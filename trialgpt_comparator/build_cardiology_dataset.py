#!/usr/bin/env python3
"""
Build TrialGPT-compatible cardiology dataset from:
  - MIMIC FHIR patient bundles (top-200 cardiology patients)
  - Cureva enriched cardiology trials CSV (11,231 trials with eligibility criteria)

Outputs to dataset/cardiology/:
  corpus.jsonl   — one trial per line in TrialGPT format
  queries.jsonl  — one patient per line as free-text clinical summary
  qrels/test.tsv — stub qrels (all patients listed, no ground-truth labels)
"""

import csv
import json
import os
import re
from collections import Counter, OrderedDict
from datetime import date, datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PATIENTS_DIR = Path(
    "/Users/andrew/Desktop/cureva-main-new/PublicationData/MIMIC_FHIR_Patients/patients"
)
TOP200_FILE = Path(
    "/Users/andrew/Desktop/cureva-main-new/PublicationData/MIMIC_FHIR_Patients"
    "/cardiology_trial_top200_patient_ids.txt"
)
TRIALS_CSV = Path(
    "/Users/andrew/Desktop/cureva-main-new/PublicationData/Cardio_Trials_Snapshot"
    "/full_publication_run_2026-03-05/trials_enriched_from_snapshot.csv"
)
OUT_DIR = Path("/Users/andrew/Desktop/TrialGPT/dataset/cardiology")

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def unique_ordered(items):
    """Deduplicate while preserving order."""
    seen = set()
    out = []
    for item in items:
        key = item.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(item.strip())
    return out


def clean_paren(text: str) -> str:
    """Remove parenthetical clarifications that add noise."""
    return re.sub(r"\s*\(.*?\)", "", text).strip()


def calc_age_from_fhir(birth_date_str: str, encounter_dates: list[str]) -> int | None:
    """
    MIMIC dates are shifted forward; compute age relative to the latest
    encounter date to get a clinically meaningful age.
    """
    if not birth_date_str:
        return None
    try:
        bd = datetime.strptime(birth_date_str[:10], "%Y-%m-%d").date()
    except ValueError:
        return None
    anchors = []
    for ds in encounter_dates:
        try:
            anchors.append(datetime.strptime(ds[:10], "%Y-%m-%d").date())
        except (ValueError, TypeError):
            pass
    if not anchors:
        return None
    anchor = max(anchors)
    years = int((anchor - bd).days / 365.25)
    return years if 0 <= years <= 120 else None


# ---------------------------------------------------------------------------
# Patient FHIR → free-text summary  (pure-JSON, no fhirclient needed)
# ---------------------------------------------------------------------------

def parse_fhir_bundle(bundle: dict, patient_id: str) -> str:
    """Convert a MIMIC FHIR bundle dict to a TrialGPT-style clinical summary."""
    gender = None
    birth_date = None
    encounter_dates: list[str] = []
    conditions: list[str] = []
    meds: list[str] = []
    key_labs: list[str] = []

    for entry in bundle.get("entry", []):
        r = entry.get("resource", {})
        rtype = r.get("resourceType", "")

        if rtype == "Patient":
            gender = r.get("gender", "unknown")
            birth_date = r.get("birthDate")

        elif rtype == "Condition":
            code_obj = r.get("code", {})
            # prefer text > coding display
            text = code_obj.get("text")
            if not text:
                codings = code_obj.get("coding", [])
                if codings:
                    text = codings[0].get("display") or codings[0].get("text")
            if text:
                conditions.append(clean_paren(text))

        elif rtype == "MedicationRequest":
            med_cc = r.get("medicationCodeableConcept", {})
            codings = med_cc.get("coding", [])
            name = med_cc.get("text")
            if not name and codings:
                name = codings[0].get("display") or codings[0].get("text")
            if name and name.strip():
                meds.append(name.strip())

        elif rtype == "Observation":
            # Collect encounter anchor dates
            eff = r.get("effectiveDateTime") or r.get("effectivePeriod", {}).get("start")
            if eff:
                encounter_dates.append(eff)
            # Collect lab values
            categories = r.get("category", [])
            is_lab = any(
                coding.get("code") == "laboratory"
                for cat in categories
                for coding in cat.get("coding", [])
            )
            if is_lab:
                code_obj = r.get("code", {})
                codings = code_obj.get("coding", [])
                test_name = code_obj.get("text")
                if not test_name and codings:
                    test_name = codings[0].get("display")
                vq = r.get("valueQuantity", {})
                if test_name and vq.get("value") is not None:
                    unit = vq.get("unit", "")
                    key_labs.append(f"{test_name}: {vq['value']} {unit}".strip())

        elif rtype == "Encounter":
            period = r.get("period", {})
            for ts in [period.get("start"), period.get("end")]:
                if ts:
                    encounter_dates.append(ts)

    age = calc_age_from_fhir(birth_date, encounter_dates)
    gender_str = gender if gender else "unknown sex"
    age_str = f"{age}-year-old" if age is not None else "adult"

    # Deduplicate and limit lists
    cond_list = unique_ordered(conditions)
    med_list = unique_ordered(meds)[:25]

    # Prioritise informative lab names (skip generic numeric codes)
    lab_counter: dict[str, str] = {}
    for lab in key_labs:
        name = lab.split(":")[0]
        if name and len(name) > 3:
            lab_counter[name] = lab
    lab_list = list(lab_counter.values())[:20]

    lines = [
        f"Patient {patient_id} is a {age_str} {gender_str}.",
        f"Medical history includes: {', '.join(cond_list[:40]) if cond_list else 'No conditions recorded'}.",
    ]
    if med_list:
        lines.append(f"Current medications include: {', '.join(med_list)}.")
    if lab_list:
        lines.append(f"Notable lab values: {'; '.join(lab_list[:10])}.")

    return " ".join(lines)


# ---------------------------------------------------------------------------
# Trial eligibility criteria parser
# ---------------------------------------------------------------------------

def split_eligibility(raw: str) -> tuple[str, str]:
    """
    Split a raw eligibility_criteria string into (inclusion, exclusion).
    Handles both 'INCLUSION CRITERIA:' and 'Inclusion Criteria:' styles.
    Returns ('', '') if text is empty.
    """
    if not raw or not raw.strip():
        return "", ""

    text = raw.strip()
    # Normalise headers to a canonical form
    text_norm = re.sub(
        r"(?i)(inclusion\s+criteria\s*:)",
        "\n__INCLUSION__\n",
        text,
    )
    text_norm = re.sub(
        r"(?i)(exclusion\s+criteria\s*:)",
        "\n__EXCLUSION__\n",
        text_norm,
    )

    inclusion = ""
    exclusion = ""

    if "__INCLUSION__" in text_norm and "__EXCLUSION__" in text_norm:
        parts = text_norm.split("__INCLUSION__", 1)
        after_incl = parts[1]
        if "__EXCLUSION__" in after_incl:
            inc_raw, exc_raw = after_incl.split("__EXCLUSION__", 1)
            inclusion = "inclusion criteria:\n\n" + inc_raw.strip()
            exclusion = exc_raw.strip()
        else:
            inclusion = "inclusion criteria:\n\n" + after_incl.strip()
    elif "__EXCLUSION__" in text_norm:
        parts = text_norm.split("__EXCLUSION__", 1)
        exclusion = parts[1].strip()
        inclusion = ""
    elif "__INCLUSION__" in text_norm:
        parts = text_norm.split("__INCLUSION__", 1)
        inclusion = "inclusion criteria:\n\n" + parts[1].strip()
    else:
        # No clear markers — treat whole block as inclusion
        inclusion = "inclusion criteria:\n\n" + text

    return inclusion, exclusion


# ---------------------------------------------------------------------------
# Build corpus.jsonl
# ---------------------------------------------------------------------------

def build_corpus():
    print("Building corpus.jsonl from enriched trials CSV …")
    out_path = OUT_DIR / "corpus.jsonl"
    skipped = 0
    written = 0

    with open(TRIALS_CSV, newline="", encoding="utf-8") as csvfile, \
         open(out_path, "w", encoding="utf-8") as outfile:

        reader = csv.DictReader(csvfile)
        for row in reader:
            nct_id = row.get("nct_id", "").strip()
            if not nct_id:
                skipped += 1
                continue

            title = (row.get("title") or "").strip()
            brief_summary = (row.get("brief_summary") or "").strip()
            conditions_raw = (row.get("conditions") or "").strip()
            eligibility_raw = row.get("eligibility_criteria") or ""

            inclusion, exclusion = split_eligibility(eligibility_raw)

            # Parse diseases list from pipe-separated conditions field
            diseases_list = [c.strip() for c in re.split(r"[|,]", conditions_raw) if c.strip()]

            # Build full text (what TrialGPT BM25 / MedCPT will index)
            text_parts = []
            if brief_summary:
                text_parts.append(f"Summary: {brief_summary}")
            if inclusion:
                text_parts.append(f"Inclusion criteria: {inclusion}")
            if exclusion:
                text_parts.append(f"Exclusion criteria: {exclusion}")
            text = "\n".join(text_parts)

            if not text.strip():
                skipped += 1
                continue

            record = {
                "_id": nct_id,
                "title": title or nct_id,
                "text": text,
                "metadata": {
                    "brief_title": title,
                    "phase": "",
                    "drugs": "[]",
                    "drugs_list": [],
                    "diseases": str(diseases_list),
                    "diseases_list": diseases_list,
                    "enrollment": "",
                    "inclusion_criteria": inclusion,
                    "exclusion_criteria": exclusion,
                    "brief_summary": brief_summary,
                },
            }
            outfile.write(json.dumps(record) + "\n")
            written += 1

    print(f"  → {written} trials written, {skipped} skipped")
    return written


# ---------------------------------------------------------------------------
# Build queries.jsonl
# ---------------------------------------------------------------------------

def build_queries():
    print("Building queries.jsonl from MIMIC FHIR patient bundles …")
    patient_ids = TOP200_FILE.read_text().strip().splitlines()
    out_path = OUT_DIR / "queries.jsonl"
    written = 0
    missing = 0

    with open(out_path, "w", encoding="utf-8") as outfile:
        for pid in patient_ids:
            pid = pid.strip()
            if not pid:
                continue
            fhir_path = PATIENTS_DIR / f"Patient_{pid}.json"
            if not fhir_path.exists():
                print(f"  [WARN] Patient file not found: {fhir_path.name}")
                missing += 1
                continue
            bundle = json.loads(fhir_path.read_text())
            summary = parse_fhir_bundle(bundle, pid)
            record = {"_id": f"cardio-{pid}", "text": summary}
            outfile.write(json.dumps(record) + "\n")
            written += 1

    print(f"  → {written} patients written, {missing} missing files")
    return written, patient_ids


# ---------------------------------------------------------------------------
# Build stub qrels (required by BEIR GenericDataLoader)
# ---------------------------------------------------------------------------

def build_stub_qrels(patient_ids: list[str], first_trial_id: str):
    """
    TrialGPT's retrieval code needs qrels to iterate over patients.
    We write a stub with a single placeholder entry per patient so
    the GenericDataLoader is satisfied.  Recall/precision metrics will
    be meaningless (no ground truth), but the retrieval pipeline runs.
    """
    print("Writing stub qrels/test.tsv …")
    qrels_path = OUT_DIR / "qrels" / "test.tsv"
    with open(qrels_path, "w") as f:
        f.write("query-id\tcorpus-id\tscore\n")
        for pid in patient_ids:
            pid = pid.strip()
            if pid:
                f.write(f"cardio-{pid}\t{first_trial_id}\t0\n")
    print(f"  → stub qrels for {len([p for p in patient_ids if p.strip()])} patients")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "qrels").mkdir(exist_ok=True)

    n_trials = build_corpus()
    n_patients, patient_ids = build_queries()

    # Get first trial ID for stub qrels
    with open(OUT_DIR / "corpus.jsonl") as f:
        first_trial_id = json.loads(f.readline())["_id"]

    build_stub_qrels(patient_ids, first_trial_id)

    print()
    print("=" * 60)
    print("Cardiology dataset ready:")
    print(f"  Corpus  : {OUT_DIR}/corpus.jsonl  ({n_trials:,} trials)")
    print(f"  Queries : {OUT_DIR}/queries.jsonl ({n_patients} patients)")
    print(f"  Qrels   : {OUT_DIR}/qrels/test.tsv (stub — no ground truth)")
    print()
    print("Next steps:")
    print("  1. Generate retrieval keywords (optional, costs ~200 API calls):")
    print("     python trialgpt_retrieval/keyword_generation.py cardiology gpt-4-turbo")
    print()
    print("  2. Run hybrid retrieval (BM25-only first, MedCPT optional):")
    print("     python run_cardiology_retrieval.py")
    print()
    print("  3. Run matching (expensive — ~200 × top_K × 2 API calls):")
    print("     python trialgpt_matching/run_matching.py cardiology gpt-4-turbo")
    print()
    print("  4. Run ranking aggregation:")
    print("     python trialgpt_ranking/run_aggregation.py cardiology gpt-4-turbo \\")
    print("       results/matching_results_cardiology_gpt-4-turbo.json")
    print()
    print("  5. Get top-5 matches per patient:")
    print("     python top5_cardiology.py \\")
    print("       results/matching_results_cardiology_gpt-4-turbo.json \\")
    print("       results/aggregation_results_cardiology_gpt-4-turbo.json")
