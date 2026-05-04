#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def norm_space(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def manual_label(value: str) -> str:
    v = norm_space(value).lower()
    if v == "yes":
        return "yes"
    if v == "no":
        return "no"
    return "unreviewed"


def categorize_trial(title: str, conditions: str) -> tuple[str, str]:
    hay = f"{title} || {conditions}".lower()

    rules = [
        (["ckm", "kidney-metabolic", "health coach"], ("CKM", "Cardiovascular-kidney-metabolic intervention")),
        (["cardiorenal syndrome"], ("Cardiorenal / HF-CKD", "Cardiorenal syndrome study")),
        (["heart failure", "hfpef", "hfrEF".lower()], ("Heart Failure", "Heart failure-focused study")),
        (["atrial fibrillation", "cryoballoon", "pvi", "pulmonary vein isolation"], ("Atrial Fibrillation", "AF ablation / rhythm-control study")),
        (["acute coronary syndrome", "stemi", "nstemi", "myocardial infarction", "acute mi"], ("ACS / MI", "Acute coronary syndrome or MI study")),
        (["peripheral artery", "pad", "critical limb", "claudication"], ("PAD / Peripheral Intervention", "Peripheral arterial disease study")),
        (["oct", "optical coherence tomography", "ivus", "intravascular ultrasound", "coronary imaging"], ("PCI / Coronary Imaging", "PCI registry or coronary imaging study")),
        (["chronic total occlusion", "cto"], ("PCI / CTO", "Chronic total occlusion PCI study")),
        (["calcified", "atherectomy"], ("PCI / Calcified CAD", "Calcified coronary intervention study")),
        (["cabg", "secondary prevention", "post-cabg", "post pci"], ("Post-CABG / Post-PCI Secondary Prevention", "Secondary prevention after revascularization")),
        (["diabetes", "a1c", "hba1c"], ("CKM / Diabetes", "Diabetes-oriented cardiometabolic study")),
        (["lifestyle medicine", "exercise", "nutrition", "health coach", "behavior"], ("CKM / Lifestyle", "Lifestyle or coaching intervention")),
        (["valve", "valvular", "prosthetic valve", "paravalvular"], ("Structural Heart / Valvular", "Structural or valvular heart study")),
        (["lipid", "ascvd", "cholesterol"], ("ASCVD / Lipids", "ASCVD or lipid-lowering study")),
        (["oncology", "lymphoma", "leukemia", "tumor", "myeloma", "pet"], ("Oncology / Imaging", "Cancer, hematology, or oncologic imaging study")),
        (["elderly comorbidity medical database"], ("Broad Cardiology Registry", "Elderly comorbidity observational registry")),
        (["kidney disease", "dialysis", "renal progression", "ckd"], ("CKD / Renal", "Kidney disease or renal progression study")),
    ]

    for needles, result in rules:
        if any(needle in hay for needle in needles):
            return result
    return ("Other", "Unclassified trial category")


MECHANISM_PATTERNS: list[tuple[str, list[str]]] = [
    ("Temporal / Current-Episode Criterion Not Met", ["within one week", "recent hospitalization", "current episode", "timing", "enrollment window"]),
    ("Numeric / Threshold Mismatch", ["a1c", "hb a1c", "threshold", "greater than", "less than", "platelets", "anc ", "creatinine", "bilirubin"]),
    ("Medication Exclusion / Follow-Up Barrier", ["medication", "cannot follow", "noncompliance", "underdosing"]),
    ("Consent / Follow-Up Capacity Barrier", ["consent", "legally authorized representative", "unable to communicate", "follow-up", "caregiver support"]),
    ("Cognitive Impairment / Neurocognitive Barrier", ["dementia", "encephalopathy", "delirium", "cognitive", "schizoaffective"]),
    ("Procedure-Specific Confirmation Missing", ["scheduled", "planned", "plan", "not explicitly documented", "not documented", "if performed or is planned"]),
    ("Required Planned Procedure Not Documented", ["cryoballoon", "pulmonary vein isolation", "pvi", "scheduled for cryoballoon"]),
    ("Imaging-Derived Criterion Missing", ["oct", "ivus", "imaging", "scanner", "gantry", "lie still"]),
    ("Prior Procedure History Conflict", ["prior mitral valve surgery", "left atrial appendage destruction", "prior surgery", "cabg history", "post-surgical"]),
    ("Prior Procedure / Structural Heart Conflict", ["prosthetic valve", "valve replacement", "valvular", "rheumatic mitral valve"]),
    ("Hard Exclusion by Arrhythmia / Comorbidity", ["atrial fibrillation", "svt", "arrhythmia", "hard protocol-level exclusion"]),
    ("Competing Illness Burden / Investigator Unsuitability", ["frailty", "palliative", "goals of care", "septic shock", "metastatic", "too sick", "clinically stable enough"]),
    ("Population / Demographic Mismatch", ["male patient", "female patient", "pregnant", "age requirement", "over 18", "≥ 65"]),
    ("Phenotype Mismatch", ["poor fit", "different substrate", "disease scope", "not the intended", "mismatched"]),
]


MECHANISM_TO_BROAD_BUCKET = {
    "Temporal / Current-Episode Criterion Not Met": "Temporal / Current-Episode Criterion Not Met",
    "Numeric / Threshold Mismatch": "Numeric / Threshold Mismatch",
    "Medication Exclusion / Follow-Up Barrier": "Consent / Follow-Up / Capacity Barrier",
    "Consent / Follow-Up Capacity Barrier": "Consent / Follow-Up / Capacity Barrier",
    "Cognitive Impairment / Neurocognitive Barrier": "Consent / Follow-Up / Capacity Barrier",
    "Procedure-Specific Confirmation Missing": "Procedure / Imaging Requirement Not Met",
    "Required Planned Procedure Not Documented": "Procedure / Imaging Requirement Not Met",
    "Imaging-Derived Criterion Missing": "Procedure / Imaging Requirement Not Met",
    "Prior Procedure History Conflict": "Prior Procedure / Structural Heart Conflict",
    "Prior Procedure / Structural Heart Conflict": "Prior Procedure / Structural Heart Conflict",
    "Hard Exclusion by Arrhythmia / Comorbidity": "Procedure / Imaging Requirement Not Met",
    "Competing Illness Burden / Investigator Unsuitability": "Competing Illness Burden / Investigator Unsuitability",
    "Population / Demographic Mismatch": "Population / Demographic Mismatch",
    "Phenotype Mismatch": "Ambiguous / Partial Inclusion Fit",
}


def extract_failure_mechanisms(text: str) -> tuple[str, list[str], str]:
    lower = (text or "").lower()
    hits: list[str] = []
    for label, patterns in MECHANISM_PATTERNS:
        if any(pattern in lower for pattern in patterns):
            hits.append(label)

    if not hits:
        hits = ["Ambiguous / Partial Inclusion Fit"]

    unique_hits: list[str] = []
    for hit in hits:
        if hit not in unique_hits:
            unique_hits.append(hit)

    primary = unique_hits[0]
    secondary = unique_hits[1:]

    summary_parts = []
    if primary == "Hard Exclusion by Arrhythmia / Comorbidity":
        summary_parts.append("Formal exclusion appears present from documented arrhythmia/comorbidity.")
    if primary == "Required Planned Procedure Not Documented":
        summary_parts.append("Core qualifying procedure or planned intervention was not documented.")
    if primary == "Imaging-Derived Criterion Missing":
        summary_parts.append("Trial depends on imaging-specific confirmation that was not evident in the chart.")
    if primary == "Competing Illness Burden / Investigator Unsuitability":
        summary_parts.append("Underlying disease fit existed, but competing illness burden made the patient a poor practical candidate.")
    if not summary_parts:
        summary_parts.append(primary)
    return primary, secondary, " ".join(summary_parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Mirror the in-house validation analyses for TrialGPT high-likelihood matches.")
    parser.add_argument("--manual-sheet", required=True)
    parser.add_argument("--high-matches", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    manual_rows = read_csv(Path(args.manual_sheet))
    high_rows = read_csv(Path(args.high_matches))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    condensed_rows = []
    paired_rows = []

    for idx, manual_row in enumerate(manual_rows, start=1):
        condensed_rows.append(
            {
                "row_index": idx,
                "review_row_number": manual_row.get("Patient ", "").strip(),
                "manual_validation": manual_label(manual_row.get("Match? ", "")),
            }
        )

    for idx, high_row in enumerate(high_rows, start=1):
        manual_row = manual_rows[idx - 1] if idx <= len(manual_rows) else None
        validation = manual_label((manual_row or {}).get("Match? ", ""))
        review_row_number = (manual_row or {}).get("Patient ", "").strip()
        raw_text = (manual_row or {}).get("Output GPT", "")
        trial_category, trial_category_detail = categorize_trial(high_row.get("trial_title", ""), high_row.get("conditions", ""))
        paired_rows.append(
            {
                "row_index": idx,
                "review_row_number": review_row_number or "",
                "manual_validation": validation,
                "patient_id": high_row.get("patient_id", ""),
                "trial_nct_id": high_row.get("trial_nct_id", ""),
                "trial_title": high_row.get("trial_title", ""),
                "trial_category": trial_category,
                "trial_category_detail": trial_category_detail,
                "llm_confidence": high_row.get("confidence", ""),
                "llm_reason": high_row.get("reason", ""),
                "trial_conditions": high_row.get("conditions", ""),
                "raw_review_text": raw_text,
            }
        )

    reviewed_pairs = [row for row in paired_rows if row["manual_validation"] in {"yes", "no"}]
    no_pairs = [row for row in reviewed_pairs if row["manual_validation"] == "no"]

    unique_nct: dict[str, dict] = {}
    for row in reviewed_pairs:
        nct_id = row["trial_nct_id"]
        entry = unique_nct.setdefault(
            nct_id,
            {
                "trial_nct_id": nct_id,
                "trial_title": row["trial_title"],
                "trial_category": row["trial_category"],
                "trial_category_detail": row["trial_category_detail"],
                "trial_conditions": row["trial_conditions"],
                "manual_yes_count": 0,
                "manual_no_count": 0,
                "total_pairs": 0,
            },
        )
        entry["total_pairs"] += 1
        if row["manual_validation"] == "yes":
            entry["manual_yes_count"] += 1
        elif row["manual_validation"] == "no":
            entry["manual_no_count"] += 1

    category_rollup: dict[str, dict] = {}
    for row in reviewed_pairs:
        category = row["trial_category"]
        entry = category_rollup.setdefault(
            category,
            {"trial_category": category, "yes_count": 0, "no_count": 0, "total_pairs": 0, "nct_ids": set()},
        )
        entry["total_pairs"] += 1
        entry["nct_ids"].add(row["trial_nct_id"])
        if row["manual_validation"] == "yes":
            entry["yes_count"] += 1
        elif row["manual_validation"] == "no":
            entry["no_count"] += 1

    category_rows = []
    for entry in sorted(category_rollup.values(), key=lambda x: (-x["yes_count"] / x["total_pairs"], -x["total_pairs"], x["trial_category"])):
        accuracy_fraction = entry["yes_count"] / entry["total_pairs"] if entry["total_pairs"] else 0.0
        category_rows.append(
            {
                "trial_category": entry["trial_category"],
                "yes_count": entry["yes_count"],
                "no_count": entry["no_count"],
                "total_pairs": entry["total_pairs"],
                "accuracy_fraction": round(accuracy_fraction, 6),
                "accuracy_percent": round(100.0 * accuracy_fraction, 2),
                "unique_nct_count": len(entry["nct_ids"]),
                "nct_ids": "|".join(sorted(entry["nct_ids"])),
            }
        )

    annotated_failures = []
    mechanism_counter = Counter()
    mechanism_counter_all = Counter()
    broad_bucket_counter = Counter()
    failure_denominator = len(no_pairs)
    for row in no_pairs:
        primary, secondary, summary = extract_failure_mechanisms(row["raw_review_text"])
        annotated_failures.append(
            {
                "row_index": row["row_index"],
                "manual_validation": row["manual_validation"],
                "patient_id": row["patient_id"],
                "trial_nct_id": row["trial_nct_id"],
                "trial_title": row["trial_title"],
                "trial_category": row["trial_category"],
                "primary_failure_mechanism": primary,
                "secondary_failure_mechanisms": "|".join(secondary),
                "failure_summary": summary,
                "raw_failure_text": row["raw_review_text"],
            }
        )
        mechanism_counter[primary] += 1
        for mech in [primary, *secondary]:
            mechanism_counter_all[mech] += 1
            broad_bucket_counter[MECHANISM_TO_BROAD_BUCKET.get(mech, "Ambiguous / Partial Inclusion Fit")] += 1

    primary_mechanism_rows = [
        {
            "failure_mechanism": mech,
            "count": count,
            "percent_of_failures": round(100.0 * count / failure_denominator, 2) if failure_denominator else 0.0,
            "mechanism_role": "primary_only",
        }
        for mech, count in mechanism_counter.most_common()
    ]
    secondary_mechanism_rows = [
        {
            "failure_mechanism": mech,
            "count": count,
            "percent_of_failures": round(100.0 * count / failure_denominator, 2) if failure_denominator else 0.0,
            "mechanism_role": "primary_or_secondary",
        }
        for mech, count in mechanism_counter_all.most_common()
    ]
    failure_breakdown_rows = primary_mechanism_rows + secondary_mechanism_rows

    broad_bucket_rows = [
        {
            "broad_failure_bucket": bucket,
            "count": count,
            "percent_of_failures": round(100.0 * count / failure_denominator, 2) if failure_denominator else 0.0,
        }
        for bucket, count in broad_bucket_counter.most_common()
    ]

    prefix = "trialgpt"
    write_csv(
        out_dir / f"{prefix}_condensed_manual_validation.csv",
        condensed_rows,
        ["row_index", "review_row_number", "manual_validation"],
    )
    write_json(out_dir / f"{prefix}_condensed_manual_validation.json", condensed_rows)

    pair_fields = [
        "row_index",
        "review_row_number",
        "manual_validation",
        "patient_id",
        "trial_nct_id",
        "trial_title",
        "trial_category",
        "trial_category_detail",
        "llm_confidence",
        "llm_reason",
        "trial_conditions",
        "raw_review_text",
    ]
    write_csv(out_dir / f"{prefix}_manual_validation_pairs_with_trial_category.csv", paired_rows, pair_fields)
    write_json(out_dir / f"{prefix}_manual_validation_pairs_with_trial_category.json", paired_rows)

    unique_rows = sorted(unique_nct.values(), key=lambda x: (-x["manual_yes_count"], -x["total_pairs"], x["trial_nct_id"]))
    write_csv(
        out_dir / f"{prefix}_unique_nct_trial_catalog.csv",
        unique_rows,
        ["trial_nct_id", "trial_title", "trial_category", "trial_category_detail", "trial_conditions", "manual_yes_count", "manual_no_count", "total_pairs"],
    )
    write_json(out_dir / f"{prefix}_unique_nct_trial_catalog.json", unique_rows)

    write_csv(
        out_dir / f"{prefix}_accuracy_by_trial_category.csv",
        category_rows,
        ["trial_category", "yes_count", "no_count", "total_pairs", "accuracy_fraction", "accuracy_percent", "unique_nct_count", "nct_ids"],
    )
    write_json(out_dir / f"{prefix}_accuracy_by_trial_category.json", category_rows)

    write_csv(
        out_dir / f"{prefix}_failure_mechanisms_annotated_pairs.csv",
        annotated_failures,
        ["row_index", "manual_validation", "patient_id", "trial_nct_id", "trial_title", "trial_category", "primary_failure_mechanism", "secondary_failure_mechanisms", "failure_summary", "raw_failure_text"],
    )
    write_json(out_dir / f"{prefix}_failure_mechanisms_annotated_pairs.json", annotated_failures)

    write_csv(
        out_dir / f"{prefix}_failure_mechanisms_breakdown.csv",
        failure_breakdown_rows,
        ["failure_mechanism", "count", "percent_of_failures", "mechanism_role"],
    )
    write_json(out_dir / f"{prefix}_failure_mechanisms_breakdown.json", failure_breakdown_rows)

    write_csv(
        out_dir / f"{prefix}_failure_mechanisms_broad_buckets.csv",
        broad_bucket_rows,
        ["broad_failure_bucket", "count", "percent_of_failures"],
    )
    write_json(out_dir / f"{prefix}_failure_mechanisms_broad_buckets.json", broad_bucket_rows)

    summary = {
        "manual_sheet_rows": len(manual_rows),
        "high_likelihood_pairs": len(high_rows),
        "reviewed_pairs": len(reviewed_pairs),
        "yes_pairs": sum(1 for row in reviewed_pairs if row["manual_validation"] == "yes"),
        "no_pairs": failure_denominator,
        "unreviewed_pairs": sum(1 for row in paired_rows if row["manual_validation"] == "unreviewed"),
        "unique_reviewed_ncts": len(unique_nct),
    }
    write_json(out_dir / f"{prefix}_analysis_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
