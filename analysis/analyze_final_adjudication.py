#!/usr/bin/env python3
"""Import the completed blinded reviews and build final ablation results."""

from __future__ import annotations

import csv
import hashlib
import math
import os
import random
import re
import zipfile
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from xml.etree import ElementTree as ET


# Inputs are restricted (they contain MIMIC-IV patient identifiers) and are not
# distributed with this repository. Point LUMEN_ANALYSIS_DIR at a local folder
# holding them, laid out as below.
ROOT = Path(os.environ.get("LUMEN_ANALYSIS_DIR", "analysis_data"))
OUT = ROOT / "final_adjudication"
ASSIGNMENTS = ROOT / "reviewer_assignments"
ORIGINAL_LABELS = ROOT / "high_likelihood_manual_validation_pairs_with_trial_category.csv"
ABLATION_OUTPUTS = ROOT / "all_ablation_outputs.csv"
RETURNED_WORKBOOKS = sorted((ROOT / "returned_workbooks").glob("*.xlsx"))

CONDITION_LABELS = {
    "full_lumen": "Full Lumen",
    "no_second_pass": "No second-pass review",
    "no_llm_reranker": "No LLM reranker",
    "no_deterministic_reranker": "No deterministic reranker",
    "decision_context_removed": "Second pass without decision context",
}

FAILURE_MAP = {
    "ambiguous fit": "Ambiguous or partial inclusion fit",
    "competing illness": "Competing illness or investigator unsuitability",
    "consent/follow-up": "Consent, follow-up, or capacity barrier",
    "numeric threshold": "Numeric or threshold mismatch",
    "other": "Other",
    "population mismatch": "Population or demographic mismatch",
    "prior procedure conflict": "Prior procedure or structural conflict",
    "procedure/imaging unmet": "Procedure or imaging requirement unmet",
    "temporal mismatch": "Temporal or current-episode mismatch",
    "wrong disease": "Wrong disease or indication",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for row in rows:
            for field in row:
                if field not in fields:
                    fields.append(field)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def xlsx_rows(path: Path) -> list[dict[str, str]]:
    ns = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path) as archive:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in root.findall("x:si", ns):
                shared.append("".join(node.text or "" for node in item.findall(".//x:t", ns)))
        sheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))

    matrix: list[dict[str, str]] = []
    for row in sheet.findall(".//x:sheetData/x:row", ns):
        values: dict[str, str] = {}
        for cell in row.findall("x:c", ns):
            match = re.match(r"([A-Z]+)", cell.attrib.get("r", ""))
            if not match:
                continue
            cell_type = cell.attrib.get("t", "")
            value_node = cell.find("x:v", ns)
            if cell_type == "inlineStr":
                value = "".join(node.text or "" for node in cell.findall(".//x:t", ns))
            elif value_node is None:
                value = ""
            elif cell_type == "s":
                value = shared[int(value_node.text or "0")]
            else:
                value = value_node.text or ""
            values[match.group(1)] = value
        matrix.append(values)

    if not matrix:
        return []
    columns = ["A", "B", "C", "D", "E", "F"]
    headers = [matrix[0].get(column, "").strip() for column in columns]
    return [
        {header: raw.get(column, "").strip() for header, column in zip(headers, columns)}
        for raw in matrix[1:]
        if raw.get("A", "").strip()
    ]


def pair_key(patient_id: str, trial_nct_id: str) -> tuple[str, str]:
    return patient_id.strip(), trial_nct_id.strip()


def wilson(successes: int, total: int) -> tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return center - radius, center + radius


def percent(value: float) -> str:
    return f"{100 * value:.1f}"


def exact_mcnemar_p(b: int, c: int) -> float:
    discordant = b + c
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, i) for i in range(0, min(b, c) + 1)) / (2**discordant)
    return min(1.0, 2 * tail)


def load_expected() -> dict[str, dict[str, str]]:
    expected: dict[str, dict[str, str]] = {}
    for path in sorted(ASSIGNMENTS.glob("reviewer_*_matches.csv")):
        for row in read_csv(path):
            pair_id = row["pair_id"].strip()
            if pair_id in expected:
                raise RuntimeError(f"Duplicate expected Pair ID: {pair_id}")
            expected[pair_id] = {
                "assignment": row["reviewer_assignment"].strip(),
                "patient_id": row["patient_id"].strip(),
                "trial_nct_id": row["trial_nct_id"].strip(),
                "trial_title": row["trial_title"].strip(),
            }
    return expected


def import_returns(expected: dict[str, dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    imported: list[dict[str, str]] = []
    issues: list[dict[str, str]] = []
    seen: set[str] = set()
    for path in RETURNED_WORKBOOKS:
        for row_number, row in enumerate(xlsx_rows(path), start=2):
            pair_id = row.get("Pair ID", "").strip()
            label = row.get("Match", "").strip().lower()
            confidence = row.get("Confidence", "").strip().lower()
            failure_raw = row.get("Primary Failure Mechanism", "").strip()
            reviewer = row.get("Reviewer Name", "").strip()
            if pair_id not in expected:
                issues.append({"severity": "error", "source_file": path.name, "row": row_number, "pair_id": pair_id, "issue": "Pair ID not in frozen assignment universe"})
                continue
            if pair_id in seen:
                issues.append({"severity": "error", "source_file": path.name, "row": row_number, "pair_id": pair_id, "issue": "Duplicate returned decision"})
                continue
            seen.add(pair_id)
            if label not in {"yes", "no"}:
                issues.append({"severity": "error", "source_file": path.name, "row": row_number, "pair_id": pair_id, "issue": f"Invalid binary decision: {label!r}"})
            if confidence not in {"high", "medium", "low"}:
                issues.append({"severity": "error", "source_file": path.name, "row": row_number, "pair_id": pair_id, "issue": f"Invalid confidence: {confidence!r}"})
            if label == "no" and not failure_raw:
                issues.append({"severity": "warning", "source_file": path.name, "row": row_number, "pair_id": pair_id, "issue": "No decision lacks a failure mechanism"})
            reviewer_inferred = ""
            if not reviewer:
                same_file_names = Counter(
                    candidate.get("Reviewer Name", "").strip()
                    for candidate in xlsx_rows(path)
                    if candidate.get("Reviewer Name", "").strip()
                )
                if len(same_file_names) == 1:
                    reviewer_inferred = next(iter(same_file_names))
                issues.append({"severity": "warning", "source_file": path.name, "row": row_number, "pair_id": pair_id, "issue": f"Reviewer name missing; workbook-level identity is {reviewer_inferred or 'not inferable'}"})
            expected_row = expected[pair_id]
            failure_key = failure_raw.casefold()
            imported.append({
                "pair_id": pair_id,
                **expected_row,
                "manual_label": label,
                "confidence": confidence,
                "failure_mechanism_raw": failure_raw,
                "failure_mechanism_normalized": FAILURE_MAP.get(failure_key, failure_raw),
                "reviewer_name_reported": reviewer,
                "reviewer_name_effective": reviewer or reviewer_inferred,
                "reviewer_identity_inferred": "yes" if reviewer_inferred else "no",
                "notes": row.get("Notes", "").strip(),
                "source_file": path.name,
                "source_sha256": sha256(path),
            })
    missing = sorted(set(expected) - seen)
    for pair_id in missing:
        issues.append({"severity": "error", "source_file": "", "row": "", "pair_id": pair_id, "issue": "Assigned pair has no returned decision"})
    return imported, issues


def load_original_labels() -> dict[tuple[str, str], dict[str, str]]:
    labels: dict[tuple[str, str], dict[str, str]] = {}
    for row in read_csv(ORIGINAL_LABELS):
        patient = (row.get("patient_id") or row.get("Patient ID") or "").strip()
        nct = (row.get("nct_id") or row.get("trial_nct_id") or row.get("NCT ID") or "").strip()
        value = (row.get("manual_validation") or row.get("manual_label") or "").strip().lower()
        if patient and nct and value in {"yes", "no"}:
            labels[pair_key(patient, nct)] = {
                "patient_id": patient,
                "trial_nct_id": nct,
                "manual_label": value,
                "confidence": "",
                "failure_mechanism_normalized": "",
                "reviewer_name_effective": "",
                "source_type": "original high-likelihood manual validation",
                "source_file": ORIGINAL_LABELS.name,
            }
    return labels


def condition_metrics(
    condition: str,
    rows: list[dict[str, str]],
    labels: dict[tuple[str, str], dict[str, str]],
) -> dict[str, object]:
    unique = {(row["patient_id"], row["trial_nct_id"]): row for row in rows}
    patient_pairs: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for pair in unique:
        patient_pairs[pair[0]].append(pair)
    high = {
        pair for pair, row in unique.items()
        if row.get("second_pass_label", "").strip() == "high_likelihood"
    }
    yes_pairs = {pair for pair in unique if labels[pair]["manual_label"] == "yes"}
    no_pairs = set(unique) - yes_pairs
    high_yes = high & yes_pairs
    high_no = high & no_pairs
    non_high = set(unique) - high
    non_high_yes = non_high & yes_pairs
    patient_hits = {
        patient for patient, pairs in patient_pairs.items()
        if any(pair in yes_pairs for pair in pairs)
    }
    high_patient_hits = {pair[0] for pair in high_yes}
    pair_lo, pair_hi = wilson(len(yes_pairs), len(unique))
    patient_lo, patient_hi = wilson(len(patient_hits), len(patient_pairs))
    high_lo, high_hi = wilson(len(high_yes), len(high)) if high else (0.0, 0.0)
    high_patient_lo, high_patient_hi = wilson(len(high_patient_hits), len(patient_pairs))
    sensitivity = len(high_yes) / len(yes_pairs) if yes_pairs else 0.0
    specificity = len(non_high - yes_pairs) / len(no_pairs) if no_pairs else 0.0
    enrichment = (
        (len(high_yes) / len(high)) / (len(non_high_yes) / len(non_high))
        if high and non_high_yes else 0.0
    )
    return {
        "condition": condition,
        "display_name": CONDITION_LABELS[condition],
        "patients": len(patient_pairs),
        "top5_pairs": len(unique),
        "top5_plausible_pairs": len(yes_pairs),
        "top5_pair_plausibility_percent": percent(len(yes_pairs) / len(unique)),
        "top5_pair_wilson95_low_percent": percent(pair_lo),
        "top5_pair_wilson95_high_percent": percent(pair_hi),
        "patients_with_at_least_one_plausible_top5": len(patient_hits),
        "top5_patient_yield_percent": percent(len(patient_hits) / len(patient_pairs)),
        "top5_patient_yield_wilson95_low_percent": percent(patient_lo),
        "top5_patient_yield_wilson95_high_percent": percent(patient_hi),
        "high_likelihood_pairs_advanced": len(high),
        "high_likelihood_plausible": len(high_yes),
        "high_likelihood_false_positive": len(high_no),
        "high_likelihood_precision_percent": percent(len(high_yes) / len(high)) if high else "not applicable",
        "high_likelihood_precision_wilson95_low_percent": percent(high_lo) if high else "",
        "high_likelihood_precision_wilson95_high_percent": percent(high_hi) if high else "",
        "confirmed_high_likelihood_patients": len(high_patient_hits),
        "confirmed_high_likelihood_patient_yield_percent": percent(len(high_patient_hits) / len(patient_pairs)),
        "confirmed_high_likelihood_patient_yield_wilson95_low_percent": percent(high_patient_lo),
        "confirmed_high_likelihood_patient_yield_wilson95_high_percent": percent(high_patient_hi),
        "non_high_pairs": len(non_high),
        "non_high_plausible_pairs": len(non_high_yes),
        "non_high_pair_plausibility_percent": percent(len(non_high_yes) / len(non_high)) if non_high else "",
        "second_pass_pair_sensitivity_percent": percent(sensitivity) if high else "not applicable",
        "second_pass_pair_specificity_percent": percent(specificity) if high else "not applicable",
        "high_vs_non_high_enrichment_ratio": f"{enrichment:.2f}" if high else "not applicable",
        "review_burden_reduction_vs_all_top5_percent": percent(1 - len(high) / len(unique)) if high else "0.0",
    }


def paired_comparisons(
    by_condition: dict[str, list[dict[str, str]]],
    labels: dict[tuple[str, str], dict[str, str]],
) -> list[dict[str, object]]:
    baseline = by_condition["full_lumen"]

    def hits(rows: list[dict[str, str]]) -> dict[str, bool]:
        grouped: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for row in rows:
            grouped[row["patient_id"]].append((row["patient_id"], row["trial_nct_id"]))
        return {patient: any(labels[pair]["manual_label"] == "yes" for pair in pairs) for patient, pairs in grouped.items()}

    baseline_hits = hits(baseline)
    output = []
    for condition, rows in by_condition.items():
        if condition in {"full_lumen", "no_second_pass", "decision_context_removed"}:
            continue
        current_hits = hits(rows)
        patients = sorted(baseline_hits)
        differences = [int(current_hits[p]) - int(baseline_hits[p]) for p in patients]
        b = sum(baseline_hits[p] and not current_hits[p] for p in patients)
        c = sum(current_hits[p] and not baseline_hits[p] for p in patients)
        rng = random.Random(20260817)
        boot = []
        for _ in range(10000):
            draw = [differences[rng.randrange(len(differences))] for _ in differences]
            boot.append(100 * sum(draw) / len(draw))
        boot.sort()
        output.append({
            "comparison": f"{CONDITION_LABELS[condition]} vs Full Lumen",
            "condition_minus_full_lumen_patient_yield_points": f"{100 * sum(differences) / len(differences):.1f}",
            "paired_bootstrap_95ci_low_points": f"{boot[249]:.1f}",
            "paired_bootstrap_95ci_high_points": f"{boot[9749]:.1f}",
            "full_only_patient_hits": b,
            "ablation_only_patient_hits": c,
            "exact_mcnemar_p": f"{exact_mcnemar_p(b, c):.4f}",
            "bootstrap_resamples": 10000,
            "bootstrap_seed": 20260817,
        })
    return output


def high_likelihood_patient_comparisons(
    by_condition: dict[str, list[dict[str, str]]],
    labels: dict[tuple[str, str], dict[str, str]],
) -> list[dict[str, object]]:
    def confirmed_hits(rows: list[dict[str, str]]) -> dict[str, bool]:
        patients = {row["patient_id"] for row in rows}
        hits = {patient: False for patient in patients}
        for row in rows:
            pair = (row["patient_id"], row["trial_nct_id"])
            if row.get("second_pass_label", "").strip() == "high_likelihood" and labels[pair]["manual_label"] == "yes":
                hits[row["patient_id"]] = True
        return hits

    baseline_hits = confirmed_hits(by_condition["full_lumen"])
    output = []
    for condition in ["no_llm_reranker", "no_deterministic_reranker", "decision_context_removed"]:
        current_hits = confirmed_hits(by_condition[condition])
        patients = sorted(baseline_hits)
        differences = [int(current_hits[p]) - int(baseline_hits[p]) for p in patients]
        b = sum(baseline_hits[p] and not current_hits[p] for p in patients)
        c = sum(current_hits[p] and not baseline_hits[p] for p in patients)
        rng = random.Random(20260817)
        boot = []
        for _ in range(10000):
            draw = [differences[rng.randrange(len(differences))] for _ in differences]
            boot.append(100 * sum(draw) / len(draw))
        boot.sort()
        output.append({
            "comparison": f"{CONDITION_LABELS[condition]} vs Full Lumen",
            "condition_minus_full_lumen_confirmed_high_likelihood_yield_points": f"{100 * sum(differences) / len(differences):.1f}",
            "paired_bootstrap_95ci_low_points": f"{boot[249]:.1f}",
            "paired_bootstrap_95ci_high_points": f"{boot[9749]:.1f}",
            "full_only_confirmed_patient_hits": b,
            "ablation_only_confirmed_patient_hits": c,
            "exact_mcnemar_p": f"{exact_mcnemar_p(b, c):.4f}",
            "bootstrap_resamples": 10000,
            "bootstrap_seed": 20260817,
        })
    return output


def top_k_metrics(
    by_condition: dict[str, list[dict[str, str]]],
    labels: dict[tuple[str, str], dict[str, str]],
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for condition in CONDITION_LABELS:
        grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
        # These two conditions retain the exact full-Lumen top-five shortlist.
        # Their files were serialized in a different order, which is not a new rank.
        rank_source = "full_lumen" if condition in {"no_second_pass", "decision_context_removed"} else condition
        for row in by_condition[rank_source]:
            grouped[row["patient_id"]].append(row)
        for k in range(1, 6):
            hits = 0
            for rows in grouped.values():
                hits += any(
                    labels[(row["patient_id"], row["trial_nct_id"])]["manual_label"] == "yes"
                    for row in rows[:k]
                )
            lo, hi = wilson(hits, len(grouped))
            output.append({
                "condition": condition,
                "display_name": CONDITION_LABELS[condition],
                "top_k": k,
                "patients_with_at_least_one_plausible_match": hits,
                "patients": len(grouped),
                "patient_yield_percent": percent(hits / len(grouped)),
                "wilson95_low_percent": percent(lo),
                "wilson95_high_percent": percent(hi),
                "rank_definition": "Cumulative position in the frozen exported shortlist",
            })
    return output


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    expected = load_expected()
    returned, issues = import_returns(expected)
    write_csv(OUT / "normalized_returned_reviews.csv", sorted(returned, key=lambda row: row["pair_id"]))
    write_csv(OUT / "validation_issues.csv", issues, ["severity", "source_file", "row", "pair_id", "issue"])

    labels = load_original_labels()
    for row in returned:
        key = pair_key(row["patient_id"], row["trial_nct_id"])
        if key in labels:
            raise RuntimeError(f"Returned review overlaps original adjudication: {key}")
        labels[key] = {
            **row,
            "source_type": "completed blinded ablation review",
        }
    label_rows = [labels[key] for key in sorted(labels)]
    write_csv(OUT / "all_manual_labels_final.csv", label_rows)

    outputs = read_csv(ABLATION_OUTPUTS)
    by_condition: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in outputs:
        by_condition[row["condition"]].append(row)
    universe = {(row["patient_id"], row["trial_nct_id"]) for row in outputs}
    missing_labels = sorted(universe - set(labels))
    extra_labels = sorted(set(labels) - universe)
    if missing_labels or extra_labels:
        raise RuntimeError(f"Label-universe mismatch: missing={len(missing_labels)}, extra={len(extra_labels)}")

    metrics = [condition_metrics(condition, by_condition[condition], labels) for condition in CONDITION_LABELS]
    write_csv(OUT / "condition_metrics.csv", metrics)
    comparisons = paired_comparisons(by_condition, labels)
    write_csv(OUT / "paired_patient_comparisons.csv", comparisons)
    write_csv(
        OUT / "paired_high_likelihood_patient_comparisons.csv",
        high_likelihood_patient_comparisons(by_condition, labels),
    )
    write_csv(OUT / "patient_level_top_k.csv", top_k_metrics(by_condition, labels))

    reviewer_summary = []
    for reviewer, rows in sorted(defaultdict(list, {
        name: [row for row in returned if row["reviewer_name_effective"] == name]
        for name in sorted({row["reviewer_name_effective"] for row in returned})
    }).items()):
        decisions = Counter(row["manual_label"] for row in rows)
        confidence = Counter(row["confidence"] for row in rows)
        reviewer_summary.append({
            "reviewer": reviewer,
            "pairs": len(rows),
            "yes": decisions["yes"],
            "no": decisions["no"],
            "yes_percent": percent(decisions["yes"] / len(rows)),
            "high_confidence": confidence["high"],
            "medium_confidence": confidence["medium"],
            "low_confidence": confidence["low"],
        })
    write_csv(OUT / "reviewer_summary.csv", reviewer_summary)

    failures = Counter(row["failure_mechanism_normalized"] or "Unspecified" for row in returned if row["manual_label"] == "no")
    failure_rows = [
        {"failure_mechanism": mechanism, "no_pairs": count, "percent_of_returned_no_decisions": percent(count / sum(failures.values()))}
        for mechanism, count in failures.most_common()
    ]
    write_csv(OUT / "failure_mechanisms.csv", failure_rows)

    audit = [{
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "returned_workbooks": len(RETURNED_WORKBOOKS),
        "new_returned_binary_decisions": len(returned),
        "original_high_likelihood_decisions": len(load_original_labels()),
        "final_unique_labels": len(labels),
        "ablation_pair_universe": len(universe),
        "missing_labels": len(missing_labels),
        "extra_labels": len(extra_labels),
        "validation_errors": sum(row["severity"] == "error" for row in issues),
        "validation_warnings": sum(row["severity"] == "warning" for row in issues),
    }]
    write_csv(OUT / "adjudication_audit.csv", audit)
    print(f"labels={len(labels)} universe={len(universe)} errors={audit[0]['validation_errors']} warnings={audit[0]['validation_warnings']}")


if __name__ == "__main__":
    main()
