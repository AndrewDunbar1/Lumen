from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class BenchmarkComparison:
    baseline_hit_at_1: float
    baseline_hit_at_3: float
    baseline_mrr_at_3: float
    candidate_hit_at_1: float
    candidate_hit_at_3: float
    candidate_mrr_at_3: float
    candidate_patients_with_match: int
    baseline_patients_with_match: int
    top_false_negative_trials: list[dict[str, Any]] = field(default_factory=list)
    repeated_candidate_trials: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkReport:
    baseline_summary: dict[str, Any]
    candidate_summary: dict[str, Any]
    comparison: BenchmarkComparison

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_summary": self.baseline_summary,
            "candidate_summary": self.candidate_summary,
            "comparison": self.comparison.to_dict(),
        }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _doctor_truth_map(agreement_rows: list[dict[str, str]]) -> dict[tuple[str, str], bool]:
    truth = {}
    for row in agreement_rows:
        key = (str(row.get("patient_id", "")).strip(), str(row.get("trial_nct_id", "")).strip())
        truth[key] = str(row.get("doctor_assessment", "")).strip() == "Likely eligible"
    return truth


def _score_rows(rows: list[dict[str, Any]], truth_map: dict[tuple[str, str], bool], rank_field: str = "rank") -> tuple[float, float, float, int, Counter[str]]:
    by_patient: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_patient[str(row["patient_id"])].append(row)

    hit1 = 0
    hit3 = 0
    mrr = 0.0
    false_neg_trials: Counter[str] = Counter()
    with_match = 0

    for patient_id, patient_rows in by_patient.items():
        ranked = sorted(patient_rows, key=lambda item: int(item.get(rank_field, 9999)))
        found_rank = None
        for index, row in enumerate(ranked[:3], start=1):
            key = (patient_id, str(row["trial_nct_id"]))
            if truth_map.get(key):
                found_rank = index
                break
        if found_rank == 1:
            hit1 += 1
        if found_rank is not None:
            hit3 += 1
            with_match += 1
            mrr += 1.0 / found_rank
        else:
            for row in ranked[:3]:
                false_neg_trials[str(row["trial_nct_id"])] += 1

    denom = max(1, len(by_patient))
    return hit1 / denom, hit3 / denom, mrr / denom, with_match, false_neg_trials


def compare_candidate_to_baseline(
    baseline_pack_dir: Path,
    candidate_master_matches: list[dict[str, Any]],
    out_json: Path | None = None,
    out_csv: Path | None = None,
) -> BenchmarkReport:
    agreement_rows = _read_csv(baseline_pack_dir / "doctor_adjudication_top3_agreement.csv")
    truth_map = _doctor_truth_map(agreement_rows)
    baseline_stats = json.loads((baseline_pack_dir / "doctor_adjudication_top3_stats.json").read_text())
    baseline_master = _read_csv(baseline_pack_dir / "master_matches.csv")

    baseline_rows = []
    by_patient_count: defaultdict[str, int] = defaultdict(int)
    for row in baseline_master:
        patient_id = str(row["patient_id"])
        by_patient_count[patient_id] += 1
        baseline_rows.append(
            {
                "patient_id": patient_id,
                "trial_nct_id": str(row["trial_nct_id"]),
                "rank": int(row.get("retrieval_rank") or row.get("rerank_rank") or by_patient_count[patient_id]),
            }
        )

    candidate_rows = []
    for row in candidate_master_matches:
        candidate_rows.append(
            {
                "patient_id": str(row["patient_id"]),
                "trial_nct_id": str(row.get("trial_nct_id") or row.get("nct_id") or ""),
                "rank": int(row.get("rerank_rank") or row.get("retrieval_rank") or 9999),
            }
        )

    b_hit1, b_hit3, b_mrr, b_with_match, _ = _score_rows(baseline_rows, truth_map)
    c_hit1, c_hit3, c_mrr, c_with_match, false_neg_trials = _score_rows(candidate_rows, truth_map)
    repeated_trials = Counter(row["trial_nct_id"] for row in candidate_rows if int(row["rank"]) <= 3)

    comparison = BenchmarkComparison(
        baseline_hit_at_1=b_hit1,
        baseline_hit_at_3=b_hit3,
        baseline_mrr_at_3=b_mrr,
        candidate_hit_at_1=c_hit1,
        candidate_hit_at_3=c_hit3,
        candidate_mrr_at_3=c_mrr,
        candidate_patients_with_match=c_with_match,
        baseline_patients_with_match=b_with_match,
        top_false_negative_trials=[
            {"trial_nct_id": nct_id, "count": count}
            for nct_id, count in false_neg_trials.most_common(15)
        ],
        repeated_candidate_trials=[
            {"trial_nct_id": nct_id, "count": count}
            for nct_id, count in repeated_trials.most_common(15)
        ],
    )
    report = BenchmarkReport(
        baseline_summary=baseline_stats,
        candidate_summary={
            "patient_count": len({row["patient_id"] for row in candidate_rows}),
            "top3_rows": len([row for row in candidate_rows if int(row["rank"]) <= 3]),
        },
        comparison=comparison,
    )

    if out_json is not None:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(report.to_dict(), indent=2))

    if out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["metric", "baseline", "candidate"])
            writer.writerow(["hit_at_1", b_hit1, c_hit1])
            writer.writerow(["hit_at_3", b_hit3, c_hit3])
            writer.writerow(["mrr_at_3", b_mrr, c_mrr])
            writer.writerow(["patients_with_match", b_with_match, c_with_match])

    return report
