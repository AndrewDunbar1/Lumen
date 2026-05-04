#!/usr/bin/env python3
"""
Output the top-N (default 5) clinical trial matches per patient.

Reads TrialGPT matching + aggregation results and saves a clean JSON
report to results/top5_cardiology.json

Usage:
  python top5_cardiology.py \
    results/matching_results_cardiology_gpt-4-turbo.json \
    results/aggregation_results_cardiology_gpt-4-turbo.json \
    [--n 5] [--output results/top5_cardiology.json]
"""

import argparse
import json
from pathlib import Path

CORPUS = "cardiology"
eps = 1e-9


# ---------------------------------------------------------------------------
# Scoring (from rank_results.py, unchanged)
# ---------------------------------------------------------------------------

def get_matching_score(matching: dict) -> float:
    included = not_inc = no_info_inc = 0
    excluded = 0

    inc_raw = matching.get("inclusion", {})
    exc_raw = matching.get("exclusion", {})
    if not isinstance(inc_raw, dict):
        inc_raw = {}
    if not isinstance(exc_raw, dict):
        exc_raw = {}

    for _, info in inc_raw.items():
        if not isinstance(info, (list, tuple)) or len(info) != 3:
            continue
        if info[2] == "included":
            included += 1
        elif info[2] == "not included":
            not_inc += 1
        elif info[2] == "not enough information":
            no_info_inc += 1

    for _, info in exc_raw.items():
        if not isinstance(info, (list, tuple)) or len(info) != 3:
            continue
        if info[2] == "excluded":
            excluded += 1

    score = included / (included + not_inc + no_info_inc + eps)
    if not_inc > 0:
        score -= 1
    if excluded > 0:
        score -= 1
    return score


def get_agg_score(assessment: dict) -> float:
    try:
        rel = float(assessment.get("relevance_score_R", 0))
        eli = float(assessment.get("eligibility_score_E", 0))
    except (TypeError, ValueError):
        rel = eli = 0.0
    return (rel + eli) / 100.0


# ---------------------------------------------------------------------------
# Load trial metadata for display
# ---------------------------------------------------------------------------

def load_trial_display(corpus: str) -> dict[str, dict]:
    info: dict[str, dict] = {}
    with open(f"dataset/{corpus}/corpus.jsonl") as f:
        for line in f:
            e = json.loads(line)
            m = e["metadata"]
            info[e["_id"]] = {
                "title": m.get("brief_title", e.get("title", "")),
                "conditions": m.get("diseases_list", []),
                "brief_summary": (m.get("brief_summary", "") or "")[:300],
                "phase": m.get("phase", ""),
            }
    return info


def load_queries(corpus: str) -> dict[str, str]:
    queries: dict[str, str] = {}
    with open(f"dataset/{corpus}/queries.jsonl") as f:
        for line in f:
            e = json.loads(line)
            queries[e["_id"]] = e["text"]
    return queries


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("matching_results", help="Path to matching_results_*.json")
    parser.add_argument("aggregation_results", help="Path to aggregation_results_*.json")
    parser.add_argument("--n", type=int, default=5, help="Number of top matches (default 5)")
    parser.add_argument(
        "--output",
        default=f"results/top{5}_cardiology.json",
        help="Output JSON path",
    )
    args = parser.parse_args()
    args.output = args.output.replace("top5", f"top{args.n}")

    print(f"Loading matching results  : {args.matching_results}")
    matching_results = json.load(open(args.matching_results))
    print(f"Loading aggregation results: {args.aggregation_results}")
    agg_results = json.load(open(args.aggregation_results))
    trial_display = load_trial_display(CORPUS)
    queries = load_queries(CORPUS)

    report: dict[str, object] = {}
    patients_with_matches = 0

    for patient_id, label2trial2results in matching_results.items():
        trial2score: dict[str, float] = {}

        for _label, trial2results in label2trial2results.items():
            for trial_id, results in trial2results.items():
                m_score = get_matching_score(results)
                a_score = 0.0
                if patient_id in agg_results and trial_id in agg_results[patient_id]:
                    a_score = get_agg_score(agg_results[patient_id][trial_id])
                trial2score[trial_id] = m_score + a_score

        if not trial2score:
            continue

        sorted_trials = sorted(trial2score.items(), key=lambda x: -x[1])
        top_n = sorted_trials[: args.n]

        # Collect criterion-level summaries for display
        top_details = []
        for trial_id, score in top_n:
            matching_detail = {}
            for _label, trial2results in label2trial2results.items():
                if trial_id in trial2results:
                    matching_detail = trial2results[trial_id]
                    break

            inc = matching_detail.get("inclusion", {})
            exc = matching_detail.get("exclusion", {})
            if not isinstance(inc, dict):
                inc = {}
            if not isinstance(exc, dict):
                exc = {}

            met_inc = sum(
                1 for v in inc.values()
                if isinstance(v, (list, tuple)) and len(v) == 3 and v[2] == "included"
            )
            total_inc = sum(1 for v in inc.values() if isinstance(v, (list, tuple)) and len(v) == 3)
            met_exc = sum(
                1 for v in exc.values()
                if isinstance(v, (list, tuple)) and len(v) == 3 and v[2] == "excluded"
            )
            total_exc = sum(1 for v in exc.values() if isinstance(v, (list, tuple)) and len(v) == 3)

            agg_info = {}
            if patient_id in agg_results and trial_id in agg_results[patient_id]:
                agg_info = agg_results[patient_id][trial_id]

            disp = trial_display.get(trial_id, {})
            top_details.append(
                {
                    "rank": len(top_details) + 1,
                    "nct_id": trial_id,
                    "title": disp.get("title", trial_id),
                    "conditions": disp.get("conditions", []),
                    "phase": disp.get("phase", ""),
                    "brief_summary": disp.get("brief_summary", ""),
                    "composite_score": round(score, 4),
                    "matching_score": round(trial2score[trial_id] - get_agg_score(agg_info), 4),
                    "aggregation_score": round(get_agg_score(agg_info), 4),
                    "relevance_score_R": agg_info.get("relevance_score_R"),
                    "eligibility_score_E": agg_info.get("eligibility_score_E"),
                    "inclusion_criteria_met": f"{met_inc}/{total_inc}",
                    "exclusion_criteria_met": f"{met_exc}/{total_exc}",
                    "relevance_explanation": agg_info.get("relevance_explanation", ""),
                    "eligibility_explanation": agg_info.get("eligibility_explanation", ""),
                }
            )

        patient_text = queries.get(patient_id, "")
        report[patient_id] = {
            "patient_id": patient_id,
            "patient_summary": patient_text[:400] + "…" if len(patient_text) > 400 else patient_text,
            "total_trials_scored": len(trial2score),
            f"top_{args.n}_matches": top_details,
        }

        if top_details:
            patients_with_matches += 1

    # Save JSON report
    out_path = Path(args.output)
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))

    print(f"\n{'='*60}")
    print(f"Top-{args.n} matches saved → {out_path}")
    print(f"Patients with results : {patients_with_matches}")
    print()

    # Print a human-readable summary to stdout
    for patient_id, entry in report.items():
        print(f"Patient: {patient_id}")
        print(f"  Summary: {entry['patient_summary'][:120]}…")
        print(f"  Trials scored: {entry['total_trials_scored']}")
        for match in entry[f"top_{args.n}_matches"]:
            print(f"  [{match['rank']}] {match['nct_id']}  score={match['composite_score']:.3f}")
            print(f"      {match['title'][:80]}")
            print(f"      Incl met: {match['inclusion_criteria_met']}  "
                  f"Excl met: {match['exclusion_criteria_met']}  "
                  f"R={match['relevance_score_R']}  E={match['eligibility_score_E']}")
        print()
