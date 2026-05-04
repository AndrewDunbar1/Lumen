#!/usr/bin/env python3
"""
TrialGPT aggregation step for the cardiology dataset.

Replacement for trialgpt_ranking/run_aggregation.py that:
  - Does NOT need GenericDataLoader / qrels
  - Reads trial info from our corpus.jsonl instead of trial_info.json
    (so all 11,231 cardiology trials are covered)

Usage:
  python run_cardiology_aggregation.py \
    results/matching_results_cardiology_gpt-4-turbo.json \
    gpt-4-turbo
"""

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "trialgpt_ranking"))
from TrialGPT import trialgpt_aggregation  # noqa: E402

CORPUS = "cardiology"


def load_trial2info(corpus: str) -> dict[str, dict]:
    """Load trial metadata from corpus.jsonl (covers all trials, not just NCBI snapshot)."""
    trial2info: dict[str, dict] = {}
    with open(f"dataset/{corpus}/corpus.jsonl") as f:
        for line in f:
            entry = json.loads(line)
            nct_id = entry["_id"]
            m = entry["metadata"]
            trial2info[nct_id] = m
    return trial2info


def load_queries(corpus: str) -> dict[str, str]:
    """Return {patient_id: text} from queries.jsonl."""
    queries: dict[str, str] = {}
    with open(f"dataset/{corpus}/queries.jsonl") as f:
        for line in f:
            e = json.loads(line)
            queries[e["_id"]] = e["text"]
    return queries


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python run_cardiology_aggregation.py <matching_results_path> <model>")
        sys.exit(1)

    matching_results_path = sys.argv[1]
    model = sys.argv[2]

    print(f"Loading matching results: {matching_results_path}")
    matching_results = json.load(open(matching_results_path))

    print("Loading trial info from corpus.jsonl …")
    trial2info = load_trial2info(CORPUS)
    print(f"  {len(trial2info):,} trials loaded")

    print("Loading patient queries …")
    queries = load_queries(CORPUS)
    print(f"  {len(queries)} patients loaded")

    output_path = f"results/aggregation_results_{CORPUS}_{model}.json"
    if os.path.exists(output_path):
        output = json.load(open(output_path))
        print(f"Resuming from existing output ({len(output)} patients already done)")
    else:
        output = {}

    skipped_trials = 0
    processed = 0

    for patient_id, label2trial2results in matching_results.items():
        patient_text = queries.get(patient_id, "")
        if not patient_text:
            print(f"  [WARN] No patient text for {patient_id}, skipping")
            continue

        if patient_id not in output:
            output[patient_id] = {}

        for label, trial2results in label2trial2results.items():
            for trial_id, results in trial2results.items():
                # skip if already aggregated
                if trial_id in output[patient_id]:
                    continue

                if trial_id not in trial2info:
                    skipped_trials += 1
                    continue

                try:
                    # Signature: trialgpt_aggregation(patient, trial_results, trial_info, model)
                    agg_result = trialgpt_aggregation(
                        patient_text,
                        results,
                        trial2info[trial_id],
                        model,
                    )
                    output[patient_id][trial_id] = agg_result
                    processed += 1

                    with open(output_path, "w") as f:
                        json.dump(output, f, indent=4)

                except Exception as exc:
                    print(f"  [ERROR] {patient_id} / {trial_id}: {exc}")
                    continue

    print(f"\n✓ Aggregation complete")
    print(f"  Processed  : {processed} patient-trial pairs")
    print(f"  Skipped    : {skipped_trials} (trial not in corpus)")
    print(f"  Output     : {output_path}")
