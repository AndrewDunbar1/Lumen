#!/usr/bin/env python3
"""
Parallel TrialGPT aggregation — resumes from cache, skips malformed results.

Usage:
  python run_aggregation_parallel.py <matching_results_path> <model> [--workers 20]
"""

import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "trialgpt_ranking"))
from TrialGPT import trialgpt_aggregation  # noqa: E402

CORPUS = "cardiology"

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("matching_results_path")
parser.add_argument("model")
parser.add_argument("--workers", type=int, default=20)
args = parser.parse_args()

model = args.model
output_path = f"results/aggregation_results_{CORPUS}_{model}.json"

# ── Load inputs ───────────────────────────────────────────────────────────────
print(f"Loading matching results: {args.matching_results_path}")
matching_results = json.load(open(args.matching_results_path))

print("Loading trial info from corpus.jsonl …")
trial2info: dict = {}
with open(f"dataset/{CORPUS}/corpus.jsonl") as f:
    for line in f:
        entry = json.loads(line)
        trial2info[entry["_id"]] = entry["metadata"]
print(f"  {len(trial2info):,} trials loaded")

print("Loading patient queries …")
queries: dict = {}
with open(f"dataset/{CORPUS}/queries.jsonl") as f:
    for line in f:
        e = json.loads(line)
        queries[e["_id"]] = e["text"]
print(f"  {len(queries)} patients loaded")

# ── Load existing cache ───────────────────────────────────────────────────────
if os.path.exists(output_path):
    output: dict = json.load(open(output_path))
    cached = sum(len(v) for v in output.values())
    print(f"Resuming — {cached} pairs already cached")
else:
    output = {}
    cached = 0

write_lock = threading.Lock()

def save():
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

# ── Build work list ───────────────────────────────────────────────────────────
tasks = []
skipped_no_trial = 0
skipped_cached = 0

for patient_id, label2trial2results in matching_results.items():
    patient_text = queries.get(patient_id, "")
    if not patient_text:
        continue

    with write_lock:
        if patient_id not in output:
            output[patient_id] = {}

    for label, trial2results in label2trial2results.items():
        for trial_id, results in trial2results.items():
            if trial_id in output.get(patient_id, {}):
                skipped_cached += 1
                continue
            if trial_id not in trial2info:
                skipped_no_trial += 1
                continue

            # Sanitize: ensure inclusion/exclusion are dicts (not strings)
            if not isinstance(results, dict):
                continue
            clean_results = {
                "inclusion": results.get("inclusion") if isinstance(results.get("inclusion"), dict) else {},
                "exclusion": results.get("exclusion") if isinstance(results.get("exclusion"), dict) else {},
            }

            tasks.append((patient_id, patient_text, trial_id, clean_results))

total = len(tasks) + skipped_cached
print(f"Total pairs    : {total}")
print(f"Already cached : {skipped_cached}")
print(f"No trial info  : {skipped_no_trial}")
print(f"To process     : {len(tasks)}")
print(f"Workers        : {args.workers}")
print()

# ── Worker ────────────────────────────────────────────────────────────────────
def process_pair(task):
    patient_id, patient_text, trial_id, results = task
    try:
        agg = trialgpt_aggregation(patient_text, results, trial2info[trial_id], model)
        with write_lock:
            output[patient_id][trial_id] = agg
            save()
        return ("ok", patient_id, trial_id)
    except Exception as exc:
        return ("err", patient_id, trial_id, str(exc))

# ── Run ───────────────────────────────────────────────────────────────────────
completed = skipped_cached
errors = 0

with ThreadPoolExecutor(max_workers=args.workers) as pool:
    futures = {pool.submit(process_pair, t): t for t in tasks}
    for fut in as_completed(futures):
        res = fut.result()
        completed += 1
        if res[0] == "err":
            errors += 1
            print(f"  [ERR] {res[1]} / {res[2]}: {res[3]}", flush=True)
        if completed % 100 == 0 or completed == total:
            pairs_done = sum(len(v) for v in output.values())
            print(f"  {completed}/{total} tasks  |  {pairs_done} pairs saved  |  {errors} errors",
                  flush=True)

print(f"\n✓ Aggregation complete — {output_path}")
