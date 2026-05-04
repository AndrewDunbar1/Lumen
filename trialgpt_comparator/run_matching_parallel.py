#!/usr/bin/env python3
"""
Parallel TrialGPT matching — drops-in for trialgpt_matching/run_matching.py
but uses a thread pool so many pairs run concurrently.

Usage:
  python run_matching_parallel.py <corpus> <model> [--workers 20]
"""

import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from nltk.tokenize import sent_tokenize

sys.path.insert(0, str(Path(__file__).parent / "trialgpt_matching"))
from TrialGPT import trialgpt_matching  # noqa: E402

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("corpus")
parser.add_argument("model")
parser.add_argument("--workers", type=int, default=20,
                    help="Concurrent API threads (default 20)")
args = parser.parse_args()

corpus = args.corpus
model  = args.model
workers = args.workers

output_path = f"results/matching_results_{corpus}_{model}.json"
dataset     = json.load(open(f"dataset/{corpus}/retrieved_trials.json"))

# Load existing cache
if os.path.exists(output_path):
    output: dict = json.load(open(output_path))
    print(f"Resuming — {sum(len(v.get('1',{})) for v in output.values())} pairs already cached")
else:
    output = {}

write_lock = threading.Lock()
done_count = 0
skip_count = 0

def save():
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

# ── Build work list ───────────────────────────────────────────────────────────
tasks = []   # (patient_id, formatted_patient, label, trial)

for instance in dataset:
    patient_id = instance["patient_id"]
    patient    = instance["patient"]
    sents = sent_tokenize(patient)
    sents.append(
        "The patient will provide informed consent, and will comply "
        "with the trial protocol without any practical issues."
    )
    sents   = [f"{idx}. {sent}" for idx, sent in enumerate(sents)]
    patient = "\n".join(sents)

    with write_lock:
        if patient_id not in output:
            output[patient_id] = {"0": {}, "1": {}, "2": {}}

    for label in ["2", "1", "0"]:
        if label not in instance:
            continue
        for trial in instance[label]:
            trial_id = trial["NCTID"]
            with write_lock:
                if trial_id in output[patient_id][label]:
                    skip_count += 1
                    continue
            tasks.append((patient_id, patient, label, trial, trial_id))

total = len(tasks) + skip_count
print(f"Total pairs  : {total}")
print(f"Already done : {skip_count}")
print(f"To process   : {len(tasks)}")
print(f"Workers      : {workers}")
print()

# ── Worker function ───────────────────────────────────────────────────────────
def process_pair(task):
    patient_id, patient, label, trial, trial_id = task
    try:
        result = trialgpt_matching(trial, patient, model)
        with write_lock:
            output[patient_id][label][trial_id] = result
            save()
        return ("ok", patient_id, trial_id)
    except Exception as exc:
        return ("err", patient_id, trial_id, str(exc))

# ── Run ───────────────────────────────────────────────────────────────────────
completed = skip_count
errors    = 0

with ThreadPoolExecutor(max_workers=workers) as pool:
    futures = {pool.submit(process_pair, t): t for t in tasks}
    for fut in as_completed(futures):
        res = fut.result()
        completed += 1
        if res[0] == "err":
            errors += 1
            print(f"  [ERR] {res[1]} / {res[2]}: {res[3]}", flush=True)
        if completed % 50 == 0 or completed == total:
            pairs_done = sum(len(v.get("1", {})) for v in output.values())
            print(f"  {completed}/{total} tasks done  |  {pairs_done} pairs in file  |  {errors} errors",
                  flush=True)

print(f"\n✓ Matching complete — {output_path}")
