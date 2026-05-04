#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from matcher import compare_candidate_to_baseline


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare candidate matcher outputs to the adjudicated baseline.")
    parser.add_argument("--baseline-pack-dir", required=True)
    parser.add_argument("--candidate-master-matches", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()

    report = compare_candidate_to_baseline(
        baseline_pack_dir=Path(args.baseline_pack_dir),
        candidate_master_matches=read_csv(Path(args.candidate_master_matches)),
        out_json=Path(args.output_json),
        out_csv=Path(args.output_csv),
    )
    print(f"Candidate hit@3: {report.comparison.candidate_hit_at_3:.4f}")
    print(f"Baseline hit@3: {report.comparison.baseline_hit_at_3:.4f}")
    print(f"Wrote: {args.output_json}")
    print(f"Wrote: {args.output_csv}")


if __name__ == "__main__":
    main()
