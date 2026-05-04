#!/usr/bin/env python3
"""
Publication run for patient->trial matching on the cardiology cohort.

Pipeline:
1) Enrich snapshot NCTs with eligibility criteria from ClinicalTrials.gov API v2.
2) Normalize trials and build the hybrid retrieval index.
3) Build structured patient profiles and retrieve/rerank candidates.
4) Evaluate criterion-level eligibility with calibrated decisioning.
5) Write review-ready artifacts plus regression/benchmark reports.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import List, Tuple

import requests
from dotenv import load_dotenv
from fhirclient.models.bundle import Bundle

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from matcher import compare_candidate_to_baseline
from matcher.pipeline import build_trial_matcher, evaluate_patient_against_trials, write_json
from patient_parser.patient_class import PatientData
from post_processing.post_processing import prepare_frontend_response


CTGOV_STUDY_URL = "https://clinicaltrials.gov/api/v2/studies/{nct_id}"


def now_utc() -> str:
    return datetime.now(UTC).isoformat()


def load_patient(path: Path) -> PatientData:
    with path.open() as handle:
        bundle = Bundle.with_json(json.load(handle))
    return PatientData(bundle)


def parse_trial_from_ctgov(study_json: dict) -> dict:
    protocol = study_json.get("protocolSection", {})
    ident = protocol.get("identificationModule", {})
    locations = protocol.get("locationsModule", {})
    conditions_mod = protocol.get("conditionsModule", {})
    elig_mod = protocol.get("eligibilityModule", {})
    desc_mod = protocol.get("descriptionModule", {})
    status_mod = protocol.get("statusModule", {})

    raw_conditions = conditions_mod.get("conditions", [])
    if isinstance(raw_conditions, list):
        conditions = "| ".join(raw_conditions)
    elif raw_conditions is None:
        conditions = ""
    else:
        conditions = str(raw_conditions)

    eligibility = elig_mod.get("eligibilityCriteria", "") or ""
    if isinstance(eligibility, str):
        eligibility = "\n        " + eligibility.replace("\n", "\n        ").replace("*", "-").replace("\\", "")

    return {
        "nct_id": ident.get("nctId", ""),
        "title": ident.get("officialTitle") or ident.get("briefTitle") or "",
        "has_us_facility": any(
            "United States" in (loc.get("country", "") or "")
            for loc in locations.get("locations", []) or []
        ),
        "conditions": conditions,
        "eligibility_criteria": eligibility,
        "brief_summary": "\n" + (desc_mod.get("briefSummary", "") or ""),
        "detailed_description": "\n" + (desc_mod.get("detailedDescription", "") or ""),
        "overall_status": status_mod.get("overallStatus", ""),
    }


def fetch_trial_with_retry(nct_id: str, retries: int = 4, timeout_sec: int = 30) -> dict | None:
    backoff = 1.0
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(CTGOV_STUDY_URL.format(nct_id=nct_id), timeout=timeout_sec)
            if resp.status_code == 200:
                return parse_trial_from_ctgov(resp.json())
            if resp.status_code == 404:
                return None
        except requests.RequestException:
            pass
        if attempt < retries:
            time.sleep(backoff)
            backoff *= 2
    return None


def read_existing_nct_ids_from_csv(csv_path: Path) -> set[str]:
    if not csv_path.exists():
        return set()
    with csv_path.open(newline="", encoding="utf-8") as handle:
        return {str(row.get("nct_id", "")).strip() for row in csv.DictReader(handle) if str(row.get("nct_id", "")).strip()}


def enrich_trials_csv(nct_ids: List[str], enriched_csv: Path, workers: int, retries: int) -> Tuple[int, int]:
    enriched_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "nct_id",
        "title",
        "has_us_facility",
        "conditions",
        "eligibility_criteria",
        "brief_summary",
        "detailed_description",
        "overall_status",
    ]
    existing = read_existing_nct_ids_from_csv(enriched_csv)
    pending = [nct for nct in nct_ids if nct not in existing]
    if not pending:
        print(f"[enrich] Reusing existing enriched CSV ({len(existing)} trials): {enriched_csv}")
        return len(existing), 0

    write_header = not enriched_csv.exists() or enriched_csv.stat().st_size == 0
    ok_count = 0
    fail_count = 0
    with enriched_csv.open("a", newline="", encoding="utf-8") as out_handle:
        writer = csv.DictWriter(out_handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(fetch_trial_with_retry, nct, retries): nct for nct in pending}
            for index, future in enumerate(as_completed(futures), start=1):
                trial = None
                try:
                    trial = future.result()
                except Exception:
                    trial = None
                if trial and trial.get("nct_id"):
                    writer.writerow(trial)
                    ok_count += 1
                else:
                    fail_count += 1
                if index % 100 == 0 or index == len(pending):
                    print(f"[enrich] {index}/{len(pending)} fetched | success={ok_count} fail={fail_count}", flush=True)
    return len(read_existing_nct_ids_from_csv(enriched_csv)), fail_count


def read_patient_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def read_snapshot_nct_ids(snapshot_json_path: Path) -> list[str]:
    if snapshot_json_path.suffix.lower() == ".csv":
        with snapshot_json_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    else:
        with snapshot_json_path.open() as handle:
            rows = json.load(handle)
        if isinstance(rows, dict):
            if isinstance(rows.get("rows"), list):
                rows = rows["rows"]
            elif isinstance(rows.get("studies"), list):
                rows = rows["studies"]
            else:
                rows = []
    seen: set[str] = set()
    deduped: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        nct = str(row.get("nct_id") or row.get("nctId") or "").strip()
        if nct and nct not in seen:
            seen.add(nct)
            deduped.append(nct)
    return deduped


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_pairs_tsv(path: Path, rows: list[list]) -> None:
    new_file = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        if new_file:
            writer.writerow(
                [
                    "patient_id",
                    "patient_rank",
                    "retrieval_rank",
                    "rerank_rank",
                    "nct_id",
                    "llm_eligible",
                    "llm_score",
                    "llm_raw_score",
                    "llm_max_possible",
                    "eligibility_probability",
                    "match_recommendation",
                    "age_sex_gate_status",
                    "retrieval_score_sparse",
                    "retrieval_score_dense",
                    "rerank_score",
                    "llm_rerank_score",
                    "disease_match_strength",
                    "intervention_match_strength",
                    "generic_trial_penalty",
                    "diversity_penalty",
                    "llm_rejection_reasons",
                    "trial_title",
                ]
            )
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Full publication cardio run with hybrid retrieval and calibrated decisioning.")
    parser.add_argument("--patient-ids", required=True)
    parser.add_argument("--patients-dir", required=True)
    parser.add_argument("--snapshot-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--candidate-pool", type=int, default=100)
    parser.add_argument("--shortlist-size", type=int, default=20)
    parser.add_argument("--llm-rerank-top-n", type=int, default=30)
    parser.add_argument("--llm-model", default="gpt-5-mini")
    parser.add_argument("--limit-patients", type=int, default=0)
    parser.add_argument("--enrich-workers", type=int, default=8)
    parser.add_argument("--enrich-retries", type=int, default=4)
    parser.add_argument("--baseline-pack-dir", default="")
    args = parser.parse_args()

    load_dotenv(dotenv_path=".env")

    patient_ids_path = Path(args.patient_ids)
    patients_dir = Path(args.patients_dir)
    snapshot_json_path = Path(args.snapshot_json)
    output_dir = Path(args.output_dir)

    ensure_dir(output_dir)
    ensure_dir(output_dir / "frontend")
    ensure_dir(output_dir / "evaluations")
    ensure_dir(output_dir / "summary")
    ensure_dir(output_dir / "audit")
    ensure_dir(output_dir / "benchmark")

    print("[run] Starting v2 publication pipeline", flush=True)
    print(f"[run] started_utc={now_utc()}", flush=True)

    nct_ids = read_snapshot_nct_ids(snapshot_json_path)
    enriched_csv = output_dir / "trials_enriched_from_snapshot.csv"
    total_enriched, enrich_fail_count = enrich_trials_csv(
        nct_ids=nct_ids,
        enriched_csv=enriched_csv,
        workers=max(1, args.enrich_workers),
        retries=max(1, args.enrich_retries),
    )
    print(f"[run] enriched trials available: {total_enriched} | fetch_failures_in_last_pass={enrich_fail_count}", flush=True)

    trials, matcher = build_trial_matcher(enriched_csv, max_candidates=args.candidate_pool)
    print(f"[run] normalized trials loaded={len(trials)}", flush=True)

    patient_ids = read_patient_ids(patient_ids_path)
    if args.limit_patients > 0:
        patient_ids = patient_ids[: args.limit_patients]

    pairs_tsv = output_dir / "summary" / "publication_patient_trial_pairs_topN.tsv"
    all_master_rows: list[dict] = []

    for index, patient_id in enumerate(patient_ids, start=1):
        patient_label = f"Patient_{patient_id}"
        patient_summary_json = output_dir / "summary" / f"{patient_label}_summary.json"
        if patient_summary_json.exists():
            print(f"[patient {index}/{len(patient_ids)}] {patient_label}: already complete, skipping", flush=True)
            continue

        patient_path = patients_dir / f"{patient_label}.json"
        if not patient_path.exists():
            print(f"[patient {index}/{len(patient_ids)}] {patient_label}: missing patient JSON, skipping", flush=True)
            continue

        patient = load_patient(patient_path)
        summary, evaluations, audit_rows = evaluate_patient_against_trials(
            patient=patient,
            matcher=matcher,
            shortlist_size=args.shortlist_size,
            candidate_pool=args.candidate_pool,
            llm_rerank_top_n=args.llm_rerank_top_n,
            llm_model=args.llm_model,
        )

        eval_path = output_dir / "evaluations" / f"evaluations_{patient_label}.json"
        frontend_path = output_dir / "frontend" / f"frontend_data_{patient_label}.json"
        audit_path = output_dir / "audit" / f"rank_trace_{patient_label}.json"
        write_json(eval_path, evaluations)
        write_json(frontend_path, prepare_frontend_response(evaluations, patient_label))
        write_json(audit_path, audit_rows)

        pair_rows: list[list] = []
        for rank, row in enumerate(evaluations, start=1):
            retrieval = row.get("retrieval_features", {})
            pair_rows.append(
                [
                    patient_id,
                    rank,
                    row.get("retrieval_rank"),
                    row.get("rerank_rank"),
                    row.get("nct_id"),
                    row.get("eligible"),
                    row.get("score"),
                    row.get("raw_score"),
                    row.get("max_possible"),
                    row.get("eligibility_probability"),
                    row.get("match_recommendation"),
                    row.get("age_sex_gate_status"),
                    retrieval.get("retrieval_score_sparse"),
                    retrieval.get("retrieval_score_dense"),
                    retrieval.get("rerank_score"),
                    retrieval.get("llm_rerank_score"),
                    retrieval.get("disease_match_strength"),
                    retrieval.get("intervention_match_strength"),
                    retrieval.get("generic_trial_penalty"),
                    retrieval.get("diversity_penalty"),
                    " | ".join(row.get("rejection_reasons", []) or []),
                    row.get("title"),
                ]
            )
        write_pairs_tsv(pairs_tsv, pair_rows)

        patient_summary = {
            "patient_id": patient_id,
            "patient_label": patient_label,
            "total_trials_scored": len(trials),
                "candidate_pool_size": summary["candidate_pool_size"],
                "top_n_for_review": summary["shortlist_size"],
                "eligible_count": summary["eligible_count"],
                "llm_rerank_used": summary["llm_rerank_used"],
                "llm_rerank_top_n": summary["llm_rerank_top_n"],
                "llm_model": summary["llm_model"],
                "generated_at_utc": now_utc(),
                "matcher_index_summary": matcher.export_index_summary(),
            "artifacts": {
                "evaluations_json": str(eval_path),
                "frontend_json": str(frontend_path),
                "audit_json": str(audit_path),
            },
        }
        write_json(patient_summary_json, patient_summary)
        all_master_rows.extend(evaluations)
        print(f"[patient {index}/{len(patient_ids)}] {patient_label}: complete", flush=True)

    meta_path = output_dir / "summary" / "run_meta.json"
    write_json(
        meta_path,
        {
            "generated_at_utc": now_utc(),
            "inputs": {
                "patient_ids": str(patient_ids_path),
                "patients_dir": str(patients_dir),
                "snapshot_json": str(snapshot_json_path),
            },
            "outputs": {
                "output_dir": str(output_dir),
                "enriched_trials_csv": str(enriched_csv),
                "pairs_topN_tsv": str(pairs_tsv),
            },
            "parameters": {
                "candidate_pool": args.candidate_pool,
                "shortlist_size": args.shortlist_size,
                "llm_rerank_top_n": args.llm_rerank_top_n,
                "llm_model": args.llm_model,
            },
            "counts": {
                "patients_requested": len(read_patient_ids(patient_ids_path)),
                "patients_processed_target": len(patient_ids),
                "snapshot_nct_ids": len(nct_ids),
                "enriched_trials_available": len(trials),
            },
        },
    )

    if args.baseline_pack_dir and all_master_rows:
        compare_candidate_to_baseline(
            baseline_pack_dir=Path(args.baseline_pack_dir),
            candidate_master_matches=all_master_rows,
            out_json=output_dir / "benchmark" / "benchmark_report.json",
            out_csv=output_dir / "benchmark" / "benchmark_report.csv",
        )

    print(f"[run] complete_utc={now_utc()}", flush=True)
    print(f"[run] meta={meta_path}", flush=True)


if __name__ == "__main__":
    main()
