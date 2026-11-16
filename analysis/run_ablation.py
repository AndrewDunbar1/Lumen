#!/usr/bin/env python3
"""Execute frozen Lumen component-removal ablations with API telemetry."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from fhirclient.models.bundle import Bundle
from openai import OpenAI


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("LUMEN_DATA_DIR", REPO_ROOT / "PublicationData"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from matcher.criterion_eval import CriterionEvaluator
from matcher.decisioning import MatchDecisionEngine
from matcher.patient_profiles import build_patient_profile
from matcher.pipeline import build_trial_matcher
from patient_parser.patient_class import PatientData


FROZEN_PATIENT_IDS = DATA_ROOT / "MIMIC_FHIR_Patients/cardiology_trial_top200_patient_ids.txt"
FROZEN_PATIENTS_DIR = DATA_ROOT / "MIMIC_FHIR_Patients/patients"
FROZEN_TRIALS = DATA_ROOT / "Cardio_Trials_Snapshot/full_publication_run_top20_top5_200_manual/trials_enriched_from_snapshot.csv"
BASELINE_PACK = DATA_ROOT / "Cardio_Trials_Snapshot/full_publication_run_top20_top5_200_manual/adjudication_pack_top5/master_matches.csv"
BASELINE_REVIEWS = DATA_ROOT / "Cardio_Trials_Snapshot/full_publication_run_top20_top5_200_manual/llm_top5_review/llm_top5_patient_reviews.json"

MODEL = "gpt-5-mini"
CANDIDATE_POOL = 100
RERANK_POOL = 20
TOP_K = 5
INPUT_USD_PER_M = 0.125
OUTPUT_USD_PER_M = 1.00
CURRENT_INPUT_USD_PER_M = 0.25
CURRENT_CACHED_INPUT_USD_PER_M = 0.025
CURRENT_OUTPUT_USD_PER_M = 2.00


def now_utc() -> str:
    return datetime.now(UTC).isoformat()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))


def load_patient(path: Path) -> PatientData:
    with path.open(encoding="utf-8") as handle:
        return PatientData(Bundle.with_json(json.load(handle)))


def patient_context(patient_path: Path, patient_id: str) -> dict[str, Any]:
    profile = build_patient_profile(load_patient(patient_path), patient_id=patient_id)
    return {
        "patient_id": profile.patient_id,
        "age": profile.age,
        "sex": profile.sex,
        "demographics_text": profile.demographics_text,
        "diagnoses": profile.diagnoses,
        "normalized_diagnoses": profile.normalized_diagnoses,
        "active_diagnoses": profile.active_diagnoses,
        "procedures": profile.procedures,
        "medications": profile.medications,
        "labs": profile.labs[:40],
        "note_signals": profile.note_signals,
        "clinical_tags": profile.clinical_tags,
        "normalized_patient_text": profile.normalized_patient_text[:12000],
        "narrative_text": profile.narrative_text[:12000],
        "evidence": [item.to_dict() for item in profile.evidence[:80]],
    }


class TelemetryOpenAI:
    """Counts logical calls, HTTP attempts, latency, response model, and token usage."""

    def __init__(self, model: str) -> None:
        self.model = model
        self.records: list[dict[str, Any]] = []
        self.http_attempts = 0

        def on_request(_request: httpx.Request) -> None:
            self.http_attempts += 1

        http_client = httpx.Client(event_hooks={"request": [on_request]})
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], http_client=http_client)

    def create(self, stage: str, patient_id: str, **kwargs: Any):
        started = time.perf_counter()
        before_attempts = self.http_attempts
        try:
            response = self.client.chat.completions.create(**kwargs)
        except Exception as exc:
            self.records.append(
                {
                    "stage": stage,
                    "patient_id": patient_id,
                    "success": False,
                    "error": repr(exc),
                    "seconds": round(time.perf_counter() - started, 3),
                    "http_attempts": self.http_attempts - before_attempts,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "cached_input_tokens": 0,
                    "requested_model": kwargs.get("model"),
                    "response_model": "",
                }
            )
            raise

        usage = response.usage
        prompt_details = getattr(usage, "prompt_tokens_details", None) if usage else None
        self.records.append(
            {
                "stage": stage,
                "patient_id": patient_id,
                "success": True,
                "error": "",
                "seconds": round(time.perf_counter() - started, 3),
                "http_attempts": self.http_attempts - before_attempts,
                "input_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
                "output_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
                "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
                "cached_input_tokens": int(getattr(prompt_details, "cached_tokens", 0) or 0),
                "requested_model": kwargs.get("model"),
                "response_model": getattr(response, "model", "") or "",
            }
        )
        return response

    def summary(self) -> dict[str, Any]:
        input_tokens = sum(row["input_tokens"] for row in self.records)
        output_tokens = sum(row["output_tokens"] for row in self.records)
        cached_input_tokens = sum(row["cached_input_tokens"] for row in self.records)
        uncached_input_tokens = max(0, input_tokens - cached_input_tokens)
        return {
            "logical_calls": len(self.records),
            "successful_calls": sum(bool(row["success"]) for row in self.records),
            "failed_calls": sum(not bool(row["success"]) for row in self.records),
            "http_attempts": self.http_attempts,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": sum(row["total_tokens"] for row in self.records),
            "cached_input_tokens": cached_input_tokens,
            "cost_usd_at_frozen_submitted_rates": round(
                input_tokens / 1_000_000 * INPUT_USD_PER_M
                + output_tokens / 1_000_000 * OUTPUT_USD_PER_M,
                6,
            ),
            "rates_usd_per_million_tokens": {
                "input": INPUT_USD_PER_M,
                "output": OUTPUT_USD_PER_M,
            },
            "cost_usd_at_execution_date_rates": round(
                uncached_input_tokens / 1_000_000 * CURRENT_INPUT_USD_PER_M
                + cached_input_tokens / 1_000_000 * CURRENT_CACHED_INPUT_USD_PER_M
                + output_tokens / 1_000_000 * CURRENT_OUTPUT_USD_PER_M,
                6,
            ),
            "execution_date_rates_usd_per_million_tokens": {
                "input": CURRENT_INPUT_USD_PER_M,
                "cached_input": CURRENT_CACHED_INPUT_USD_PER_M,
                "output": CURRENT_OUTPUT_USD_PER_M,
            },
            "response_models": dict(Counter(row["response_model"] for row in self.records if row["response_model"])),
            "stage_counts": dict(Counter(row["stage"] for row in self.records)),
            "records": self.records,
        }


def clean_json(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    return cleaned[start : end + 1] if start != -1 and end >= start else cleaned


def display_field(value: Any) -> str:
    if isinstance(value, list):
        return " | ".join(str(item).strip() for item in value if str(item).strip())
    return str(value or "").strip()


def llm_rerank(telemetry: TelemetryOpenAI, profile, rows, trials_by_id) -> dict[str, dict[str, Any]]:
    patient_summary = {
        "patient_id": profile.patient_id,
        "age": profile.age,
        "sex": profile.sex,
        "clinical_tags": profile.clinical_tags,
        "diagnoses": profile.normalized_diagnoses[:20],
        "procedures": profile.procedures[:20],
        "medications": profile.medications[:15],
        "signals": profile.note_signals[:15],
    }
    trials_payload = []
    for row in rows:
        trial = trials_by_id[row.nct_id]
        trials_payload.append(
            {
                "nct_id": trial.nct_id,
                "title": trial.title,
                "condition_tags": trial.condition_tags[:12],
                "intervention_tags": trial.intervention_tags[:12],
                "disease_tags": trial.disease_tags[:12],
                "brief_summary": (trial.brief_summary or "")[:500],
                "inclusion_preview": [c.text for c in trial.inclusion_criteria[:4]],
                "exclusion_preview": [c.text for c in trial.exclusion_criteria[:3]],
                "deterministic_rerank_score": row.rerank_score,
                "age_sex_gate_status": row.age_sex_gate_status,
            }
        )
    system_prompt = (
        "You are reranking cardiology clinical trials for a patient. "
        "Favor specific clinical fit over broad plausibility. "
        "Do not reward generic observational studies just because they are broadly relevant. "
        "Return strict JSON only."
    )
    user_prompt = {
        "task": (
            "For each candidate trial, assign a rerank score from 0.0 to 1.0 for how likely "
            "the trial is to be a truly good top-3 match for this patient. "
            "Prioritize specific disease/intervention alignment, likely real eligibility, and avoid broad overmatching."
        ),
        "patient": patient_summary,
        "candidate_trials": trials_payload,
        "output_schema": {
            "trials": [
                {
                    "nct_id": "string",
                    "llm_rerank_score": "float 0.0-1.0",
                    "specific_fit_reason": "short string",
                }
            ]
        },
    }
    response = telemetry.create(
        "llm_reranker",
        profile.patient_id,
        model=MODEL,
        timeout=120,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_prompt)},
        ],
    )
    payload = json.loads(clean_json((response.choices[0].message.content or "").strip()))
    output: dict[str, dict[str, Any]] = {}
    for item in payload.get("trials", []):
        try:
            score = max(0.0, min(1.0, float(item.get("llm_rerank_score"))))
        except (TypeError, ValueError):
            continue
        nct_id = str(item.get("nct_id") or "").strip()
        if nct_id:
            output[nct_id] = {
                "llm_rerank_score": score,
                "specific_fit_reason": str(item.get("specific_fit_reason") or "").strip(),
            }
    return output


def evaluate_rows(profile, matcher, ranked_rows) -> list[dict[str, Any]]:
    criterion_evaluator = CriterionEvaluator()
    decision_engine = MatchDecisionEngine()
    evaluations: list[dict[str, Any]] = []
    for rank, candidate in enumerate(ranked_rows, start=1):
        trial = matcher.get_trial(candidate.nct_id)
        criteria = criterion_evaluator.evaluate_trial(profile, trial)
        decision = decision_engine.decide(profile, trial, candidate, criteria, rank)
        payload = decision.to_dict()
        payload["raw_criteria_results"] = decision.raw_criteria_results
        evaluations.append(payload)
    return evaluations


def second_pass_review(
    telemetry: TelemetryOpenAI,
    patient_id: str,
    context: dict[str, Any],
    rows: list[dict[str, Any]],
    trial_map: dict[str, dict[str, str]],
    remove_decision_context: bool = False,
) -> dict[str, Any]:
    trials_payload = []
    for row in rows:
        nct_id = str(row.get("trial_nct_id") or row.get("nct_id") or "").strip()
        trial = trial_map.get(nct_id, {})
        trials_payload.append(
            {
                "trial_nct_id": nct_id,
                "trial_title": str(row.get("trial_title") or row.get("title") or trial.get("title") or "").strip(),
                "match_recommendation": "" if remove_decision_context else display_field(row.get("match_recommendation")),
                "eligibility_probability": "" if remove_decision_context else display_field(row.get("eligibility_probability")),
                "system_rationale_short": "" if remove_decision_context else display_field(row.get("system_rationale_short")),
                "open_questions": "" if remove_decision_context else display_field(row.get("open_questions")),
                "hard_blockers": "" if remove_decision_context else display_field(row.get("hard_blockers")),
                "eligibility_criteria": str(trial.get("eligibility_criteria") or "").strip()[:5000],
            }
        )
    system_prompt = (
        "You are a clinical trial eligibility reviewer. "
        "You are reviewing the final top-5 candidate trials for one patient. "
        "Focus on core clinical fit only. "
        "Assume procedural/logistics items like consent, scheduling, obtaining extra labs, extra imaging, and routine screening can be completed unless there is a direct clinical contradiction. "
        "Do not assume away hard clinical contradictions like wrong disease, wrong sex, wrong age, dialysis requirement when not on dialysis, cancer trial for non-cancer patient, or highly specific procedure history that is absent."
    )
    user_prompt = {
        "task": (
            "For each of the 5 trials, classify the patient-trial pair as one of: "
            "'high_likelihood', 'possible', 'unlikely', or 'no_match'. "
            "Then decide whether this patient has at least one 'high_likelihood' trial in the top 5. "
            "Only call a pair 'high_likelihood' if the core disease/intervention fit is strong and there is no obvious clinical contradiction. "
            "Call it 'possible' if it is clinically plausible but still uncertain. "
            "Return strict JSON only."
        ),
        "patient_id": patient_id,
        "patient_context": context,
        "candidate_trials": trials_payload,
        "output_schema": {
            "patient_id": "string",
            "has_high_likelihood_match": "boolean",
            "best_trial_nct_id": "string or empty",
            "best_trial_label": "high_likelihood|possible|unlikely|no_match",
            "patient_level_reason": "short string",
            "trials": [
                {
                    "trial_nct_id": "string",
                    "label": "high_likelihood|possible|unlikely|no_match",
                    "confidence": "float 0.0-1.0",
                    "reason": "short string",
                }
            ],
        },
    }
    response = telemetry.create(
        "second_pass_review_no_decision_context" if remove_decision_context else "second_pass_review",
        patient_id,
        model=MODEL,
        timeout=120,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_prompt)},
        ],
    )
    return json.loads(clean_json((response.choices[0].message.content or "").strip()))


def trial_map_from_csv(path: Path) -> dict[str, dict[str, str]]:
    return {row["nct_id"].strip(): row for row in read_csv(path) if row.get("nct_id", "").strip()}


def flatten_reviews(reviews: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for patient_id, review in reviews.items():
        for trial in review.get("trials") or []:
            rows.append(
                {
                    "patient_id": patient_id,
                    "trial_nct_id": str(trial.get("trial_nct_id") or "").strip(),
                    "second_pass_label": str(trial.get("label") or "").strip(),
                    "second_pass_confidence": trial.get("confidence"),
                    "second_pass_reason": str(trial.get("reason") or "").strip(),
                }
            )
    return rows


def run_rank_ablation(
    mode: str,
    output_dir: Path,
    telemetry: TelemetryOpenAI,
    limit_patients: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    index_t0 = time.perf_counter()
    trials, matcher = build_trial_matcher(FROZEN_TRIALS, max_candidates=CANDIDATE_POOL)
    index_seconds = time.perf_counter() - index_t0
    trials_by_id = {trial.nct_id: trial for trial in trials}
    patient_ids = [line.strip() for line in FROZEN_PATIENT_IDS.read_text().splitlines() if line.strip()]
    if limit_patients > 0:
        patient_ids = patient_ids[:limit_patients]
    eval_dir = output_dir / "evaluations_top20"
    eval_dir.mkdir(parents=True, exist_ok=True)

    matching_t0 = time.perf_counter()
    top5_rows: list[dict[str, Any]] = []
    patient_seconds: dict[str, float] = {}
    for index, patient_id in enumerate(patient_ids, start=1):
        patient_t0 = time.perf_counter()
        patient = load_patient(FROZEN_PATIENTS_DIR / f"Patient_{patient_id}.json")
        profile = build_patient_profile(patient, patient_id=patient_id)
        candidates = matcher.generate_candidates(profile, top_k=CANDIDATE_POOL)

        if mode == "no_llm_reranker":
            ranked = matcher.rerank_candidates(profile, candidates, top_n=RERANK_POOL)
        elif mode == "no_deterministic_reranker":
            ranked = candidates[:RERANK_POOL]
            try:
                scores = llm_rerank(telemetry, profile, ranked, trials_by_id)
            except Exception as exc:
                print(f"[warning] LLM rerank failed for {patient_id}: {exc!r}", flush=True)
                scores = {}
            for candidate in ranked:
                if candidate.nct_id in scores:
                    candidate.llm_rerank_score = scores[candidate.nct_id]["llm_rerank_score"]
                    candidate.llm_rerank_reason = scores[candidate.nct_id]["specific_fit_reason"]
                    candidate.rerank_score += 0.65 * candidate.llm_rerank_score
            ranked = sorted(ranked, key=lambda row: row.rerank_score, reverse=True)
        else:
            raise ValueError(mode)

        evaluations = evaluate_rows(profile, matcher, ranked)
        write_json(eval_dir / f"evaluations_Patient_{patient_id}.json", evaluations)
        for rank, row in enumerate(evaluations[:TOP_K], start=1):
            flat = dict(row)
            flat["patient_id"] = patient_id
            flat["trial_nct_id"] = row["nct_id"]
            flat["trial_title"] = row["title"]
            flat["top5_rank"] = rank
            top5_rows.append(flat)
        patient_seconds[patient_id] = round(time.perf_counter() - patient_t0, 3)
        print(f"[{mode}] matching {index}/{len(patient_ids)} patient={patient_id} seconds={patient_seconds[patient_id]}", flush=True)

    matching_seconds = time.perf_counter() - matching_t0
    write_csv(output_dir / "top5_pairs.csv", top5_rows)

    trial_map = trial_map_from_csv(FROZEN_TRIALS)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in top5_rows:
        grouped.setdefault(row["patient_id"], []).append(row)
    reviews: dict[str, Any] = {}
    review_t0 = time.perf_counter()
    for index, patient_id in enumerate(patient_ids, start=1):
        context = patient_context(FROZEN_PATIENTS_DIR / f"Patient_{patient_id}.json", patient_id)
        try:
            reviews[patient_id] = second_pass_review(telemetry, patient_id, context, grouped[patient_id], trial_map)
        except Exception as exc:
            print(f"[error] second-pass failed for {patient_id}: {exc!r}", flush=True)
            reviews[patient_id] = {"patient_id": patient_id, "error": repr(exc), "trials": []}
        write_json(output_dir / "second_pass_reviews.json", reviews)
        print(f"[{mode}] review {index}/{len(patient_ids)} patient={patient_id}", flush=True)
    review_seconds = time.perf_counter() - review_t0
    write_csv(output_dir / "second_pass_pair_reviews.csv", flatten_reviews(reviews))
    return top5_rows, {
        "trial_index_seconds": round(index_seconds, 3),
        "matching_seconds": round(matching_seconds, 3),
        "second_pass_review_seconds": round(review_seconds, 3),
        "patient_matching_seconds": patient_seconds,
    }


def run_decision_context_analysis(
    output_dir: Path,
    telemetry: TelemetryOpenAI,
    limit_patients: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = read_csv(BASELINE_PACK)
    rows.sort(key=lambda row: (int(row["patient_id"]), int(row.get("rerank_rank") or 0)))
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["patient_id"], []).append(row)
    patient_ids = [line.strip() for line in FROZEN_PATIENT_IDS.read_text().splitlines() if line.strip()]
    if limit_patients > 0:
        patient_ids = patient_ids[:limit_patients]
        allowed = set(patient_ids)
        rows = [row for row in rows if row["patient_id"] in allowed]
        grouped = {}
        for row in rows:
            grouped.setdefault(row["patient_id"], []).append(row)
    write_csv(output_dir / "top5_pairs.csv", rows)
    trial_map = trial_map_from_csv(FROZEN_TRIALS)
    reviews: dict[str, Any] = {}
    review_t0 = time.perf_counter()
    for index, patient_id in enumerate(patient_ids, start=1):
        context = patient_context(FROZEN_PATIENTS_DIR / f"Patient_{patient_id}.json", patient_id)
        try:
            reviews[patient_id] = second_pass_review(
                telemetry,
                patient_id,
                context,
                grouped[patient_id],
                trial_map,
                remove_decision_context=True,
            )
        except Exception as exc:
            print(f"[error] decision-context review failed for {patient_id}: {exc!r}", flush=True)
            reviews[patient_id] = {"patient_id": patient_id, "error": repr(exc), "trials": []}
        write_json(output_dir / "second_pass_reviews.json", reviews)
        print(f"[decision_context_removed] review {index}/{len(patient_ids)} patient={patient_id}", flush=True)
    review_seconds = time.perf_counter() - review_t0
    write_csv(output_dir / "second_pass_pair_reviews.csv", flatten_reviews(reviews))
    return rows, {
        "trial_index_seconds": 0.0,
        "matching_seconds": 0.0,
        "second_pass_review_seconds": round(review_seconds, 3),
        "patient_matching_seconds": {},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        required=True,
        choices=["no_llm_reranker", "no_deterministic_reranker", "decision_context_removed"],
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit-patients", type=int, default=0)
    args = parser.parse_args()

    load_dotenv(REPO_ROOT / ".env")
    if not os.getenv("OPENAI_API_KEY", "").strip():
        raise RuntimeError("OPENAI_API_KEY is required")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    started_at = now_utc()
    total_t0 = time.perf_counter()
    telemetry = TelemetryOpenAI(MODEL)

    if args.mode == "decision_context_removed":
        top5_rows, timings = run_decision_context_analysis(output_dir, telemetry, args.limit_patients)
    else:
        top5_rows, timings = run_rank_ablation(args.mode, output_dir, telemetry, args.limit_patients)

    telemetry_payload = telemetry.summary()
    write_json(output_dir / "api_telemetry.json", telemetry_payload)
    meta = {
        "mode": args.mode,
        "started_at_utc": started_at,
        "completed_at_utc": now_utc(),
        "total_wall_clock_seconds": round(time.perf_counter() - total_t0, 3),
        "timings": timings,
        "configuration": {
            "model": MODEL,
            "candidate_pool": CANDIDATE_POOL,
            "rerank_pool": RERANK_POOL,
            "top_k": TOP_K,
            "temperature": "API default; parameter omitted",
            "seed": "not set; production code did not set one",
            "response_format": "json_object",
            "timeout_seconds": 120,
            "OpenAI_client_max_retries": 2,
            "OpenAI_python_version": "1.109.1",
            "patient_processing": "serial, preserving production patient order and stateful popularity counter",
            "input_token_rate_usd_per_million": INPUT_USD_PER_M,
            "output_token_rate_usd_per_million": OUTPUT_USD_PER_M,
            "execution_date_input_token_rate_usd_per_million": CURRENT_INPUT_USD_PER_M,
            "execution_date_cached_input_token_rate_usd_per_million": CURRENT_CACHED_INPUT_USD_PER_M,
            "execution_date_output_token_rate_usd_per_million": CURRENT_OUTPUT_USD_PER_M,
        },
        "frozen_inputs": {
            "patient_ids": str(FROZEN_PATIENT_IDS),
            "patients_dir": str(FROZEN_PATIENTS_DIR),
            "trials_csv": str(FROZEN_TRIALS),
            "baseline_pack": str(BASELINE_PACK),
            "baseline_reviews": str(BASELINE_REVIEWS),
        },
        "counts": {
            "top5_pairs": len(top5_rows),
            "patients": len({str(row.get("patient_id")) for row in top5_rows}),
        },
        "api": {key: value for key, value in telemetry_payload.items() if key != "records"},
    }
    write_json(output_dir / "run_meta.json", meta)
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
