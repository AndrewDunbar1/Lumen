#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fhirclient.models.bundle import Bundle
from openai import OpenAI

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from matcher.patient_profiles import build_patient_profile
from patient_parser.patient_class import PatientData


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_trial_map(trials_csv: Path) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for row in read_csv(trials_csv):
        nct_id = str(row.get("nct_id") or "").strip()
        if not nct_id:
            continue
        out[nct_id] = row
    return out


def group_rows_by_patient(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        patient_id = str(row.get("patient_id") or "").strip()
        if patient_id:
            grouped[patient_id].append(row)
    return grouped


def load_summary_cache(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def load_patient(path: Path) -> PatientData:
    with path.open(encoding="utf-8") as handle:
        bundle = Bundle.with_json(json.load(handle))
    return PatientData(bundle)


def build_review_patient_context(patient_path: Path, patient_id: str) -> dict[str, Any]:
    patient = load_patient(patient_path)
    profile = build_patient_profile(patient, patient_id=patient_id)
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


class LLMTop5Reviewer:
    def __init__(self, model: str) -> None:
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for LLM top-5 review.")
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def review_patient(
        self,
        patient_id: str,
        patient_context: dict[str, Any],
        candidate_rows: list[dict[str, str]],
        trial_map: dict[str, dict[str, str]],
    ) -> dict[str, Any]:
        trials_payload: list[dict[str, Any]] = []
        for row in candidate_rows:
            nct_id = str(row.get("trial_nct_id") or row.get("nct_id") or "").strip()
            trial = trial_map.get(nct_id, {})
            trials_payload.append(
                {
                    "trial_nct_id": nct_id,
                    "trial_title": str(row.get("trial_title") or trial.get("title") or "").strip(),
                    "match_recommendation": str(row.get("match_recommendation") or "").strip(),
                    "eligibility_probability": str(row.get("eligibility_probability") or "").strip(),
                    "system_rationale_short": str(row.get("system_rationale_short") or "").strip(),
                    "open_questions": str(row.get("open_questions") or "").strip(),
                    "hard_blockers": str(row.get("hard_blockers") or "").strip(),
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
            "patient_context": patient_context,
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

        response = self.client.chat.completions.create(
            model=self.model,
            timeout=120,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_prompt)},
            ],
        )
        content = (response.choices[0].message.content or "").strip()
        return json.loads(content)


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM review pass over top-5 patient-trial matches.")
    parser.add_argument("--pack-dir", required=True)
    parser.add_argument("--trials-csv", required=True)
    parser.add_argument("--patients-dir", required=True)
    parser.add_argument("--summary-cache", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--limit-patients", type=int, default=0)
    args = parser.parse_args()

    pack_dir = Path(args.pack_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_csv(pack_dir / "master_matches.csv")
    grouped = group_rows_by_patient(rows)
    patient_ids = sorted(grouped.keys(), key=lambda x: int(x) if x.isdigit() else x)
    if args.limit_patients > 0:
        patient_ids = patient_ids[: args.limit_patients]

    summary_cache = load_summary_cache(Path(args.summary_cache)) if args.summary_cache else {}
    patients_dir = Path(args.patients_dir)
    trial_map = load_trial_map(Path(args.trials_csv))
    reviewer = LLMTop5Reviewer(model=args.model)

    raw_json_path = output_dir / "llm_top5_patient_reviews.json"
    pair_csv_path = output_dir / "llm_top5_pair_reviews.csv"
    patient_csv_path = output_dir / "llm_top5_patient_rollup.csv"
    high_csv_path = output_dir / "llm_top5_high_likelihood_matches.csv"

    existing_reviews: dict[str, Any] = {}
    if raw_json_path.exists():
        existing_reviews = json.loads(raw_json_path.read_text())

    for index, patient_id in enumerate(patient_ids, start=1):
        if patient_id in existing_reviews:
            continue
        patient_path = patients_dir / f"Patient_{patient_id}.json"
        if patient_path.exists():
            patient_context = build_review_patient_context(patient_path, patient_id)
        else:
            patient_summary = summary_cache.get(patient_id, "").strip()
            patient_context = {
                "patient_id": patient_id,
                "summary_fallback": patient_summary or f"Patient {patient_id} summary unavailable.",
            }
        review = reviewer.review_patient(
            patient_id=patient_id,
            patient_context=patient_context,
            candidate_rows=grouped[patient_id],
            trial_map=trial_map,
        )
        existing_reviews[patient_id] = review
        raw_json_path.write_text(json.dumps(existing_reviews, indent=2))
        print(f"[review] {index}/{len(patient_ids)} patient {patient_id}", flush=True)

    pair_rows: list[dict[str, Any]] = []
    patient_rows: list[dict[str, Any]] = []
    high_rows: list[dict[str, Any]] = []

    for patient_id in patient_ids:
        review = existing_reviews[patient_id]
        patient_rows.append(
            {
                "patient_id": patient_id,
                "has_high_likelihood_match": bool(review.get("has_high_likelihood_match")),
                "best_trial_nct_id": str(review.get("best_trial_nct_id") or "").strip(),
                "best_trial_label": str(review.get("best_trial_label") or "").strip(),
                "patient_level_reason": str(review.get("patient_level_reason") or "").strip(),
            }
        )
        trials = review.get("trials") or []
        for trial in trials:
            row = {
                "patient_id": patient_id,
                "trial_nct_id": str(trial.get("trial_nct_id") or "").strip(),
                "label": str(trial.get("label") or "").strip(),
                "confidence": trial.get("confidence"),
                "reason": str(trial.get("reason") or "").strip(),
            }
            pair_rows.append(row)
            if row["label"] == "high_likelihood":
                high_rows.append(row)

    write_csv(pair_csv_path, pair_rows)
    write_csv(patient_csv_path, patient_rows)
    write_csv(high_csv_path, high_rows)

    print(f"Wrote: {raw_json_path}")
    print(f"Wrote: {pair_csv_path}")
    print(f"Wrote: {patient_csv_path}")
    print(f"Wrote: {high_csv_path}")
    print(f"Patients reviewed: {len(patient_rows)}")
    print(f"High-likelihood pairs: {len(high_rows)}")


if __name__ == "__main__":
    main()
