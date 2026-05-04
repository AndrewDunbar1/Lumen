from __future__ import annotations

import json
import logging
import os
from typing import Any

from openai import OpenAI

from .schemas import CandidateFeatureRow, PatientProfile, TrialRecord

logger = logging.getLogger(__name__)


class LLMTrialReranker:
    def __init__(self, model_name: str = "gpt-5-mini") -> None:
        self.model_name = model_name
        self.api_key = os.getenv("OPENAI_API_KEY", "").strip()
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None

    @property
    def enabled(self) -> bool:
        return self.client is not None

    def rerank(
        self,
        patient: PatientProfile,
        candidate_rows: list[CandidateFeatureRow],
        trials_by_id: dict[str, TrialRecord],
    ) -> dict[str, dict[str, Any]]:
        if not self.enabled or not candidate_rows:
            return {}

        patient_summary = {
            "patient_id": patient.patient_id,
            "age": patient.age,
            "sex": patient.sex,
            "clinical_tags": patient.clinical_tags,
            "diagnoses": patient.normalized_diagnoses[:20],
            "procedures": patient.procedures[:20],
            "medications": patient.medications[:15],
            "signals": patient.note_signals[:15],
        }

        trials_payload = []
        for row in candidate_rows:
            trial = trials_by_id[row.nct_id]
            trials_payload.append(
                {
                    "nct_id": trial.nct_id,
                    "title": trial.title,
                    "condition_tags": trial.condition_tags[:12],
                    "intervention_tags": trial.intervention_tags[:12],
                    "disease_tags": trial.disease_tags[:12],
                    "brief_summary": (trial.brief_summary or "")[:500],
                    "inclusion_preview": [criterion.text for criterion in trial.inclusion_criteria[:4]],
                    "exclusion_preview": [criterion.text for criterion in trial.exclusion_criteria[:3]],
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

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                timeout=120,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(user_prompt)},
                ],
            )
            content = (response.choices[0].message.content or "").strip()
            payload = json.loads(self._clean_json(content))
        except Exception as exc:
            logger.warning(
                "LLM reranker failed for patient %s: %s",
                patient.patient_id,
                exc,
            )
            return {}

        if not isinstance(payload, dict) or not isinstance(payload.get("trials"), list):
            return {}

        reranked: dict[str, dict[str, Any]] = {}
        for item in payload["trials"]:
            if not isinstance(item, dict):
                continue
            nct_id = str(item.get("nct_id") or "").strip()
            if not nct_id:
                continue
            try:
                llm_score = float(item.get("llm_rerank_score"))
            except (TypeError, ValueError):
                continue
            reranked[nct_id] = {
                "llm_rerank_score": max(0.0, min(1.0, llm_score)),
                "specific_fit_reason": str(item.get("specific_fit_reason") or "").strip(),
            }
        return reranked

    def _clean_json(self, text: str) -> str:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.startswith("json"):
                cleaned = cleaned[4:].strip()
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end >= start:
            return cleaned[start : end + 1]
        return cleaned
