from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .criterion_eval import CriterionEvaluator
from .decisioning import MatchDecisionEngine
from .llm_reranker import LLMTrialReranker
from .patient_profiles import build_patient_profile
from .retrieval import HybridTrialMatcher
from .trial_normalization import normalize_trial_record


def load_trials_from_csv(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_trial_matcher(csv_path: Path, max_candidates: int = 100) -> tuple[list, HybridTrialMatcher]:
    rows = load_trials_from_csv(csv_path)
    trials = [normalize_trial_record(row) for row in rows if str(row.get("nct_id", "")).strip()]
    matcher = HybridTrialMatcher(trials=trials, max_candidates=max_candidates)
    return trials, matcher


def evaluate_patient_against_trials(
    patient,
    matcher: HybridTrialMatcher,
    shortlist_size: int = 20,
    candidate_pool: int = 100,
    llm_rerank_top_n: int = 0,
    llm_model: str = "gpt-5-mini",
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    profile = build_patient_profile(patient)
    candidates = matcher.generate_candidates(profile, top_k=candidate_pool)
    rerank_pool = matcher.rerank_candidates(profile, candidates, top_n=max(shortlist_size, llm_rerank_top_n, 1))
    llm_reranker = LLMTrialReranker(model_name=llm_model)

    llm_used = False
    if llm_rerank_top_n > 0 and llm_reranker.enabled:
        llm_input = rerank_pool[:llm_rerank_top_n]
        llm_scores = llm_reranker.rerank(
            patient=profile,
            candidate_rows=llm_input,
            trials_by_id={trial.nct_id: trial for trial in matcher.trials},
        )
        for candidate in rerank_pool:
            if candidate.nct_id in llm_scores:
                candidate.llm_rerank_score = llm_scores[candidate.nct_id]["llm_rerank_score"]
                candidate.llm_rerank_reason = llm_scores[candidate.nct_id]["specific_fit_reason"]
                candidate.rerank_score += 0.65 * candidate.llm_rerank_score
                candidate.feature_debug["llm_rerank_score"] = candidate.llm_rerank_score
                candidate.feature_debug["llm_rerank_reason"] = candidate.llm_rerank_reason
        rerank_pool = sorted(rerank_pool, key=lambda row: row.rerank_score, reverse=True)
        llm_used = bool(llm_scores)

    shortlist = rerank_pool[:shortlist_size]

    criterion_evaluator = CriterionEvaluator()
    decision_engine = MatchDecisionEngine()
    evaluations: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []

    for rerank_rank, candidate in enumerate(shortlist, start=1):
        trial = matcher.get_trial(candidate.nct_id)
        criterion_results = criterion_evaluator.evaluate_trial(profile, trial)
        decision = decision_engine.decide(
            patient=profile,
            trial=trial,
            candidate=candidate,
            criterion_evaluations=criterion_results,
            rerank_rank=rerank_rank,
        )
        payload = decision.to_dict()
        payload["raw_criteria_results"] = decision.raw_criteria_results
        evaluations.append(payload)

        audit_rows.append(
            {
                "patient_id": profile.patient_id,
                "trial_nct_id": trial.nct_id,
                "trial_title": trial.title,
                "retrieval_rank": candidate.retrieval_rank,
                "rerank_rank": rerank_rank,
                "retrieval_score_sparse": candidate.retrieval_score_sparse,
                "retrieval_score_dense": candidate.retrieval_score_dense,
                "rerank_score": candidate.rerank_score,
                "age_sex_gate_status": candidate.age_sex_gate_status,
                "disease_match_strength": candidate.disease_match_strength,
                "intervention_match_strength": candidate.intervention_match_strength,
                "medication_match_strength": candidate.medication_match_strength,
                "note_signal_match_strength": candidate.note_signal_match_strength,
                "required_signal_match_strength": candidate.required_signal_match_strength,
                "missing_required_signal_penalty": candidate.missing_required_signal_penalty,
                "llm_rerank_score": candidate.llm_rerank_score,
                "llm_rerank_reason": candidate.llm_rerank_reason,
                "generic_trial_penalty": candidate.generic_trial_penalty,
                "repeated_popularity_penalty": candidate.repeated_popularity_penalty,
                "diversity_penalty": candidate.diversity_penalty,
                "match_recommendation": decision.match_recommendation,
                "eligibility_probability": decision.eligibility_probability,
                "clinical_fit_score": decision.clinical_fit_score,
                "evidence_completeness_score": decision.evidence_completeness_score,
                "hard_blockers": " | ".join(decision.hard_blockers),
                "open_questions": " | ".join(decision.open_questions),
                "system_rationale_short": decision.system_rationale_short,
            }
        )

    summary = {
        "patient_id": profile.patient_id,
        "candidate_pool_size": len(candidates),
        "shortlist_size": len(shortlist),
        "eligible_count": len([item for item in evaluations if item.get("eligible")]),
        "llm_rerank_used": llm_used,
        "llm_rerank_top_n": llm_rerank_top_n,
        "llm_model": llm_model,
        "profile": profile.to_dict(),
    }
    return summary, evaluations, audit_rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))
