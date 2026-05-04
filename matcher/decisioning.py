from __future__ import annotations

from .schemas import CandidateFeatureRow, CriterionEvaluation, MatchDecision, PatientProfile, TrialRecord


class MatchDecisionEngine:
    def decide(
        self,
        patient: PatientProfile,
        trial: TrialRecord,
        candidate: CandidateFeatureRow,
        criterion_evaluations: list[CriterionEvaluation],
        rerank_rank: int,
    ) -> MatchDecision:
        criteria_stats = {
            "inclusion": {"total": 0, "met": 0, "not_met": 0, "unknown": 0, "not_applicable": 0},
            "exclusion": {"total": 0, "met": 0, "not_met": 0, "unknown": 0, "not_applicable": 0},
        }
        hard_blockers: list[str] = []
        open_questions: list[str] = []
        clinical_score = 0.0
        max_possible = 0.0

        for evaluation in criterion_evaluations:
            bucket = criteria_stats.setdefault(
                evaluation.criterion_type,
                {"total": 0, "met": 0, "not_met": 0, "unknown": 0, "not_applicable": 0},
            )
            bucket["total"] += 1
            bucket[evaluation.status] = bucket.get(evaluation.status, 0) + 1
            if evaluation.status == "not_applicable":
                continue

            max_possible += 1.0
            if evaluation.criterion_type == "inclusion":
                if evaluation.status == "met":
                    clinical_score += 1.0 * evaluation.confidence
                elif evaluation.status == "unknown":
                    clinical_score += 0.45 * evaluation.confidence
                    open_questions.append(evaluation.criterion_text)
                elif evaluation.status == "not_met" and evaluation.blocker_type == "missing_key_inclusion":
                    hard_blockers.append(evaluation.criterion_text)
            else:
                if evaluation.status == "not_met":
                    clinical_score += 0.9 * evaluation.confidence
                elif evaluation.status == "unknown":
                    clinical_score += 0.4 * evaluation.confidence
                    open_questions.append(evaluation.criterion_text)
                elif evaluation.status == "met":
                    hard_blockers.append(evaluation.criterion_text)

        fit_ratio = (clinical_score / max_possible) if max_possible else 0.0
        retrieval_bonus = max(candidate.rerank_score, 0.0)
        evidence_completeness = 1.0 - (
            (criteria_stats["inclusion"].get("unknown", 0) + criteria_stats["exclusion"].get("unknown", 0))
            / max(1, criteria_stats["inclusion"]["total"] + criteria_stats["exclusion"]["total"])
        )
        eligibility_probability = max(0.0, min(1.0, 0.55 * fit_ratio + 0.25 * evidence_completeness + 0.20 * min(retrieval_bonus, 1.0)))

        if candidate.age_sex_gate_status != "compatible":
            hard_blockers.append(candidate.age_sex_gate_status)

        if hard_blockers:
            recommendation = "unlikely_match"
        elif eligibility_probability >= 0.78 and evidence_completeness >= 0.6:
            recommendation = "strong_match"
        elif evidence_completeness < 0.45 or open_questions:
            recommendation = "manual_review"
        elif eligibility_probability >= 0.55:
            recommendation = "possible_match"
        else:
            recommendation = "unlikely_match"

        eligible = recommendation in {"strong_match", "possible_match", "manual_review"} and not hard_blockers
        normalized_score = round(eligibility_probability * 100, 2)
        short_reason = (
            f"{recommendation.replace('_', ' ')}; disease={candidate.disease_match_strength:.2f}; "
            f"intervention={candidate.intervention_match_strength:.2f}; blockers={len(hard_blockers)}"
        )

        raw_criteria = []
        for evaluation in criterion_evaluations:
            raw_criteria.append(
                {
                    "criterion": evaluation.criterion_text,
                    "type": evaluation.criterion_type,
                    "status": evaluation.status.replace("_", " "),
                    "confidence": evaluation.confidence,
                    "evidence": evaluation.evidence_span_text,
                    "sources_used": evaluation.evidence_source_refs,
                    "criterion_id": evaluation.criterion_id,
                    "criterion_group_id": evaluation.criterion_group_id,
                    "criterion_priority": evaluation.criterion_priority,
                    "blocker_type": evaluation.blocker_type,
                    "decision_role": evaluation.decision_role,
                }
            )

        return MatchDecision(
            patient_id=patient.patient_id,
            nct_id=trial.nct_id,
            title=trial.title,
            retrieval_rank=candidate.retrieval_rank,
            rerank_rank=rerank_rank,
            eligible=eligible,
            score=normalized_score,
            raw_score=round(clinical_score, 4),
            max_possible=round(max_possible, 4),
            eligibility_probability=round(eligibility_probability, 6),
            clinical_fit_score=round(fit_ratio, 6),
            evidence_completeness_score=round(evidence_completeness, 6),
            match_recommendation=recommendation,
            age_sex_gate_status=candidate.age_sex_gate_status,
            hard_blockers=hard_blockers,
            open_questions=open_questions[:8],
            rejection_reasons=hard_blockers.copy(),
            system_rationale_short=short_reason,
            criteria_stats=criteria_stats,
            raw_criteria_results=raw_criteria,
            retrieval_features=candidate.to_dict(),
        )
