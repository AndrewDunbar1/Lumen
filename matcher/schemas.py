from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class PatientEvidence:
    source_type: str
    source_ref: str
    text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PatientProfile:
    patient_id: str
    age: int | None
    sex: str
    demographics_text: str
    diagnoses: list[str] = field(default_factory=list)
    normalized_diagnoses: list[str] = field(default_factory=list)
    active_diagnoses: list[str] = field(default_factory=list)
    procedures: list[str] = field(default_factory=list)
    medications: list[str] = field(default_factory=list)
    labs: list[str] = field(default_factory=list)
    note_signals: list[str] = field(default_factory=list)
    clinical_tags: list[str] = field(default_factory=list)
    normalized_patient_text: str = ""
    evidence: list[PatientEvidence] = field(default_factory=list)
    sparse_text: str = ""
    narrative_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["evidence"] = [item.to_dict() for item in self.evidence]
        return payload


@dataclass
class TrialCriterion:
    criterion_id: str
    group_id: str
    text: str
    criterion_type: str
    priority: str
    is_administrative: bool
    is_or_group: bool
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TrialRecord:
    nct_id: str
    title: str
    conditions_text: str
    overall_status: str
    brief_summary: str
    detailed_description: str
    eligibility_criteria: str
    inclusion_criteria: list[TrialCriterion] = field(default_factory=list)
    exclusion_criteria: list[TrialCriterion] = field(default_factory=list)
    condition_tags: list[str] = field(default_factory=list)
    intervention_tags: list[str] = field(default_factory=list)
    disease_tags: list[str] = field(default_factory=list)
    age_min: int | None = None
    age_max: int | None = None
    sex: str = "all"
    genericity_penalty: float = 0.0
    required_patient_signals: list[str] = field(default_factory=list)
    sparse_text: str = ""
    dense_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["inclusion_criteria"] = [item.to_dict() for item in self.inclusion_criteria]
        payload["exclusion_criteria"] = [item.to_dict() for item in self.exclusion_criteria]
        return payload


@dataclass
class CandidateFeatureRow:
    patient_id: str
    nct_id: str
    title: str
    retrieval_rank: int
    retrieval_score_sparse: float
    retrieval_score_dense: float
    rerank_score: float
    age_sex_gate_status: str
    disease_match_strength: float
    intervention_match_strength: float
    medication_match_strength: float
    note_signal_match_strength: float
    generic_trial_penalty: float
    repeated_popularity_penalty: float
    required_signal_match_strength: float = 0.0
    missing_required_signal_penalty: float = 0.0
    diversity_penalty: float = 0.0
    llm_rerank_score: float | None = None
    llm_rerank_reason: str = ""
    feature_debug: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CriterionEvaluation:
    criterion_id: str
    criterion_group_id: str
    criterion_text: str
    criterion_type: str
    criterion_priority: str
    blocker_type: str
    decision_role: str
    status: str
    confidence: float
    evidence_span_text: str
    evidence_source_refs: list[str] = field(default_factory=list)
    rationale: str = ""
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MatchDecision:
    patient_id: str
    nct_id: str
    title: str
    retrieval_rank: int
    rerank_rank: int
    eligible: bool
    score: float
    raw_score: float
    max_possible: float
    eligibility_probability: float
    clinical_fit_score: float
    evidence_completeness_score: float
    match_recommendation: str
    age_sex_gate_status: str
    hard_blockers: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    rejection_reasons: list[str] = field(default_factory=list)
    system_rationale_short: str = ""
    criteria_stats: dict[str, dict[str, int]] = field(default_factory=dict)
    raw_criteria_results: list[dict[str, Any]] = field(default_factory=list)
    retrieval_features: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
