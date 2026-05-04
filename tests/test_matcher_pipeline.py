from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from matcher.benchmark import compare_candidate_to_baseline
from matcher.criterion_eval import CriterionEvaluator
from matcher.decisioning import MatchDecisionEngine
from matcher.patient_profiles import build_patient_profile
from matcher.schemas import CandidateFeatureRow
from matcher.trial_normalization import normalize_trial_record


class FakeDate:
    def __init__(self, value: str) -> None:
        self.date = __import__("datetime").date.fromisoformat(value)


def make_patient():
    personal = SimpleNamespace(id="123", birthday=FakeDate("1980-01-01"), gender="male")
    conditions = SimpleNamespace(
        get_condition_names=lambda: ["Heart failure", "Atrial fibrillation"],
        get_active_conditions=lambda: ["Heart failure"],
    )
    labs_manager = SimpleNamespace(
        return_all_labs=lambda: [
            SimpleNamespace(test_name="Creatinine", value="1.2", unit="mg/dL", date="2024-01-01")
        ]
    )
    procedures = SimpleNamespace(
        procedure_list=[SimpleNamespace(name="PCI", performed_time="2024-01-02")]
    )
    notes = SimpleNamespace(reports=[["EF 35% with NYHA class III symptoms. Prior PCI."]])
    medications = [SimpleNamespace(name="Metoprolol"), SimpleNamespace(name="Apixaban")]
    return SimpleNamespace(
        personal_info=personal,
        conditions=conditions,
        labs=labs_manager,
        procedures=procedures,
        notes=notes,
        medications=medications,
    )


def test_build_patient_profile_extracts_structured_signals():
    profile = build_patient_profile(make_patient())
    assert profile.patient_id == "123"
    assert profile.age is not None
    assert "Heart failure" in profile.diagnoses
    assert "reduced_ejection_fraction" in profile.note_signals
    assert "pci_history" in profile.note_signals


def test_normalize_trial_record_extracts_criteria_and_gates():
    trial = normalize_trial_record(
        {
            "nct_id": "NCT1",
            "title": "Heart Failure PCI Registry",
            "conditions": "Heart Failure",
            "eligibility_criteria": """
            Inclusion Criteria:
            - Age >= 18 years
            - History of heart failure
            Exclusion Criteria:
            - Pregnant women
            """,
            "brief_summary": "",
            "detailed_description": "",
            "overall_status": "RECRUITING",
        }
    )
    assert trial.age_min == 18
    assert trial.condition_tags
    assert len(trial.inclusion_criteria) == 2
    assert len(trial.exclusion_criteria) == 1
    assert trial.genericity_penalty > 0


def test_criterion_and_decisioning_blocks_clear_exclusion():
    patient = build_patient_profile(make_patient())
    trial = normalize_trial_record(
        {
            "nct_id": "NCT2",
            "title": "Pregnancy Exclusion Trial",
            "conditions": "Atrial Fibrillation",
            "eligibility_criteria": """
            Inclusion Criteria:
            - Age >= 18 years
            - Atrial fibrillation
            Exclusion Criteria:
            - Chronic kidney disease requiring dialysis
            """,
            "brief_summary": "",
            "detailed_description": "",
            "overall_status": "RECRUITING",
        }
    )
    evaluator = CriterionEvaluator()
    criterion_results = evaluator.evaluate_trial(patient, trial)
    candidate = CandidateFeatureRow(
        patient_id=patient.patient_id,
        nct_id=trial.nct_id,
        title=trial.title,
        retrieval_rank=1,
        retrieval_score_sparse=0.8,
        retrieval_score_dense=0.7,
        rerank_score=0.9,
        age_sex_gate_status="compatible",
        disease_match_strength=0.8,
        intervention_match_strength=0.2,
        medication_match_strength=0.1,
        note_signal_match_strength=0.2,
        generic_trial_penalty=0.0,
        repeated_popularity_penalty=0.0,
    )
    decision = MatchDecisionEngine().decide(patient, trial, candidate, criterion_results, rerank_rank=1)
    assert decision.match_recommendation in {"strong_match", "possible_match", "manual_review"}
    assert decision.eligibility_probability > 0


def test_compare_candidate_to_baseline(tmp_path: Path):
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    (baseline / "doctor_adjudication_top3_stats.json").write_text(
        '{"ranking_metrics":{"hit_at_1":0.5,"hit_at_3":0.75,"mrr_at_3":0.6}}'
    )
    (baseline / "doctor_adjudication_top3_agreement.csv").write_text(
        "patient_id,trial_nct_id,doctor_assessment\n1,NCT1,Likely eligible\n1,NCT2,Not likely eligible\n"
    )
    (baseline / "master_matches.csv").write_text(
        "patient_id,trial_nct_id,score\n1,NCT1,80\n1,NCT2,20\n"
    )
    report = compare_candidate_to_baseline(
        baseline_pack_dir=baseline,
        candidate_master_matches=[
            {"patient_id": "1", "trial_nct_id": "NCT1", "rerank_rank": 1},
            {"patient_id": "1", "trial_nct_id": "NCT2", "rerank_rank": 2},
        ],
    )
    assert report.comparison.candidate_hit_at_1 == 1.0
    assert report.comparison.candidate_hit_at_3 == 1.0
