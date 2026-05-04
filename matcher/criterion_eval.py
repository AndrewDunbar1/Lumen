from __future__ import annotations

import re

from .schemas import CriterionEvaluation, PatientProfile, TrialCriterion, TrialRecord
from .text_utils import normalize_text


class CriterionEvaluator:
    def __init__(self) -> None:
        self._affirmative_patterns = {
            "pregnancy": ["pregnant", "pregnancy", "breastfeeding"],
            "malignancy": ["cancer", "malignancy", "metastatic"],
            "renal_impairment": ["ckd", "chronic kidney disease", "dialysis", "renal failure"],
            "heart_failure": ["heart failure", "reduced ejection fraction", "preserved ejection fraction"],
            "atrial_fibrillation": ["atrial fibrillation", "atrial flutter", "afib"],
            "pci": ["pci", "stent", "angioplasty"],
            "cabg": ["cabg", "coronary artery bypass"],
            "tavr": ["tavr", "tavi", "transcatheter aortic valve"],
            "hemodialysis": ["hemodialysis", "dialysis"],
            "dialysis_access_malfunction": ["dialysis access malfunction", "access malfunction"],
            "fistulogram": ["fistulogram"],
            "ivus": ["ivus", "intravascular ultrasound"],
            "nirs_ivus": ["nirs", "near-infrared spectroscopy", "near infrared spectroscopy", "nirs-ivus"],
            "acute_decompensated_heart_failure": ["acute decompensated heart failure", "adhf"],
            "acute_coronary_syndrome": ["acute coronary syndrome", "acs", "stemi", "nstemi"],
            "drug_eluting_stent": ["drug-eluting stent", "drug eluting stent"],
            "cardiorenal_syndrome": ["cardiorenal syndrome"],
        }

    def evaluate_trial(self, patient: PatientProfile, trial: TrialRecord) -> list[CriterionEvaluation]:
        evaluations: list[CriterionEvaluation] = []
        for criterion in trial.inclusion_criteria + trial.exclusion_criteria:
            evaluations.append(self._evaluate_criterion(patient, criterion))
        return evaluations

    def _evaluate_criterion(self, patient: PatientProfile, criterion: TrialCriterion) -> CriterionEvaluation:
        lowered = normalize_text(criterion.text)
        if criterion.is_administrative:
            return CriterionEvaluation(
                criterion_id=criterion.criterion_id,
                criterion_group_id=criterion.group_id,
                criterion_text=criterion.text,
                criterion_type=criterion.criterion_type,
                criterion_priority=criterion.priority,
                blocker_type="administrative",
                decision_role="supporting",
                status="not_applicable",
                confidence=0.8,
                evidence_span_text="Administrative or consent criterion deferred to manual screening.",
                evidence_source_refs=[],
                rationale="Administrative criterion should not auto-reject an otherwise good clinical match.",
                tags=criterion.tags,
            )

        status = "unknown"
        confidence = 0.45
        evidence_text = ""
        evidence_refs: list[str] = []
        blocker_type = "none"
        decision_role = "supporting"

        age_eval = self._evaluate_age(patient, lowered)
        if age_eval is not None:
            status, confidence, evidence_text = age_eval
            decision_role = "blocking"
        sex_eval = self._evaluate_sex(patient, lowered)
        if sex_eval is not None:
            status, confidence, evidence_text = sex_eval
            decision_role = "blocking"

        if status == "unknown":
            keyword_eval = self._evaluate_keyword_match(patient, criterion, lowered)
            if keyword_eval is not None:
                status, confidence, evidence_text, evidence_refs, blocker_type = keyword_eval
                decision_role = "blocking" if blocker_type != "none" or criterion.priority == "high" else "supporting"

        if criterion.is_or_group and status == "not_met":
            status = "unknown"
            confidence = min(confidence, 0.55)
            evidence_text = "Criterion appears to be an OR-group; deterministic evidence was insufficient for a hard negative."

        if criterion.criterion_type == "exclusion" and status == "met":
            blocker_type = blocker_type if blocker_type != "none" else "explicit_exclusion"
        if criterion.criterion_type == "inclusion" and status == "not_met" and criterion.priority == "high":
            blocker_type = blocker_type if blocker_type != "none" else "missing_key_inclusion"

        return CriterionEvaluation(
            criterion_id=criterion.criterion_id,
            criterion_group_id=criterion.group_id,
            criterion_text=criterion.text,
            criterion_type=criterion.criterion_type,
            criterion_priority=criterion.priority,
            blocker_type=blocker_type,
            decision_role=decision_role,
            status=status,
            confidence=round(confidence, 4),
            evidence_span_text=evidence_text,
            evidence_source_refs=evidence_refs,
            rationale=evidence_text,
            tags=criterion.tags,
        )

    def _evaluate_age(self, patient: PatientProfile, lowered: str) -> tuple[str, float, str] | None:
        if "age" not in lowered and "years" not in lowered:
            return None
        if patient.age is None:
            return "unknown", 0.4, "Patient age could not be derived reliably from the FHIR bundle."
        min_match = re.search(r"(?:older than|>=|at least|age\s*>=?)\s*(\d{1,3})", lowered)
        max_match = re.search(r"(?:younger than|<=|under|less than)\s*(\d{1,3})", lowered)
        if min_match and patient.age < int(min_match.group(1)):
            return "not_met", 0.95, f"Patient age {patient.age} is below the stated minimum age threshold."
        if max_match and patient.age > int(max_match.group(1)):
            return "not_met", 0.95, f"Patient age {patient.age} is above the stated maximum age threshold."
        return "met", 0.9, f"Patient age {patient.age} is compatible with the criterion."

    def _evaluate_sex(self, patient: PatientProfile, lowered: str) -> tuple[str, float, str] | None:
        sex = (patient.sex or "unknown").lower()
        if "female" in lowered and "male" not in lowered:
            if sex == "female":
                return "met", 0.95, "Patient sex is female and matches the sex-restricted criterion."
            if sex == "male":
                return "not_met", 0.95, "Patient sex is male and conflicts with a female-only criterion."
        if "male" in lowered and "female" not in lowered:
            if sex == "male":
                return "met", 0.95, "Patient sex is male and matches the sex-restricted criterion."
            if sex == "female":
                return "not_met", 0.95, "Patient sex is female and conflicts with a male-only criterion."
        return None

    def _evaluate_keyword_match(
        self,
        patient: PatientProfile,
        criterion: TrialCriterion,
        lowered: str,
    ) -> tuple[str, float, str, list[str], str] | None:
        evidence_texts = []
        evidence_refs = []
        patient_space = " ".join(
            patient.normalized_diagnoses + patient.diagnoses + patient.active_diagnoses + patient.procedures + patient.medications + patient.labs + patient.note_signals
        ).lower()
        for label, phrases in self._affirmative_patterns.items():
            if any(phrase in lowered for phrase in phrases):
                has_match = any(phrase in patient_space for phrase in phrases)
                if has_match:
                    for item in patient.evidence[:30]:
                        item_lower = item.text.lower()
                        if any(phrase in item_lower for phrase in phrases):
                            evidence_texts.append(item.text)
                            evidence_refs.append(item.source_ref)
                    evidence = evidence_texts[0] if evidence_texts else f"Patient evidence matched keyword group {label}."
                    return "met", 0.82, evidence, evidence_refs[:3], "explicit_exclusion" if criterion.criterion_type == "exclusion" else "none"
                blocker = "missing_key_inclusion" if criterion.criterion_type == "inclusion" and criterion.priority == "high" else "none"
                if label in {
                    "dialysis_access_malfunction",
                    "fistulogram",
                    "nirs_ivus",
                    "ivus",
                    "acute_decompensated_heart_failure",
                    "acute_coronary_syndrome",
                }:
                    blocker = "missing_key_inclusion" if criterion.criterion_type == "inclusion" else blocker
                negative_status = "not_met" if criterion.criterion_type == "inclusion" else "not_met"
                return negative_status, 0.7, f"No patient evidence was found for keyword group {label}.", [], blocker

        if criterion.tags:
            matched = [tag for tag in criterion.tags if tag in {value.lower() for value in patient.normalized_diagnoses + patient.note_signals + patient.procedures}]
            if matched:
                return "met", 0.74, f"Patient profile overlaps trial criterion tags: {', '.join(matched)}.", [], "none"
            if criterion.criterion_type == "inclusion" and criterion.priority == "high":
                return "unknown", 0.5, "Deterministic signals were insufficient to confirm a high-priority clinical inclusion.", [], "none"
        return None
