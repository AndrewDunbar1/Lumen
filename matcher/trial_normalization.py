from __future__ import annotations

import re

from .clinical_tags import extract_clinical_tags, extract_clinical_tags_from_normalized
from .schemas import TrialCriterion, TrialRecord
from .text_utils import normalize_space, normalize_text, unique_preserve_order


ADMIN_PATTERNS = [
    "informed consent",
    "willing to comply",
    "ability to comply",
    "follow-up requirements",
    "signed consent",
]

CONDITION_TAGS = {
    "heart failure": ["heart failure", "reduced ejection fraction", "preserved ejection fraction", "hf"],
    "atrial fibrillation": ["atrial fibrillation", "atrial flutter", "afib"],
    "coronary artery disease": ["coronary artery disease", "cad", "ischemic heart disease"],
    "myocardial infarction": ["myocardial infarction", "stemi", "nstemi", "acute coronary syndrome"],
    "aortic stenosis": ["aortic stenosis", "valve stenosis"],
    "hypertension": ["hypertension", "blood pressure"],
    "pulmonary hypertension": ["pulmonary hypertension"],
    "cardiorenal syndrome": ["cardiorenal syndrome"],
}

INTERVENTION_TAGS = {
    "pci": ["pci", "stent", "angioplasty", "percutaneous coronary intervention"],
    "cabg": ["cabg", "coronary artery bypass"],
    "tavr": ["tavr", "tavi", "transcatheter aortic valve"],
    "valve_repair": ["valve replacement", "valve repair", "aortic valve replacement", "mitral valve repair"],
    "catheterization": ["angiography", "catheterization", "right heart catheterization"],
}

GENERIC_TRIAL_PATTERNS = [
    "registry",
    "repository",
    "survey",
    "observational",
    "cohort",
    "database",
    "quality improvement",
    "longitudinal",
]

REQUIRED_SIGNAL_PATTERNS = {
    "hemodialysis": ["hemodialysis", "renal dialysis", "dialysis"],
    "dialysis_access_malfunction": ["dialysis access malfunction", "access malfunction", "failing dialysis access"],
    "fistulogram": ["fistulogram"],
    "ivus": ["ivus", "intravascular ultrasound"],
    "nirs_ivus": ["nirs ivus", "near infrared spectroscopy", "nirs-ivus"],
    "acute_decompensated_heart_failure": ["acute decompensated heart failure", "adhf"],
    "elevated_bnp": ["pro bnp", "pro-bnp", "bnp > 800", "nt probnp"],
    "acute_coronary_syndrome": ["acute coronary syndrome", "acs", "stemi", "nstemi"],
    "successful_pci": ["successful pci", "after successful pci", "post pci"],
    "drug_eluting_stent": ["drug eluting stent", "drug-eluting stent"],
    "primary_hypertension": ["primary hypertension"],
    "cardiorenal_syndrome": ["cardiorenal syndrome"],
    "diuretic_resistance": ["diuretic resistance"],
    "rotational_atherectomy": ["rotational atherectomy"],
    "calcified_coronary_lesion": ["calcified coronary lesion", "calcified lesion", "calcified nodule"],
    "carotid_stenting": ["carotid stenting", "carotid stenosis", "carotid revascularization"],
    "limb_artery_disease": ["limb arteries", "peripheral artery", "lower extremity artery"],
}


def split_criteria_blocks(text: str) -> tuple[str, str]:
    text = (text or "").replace("\\", " ")
    inclusion_match = re.search(
        r"inclusion criteria:(.*?)(?=exclusion criteria:)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    exclusion_match = re.search(
        r"exclusion criteria:(.*)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    inclusion = inclusion_match.group(1).strip().replace("\r", "\n") if inclusion_match else ""
    exclusion = exclusion_match.group(1).strip().replace("\r", "\n") if exclusion_match else ""
    if not inclusion and not exclusion:
        inclusion = (text or "").strip()
    return inclusion, exclusion


def _criterion_priority(text: str, is_admin: bool) -> str:
    if is_admin:
        return "administrative"
    lowered = normalize_text(text)
    if any(token in lowered for token in ["must", "required", "diagnosis of", "history of", "age", "male", "female"]):
        return "high"
    if any(token in lowered for token in ["one of the following", "at least", "prior", "current treatment"]):
        return "medium"
    return "low"


def _extract_tags(text: str, mapping: dict[str, list[str]]) -> list[str]:
    lowered = normalize_text(text)
    tags = []
    for canonical, variants in mapping.items():
        if any(variant in lowered for variant in variants):
            tags.append(canonical)
    return unique_preserve_order(tags)


def _extract_age_gates(text: str) -> tuple[int | None, int | None]:
    lowered = normalize_text(text)
    age_min = None
    age_max = None
    min_match = re.search(r"(?:age|aged|older than|>=|greater than|at least)\s*(?:of\s*)?(\d{1,3})", lowered)
    max_match = re.search(r"(?:younger than|<=|less than|under)\s*(?:age\s*)?(\d{1,3})", lowered)
    between_match = re.search(r"between\s+(\d{1,3})\s+and\s+(\d{1,3})", lowered)
    if between_match:
        age_min = int(between_match.group(1))
        age_max = int(between_match.group(2))
    else:
        if min_match:
            age_min = int(min_match.group(1))
        if max_match:
            age_max = int(max_match.group(1))
    return age_min, age_max


def _extract_sex_gate(text: str) -> str:
    lowered = normalize_text(text)
    if "female" in lowered and "male" not in lowered:
        return "female"
    if "male" in lowered and "female" not in lowered:
        return "male"
    return "all"


def _split_criteria_list(text: str, prefix: str) -> list[TrialCriterion]:
    if not text:
        return []
    normalized = (text or "").replace("\r", "\n").replace("•", "\n- ").replace("*", "\n- ")
    normalized = re.sub(r"(?<!\n)\s*-\s+", "\n- ", normalized)
    normalized = re.sub(r"(?<!\n)\s+(?=\d+\.)", "\n", normalized)
    lines = [normalize_space(line) for line in normalized.splitlines() if normalize_space(line)]
    criteria: list[TrialCriterion] = []
    group_counter = 0
    for idx, line in enumerate(lines, start=1):
        clean = re.sub(r"^(?:[-*]|\d+[.)]|[a-z][.)])\s*", "", line).strip()
        if not clean:
            continue
        group_counter += 1
        lowered = normalize_text(clean)
        is_or_group = " one of the following" in f" {lowered}" or " or " in lowered
        is_admin = any(pattern in lowered for pattern in ADMIN_PATTERNS)
        criteria.append(
            TrialCriterion(
                criterion_id=f"{prefix}_{idx}",
                group_id=f"{prefix}_group_{group_counter}",
                text=clean,
                criterion_type=prefix,
                priority=_criterion_priority(clean, is_admin),
                is_administrative=is_admin,
                is_or_group=is_or_group,
                tags=unique_preserve_order(_extract_tags(clean, CONDITION_TAGS) + _extract_tags(clean, INTERVENTION_TAGS)),
            )
        )
    return criteria


def _extract_required_patient_signals(inclusion_text: str, title: str, brief_summary: str) -> list[str]:
    lowered = normalize_text(" ".join([title, inclusion_text, brief_summary]))
    required: list[str] = []
    for signal, variants in REQUIRED_SIGNAL_PATTERNS.items():
        if any(variant in lowered for variant in variants):
            required.append(signal)
    return unique_preserve_order(required)


def normalize_trial_record(row: dict[str, str]) -> TrialRecord:
    inclusion_text, exclusion_text = split_criteria_blocks(str(row.get("eligibility_criteria", "") or ""))
    source_text = " ".join(
        [
            str(row.get("title", "") or ""),
            str(row.get("conditions", "") or ""),
            inclusion_text,
            exclusion_text,
            str(row.get("brief_summary", "") or ""),
            str(row.get("detailed_description", "") or ""),
        ]
    )
    age_min, age_max = _extract_age_gates(source_text)
    sex = _extract_sex_gate(source_text)
    lowered_source = normalize_text(source_text)
    padded_lowered_source = f" {lowered_source} "
    source_clinical_tags = extract_clinical_tags_from_normalized(padded_lowered_source)
    genericity_penalty = sum(0.08 for pattern in GENERIC_TRIAL_PATTERNS if pattern in lowered_source)

    condition_tags = unique_preserve_order(_extract_tags(source_text, CONDITION_TAGS) + source_clinical_tags)
    intervention_tags = unique_preserve_order(_extract_tags(source_text, INTERVENTION_TAGS) + [tag for tag in source_clinical_tags if tag in {"pci", "cabg", "tavr"}])
    criteria_inclusion = _split_criteria_list(inclusion_text, "inclusion")
    criteria_exclusion = _split_criteria_list(exclusion_text, "exclusion")
    required_patient_signals = _extract_required_patient_signals(
        inclusion_text=inclusion_text,
        title=str(row.get("title", "") or ""),
        brief_summary=str(row.get("brief_summary", "") or ""),
    )
    disease_tags = unique_preserve_order(condition_tags + [tag for item in criteria_inclusion for tag in item.tags] + extract_clinical_tags([inclusion_text, exclusion_text, str(row.get("conditions", "") or "")]))
    if any(term in lowered_source for term in ["rotational atherectomy", "intravascular ultrasound", "calcified coronary lesion", "coronary lesion"]):
        disease_tags = unique_preserve_order(disease_tags + ["coronary_artery_disease", "pci"])
        intervention_tags = unique_preserve_order(intervention_tags + ["pci", "catheterization"])
        genericity_penalty = max(0.0, genericity_penalty - 0.16)
    if "cardiorenal syndrome" in lowered_source:
        disease_tags = unique_preserve_order(disease_tags + ["cardiorenal_syndrome", "heart_failure", "chronic_kidney_disease"])
    if "health coach" in lowered_source and "randomized clinical trial" in lowered_source:
        genericity_penalty += 0.06

    sparse_text = "\n".join(
        part
        for part in [
            str(row.get("title", "") or ""),
            str(row.get("conditions", "") or ""),
            " ".join(condition_tags),
            " ".join(intervention_tags),
            " ".join(criterion.text for criterion in criteria_inclusion[:25]),
            " ".join(criterion.text for criterion in criteria_exclusion[:25]),
        ]
        if normalize_space(part)
    )

    dense_text = "\n".join(
        part
        for part in [
            str(row.get("title", "") or ""),
            str(row.get("brief_summary", "") or ""),
            str(row.get("detailed_description", "") or ""),
            inclusion_text,
            exclusion_text,
        ]
        if normalize_space(part)
    )

    return TrialRecord(
        nct_id=str(row.get("nct_id", "") or "").strip(),
        title=str(row.get("title", "") or "").strip(),
        conditions_text=str(row.get("conditions", "") or "").strip(),
        overall_status=str(row.get("overall_status", "") or "").strip(),
        brief_summary=str(row.get("brief_summary", "") or "").strip(),
        detailed_description=str(row.get("detailed_description", "") or "").strip(),
        eligibility_criteria=str(row.get("eligibility_criteria", "") or "").strip(),
        inclusion_criteria=criteria_inclusion,
        exclusion_criteria=criteria_exclusion,
        condition_tags=condition_tags,
        intervention_tags=intervention_tags,
        disease_tags=disease_tags,
        age_min=age_min,
        age_max=age_max,
        sex=sex,
        genericity_penalty=min(genericity_penalty, 0.4),
        required_patient_signals=required_patient_signals,
        sparse_text=sparse_text,
        dense_text=dense_text,
    )
