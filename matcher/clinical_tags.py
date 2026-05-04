from __future__ import annotations

from .text_utils import normalize_text, unique_preserve_order


CLINICAL_TAG_KEYWORDS: dict[str, list[str]] = {
    "heart_failure": [
        "heart failure",
        "congestive heart failure",
        "hfpef",
        "hfref",
        "hfmref",
        "hfmr ef",
        "hfmr ef",
        "cardiomyopathy",
        "end stage heart failure",
    ],
    "atrial_fibrillation": [
        "atrial fibrillation",
        "atrial flutter",
        "afib",
        "a fib",
    ],
    "coronary_artery_disease": [
        "coronary artery disease",
        "coronary heart disease",
        "ischemic heart disease",
        "atherosclerotic heart disease",
        "coronary atherosclerosis",
        "coronary lesion",
        "coronary stenosis",
        "epicardial coronary artery",
        " cad ",
    ],
    "myocardial_infarction": [
        "myocardial infarction",
        "old myocardial infarction",
        "acute mi",
        "nstemi",
        "stemi",
        "acute coronary syndrome",
        "unstable angina",
    ],
    "chronic_kidney_disease": [
        "chronic kidney disease",
        " ckd ",
        "end stage renal disease",
        " esrd ",
        "renal dialysis",
        " dialysis ",
        "acute kidney failure",
        "acute kidney injury",
        " aki ",
        " aki1 ",
        "kidney disease",
        "kidney injury",
        "renal disease",
    ],
    "diabetes": [
        "diabetes",
        "diabetic",
        "diabetes mellitus",
        "type 2 diabetes",
        " t2dm ",
        " dm ",
        "hba1c",
        " a1c ",
    ],
    "hypertension": [
        "hypertension",
        "hypertensive",
        "high blood pressure",
        " htn ",
    ],
    "aortic_stenosis": [
        "aortic stenosis",
        "calcific aortic stenosis",
        "valve stenosis",
    ],
    "sudden_cardiac_death": [
        "sudden cardiac death",
        "cardiac arrest",
        "ventricular fibrillation",
        "ventricular tachycardia",
        " icd ",
    ],
    "metabolic_disease": [
        "metabolic syndrome",
        "obesity",
        "dyslipidemia",
        "hyperlipidemia",
        "metabolic disease",
    ],
    "pci": [
        " pci ",
        "percutaneous coronary intervention",
        "angioplasty",
        " ptca ",
        " stent ",
        "drug eluting coronary artery stent",
        "coronary angioplasty",
        "rotational atherectomy",
        "intravascular ultrasound",
        " ivus ",
        "calcified coronary lesion",
    ],
    "cabg": [
        " cabg ",
        "coronary artery bypass",
        "aortocoronary bypass",
        "bypass graft",
    ],
    "tavr": [
        " tavr ",
        " tavi ",
        "transcatheter aortic valve",
        "aortic valve replacement",
    ],
    "cardiovascular_disease": [
        "cardiovascular disease",
        " cvd ",
        "cardiology",
    ],
    "cardiorenal_syndrome": [
        "cardiorenal syndrome",
        " crs ",
        " ckm ",
    ],
}


def extract_clinical_tags(texts: list[str]) -> list[str]:
    combined = "\n".join(texts)
    lowered = f" {normalize_text(combined)} "
    return extract_clinical_tags_from_normalized(lowered)


def extract_clinical_tags_from_normalized(normalized_text: str) -> list[str]:
    lowered = normalized_text if normalized_text.startswith(" ") else f" {normalized_text} "
    tags = []
    for tag, keywords in CLINICAL_TAG_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            tags.append(tag)
    return unique_preserve_order(tags)


def canonicalize_condition_name(name: str) -> str:
    lowered = f" {normalize_text(name)} "
    for tag, keywords in CLINICAL_TAG_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            return tag
    return name.strip()
