from __future__ import annotations

from datetime import date, datetime
import re
from typing import Iterable

from .clinical_tags import canonicalize_condition_name, extract_clinical_tags
from .schemas import PatientEvidence, PatientProfile
from .text_utils import normalize_space, normalize_text, sentence_split, unique_preserve_order

NOTE_SIGNAL_PATTERNS: dict[str, str] = {
    "reduced_ejection_fraction": r"\b(?:ef|ejection fraction)\b.{0,15}\b(?:[1-4]?\d)\s?%",
    "preserved_ejection_fraction": r"\b(?:ef|ejection fraction)\b.{0,15}\b(?:5\d|6\d|70)\s?%",
    "nyha_class": r"\bnyha\s+(?:class\s+)?(?:ii|iii|iv|2|3|4)\b",
    "atrial_fibrillation": r"\b(?:afib|a fib|atrial fibrillation|atrial flutter)\b",
    "acute_coronary_syndrome": r"\b(?:acs|stemi|nstemi|acute coronary syndrome|myocardial infarction)\b",
    "pci_history": r"\b(?:pci|stent|angioplasty)\b",
    "cabg_history": r"\b(?:cabg|coronary artery bypass)\b",
    "tavr_history": r"\b(?:tavr|tavi|transcatheter aortic valve)\b",
    "renal_impairment": r"\b(?:ckd|chronic kidney disease|renal failure|dialysis|creatinine)\b",
    "hepatic_impairment": r"\b(?:cirrhosis|hepatic failure|liver failure|transaminase)\b",
    "pregnancy": r"\b(?:pregnant|pregnancy|breastfeeding)\b",
    "malignancy": r"\b(?:cancer|malignancy|tumor|metastatic)\b",
}


def _safe_isodate(value: object) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if hasattr(value, "date"):
        maybe_date = value.date
        if callable(maybe_date):
            try:
                return maybe_date()
            except TypeError:
                pass
        if isinstance(maybe_date, datetime):
            return maybe_date.date()
        if isinstance(maybe_date, date):
            return maybe_date
    text = str(value)
    match = re.match(r"^(\d{4})-(\d{2})-(\d{2})", text)
    if not match:
        return None
    try:
        return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    except ValueError:
        return None


def _calc_age(birth_date: date | None, anchor_dates: Iterable[date]) -> int | None:
    if birth_date is None:
        return None
    valid = sorted(item for item in anchor_dates if item and item >= birth_date)
    if not valid:
        return None
    anchor = valid[-1]
    years = int((anchor - birth_date).days // 365.2425)
    return years if 0 <= years <= 120 else None

def _dedupe(items: Iterable[str]) -> list[str]:
    return unique_preserve_order([normalize_space(item) for item in items if normalize_space(item)])


def _flatten_notes(note_groups: Iterable[object], max_chars: int = 20000) -> list[str]:
    out: list[str] = []
    for group in note_groups:
        if isinstance(group, list):
            for item in group:
                text = normalize_space(item)
                if text:
                    out.append(text)
        else:
            text = normalize_space(group)
            if text:
                out.append(text)
    joined = "\n\n".join(out)
    if len(joined) > max_chars:
        joined = joined[:max_chars]
    return sentence_split(joined)


def build_patient_profile(patient, patient_id: str | None = None) -> PatientProfile:
    personal = patient.personal_info
    resolved_patient_id = patient_id or str(getattr(personal, "id", "") or "").replace("Patient_", "")

    diagnoses = _dedupe(patient.conditions.get_condition_names())
    normalized = _dedupe(canonicalize_condition_name(value) for value in diagnoses)
    active = _dedupe(patient.conditions.get_active_conditions())
    procedures = _dedupe(getattr(proc, "name", "") for proc in patient.procedures.procedure_list)
    medications = _dedupe(getattr(med, "name", "") for med in patient.medications)

    labs: list[str] = []
    anchor_dates: list[date] = []
    for lab in patient.labs.return_all_labs():
        name = normalize_space(getattr(lab, "test_name", ""))
        value = getattr(lab, "value", "")
        unit = normalize_space(getattr(lab, "unit", ""))
        if name:
            labs.append(f"{name}: {value} {unit}".strip())
        anchor = _safe_isodate(getattr(lab, "date", None))
        if anchor:
            anchor_dates.append(anchor)

    for proc in patient.procedures.procedure_list:
        anchor = _safe_isodate(getattr(proc, "performed_time", None))
        if anchor:
            anchor_dates.append(anchor)

    birth_date = _safe_isodate(getattr(personal, "birthday", None))
    age = _calc_age(birth_date, anchor_dates)
    sex = normalize_space(getattr(personal, "gender", "") or "unknown")

    note_sentences = _flatten_notes(patient.notes.reports)
    note_signals: list[str] = []
    evidence: list[PatientEvidence] = []

    for index, sentence in enumerate(note_sentences):
        evidence.append(PatientEvidence("note", f"note:{index + 1}", sentence))
        lowered = sentence.lower()
        for signal_name, pattern in NOTE_SIGNAL_PATTERNS.items():
            if re.search(pattern, lowered):
                note_signals.append(signal_name)

    for index, diagnosis in enumerate(diagnoses):
        evidence.append(PatientEvidence("condition", f"condition:{index + 1}", diagnosis))
    for index, procedure in enumerate(procedures):
        evidence.append(PatientEvidence("procedure", f"procedure:{index + 1}", procedure))
    for index, medication in enumerate(medications):
        evidence.append(PatientEvidence("medication", f"medication:{index + 1}", medication))
    for index, lab in enumerate(labs):
        evidence.append(PatientEvidence("lab", f"lab:{index + 1}", lab))

    demographics_parts = [f"sex={sex}"]
    if age is not None:
        demographics_parts.insert(0, f"age={age}")
    demographics_text = " ".join(demographics_parts)
    note_signals = _dedupe(note_signals)
    patient_tags = extract_clinical_tags(diagnoses + normalized + active + procedures + medications + labs + note_sentences + note_signals)
    normalized_patient_text = normalize_text(
        " ".join(
            [
                demographics_text,
                " ".join(diagnoses),
                " ".join(normalized),
                " ".join(active),
                " ".join(procedures),
                " ".join(medications),
                " ".join(labs),
                " ".join(note_sentences),
                " ".join(note_signals),
                " ".join(patient_tags),
            ]
        )
    )

    sparse_sections = [
        demographics_text,
        " ".join(diagnoses),
        " ".join(normalized),
        " ".join(active),
        " ".join(procedures),
        " ".join(medications),
        " ".join(labs),
        " ".join(note_signals),
        " ".join(patient_tags),
    ]
    sparse_text = "\n".join(part for part in sparse_sections if part)

    narrative_lines = [
        f"Patient {resolved_patient_id}",
        f"Demographics: {demographics_text}",
        f"Diagnoses: {', '.join(diagnoses) if diagnoses else 'None recorded'}",
        f"Normalized diagnoses: {', '.join(normalized) if normalized else 'None derived'}",
        f"Active diagnoses: {', '.join(active) if active else 'None recorded'}",
        f"Procedures: {', '.join(procedures) if procedures else 'None recorded'}",
        f"Medications: {', '.join(medications) if medications else 'None recorded'}",
        f"Labs: {', '.join(labs[:20]) if labs else 'None recorded'}",
        f"Signals: {', '.join(note_signals) if note_signals else 'None found'}",
        f"Clinical tags: {', '.join(patient_tags) if patient_tags else 'None derived'}",
        "Evidence snippets:",
    ]
    for item in evidence[:25]:
        narrative_lines.append(f"- [{item.source_ref}] {item.text}")
    narrative_text = "\n".join(narrative_lines)

    return PatientProfile(
        patient_id=resolved_patient_id,
        age=age,
        sex=sex,
        demographics_text=demographics_text,
        diagnoses=diagnoses,
        normalized_diagnoses=normalized,
        active_diagnoses=active,
        procedures=procedures,
        medications=medications,
        labs=labs,
        note_signals=note_signals,
        clinical_tags=patient_tags,
        normalized_patient_text=normalized_patient_text,
        evidence=evidence,
        sparse_text=sparse_text,
        narrative_text=narrative_text,
    )
