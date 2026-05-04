from __future__ import annotations

from collections import Counter

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .schemas import CandidateFeatureRow, PatientProfile, TrialRecord
from .text_utils import keyword_overlap, unique_preserve_order


class HybridTrialMatcher:
    def __init__(self, trials: list[TrialRecord], max_candidates: int = 100) -> None:
        self.trials = trials
        self.max_candidates = max_candidates
        self.popularity_counter = Counter()
        self._fit()

    def _fit(self) -> None:
        sparse_corpus = [trial.sparse_text or trial.title for trial in self.trials]
        dense_corpus = [trial.dense_text or trial.sparse_text or trial.title for trial in self.trials]
        self.sparse_vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_features=20000,
            stop_words="english",
        )
        self.dense_vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_features=30000,
            stop_words="english",
        )
        self.sparse_matrix = self.sparse_vectorizer.fit_transform(sparse_corpus)
        dense_matrix = self.dense_vectorizer.fit_transform(dense_corpus)

        if dense_matrix.shape[1] <= 2:
            self.svd = None
            self.dense_matrix = dense_matrix
        else:
            dense_dim = max(2, min(64, dense_matrix.shape[1] - 1))
            self.svd = TruncatedSVD(n_components=dense_dim, random_state=42)
            self.dense_matrix = self.svd.fit_transform(dense_matrix)
        self.trial_ids = [trial.nct_id for trial in self.trials]

    def _gate_status(self, patient: PatientProfile, trial: TrialRecord) -> str:
        sex = (patient.sex or "unknown").lower()
        if trial.sex in {"male", "female"} and sex and sex != "unknown" and sex != trial.sex:
            return "sex_mismatch"
        if patient.age is not None:
            if trial.age_min is not None and patient.age < trial.age_min:
                return "age_below_min"
            if trial.age_max is not None and patient.age > trial.age_max:
                return "age_above_max"
        return "compatible"

    def _dense_similarity(self, patient_text: str) -> np.ndarray:
        dense_query = self.dense_vectorizer.transform([patient_text])
        if self.svd is None:
            return cosine_similarity(dense_query, self.dense_matrix).ravel()
        transformed = self.svd.transform(dense_query)
        return cosine_similarity(transformed, self.dense_matrix).ravel()

    def _required_signal_match(self, patient: PatientProfile, trial: TrialRecord) -> tuple[float, float]:
        if not trial.required_patient_signals:
            return 0.0, 0.0

        patient_text = f" {patient.normalized_patient_text} "
        signal_aliases = {
            "hemodialysis": [" hemodialysis ", " dialysis ", " renal dialysis "],
            "dialysis_access_malfunction": [" dialysis access malfunction ", " access malfunction "],
            "fistulogram": [" fistulogram "],
            "ivus": [" ivus ", " intravascular ultrasound "],
            "nirs_ivus": [" nirs ivus ", " nirs-ivus ", " near infrared spectroscopy "],
            "acute_decompensated_heart_failure": [" acute decompensated heart failure ", " adhf "],
            "elevated_bnp": [" bnp ", " pro bnp ", " pro-bnp ", " nt probnp "],
            "acute_coronary_syndrome": [" acute coronary syndrome ", " acs ", " stemi ", " nstemi "],
            "successful_pci": [" successful pci ", " post pci ", " pci ", " stent "],
            "drug_eluting_stent": [" drug eluting coronary artery stent ", " drug-eluting coronary artery stent ", " drug eluting stent "],
            "primary_hypertension": [" primary hypertension ", " hypertension "],
            "cardiorenal_syndrome": [" cardiorenal syndrome ", " heart_failure ", " chronic_kidney_disease "],
            "diuretic_resistance": [" diuretic resistance ", " loop diuretic "],
            "rotational_atherectomy": [" rotational atherectomy "],
            "calcified_coronary_lesion": [" calcified coronary lesion ", " calcified lesion ", " calcified nodule "],
            "carotid_stenting": [" carotid stenting ", " carotid stenosis ", " carotid revascularization "],
            "limb_artery_disease": [" limb arteries ", " peripheral vascular disease ", " lower extremity arter", " femoral artery "],
        }

        matched = 0
        critical_missing = 0
        critical_signals = {
            "dialysis_access_malfunction",
            "fistulogram",
            "nirs_ivus",
            "ivus",
            "acute_decompensated_heart_failure",
            "acute_coronary_syndrome",
            "rotational_atherectomy",
            "calcified_coronary_lesion",
            "carotid_stenting",
            "limb_artery_disease",
        }
        for signal in trial.required_patient_signals:
            aliases = signal_aliases.get(signal, [f" {signal.replace('_', ' ')} "])
            has_signal = any(alias in patient_text for alias in aliases)
            if has_signal:
                matched += 1
            elif signal in critical_signals:
                critical_missing += 1

        match_strength = matched / len(trial.required_patient_signals)
        penalty = min(0.18 * critical_missing, 0.54)
        return match_strength, penalty

    def _domain_drift_penalty(self, patient: PatientProfile, trial: TrialRecord) -> float:
        patient_text = f" {patient.normalized_patient_text} "
        trial_text = f" {trial.dense_text.lower()} {trial.title.lower()} "

        penalty = 0.0
        cerebrovascular_terms = [" carotid ", " cerebral ", " stroke ", " retinal artery ", " vertebral artery ", " patent foramen ovale "]
        peripheral_terms = [" limb arter", " lower extremity ", " peripheral artery ", " femoral artery "]

        if any(term in trial_text for term in cerebrovascular_terms):
            if not any(term in patient_text for term in cerebrovascular_terms):
                penalty += 0.22
        if any(term in trial_text for term in peripheral_terms):
            if not any(term in patient_text for term in peripheral_terms):
                penalty += 0.18
        return min(penalty, 0.3)

    def _trial_buckets(self, trial: TrialRecord) -> set[str]:
        buckets: set[str] = set()
        title_lower = trial.title.lower()
        disease_tags = set(trial.disease_tags)
        intervention_tags = set(trial.intervention_tags)
        required_signals = set(trial.required_patient_signals)

        if (
            {"heart_failure", "chronic_kidney_disease"} <= disease_tags
            or "cardiorenal_syndrome" in disease_tags
            or {"acute_decompensated_heart_failure", "diuretic_resistance", "cardiorenal_syndrome"} & required_signals
        ):
            buckets.add("cardiorenal_specific")
        if (
            {"pci", "catheterization"} & intervention_tags
            or {"coronary_artery_disease", "myocardial_infarction"} & disease_tags
            or any(term in title_lower for term in ["atherectomy", "ivus", "stemi", "acute myocardial", "coronary"])
        ):
            buckets.add("coronary_interventional")
        if "metabolic_disease" in disease_tags or "ckm" in title_lower or "health coach" in title_lower:
            buckets.add("ckm_broad")
        if "hypertension" in disease_tags and "coronary_interventional" not in buckets:
            buckets.add("hypertension_specific")
        return buckets

    def _patient_target_buckets(self, patient: PatientProfile) -> list[str]:
        tags = set(patient.clinical_tags)
        targets: list[str] = []
        if {"heart_failure", "chronic_kidney_disease"} <= tags or "cardiorenal_syndrome" in tags:
            targets.append("cardiorenal_specific")
        if {"coronary_artery_disease", "pci"} & tags or "myocardial_infarction" in tags:
            targets.append("coronary_interventional")
        if {"diabetes", "metabolic_disease"} & tags and ({"heart_failure", "chronic_kidney_disease", "coronary_artery_disease"} & tags):
            targets.append("ckm_broad")
        return unique_preserve_order(targets)

    def generate_candidates(self, patient: PatientProfile, top_k: int | None = None) -> list[CandidateFeatureRow]:
        requested_k = top_k or self.max_candidates
        sparse_query = self.sparse_vectorizer.transform([patient.sparse_text])
        sparse_scores = cosine_similarity(sparse_query, self.sparse_matrix).ravel()
        dense_scores = self._dense_similarity(patient.narrative_text)

        patient_diag = unique_preserve_order(patient.normalized_diagnoses + patient.diagnoses + patient.active_diagnoses + patient.clinical_tags)
        patient_proc = patient.procedures
        patient_meds = unique_preserve_order(patient.medications + patient.clinical_tags)
        patient_signals = unique_preserve_order(patient.note_signals + patient.clinical_tags)

        features: list[CandidateFeatureRow] = []
        for idx, trial in enumerate(self.trials):
            gate = self._gate_status(patient, trial)
            disease_match = keyword_overlap(patient_diag, trial.disease_tags or trial.condition_tags)
            intervention_match = keyword_overlap(patient_proc, trial.intervention_tags)
            medication_match = keyword_overlap(patient_meds, trial.intervention_tags)
            signal_match = keyword_overlap(patient_signals, trial.condition_tags + trial.intervention_tags)
            required_signal_match, missing_required_penalty = self._required_signal_match(patient, trial)
            domain_drift_penalty = self._domain_drift_penalty(patient, trial)
            trial_tags = unique_preserve_order(trial.disease_tags + trial.intervention_tags + trial.condition_tags)
            patient_tags = patient.clinical_tags
            tag_overlap = keyword_overlap(patient_tags, trial_tags)
            trial_buckets = self._trial_buckets(trial)
            target_buckets = set(self._patient_target_buckets(patient))
            repeated_penalty = min(self.popularity_counter[trial.nct_id] * 0.005, 0.2)
            gate_penalty = 0.25 if gate != "compatible" else 0.0
            broad_cardio_bonus = 0.0
            if {"heart_failure", "chronic_kidney_disease"} <= set(patient.clinical_tags) and "cardiorenal_syndrome" in trial.disease_tags:
                broad_cardio_bonus += 0.45
            if {"coronary_artery_disease", "pci"} & set(patient.clinical_tags) and "pci" in trial.intervention_tags:
                broad_cardio_bonus += 0.45
            if "diabetes" in patient.clinical_tags and ("metabolic_disease" in trial.disease_tags or "cardiovascular_disease" in trial.disease_tags):
                broad_cardio_bonus += 0.08
            specificity_bonus = 0.0
            trial_title_lower = trial.title.lower()
            if {"heart_failure", "chronic_kidney_disease"} <= set(patient.clinical_tags) and "cardiorenal_syndrome" in trial.disease_tags:
                if len(trial.disease_tags) <= 8:
                    specificity_bonus += 0.25
            if {"coronary_artery_disease", "pci"} <= set(patient.clinical_tags) and "pci" in trial.intervention_tags:
                if any(term in trial_title_lower for term in ["rotational atherectomy", "intravascular ultrasound", "calcified coronary lesion"]):
                    specificity_bonus += 0.35
            if "sudden_cardiac_death" in patient.clinical_tags and "sudden_cardiac_death" in trial.disease_tags:
                specificity_bonus += 0.18
            coverage_boost = 0.0
            if "cardiorenal_specific" in trial_buckets and "cardiorenal_specific" in target_buckets:
                coverage_boost += 0.28
            if "coronary_interventional" in trial_buckets and "coronary_interventional" in target_buckets:
                coverage_boost += 0.22
            if "ckm_broad" in trial_buckets and "ckm_broad" in target_buckets:
                coverage_boost += 0.08
            ckm_specificity_penalty = 0.0
            if "ckm_broad" in trial_buckets and ({"cardiorenal_specific", "coronary_interventional"} & target_buckets):
                ckm_specificity_penalty += 0.06
            broadness_penalty = 0.0
            if len(trial_tags) >= 9 and tag_overlap >= 0.35:
                broadness_penalty += 0.10
            rerank_score = (
                0.22 * float(sparse_scores[idx])
                + 0.14 * float(dense_scores[idx])
                + 0.28 * disease_match
                + 0.16 * intervention_match
                + 0.06 * medication_match
                + 0.06 * signal_match
                + 0.24 * tag_overlap
                + 0.12 * required_signal_match
                + broad_cardio_bonus
                + specificity_bonus
                + coverage_boost
                - broadness_penalty
                - ckm_specificity_penalty
                - missing_required_penalty
                - domain_drift_penalty
                - (trial.genericity_penalty * 0.35)
                - repeated_penalty
                - gate_penalty
            )
            features.append(
                CandidateFeatureRow(
                    patient_id=patient.patient_id,
                    nct_id=trial.nct_id,
                    title=trial.title,
                    retrieval_rank=0,
                    retrieval_score_sparse=float(sparse_scores[idx]),
                    retrieval_score_dense=float(dense_scores[idx]),
                    rerank_score=rerank_score,
                    age_sex_gate_status=gate,
                    disease_match_strength=disease_match,
                    intervention_match_strength=intervention_match,
                    medication_match_strength=medication_match,
                    note_signal_match_strength=signal_match,
                    required_signal_match_strength=required_signal_match,
                    missing_required_signal_penalty=missing_required_penalty,
                    generic_trial_penalty=trial.genericity_penalty,
                    repeated_popularity_penalty=repeated_penalty,
                    feature_debug={
                        "condition_tags": trial.condition_tags,
                        "intervention_tags": trial.intervention_tags,
                        "patient_diag_tokens": patient_diag,
                        "patient_clinical_tags": patient.clinical_tags,
                        "tag_overlap": tag_overlap,
                        "required_patient_signals": trial.required_patient_signals,
                        "trial_buckets": sorted(trial_buckets),
                        "target_buckets": sorted(target_buckets),
                        "required_signal_match": required_signal_match,
                        "missing_required_signal_penalty": missing_required_penalty,
                        "domain_drift_penalty": domain_drift_penalty,
                        "coverage_boost": coverage_boost,
                        "ckm_specificity_penalty": ckm_specificity_penalty,
                        "broad_cardio_bonus": broad_cardio_bonus,
                        "specificity_bonus": specificity_bonus,
                        "broadness_penalty": broadness_penalty,
                    },
                )
            )

        features.sort(
            key=lambda row: (
                row.age_sex_gate_status != "compatible",
                -row.rerank_score,
                -row.retrieval_score_dense,
                -row.retrieval_score_sparse,
            )
        )

        selected: list[CandidateFeatureRow] = []
        covered_condition_groups: Counter[str] = Counter()
        for row in features:
            trial = self.get_trial(row.nct_id)
            major_group = ",".join(trial.condition_tags[:2]) or "generic"
            diversity_penalty = max(0, covered_condition_groups[major_group] - 1) * 0.05
            row.diversity_penalty = diversity_penalty
            row.rerank_score -= diversity_penalty
            selected.append(row)
            covered_condition_groups[major_group] += 1
            if len(selected) >= requested_k:
                break

        selected.sort(key=lambda row: row.rerank_score, reverse=True)
        for rank, row in enumerate(selected, start=1):
            row.retrieval_rank = rank
            self.popularity_counter[row.nct_id] += 1
        return selected

    def rerank_candidates(self, patient: PatientProfile, candidates: list[CandidateFeatureRow], top_n: int = 20) -> list[CandidateFeatureRow]:
        reranked = sorted(
            candidates,
            key=lambda row: (
                row.age_sex_gate_status != "compatible",
                -row.rerank_score,
                -row.disease_match_strength,
                -row.intervention_match_strength,
            ),
        )
        target_buckets = self._patient_target_buckets(patient)
        if not target_buckets:
            return reranked[:top_n]

        prioritized: list[CandidateFeatureRow] = []
        used_ids: set[str] = set()
        for bucket in target_buckets:
            for row in reranked:
                if row.nct_id in used_ids:
                    continue
                trial = self.get_trial(row.nct_id)
                if bucket in self._trial_buckets(trial):
                    prioritized.append(row)
                    used_ids.add(row.nct_id)
                    break

        for row in reranked:
            if row.nct_id in used_ids:
                continue
            prioritized.append(row)
            used_ids.add(row.nct_id)
            if len(prioritized) >= top_n:
                break
        return prioritized[:top_n]

    def get_trial(self, nct_id: str) -> TrialRecord:
        index = self.trial_ids.index(nct_id)
        return self.trials[index]

    def export_index_summary(self) -> dict[str, float | int]:
        generic_trials = sum(1 for trial in self.trials if trial.genericity_penalty > 0)
        penalties = [trial.genericity_penalty for trial in self.trials]
        return {
            "trial_count": len(self.trials),
            "generic_trial_count": generic_trials,
            "mean_genericity_penalty": float(np.mean(penalties)) if penalties else 0.0,
        }
