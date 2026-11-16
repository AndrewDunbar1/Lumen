# Lumen AI

Lumen AI is an open-source pipeline that narrows a large clinical trial corpus to a short, evidence-linked list of trials for each patient, for review by a clinician or study team. It produces a screening shortlist; it does not determine eligibility.

This repository contains the code for:

> Dunbar A, Elfayoumy H, Broder L, Barakat N, Kunaparaju A, Jog S, Patel S, Lam T, Ganatra S, Dani SS. Lumen AI: Compressing Large Clinical Trial Search Spaces into Clinician-Reviewable Shortlists. *JAMIA Open*. 2026.

## How it works

For each patient, Lumen:

1. Parses a FHIR patient bundle into a structured profile (`patient_parser/`, `matcher/patient_profiles.py`).
2. Normalizes trial records and splits eligibility criteria (`matcher/trial_normalization.py`).
3. Retrieves 100 candidate trials from the corpus (`matcher/retrieval.py`).
4. Reranks to 20 candidates with deterministic clinical features, then with an LLM (`matcher/llm_reranker.py`).
5. Evaluates criteria individually and combines the results into a match decision (`matcher/criterion_eval.py`, `matcher/decisioning.py`).
6. Exports a five-trial shortlist per patient.
7. Runs a second-pass LLM review that labels each shortlisted pair as `high_likelihood`, `possible`, `unlikely` or `no_match` (`scripts/review_top5_matches_with_llm.py`).

In the published evaluation the LLM stages used `gpt-5-mini`.

## Repository layout

| Path | Contents |
| --- | --- |
| `matcher/` | Core matching package: profiles, trial normalization, retrieval, reranking, criterion evaluation, decisioning |
| `patient_parser/` | FHIR bundle parsing |
| `post_processing/` | Export helpers used by the runner |
| `llm/` | Condition-normalization helper |
| `scripts/run_cardio_publication_full_v2.py` | Main pipeline runner |
| `scripts/review_top5_matches_with_llm.py` | Second-pass review of the top-five shortlist |
| `scripts/build_high_likelihood_dossier.py` | Reviewer dossier builder |
| `scripts/benchmark_cardio_matcher.py`, `scripts/analyze_trialgpt_manual_results.py` | Evaluation utilities |
| `analysis/run_ablation.py` | Component-removal (ablation) runs with API telemetry |
| `analysis/analyze_final_adjudication.py` | Final statistics: Wilson intervals, top-k yield, paired bootstrap and exact McNemar tests |
| `trialgpt_comparator/` | TrialGPT-style comparator used in the paper |
| `tests/` | Unit tests for the matcher |

## Installation

Python 3.11 or later.

```bash
pip install -r requirements.txt
```

The LLM stages read the OpenAI API key from the `OPENAI_API_KEY` environment variable or a local `.env` file.

The comparator has its own dependencies:

```bash
pip install -r trialgpt_comparator/requirements.txt
```

## Data

No patient data or trial text is distributed with this repository.

- **Patients.** The evaluation used 200 cardiology patients from MIMIC-IV, represented as FHIR bundles. MIMIC-IV is available from [PhysioNet](https://physionet.org/content/mimiciv/) to credentialed users who complete its data use agreement. It cannot be redistributed here.
- **Trials.** The corpus was 11,231 cardiology trials from [ClinicalTrials.gov](https://clinicaltrials.gov/), snapshot of March 5, 2026, enriched through the ClinicalTrials.gov API v2.
- **Results.** Aggregate results behind the paper's tables and figures, and the registry identifiers of the trial corpus, are deposited in Dryad: https://doi.org/10.5061/dryad.gmsbcc36k

## Usage

Run the pipeline:

```bash
python scripts/run_cardio_publication_full_v2.py \
  --patient-ids /path/to/patient_ids.txt \
  --patients-dir /path/to/patient_bundles \
  --snapshot-json /path/to/cardio_snapshot.csv \
  --output-dir /path/to/output_dir \
  --patient-workers 4 \
  --llm-rerank-top-n 20 \
  --llm-model gpt-5-mini
```

Each script documents its remaining options under `--help`.

The component-removal runs and the final analysis read their inputs from local folders set by `LUMEN_DATA_DIR` and `LUMEN_ANALYSIS_DIR`; see the top of each file in `analysis/` for the expected layout. The comparator workflow is described in `trialgpt_comparator/README.md`.

The LLM calls do not set a seed, so repeated runs can differ slightly.

## Tests

```bash
pytest
```

## TrialGPT-style comparator

`trialgpt_comparator/` adapts the TrialGPT matching and ranking code (Jin et al., *Nat Commun* 2024) to the same cohort and corpus. It is an adapted reproduction and not a release of the original TrialGPT project. The upstream public-domain notice is kept in `trialgpt_comparator/LICENSE`.

## Citation

If you use this code, please cite the paper above. Citation metadata is in `CITATION.cff`.

Hussein Elfayoumy and Luca Broder contributed equally as co-second authors.

## License

MIT. See `LICENSE`.
