# Lumen

Lumen is an open-source clinical trial shortlist-enrichment pipeline for matching structured patient records to candidate cardiology trials.

This public repo contains two code paths:

- `matcher/` and `scripts/`: the in-house Lumen pipeline used for structured patient profiling, candidate generation, reranking, criterion-level evaluation, and final top-5 review
- `trialgpt_comparator/`: the TrialGPT-style comparator reproduction used in the paired manuscript comparison

The repository is code-first and intentionally excludes protected patient data, local trial snapshots, review workbooks, and large run artifacts.

## Repository Layout

- `matcher/`: core patient-trial matching package
- `patient_parser/`: FHIR patient bundle parsing utilities
- `post_processing/`: review/export helpers used by the pipeline
- `scripts/run_cardio_publication_full_v2.py`: main in-house production runner
- `scripts/benchmark_cardio_matcher.py`: benchmark and comparator evaluation
- `scripts/review_top5_matches_with_llm.py`: second-pass top-5 review layer
- `scripts/build_high_likelihood_dossier.py`: print-ready dossier builder for reviewed matches
- `scripts/analyze_trialgpt_manual_results.py`: manual validation analysis utility
- `tests/test_matcher_pipeline.py`: focused matcher tests
- `trialgpt_comparator/`: TrialGPT-style retrieval, matching, aggregation, and ranking scripts

## What Lumen Does

The in-house pipeline follows this sequence:

1. Parse FHIR-style patient bundles into structured profiles.
2. Normalize trial records and eligibility criteria.
3. Generate a high-recall candidate set.
4. Rerank candidates using deterministic clinical signals and optional `gpt-5-mini` reranking.
5. Evaluate criterion-level eligibility and calibrated fit.
6. Export a final top-5 shortlist.
7. Optionally run a second-pass LLM review over the top-5 pairs.

The TrialGPT-style comparator follows a different sequence:

1. Convert patient bundles into compact free-text summaries.
2. Build a BEIR-style trial corpus.
3. Retrieve top candidates with BM25.
4. Run TrialGPT-style inclusion/exclusion matching per pair.
5. Run TrialGPT-style aggregation scoring per pair.
6. Rank by composite score and export the top 5.

## Quick Start

### In-house Lumen pipeline

Install the core dependencies:

```bash
pip install -r requirements.txt
```

Set `OPENAI_API_KEY` if you want LLM reranking or top-5 review.

Example run:

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

Example benchmark:

```bash
python scripts/benchmark_cardio_matcher.py \
  --run-dir /path/to/output_dir \
  --baseline-pack-dir /path/to/adjudicated_top3_pack
```

### TrialGPT-style comparator

The comparator has its own dependency file:

```bash
pip install -r trialgpt_comparator/requirements.txt
```

Then see:

- `trialgpt_comparator/build_cardiology_dataset.py`
- `trialgpt_comparator/run_cardiology_retrieval.py`
- `trialgpt_comparator/run_matching_parallel.py`
- `trialgpt_comparator/run_aggregation_parallel.py`
- `trialgpt_comparator/top5_cardiology.py`

## Public Release Notes

- No patient data is included.
- No local trial snapshots or adjudication outputs are included.
- Environment-specific `.env` files are intentionally excluded.
- The TrialGPT-style comparator is included as a reproduction used for side-by-side evaluation, not as an official release of the original TrialGPT project.

## Testing

```bash
pytest tests/test_matcher_pipeline.py
```

## License

This repository is released under the MIT License. See `LICENSE`.
