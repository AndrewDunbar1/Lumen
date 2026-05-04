# TrialGPT-style Comparator

This directory contains the comparator code path used to reproduce a TrialGPT-like architecture on the same cardiology cohort and trial corpus as the in-house Lumen pipeline.

It is not a byte-for-byte copy of the original TrialGPT paper environment. The main deviations in the reproduction were:

- BM25-only retrieval instead of BM25 + MedCPT hybrid retrieval
- top-20 candidate pool instead of top-50
- structured FHIR-derived patient summaries instead of full clinical notes
- `gpt-5-mini` instead of `gpt-4-turbo`

What was preserved closely:

- criterion-by-criterion inclusion matching logic
- criterion-by-criterion exclusion matching logic
- TrialGPT-style aggregation prompts and `R` / `E` scoring
- composite ranking flow

Main entry points:

- `build_cardiology_dataset.py`
- `run_cardiology_retrieval.py`
- `run_matching_parallel.py`
- `run_aggregation_parallel.py`
- `top5_cardiology.py`
- `run_cardiology_pipeline.sh`

This directory intentionally excludes local datasets, result dumps, and API logs.
