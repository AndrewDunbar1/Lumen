#!/bin/bash
# TrialGPT Cardiology Pipeline
# Full end-to-end run for 200 MIMIC cardiology patients × 11,231 trials
#
# Usage:
#   bash run_cardiology_pipeline.sh [--top_n 50] [--model gpt-4-turbo] [--skip-keywords]
#
# Stages:
#   1. (Optional) Keyword generation  — ~200 GPT-4 calls, ~$1-2
#   2. BM25 retrieval                 — free, ~2 min
#   3. LLM Matching                   — ~200×top_n×2 calls, see cost note below
#   4. LLM Aggregation                — ~200×top_n calls
#   5. Top-5 ranking output           — free
#
# Cost estimate (gpt-4-turbo, top_n=50):
#   Step 3: 200 patients × 50 trials × 2 calls = 20,000 calls  (~$20-60)
#   Step 4: 200 patients × 50 trials × 1 call  = 10,000 calls  (~$10-30)
#   Total: roughly $30-90 for the full 200-patient run
#
# For a cheaper test run, use --top_n 10 and a subset of patients.

set -e
cd "$(dirname "$0")"

# Load .env if present
if [ -f "../cureva-main/.env" ]; then
  export $(grep -v '^#' ../cureva-main/.env | xargs)
elif [ -f ".env" ]; then
  export $(grep -v '^#' .env | xargs)
fi

if [ -z "$OPENAI_API_KEY" ]; then
  echo "ERROR: OPENAI_API_KEY not set. Export it or add to .env"
  exit 1
fi

MODEL="${MODEL:-gpt-4-turbo}"
TOP_N="${TOP_N:-50}"
SKIP_KEYWORDS="${SKIP_KEYWORDS:-0}"

echo "========================================"
echo "TrialGPT Cardiology Pipeline"
echo "  Model   : $MODEL"
echo "  Top-N   : $TOP_N trials per patient"
echo "========================================"

# ── Stage 1: Keyword generation (optional) ──────────────────────────────
if [ "$SKIP_KEYWORDS" = "0" ]; then
  echo
  echo "Stage 1/5: Generating retrieval keywords (~200 API calls) ..."
  venv/bin/python trialgpt_retrieval/keyword_generation.py cardiology "$MODEL"
  Q_TYPE="$MODEL"
else
  echo "Stage 1/5: Skipping keyword generation (using raw patient text)"
  Q_TYPE="raw"
fi

# ── Stage 2: BM25 Retrieval ───────────────────────────────────────────────
echo
echo "Stage 2/5: Running BM25 retrieval (top $TOP_N trials per patient) ..."
venv/bin/python run_cardiology_retrieval.py --q_type "$Q_TYPE" --top_n "$TOP_N" --bm25_only

# ── Stage 3: LLM Matching ────────────────────────────────────────────────
echo
echo "Stage 3/5: Running TrialGPT-Matching (this is the slow/costly step) ..."
echo "           ~$(( 200 * TOP_N * 2 )) API calls expected"
venv/bin/python trialgpt_matching/run_matching.py cardiology "$MODEL"

MATCHING_RESULTS="results/matching_results_cardiology_${MODEL}.json"

# ── Stage 4: LLM Aggregation ─────────────────────────────────────────────
echo
echo "Stage 4/5: Running TrialGPT-Ranking aggregation ..."
venv/bin/python run_cardiology_aggregation.py "$MATCHING_RESULTS" "$MODEL"

AGG_RESULTS="results/aggregation_results_cardiology_${MODEL}.json"

# ── Stage 5: Top-5 output ─────────────────────────────────────────────────
echo
echo "Stage 5/5: Generating top-5 match report ..."
venv/bin/python top5_cardiology.py "$MATCHING_RESULTS" "$AGG_RESULTS" --n 5

echo
echo "========================================"
echo "Pipeline complete!"
echo "  Top-5 report: results/top5_cardiology.json"
echo "========================================"
