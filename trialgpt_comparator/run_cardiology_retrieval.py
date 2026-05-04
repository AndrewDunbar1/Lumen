#!/usr/bin/env python3
"""
TrialGPT retrieval step for the cardiology dataset.

Outputs dataset/cardiology/retrieved_trials.json in the exact format
expected by trialgpt_matching/run_matching.py.

Key differences from original hybrid_fusion_retrieval.py:
  - No ground-truth qrels needed (custom data has no labels)
  - MPS (Apple Silicon) or CPU — no CUDA requirement
  - Reads cached keywords if available; falls back to raw patient text
  - Embeds full trial info in each retrieved item (for run_matching.py)

Usage (from TrialGPT root, with venv activated):
  python run_cardiology_retrieval.py [options]

  --q_type   keyword source: 'raw' or model name (default: gpt-4-turbo if
             results/retrieval_keywords_gpt-4-turbo_cardiology.json exists,
             else 'raw')
  --k        RRF constant (default: 20)
  --top_n    trials to return per patient (default: 50)
  --bm25_only  skip MedCPT — faster, no GPU / large model download needed
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import tqdm
from nltk import word_tokenize
from rank_bm25 import BM25Okapi

CORPUS = "cardiology"


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------

def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Load corpus metadata (for populating retrieved trial objects)
# ---------------------------------------------------------------------------

def load_corpus_meta(corpus: str) -> dict[str, dict]:
    """Return {nct_id: metadata_dict} from corpus.jsonl."""
    meta: dict[str, dict] = {}
    with open(f"dataset/{corpus}/corpus.jsonl") as f:
        for line in f:
            entry = json.loads(line)
            nct_id = entry["_id"]
            m = entry["metadata"]
            meta[nct_id] = {
                "NCTID": nct_id,
                "brief_title": m.get("brief_title", ""),
                "phase": m.get("phase", ""),
                "drugs": m.get("drugs", "[]"),
                "drugs_list": m.get("drugs_list", []),
                "diseases": m.get("diseases", "[]"),
                "diseases_list": m.get("diseases_list", []),
                "enrollment": m.get("enrollment", ""),
                "inclusion_criteria": m.get("inclusion_criteria", ""),
                "exclusion_criteria": m.get("exclusion_criteria", ""),
                "brief_summary": m.get("brief_summary", ""),
            }
    return meta


# ---------------------------------------------------------------------------
# BM25 corpus index
# ---------------------------------------------------------------------------

def get_bm25_index(corpus: str, meta: dict[str, dict]):
    cache = Path(f"trialgpt_retrieval/bm25_corpus_{corpus}.json")
    if cache.exists():
        print("  Loading cached BM25 index …")
        data = json.loads(cache.read_text())
        tokenized_corpus = data["tokenized_corpus"]
        corpus_nctids = data["corpus_nctids"]
    else:
        print("  Building BM25 index …")
        tokenized_corpus = []
        corpus_nctids = []
        with open(f"dataset/{corpus}/corpus.jsonl") as f:
            lines = f.readlines()
        for line in tqdm.tqdm(lines, desc="Tokenising corpus"):
            entry = json.loads(line)
            nct_id = entry["_id"]
            corpus_nctids.append(nct_id)
            tokens = word_tokenize(entry["title"].lower()) * 3
            for disease in meta.get(nct_id, {}).get("diseases_list", []):
                tokens += word_tokenize(disease.lower()) * 2
            tokens += word_tokenize(entry["text"].lower())
            tokenized_corpus.append(tokens)
        data = {"tokenized_corpus": tokenized_corpus, "corpus_nctids": corpus_nctids}
        cache.write_text(json.dumps(data))
        print(f"  BM25 index cached → {cache}")
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25, corpus_nctids


# ---------------------------------------------------------------------------
# MedCPT corpus index  (optional)
# ---------------------------------------------------------------------------

def get_medcpt_index(corpus: str, device: str):
    import faiss
    from transformers import AutoModel, AutoTokenizer

    embed_path = Path(f"trialgpt_retrieval/{corpus}_embeds.npy")
    nctids_path = Path(f"trialgpt_retrieval/{corpus}_nctids.json")

    if embed_path.exists() and nctids_path.exists():
        print("  Loading cached MedCPT embeddings …")
        embeds = np.load(str(embed_path))
        corpus_nctids = json.loads(nctids_path.read_text())
    else:
        print(f"  Encoding corpus with MedCPT on {device} (one-time, may take ~30 min on CPU) …")
        model = AutoModel.from_pretrained("ncbi/MedCPT-Article-Encoder").to(device)
        tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Article-Encoder")
        embeds = []
        corpus_nctids = []
        with open(f"dataset/{corpus}/corpus.jsonl") as f:
            lines = f.readlines()
        for line in tqdm.tqdm(lines, desc="Encoding trials"):
            entry = json.loads(line)
            corpus_nctids.append(entry["_id"])
            title = entry["title"]
            text = entry["text"][:512]
            with torch.no_grad():
                encoded = tokenizer(
                    [[title, text]],
                    truncation=True,
                    padding=True,
                    return_tensors="pt",
                    max_length=512,
                ).to(device)
                embed = model(**encoded).last_hidden_state[:, 0, :]
                embeds.append(embed[0].cpu().numpy())
        embeds = np.array(embeds)
        np.save(str(embed_path), embeds)
        nctids_path.write_text(json.dumps(corpus_nctids))
        print("  MedCPT embeddings cached.")

    index = faiss.IndexFlatIP(768)
    index.add(embeds)
    return index, corpus_nctids


def load_query_encoder(device: str):
    from transformers import AutoModel, AutoTokenizer
    model = AutoModel.from_pretrained("ncbi/MedCPT-Query-Encoder").to(device)
    tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Query-Encoder")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_retrieval(q_type: str, k: int, top_n: int, bm25_only: bool):
    device = get_device()
    print(f"Device: {device}")

    # ── Load queries ──────────────────────────────────────────────────────
    queries: dict[str, str] = {}
    with open(f"dataset/{CORPUS}/queries.jsonl") as f:
        for line in f:
            e = json.loads(line)
            queries[e["_id"]] = e["text"]
    print(f"Loaded {len(queries)} patients")

    # ── Load corpus metadata ──────────────────────────────────────────────
    print("Loading corpus metadata …")
    corpus_meta = load_corpus_meta(CORPUS)
    print(f"  {len(corpus_meta):,} trials in corpus")

    # ── Resolve keyword lists ─────────────────────────────────────────────
    kw_file = Path(f"results/retrieval_keywords_{q_type}_{CORPUS}.json")
    id2conditions: dict[str, list[str]] = {}

    if q_type != "raw" and kw_file.exists():
        print(f"  Using cached keywords: {kw_file.name}")
        raw_kw = json.loads(kw_file.read_text())
        for qid, kw_data in raw_kw.items():
            if isinstance(kw_data, dict):
                id2conditions[qid] = kw_data.get("conditions", [])
            elif isinstance(kw_data, list):
                id2conditions[qid] = kw_data
    else:
        if q_type != "raw":
            print(f"  [INFO] Keyword file not found — using raw patient text")
        for qid, text in queries.items():
            id2conditions[qid] = [text]

    # ── Build BM25 index ──────────────────────────────────────────────────
    bm25, bm25_nctids = get_bm25_index(CORPUS, corpus_meta)

    # ── Optionally build MedCPT index ─────────────────────────────────────
    medcpt_index = medcpt_nctids = qenc_model = qenc_tokenizer = None
    if not bm25_only:
        try:
            medcpt_index, medcpt_nctids = get_medcpt_index(CORPUS, device)
            qenc_model, qenc_tokenizer = load_query_encoder(device)
            print("  MedCPT retriever ready")
        except Exception as exc:
            print(f"  [WARN] MedCPT unavailable ({exc}). Using BM25-only.")

    # ── Retrieve for each patient ─────────────────────────────────────────
    N = top_n
    output: list[dict] = []

    for qid, patient_text in tqdm.tqdm(queries.items(), desc="Retrieving"):
        conditions = id2conditions.get(qid) or [patient_text]

        nctid2score: dict[str, float] = {}

        # BM25 scoring
        for cond_idx, cond in enumerate(conditions):
            tokens = word_tokenize(cond.lower())
            top = bm25.get_top_n(tokens, bm25_nctids, n=N * 2)
            for rank, nctid in enumerate(top):
                nctid2score[nctid] = nctid2score.get(nctid, 0.0) + (
                    1 / (rank + k)
                ) * (1 / (cond_idx + 1))

        # MedCPT scoring (if available)
        if medcpt_index is not None:
            with torch.no_grad():
                enc = qenc_tokenizer(
                    conditions,
                    truncation=True,
                    padding=True,
                    return_tensors="pt",
                    max_length=256,
                ).to(device)
                embeds = qenc_model(**enc).last_hidden_state[:, 0, :].cpu().numpy()
            scores, inds = medcpt_index.search(embeds, k=N * 2)
            for cond_idx, ind_list in enumerate(inds):
                for rank, ind in enumerate(ind_list):
                    nctid = medcpt_nctids[ind]
                    nctid2score[nctid] = nctid2score.get(nctid, 0.0) + (
                        1 / (rank + k)
                    ) * (1 / (cond_idx + 1))

        sorted_results = sorted(nctid2score.items(), key=lambda x: -x[1])
        top_nctids = [nctid for nctid, _ in sorted_results[:N]]

        # Build the trial objects that run_matching.py expects
        trial_objects = [
            corpus_meta[nctid]
            for nctid in top_nctids
            if nctid in corpus_meta
        ]

        # All retrieved trials go into label "1" (no ground-truth labels)
        output.append(
            {
                "patient_id": qid,
                "patient": patient_text,
                "1": trial_objects,
            }
        )

    # ── Save ──────────────────────────────────────────────────────────────
    out_path = Path(f"dataset/{CORPUS}/retrieved_trials.json")
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\n✓ Saved {len(output)} patients → {out_path}")
    print(f"  Each patient has up to {N} candidate trials (label '1')")

    # Also save a simpler qid→nctids mapping for reference
    simple_path = Path(f"results/retrieved_cardiology_simple.json")
    simple = {row["patient_id"]: [t["NCTID"] for t in row["1"]] for row in output}
    simple_path.parent.mkdir(exist_ok=True)
    simple_path.write_text(json.dumps(simple, indent=2))
    print(f"  Simple mapping → {simple_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--q_type", default="gpt-4-turbo")
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--top_n", type=int, default=50)
    parser.add_argument("--bm25_only", action="store_true")
    args = parser.parse_args()
    run_retrieval(args.q_type, args.k, args.top_n, args.bm25_only)
