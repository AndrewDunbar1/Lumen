from __future__ import annotations

import math
import re
from collections import Counter


TOKEN_RE = re.compile(r"[a-z0-9]+")


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def normalize_text(text: str) -> str:
    return normalize_space(str(text or "").lower())


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(normalize_text(text))


def unique_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        clean = normalize_space(item)
        key = clean.lower()
        if clean and key not in seen:
            seen.add(key)
            out.append(clean)
    return out


def keyword_overlap(a_values: list[str], b_values: list[str]) -> float:
    a_set = {normalize_text(value) for value in a_values if normalize_text(value)}
    b_set = {normalize_text(value) for value in b_values if normalize_text(value)}
    if not a_set or not b_set:
        return 0.0
    inter = len(a_set & b_set)
    union = len(a_set | b_set)
    return inter / union if union else 0.0


def cosine_from_counters(a: Counter[str], b: Counter[str]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(a[token] * b[token] for token in set(a) & set(b))
    norm_a = math.sqrt(sum(value * value for value in a.values()))
    norm_b = math.sqrt(sum(value * value for value in b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def sentence_split(text: str) -> list[str]:
    raw = re.split(r"(?:\n+|(?<=[.!?])\s+)", str(text or ""))
    return [normalize_space(item) for item in raw if normalize_space(item)]
