from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping

TOKEN_RE = re.compile(r"[a-z0-9_+*/-]+")


def tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def token_profile(texts: Iterable[str], *, limit: int = 32) -> dict[str, float]:
    counts: Counter[str] = Counter()
    documents = 0
    for text in texts:
        documents += 1
        counts.update(set(tokens(text)))
    if not documents:
        return {}
    return {token: count / documents for token, count in counts.most_common(limit)}


def cosine(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    common = set(left) & set(right)
    dot = sum(left[key] * right[key] for key in common)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)


def task_profile(task_text: str) -> dict[str, float]:
    return {token: 1.0 for token in set(tokens(task_text))}
