"""Noise-quantile relevance floor (floor-calibration).

At nightly Dream, embed word-shuffled probes and estimate the high quantile of
the null cosine distribution for THIS (embedder, corpus). That quantile is the
smallest similarity that still beats noise → a data-driven MEMO_RECALL_MIN_SIM
floor. Runs only at Dream (MLX allowed, deferred import). Pure math here; the
Dream pass owns the gate + overlay write.
"""

from __future__ import annotations

import itertools
from typing import Any


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def estimate_noise_floor(
    embedder: Any, *, probes: list[str], quantile: float = 0.95, dims: int | None = None
) -> float | None:
    """High-quantile pairwise cosine among unrelated probe embeddings = the null
    ceiling. Returns None when there are too few probes to form a pair."""
    if len(probes) < 2:
        return None
    vecs = embedder.embed(list(probes))  # Sequence[str] — never a bare str
    sims = sorted(_cosine(vecs[i], vecs[j]) for i, j in itertools.combinations(range(len(vecs)), 2))
    if not sims:
        return None
    idx = min(len(sims) - 1, round(quantile * (len(sims) - 1)))
    return max(0.0, min(1.0, round(sims[idx], 4)))
