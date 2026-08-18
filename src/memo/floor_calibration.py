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
    """High-quantile similarity between UNRELATED query/document pairs = the
    null ceiling. Returns None when there are too few probes to form a pair.

    The pairs are asymmetric on purpose — each probe is encoded once as a QUERY
    (`embed_query`, with the retrieval instruction prefix) and once as a
    DOCUMENT (`embed`, without it), and only cross pairs `i != j` are scored.
    That is the geometry the floor is actually compared against in
    `rank_hits._passes`. Scoring document-document pairs instead put the null
    ceiling in a different region of the space entirely and proposed 0.8033 on
    a live corpus whose real query-doc scores sit far lower — a floor that
    buried most of recall.
    """
    if len(probes) < 2:
        return None
    doc_vecs = embedder.embed(list(probes))  # Sequence[str] — never a bare str
    query_vecs = [embedder.embed_query(p) for p in probes]
    sims = sorted(
        _cosine(query_vecs[i], doc_vecs[j])
        for i, j in itertools.permutations(range(len(probes)), 2)
    )
    if not sims:
        return None
    idx = min(len(sims) - 1, round(quantile * (len(sims) - 1)))
    return max(0.0, min(1.0, round(sims[idx], 4)))
