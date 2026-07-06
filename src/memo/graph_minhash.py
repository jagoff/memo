"""MinHash + LSH blocking for entity-name near-duplicate candidates.

Pure Python (hashlib + math — no new deps). Used by the nightly
`memo dream entity-canon` pass to propose entity merge candidates WITHOUT
an O(n²) LLM sweep: 3-gram character shingles over the name → 64-permutation
MinHash signature (blake2b with per-permutation salt) → banded LSH bucketing.
Only pairs sharing ≥1 band bucket become candidates, and a Shannon-entropy +
length gate drops short/low-information names ("api", "aaaa") whose shingle
sets produce garbage similarity.

NOT in the recall-hook path: imported only by the nightly dream pass.
Dependency-free by design (mirrors graph_canonical.py).
"""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from collections.abc import Iterable

__all__ = [
    "candidate_pairs",
    "estimated_jaccard",
    "minhash_signature",
    "name_entropy",
    "shingles",
]

_NUM_HASHES = 64
_BANDS = 16  # 16 bands × 4 rows → catches Jaccard ≥ ~0.5 with high probability
_ROWS = _NUM_HASHES // _BANDS
_MIN_ENTROPY_BITS = 2.0
_MIN_NAME_LEN = 6


def shingles(name: str, n: int = 3) -> set[str]:
    """Character n-gram shingles over a name (whitespace collapsed)."""
    s = " ".join((name or "").lower().split())
    if not s:
        return set()
    if len(s) < n:
        return {s}
    return {s[i : i + n] for i in range(len(s) - n + 1)}


def name_entropy(name: str) -> float:
    """Shannon entropy (bits) of the character distribution of ``name``."""
    s = (name or "").lower()
    if not s:
        return 0.0
    counts: dict[str, int] = defaultdict(int)
    for ch in s:
        counts[ch] += 1
    total = len(s)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def _hash(shingle: str, salt: int) -> int:
    h = hashlib.blake2b(f"{salt}:{shingle}".encode(), digest_size=8)
    return int.from_bytes(h.digest(), "big")


def minhash_signature(
    shingle_set: set[str], num_hashes: int = _NUM_HASHES
) -> tuple[int, ...]:
    """Per-salt minimum blake2b hash over the shingle set."""
    if not shingle_set:
        return tuple([0] * num_hashes)
    return tuple(
        min(_hash(sh, salt) for sh in shingle_set) for salt in range(num_hashes)
    )


def estimated_jaccard(sig_a: tuple[int, ...], sig_b: tuple[int, ...]) -> float:
    """Fraction of agreeing signature slots ≈ Jaccard of the shingle sets."""
    if not sig_a or len(sig_a) != len(sig_b):
        return 0.0
    return sum(1 for a, b in zip(sig_a, sig_b, strict=True) if a == b) / len(sig_a)


def candidate_pairs(
    names: Iterable[str],
    *,
    min_jaccard: float = 0.5,
    min_entropy_bits: float = _MIN_ENTROPY_BITS,
    min_len: int = _MIN_NAME_LEN,
) -> list[tuple[str, str, float]]:
    """LSH-blocked near-duplicate name pairs, sorted by estimated Jaccard desc.

    Names failing the entropy/length gate are skipped entirely. Only pairs
    sharing at least one LSH band bucket are compared; each surviving pair
    carries its estimated Jaccard.
    """
    gated = [
        n
        for n in dict.fromkeys(names)
        if len(n) >= min_len and name_entropy(n) >= min_entropy_bits
    ]
    sigs = {n: minhash_signature(shingles(n)) for n in gated}
    buckets: dict[tuple[int, tuple[int, ...]], list[str]] = defaultdict(list)
    for n, sig in sigs.items():
        for band in range(_BANDS):
            key = (band, sig[band * _ROWS : (band + 1) * _ROWS])
            buckets[key].append(n)
    seen: set[tuple[str, str]] = set()
    out: list[tuple[str, str, float]] = []
    for members in buckets.values():
        if len(members) < 2:
            continue
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a, b = sorted((members[i], members[j]))
                if a == b or (a, b) in seen:
                    continue
                seen.add((a, b))
                est = estimated_jaccard(sigs[a], sigs[b])
                if est >= min_jaccard:
                    out.append((a, b, est))
    out.sort(key=lambda t: -t[2])
    return out
