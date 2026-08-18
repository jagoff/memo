"""Regression tests for `_ConsolidateOpsMixin._average_link_cluster` — the
merge-consolidation clustering primitive that replaced `_greedy_cluster`
(`_cluster_within_scope` is its only caller).

Two failure modes drove the change, both reproduced here with hand-derived
unit vectors (exact cosines, no floating-point guesswork):

- `test_greedy_splits_an_above_threshold_pair_average_link_does_not`: greedy
  compares a new item only to each existing cluster's FIRST member, frozen as
  its "representative" forever. Q and R here are each other's best match
  (cos ≈ 0.978), but Q gets folded into P's cluster before R arrives, and R is
  only ever compared to P (cos ≈ 0.891, just under threshold) — so greedy
  splits an above-threshold pair across two different clusters. Measured on
  the live corpus: 38.4% of above-threshold pairs split this way.
- `test_average_link_does_not_chain_like_single_linkage`: A-B and B-C are each
  individually above threshold, but A and C share nothing (cos ≈ 0.61).
  Single-linkage (connected components of the threshold graph) transitively
  chains all three into one cluster through the B bridge — measured on the
  live corpus, a single (project, type) bucket chained 157 of its 950 memories
  this way. Average-link's cutoff is on the CLUSTER AVERAGE, so once A and B
  merge (the stronger of the two edges — deliberately unequal so the merge
  order is deterministic, not a near-tie the numpy vs pure-python cosine
  rounding could break differently), C's average similarity to {A, B} (0.74)
  dilutes below threshold and the merge stops.

See `_average_link_cluster`'s docstring in
`src/memo/memory/consolidate_ops.py` for the full empirical comparison
(pair-recall / max-cluster-size / purity) against greedy and single-linkage.
"""

import math
import sys

from memo.memory.consolidate_ops import _ConsolidateOpsMixin


def _unit(deg: float) -> list[float]:
    rad = math.radians(deg)
    return [math.cos(rad), math.sin(rad)]


def _items(vectors: list[list[float]]) -> list[dict]:
    return [{"id": str(i), "emb": v} for i, v in enumerate(vectors)]


def test_greedy_splits_an_above_threshold_pair_average_link_does_not():
    # P at 0°, Q at 15° (cos(P,Q)=0.966), R at 27° (cos(P,R)=0.891, cos(Q,R)=0.978).
    # Threshold 0.9: P-Q and Q-R clear it; P-R does not.
    vectors = [_unit(0), _unit(15), _unit(27)]
    items = _items(vectors)
    threshold = 0.9

    greedy = _ConsolidateOpsMixin._greedy_cluster(items, threshold)
    greedy_of = {i: ci for ci, cluster in enumerate(greedy) for i in cluster}
    # The documented bug, reproduced: Q (1) and R (2) are each other's closest
    # match and both clear the threshold, but greedy splits them because R is
    # only ever compared to P's frozen representative.
    assert greedy_of[1] != greedy_of[2], "expected the pre-existing greedy bug to reproduce"

    avg_link = _ConsolidateOpsMixin._average_link_cluster(items, threshold)
    avg_of = {i: ci for ci, cluster in enumerate(avg_link) for i in cluster}
    assert avg_of[1] == avg_of[2], "average-link must keep the above-threshold Q,R pair together"


def test_average_link_does_not_chain_like_single_linkage():
    # A-B (cos=0.92) and B-C (cos=0.87) are each >= threshold, but deliberately
    # unequal — a real tie between them (both exactly 0.9, say) would let the
    # numpy matmul path and the pure-python dot-product path round to opposite
    # winners and merge in a different order. A-C is cos=0.61, far below.
    d_ab = math.degrees(math.acos(0.92))
    d_bc = math.degrees(math.acos(0.87))
    vectors = [_unit(0), _unit(d_ab), _unit(d_ab + d_bc)]
    items = _items(vectors)
    threshold = 0.85

    clusters = _ConsolidateOpsMixin._average_link_cluster(items, threshold)
    sizes = sorted(len(c) for c in clusters)
    # NOT single-linkage's one chained cluster of 3 — A and C's shared
    # dissimilarity (cos 0.61) drags the {A,B}-vs-C average (0.74) below
    # threshold once A and B have merged, so the merge stops there.
    assert sizes == [1, 2], f"expected a bounded {{A,B}},{{C}} split, got sizes {sizes}"

    # The pair that IS above threshold on both sides (A,B) stays together;
    # the unrelated pair (A,C) does not get chained in via the B bridge.
    a_cluster = next(c for c in clusters if 0 in c)
    assert 1 in a_cluster
    assert 2 not in a_cluster


def test_average_link_pure_python_fallback_matches_numpy_path(monkeypatch):
    """The numpy-less fallback (exercised when numpy isn't installed — an
    optional dependency, per pyproject's mypy-override comment) must produce
    the same clustering as the numpy path for the same inputs.

    The four consecutive similarities are deliberately unequal (0.93/0.88/0.91,
    not a repeated 0.9): a near-exact tie between two candidate merge pairs
    would let numpy's vectorised matmul and the pure-python manual dot product
    round to opposite winners (this happened during development — an early
    all-0.9 chain merged (0,1) first under one path and (1,2) first under the
    other, diverging from there) and fail this assertion by construction,
    not because either path is wrong."""
    d1 = math.degrees(math.acos(0.93))
    d2 = math.degrees(math.acos(0.88))
    d3 = math.degrees(math.acos(0.91))
    angles = [0.0, d1, d1 + d2, d1 + d2 + d3]
    vectors = [_unit(a) for a in angles]
    items = _items(vectors)
    threshold = 0.85

    numpy_result = _ConsolidateOpsMixin._average_link_cluster(items, threshold)

    monkeypatch.setitem(sys.modules, "numpy", None)
    fallback_result = _ConsolidateOpsMixin._average_link_cluster(items, threshold)

    def _as_sets(clusters: list[list[int]]) -> set[frozenset[int]]:
        return {frozenset(c) for c in clusters}

    assert _as_sets(numpy_result) == _as_sets(fallback_result)


def test_average_link_singletons_pass_through_untouched():
    """Items with no above-threshold partner at all stay singleton clusters —
    matching `_greedy_cluster`'s behaviour, so callers that drop clusters of
    size 1 (`_consolidate_in_process`) see the same shape either way."""
    vectors = [_unit(0), _unit(90), _unit(180)]
    items = _items(vectors)

    clusters = _ConsolidateOpsMixin._average_link_cluster(items, threshold=0.85)

    assert sorted(len(c) for c in clusters) == [1, 1, 1]


def test_average_link_empty_input_returns_empty():
    assert _ConsolidateOpsMixin._average_link_cluster([], threshold=0.85) == []


def test_dense_component_clusters_in_bounded_time():
    """One dense component of 500 near-identical vectors — the shape the
    corpus-scale conformance fixture produces (20 topics × ~500 members whose
    within-topic cosine is ~0.999). The first UPGMA implementation recomputed
    every cross-cluster block mean per candidate pair per merge (O(k^4) element
    touches), which turned `memo_consolidate` over that fixture into ~2h of
    GIL-holding work — CI's conformance step timed out twice on it, and the
    still-running FastMCP worker starved the *next* test in the process too.
    Lance-Williams updates make the same merge O(k^3) at C speed (~0.1s).

    The 30s bound is ~300× the measured time — loose enough for any CI runner,
    tight enough that an O(k^4) regression (minutes at minimum) can never pass.
    """
    import random
    import time

    rng = random.Random(7)  # noqa: S311 — deterministic test fixture, not crypto
    base = [rng.gauss(0, 1) for _ in range(64)]
    items = []
    for i in range(500):
        v = [b + rng.gauss(0, 0.005) for b in base]
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        items.append({"id": str(i), "emb": [x / norm for x in v]})

    start = time.monotonic()
    clusters = _ConsolidateOpsMixin._average_link_cluster(items, threshold=0.85)
    elapsed = time.monotonic() - start

    assert elapsed < 30, f"dense-component clustering took {elapsed:.1f}s (O(k^4) regression?)"
    # Near-identical vectors are one coherent cluster, not a shredded pile.
    assert max(len(c) for c in clusters) == 500
