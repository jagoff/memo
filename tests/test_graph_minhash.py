"""Tests for graph_minhash — pure MinHash+LSH blocking for entity names."""

from __future__ import annotations

from memo.graph_minhash import (
    candidate_pairs,
    estimated_jaccard,
    minhash_signature,
    name_entropy,
    shingles,
)


def test_shingles_3gram_and_short_name():
    assert shingles("fastapi") == {"fas", "ast", "sta", "tap", "api"}
    assert shingles("ab") == {"ab"}
    assert shingles("") == set()


def test_name_entropy_flags_degenerate_names():
    assert name_entropy("aaaaaaaa") == 0.0
    assert name_entropy("memo recall daemon") > 2.0


def test_minhash_signature_deterministic_and_sized():
    sig = minhash_signature(shingles("memo recall daemon"))
    assert len(sig) == 64
    assert sig == minhash_signature(shingles("memo recall daemon"))


def test_estimated_jaccard_orders_near_dupes_above_strangers():
    a = minhash_signature(shingles("memo recall daemon"))
    b = minhash_signature(shingles("memo recall daemons"))
    c = minhash_signature(shingles("kubernetes networking"))
    assert estimated_jaccard(a, a) == 1.0
    assert estimated_jaccard(a, b) > estimated_jaccard(a, c)


def test_candidate_pairs_blocks_near_dupes_only():
    pairs = candidate_pairs(
        ["memo recall daemon", "memo recall daemons", "kubernetes networking"]
    )
    assert [(a, b) for a, b, _ in pairs] == [
        ("memo recall daemon", "memo recall daemons")
    ]


def test_candidate_pairs_entropy_and_length_gate():
    # "aaaaaaaa"/"aaaaaaaas" fail the entropy gate; "api"/"apis" fail min_len.
    assert candidate_pairs(["aaaaaaaa", "aaaaaaaas", "api", "apis"]) == []
