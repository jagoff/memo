"""Tests for entity_extractor — regex path + flag gating + match scoring.

The regex extractor is dependency-free and is the only extraction path.
"""

from __future__ import annotations

import pytest

from memo import entity_extractor as ee


def test_entity_retrieval_enabled_reads_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MEMO_ENTITY_RETRIEVAL_ENABLED", raising=False)
    assert ee.entity_retrieval_enabled() is False
    monkeypatch.setenv("MEMO_ENTITY_RETRIEVAL_ENABLED", "true")  # not just "1"
    assert ee.entity_retrieval_enabled() is True
    monkeypatch.setenv("MEMO_ENTITY_RETRIEVAL_ENABLED", "0")
    assert ee.entity_retrieval_enabled() is False


def test_extract_entities_empty() -> None:
    assert ee.extract_entities("") == []
    assert ee.extract_entities("   ") == []


def test_extract_entities_regex_finds_proper_nouns() -> None:
    ents = ee.extract_entities("Fernando works on Synapse and Memflow in Buenos Aires")
    low = {e.lower() for e in ents}
    assert "synapse" in low
    assert "memflow" in low


def test_extract_entities_backtick_identifiers() -> None:
    ents = ee.extract_entities("the `VecStore` class uses `sqlite-vec`")
    assert "VecStore" in ents
    assert "sqlite-vec" in ents


def test_extract_entities_dedups() -> None:
    ents = ee.extract_entities("Synapse Synapse Synapse")
    assert ents.count("Synapse") == 1


def test_entity_match_score_overlap_and_cap() -> None:
    assert ee.entity_match_score([], ["a"]) == 0.0
    assert ee.entity_match_score(["a"], []) == 0.0
    # one overlap (case-insensitive)
    assert ee.entity_match_score(["Synapse"], ["synapse"]) == pytest.approx(0.05)
    # capped at max_boost
    q = [f"e{i}" for i in range(10)]
    assert ee.entity_match_score(q, q) == pytest.approx(0.2)
