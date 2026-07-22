"""Reference-tier noise gate: near-empty bulk chunks with no heading/link must
not be indexable via `memo_save`, while short *durable* facts stay allowed.
"""

from __future__ import annotations

import pytest

from memo.memory.record import is_reference_noise


def test_is_reference_noise_predicate():
    assert is_reference_noise("...") is True
    assert is_reference_noise("   \n  ") is True
    assert is_reference_noise("frag") is True
    # heading → kept
    assert is_reference_noise("# Title") is False
    # wikilink / md link / URL → kept
    assert is_reference_noise("[[Ideas]]") is False
    assert is_reference_noise("[docs](url)") is False
    assert is_reference_noise("see https://example.com") is False
    # long enough → kept regardless
    assert is_reference_noise("x" * 80) is False
    # pathological ReDoS input must not hang (linear substring scans)
    assert is_reference_noise("[a](" * 60_000) is False


def test_save_rejects_reference_noise(mock_memory):
    with pytest.raises(ValueError, match="near-empty noise"):
        mock_memory.save(content="frag", type_="reference")


def test_save_allows_short_durable_fact(mock_memory):
    # Same short body, durable tier — a real preference must NOT be rejected.
    rec = mock_memory.save(content="User prefers dark mode", type_="preference")
    assert rec.id


def test_save_allows_reference_with_heading(mock_memory):
    rec = mock_memory.save(content="# Arquitectura", type_="reference")
    assert rec.id
