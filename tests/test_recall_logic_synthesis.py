"""Tests for _deduplicate_synthesis in recall_logic."""

from __future__ import annotations

from types import SimpleNamespace

from memo.recall_logic import _deduplicate_synthesis


def _hit(id: str, type: str = "note", extra: dict | None = None) -> SimpleNamespace:
    return SimpleNamespace(id=id, type=type, extra=extra)


def test_deduplicate_synthesis_no_synthesis() -> None:
    """No synthesis hits — list is returned unchanged."""
    hits = [_hit("aaa"), _hit("bbb"), _hit("ccc")]
    result = _deduplicate_synthesis(hits)
    assert [h.id for h in result] == ["aaa", "bbb", "ccc"]


def test_deduplicate_synthesis_removes_sources() -> None:
    """Synthesis covers two sources — sources are removed."""
    synth = _hit("synth1", type="synthesis", extra={"synthesis_sources": ["src1", "src2"]})
    src1 = _hit("src1")
    src2 = _hit("src2")
    hits = [synth, src1, src2]
    result = _deduplicate_synthesis(hits)
    assert [h.id for h in result] == ["synth1"]


def test_deduplicate_synthesis_partial() -> None:
    """Synthesis covers 2 of 3 sources; uncovered source stays."""
    synth = _hit("synth1", type="synthesis", extra={"synthesis_sources": ["src1", "src2"]})
    src1 = _hit("src1")
    src2 = _hit("src2")
    src3 = _hit("src3")
    hits = [synth, src1, src2, src3]
    result = _deduplicate_synthesis(hits)
    assert [h.id for h in result] == ["synth1", "src3"]


def test_deduplicate_synthesis_bad_extra() -> None:
    """Synthesis with extra=None must not raise; list is returned unchanged."""
    synth = _hit("synth1", type="synthesis", extra=None)
    src = _hit("src1")
    hits = [synth, src]
    result = _deduplicate_synthesis(hits)
    # extra=None → no covered_ids → nothing removed
    assert [h.id for h in result] == ["synth1", "src1"]
