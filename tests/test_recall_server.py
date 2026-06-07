"""Unit tests for recall_server's pure ranking/dedup helpers.

These run without MLX (recall_server defers all MLX imports). The socket /
percentile surface is covered by test_embedder_client.py; this fills the
ranking-logic gap: project boost, near-duplicate collapse, and the cheap
session-context continuity read.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from memo.recall_server import (
    _apply_project_boost,
    _dedup_key,
    _session_context,
    dedup_hits,
)


@dataclass(frozen=True)
class _Hit:
    id: str
    title: str = ""
    body: str = ""
    type: str = "note"
    score: float | None = None
    tags: tuple[str, ...] = field(default_factory=tuple)


# -- _apply_project_boost --------------------------------------------------


def test_project_boost_noop_without_tag():
    hits = [_Hit("1", score=0.5), _Hit("2", score=0.9)]
    out = _apply_project_boost(hits, None, 0.15)
    assert [h.id for h in out] == ["1", "2"]  # untouched order, no boost


def test_project_boost_lifts_and_resorts_matching_tag():
    hits = [
        _Hit("plain", score=0.80, tags=("other",)),
        _Hit("proj", score=0.70, tags=("project:memo",)),
    ]
    out = _apply_project_boost(hits, "project:memo", 0.15)
    # proj 0.70+0.15=0.85 now outranks plain 0.80
    assert out[0].id == "proj"
    assert out[0].score == 0.85
    assert out[1].score == 0.80  # untagged hit unchanged


def test_project_boost_skips_hits_without_score():
    hits = [_Hit("noscore", score=None, tags=("project:memo",))]
    out = _apply_project_boost(hits, "project:memo", 0.15)
    assert out[0].score is None


# -- _dedup_key / dedup_hits -----------------------------------------------


def test_dedup_key_normalises_whitespace_and_case():
    a = _Hit("1", title="  Hello   World ", body="Body Text")
    b = _Hit("2", title="hello world", body="body text")
    assert _dedup_key(a) == _dedup_key(b)


def test_dedup_hits_drops_duplicate_ids_keeping_first():
    hits = [_Hit("x", title="a"), _Hit("x", title="b")]
    out = dedup_hits(hits)
    assert len(out) == 1
    assert out[0].title == "a"  # first (highest-ranked) kept


def test_dedup_hits_drops_near_identical_title_body():
    hits = [
        _Hit("1", title="Same Fact", body="detail"),
        _Hit("2", title="same   fact", body="DETAIL"),
        _Hit("3", title="Different", body="x"),
    ]
    out = dedup_hits(hits)
    assert [h.id for h in out] == ["1", "3"]


# -- _session_context ------------------------------------------------------


class _FakeStore:
    def __init__(self, rows):
        self._rows = rows
        self.seen_kwargs = None

    def list_recent(self, limit, exclude_types=None):
        self.seen_kwargs = {"limit": limit, "exclude_types": exclude_types}
        return self._rows


class _FakeMem:
    def __init__(self, store):
        self.store = store


def test_session_context_joins_titles_and_caps():
    rows = [{"title": f"loop {i}"} for i in range(10)]
    mem = _FakeMem(_FakeStore(rows))
    out = _session_context(mem, exclude_types={"reference"}, max_titles=3)
    assert out == "loop 0 ; loop 1 ; loop 2"
    assert mem.store.seen_kwargs["exclude_types"] == {"reference"}


def test_session_context_filters_blank_titles():
    rows = [{"title": "  "}, {"title": "real"}, {"title": None}]
    mem = _FakeMem(_FakeStore(rows))
    assert _session_context(mem, None) == "real"


def test_session_context_returns_empty_on_error():
    class _BoomMem:
        @property
        def store(self):
            raise RuntimeError("no store")

    assert _session_context(_BoomMem(), None) == ""
