"""Tests for the ⛔ AVOID surface in El Briefing / SessionStart.

Covers ``memo.briefing.negative_recall_lines`` and its wiring into
``memo_native_briefing_lines``: the section appears when relevant
failure_patterns exist, is absent when none / the flag is off, is size-capped,
prefers the current project, and stays off-cognition (surfaces the stored fact,
never a suggestion).

No real MLX: the pure-builder tests drive a fake ``mem`` exposing only ``list``;
the integration tests use the ``mock_memory`` fixture's stubbed embedder.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from memo.briefing import (
    _NEGATIVE_RECALL_MAX_ITEMS,
    memo_native_briefing_lines,
    negative_recall_lines,
)
from memo.negative_recall import AVOID_BLOCK_HEADER

_SECTION_HEADER = "### ⛔ Known pitfalls"


# ── fakes (no store, no MLX) ─────────────────────────────────────────────────


@dataclass(frozen=True)
class _Rec:
    """Minimal record matching the AvoidHit / MemoryLike structural protocol."""

    id: str
    title: str
    body: str
    type: str = "failure_pattern"
    extra: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)


class _FakeMem:
    """Mem double exposing only ``list`` — the read path the builder uses."""

    def __init__(self, records: list[_Rec]):
        self._records = list(records)

    def list(
        self,
        *,
        type_: str | None = None,
        limit: int = 20,
        include_forgotten: bool = False,
        updated_since: str | None = None,
    ) -> list[_Rec]:
        rows = [r for r in self._records if type_ is None or r.type == type_]
        return rows[:limit]


def _fp(
    idx: int, *, wrong: str = "did the wrong thing", right: str = "do the right thing", tags=()
):
    return _Rec(
        id=f"{idx:032x}",
        title=f"Avoid mistake {idx}",
        body=f"Pattern: pattern {idx}\nContext: ctx {idx}\nWrong: {wrong}\nRight: {right}",
        tags=list(tags),
    )


def _enable(monkeypatch) -> None:
    monkeypatch.setenv("MEMO_NEGATIVE_RECALL_ENABLED", "1")
    # Pin the current project to a value nothing matches so ordering stays pure
    # recency and no git subprocess runs (tests that exercise project-preference
    # override this).
    monkeypatch.setenv("MEMO_PROJECT_TAG", "project:zzz-nomatch")


# ── flag gating ──────────────────────────────────────────────────────────────


def test_section_absent_when_flag_off(monkeypatch):
    monkeypatch.delenv("MEMO_NEGATIVE_RECALL_ENABLED", raising=False)
    mem = _FakeMem([_fp(1), _fp(2)])

    assert negative_recall_lines(mem) == []


def test_section_appears_when_failure_patterns_exist(monkeypatch):
    _enable(monkeypatch)
    mem = _FakeMem([_fp(1, wrong="reindexed with an empty data_dir", right="mv + reclone")])

    lines = negative_recall_lines(mem)
    joined = "\n".join(lines)

    assert lines[0] == _SECTION_HEADER
    assert AVOID_BLOCK_HEADER in joined
    assert "reindexed with an empty data_dir" in joined
    assert "mv + reclone" in joined
    # The record id prefix anchors the surfaced fact to a real memory.
    assert _fp(1).id[:8] in joined


# ── empty corpus ─────────────────────────────────────────────────────────────


def test_section_absent_when_no_failure_patterns(monkeypatch):
    _enable(monkeypatch)
    # Corpus has non-failure records only; the type_ filter yields nothing.
    mem = _FakeMem([_Rec(id="a" * 32, title="a note", body="just a note", type="note")])

    assert negative_recall_lines(mem) == []


def test_section_absent_when_list_raises(monkeypatch):
    _enable(monkeypatch)

    class _Boom:
        def list(self, **_kw):
            raise RuntimeError("store offline")

    assert negative_recall_lines(_Boom()) == []


# ── size / count caps ────────────────────────────────────────────────────────


def test_section_capped_by_default_limit(monkeypatch):
    _enable(monkeypatch)
    mem = _FakeMem([_fp(i) for i in range(1, 8)])  # 7 available

    lines = negative_recall_lines(mem)
    joined = "\n".join(lines)

    # Only the first _NEGATIVE_RECALL_MAX_ITEMS numbered entries render.
    numbered_items = [ln for ln in joined.splitlines() if ln and ln[0].isdigit() and ". [" in ln]
    assert len(numbered_items) == _NEGATIVE_RECALL_MAX_ITEMS
    assert f"{_NEGATIVE_RECALL_MAX_ITEMS + 1}. [" not in joined


def test_explicit_limit_caps_items(monkeypatch):
    _enable(monkeypatch)
    mem = _FakeMem([_fp(i) for i in range(1, 6)])

    lines = negative_recall_lines(mem, limit=1)
    joined = "\n".join(lines)

    assert "1. [" in joined
    assert "2. [" not in joined


def test_zero_limit_yields_nothing(monkeypatch):
    _enable(monkeypatch)
    mem = _FakeMem([_fp(1)])

    assert negative_recall_lines(mem, limit=0) == []


def test_long_fields_are_truncated(monkeypatch):
    _enable(monkeypatch)
    long_wrong = "x" * 500
    mem = _FakeMem([_fp(1, wrong=long_wrong)])

    lines = negative_recall_lines(mem)
    wrong_line = next(ln for ln in "\n".join(lines).splitlines() if ln.strip().startswith("✗"))

    assert "…" in wrong_line  # truncated
    assert len(wrong_line) < 300  # bounded well below the 500-char field


# ── project preference ───────────────────────────────────────────────────────


def test_current_project_failure_patterns_surface_first(monkeypatch):
    monkeypatch.setenv("MEMO_NEGATIVE_RECALL_ENABLED", "1")
    monkeypatch.setenv("MEMO_PROJECT_TAG", "project:alpha")
    # Recency order (as mem.list returns): global first, project:alpha second.
    global_fp = _fp(1, wrong="global mistake")
    alpha_fp = _fp(2, wrong="alpha mistake", tags=["project:alpha"])
    mem = _FakeMem([global_fp, alpha_fp])

    lines = negative_recall_lines(mem)
    joined = "\n".join(lines)

    # Despite being less recent, the current-project pitfall ranks first.
    assert joined.index("alpha mistake") < joined.index("global mistake")


def test_unresolved_project_keeps_recency_order(monkeypatch, tmp_path):
    # No project pin + a non-git cwd → current_project_tag returns None, so the
    # order is left as mem.list returned it (recency).
    monkeypatch.setenv("MEMO_NEGATIVE_RECALL_ENABLED", "1")
    monkeypatch.delenv("MEMO_PROJECT_TAG", raising=False)
    first = _fp(1, wrong="most recent", tags=["project:alpha"])
    second = _fp(2, wrong="older")
    mem = _FakeMem([first, second])

    joined = "\n".join(negative_recall_lines(mem, cwd=str(tmp_path)))

    # Even though `first` carries a project tag, an unresolved current project
    # means no reordering — recency order is preserved.
    assert joined.index("most recent") < joined.index("older")


# ── off-cognition ────────────────────────────────────────────────────────────


def test_block_is_off_cognition(monkeypatch):
    _enable(monkeypatch)
    mem = _FakeMem([_fp(1)])

    joined = "\n".join(negative_recall_lines(mem)).lower()

    # Framed as data, not a directive.
    assert "data, not an instruction" in joined
    for verb in ("you should", "i suggest", "i recommend", "you must"):
        assert verb not in joined


# ── integration through the shared briefing builder (stubbed embed) ──────────


def test_appears_in_native_briefing_when_enabled(mock_memory, monkeypatch):
    _enable(monkeypatch)
    mock_memory.save(
        content="Pattern: p\nContext: c\nWrong: ran rebuild on empty data_dir\nRight: reclone",
        title="Avoid empty rebuild",
        type_="failure_pattern",
    )

    lines = memo_native_briefing_lines(mock_memory)
    joined = "\n".join(lines)

    assert _SECTION_HEADER in joined
    assert AVOID_BLOCK_HEADER in joined
    assert "ran rebuild on empty data_dir" in joined


def test_native_briefing_only_surfaces_failure_patterns(mock_memory, monkeypatch):
    _enable(monkeypatch)
    mock_memory.save(
        content="a plain note that is not a pitfall",
        title="Just a note",
        type_="note",
    )
    mock_memory.save(
        content="Pattern: p\nContext: c\nWrong: the pitfall body\nRight: the fix",
        title="Avoid pitfall",
        type_="failure_pattern",
    )

    # Isolated section exercises the real `mem.list(type_="failure_pattern")`
    # store filter — the plain note must not leak into the ⛔ block.
    block = "\n".join(negative_recall_lines(mock_memory))
    assert "the pitfall body" in block
    assert "Just a note" not in block
    assert "a plain note that is not a pitfall" not in block


def test_absent_from_native_briefing_when_flag_off(mock_memory, monkeypatch):
    monkeypatch.delenv("MEMO_NEGATIVE_RECALL_ENABLED", raising=False)
    mock_memory.save(
        content="Pattern: p\nContext: c\nWrong: w\nRight: r",
        title="Avoid something",
        type_="failure_pattern",
    )

    joined = "\n".join(memo_native_briefing_lines(mock_memory))

    assert _SECTION_HEADER not in joined
    assert AVOID_BLOCK_HEADER not in joined
