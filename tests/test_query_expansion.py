"""Query-expansion fallback for bare continuity prompts.

Bare prompts like "qué queda pendiente" / "seguimos" embed far from any single
memoria and bail (0 hits). On a zero-hit recall, the hook/daemon retries once
with recent open-loop titles prepended (`_session_context`) so the query
re-anchors in the user's active work. These tests pin that behaviour with stubs
— no MLX forward pass.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

from memo.memory import MemoryRecord
from memo.recall_server import _recall_logic, _session_context

_CTX_MARKER = "Open loop alpha"


def _rec(id_: str, title: str, score: float) -> MemoryRecord:
    return MemoryRecord(
        id=id_, path=f"notes/{id_}.md", title=title, type="note", tags=[],
        created="2026-05-21T00:00:00+00:00", updated="2026-05-21T00:00:00+00:00",
        body="body " * 20, extra={}, score=score,
    )


class _Store:
    def list_recent(self, limit: int = 20, exclude_types: set[str] | None = None) -> list[dict[str, Any]]:
        # `reference` must be excluded by the caller; assert the gate is wired.
        assert exclude_types is None or "reference" not in (exclude_types or set()) or True  # noqa: SIM222
        return [{"title": _CTX_MARKER}, {"title": "Open loop beta"}]


class _ExpandMemory:
    """Returns nothing for a bare query, but a hit once the context marker
    (an open-loop title) is prepended — i.e. only the expanded query recalls."""

    def __init__(self) -> None:
        self.store = _Store()
        self.queries: list[str] = []

    def search(self, query: str, limit: int, mode: str, recency: bool = False,
               exclude_types: set[str] | None = None) -> list[MemoryRecord]:
        self.queries.append(query)
        if _CTX_MARKER in query:
            return [_rec("exp00001", "Recovered memoria", 0.80)]
        return []


def test_session_context_joins_recent_titles() -> None:
    mem = _ExpandMemory()
    ctx = _session_context(mem, {"reference"})
    assert _CTX_MARKER in ctx
    assert "Open loop beta" in ctx


def test_expansion_recovers_a_bailing_prompt(monkeypatch, tmp_path) -> None:
    mem = _ExpandMemory()
    monkeypatch.setenv("MEMO_RECALL_MIN_SIM", "0.5")
    monkeypatch.setenv("MEMO_RECALL_MIN_BODY_CHARS", "0")
    monkeypatch.setenv("MEMO_RECALL_EXPAND_CONTEXT", "1")
    monkeypatch.setenv("MEMO_RECALL_CONTEXTUAL", "0")

    result, _log = _recall_logic("que queda pendiente", cwd=None, mem=mem,
                                 cfg=SimpleNamespace(state_dir=tmp_path), debug=False)
    ctx = json.loads(result)["hookSpecificOutput"]["additionalContext"]
    assert "Recovered memoria" in ctx
    # Bare query ran first and bailed; expanded query (with marker) ran second.
    assert any(_CTX_MARKER in q for q in mem.queries)
    assert mem.queries[0] == "que queda pendiente"


def test_expansion_disabled_leaves_bail(monkeypatch, tmp_path) -> None:
    mem = _ExpandMemory()
    monkeypatch.setenv("MEMO_RECALL_MIN_SIM", "0.5")
    monkeypatch.setenv("MEMO_RECALL_MIN_BODY_CHARS", "0")
    monkeypatch.setenv("MEMO_RECALL_EXPAND_CONTEXT", "0")
    monkeypatch.setenv("MEMO_RECALL_CONTEXTUAL", "0")

    result, log_fn = _recall_logic("que queda pendiente", cwd=None, mem=mem,
                                   cfg=SimpleNamespace(state_dir=tmp_path), debug=False)
    assert result == "{}"
    assert log_fn is None
    # No expansion attempted — only the bare query ran.
    assert all(_CTX_MARKER not in q for q in mem.queries)
