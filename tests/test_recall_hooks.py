from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from memo.memory import MemoryRecord
from memo.recall_server import _apply_project_boost, _recall_logic


def _rec(id_: str, title: str, score: float, tags: list[str] | None = None) -> MemoryRecord:
    return MemoryRecord(
        id=id_,
        path=f"notes/{id_}.md",
        title=title,
        type="note",
        tags=tags or [],
        created="2026-05-21T00:00:00+00:00",
        updated="2026-05-21T00:00:00+00:00",
        body="body " * 20,
        extra={},
        score=score,
    )


def test_apply_project_boost_copies_frozen_records_and_resorts() -> None:
    global_hit = _rec("global01", "Global", 0.70)
    project_hit = _rec("project1", "Project", 0.60, ["project:memo"])

    boosted = _apply_project_boost([global_hit, project_hit], "project:memo", 0.15)

    assert [h.id for h in boosted] == ["project1", "global01"]
    assert boosted[0].score == pytest.approx(0.75)
    assert project_hit.score == pytest.approx(0.60)


def test_recall_logic_project_boost_handles_frozen_records(monkeypatch, tmp_path) -> None:
    global_hit = _rec("global01", "Global", 0.70)
    project_hit = _rec("project1", "Project", 0.60, ["project:memo"])

    class StubMemory:
        def search(self, query: str, limit: int, mode: str) -> list[MemoryRecord]:
            return [global_hit, project_hit]

    monkeypatch.setenv("MEMO_PROJECT_TAG", "memo")
    monkeypatch.setenv("MEMO_RECALL_MIN_SIM", "0.0")
    monkeypatch.setenv("MEMO_RECALL_MIN_BODY_CHARS", "0")

    result = _recall_logic(
        "project-specific query",
        cwd=str(tmp_path),
        mem=StubMemory(),
        cfg=SimpleNamespace(state_dir=tmp_path),
        debug=False,
    )

    payload = json.loads(result)
    context = payload["hookSpecificOutput"]["additionalContext"]

    assert context.index("Project") < context.index("Global")
    assert "score 0.75" in context
