from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from memo.memory import MemoryRecord
from memo.recall_server import (
    RECALL_DIRECTIVE,
    _apply_preference_boost,
    _apply_project_boost,
    _recall_logic,
    dedup_hits,
)


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
        def search(self, query: str, limit: int, mode: str, recency: bool = False, exclude_types=None) -> list[MemoryRecord]:
            return [global_hit, project_hit]

    monkeypatch.setenv("MEMO_PROJECT_TAG", "memo")
    monkeypatch.setenv("MEMO_RECALL_MIN_SIM", "0.0")
    monkeypatch.setenv("MEMO_RECALL_MIN_BODY_CHARS", "0")

    result, _log = _recall_logic(
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


def test_dedup_hits_drops_duplicate_id_and_near_identical_content() -> None:
    a = _rec("id000001", "Decisión MLX", 0.80)
    a_dup_id = _rec("id000001", "Decisión MLX", 0.50)  # same id, lower score
    a_near = _rec("id000002", "Decisión MLX", 0.70)    # different id, same title+body
    b = _rec("id000003", "Otra cosa distinta", 0.65)

    out = dedup_hits([a, a_dup_id, a_near, b])

    ids = [h.id for h in out]
    assert ids == ["id000001", "id000003"]  # dup id + near-dup content collapsed


def test_recall_logic_emits_authority_directive(monkeypatch, tmp_path) -> None:
    hit = _rec("auth0001", "Some fact", 0.80)

    class StubMemory:
        def search(self, query: str, limit: int, mode: str, recency: bool = False, exclude_types=None) -> list[MemoryRecord]:
            return [hit]

    monkeypatch.setenv("MEMO_RECALL_MIN_SIM", "0.0")
    monkeypatch.setenv("MEMO_RECALL_MIN_BODY_CHARS", "0")

    result, _log = _recall_logic(
        "anything", cwd=None, mem=StubMemory(),
        cfg=SimpleNamespace(state_dir=tmp_path), debug=False,
    )
    context = json.loads(result)["hookSpecificOutput"]["additionalContext"]
    assert RECALL_DIRECTIVE in context
    assert "authoritative" in context.lower()


def test_recall_logic_passes_recency_to_search(monkeypatch, tmp_path) -> None:
    seen = {}

    class StubMemory:
        def search(self, query: str, limit: int, mode: str, recency: bool = False, exclude_types=None) -> list[MemoryRecord]:
            seen["recency"] = recency
            return [_rec("r0000001", "Fresh", 0.9)]

    monkeypatch.setenv("MEMO_RECALL_MIN_SIM", "0.0")
    monkeypatch.setenv("MEMO_RECALL_MIN_BODY_CHARS", "0")
    _recall_logic("q", cwd=None, mem=StubMemory(),
                  cfg=SimpleNamespace(state_dir=tmp_path), debug=False)
    assert seen["recency"] is True


def test_apply_preference_boost_reorders_by_learned_type() -> None:
    note = _rec("n0000001", "a note", 0.70)
    decision = _rec("d0000001", "a decision", 0.68)
    object.__setattr__(decision, "type", "decision")  # frozen record

    prefs = SimpleNamespace(preferred_types={"decision": 0.9})
    out = _apply_preference_boost([note, decision], prefs)

    # decision was behind on raw score but the learned type pref lifts it
    assert next(h.id for h in out) == "d0000001"
    # empty prefs → unchanged order
    same = _apply_preference_boost([note, decision], SimpleNamespace(preferred_types={}))
    assert [h.id for h in same] == ["n0000001", "d0000001"]


def test_recall_logic_records_what_surfaced(monkeypatch, tmp_path) -> None:
    recorded = {}

    class FakeContextual:
        class context:
            @staticmethod
            def get_preferences():
                return SimpleNamespace(preferred_types={})
        @staticmethod
        def record_search(prompt, ids):
            recorded["prompt"] = prompt
            recorded["ids"] = ids

    class StubMemory:
        contextual = FakeContextual()
        def search(self, query: str, limit: int, mode: str, recency: bool = False, exclude_types=None) -> list[MemoryRecord]:
            return [_rec("surf0001", "surfaced", 0.9)]

    monkeypatch.setenv("MEMO_RECALL_MIN_SIM", "0.0")
    monkeypatch.setenv("MEMO_RECALL_MIN_BODY_CHARS", "0")
    _recall_logic("mi pregunta", cwd=None, mem=StubMemory(),
                  cfg=SimpleNamespace(state_dir=tmp_path), debug=False)
    assert recorded["prompt"] == "mi pregunta"
    assert recorded["ids"] == ["surf0001"]


def test_recall_logic_adds_related_nudge_below_the_cut(monkeypatch, tmp_path) -> None:
    hits = [_rec(f"id{i:07d}", f"hit {i}", 0.9 - i * 0.05) for i in range(5)]

    class StubMemory:
        def search(self, query: str, limit: int, mode: str, recency: bool = False, exclude_types=None) -> list[MemoryRecord]:
            return hits

    monkeypatch.setenv("MEMO_RECALL_TOP_K", "3")
    monkeypatch.setenv("MEMO_RECALL_MIN_SIM", "0.0")
    monkeypatch.setenv("MEMO_RECALL_MIN_BODY_CHARS", "0")
    monkeypatch.setenv("MEMO_RECALL_CONTEXTUAL", "0")  # isolate from prefs

    result, _log = _recall_logic("q", cwd=None, mem=StubMemory(),
                           cfg=SimpleNamespace(state_dir=tmp_path), debug=False)
    context = json.loads(result)["hookSpecificOutput"]["additionalContext"]
    # top-3 in the main block, next 2 in the related nudge
    nudge_line = context.split("relacionado):", 1)[1]
    assert "hit 3" in nudge_line and "hit 4" in nudge_line
    assert "hit 0" not in nudge_line  # top hits stay in the main block


class _StubMicroEmbedder:
    """Deterministic 2-dim embedder for the cold-embedder fallback path."""

    def __init__(self, query_vec: list[float], doc_vecs: list[list[float]]) -> None:
        self._query_vec = query_vec
        self._doc_vecs = doc_vecs

    def embed_query(self, text: str) -> list[float]:
        return self._query_vec

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._doc_vecs[: len(texts)]


def test_fallback_scoring_does_not_mutate_shared_hits(monkeypatch, tmp_path) -> None:
    """The micro-embedder fallback must score into NEW records, never mutate
    the shared frozen hits returned by `search`."""
    def _body_rec(id_: str, title: str) -> MemoryRecord:
        # distinct bodies so dedup doesn't collapse them as near-identical
        return MemoryRecord(
            id=id_, path=f"notes/{id_}.md", title=title, type="note", tags=[],
            created="2026-05-21T00:00:00+00:00", updated="2026-05-21T00:00:00+00:00",
            body=f"unique body for {title} " * 8, extra={}, score=0.0,
        )

    a = _body_rec("aaaa0001", "Alpha")
    b = _body_rec("bbbb0001", "Beta")
    candidates = [a, b]

    class StubMemory:
        embedder = SimpleNamespace(is_warm=False)

        def search(self, query, limit, mode, recency=False, exclude_types=None):
            return candidates

        def _read_body(self, path):
            return "body"

    # query closer to Beta (doc index 1) than Alpha → Beta must rank first
    micro = _StubMicroEmbedder(query_vec=[0.0, 1.0], doc_vecs=[[1.0, 0.0], [0.0, 1.0]])

    monkeypatch.setenv("MEMO_RECALL_MIN_SIM", "0.0")
    monkeypatch.setenv("MEMO_RECALL_MIN_BODY_CHARS", "0")
    monkeypatch.setenv("MEMO_RECALL_CONTEXTUAL", "0")
    # neutralise dev-env trimming so both hits surface deterministically
    monkeypatch.setenv("MEMO_RECALL_GAP_THRESHOLD", "0")
    monkeypatch.setenv("MEMO_RECALL_SKIP_BELOW", "0")

    result, _log = _recall_logic(
        "q", cwd=None, mem=StubMemory(),
        cfg=SimpleNamespace(state_dir=tmp_path), debug=False,
        micro_embedder=micro,
    )
    context = json.loads(result)["hookSpecificOutput"]["additionalContext"]
    assert context.index("Beta") < context.index("Alpha")  # rescored + resorted
    # the shared frozen hits keep their original (None-equivalent 0.0) score
    assert a.score == 0.0 and b.score == 0.0
    # original list order is untouched (we sorted a NEW list)
    assert [h.id for h in candidates] == ["aaaa0001", "bbbb0001"]


def test_project_tag_failure_is_logged_not_silent(monkeypatch, tmp_path, caplog) -> None:
    """A failing project_tag resolution must be swallowed but observable."""
    import memo.recall_logic as rl

    def _boom(cwd):
        raise RuntimeError("project resolution blew up")

    monkeypatch.setattr("memo.project.current_project_tag", _boom)
    monkeypatch.setenv("MEMO_RECALL_MIN_SIM", "0.0")
    monkeypatch.setenv("MEMO_RECALL_MIN_BODY_CHARS", "0")

    class StubMemory:
        def search(self, query, limit, mode, recency=False, exclude_types=None):
            return [_rec("ok000001", "Surfaced", 0.9)]

    with caplog.at_level("DEBUG", logger=rl._logger.name):
        result, _log = _recall_logic(
            "q", cwd=str(tmp_path), mem=StubMemory(),
            cfg=SimpleNamespace(state_dir=tmp_path), debug=False,
        )
    # control flow preserved: recall still succeeds
    assert "Surfaced" in json.loads(result)["hookSpecificOutput"]["additionalContext"]
    # but the failure is now observable
    assert any("project_tag resolution failed" in r.message for r in caplog.records)
