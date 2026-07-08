"""Recall tiering — reference-tier predicate, exclude_types filtering, retier.

Covers the relevance fix that keeps bulk-ingested vault material (the
`reference` tier) out of the auto-recall hook so durable knowledge isn't
drowned. See `memo.tiers`, `VecStore`, and the recall hook.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from memo.cli import cli
from memo.store import VecStore
from memo.tiers import DURABLE_TYPES, REFERENCE_TYPES, is_reference_candidate


@pytest.fixture
def store(tmp_path):
    s = VecStore(tmp_path / "test.db", dims=4)
    yield s
    s.close()


def _emb(*vals: float) -> list[float]:
    v = list(vals) + [0.0] * 4
    return v[:4]


def _add(store, id_, title, type_="note", tags=None, path=None, emb=None):
    store.upsert(
        id_=id_,
        path=path if path is not None else f"{id_}.md",
        title=title,
        type_=type_,
        tags=tags or [],
        created="2026-01-01T00:00:00Z",
        updated="2026-01-01T00:00:00Z",
        body_hash="h",
        embedding=emb or _emb(1.0),
        body_text=title + " body",
    )


# --- predicate ---------------------------------------------------------------


def test_vault_path_is_reference():
    assert is_reference_candidate("notes/04-Archive/moka.md", [], "MoKa notes")
    assert is_reference_candidate("work/01-Projects/x.md", ["team"], "Plan")


def test_chunk_tag_is_reference():
    assert is_reference_candidate(None, ["pdf", "chunk"], "Doc")


def test_chunk_marker_title_is_reference():
    assert is_reference_candidate(None, [], "MoKa Notas (§54/130)")


def test_hand_saved_note_stays_durable():
    # memo's own saves live under date-shaped paths, no vault prefix / chunk.
    assert not is_reference_candidate("2026/05/my-decision.md", ["proj"], "A decision")
    assert not is_reference_candidate(None, [], "Plain hand note")


def test_retier_dry_run_closes_memory() -> None:
    mock_memory = MagicMock()
    mock_memory.store.list_recent.return_value = []

    with patch("memo.memory.Memory", return_value=mock_memory):
        result = CliRunner().invoke(cli, ["retier"], env={"MEMO_NONINTERACTIVE": "1"})

    assert result.exit_code == 0, result.output
    mock_memory.close.assert_called_once_with()


def test_tiers_are_disjoint():
    assert "reference" in REFERENCE_TYPES
    assert "reference" not in DURABLE_TYPES
    assert {"decision", "fact", "preference", "bug", "note"} <= DURABLE_TYPES


# --- exclude_types in every search path --------------------------------------


def test_exclude_types_vec_search(store):
    _add(store, "d1", "Decision", type_="decision")
    _add(store, "r1", "Ref chunk", type_="reference")
    ids = {r["id"] for r in store.search(_emb(1.0), limit=10, exclude_types={"reference"})}
    assert ids == {"d1"}


def test_exclude_types_bm25_search(store):
    _add(store, "d1", "alpha", type_="decision")
    _add(store, "r1", "alpha", type_="reference")
    ids = {r["id"] for r in store.search_bm25("alpha", limit=10, exclude_types={"reference"})}
    assert ids == {"d1"}


def test_exclude_types_list_recent(store):
    _add(store, "d1", "Decision", type_="decision")
    _add(store, "r1", "Ref", type_="reference")
    ids = {r["id"] for r in store.list_recent(limit=10, exclude_types={"reference"})}
    assert ids == {"d1"}


def test_no_exclude_returns_all(store):
    _add(store, "d1", "Decision", type_="decision")
    _add(store, "r1", "Ref", type_="reference")
    ids = {r["id"] for r in store.list_recent(limit=10)}
    assert ids == {"d1", "r1"}


# --- retier migration --------------------------------------------------------


def test_bulk_update_type_retiers_notes(store):
    _add(store, "n1", "MoKa (§1/9)", type_="note", path="notes/04-Archive/m.md", tags=["chunk"])
    _add(store, "h1", "Hand note", type_="note", path="2026/05/h.md", tags=["proj"])
    _add(store, "d1", "Decision", type_="decision")
    rows = store.list_recent(limit=50)
    to_move = [
        r["id"]
        for r in rows
        if r["type"] == "note" and is_reference_candidate(r["path"], r["tags"], r["title"])
    ]
    assert to_move == ["n1"]
    assert store.bulk_update_type(to_move, "reference") == 1
    # n1 now excluded from durable recall; h1 + d1 remain.
    durable = {r["id"] for r in store.list_recent(limit=50, exclude_types={"reference"})}
    assert durable == {"h1", "d1"}


def test_bulk_update_type_empty_is_noop(store):
    assert store.bulk_update_type([], "reference") == 0


# --- contextual learning must not amplify the reference tier ------------------


def test_record_feedback_ignores_reference_tier(tmp_path):
    from memo.contextual import ContextStore

    cs = ContextStore(tmp_path)
    cs.record_feedback("m1", "reference", [])
    assert "reference" not in cs.get_preferences().preferred_types
    cs.record_feedback("m2", "decision", [])
    assert "decision" in cs.get_preferences().preferred_types


# --- recall health metric (is memo consulted + returning confident hits) -----


def test_recall_health_summarises_log(tmp_path):
    from memo.dashboard import append_recall_log, recall_health

    append_recall_log(
        tmp_path,
        prompt="a useful question here",
        hits=[{"id": "1", "score": 0.9, "title": "T"}],
        mode="vec",
        latency_ms=120,
        via="daemon",
    )
    append_recall_log(
        tmp_path,
        prompt="another good question",
        hits=[{"id": "2", "score": 0.8, "title": "U"}],
        mode="vec",
        latency_ms=140,
        via="subprocess",
    )
    append_recall_log(tmp_path, prompt="x", hits=[], via="bail", reason="prompt too short")

    h = recall_health(tmp_path)
    assert h["sampled"] == 3
    assert h["fired"] == 2
    assert h["bailed"] == 1
    assert h["hit_rate"] == 1.0  # both fired recalls had a hit
    assert h["median_top_score"] == 0.85  # true median of [0.8, 0.9]
    assert h["p50_latency_ms"] == 130.0  # true median of [120, 140]


def test_recall_health_empty_is_safe(tmp_path):
    from memo.dashboard import recall_health

    h = recall_health(tmp_path)
    assert h["sampled"] == 0
    assert h["hit_rate"] is None


def test_procedural_types_are_durable():
    assert {"procedure", "failure_pattern"} <= DURABLE_TYPES


def test_save_accepts_procedural_types(mem_with_stub):
    rec = mem_with_stub.save(
        content="To rebuild the index from disk, run `memo reindex --rebuild` — never rm memvec.db.",
        title="Rebuild index procedure",
        type_="procedure",
    )
    assert rec.type == "procedure"
    rec2 = mem_with_stub.save(
        content=(
            "Pattern: git add -A in the shared worktree\n"
            "Context: concurrent agent sessions share one working tree\n"
            "Wrong: stage everything with -A\n"
            "Right: stage explicit paths only"
        ),
        title="Shared worktree staging failure",
        type_="failure_pattern",
    )
    assert rec2.type == "failure_pattern"
