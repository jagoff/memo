"""Tests for the HyPE question-space index (flags + HypeStore)."""
from __future__ import annotations

from pathlib import Path

import pytest

from memo.store.hype_store import HypeStore

DIMS = 4


def test_hype_flags_registered_defaults():
    from memo.flags import REGISTRY

    assert REGISTRY["MEMO_HYPE_ENABLED"].default is False
    assert REGISTRY["MEMO_DREAM_HYPE_ENABLED"].default is False
    assert REGISTRY["MEMO_HYPE_QUESTIONS_PER_MEMORY"].default == 3
    assert REGISTRY["MEMO_HYPE_NIGHT_CAP"].default == 400
    assert REGISTRY["MEMO_HYPE_FOLD_POOL"].default == 30


@pytest.fixture
def store(tmp_path: Path) -> HypeStore:
    hype_store = HypeStore(tmp_path / "test.db", dims=DIMS)
    yield hype_store
    hype_store.close()


def test_replace_for_memory_inserts_and_reinsert_swaps_rows(store: HypeStore):
    inserted = store.replace_for_memory(
        "mem-1",
        "hash-a",
        "helper-model",
        [
            ("What is X?", [1.0, 0.0, 0.0, 0.0]),
            ("How does X work?", [0.0, 1.0, 0.0, 0.0]),
        ],
    )
    assert inserted == 2
    assert store.stats() == {"memories": 1, "questions": 2}

    # Re-replace with a new question set (e.g. after a body edit) swaps rows —
    # old questions must be gone, not merely appended to.
    inserted_again = store.replace_for_memory(
        "mem-1",
        "hash-b",
        "helper-model",
        [("What is Y?", [0.0, 0.0, 1.0, 0.0])],
    )
    assert inserted_again == 1
    assert store.stats() == {"memories": 1, "questions": 1}

    rows = store._conn.execute(
        "SELECT question FROM hype_questions WHERE memory_id = ?", ("mem-1",)
    ).fetchall()
    questions = {r["question"] for r in rows}
    assert questions == {"What is Y?"}
    assert store.body_hash_for("mem-1") == "hash-b"


def test_body_hash_for_unknown_memory_returns_none(store: HypeStore):
    assert store.body_hash_for("does-not-exist") is None


def test_body_hash_for_returns_stored_hash(store: HypeStore):
    store.replace_for_memory(
        "mem-1", "hash-a", "helper-model", [("What is X?", [1.0, 0.0, 0.0, 0.0])]
    )
    assert store.body_hash_for("mem-1") == "hash-a"


def test_knn_returns_best_question_per_memory_sorted_by_score(store: HypeStore):
    # mem-1's question is exactly aligned with the query vector -> distance 0,
    # score 1.0. mem-2's question is orthogonal -> cosine distance 1.0, score 0.0.
    store.replace_for_memory(
        "mem-1", "hash-a", "helper-model", [("closest question", [1.0, 0.0, 0.0, 0.0])]
    )
    store.replace_for_memory(
        "mem-2", "hash-b", "helper-model", [("farther question", [0.0, 1.0, 0.0, 0.0])]
    )

    results = store.knn([1.0, 0.0, 0.0, 0.0], k=10)

    assert [r["memory_id"] for r in results] == ["mem-1", "mem-2"]
    assert results[0]["question"] == "closest question"
    assert results[0]["score"] == pytest.approx(1.0)
    assert results[1]["score"] == pytest.approx(0.0)


def test_knn_picks_best_of_multiple_questions_per_memory(store: HypeStore):
    store.replace_for_memory(
        "mem-1",
        "hash-a",
        "helper-model",
        [
            ("far question", [0.0, 1.0, 0.0, 0.0]),
            ("near question", [1.0, 0.0, 0.0, 0.0]),
        ],
    )

    results = store.knn([1.0, 0.0, 0.0, 0.0], k=10)

    assert len(results) == 1
    assert results[0]["memory_id"] == "mem-1"
    assert results[0]["question"] == "near question"
    assert results[0]["score"] == pytest.approx(1.0)


def test_prune_orphans_removes_vec_and_text_rows(store: HypeStore):
    store.replace_for_memory(
        "mem-1", "hash-a", "helper-model", [("question one", [1.0, 0.0, 0.0, 0.0])]
    )
    store.replace_for_memory(
        "mem-2", "hash-b", "helper-model", [("question two", [0.0, 1.0, 0.0, 0.0])]
    )

    removed = store.prune_orphans({"mem-1"})

    assert removed == 1
    assert store.stats() == {"memories": 1, "questions": 1}
    assert store.body_hash_for("mem-2") is None
    remaining_vec = store._conn.execute("SELECT COUNT(*) AS n FROM hype_vec").fetchone()
    assert remaining_vec["n"] == 1


def test_stats_counts_memories_and_questions(store: HypeStore):
    assert store.stats() == {"memories": 0, "questions": 0}

    store.replace_for_memory(
        "mem-1",
        "hash-a",
        "helper-model",
        [
            ("question one", [1.0, 0.0, 0.0, 0.0]),
            ("question two", [0.0, 1.0, 0.0, 0.0]),
        ],
    )
    store.replace_for_memory(
        "mem-2", "hash-b", "helper-model", [("question three", [0.0, 0.0, 1.0, 0.0])]
    )

    assert store.stats() == {"memories": 2, "questions": 3}
