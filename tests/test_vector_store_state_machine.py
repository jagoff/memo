from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pytest
from hypothesis import HealthCheck, settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, rule

from memo.store import VecStore

pytestmark = [pytest.mark.db_contract, pytest.mark.resource_hygiene]


@dataclass
class ModelRow:
    slot: int
    version: int
    deleted: bool = False


def _rank_value(slot: int, version: int) -> float:
    return (slot * 10 + version + 1) / 100


def _body_token(slot: int, version: int) -> str:
    return f"slot{slot}version{version}"


def _write(store: VecStore, slot: int, version: int) -> None:
    memory_id = f"memory-{slot}"
    embedding = [1.0, _rank_value(slot, version), 0.0, 0.0]
    store.upsert(
        id_=memory_id,
        path=f"memory/{memory_id}.md",
        title=f"Memory {slot} v{version}",
        type_="decision" if version % 2 else "note",
        tags=[
            f"slot-{slot}",
            f"version-{version}",
            f"slot-{(slot + 1) % 6}-suffix",
        ],
        created="2026-01-01T00:00:00+00:00",
        updated=f"2026-01-{version + 1:02d}T00:00:00+00:00",
        body_hash=f"hash-{slot}-{version}",
        embedding=embedding,
        body_text=f"body {_body_token(slot, version)}",
    )


def _derived_row_counts(store: VecStore) -> dict[str, int]:
    return {
        "meta": store.connection.execute("SELECT COUNT(*) FROM meta").fetchone()[0],
        "vec": store.connection.execute("SELECT COUNT(*) FROM vec").fetchone()[0],
        "fts": store.connection.execute("SELECT COUNT(*) FROM fts").fetchone()[0],
    }


def _signal_payload_for_ids(
    store: VecStore,
    memory_ids: set[str],
) -> dict[str, list[tuple[object, ...]]]:
    access = [
        (row["id"], int(row["access_count"]), row["last_accessed"])
        for row in store.connection.execute(
            "SELECT id, access_count, last_accessed FROM access ORDER BY id"
        ).fetchall()
        if row["id"] in memory_ids
    ]
    health = [
        (
            row["id"],
            float(row["confidence"]),
            float(row["roi_score"]),
            row["updated_at"],
            int(row["support_count"]),
        )
        for row in store.connection.execute(
            "SELECT id, confidence, roi_score, updated_at, support_count "
            "FROM memory_health ORDER BY id"
        ).fetchall()
        if row["id"] in memory_ids
    ]
    feedback = [
        (
            row["id"],
            row["source_id"],
            row["query_text"],
            int(row["rating"]),
            row["created_at"],
            row["extra_json"],
        )
        for row in store.connection.execute(
            "SELECT id, source_id, query_text, rating, created_at, extra_json "
            "FROM source_feedback ORDER BY id"
        ).fetchall()
        if row["source_id"] in memory_ids
    ]
    feedback_vectors = [
        (
            row["feedback_id"],
            row["source_id"],
            bytes(row["query_emb"]),
        )
        for row in store.connection.execute(
            "SELECT feedback_id, source_id, query_emb FROM source_feedback_vec ORDER BY feedback_id"
        ).fetchall()
        if row["source_id"] in memory_ids
    ]
    return {
        "access": access,
        "memory_health": health,
        "source_feedback": feedback,
        "source_feedback_vec": feedback_vectors,
    }


def _assert_store_matches_model(
    store: VecStore,
    model: dict[str, ModelRow],
    purged: set[str],
) -> None:
    active = {memory_id for memory_id, row in model.items() if not row.deleted}
    deleted = {memory_id for memory_id, row in model.items() if row.deleted}
    assert store.count() == len(active)
    assert set(store.list_soft_deleted()) == deleted

    for memory_id, expected in model.items():
        row = store.get(memory_id)
        vector_count = store.connection.execute(
            "SELECT COUNT(*) FROM vec WHERE id = ?", (memory_id,)
        ).fetchone()[0]
        fts_count = store.connection.execute(
            "SELECT COUNT(*) FROM fts WHERE id = ?", (memory_id,)
        ).fetchone()[0]
        if expected.deleted:
            assert row is None
            assert vector_count == 0, "vector row count for tombstone"
            assert fts_count == 0, "FTS row count for tombstone"
            assert store.search_bm25(_body_token(expected.slot, expected.version), limit=20) == []
            continue

        assert row is not None
        assert row["title"] == f"Memory {expected.slot} v{expected.version}"
        assert row["type"] == ("decision" if expected.version % 2 else "note")
        assert row["tags"] == [
            f"slot-{expected.slot}",
            f"version-{expected.version}",
            f"slot-{(expected.slot + 1) % 6}-suffix",
        ]
        assert vector_count == 1, "vector row count for active memory"
        assert fts_count == 1, "FTS row count for active memory"
        body = store.get_fts_body_by_path(f"memory/{memory_id}.md")
        assert body == f"body {_body_token(expected.slot, expected.version)}"
        assert [
            hit["id"]
            for hit in store.search_bm25(_body_token(expected.slot, expected.version), limit=20)
        ] == [memory_id]
        for old_version in range(9):
            if old_version != expected.version:
                assert store.search_bm25(_body_token(expected.slot, old_version), limit=20) == []

    expected_vector_order = [
        memory_id
        for memory_id, row in sorted(
            ((memory_id, row) for memory_id, row in model.items() if not row.deleted),
            key=lambda item: _rank_value(item[1].slot, item[1].version),
        )
    ]
    vector_ids = [hit["id"] for hit in store.search([1.0, 0.0, 0.0, 0.0], limit=20)]
    assert vector_ids == expected_vector_order

    for type_ in ("decision", "note"):
        expected_type_ids = {
            memory_id
            for memory_id, row in model.items()
            if not row.deleted and ("decision" if row.version % 2 else "note") == type_
        }
        assert {
            hit["id"] for hit in store.search([1.0, 0.0, 0.0, 0.0], limit=20, type_=type_)
        } == expected_type_ids

    for slot in range(6):
        expected_tag_ids = {
            memory_id for memory_id, row in model.items() if not row.deleted and row.slot == slot
        }
        decoy_only_ids = {
            memory_id
            for memory_id, row in model.items()
            if not row.deleted and (row.slot + 1) % 6 == slot
        }
        actual_tag_ids = {row["id"] for row in store.list_by_tag(f"slot-{slot}")}
        assert actual_tag_ids == expected_tag_ids
        assert actual_tag_ids.isdisjoint(decoy_only_ids)

    for memory_id in purged:
        for table, column in (
            ("meta", "id"),
            ("vec", "id"),
            ("fts", "id"),
            ("access", "id"),
            ("memory_health", "id"),
            ("source_feedback", "source_id"),
            ("source_feedback_vec", "source_id"),
        ):
            count = store.connection.execute(
                f"SELECT COUNT(*) FROM {table} WHERE {column} = ?",  # noqa: S608
                (memory_id,),
            ).fetchone()[0]
            assert count == 0, f"{table} rows for hard-deleted memory"


def test_model_oracle_detects_a_missing_vector(tmp_path: Path) -> None:
    with patch.dict(
        os.environ,
        {"MEMO_SOFT_DELETE": "1", "MEMO_TANTIVY_ENABLED": "0"},
    ):
        assert os.environ["MEMO_TANTIVY_ENABLED"] == "0"
        assert os.environ["MEMO_SOFT_DELETE"] == "1"
        store = VecStore(tmp_path / "vectors.db", dims=4, vec_quant="off")
        try:
            _write(store, 1, 1)
            store.connection.execute("DELETE FROM vec WHERE id = ?", ("memory-1",))
            store.connection.commit()
            with pytest.raises(AssertionError, match="vector row count"):
                _assert_store_matches_model(
                    store,
                    {"memory-1": ModelRow(slot=1, version=1)},
                    purged=set(),
                )
        finally:
            store.close()


class VecStoreStateMachine(RuleBasedStateMachine):
    def __init__(self) -> None:
        super().__init__()
        self._tmp = TemporaryDirectory(prefix="memo-vec-state-")
        self._env = patch.dict(
            os.environ,
            {"MEMO_SOFT_DELETE": "1", "MEMO_TANTIVY_ENABLED": "0"},
        )
        self._env.start()
        self._db_path = Path(self._tmp.name) / "vectors.db"
        self.store = VecStore(self._db_path, dims=4, vec_quant="off")
        self.model: dict[str, ModelRow] = {}
        self.purged: set[str] = set()

    @rule(slot=st.integers(min_value=0, max_value=5), version=st.integers(0, 8))
    def upsert(self, slot: int, version: int) -> None:
        _write(self.store, slot, version)
        memory_id = f"memory-{slot}"
        self.model[memory_id] = ModelRow(slot=slot, version=version)
        self.purged.discard(memory_id)

    @rule(slot=st.integers(min_value=0, max_value=5))
    def soft_delete(self, slot: int) -> None:
        memory_id = f"memory-{slot}"
        existed = memory_id in self.model
        assert self.store.delete(memory_id) is existed
        if existed:
            self.model[memory_id].deleted = True

    @rule(slot=st.integers(min_value=0, max_value=5))
    def restore_by_upsert(self, slot: int) -> None:
        memory_id = f"memory-{slot}"
        current = self.model.get(memory_id)
        if current is None or not current.deleted:
            return
        version = (current.version + 1) % 9
        _write(self.store, slot, version)
        self.model[memory_id] = ModelRow(slot=slot, version=version)
        self.purged.discard(memory_id)

    @rule(slot=st.integers(min_value=0, max_value=5))
    def hard_delete(self, slot: int) -> None:
        memory_id = f"memory-{slot}"
        existed = memory_id in self.model
        if existed:
            self.store.touch([memory_id], ts="2026-02-01T00:00:00+00:00")
            self.store.set_confidence_batch([(memory_id, 0.42)])
            self.store.record_source_feedback(
                source_id=memory_id,
                query_text=f"feedback for {memory_id}",
                query_emb=[0.0, 1.0, 0.0, 0.0],
                rating=1,
                feedback_id=f"feedback-{memory_id}",
            )
        assert self.store.hard_delete(memory_id) is existed
        self.model.pop(memory_id, None)
        self.purged.add(memory_id)

    @rule()
    def reopen(self) -> None:
        self.store.close()
        self.store = VecStore(self._db_path, dims=4, vec_quant="off")

    @rule()
    def clear_derived_index(self) -> None:
        memory_ids = set(self.model)
        feedback_ids: set[str] = set()
        for memory_id, row in sorted(self.model.items()):
            self.store.touch(
                [memory_id],
                ts=f"2026-03-{row.slot + 1:02d}T00:00:00+00:00",
            )
            self.store.set_confidence_batch(
                [(memory_id, 0.4 + row.slot / 100 + row.version / 1000)]
            )
            expected_feedback_id = f"clear-feedback-{memory_id}-v{row.version}"
            feedback_id = self.store.record_source_feedback(
                source_id=memory_id,
                query_text=f"clear feedback {memory_id} v{row.version}",
                query_emb=[0.0, 1.0, 0.0, 0.0],
                rating=1 if row.version % 2 else -1,
                feedback_id=expected_feedback_id,
                extra={"phase": "clear", "slot": row.slot, "version": row.version},
            )
            assert feedback_id == expected_feedback_id
            feedback_ids.add(feedback_id)

        signal_before = _signal_payload_for_ids(self.store, memory_ids)
        assert {row[0] for row in signal_before["access"]} == memory_ids
        assert {row[0] for row in signal_before["memory_health"]} == memory_ids
        assert {row[1] for row in signal_before["source_feedback"]} == memory_ids
        assert {row[1] for row in signal_before["source_feedback_vec"]} == memory_ids
        assert feedback_ids <= {row[0] for row in signal_before["source_feedback"]}
        assert feedback_ids <= {row[0] for row in signal_before["source_feedback_vec"]}

        assert self.store.clear_memory_index() == len(self.model)
        assert _derived_row_counts(self.store) == {"meta": 0, "vec": 0, "fts": 0}
        assert _signal_payload_for_ids(self.store, memory_ids) == signal_before

        self.store.close()
        self.store = VecStore(self._db_path, dims=4, vec_quant="off")
        assert _derived_row_counts(self.store) == {"meta": 0, "vec": 0, "fts": 0}
        assert _signal_payload_for_ids(self.store, memory_ids) == signal_before

        self.model.clear()

    @invariant()
    def database_matches_reference_model(self) -> None:
        _assert_store_matches_model(self.store, self.model, self.purged)

    def teardown(self) -> None:
        self.store.close()
        self._env.stop()
        self._tmp.cleanup()


def test_clear_rule_requires_seeded_signal_survival() -> None:
    machine = VecStoreStateMachine()
    try:
        machine.upsert(slot=0, version=0)
        machine.clear_derived_index()
    finally:
        machine.teardown()


settings.register_profile(
    "memo_pr",
    max_examples=25,
    stateful_step_count=30,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.register_profile(
    "ci_extended",
    max_examples=100,
    stateful_step_count=75,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "memo_pr"))

TestVecStoreStateMachine = VecStoreStateMachine.TestCase
