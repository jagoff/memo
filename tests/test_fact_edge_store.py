from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

from click.testing import CliRunner

from memo.cli import cli
from memo.fact_extraction import fact_edges_from_metadata
from memo.store.fact_edge_store import FactEdgeStore


def test_fact_edge_store_upserts_and_queries_live_facts(tmp_path) -> None:
    store = FactEdgeStore(tmp_path / "facts.db")
    try:
        fact_id = store.upsert_fact(
            subject="memo",
            predicate="uses",
            object="sqlite",
            source_record_id="rec-1",
            valid_at="2026-01-01T00:00:00+00:00",
            confidence=0.8,
            provenance={"extractor": "test"},
        )

        rows = store.query(subject="memo", as_of="2026-06-01T00:00:00+00:00")

        assert [r["id"] for r in rows] == [fact_id]
        assert rows[0]["subject"] == "memo"
        assert rows[0]["predicate"] == "uses"
        assert rows[0]["object"] == "sqlite"
        assert rows[0]["source_record_id"] == "rec-1"
        assert rows[0]["valid_at"] == "2026-01-01T00:00:00+00:00"
        assert rows[0]["invalid_at"] is None
        assert rows[0]["confidence"] == 0.8
        assert rows[0]["provenance"] == {"extractor": "test"}
    finally:
        store.close()


def test_fact_edge_store_excludes_future_invalid_and_expired_edges(tmp_path) -> None:
    store = FactEdgeStore(tmp_path / "facts.db")
    try:
        live = store.upsert_fact(
            subject="agent",
            predicate="prefers",
            object="visible-context",
            valid_at="2026-01-01T00:00:00+00:00",
        )
        future = store.upsert_fact(
            subject="agent",
            predicate="prefers",
            object="temporal-store",
            valid_at="2026-08-01T00:00:00+00:00",
        )
        invalid = store.upsert_fact(
            subject="agent",
            predicate="prefers",
            object="old-context",
            valid_at="2026-01-01T00:00:00+00:00",
            invalid_at="2026-03-01T00:00:00+00:00",
        )
        expired = store.upsert_fact(
            subject="agent",
            predicate="prefers",
            object="temporary-context",
            valid_at="2026-01-01T00:00:00+00:00",
            expired_at="2026-02-01T00:00:00+00:00",
        )

        rows = store.query(subject="agent", as_of="2026-04-01T00:00:00+00:00")

        assert [r["id"] for r in rows] == [live]
        assert future not in [r["id"] for r in rows]
        assert invalid not in [r["id"] for r in rows]
        assert expired not in [r["id"] for r in rows]
    finally:
        store.close()


def test_fact_edge_store_invalidates_superseded_fact(tmp_path) -> None:
    store = FactEdgeStore(tmp_path / "facts.db")
    try:
        old_id = store.upsert_fact(
            subject="memo",
            predicate="backend",
            object="old",
            valid_at="2026-01-01T00:00:00+00:00",
        )
        new_id = store.upsert_fact(
            subject="memo",
            predicate="backend",
            object="new",
            valid_at="2026-06-01T00:00:00+00:00",
            supersedes=[old_id],
        )

        old_rows = store.query(subject="memo", as_of="2026-05-01T00:00:00+00:00")
        new_rows = store.query(subject="memo", as_of="2026-07-01T00:00:00+00:00")

        assert [r["id"] for r in old_rows] == [old_id]
        assert [r["id"] for r in new_rows] == [new_id]
        archived = store.get(old_id)
        assert archived is not None
        assert archived["invalid_at"] == "2026-06-01T00:00:00+00:00"
    finally:
        store.close()


def test_fact_edge_store_normalizes_naive_datetimes_to_utc(tmp_path) -> None:
    store = FactEdgeStore(tmp_path / "facts.db")
    try:
        fact_id = store.upsert_fact(
            subject="memo",
            predicate="shipped",
            object="visible-surface",
            valid_at=datetime(2026, 7, 10, 12, 0, 0),
        )

        row = store.get(fact_id)

        assert row is not None
        assert row["valid_at"] == datetime(2026, 7, 10, 12, 0, 0, tzinfo=UTC).isoformat()
    finally:
        store.close()


def test_memory_exposes_fact_edge_store(mock_memory) -> None:
    fact_id = mock_memory.fact_edges.upsert_fact(
        subject="memo",
        predicate="phase",
        object="deep-foundation",
        valid_at="2026-07-10T00:00:00+00:00",
    )

    rows = mock_memory.fact_edges.query(subject="memo", as_of="2026-07-11T00:00:00+00:00")

    assert [r["id"] for r in rows] == [fact_id]


def test_save_declared_fact_edges_populates_fact_store(mock_memory) -> None:
    rec = mock_memory.save(
        content="memo uses sqlite for local temporal facts",
        title="Memo fact edge",
        type_="note",
        created="2026-01-01T00:00:00+00:00",
        extra={
            "fact_edges": [
                {
                    "subject": "memo",
                    "predicate": "uses",
                    "object": "sqlite",
                    "valid_at": "2026-01-01T00:00:00+00:00",
                    "confidence": 0.9,
                }
            ]
        },
    )

    rows = mock_memory.fact_edges.query(subject="memo", as_of="2026-02-01T00:00:00+00:00")

    assert len(rows) == 1
    assert rows[0]["source_record_id"] == rec.id
    assert rows[0]["predicate"] == "uses"
    assert rows[0]["object"] == "sqlite"
    assert rows[0]["confidence"] == 0.9


def test_fact_type_memory_gets_baseline_assertion(mock_memory) -> None:
    rec = mock_memory.save(
        content="The temporal store is sqlite-backed.",
        title="Temporal store backend",
        type_="fact",
        created="2026-01-01T00:00:00+00:00",
    )

    rows = mock_memory.fact_edges.query(
        subject="memory",
        predicate="asserts",
        as_of="2026-02-01T00:00:00+00:00",
    )

    assert len(rows) == 1
    assert rows[0]["source_record_id"] == rec.id
    assert rows[0]["object"] == "Temporal store backend"


def test_fact_edge_metadata_normalizes_confidence_and_supersedes() -> None:
    edges = fact_edges_from_metadata(
        record_id="rec-1",
        title="Fact",
        type_="note",
        created="2026-01-01T00:00:00+00:00",
        updated="2026-01-02T00:00:00+00:00",
        extra={
            "fact_edges": [
                {
                    "subject": "memo",
                    "predicate": "stores",
                    "object": "facts",
                    "confidence": "not-a-number",
                    "supersedes": "old-edge",
                }
            ]
        },
    )

    assert edges[0]["confidence"] == 1.0
    assert edges[0]["supersedes"] == ["old-edge"]


def test_cli_temporal_facts_add_and_list(tmp_path, monkeypatch) -> None:
    env = {
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_DATA_DIR": str(tmp_path / "data"),
        "MEMO_STATE_DIR": str(tmp_path / "state"),
        "MEMO_VAULT_PATH": str(tmp_path / "vault"),
        "MEMO_EMBEDDER_DIMS": "4",
        "MEMO_EMBEDDER_VIA_DAEMON": "0",
        "MEMO_SKIP_MODEL_VERSION_CHECK": "1",
    }
    monkeypatch.setattr(
        "memo.embedder.MLXEmbedder.embed",
        lambda self, inputs: [[1.0, 0.0, 0.0, 0.0] for _ in inputs],
    )
    runner = CliRunner()

    add = runner.invoke(
        cli,
        [
            "temporal",
            "facts",
            "add",
            "memo",
            "backend",
            "sqlite",
            "--valid-at",
            "2026-01-01T00:00:00Z",
            "--json",
        ],
        env=env,
    )
    listed = runner.invoke(
        cli,
        [
            "temporal",
            "facts",
            "list",
            "--subject",
            "memo",
            "--as-of",
            "2026-02-01T00:00:00Z",
            "--json",
        ],
        env=env,
    )

    assert add.exit_code == 0, add.output
    assert listed.exit_code == 0, listed.output
    assert '"predicate": "backend"' in listed.output
    assert '"object": "sqlite"' in listed.output


def test_mcp_fact_edge_tools_round_trip(mock_memory) -> None:
    from memo.server_temporal import register

    server = MagicMock()
    tools: dict = {}

    def tool_decorator():
        def wrapper(fn):
            tools[fn.__name__] = fn
            return fn

        return wrapper

    server.tool = tool_decorator
    register(server, mock_memory)

    saved = tools["memo_fact_edge_save"](
        subject="memo",
        predicate="foundation",
        object="temporal-facts",
        valid_at="2026-07-10T00:00:00+00:00",
    )
    rows = tools["memo_fact_edges"](subject="memo", as_of="2026-07-11T00:00:00+00:00")
    invalidated = tools["memo_fact_edge_invalidate"](
        id=saved["id"],
        invalid_at="2026-07-12T00:00:00+00:00",
    )

    assert [r["id"] for r in rows] == [saved["id"]]
    assert invalidated == {"id": saved["id"], "invalidated": True}
