"""Safety and failure-isolation contracts for destructive housekeeping."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from memo.cli import cli
from memo.session import prune_lru
from memo.sqlite_snapshot import snapshot_sqlite_database
from memo.store import VecStore

pytestmark = [pytest.mark.db_contract, pytest.mark.resource_hygiene]


def _env(tmp_path: Path) -> dict[str, str]:
    return {
        "MEMO_CONFIG_FILE": str(tmp_path / "missing-config.toml"),
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_DATA_DIR": str(tmp_path / "data"),
        "MEMO_STATE_DIR": str(tmp_path / "state"),
        "MEMO_OUTCOME_RANKING_ENABLED": "0",
        "MEMO_SYNTHESIS_ENABLED": "0",
        "MEMO_MAINT_SYNTHESIZE": "0",
        "MEMO_VERIFICATION_STATE_TRACKING": "0",
    }


def _vacuum_args(*extra: str) -> list[str]:
    return [
        "maintain",
        "--vacuum",
        "--skip-contradict",
        "--skip-consolidate",
        "--skip-stale",
        "--skip-synthesize",
        "--json",
        *extra,
    ]


def _vacuum_memory(ids: list[str]) -> MagicMock:
    mem = MagicMock()
    mem.lifecycle.enforce_forget_ttl.return_value = []
    mem.store.list_soft_deleted.return_value = ids
    return mem


def test_vacuum_dry_run_reports_candidates_without_deleting(tmp_path: Path) -> None:
    mem = _vacuum_memory(["old-a", "old-b"])

    with patch("memo.cli_maintain._get_memory", return_value=mem):
        result = CliRunner().invoke(
            cli,
            _vacuum_args("--vacuum-days", "30", "--dry-run"),
            env=_env(tmp_path),
        )

    assert result.exit_code == 0, result.output
    receipt = json.loads(result.output)
    assert receipt["vacuumed"] == 2
    mem.store.hard_delete_if_soft_deleted_before.assert_not_called()
    cutoff = mem.store.list_soft_deleted.call_args.kwargs["before"]
    assert isinstance(cutoff, str) and cutoff.endswith("+00:00")


def test_vacuum_continues_after_one_record_fails_and_counts_successes(
    tmp_path: Path,
) -> None:
    """One corrupt tombstone must not block cleanup of every later row."""
    mem = _vacuum_memory(["broken", "deleted", "already-gone"])

    def hard_delete(id_: str, *, before: str) -> bool:
        assert before.endswith("+00:00")
        if id_ == "broken":
            raise OSError("database page unavailable")
        return id_ == "deleted"

    mem.store.hard_delete_if_soft_deleted_before.side_effect = hard_delete

    with patch("memo.cli_maintain._get_memory", return_value=mem):
        result = CliRunner().invoke(
            cli,
            _vacuum_args("--vacuum-days", "90"),
            env=_env(tmp_path),
        )

    # Failure isolation is the contract: every later row is still processed and
    # the receipt is complete. Since the P1 audit the exit code reports that a
    # row failed (see receipt["errors"] assertions below).
    assert result.exit_code == 1, result.output
    receipt = json.loads(result.output)
    assert receipt["vacuumed"] == 1
    assert [
        call.args[0] for call in mem.store.hard_delete_if_soft_deleted_before.call_args_list
    ] == [
        "broken",
        "deleted",
        "already-gone",
    ]
    assert any("vacuum broken: OSError: database page unavailable" in e for e in receipt["errors"])
    assert any(
        "vacuum already-gone: record is no longer deleted before cutoff" in e
        for e in receipt["errors"]
    )


def test_vacuum_days_rejects_negative_retention_before_loading_memory(
    tmp_path: Path,
) -> None:
    with patch("memo.cli_maintain._get_memory") as get_memory:
        result = CliRunner().invoke(
            cli,
            _vacuum_args("--vacuum-days", "-1"),
            env=_env(tmp_path),
        )

    assert result.exit_code == 2
    assert "vacuum-days" in result.output
    get_memory.assert_not_called()


def test_vacuum_days_overflow_is_reported_without_aborting_command(tmp_path: Path) -> None:
    mem = _vacuum_memory([])

    with patch("memo.cli_maintain._get_memory", return_value=mem):
        result = CliRunner().invoke(
            cli,
            _vacuum_args("--vacuum-days", str(10**20)),
            env=_env(tmp_path),
        )

    assert result.exit_code == 0, result.output
    receipt = json.loads(result.output)
    assert receipt["vacuumed"] == 0
    assert any(error.startswith("vacuum: OverflowError:") for error in receipt["errors"])
    mem.store.list_soft_deleted.assert_not_called()


def test_conditional_vacuum_preserves_record_restored_after_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reactivated row must win a race with a stale vacuum candidate list."""

    monkeypatch.setenv("MEMO_TANTIVY_ENABLED", "0")
    monkeypatch.setenv("MEMO_SOFT_DELETE", "1")
    store = VecStore(
        tmp_path / "vectors.db",
        dims=4,
        embedder_model="tests/housekeeping-v1",
        vec_quant="off",
    )
    embedding = [1.0, 0.0, 0.0, 0.0]
    upsert = {
        "id_": "restored",
        "path": "memory/restored.md",
        "title": "Restored record",
        "type_": "note",
        "tags": ["durable"],
        "created": "2026-01-01T00:00:00+00:00",
        "updated": "2026-01-02T00:00:00+00:00",
        "body_hash": "hash-restored",
        "embedding": embedding,
        "body_text": "restored lexical marker",
        "extra": {"origin": "keep"},
    }
    cutoff = "2026-02-01T00:00:00+00:00"

    try:
        store.upsert(**upsert)
        store.touch(["restored"], ts="2026-01-03T00:00:00+00:00")
        store.set_confidence_batch([("restored", 0.35)])
        feedback_id = store.record_source_feedback(
            source_id="restored",
            query_text="restore this record",
            query_emb=[0.0, 1.0, 0.0, 0.0],
            rating=1,
            feedback_id="feedback-restored",
        )
        assert store.delete("restored") is True
        store.connection.execute(
            "UPDATE meta SET deleted_at = ? WHERE id = ?",
            ("2026-01-01T00:00:00+00:00", "restored"),
        )
        store.connection.commit()
        assert store.list_soft_deleted(before=cutoff) == ["restored"]

        # Reindex/upsert restores the row and all of its searchable indexes
        # after vacuum selected it but before vacuum attempts the hard delete.
        store.upsert(**upsert)

        assert store.hard_delete_if_soft_deleted_before("restored", before=cutoff) is False
        restored = store.get("restored")
        assert restored is not None
        assert restored["extra"] == {"origin": "keep"}
        assert store.has_vector("restored") is True
        assert [row["id"] for row in store.search(embedding, limit=5)] == ["restored"]
        assert [row["id"] for row in store.search_bm25("lexical marker", limit=5)] == ["restored"]
        assert store.get_access("restored")["access_count"] == 1
        signal = store.dump_signal()
        assert any(row["id"] == "restored" for row in signal["memory_health"])
        assert [
            row["id"]
            for row in store.find_feedback_for_source(
                "restored",
                [0.0, 1.0, 0.0, 0.0],
                threshold=0.99,
            )
        ] == [feedback_id]
    finally:
        store.close()


def test_conditional_vacuum_matches_selection_for_malformed_tombstones(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every selected tombstone must remain eligible for the atomic purge."""

    monkeypatch.setenv("MEMO_TANTIVY_ENABLED", "0")
    monkeypatch.setenv("MEMO_SOFT_DELETE", "1")
    store = VecStore(
        tmp_path / "vectors.db",
        dims=4,
        embedder_model="tests/housekeeping-v1",
        vec_quant="off",
    )
    cutoff = "2026-02-01T00:00:00+00:00"

    def upsert(id_: str) -> None:
        store.upsert(
            id_=id_,
            path=f"memory/{id_}.md",
            title=id_.replace("-", " ").title(),
            type_="note",
            tags=["vacuum"],
            created="2026-01-01T00:00:00+00:00",
            updated="2026-01-02T00:00:00+00:00",
            body_hash=f"hash-{id_}",
            embedding=[1.0, 0.0, 0.0, 0.0],
            body_text=f"{id_} lexical marker",
            extra={"origin": "housekeeping-test"},
        )

    def count(query: str, value: str) -> int:
        row = store.connection.execute(
            query,
            (value,),
        ).fetchone()
        assert row is not None
        return int(row["n"])

    try:
        for id_ in ("malformed", "valid-old", "at-cutoff"):
            upsert(id_)
            assert store.delete(id_) is True

        store.touch(["malformed"], ts="2026-01-03T00:00:00+00:00")
        store.set_confidence_batch([("malformed", 0.35)])
        store.record_source_feedback(
            source_id="malformed",
            query_text="purge malformed tombstone",
            query_emb=[0.0, 1.0, 0.0, 0.0],
            rating=-1,
            feedback_id="feedback-malformed",
        )
        store.connection.executemany(
            "UPDATE meta SET deleted_at = ? WHERE id = ?",
            [
                ("not-a-timestamp", "malformed"),
                ("2026-01-01T00:00:00+00:00", "valid-old"),
                (cutoff, "at-cutoff"),
            ],
        )
        store.connection.commit()

        assert sorted(store.list_soft_deleted(before=cutoff)) == ["malformed", "valid-old"]
        assert count("SELECT count(*) AS n FROM access WHERE id = ?", "malformed") == 1
        assert count("SELECT count(*) AS n FROM memory_health WHERE id = ?", "malformed") == 1
        assert (
            count(
                "SELECT count(*) AS n FROM source_feedback WHERE source_id = ?",
                "malformed",
            )
            == 1
        )
        assert (
            count(
                "SELECT count(*) AS n FROM source_feedback_vec WHERE source_id = ?",
                "malformed",
            )
            == 1
        )

        assert store.hard_delete_if_soft_deleted_before("malformed", before=cutoff) is True
        for query in (
            "SELECT count(*) AS n FROM meta WHERE id = ?",
            "SELECT count(*) AS n FROM vec WHERE id = ?",
            "SELECT count(*) AS n FROM fts WHERE id = ?",
            "SELECT count(*) AS n FROM access WHERE id = ?",
            "SELECT count(*) AS n FROM memory_health WHERE id = ?",
            "SELECT count(*) AS n FROM source_feedback WHERE source_id = ?",
            "SELECT count(*) AS n FROM source_feedback_vec WHERE source_id = ?",
        ):
            assert count(query, "malformed") == 0

        assert store.hard_delete_if_soft_deleted_before("valid-old", before=cutoff) is True
        assert store.hard_delete_if_soft_deleted_before("at-cutoff", before=cutoff) is False
        assert count("SELECT count(*) AS n FROM meta WHERE id = ?", "valid-old") == 0
        assert count("SELECT count(*) AS n FROM meta WHERE id = ?", "at-cutoff") == 1
        assert store.list_soft_deleted(before=cutoff) == []
    finally:
        store.close()


def test_prune_lru_rejects_negative_cap_without_deleting_sessions(tmp_path: Path) -> None:
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    paths = []
    for idx in range(3):
        path = sessions / f"session-{idx}.json"
        path.write_text(
            json.dumps(
                {
                    "session_id": f"session-{idx}",
                    "updated": f"2026-01-0{idx + 1}T00:00:00+00:00",
                }
            ),
            encoding="utf-8",
        )
        paths.append(path)

    with pytest.raises(ValueError, match="cap must be non-negative"):
        prune_lru(tmp_path, cap=-1)

    assert all(path.is_file() for path in paths)


def test_session_prune_cli_rejects_negative_cap(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        cli,
        ["session", "prune", "--cap", "-1"],
        env=_env(tmp_path),
    )

    assert result.exit_code == 2
    assert "cap" in result.output


def test_prune_lru_removes_malformed_snapshot_before_valid_sessions(tmp_path: Path) -> None:
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    (sessions / "malformed.json").write_text("{not-json", encoding="utf-8")
    for idx in range(2):
        (sessions / f"valid-{idx}.json").write_text(
            json.dumps(
                {
                    "session_id": f"valid-{idx}",
                    "updated": f"2026-02-0{idx + 1}T00:00:00+00:00",
                }
            ),
            encoding="utf-8",
        )

    assert prune_lru(tmp_path, cap=2) == 1
    assert sorted(path.name for path in sessions.glob("*.json")) == [
        "valid-0.json",
        "valid-1.json",
    ]


def test_prune_lru_counts_only_successful_unlinks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    for idx in range(3):
        (sessions / f"session-{idx}.json").write_text(
            json.dumps(
                {
                    "session_id": f"session-{idx}",
                    "updated": f"2026-03-0{idx + 1}T00:00:00+00:00",
                }
            ),
            encoding="utf-8",
        )
    real_unlink = Path.unlink

    def flaky_unlink(path: Path, *args, **kwargs) -> None:
        if path.name == "session-0.json":
            raise OSError("busy snapshot")
        real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", flaky_unlink)

    assert prune_lru(tmp_path, cap=1) == 1
    assert sorted(path.name for path in sessions.glob("*.json")) == [
        "session-0.json",
        "session-2.json",
    ]


def test_sqlite_snapshot_failure_removes_partial_output_and_scratch_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.db"
    destination = tmp_path / "snapshot.db"
    with closing(sqlite3.connect(source)) as connection:
        connection.execute("CREATE TABLE durable (value TEXT)")
        connection.execute("INSERT INTO durable VALUES ('committed')")
        connection.commit()

    def fail_sanitization(_database: Path) -> None:
        raise sqlite3.DatabaseError("integrity failure")

    monkeypatch.setattr("memo.sqlite_snapshot._sanitize_secret_store", fail_sanitization)

    with pytest.raises(sqlite3.DatabaseError, match="integrity failure"):
        snapshot_sqlite_database(source, destination)

    assert not destination.exists()
    assert list(tmp_path.glob(".sqlite-snapshot-*")) == []


def test_sqlite_snapshot_never_overwrites_existing_destination(tmp_path: Path) -> None:
    source = tmp_path / "source.db"
    destination = tmp_path / "snapshot.db"
    with closing(sqlite3.connect(source)) as connection:
        connection.execute("CREATE TABLE source_data (value TEXT)")
    sentinel = b"existing backup must survive"
    destination.write_bytes(sentinel)

    with pytest.raises(FileExistsError, match="already exists"):
        snapshot_sqlite_database(source, destination)

    assert destination.read_bytes() == sentinel
