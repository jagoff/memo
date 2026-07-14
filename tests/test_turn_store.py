"""Tests for the verbatim turn-level FTS5 index (flags + TurnStore config property)."""

from __future__ import annotations

import stat
from pathlib import Path

import pytest

from memo.config import Config
from memo.store.turn_store import TurnStore


def test_verbatim_flags_registered_defaults():
    """All three verbatim flags are registered with correct defaults."""
    from memo.flags import REGISTRY

    assert REGISTRY["MEMO_VERBATIM_INDEX"].default is False
    assert REGISTRY["MEMO_VERBATIM_MAX_DAYS"].default == 90
    assert REGISTRY["MEMO_VERBATIM_MIN_CHARS"].default == 20


def test_verbatim_db_separate_when_single_db_off(tmp_path: Path):
    """When single_db is False (default), verbatim_db is a separate file in state_dir."""
    cfg = Config(data_dir=tmp_path / "d", state_dir=tmp_path / "s")
    assert cfg.single_db is False
    assert cfg.verbatim_db == cfg.state_dir / "verbatim.db"
    assert cfg.verbatim_db != cfg.db_path


def test_verbatim_db_collapses_with_single_db_true(tmp_path: Path):
    """When single_db is True, verbatim_db collapses onto db_path."""
    cfg = Config(data_dir=tmp_path / "d", state_dir=tmp_path / "s", single_db=True)
    assert cfg.single_db is True
    assert cfg.verbatim_db == cfg.db_path


def test_verbatim_db_from_env(monkeypatch, tmp_path: Path):
    """MEMO_SINGLE_DB env var controls verbatim_db collapse."""
    monkeypatch.setenv("MEMO_CONFIG_FILE", str(tmp_path / "none.toml"))
    monkeypatch.setenv("MEMO_DATA_DIR", str(tmp_path / "d"))
    monkeypatch.setenv("MEMO_STATE_DIR", str(tmp_path / "s"))
    monkeypatch.setenv("MEMO_SINGLE_DB", "1")
    cfg = Config.from_env()
    assert cfg.single_db is True
    assert cfg.verbatim_db == cfg.db_path


# ── TurnStore (Task V2) ──────────────────────────────────────────────────


@pytest.fixture
def store(tmp_path: Path):
    ts = TurnStore(tmp_path / "verbatim.db")
    yield ts
    ts.close()


def test_replace_session_returns_count(store: TurnStore):
    turns = [
        {"idx": 0, "role": "user", "ts": "2026-07-01T10:00:00", "text": "hola"},
        {"idx": 1, "role": "assistant", "ts": "2026-07-01T10:00:05", "text": "que tal"},
    ]
    n = store.replace_session("sess-1", "claude-code", turns)
    assert n == 2
    assert store.stats() == {"sessions": 1, "turns": 2}


def test_replace_session_idempotent_swaps_rows(store: TurnStore):
    """Re-replacing a session swaps its rows; count stays stable, no dupes/orphans."""
    first = [
        {"idx": 0, "role": "user", "ts": "2026-07-01T10:00:00", "text": "primero"},
        {"idx": 1, "role": "assistant", "ts": "2026-07-01T10:00:05", "text": "segundo"},
    ]
    store.replace_session("sess-1", "claude-code", first)

    grown = [
        {"idx": 0, "role": "user", "ts": "2026-07-01T10:00:00", "text": "primero editado"},
        {"idx": 1, "role": "assistant", "ts": "2026-07-01T10:00:05", "text": "segundo"},
        {"idx": 2, "role": "user", "ts": "2026-07-01T10:00:10", "text": "tercero nuevo"},
    ]
    n = store.replace_session("sess-1", "claude-code", grown)

    assert n == 3
    assert store.stats() == {"sessions": 1, "turns": 3}
    assert store.sessions_watermark() == {"sess-1": 2}
    # search must reflect the new content, not the stale first-pass row.
    hits = store.search("editado")
    assert len(hits) == 1
    assert hits[0]["turn_idx"] == 0


def test_search_and_multitoken_diacritics_fold(store: TurnStore):
    """AND multi-token match, and 'decision' (no accent) matches 'decisión'."""
    turns = [
        {
            "idx": 0,
            "role": "assistant",
            "ts": "2026-07-01T10:00:00",
            "text": "la decisión que tomamos fue clara",
        },
        {"idx": 1, "role": "user", "ts": "2026-07-01T10:00:05", "text": "otro turno sin relación"},
    ]
    store.replace_session("sess-1", "claude-code", turns)

    hits = store.search("decision")
    assert len(hits) == 1
    assert hits[0]["turn_idx"] == 0

    hits_multi = store.search("decision tomamos")
    assert len(hits_multi) == 1
    assert hits_multi[0]["turn_idx"] == 0


def test_search_or_fallback_on_zero_and_results(store: TurnStore):
    """Two turns each with one distinct token → AND finds nothing, OR fallback returns both."""
    turns = [
        {"idx": 0, "role": "user", "ts": "2026-07-01T10:00:00", "text": "queso banana"},
        {"idx": 1, "role": "assistant", "ts": "2026-07-01T10:00:05", "text": "manzana durazno"},
    ]
    store.replace_session("sess-1", "claude-code", turns)

    hits = store.search("queso durazno")
    assert {h["turn_idx"] for h in hits} == {0, 1}


def test_search_session_id_filter(store: TurnStore):
    store.replace_session(
        "sess-a",
        "claude-code",
        [{"idx": 0, "role": "user", "ts": "2026-07-01T10:00:00", "text": "palabra clave"}],
    )
    store.replace_session(
        "sess-b",
        "claude-code",
        [{"idx": 0, "role": "user", "ts": "2026-07-01T10:00:00", "text": "palabra clave"}],
    )

    hits = store.search("palabra clave", session_id="sess-a")
    assert len(hits) == 1
    assert hits[0]["session_id"] == "sess-a"


def test_search_since_filter(store: TurnStore):
    turns = [
        {"idx": 0, "role": "user", "ts": "2026-01-01T00:00:00", "text": "palabra vieja"},
        {"idx": 1, "role": "user", "ts": "2026-07-10T00:00:00", "text": "palabra nueva"},
    ]
    store.replace_session("sess-1", "claude-code", turns)

    hits = store.search("palabra", since="2026-06-01T00:00:00")
    assert len(hits) == 1
    assert hits[0]["turn_idx"] == 1


def test_prune_older_than_removes_old_rows_from_both_tables(store: TurnStore):
    from datetime import UTC, datetime, timedelta

    old_ts = (datetime.now(UTC) - timedelta(days=200)).isoformat()
    recent_ts = datetime.now(UTC).isoformat()
    turns = [
        {"idx": 0, "role": "user", "ts": old_ts, "text": "contenido antiguo unico"},
        {"idx": 1, "role": "user", "ts": recent_ts, "text": "contenido reciente unico"},
        {"idx": 2, "role": "user", "ts": None, "text": "contenido sin fecha legado"},
    ]
    store.replace_session("sess-1", "claude-code", turns)

    removed = store.prune_older_than(90)
    assert removed == 2
    assert store.stats() == {"sessions": 1, "turns": 1}
    # FTS side-table must be pruned too — search for the old text finds nothing.
    assert store.search("antiguo") == []
    assert store.search("legado") == []
    assert len(store.search("reciente")) == 1


def test_sessions_watermark(store: TurnStore):
    store.replace_session(
        "sess-a",
        "claude-code",
        [
            {"idx": 0, "role": "user", "ts": "2026-07-01T10:00:00", "text": "a"},
            {"idx": 3, "role": "user", "ts": "2026-07-01T10:00:05", "text": "b"},
        ],
    )
    store.replace_session(
        "sess-b",
        "claude-code",
        [{"idx": 1, "role": "user", "ts": "2026-07-01T10:00:00", "text": "c"}],
    )
    assert store.sessions_watermark() == {"sess-a": 3, "sess-b": 1}


def test_stats_empty_store(store: TurnStore):
    assert store.stats() == {"sessions": 0, "turns": 0}


def test_store_enforces_private_directory_and_database_modes(tmp_path: Path):
    state_dir = tmp_path / "state"
    state_dir.mkdir(mode=0o777)
    state_dir.chmod(0o777)

    private_store = TurnStore(state_dir / "verbatim.db")
    try:
        assert stat.S_IMODE(state_dir.stat().st_mode) == 0o700
        assert stat.S_IMODE(private_store.db_path.stat().st_mode) == 0o600
        for suffix in ("-wal", "-shm"):
            sidecar = Path(f"{private_store.db_path}{suffix}")
            if sidecar.exists():
                assert stat.S_IMODE(sidecar.stat().st_mode) == 0o600
    finally:
        private_store.close()


def test_search_result_limit_is_bounded(store: TurnStore):
    store.replace_session(
        "many-turns",
        "claude-code",
        [
            {
                "idx": idx,
                "role": "assistant",
                "ts": "2026-07-01T10:00:00+00:00",
                "text": f"bounded common result number {idx}",
            }
            for idx in range(125)
        ],
    )

    assert len(store.search("common", limit=10_000)) == 100
    assert len(store.search("common", limit=0)) == 1
