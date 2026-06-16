"""F3 — cross-Mac signal export/import.

The `.md` memorias sync via git, but the user-signal tables (`access`,
`memory_health`, `source_feedback`) live only in the local rebuildable
`memvec.db`. A fresh clone + reindex on another Mac restores every memoria
but zero signal. `memo sync export-signal` dumps signal to `signal/*.json`
(committed to git); `import-signal` merges it back by memoria id.

Merge semantics (idempotent on re-pull):
  - access:         access_count = max(local, remote); last_accessed = max
  - memory_health:  keep the row with the newer updated_at
  - source_feedback: union by id (INSERT OR IGNORE)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from memo.config import Config
from memo.store import VecStore
from memo.sync_signal import export_signal, import_signal, signal_dir_for


@pytest.fixture
def store(tmp_cfg: Config) -> VecStore:
    s = VecStore(tmp_cfg.db_path, dims=tmp_cfg.embedder_dims, embedder_model=tmp_cfg.embedder_model)
    yield s
    s.close()


def _insert_feedback(store: VecStore, fid: str, source_id: str, rating: int, created: str) -> None:
    """Insert a source_feedback row directly (bypasses query embedding)."""
    with store._tx() as cx:
        cx.execute(
            "INSERT INTO source_feedback (id, source_id, query_text, rating, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (fid, source_id, "q", rating, created),
        )


# -- dump --------------------------------------------------------------------


def test_dump_signal_returns_all_three_tables(store: VecStore):
    store.touch(["a", "b"], ts="2026-01-01T00:00:00+00:00")
    store.boost_roi_batch(["a"])
    _insert_feedback(store, "f1", "a", 1, "2026-01-01T00:00:00+00:00")

    dump = store.dump_signal()

    assert {r["id"] for r in dump["access"]} == {"a", "b"}
    assert {r["id"] for r in dump["memory_health"]} == {"a"}
    assert {r["id"] for r in dump["source_feedback"]} == {"f1"}


# -- export writes json ------------------------------------------------------


def test_export_signal_writes_json_files(store: VecStore, tmp_path: Path):
    store.touch(["a"], ts="2026-01-01T00:00:00+00:00")
    sig = tmp_path / "signal"

    counts = export_signal(store, sig)

    assert (sig / "access.json").exists()
    assert (sig / "memory_health.json").exists()
    assert (sig / "source_feedback.json").exists()
    assert counts["access"] == 1
    payload = json.loads((sig / "access.json").read_text())
    assert payload["schema"] == "memo.sync.signal.v1"
    assert payload["rows"][0]["id"] == "a"


# -- import merge: access = max ---------------------------------------------


def test_import_access_takes_max_count(store: VecStore, tmp_path: Path):
    # local: a touched 3x
    store.touch(["a"], ts="2026-01-01T00:00:00+00:00")
    store.touch(["a"], ts="2026-01-02T00:00:00+00:00")
    store.touch(["a"], ts="2026-01-03T00:00:00+00:00")
    sig = tmp_path / "signal"
    # remote payload: a with higher count + later access, b new
    sig.mkdir()
    (sig / "access.json").write_text(
        json.dumps(
            {
                "schema": "memo.sync.signal.v1",
                "rows": [
                    {"id": "a", "access_count": 10, "last_accessed": "2026-06-01T00:00:00+00:00"},
                    {"id": "b", "access_count": 5, "last_accessed": "2026-05-01T00:00:00+00:00"},
                ],
            }
        )
    )
    (sig / "memory_health.json").write_text(json.dumps({"schema": "memo.sync.signal.v1", "rows": []}))
    (sig / "source_feedback.json").write_text(json.dumps({"schema": "memo.sync.signal.v1", "rows": []}))

    import_signal(store, sig)

    assert store.get_access("a")["access_count"] == 10  # max(3, 10)
    assert store.get_access("a")["last_accessed"] == "2026-06-01T00:00:00+00:00"
    assert store.get_access("b")["access_count"] == 5  # new row


def test_import_access_keeps_local_when_higher(store: VecStore, tmp_path: Path):
    for _ in range(8):
        store.touch(["a"], ts="2026-01-01T00:00:00+00:00")
    sig = tmp_path / "signal"
    sig.mkdir()
    (sig / "access.json").write_text(
        json.dumps(
            {
                "schema": "memo.sync.signal.v1",
                "rows": [{"id": "a", "access_count": 2, "last_accessed": "2025-01-01T00:00:00+00:00"}],
            }
        )
    )
    (sig / "memory_health.json").write_text(json.dumps({"schema": "memo.sync.signal.v1", "rows": []}))
    (sig / "source_feedback.json").write_text(json.dumps({"schema": "memo.sync.signal.v1", "rows": []}))

    import_signal(store, sig)

    assert store.get_access("a")["access_count"] == 8  # local wins


# -- import merge: health = latest updated_at -------------------------------


def test_import_health_keeps_newer(store: VecStore, tmp_path: Path):
    # local health for 'a' stamped older; remote newer must win
    with store._tx() as cx:
        cx.execute(
            "INSERT INTO memory_health (id, confidence, roi_score, updated_at) VALUES "
            "('a', 0.5, 0.5, '2026-01-01T00:00:00+00:00')"
        )
    sig = tmp_path / "signal"
    sig.mkdir()
    (sig / "access.json").write_text(json.dumps({"schema": "memo.sync.signal.v1", "rows": []}))
    (sig / "memory_health.json").write_text(
        json.dumps(
            {
                "schema": "memo.sync.signal.v1",
                "rows": [
                    {"id": "a", "confidence": 0.9, "roi_score": 1.3, "updated_at": "2026-06-01T00:00:00+00:00"},
                    {"id": "b", "confidence": 0.7, "roi_score": 1.0, "updated_at": "2026-06-01T00:00:00+00:00"},
                ],
            }
        )
    )
    (sig / "source_feedback.json").write_text(json.dumps({"schema": "memo.sync.signal.v1", "rows": []}))

    import_signal(store, sig)

    h = store.get_health_batch(["a", "b"])
    assert h["a"]["confidence"] == 0.9  # remote newer wins
    assert h["b"]["confidence"] == 0.7  # new row


def test_import_health_keeps_local_when_newer(store: VecStore, tmp_path: Path):
    with store._tx() as cx:
        cx.execute(
            "INSERT INTO memory_health (id, confidence, roi_score, updated_at) VALUES "
            "('a', 0.9, 1.4, '2026-06-01T00:00:00+00:00')"
        )
    sig = tmp_path / "signal"
    sig.mkdir()
    (sig / "access.json").write_text(json.dumps({"schema": "memo.sync.signal.v1", "rows": []}))
    (sig / "memory_health.json").write_text(
        json.dumps(
            {
                "schema": "memo.sync.signal.v1",
                "rows": [{"id": "a", "confidence": 0.2, "roi_score": 0.2, "updated_at": "2026-01-01T00:00:00+00:00"}],
            }
        )
    )
    (sig / "source_feedback.json").write_text(json.dumps({"schema": "memo.sync.signal.v1", "rows": []}))

    import_signal(store, sig)

    assert store.get_health_batch(["a"])["a"]["confidence"] == 0.9  # local newer wins


# -- import merge: feedback = union by id -----------------------------------


def test_import_feedback_union_by_id(store: VecStore, tmp_path: Path):
    _insert_feedback(store, "f1", "a", 1, "2026-01-01T00:00:00+00:00")
    sig = tmp_path / "signal"
    sig.mkdir()
    (sig / "access.json").write_text(json.dumps({"schema": "memo.sync.signal.v1", "rows": []}))
    (sig / "memory_health.json").write_text(json.dumps({"schema": "memo.sync.signal.v1", "rows": []}))
    (sig / "source_feedback.json").write_text(
        json.dumps(
            {
                "schema": "memo.sync.signal.v1",
                "rows": [
                    {"id": "f1", "source_id": "a", "query_text": "q", "rating": 1, "created_at": "2026-01-01T00:00:00+00:00", "extra_json": None},
                    {"id": "f2", "source_id": "b", "query_text": "q", "rating": -1, "created_at": "2026-02-01T00:00:00+00:00", "extra_json": None},
                ],
            }
        )
    )

    import_signal(store, sig)

    dump = store.dump_signal()
    assert {r["id"] for r in dump["source_feedback"]} == {"f1", "f2"}


# -- round trip + idempotency ------------------------------------------------


def test_roundtrip_export_import_idempotent(store: VecStore, tmp_path: Path):
    store.touch(["a", "b"], ts="2026-01-01T00:00:00+00:00")
    store.touch(["a"], ts="2026-01-02T00:00:00+00:00")
    store.boost_roi_batch(["a"])
    _insert_feedback(store, "f1", "a", 1, "2026-01-01T00:00:00+00:00")
    sig = tmp_path / "signal"
    export_signal(store, sig)

    # importing our own export back must not inflate counts
    before = store.get_access("a")["access_count"]
    import_signal(store, sig)
    import_signal(store, sig)  # twice
    assert store.get_access("a")["access_count"] == before


def test_signal_dir_for_is_sibling_of_memorias(tmp_cfg: Config):
    # signal/ lives next to the memorias dir under the repo root
    d = signal_dir_for(tmp_cfg)
    assert d.name == "signal"
    assert d.parent == Path(tmp_cfg.data_dir).parent


# -- CLI wiring --------------------------------------------------------------


def test_cli_export_then_import_signal(tmp_path: Path):
    from click.testing import CliRunner

    from memo.cli import cli

    data = tmp_path / "data"
    data.mkdir()
    env = {
        "MEMO_CONFIG_FILE": str(tmp_path / "memo-config.toml"),
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_DATA_DIR": str(data),
        "MEMO_STATE_DIR": str(tmp_path / "state"),
        "MEMO_EMBEDDER_DIMS": "4",
        "MEMO_EMBEDDER_MODEL": "stub",
    }
    runner = CliRunner()

    r = runner.invoke(cli, ["sync", "export-signal", "--json"], env=env)
    assert r.exit_code == 0, r.output
    sig = data.parent / "signal"
    assert (sig / "access.json").exists()

    r2 = runner.invoke(cli, ["sync", "import-signal", "--json"], env=env)
    assert r2.exit_code == 0, r2.output
