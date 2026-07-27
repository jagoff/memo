"""Regression tests for deterministic SQLite cleanup."""

from __future__ import annotations

import gc
import sqlite3
import threading
import warnings
from contextlib import closing
from pathlib import Path

import pytest

from memo import sqlite_snapshot
from memo.contradict import ContradictionStore
from memo.crossref import CrossReferenceIndex
from memo.graph import GraphStore
from memo.history import HistoryStore
from memo.store import VecStore
from memo.store.episode_store import EpisodeStore
from memo.store.fact_edge_store import FactEdgeStore
from memo.store.hype_store import HypeStore
from memo.store.turn_store import TurnStore
from memo.versioning import VersionManager, VersionStore

pytestmark = pytest.mark.resource_hygiene


def test_snapshot_backup_resolves_source_strictly(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source.db"
    destination = tmp_path / "destination.db"
    with closing(sqlite3.connect(source)) as connection:
        connection.execute("CREATE TABLE items (id INTEGER PRIMARY KEY)")

    original_resolve = Path.resolve
    strict_values: list[bool] = []

    def recording_resolve(path: Path, strict: bool = False) -> Path:
        if path == source:
            strict_values.append(strict)
        return original_resolve(path, strict=strict)

    monkeypatch.setattr(Path, "resolve", recording_resolve)

    sqlite_snapshot._backup_database(source, destination)

    assert strict_values == [True]
    with closing(sqlite3.connect(destination)) as connection:
        assert connection.execute("PRAGMA integrity_check").fetchone() == ("ok",)


def test_snapshot_contract_uses_private_sanitized_scratch(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "source.db"
    destination = tmp_path / "nested" / "deeper" / "published.db"
    with closing(sqlite3.connect(source)) as connection:
        connection.execute("CREATE TABLE items (id INTEGER PRIMARY KEY)")

    scratch = destination.parent / ".fixed-scratch"
    temporary_args: list[tuple[str | None, Path | None]] = []
    backup_calls: list[tuple[Path, Path]] = []
    sanitized_calls: list[Path] = []

    class FakeTemporaryDirectory:
        def __init__(self, *, prefix: str | None = None, dir: Path | None = None) -> None:
            temporary_args.append((prefix, dir))
            scratch.mkdir()

        def __enter__(self) -> str:
            return str(scratch)

        def __exit__(self, *_exc: object) -> None:
            return None

    def fake_backup(backup_source: Path, backup_destination: Path) -> None:
        backup_calls.append((backup_source, backup_destination))
        with closing(sqlite3.connect(backup_destination)) as connection:
            connection.execute("CREATE TABLE items (id INTEGER PRIMARY KEY)")

    def fake_sanitize(database: Path) -> None:
        sanitized_calls.append(database)

    monkeypatch.setattr(sqlite_snapshot.tempfile, "TemporaryDirectory", FakeTemporaryDirectory)
    monkeypatch.setattr(sqlite_snapshot, "_backup_database", fake_backup)
    monkeypatch.setattr(sqlite_snapshot, "_sanitize_secret_store", fake_sanitize)

    sqlite_snapshot.snapshot_sqlite_database(source, destination)

    sanitized = scratch / "sanitized.db"
    assert destination.parent.is_dir()
    assert temporary_args == [(".sqlite-snapshot-", destination.parent)]
    assert backup_calls == [(source, sanitized), (sanitized, destination)]
    assert sanitized_calls == [sanitized]
    assert destination.is_file()


def _cleanup_without_warnings(factory) -> list[warnings.WarningMessage]:
    # Drain unreachable objects left by earlier tests before attributing any
    # ResourceWarning to the object created below.  This matters under xdist,
    # where a worker may run unrelated SQLite tests immediately beforehand.
    gc.collect()
    gc.collect()
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", ResourceWarning)
        obj = factory()
        del obj
        gc.collect()
        gc.collect()
    return captured


def test_cleanup_warning_capture_ignores_preexisting_garbage(tmp_path: Path) -> None:
    class CyclicConnectionHolder:
        def __init__(self) -> None:
            self.connection = sqlite3.connect(tmp_path / "ambient.db")
            self.cycle = self

    holder = CyclicConnectionHolder()
    del holder

    # The preexisting connection is deliberately leaked to exercise the
    # attribution boundary. Capture its warning here so it does not pollute the
    # suite summary; the helper's inner filter still records warnings produced
    # by VersionStore itself.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        warnings_seen = _cleanup_without_warnings(lambda: VersionStore(tmp_path / "versions.db"))
    assert not any(issubclass(w.category, ResourceWarning) for w in warnings_seen)


@pytest.mark.parametrize(
    "factory",
    [
        lambda tmp_path: VecStore(tmp_path / "vec.db", dims=4),
        lambda tmp_path: GraphStore(tmp_path / "graph.db"),
        lambda tmp_path: HistoryStore(tmp_path / "history.db"),
        lambda tmp_path: CrossReferenceIndex(tmp_path / "crossref.db"),
        lambda tmp_path: ContradictionStore(tmp_path / "contradictions.db"),
    ],
)
def test_sqlite_objects_do_not_emit_resourcewarnings(tmp_path: Path, factory) -> None:
    warnings_seen = _cleanup_without_warnings(lambda: factory(tmp_path))
    assert not any(issubclass(w.category, ResourceWarning) for w in warnings_seen)


def test_versioning_objects_do_not_emit_resourcewarnings(tmp_path: Path, mock_memory) -> None:
    store_warnings = _cleanup_without_warnings(lambda: VersionStore(tmp_path / "versions.db"))
    assert not any(issubclass(w.category, ResourceWarning) for w in store_warnings)

    manager_warnings = _cleanup_without_warnings(lambda: VersionManager(mock_memory))
    assert not any(issubclass(w.category, ResourceWarning) for w in manager_warnings)


@pytest.mark.parametrize(
    "cls",
    [VecStore, GraphStore, HistoryStore, CrossReferenceIndex, ContradictionStore, VersionStore],
)
def test_best_effort_destructors_suppress_shutdown_interrupts(cls) -> None:
    obj = object.__new__(cls)

    def raising_close() -> None:
        raise KeyboardInterrupt()

    obj.close = raising_close
    cls.__del__(obj)


@pytest.mark.parametrize(
    ("factory", "reader"),
    [
        pytest.param(
            lambda tmp_path: VecStore(tmp_path / "vec.db", dims=4),
            lambda store: store.count(),
            id="VecStore",
        ),
        pytest.param(
            lambda tmp_path: FactEdgeStore(tmp_path / "facts.db"),
            lambda store: store.get("absent"),
            id="FactEdgeStore",
        ),
        pytest.param(
            lambda tmp_path: HypeStore(tmp_path / "hype.db", dims=4),
            lambda store: store.stats(),
            id="HypeStore",
        ),
        pytest.param(
            lambda tmp_path: EpisodeStore(tmp_path / "episodes.db", dims=4),
            lambda store: store.count(),
            id="EpisodeStore",
        ),
        pytest.param(
            lambda tmp_path: TurnStore(tmp_path / "verbatim.db"),
            lambda store: store.stats(),
            id="TurnStore",
        ),
    ],
)
def test_store_closes_worker_thread_connections(tmp_path: Path, factory, reader) -> None:
    # Same deterministic-cleanup contract as VecStore: FactEdgeStore is shared
    # across the FastMCP HTTP threadpool for the process lifetime, so a WeakSet
    # holder set would let a worker connection vanish before close()/sweep runs.
    store = factory(tmp_path)

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", ResourceWarning)
        threads = [threading.Thread(target=reader, args=(store,)) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        # Worker connections are strongly retained (never left to
        # platform-dependent thread-local finalizers) until either close()
        # or the dead-owner sweep on a later thread's _connect prunes them.
        assert 2 <= len(store._conn_holders) <= 5  # main + last worker at minimum
        # A new thread's _connect sweeps the (now dead) workers' holders.
        sweeper = threading.Thread(target=reader, args=(store,))
        sweeper.start()
        sweeper.join()
        assert len(store._conn_holders) == 2  # main + sweeper; dead workers pruned
        store.close()
        gc.collect()

    assert len(store._conn_holders) == 0
    assert not any(issubclass(w.category, ResourceWarning) for w in captured)
