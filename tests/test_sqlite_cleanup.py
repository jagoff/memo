"""Regression tests for deterministic SQLite cleanup."""

from __future__ import annotations

import gc
import sqlite3
import threading
import warnings
from pathlib import Path

import pytest

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
