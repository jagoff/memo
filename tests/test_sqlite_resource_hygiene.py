from __future__ import annotations

import gc
import warnings

import pytest

from memo.config import Config
from memo.memory import Memory


def _sqlite_resource_warnings(
    caught: list[warnings.WarningMessage],
) -> list[warnings.WarningMessage]:
    return [
        w
        for w in caught
        if issubclass(w.category, ResourceWarning) and "unclosed database" in str(w.message).lower()
    ]


def _drain_preexisting_resource_warnings() -> None:
    """Collect garbage from earlier tests outside the attribution window."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        gc.collect()
        gc.collect()


def test_memory_close_releases_sqlite_connections(
    tmp_cfg: Config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    embedding = [1.0, *([0.0] * (tmp_cfg.embedder_dims - 1))]
    monkeypatch.setattr(
        "memo.embedder.MLXEmbedder.embed",
        lambda self, inputs: [embedding for _ in inputs],
    )

    _drain_preexisting_resource_warnings()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ResourceWarning)
        mem = Memory(tmp_cfg)
        mem.save(content="sqlite cleanup probe", title="SQLite Cleanup Probe")
        assert mem.search("sqlite cleanup probe", mode="bm25", limit=1)
        mem.close()
        del mem
        gc.collect()

    assert _sqlite_resource_warnings(caught) == []


def test_memory_close_is_idempotent_after_lazy_connections(
    tmp_cfg: Config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    embedding = [1.0, *([0.0] * (tmp_cfg.embedder_dims - 1))]
    monkeypatch.setattr(
        "memo.embedder.MLXEmbedder.embed",
        lambda self, inputs: [embedding for _ in inputs],
    )

    _drain_preexisting_resource_warnings()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ResourceWarning)
        mem = Memory(tmp_cfg)
        _ = mem.store.get("missing-id")
        mem.close()
        mem.close()
        del mem
        gc.collect()

    assert _sqlite_resource_warnings(caught) == []


def test_memory_finalizer_releases_sqlite_connections(
    tmp_cfg: Config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "memo.embedder.MLXEmbedder.embed",
        lambda self, inputs: [[1.0] + [0.0] * (tmp_cfg.embedder_dims - 1) for _ in inputs],
    )

    _drain_preexisting_resource_warnings()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ResourceWarning)
        mem = Memory(tmp_cfg)
        mem.save(content="sqlite finalizer cleanup probe", title="SQLite Finalizer Probe")
        assert mem.search("sqlite finalizer cleanup probe", mode="bm25", limit=1)
        del mem
        gc.collect()

    assert _sqlite_resource_warnings(caught) == []
