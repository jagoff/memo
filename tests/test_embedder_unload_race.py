"""A concurrent `unload()` must not turn an in-flight embed into an AttributeError.

`_ensure_loaded` releases the load lock before the caller dereferences the
model, so an `unload()` landing in that window used to null `_model`/`_tokenizer`
mid-embed. Observed in production as
`recall-daemon: warm-up failed (AttributeError: 'NoneType' object has no
attribute 'model')` when two daemon instances raced on startup — the warm-up
then degraded to lazy-load for that process.
"""

from __future__ import annotations

import threading

import pytest

from memo.embedder import MLXEmbedder


def _embedder() -> MLXEmbedder:
    return MLXEmbedder("stub-model", 8)


def test_embed_snapshots_the_model_under_the_load_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    """The snapshot is the fix: after it, nulling the attributes cannot affect
    an embed that already read them."""
    emb = _embedder()
    emb._model = object()
    emb._tokenizer = object()
    monkeypatch.setattr(emb, "_ensure_loaded", lambda: None)

    with emb._load_lock:
        model = emb._model
        tokenizer = emb._tokenizer
    emb._model = None  # a concurrent unload landing right here
    emb._tokenizer = None

    assert model is not None and tokenizer is not None


def test_embed_raises_a_clear_error_when_unloaded_instead_of_attributeerror(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An embed racing a completed unload must fail with a sentence that names
    the cause, not `'NoneType' object has no attribute 'model'`."""
    emb = _embedder()
    monkeypatch.setattr(emb, "_ensure_loaded", lambda: None)
    emb._model = None
    emb._tokenizer = None

    with pytest.raises(RuntimeError, match="unloaded before embed"):
        emb.embed(["anything"])


def test_unload_during_a_held_snapshot_does_not_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    """The real interleaving: snapshot taken, unload runs, embed proceeds."""
    emb = _embedder()
    emb._model = object()
    emb._tokenizer = object()
    monkeypatch.setattr(emb, "_ensure_loaded", lambda: None)
    snapshot: dict[str, object] = {}

    def _take() -> None:
        with emb._load_lock:
            snapshot["model"] = emb._model

    def _unload() -> None:
        with emb._load_lock:
            emb._model = None
            emb._tokenizer = None

    t1 = threading.Thread(target=_take)
    t1.start()
    t1.join()
    t2 = threading.Thread(target=_unload)
    t2.start()
    t2.join()

    assert snapshot["model"] is not None
    assert emb._model is None
