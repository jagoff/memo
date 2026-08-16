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
    return MLXEmbedder("stub-model", expected_dims=8)


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


def test_snapshot_returns_both_when_loaded() -> None:
    """The happy path through the shared helper — without it the raise branches
    are the only thing exercised and the guard could be inverted unnoticed."""
    import threading

    from memo.embedder import _snapshot_loaded

    model, tokenizer = object(), object()
    got_model, got_tokenizer = _snapshot_loaded(
        threading.Lock(), lambda: model, lambda: tokenizer, "unused"
    )

    assert got_model is model
    assert got_tokenizer is tokenizer


def test_snapshot_message_differs_by_caller() -> None:
    """MicroEmbedder's `_ensure_loaded` SWALLOWS a load error and leaves _model
    None, so there a null means "failed to load". MLXEmbedder's raises on load
    failure, so there it can only mean a concurrent unload. Flattening the two
    into one message loses a real distinction — and broke an existing test that
    asserted "failed to load"."""
    import threading

    import pytest as _pytest

    from memo.embedder import _snapshot_loaded

    with _pytest.raises(RuntimeError, match="failed to load"):
        _snapshot_loaded(
            threading.Lock(), lambda: None, lambda: None, "MicroEmbedder: ... failed to load"
        )
    with _pytest.raises(RuntimeError, match="unloaded before embed"):
        _snapshot_loaded(
            threading.Lock(),
            lambda: None,
            lambda: object(),
            "embedder model was unloaded before embed could run",
        )


def test_snapshot_raises_when_only_the_tokenizer_is_gone() -> None:
    """unload() nulls both, but they are separate attributes: a guard checking
    only the model would sail past a half-unloaded embedder."""
    import threading

    import pytest as _pytest

    from memo.embedder import _snapshot_loaded

    with _pytest.raises(RuntimeError):
        _snapshot_loaded(threading.Lock(), lambda: object(), lambda: None, "gone")


# -- the embed bodies, exercised without MLX ---------------------------------
#
# These lines only ran on Apple Silicon before: CI's Linux runners cannot
# import mlx, so `embed()`'s body had zero coverage there. The fake below is
# deliberately tiny — it fakes only the four ops the bodies call — and each
# test asserts a real contract, not that a stub returned a stub.


class _FakeArr:
    def __init__(self, data, shape=None):
        self.data = data
        self.shape = shape or (1, 1, len(data[0]) if data and isinstance(data[0], list) else 1)

    def __mul__(self, other):
        return self

    def __truediv__(self, other):
        return self

    def __getitem__(self, key):
        return self

    def tolist(self):
        return self.data[0] if self.data else []


def _fake_mlx(monkeypatch: pytest.MonkeyPatch, *, dims: int) -> None:
    import sys
    import types

    core = types.ModuleType("mlx.core")
    core.array = lambda d: _FakeArr(d if isinstance(d[0], list) else [d])  # type: ignore[attr-defined]
    core.mean = lambda a, axis=None: _FakeArr([[1.0] * dims])  # type: ignore[attr-defined]
    core.sum = lambda a, axis=None, keepdims=False: _FakeArr([[1.0]])  # type: ignore[attr-defined]
    core.sqrt = lambda a: _FakeArr([[1.0]])  # type: ignore[attr-defined]
    core.where = lambda c, a, b: b  # type: ignore[attr-defined]
    core.ones_like = lambda a: a  # type: ignore[attr-defined]
    core.clear_cache = lambda: None  # type: ignore[attr-defined]
    mlx = types.ModuleType("mlx")
    mlx.core = core  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", core)

    from contextlib import contextmanager

    @contextmanager
    def _noop_guard():
        yield

    monkeypatch.setattr("memo.embedder.gpu_guard", _noop_guard)


class _StubTokenizer:
    eos_token_id = 7

    def encode(self, text, add_special_tokens=False):
        return [1, 2, 3]


def test_micro_embed_uses_the_snapshotted_locals(monkeypatch: pytest.MonkeyPatch) -> None:
    """Runs MicroEmbedder's pooling body end to end with the model nulled
    AFTERWARDS: it must still succeed, because the body holds locals."""
    from memo.embedder import MicroEmbedder

    _fake_mlx(monkeypatch, dims=4)
    micro = MicroEmbedder("stub", expected_dims=4)
    micro._model = type("M", (), {"model": lambda self, arr: _FakeArr([[1.0] * 4])})()
    micro._tokenizer = _StubTokenizer()
    monkeypatch.setattr(type(micro), "_ensure_loaded", lambda self: None)

    out = micro.embed(["hola"])

    assert len(out) == 1 and len(out[0]) == 4


def test_mlx_embed_rejects_a_model_whose_width_is_not_the_configured_dims(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MLX invariant 3 at runtime: a model whose hidden width disagrees with
    `embedder_dims` must fail loudly, not write wrong-width vectors into vec0.
    Reaching this guard also exercises the tokenizer/model dereferences."""
    from memo.embedder import MLXEmbedder

    _fake_mlx(monkeypatch, dims=8)
    emb = MLXEmbedder("stub-model", expected_dims=8)
    emb._model = type(
        "M", (), {"model": lambda self, arr: _FakeArr([[1.0] * 3], shape=(1, 1, 3))}
    )()
    emb._tokenizer = _StubTokenizer()
    monkeypatch.setattr(type(emb), "_ensure_loaded", lambda self: None)

    with pytest.raises(RuntimeError, match=r"dim=3.*expects 8"):
        emb.embed(["hola"])
