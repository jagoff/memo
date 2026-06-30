"""Unit tests for embedder backend selection (`embedder_select`).

Selection is exercised with a duck-typed config namespace; neither backend's
heavy runtime (MLX / sentence-transformers) is imported at construction time,
so these run on any host.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from memo import embedder_select
from memo.embedder_select import make_embedder, resolve_backend


def _cfg(**over):
    base = dict(
        embedder_backend="auto",
        embedder_model="mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ",
        st_embedder_model="Qwen/Qwen3-Embedding-0.6B",
        embedder_dims=1024,
    )
    base.update(over)
    return SimpleNamespace(**base)


@pytest.mark.parametrize("forced", ["mlx", "st"])
def test_resolve_backend_honors_explicit(forced):
    assert resolve_backend(_cfg(embedder_backend=forced)) == forced


def test_resolve_backend_auto_uses_mlx_when_available(monkeypatch):
    monkeypatch.setattr(embedder_select, "mlx_available", lambda: True)
    assert resolve_backend(_cfg(embedder_backend="auto")) == "mlx"


def test_resolve_backend_auto_falls_back_to_st(monkeypatch):
    monkeypatch.setattr(embedder_select, "mlx_available", lambda: False)
    assert resolve_backend(_cfg(embedder_backend="auto")) == "st"


def test_resolve_backend_unknown_value_treated_as_auto(monkeypatch):
    monkeypatch.setattr(embedder_select, "mlx_available", lambda: False)
    assert resolve_backend(_cfg(embedder_backend="bogus")) == "st"


def test_make_embedder_st_returns_st_backend(monkeypatch):
    monkeypatch.setattr(embedder_select, "mlx_available", lambda: False)
    emb = make_embedder(_cfg(embedder_backend="st", embedder_dims=1024))
    assert type(emb).__name__ == "STEmbedder"
    assert emb.dims == 1024
    assert emb.model_name == "Qwen/Qwen3-Embedding-0.6B"


def test_make_embedder_mlx_returns_mlx_backend(monkeypatch):
    monkeypatch.setattr(embedder_select, "mlx_available", lambda: True)
    emb = make_embedder(_cfg(embedder_backend="mlx", embedder_dims=1024))
    assert type(emb).__name__ == "MLXEmbedder"
    assert emb.dims == 1024
