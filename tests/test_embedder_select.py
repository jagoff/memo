"""Unit tests for embedder backend selection (`embedder_select`).

Selection is exercised with a duck-typed config namespace; neither backend's
heavy runtime (MLX / sentence-transformers) is imported at construction time,
so these run on any host.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from memo import embedder_select
from memo.embedder_select import active_embedder_identity, make_embedder, resolve_backend
from memo.errors import MemoError
from memo.model_pins import PINNED_MODEL_REVISIONS
from memo.repo_index_helpers import _embed_cache_model


def _cfg(**over):
    base = dict(
        embedder_backend="auto",
        embedder_model="mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ",
        embedder_revision=PINNED_MODEL_REVISIONS["mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ"],
        st_embedder_model="Qwen/Qwen3-Embedding-0.6B",
        st_embedder_revision=PINNED_MODEL_REVISIONS["Qwen/Qwen3-Embedding-0.6B"],
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
    monkeypatch.setattr(sys, "platform", "linux")
    assert resolve_backend(_cfg(embedder_backend="auto")) == "st"


def test_resolve_backend_unknown_value_treated_as_auto(monkeypatch):
    monkeypatch.setattr(embedder_select, "mlx_available", lambda: False)
    monkeypatch.setattr(sys, "platform", "linux")
    assert resolve_backend(_cfg(embedder_backend="bogus")) == "st"


@pytest.mark.parametrize("platform", ["darwin", "win32"])
def test_resolve_backend_auto_rejects_unsupported_non_linux_without_mlx(monkeypatch, platform):
    monkeypatch.setattr(embedder_select, "mlx_available", lambda: False)
    monkeypatch.setattr(sys, "platform", platform)

    with pytest.raises(MemoError, match="unsupported platform"):
        resolve_backend(_cfg(embedder_backend="auto"))


def test_make_embedder_st_returns_st_backend(monkeypatch):
    monkeypatch.setattr(embedder_select, "mlx_available", lambda: False)
    emb = make_embedder(_cfg(embedder_backend="st", embedder_dims=1024))
    assert type(emb).__name__ == "STEmbedder"
    assert emb.dims == 1024
    assert emb.model_name == (
        "Qwen/Qwen3-Embedding-0.6B@" + PINNED_MODEL_REVISIONS["Qwen/Qwen3-Embedding-0.6B"]
    )


def test_make_embedder_st_forwards_revision(monkeypatch):
    monkeypatch.setattr(embedder_select, "mlx_available", lambda: False)
    revision = "a" * 40
    emb = make_embedder(_cfg(embedder_backend="st", st_embedder_revision=revision))

    assert emb.model_name == f"Qwen/Qwen3-Embedding-0.6B@{revision}"


def test_make_embedder_mlx_returns_mlx_backend(monkeypatch):
    monkeypatch.setattr(embedder_select, "mlx_available", lambda: True)
    emb = make_embedder(_cfg(embedder_backend="mlx", embedder_dims=1024))
    assert type(emb).__name__ == "MLXEmbedder"
    assert emb.dims == 1024
    assert emb.revision == _cfg().embedder_revision


def test_active_embedder_identity_includes_st_revision():
    revision = "b" * 40
    assert (
        active_embedder_identity(_cfg(embedder_backend="st", st_embedder_revision=revision))
        == f"Qwen/Qwen3-Embedding-0.6B@{revision}"
    )


def test_active_embedder_identity_includes_mlx_revision():
    cfg = _cfg(embedder_backend="mlx")
    assert active_embedder_identity(cfg) == f"{cfg.embedder_model}@{cfg.embedder_revision}"


def test_repo_embedding_cache_identity_includes_st_revision():
    revision = "c" * 40
    cfg = _cfg(embedder_backend="st", st_embedder_revision=revision)
    embedder = make_embedder(cfg)

    assert _embed_cache_model(embedder, cfg) == f"Qwen/Qwen3-Embedding-0.6B@{revision}"
