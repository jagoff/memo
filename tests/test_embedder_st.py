"""Unit tests for the CPU (sentence-transformers) embedder backend.

`sentence_transformers` is stubbed in `sys.modules` so these run without torch
on the MLX dev box and on clean CI — they exercise STEmbedder's contract
(shapes, asymmetric query prefix, dim guard), not a real model.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from memo.embedder import _QUERY_INSTRUCTION_PREFIX, assert_valid_embedding
from memo.embedder_st import STEmbedder


def _l2_normalize(vec: list[float]) -> list[float]:
    norm = sum(x * x for x in vec) ** 0.5 or 1.0
    return [x / norm for x in vec]


class _FakeSentenceTransformer:
    """Deterministic, normalized vectors; records encode() inputs."""

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        dim: int = 4,
        revision: str | None = None,
    ) -> None:
        self.model_path = model_path
        self.device = device
        self.revision = revision
        self._dim = dim
        self.max_seq_length = 0
        self.last_inputs: list[str] = []
        self.legacy_dimension_calls = 0

    def get_embedding_dimension(self) -> int:
        return self._dim

    def get_sentence_embedding_dimension(self) -> int:
        self.legacy_dimension_calls += 1
        return self._dim

    def encode(
        self,
        items,
        normalize_embeddings=False,
        convert_to_numpy=True,
        batch_size=32,
        show_progress_bar=False,
        **_,
    ):
        self.last_inputs = list(items)
        rows = []
        for text in items:
            raw = [float(len(text) + k + 1) for k in range(self._dim)]
            rows.append(_l2_normalize(raw) if normalize_embeddings else raw)
        return rows


def _install_fake_st(monkeypatch, *, dim: int = 4) -> dict:
    captured: dict = {}

    def _factory(model_path, device="cpu", revision=None):
        inst = _FakeSentenceTransformer(model_path, device=device, dim=dim, revision=revision)
        captured["instance"] = inst
        return inst

    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = _factory  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    return captured


def test_embed_returns_normalized_vectors_of_expected_dim(monkeypatch):
    captured = _install_fake_st(monkeypatch, dim=4)
    emb = STEmbedder(model_path="/tmp/fake-model", expected_dims=4)

    out = emb.embed(["alpha", "beta gamma"])

    assert len(out) == 2
    for vec in out:
        assert len(vec) == 4
        assert_valid_embedding(vec, 4)  # passes the norm≈1 boundary check
    assert captured["instance"].legacy_dimension_calls == 0


def test_embedder_loads_exact_configured_revision(monkeypatch):
    captured = _install_fake_st(monkeypatch, dim=4)
    revision = "f" * 40
    emb = STEmbedder(model_path="fake/model", revision=revision, expected_dims=4)

    emb.embed(["alpha"])

    assert captured["instance"].revision == revision
    assert emb.model_name == f"fake/model@{revision}"


def test_non_linux_missing_st_dependency_does_not_recommend_reinstall(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    emb = STEmbedder(model_path=str(Path("/tmp/local-model")), expected_dims=4)

    with pytest.raises(RuntimeError) as excinfo:
        emb.embed(["alpha"])

    assert "explicit ST backend" in str(excinfo.value)
    assert "Reinstall" not in str(excinfo.value)


def test_embed_empty_sequence_returns_empty(monkeypatch):
    _install_fake_st(monkeypatch)
    emb = STEmbedder(expected_dims=4)
    assert emb.embed([]) == []


def test_embed_rejects_bare_string(monkeypatch):
    _install_fake_st(monkeypatch)
    emb = STEmbedder(expected_dims=4)
    with pytest.raises(TypeError):
        emb.embed("not a list")  # type: ignore[arg-type]


def test_embed_query_applies_asymmetric_prefix(monkeypatch):
    captured = _install_fake_st(monkeypatch, dim=4)
    emb = STEmbedder(model_path="/tmp/fake-model", expected_dims=4)

    vec = emb.embed_query("astor terapia")

    assert len(vec) == 4
    # The query, not a document, gets the instruction prefix (MLX invariant #1).
    sent = captured["instance"].last_inputs
    assert sent == [_QUERY_INSTRUCTION_PREFIX + "astor terapia"]


def test_embed_does_not_prefix_documents(monkeypatch):
    captured = _install_fake_st(monkeypatch, dim=4)
    emb = STEmbedder(expected_dims=4)

    emb.embed(["plain document"])

    assert captured["instance"].last_inputs == ["plain document"]


def test_embed_query_empty_returns_zero_vector(monkeypatch):
    _install_fake_st(monkeypatch, dim=4)
    emb = STEmbedder(expected_dims=4)
    assert emb.embed_query("   ") == [0.0, 0.0, 0.0, 0.0]


def test_dim_mismatch_raises_on_load(monkeypatch):
    _install_fake_st(monkeypatch, dim=8)  # model yields 8, config expects 4
    emb = STEmbedder(expected_dims=4)
    with pytest.raises(RuntimeError, match="dim"):
        emb.embed(["x"])


def test_dims_and_warm_state(monkeypatch):
    _install_fake_st(monkeypatch, dim=4)
    emb = STEmbedder(expected_dims=4)
    assert emb.dims == 4
    assert emb.is_warm is False
    emb.embed(["warm me"])
    assert emb.is_warm is True
    emb.unload()
    assert emb.is_warm is False
