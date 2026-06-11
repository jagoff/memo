from __future__ import annotations

import json
from types import SimpleNamespace

from memo.trinity_server import TrinityHandler


class _FakeEmbedder:
    def embed_query(self, text: str) -> list[float]:
        return [0.1, 0.2] if text == "hola" else []

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(i), float(i + 1)] for i, _ in enumerate(texts)]


def _handler() -> TrinityHandler:
    h = TrinityHandler.__new__(TrinityHandler)
    h.server = SimpleNamespace(
        _cfg=SimpleNamespace(embedder_model="fake-model", embedder_dims=2),
        _mem=SimpleNamespace(embedder=_FakeEmbedder()),
    )
    return h


def test_trinity_delegates_embed_query() -> None:
    out = json.loads(_handler()._delegate_to_embedder("embed_query", {"text": "hola"}))
    assert out == {
        "vector": [0.1, 0.2],
        "dim": 2,
        "dims": 2,
        "model": "fake-model",
    }


def test_trinity_delegates_embed_batch() -> None:
    out = json.loads(_handler()._delegate_to_embedder("embed_batch", {"texts": ["a", "b"]}))
    assert out == {
        "vectors": [[0.0, 1.0], [1.0, 2.0]],
        "dim": 2,
        "dims": 2,
        "model": "fake-model",
    }


def test_trinity_rejects_unknown_embedder_op() -> None:
    out = json.loads(_handler()._delegate_to_embedder("bogus", {}))
    assert out == {"error": "unknown op: 'bogus'"}
