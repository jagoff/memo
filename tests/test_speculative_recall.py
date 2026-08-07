from __future__ import annotations

import json
from unittest.mock import MagicMock

from memo.recall_socket import _RecallHandler


class MockServer:
    def __init__(self) -> None:
        self._cfg = MagicMock()
        self._cfg.embedder_model = "test-model"
        self._cfg.embedder_dims = 4
        self._mem = MagicMock()
        self._mem.embedder.embed_query.return_value = [0.1, 0.2, 0.3, 0.4]
        self._speculative_cache: dict[str, list[float]] = {}


def test_speculative_pre_fetch_cache() -> None:
    handler = _RecallHandler.__new__(_RecallHandler)
    handler.server = MockServer()  # type: ignore[assignment]

    req = {"text": "how to configure database"}

    # First call: cache miss, computes embedding
    res1_raw = handler._embed_query(req)
    res1 = json.loads(res1_raw)
    assert res1["vector"] == [0.1, 0.2, 0.3, 0.4]
    assert "how to configure database" in handler.server._speculative_cache

    # Second call: cache hit!
    res2_raw = handler._embed_query(req)
    res2 = json.loads(res2_raw)
    assert res2["vector"] == [0.1, 0.2, 0.3, 0.4]
    assert res2.get("speculative_hit") is True
