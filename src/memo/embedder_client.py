"""Shared-embedder client — socket-first, in-process fallback.

The recall daemon at `recall_server.py` keeps a single warm MLX embedder
in RAM (~2GB for Qwen3-Embedding-0.6B + reranker). Any peer process —
synapse's federator, memflow's daemon, another memo CLI — can reuse
that warm instance instead of loading its own copy. This module is the
client side of that contract.

Two public functions mirror the `MLXEmbedder` surface so callers can
drop this in wherever they would have instantiated `MLXEmbedder()`:

    from memo.embedder_client import embed, embed_query

    vec   = embed_query("astor terapia ocupacional")
    vecs  = embed(["doc 1", "doc 2", "doc 3"])

Routing logic per call:

1. Look up the daemon socket at `state_dir/recall.sock`.
2. If present, send a JSON-line request (`embed_query` / `embed_batch`)
   and return the response. Sub-millisecond on the daemon side after
   model warm-up.
3. If absent / refused / timed out, **fall back in-process**: load
   `MLXEmbedder` lazily (the first call pays MLX cold-start, ~2s) and
   call the same method directly.

The fallback is deliberate. Callers running on a peer Mac without a
memo daemon still get correct embeddings — just slower on the first
call. Set `MEMO_EMBEDDER_CLIENT_REQUIRE_DAEMON=1` to raise instead.

Environment knobs:
    MEMO_EMBEDDER_CLIENT_TIMEOUT          seconds, default "8.0"
    MEMO_EMBEDDER_CLIENT_REQUIRE_DAEMON   "1" disables fallback
"""

from __future__ import annotations

import json
import logging
import os
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from memo.recall_server import connect_and_send

_log = logging.getLogger(__name__)
_DEFAULT_TIMEOUT_S = 8.0
_state_dir_lock = threading.Lock()
_cached_state_dir: Path | None = None
_inproc_lock = threading.Lock()
_inproc_embedder: Any | None = None


def _resolve_state_dir(state_dir: Path | None) -> Path:
    """Return the state dir, caching the env-derived value."""
    if state_dir is not None:
        return state_dir
    global _cached_state_dir
    with _state_dir_lock:
        if _cached_state_dir is not None:
            return _cached_state_dir
        from memo.config import Config

        _cached_state_dir = Config.from_env().state_dir
        return _cached_state_dir


def _timeout() -> float:
    raw = os.environ.get("MEMO_EMBEDDER_CLIENT_TIMEOUT")
    if not raw:
        return _DEFAULT_TIMEOUT_S
    try:
        value = float(raw)
    except ValueError:
        return _DEFAULT_TIMEOUT_S
    return value if value > 0 else _DEFAULT_TIMEOUT_S


def _require_daemon() -> bool:
    return os.environ.get("MEMO_EMBEDDER_CLIENT_REQUIRE_DAEMON") == "1"


def _try_socket(state_dir: Path, payload: dict[str, Any]) -> dict[str, Any] | None:
    raw = connect_and_send(state_dir, payload, timeout=_timeout())
    if not raw:
        return None
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError as exc:
        _log.warning("embedder_client: malformed daemon response (%s)", exc)
        return None
    if not isinstance(decoded, dict):
        return None
    if "error" in decoded:
        _log.warning("embedder_client: daemon error: %s", decoded["error"])
        return None
    return decoded


def _inproc() -> Any:
    """Lazy-load an in-process `MLXEmbedder` instance (singleton).

    Imports are deferred so callers without MLX (Linux CI) never trigger
    the import unless they actually reach the fallback path.
    """
    global _inproc_embedder
    with _inproc_lock:
        if _inproc_embedder is not None:
            return _inproc_embedder
        from memo.config import Config
        from memo.embedder import MLXEmbedder

        cfg = Config.from_env()
        _inproc_embedder = MLXEmbedder(
            model_name=cfg.embedder_model,
            expected_dims=cfg.embedder_dims,
        )
        return _inproc_embedder


# -- public API ------------------------------------------------------------


def embed_query(text: str, *, state_dir: Path | None = None) -> list[float]:
    """Embed a single query string (asymmetric, query-prefixed).

    Returns a list of floats matching the embedder's configured
    dimensionality. Raises `RuntimeError` if the daemon is unreachable
    AND `MEMO_EMBEDDER_CLIENT_REQUIRE_DAEMON=1`.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("embed_query: empty text")
    resolved = _resolve_state_dir(state_dir)
    decoded = _try_socket(resolved, {"op": "embed_query", "text": text})
    if decoded is not None:
        vec = decoded.get("vector")
        if isinstance(vec, list):
            return [float(x) for x in vec]
        _log.warning("embedder_client: daemon embed_query missing `vector` field")
    if _require_daemon():
        raise RuntimeError(
            "embed_query: daemon unreachable and "
            "MEMO_EMBEDDER_CLIENT_REQUIRE_DAEMON=1 disables in-process fallback"
        )
    _log.warning(
        "embedder_client: daemon unreachable, falling back to in-process "
        "(first call will be slow ~2s, cold MLX load)"
    )
    return _inproc().embed_query(text)


def embed(
    texts: Sequence[str],
    *,
    state_dir: Path | None = None,
) -> list[list[float]]:
    """Embed a batch of documents (symmetric, no query prefix).

    Mirrors `MLXEmbedder.embed`. Pass a Sequence[str]; passing a bare
    `str` is rejected to surface the v0.3.1 string-as-iterable bug
    early instead of returning per-character embeddings.
    """
    if isinstance(texts, str):
        raise TypeError(
            "embed: pass Sequence[str], not bare str. Wrap as `[text]` "
            "to avoid the v0.3.1 character-iteration bug.",
        )
    items = list(texts)
    if not items:
        return []
    if not all(isinstance(t, str) for t in items):
        raise TypeError("embed: every element of `texts` must be a string")
    resolved = _resolve_state_dir(state_dir)
    decoded = _try_socket(resolved, {"op": "embed_batch", "texts": items})
    if decoded is not None:
        vectors = decoded.get("vectors")
        if isinstance(vectors, list):
            return [[float(x) for x in v] for v in vectors]
        _log.warning("embedder_client: daemon embed_batch missing `vectors` field")
    if _require_daemon():
        raise RuntimeError(
            "embed: daemon unreachable and "
            "MEMO_EMBEDDER_CLIENT_REQUIRE_DAEMON=1 disables in-process fallback"
        )
    _log.warning(
        "embedder_client: daemon unreachable, falling back to in-process "
        "(batch of %d items, first call will be slow ~2s, cold MLX load)",
        len(items),
    )
    return _inproc().embed(items)


def ping(*, state_dir: Path | None = None) -> dict[str, Any] | None:
    """Cheap warm-state probe. Returns `None` if the daemon is unreachable."""
    resolved = _resolve_state_dir(state_dir)
    return _try_socket(resolved, {"op": "ping"})


def status(*, state_dir: Path | None = None) -> dict[str, Any] | None:
    """Daemon alive/dead + uptime + model. `None` if unreachable.

    Alias for `ping()` with a name that reads better at call sites
    (`memo embed-daemon status`). Shape:
        {"ok": True, "model": "...", "dims": N,
         "started_at": <epoch>, "uptime_s": N}
    """
    return ping(state_dir=state_dir)


def stats(*, state_dir: Path | None = None) -> dict[str, Any] | None:
    """Per-op request/error counters + latency percentiles. `None` if unreachable.

    Shape:
        {"started_at": <epoch>, "uptime_s": N, "model": "...", "dims": N,
         "ops": {op: {count, errors, samples, p50_ms, p95_ms, p99_ms}}}
    """
    resolved = _resolve_state_dir(state_dir)
    return _try_socket(resolved, {"op": "stats"})


__all__ = ["embed", "embed_query", "ping", "stats", "status"]
