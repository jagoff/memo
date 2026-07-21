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
    MEMO_EMBEDDER_CLIENT_TIMEOUT          seconds; overrides the per-op
                                          defaults (query 30, batch 120,
                                          ping/stats 5)
    MEMO_EMBEDDER_CLIENT_REQUIRE_DAEMON   "1" disables fallback
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from memo.recall_server import connect_and_send

_log = logging.getLogger(__name__)
# Per-op defaults. A single flat timeout can't serve both: control ops must
# stay snappy, while document batches on the 4B "quality" profile routinely
# exceed 60s under GPU contention — timing out then makes the client fall
# back in-process, load a SECOND model copy, and fight the daemon for the
# cross-process GPU lock, which is what it was trying to avoid.
_QUERY_TIMEOUT_S = 30.0
_BATCH_TIMEOUT_S = 120.0
_CONTROL_TIMEOUT_S = 5.0
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


def _timeout(default: float) -> float:
    from memo.flags import flag_float

    value = flag_float("MEMO_EMBEDDER_CLIENT_TIMEOUT")
    if value is None:
        return default
    return value if value > 0 else default


def _require_daemon() -> bool:
    from memo.flags import flag_bool

    return flag_bool("MEMO_EMBEDDER_CLIENT_REQUIRE_DAEMON")


def _try_socket(
    state_dir: Path, payload: dict[str, Any], *, timeout: float
) -> dict[str, Any] | None:
    raw = connect_and_send(state_dir, payload, timeout=timeout)
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


def daemon_is_compatible(
    info: dict[str, Any] | None,
    *,
    expected_model: str,
    expected_dims: int,
) -> bool:
    """Return whether a daemon proves it owns the requested vector space.

    Model identity includes the immutable revision (``model@revision``).  An
    older daemon that reports only the repository name is therefore rejected:
    accepting its vectors would let an unknown snapshot contaminate an index
    stamped with the current pin.
    """
    if info is None:
        return False
    return info.get("model") == expected_model and info.get("dims") == expected_dims


def _accept_socket_response(
    decoded: dict[str, Any] | None,
    *,
    expected_model: str | None,
    expected_dims: int | None,
) -> dict[str, Any] | None:
    """Reject responses from a stale or differently configured daemon."""
    if decoded is None or (expected_model is None and expected_dims is None):
        return decoded
    actual_model = decoded.get("model")
    actual_dims = decoded.get("dims", decoded.get("dim"))
    if expected_model is not None and actual_model != expected_model:
        _log.warning(
            "embedder_client: daemon model mismatch (expected %s, got %s); falling back in-process",
            expected_model,
            actual_model,
        )
        return None
    if expected_dims is not None and actual_dims != expected_dims:
        _log.warning(
            "embedder_client: daemon dimension mismatch (expected %d, got %s); "
            "falling back in-process",
            expected_dims,
            actual_dims,
        )
        return None
    return decoded


def _inproc() -> Any:
    """Lazy-load an in-process embedder instance (singleton).

    Imports are deferred so callers without MLX (Linux CI) never trigger
    the import unless they actually reach the fallback path. `make_embedder`
    picks MLX (Apple Silicon) or the CPU sentence-transformers backend.
    """
    global _inproc_embedder
    with _inproc_lock:
        if _inproc_embedder is not None:
            return _inproc_embedder
        from memo.config import Config
        from memo.embedder_select import make_embedder

        _inproc_embedder = make_embedder(Config.from_env())
        return _inproc_embedder


# -- public API ------------------------------------------------------------


def embed_query(
    text: str,
    *,
    state_dir: Path | None = None,
    expected_model: str | None = None,
    expected_dims: int | None = None,
) -> list[float]:
    """Embed a single query string (asymmetric, query-prefixed).

    Returns a list of floats matching the embedder's configured
    dimensionality. Raises `RuntimeError` if the daemon is unreachable
    AND `MEMO_EMBEDDER_CLIENT_REQUIRE_DAEMON=1`.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("embed_query: empty text")
    resolved = _resolve_state_dir(state_dir)
    decoded = _accept_socket_response(
        _try_socket(
            resolved,
            {"op": "embed_query", "text": text},
            timeout=_timeout(_QUERY_TIMEOUT_S),
        ),
        expected_model=expected_model,
        expected_dims=expected_dims,
    )
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
    from memo.flags import flag_bool

    if flag_bool("MEMO_STRICT"):
        raise RuntimeError(
            "embed_query: daemon unreachable and MEMO_STRICT=1 disables in-process fallback. "
            "Start the daemon with 'memo recall-daemon start' or set MEMO_EMBEDDER_CLIENT_REQUIRE_DAEMON=1"
        )
    _log.warning(
        "embedder_client: daemon unreachable at %s, falling back to in-process "
        "(first call will be slow ~2s due to cold MLX load). "
        "To start the daemon: 'memo recall-daemon start'. "
        "To require daemon and fail fast: set MEMO_EMBEDDER_CLIENT_REQUIRE_DAEMON=1",
        resolved / "recall.sock",
    )
    return _inproc().embed_query(text)


def embed(
    texts: Sequence[str],
    *,
    state_dir: Path | None = None,
    expected_model: str | None = None,
    expected_dims: int | None = None,
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
    decoded = _accept_socket_response(
        _try_socket(
            resolved,
            {"op": "embed_batch", "texts": items},
            timeout=_timeout(_BATCH_TIMEOUT_S),
        ),
        expected_model=expected_model,
        expected_dims=expected_dims,
    )
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
    from memo.flags import flag_bool

    if flag_bool("MEMO_STRICT"):
        raise RuntimeError(
            "embed: daemon unreachable and MEMO_STRICT=1 disables in-process fallback. "
            "Start the daemon with 'memo recall-daemon start' or set MEMO_EMBEDDER_CLIENT_REQUIRE_DAEMON=1"
        )
    _log.warning(
        "embedder_client: daemon unreachable at %s, falling back to in-process "
        "(batch of %d items, first call will be slow ~2s due to cold MLX load). "
        "To start the daemon: 'memo recall-daemon start'. "
        "To require daemon and fail fast: set MEMO_EMBEDDER_CLIENT_REQUIRE_DAEMON=1",
        resolved / "recall.sock",
        len(items),
    )
    return _inproc().embed(items)


class SocketEmbedder:  # duck-type implements EmbedderBase (see memo.embed_base)
    """Drop-in for `MLXEmbedder` backed by the recall-daemon socket.

    Implements the `embed` / `embed_query` / `expected_dims` surface that
    `Memory` uses. Lets a long-lived process (the `memo-mcp` chat daemon)
    reuse the recall daemon's ALREADY-WARM embedder over the socket instead
    of loading its own copy in-process — keeping the chat daemon's resident
    footprint to just the synthesis model (+ reranker). Falls back in-process
    automatically (the module-level `embed`/`embed_query` do this), so it is
    never less available than `MLXEmbedder`.

    Gated in `Memory.__init__` by `MEMO_EMBEDDER_VIA_DAEMON=1`.
    """

    def __init__(
        self,
        expected_dims: int,
        *,
        expected_model: str,
        state_dir: Path | None = None,
    ) -> None:
        self.expected_dims = expected_dims
        self.expected_model = expected_model
        self._state_dir = state_dir

    @property
    def dims(self) -> int:
        return self.expected_dims

    @property
    def is_warm(self) -> bool:
        """Warm iff the recall daemon (which holds the model) answers a ping."""
        return ping(state_dir=self._state_dir) is not None

    def embed(self, inputs: Sequence[str]) -> list[list[float]]:
        return embed(
            inputs,
            state_dir=self._state_dir,
            expected_model=self.expected_model,
            expected_dims=self.expected_dims,
        )

    def embed_query(self, query: str) -> list[float]:
        return embed_query(
            query,
            state_dir=self._state_dir,
            expected_model=self.expected_model,
            expected_dims=self.expected_dims,
        )


def ping(*, state_dir: Path | None = None) -> dict[str, Any] | None:
    """Cheap warm-state probe. Returns `None` if the daemon is unreachable."""
    resolved = _resolve_state_dir(state_dir)
    return _try_socket(resolved, {"op": "ping"}, timeout=_timeout(_CONTROL_TIMEOUT_S))


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
    return _try_socket(resolved, {"op": "stats"}, timeout=_timeout(_CONTROL_TIMEOUT_S))


__all__ = ["daemon_is_compatible", "embed", "embed_query", "ping", "stats", "status"]
