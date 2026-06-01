"""Shared embedder wire protocol — the frozen contract between the recall
daemon and every client (memo, memflow, synapse).

This module is the single normative spec for the AF_UNIX, newline-delimited
JSON framing used by ``recall_server.py``. It is **stdlib-only** and imports
no project code, so it can be vendored byte-identically into peer repos
(memflow keeps a copy at ``memflow/embed_protocol.py``) without creating a
cross-package dependency — memflow must not import memo (enforced by its
``test_architecture_boundaries``). The two copies are kept in lock-step by a
contract test in each repo (``test_embed_protocol_contract.py``).

Wire contract
-------------
One JSON object per line (``\\n``-terminated), request in → response out.

Requests::

    {"op": "ping"}
    {"op": "embed_query", "text": "..."}
    {"op": "embed_batch", "texts": ["...", ...]}

Responses (fields frozen — clients may rely on these keys)::

    ping:        {"ok": true, "model": "...", "dims": N, ...}
    embed_query: {"vector": [...],   "dim": N, "dims": N, "model": "..."}
    embed_batch: {"vectors": [[...]], "dim": N, "dims": N, "model": "..."}
    on error:    {"error": "<message>"}

``dim`` and ``dims`` are emitted together for back-compat: historically the
embed ops returned ``dim`` and ``ping`` returned ``dims``; both are now
present on every embed response so clients can read either. Do not remove
either key without bumping the contract test in both repos.

Layering
--------
Two send helpers share one socket round-trip:

* :func:`send_request_line` returns the raw JSON line (``str``) — used by the
  recall path, whose response is injected verbatim and must not be re-parsed.
* :func:`send_request` parses that line into a ``dict`` — the convenience the
  embed clients use.

Both return ``None`` on any failure (missing socket, refused, timeout, runaway
response) so callers transparently fall back (to keyword retrieval in memflow,
to in-process MLX in memo). Socket resolution is per-repo policy and is NOT
frozen here: memo passes a ``Config``-derived path; memflow uses the env-based
:func:`default_socket_path`.
"""

from __future__ import annotations

import json
import socket as _socket
from pathlib import Path
from typing import Any

# -- frozen op names -------------------------------------------------------
OP_PING = "ping"
OP_EMBED_QUERY = "embed_query"
OP_EMBED_BATCH = "embed_batch"

# -- frozen response field names -------------------------------------------
FIELD_VECTOR = "vector"
FIELD_VECTORS = "vectors"
FIELD_DIM = "dim"
FIELD_DIMS = "dims"
FIELD_MODEL = "model"
FIELD_OK = "ok"
FIELD_ERROR = "error"

# Cap a single request/response line. Batch embed responses (many 1024-dim
# float vectors) can be large, so this is generous — far above any legitimate
# line, but bounded so a peer that never sends a newline can't make us buffer
# unboundedly.
MAX_LINE_BYTES = 64 * 1024 * 1024

DEFAULT_TIMEOUT_S = 5.0
PING_TIMEOUT_S = 0.5

_RECV_CHUNK = 65536


def default_socket_path() -> Path:
    """Resolve the recall socket from the environment (no project imports).

    Priority: ``MEMFLOW_EMBED_SOCKET`` (explicit) → ``MEMO_STATE_DIR``/recall.sock
    → memo's default state dir (``~/.local/share/memo/recall.sock``).

    This is the env-only resolution memflow uses. memo resolves its state dir
    through ``Config`` instead and passes the path explicitly to the send
    helpers, so it does not call this.
    """
    import os

    explicit = os.environ.get("MEMFLOW_EMBED_SOCKET", "").strip()
    if explicit:
        return Path(explicit).expanduser()
    state_dir = os.environ.get("MEMO_STATE_DIR", "").strip()
    if state_dir:
        return Path(state_dir).expanduser() / "recall.sock"
    return Path.home() / ".local" / "share" / "memo" / "recall.sock"


def encode_request(op: str, **fields: Any) -> bytes:
    """Frame a request as one UTF-8, newline-terminated JSON line."""
    payload = {"op": op, **fields}
    return (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")


def encode_payload(payload: dict[str, Any]) -> bytes:
    """Frame an already-built payload dict (must carry its own ``op``)."""
    return (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")


def send_request_line(
    sock_path: Path,
    payload: dict[str, Any],
    *,
    timeout: float = DEFAULT_TIMEOUT_S,
    max_bytes: int = MAX_LINE_BYTES,
) -> str | None:
    """Send one JSON-line request; return the raw response line (no trailing
    newline) or ``None`` on any failure.

    The raw-string form preserves the recall path's contract, whose response
    is injected verbatim and must not be re-serialized.
    """
    if not sock_path.exists():
        return None
    try:
        with _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout)
            sock.connect(str(sock_path))
            sock.sendall(encode_payload(payload))
            buf = b""
            while b"\n" not in buf:
                chunk = sock.recv(_RECV_CHUNK)
                if not chunk:
                    break
                buf += chunk
                if len(buf) >= max_bytes:
                    break  # runaway response — stop buffering
    except (FileNotFoundError, ConnectionRefusedError, OSError, TimeoutError):
        return None
    line = buf.decode("utf-8", errors="replace").strip()
    return line if line else None


def send_request(
    sock_path: Path,
    payload: dict[str, Any],
    *,
    timeout: float = DEFAULT_TIMEOUT_S,
    max_bytes: int = MAX_LINE_BYTES,
) -> dict[str, Any] | None:
    """Send one JSON-line request; return the parsed response ``dict`` or
    ``None`` on any failure (including non-JSON or non-object responses)."""
    line = send_request_line(sock_path, payload, timeout=timeout, max_bytes=max_bytes)
    if not line:
        return None
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


__all__ = [
    "DEFAULT_TIMEOUT_S",
    "FIELD_DIM",
    "FIELD_DIMS",
    "FIELD_ERROR",
    "FIELD_MODEL",
    "FIELD_OK",
    "FIELD_VECTOR",
    "FIELD_VECTORS",
    "MAX_LINE_BYTES",
    "OP_EMBED_BATCH",
    "OP_EMBED_QUERY",
    "OP_PING",
    "PING_TIMEOUT_S",
    "default_socket_path",
    "encode_payload",
    "encode_request",
    "send_request",
    "send_request_line",
]
