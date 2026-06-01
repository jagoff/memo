"""Client for the ingest worker daemon — socket-first, None on absence.

Mirrors `embedder_client`'s contract: every call returns ``None`` if the
ingest daemon socket is missing/refused/timed out, so callers transparently
fall back to running the batch op in-process. Framing is delegated to the
shared ``embed_protocol`` so all of memo's daemons speak one wire format.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from memo import embed_protocol
from memo.ingest_server import _socket_path

_log = logging.getLogger(__name__)

_DEFAULT_TIMEOUT_S = 5.0
_PING_TIMEOUT_S = 0.5


def _resolve_state_dir(state_dir: Path | None) -> Path:
    if state_dir is not None:
        return state_dir
    from memo.config import Config

    return Config.from_env().state_dir


def ping(*, state_dir: Path | None = None) -> dict[str, Any] | None:
    """Probe the daemon. Returns its status dict, or None if unreachable."""
    sock = _socket_path(_resolve_state_dir(state_dir))
    return embed_protocol.send_request(sock, {"op": "ping"}, timeout=_PING_TIMEOUT_S)


def enqueue(
    kind: str,
    payload: dict[str, Any],
    *,
    state_dir: Path | None = None,
    timeout: float = _DEFAULT_TIMEOUT_S,
) -> str | None:
    """Enqueue a batch job. Returns its job_id, or None if the daemon is
    unreachable (caller should run the op in-process)."""
    sock = _socket_path(_resolve_state_dir(state_dir))
    resp = embed_protocol.send_request(
        sock, {"op": "enqueue", "kind": kind, "payload": payload}, timeout=timeout
    )
    if not resp or "error" in resp:
        if resp and "error" in resp:
            _log.warning("ingest daemon enqueue error: %s", resp["error"])
        return None
    job_id = resp.get("job_id")
    return str(job_id) if job_id else None


def status(job_id: str, *, state_dir: Path | None = None, timeout: float = _DEFAULT_TIMEOUT_S) -> dict[str, Any] | None:
    """Poll a job's state. Returns its status dict, or None if unreachable."""
    sock = _socket_path(_resolve_state_dir(state_dir))
    resp = embed_protocol.send_request(sock, {"op": "status", "job_id": job_id}, timeout=timeout)
    return resp


__all__ = ["enqueue", "ping", "status"]
