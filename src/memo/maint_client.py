"""Client for the maintenance daemon — socket-first, None on absence.

Mirrors the other daemon clients: returns ``None`` on any
missing/refused/timed-out socket so `Memory.consolidate` falls back to
running the synthesis LLM in-process. Framing via shared ``embed_protocol``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from memo import embed_protocol
from memo.maint_server import _socket_path

_log = logging.getLogger(__name__)

# Consolidation runs an LLM over many clusters — generous timeout.
_DEFAULT_TIMEOUT_S = 600.0
_PING_TIMEOUT_S = 0.5


def _resolve_state_dir(state_dir: Path | None) -> Path:
    if state_dir is not None:
        return state_dir
    from memo.config import Config

    return Config.from_env().state_dir


def ping(*, state_dir: Path | None = None) -> dict[str, Any] | None:
    sock = _socket_path(_resolve_state_dir(state_dir))
    return embed_protocol.send_request(sock, {"op": "ping"}, timeout=_PING_TIMEOUT_S)


def consolidate(
    *,
    threshold: float = 0.85,
    max_clusters: int = 50,
    type_: str | None = None,
    state_dir: Path | None = None,
    timeout: float = _DEFAULT_TIMEOUT_S,
) -> list[dict[str, Any]] | None:
    """Run consolidation propose on the daemon. Returns the proposal list,
    or None if the daemon is unreachable (caller runs it in-process)."""
    sock = _socket_path(_resolve_state_dir(state_dir))
    resp = embed_protocol.send_request(
        sock,
        {"op": "consolidate", "params": {"threshold": threshold, "max_clusters": max_clusters, "type_": type_}},
        timeout=timeout,
    )
    if not resp or "error" in resp:
        if resp and "error" in resp:
            _log.warning("maint daemon consolidate error: %s", resp["error"])
        return None
    proposals = resp.get("proposals")
    return proposals if isinstance(proposals, list) else None


__all__ = ["consolidate", "ping"]
