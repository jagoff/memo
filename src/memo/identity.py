"""Stable identity for memo: which MACHINE, which SESSION/terminal.

The sync layer needs a stable per-machine id to attribute git commits, decide
same-machine vs cross-machine, and own the machine-level git coordinator lock.
The session/terminal id makes each open agent session addressable — so memflow
can reference "terminal X" for a task in future. memo only EXPOSES this; the
addressing/coordination is memflow's job (YAGNI here).

Machine identity is decoupled from the trinity by design: ``hostname`` is the
shared match key every tool computes identically; memo's persisted
``cfg.device_id`` (``state_dir/.device_id``) adds uniqueness. No hard dependency
on ``consciousness_contracts`` — its ``IdentityClaim`` is an assertion record,
not a machine-id provider.
"""

from __future__ import annotations

import os
import socket
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Identity:
    """Who this memo process is. Immutable snapshot resolved at use time."""

    machine_id: str  # stable, unique per machine (persisted device_id)
    hostname: str  # cross-tool match key (every trinity tool computes it the same)
    session_id: str | None  # the open agent session, if the client supplied one
    terminal: str | None  # controlling TTY, when attached — for future addressing

    @property
    def label(self) -> str:
        """Human/commit-friendly label, e.g. ``MacBook-Pro·a1b2c3d4``."""
        s = self.hostname
        if self.session_id:
            s = f"{s}·{self.session_id[:8]}"
        return s

    def as_dict(self) -> dict[str, Any]:
        return {
            "machine_id": self.machine_id,
            "hostname": self.hostname,
            "session_id": self.session_id,
            "terminal": self.terminal,
            "label": self.label,
        }


def _hostname() -> str:
    try:
        return socket.gethostname().strip() or "unknown-host"
    except OSError:
        return "unknown-host"


def _session_id() -> str | None:
    # Clients pass their session id via env (memo's own var wins). Best-effort:
    # the machine id is what sync correctness depends on; session is provenance.
    for k in ("MEMO_SESSION_ID", "CLAUDE_SESSION_ID", "CLAUDE_CODE_SESSION_ID"):
        v = os.environ.get(k)
        if v and v.strip():
            return v.strip()
    return None


def _terminal() -> str | None:
    for fd in (0, 1, 2):
        try:
            return os.ttyname(fd)
        except OSError:
            continue
    return None


def current(cfg: Any) -> Identity:
    """Resolve this process's identity. ``cfg.device_id`` is the persisted stable
    machine id; ``hostname`` is the cross-tool match key."""
    return Identity(
        machine_id=str(getattr(cfg, "device_id", "") or "unknown"),
        hostname=_hostname(),
        session_id=_session_id(),
        terminal=_terminal(),
    )
