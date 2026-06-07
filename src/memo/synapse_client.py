"""Lazy subprocess client to the `synapse` CLI.

Read-only today. Used by `Memory.save()` to respect Synapse's
freeze-write protocol: if a `RealityConflict` with
`freeze_write=true` (and lifecycle ∉ {resolved, archived}) overlaps
the topic of a pending write, the save is refused so the agent can
review the conflict before mutating state.

All calls are best-effort and never raise on missing binary /
non-zero exit / malformed JSON: this module is OPT-IN
(`MEMO_RESPECT_SYNAPSE_FREEZE=1` or per-save kwarg). The boundary is
intentionally loose — memo runs perfectly without synapse.

Wire: shells out to `synapse conflicts <query> --json` which returns
`{schema, query, conflicts: [{conflict_id, freeze_write,
lifecycle_state, summary, severity, involved_backends, ...}]}`.
See `synapse/src/synapse/cli.py:_cmd_conflicts`.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from collections.abc import Sequence
from typing import Any

_log = logging.getLogger(__name__)
_DEFAULT_TIMEOUT_S = 8.0
_BLOCKING_STATES: frozenset[str] = frozenset({"detected", "acknowledged"})


def _executable() -> str | None:
    """Return path to `synapse` binary, or None if missing.

    Override with `MEMO_SYNAPSE_EXECUTABLE=/path/to/synapse` (useful in
    tests and for non-PATH installs).
    """
    override = os.environ.get("MEMO_SYNAPSE_EXECUTABLE")
    if override:
        return override if os.path.exists(override) else None
    return shutil.which("synapse")


def _timeout() -> float:
    raw = os.environ.get("MEMO_SYNAPSE_CLIENT_TIMEOUT")
    if not raw:
        return _DEFAULT_TIMEOUT_S
    try:
        value = float(raw)
    except ValueError:
        return _DEFAULT_TIMEOUT_S
    return value if value > 0 else _DEFAULT_TIMEOUT_S


def is_available() -> bool:
    """True if a `synapse` binary is on PATH (or env-overridden)."""
    return _executable() is not None


def list_conflicts(
    query: str = "",
    *,
    k: int = 5,
    trace_id: str = "",
    timeout: float | None = None,
) -> list[dict[str, Any]]:
    """Return raw conflict dicts from `synapse conflicts <query> --json`.

    Returns `[]` on any failure (missing binary, non-zero exit, parse
    error, timeout). Callers MUST treat empty list as "no information"
    rather than "no conflicts" — see `has_blocking_freeze()` for the
    safety-aware reduction.
    """
    binary = _executable()
    if binary is None:
        return []
    args: list[str] = [binary, "conflicts"]
    if query:
        args.append(query)
    args.extend(["--k", str(max(1, int(k))), "--json"])
    env = dict(os.environ)
    if trace_id:
        env["SYNAPSE_TRACE_ID"] = trace_id
    try:
        proc = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout or _timeout(),
            env=env,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        _log.warning(
            "synapse_client.list_conflicts: timeout after %.1fs (query=%r)",
            exc.timeout,
            query,
        )
        return []
    except (FileNotFoundError, OSError) as exc:
        _log.debug("synapse_client.list_conflicts: subprocess failed: %s", exc)
        return []
    if proc.returncode != 0:
        _log.warning(
            "synapse_client.list_conflicts: exit=%s (query=%r) stderr=%r",
            proc.returncode,
            query,
            proc.stderr[:200],
        )
        return []
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        _log.warning("synapse_client.list_conflicts: bad JSON: %s", exc)
        return []
    rows = payload.get("conflicts") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return []
    return [r for r in rows if isinstance(r, dict)]


def has_blocking_freeze(
    conflicts: Sequence[dict[str, Any]],
) -> tuple[bool, dict[str, Any] | None]:
    """Return (is_blocked, first_blocking_conflict_or_None).

    A conflict blocks a write iff `freeze_write` is truthy AND its
    `lifecycle_state` is one of {detected, acknowledged}. `resolved` /
    `archived` states do not block.
    """
    for row in conflicts:
        if not row.get("freeze_write"):
            continue
        state = str(row.get("lifecycle_state") or "detected").strip().lower()
        if state in _BLOCKING_STATES:
            return True, dict(row)
    return False, None


def get_packet(
    query: str = "",
    *,
    k: int = 5,
    trace_id: str = "",
    timeout: float | None = None,
) -> dict[str, Any] | None:
    """Return the synapse `ConsciousnessPacket.v2` dict, or None on failure.

    Subprocess-calls ``synapse packet --query Q -k N --json`` and parses
    the response. Returns ``None`` on any failure (missing binary,
    non-zero exit, parse error, timeout) so callers can gracefully
    fall back to a memo-only briefing.

    The packet shape (top-level keys): ``schema``, ``trace_id``,
    ``status`` (ready/partial/degraded), ``present_state`` (memflow
    items), ``deep_memory`` (memo items), ``reality_conflicts``,
    ``attention`` / ``attention_state``, ``executive_summary``,
    ``backends``.
    """
    binary = _executable()
    if binary is None:
        return None
    args: list[str] = [binary, "packet"]
    if query:
        args.extend(["--query", query])
    args.extend(["-k", str(max(1, int(k))), "--json"])
    env = dict(os.environ)
    if trace_id:
        env["SYNAPSE_TRACE_ID"] = trace_id
    try:
        proc = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout or _timeout(),
            env=env,
            check=False,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
        _log.debug("synapse_client.get_packet: subprocess failed: %s", exc)
        return None
    if proc.returncode != 0:
        _log.debug(
            "synapse_client.get_packet: exit=%s stderr=%r",
            proc.returncode,
            proc.stderr[:200],
        )
        return None
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        _log.debug("synapse_client.get_packet: bad JSON: %s", exc)
        return None
    return payload if isinstance(payload, dict) else None


__all__ = ["get_packet", "has_blocking_freeze", "is_available", "list_conflicts"]
