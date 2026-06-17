"""Lazy subprocess client to the `synapse` CLI.

Read-only today. Used by `Memory.save()` to respect Synapse's
freeze-write protocol: if a `RealityConflict` with
`freeze_write=true` (and lifecycle ∉ {resolved, archived}) overlaps
the topic of a pending write, the save is refused so the agent can
review the conflict before mutating state.

Two safety stances, picked per call:

* **best-effort (default)** — `list_conflicts(strict=False)` returns `[]`
  on any failure (missing binary, non-zero exit, malformed JSON, timeout).
  memo runs perfectly without synapse; the freeze check is opt-in and a
  silent `[]` means "no information, proceed".
* **fail-closed (`strict=True`)** — the same failures raise
  `SynapseUnavailable` so a caller that treats the freeze gate as a real
  safety boundary can refuse the write rather than silently disarm it.
  Only `Memory.save()` under `MEMO_RESPECT_SYNAPSE_FREEZE=1` opts in.

A *missing binary* is never a fail-closed condition: "synapse not installed"
is a deliberate, stable choice to run memo standalone, not an outage. Only a
present-but-unresponsive synapse yields `SynapseUnavailable`.

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

try:
    from consciousness_contracts import BackendError, run_json

    _HAS_CONTRACTS_SUBPROCESS = True
except ImportError:  # graceful degradation — contracts is an optional dep
    _HAS_CONTRACTS_SUBPROCESS = False

_log = logging.getLogger(__name__)
_DEFAULT_TIMEOUT_S = 8.0
_BLOCKING_STATES: frozenset[str] = frozenset({"detected", "acknowledged"})


class SynapseUnavailable(RuntimeError):
    """The synapse probe could not complete, so the answer is *unknown*.

    Raised only in ``strict=True`` mode (timeout, non-zero exit, or bad JSON
    from an installed `synapse` binary). Subclasses ``RuntimeError`` so legacy
    ``except RuntimeError`` / ``except Exception`` sites keep catching it, while
    fail-closed callers can catch this precise type to distinguish "synapse
    couldn't answer" from "synapse said: no conflicts".
    """


def _executable() -> str | None:
    """Return path to `synapse` binary, or None if missing.

    Override with `MEMO_SYNAPSE_EXECUTABLE=/path/to/synapse` (useful in
    tests and for non-PATH installs).
    """
    from memo.flags import flag_str

    override = flag_str("MEMO_SYNAPSE_EXECUTABLE")
    if override:
        return override if os.path.exists(override) else None
    return shutil.which("synapse")


def _timeout() -> float:
    from memo.flags import flag_float

    value = flag_float("MEMO_SYNAPSE_CLIENT_TIMEOUT")
    if value is None:
        return _DEFAULT_TIMEOUT_S
    return value if value > 0 else _DEFAULT_TIMEOUT_S


def is_available() -> bool:
    """True if a `synapse` binary is on PATH (or env-overridden)."""
    return _executable() is not None


def _probe_json(
    binary: str,
    args: Sequence[str],
    *,
    op: str,
    query: str,
    trace_id: str,
    timeout: float,
) -> Any:
    """Run `synapse <args> --json` and return the parsed payload.

    Always raises :class:`SynapseUnavailable` on timeout / non-zero exit /
    malformed JSON so each caller decides whether that's fatal (fail-closed)
    or ignorable (best-effort). Uses the shared `consciousness_contracts`
    typed subprocess driver when present, falling back to stdlib otherwise so
    memo still runs on a clean install.
    """
    env = dict(os.environ)
    if trace_id:
        env["SYNAPSE_TRACE_ID"] = trace_id

    if _HAS_CONTRACTS_SUBPROCESS:
        try:
            return run_json(
                binary,
                list(args),
                timeout=timeout,
                env=env,
                backend_name="synapse",
                trace_id=trace_id,
            )
        except BackendError as exc:
            _log.warning("synapse_client.%s: %s (query=%r)", op, exc, query)
            raise SynapseUnavailable(str(exc)) from exc

    # Stdlib fallback (contracts absent): same raise-on-failure contract.
    try:
        proc = subprocess.run(
            [binary, *args],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        _log.warning("synapse_client.%s: timeout after %.1fs (query=%r)", op, exc.timeout, query)
        raise SynapseUnavailable(f"timeout after {exc.timeout}s") from exc
    except (FileNotFoundError, OSError) as exc:
        _log.debug("synapse_client.%s: subprocess failed: %s", op, exc)
        raise SynapseUnavailable(str(exc)) from exc
    if proc.returncode != 0:
        _log.warning(
            "synapse_client.%s: exit=%s (query=%r) stderr=%r",
            op,
            proc.returncode,
            query,
            proc.stderr[:200],
        )
        raise SynapseUnavailable(f"exit {proc.returncode}: {proc.stderr[:200]}")
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        _log.warning("synapse_client.%s: bad JSON: %s", op, exc)
        raise SynapseUnavailable(f"bad JSON: {exc}") from exc


def list_conflicts(
    query: str = "",
    *,
    k: int = 5,
    trace_id: str = "",
    timeout: float | None = None,
    strict: bool = False,
) -> list[dict[str, Any]]:
    """Return raw conflict dicts from `synapse conflicts <query> --json`.

    With ``strict=False`` (default) returns `[]` on any failure — callers
    MUST treat empty as "no information", not "no conflicts" (see
    `has_blocking_freeze`). With ``strict=True`` the same failures raise
    :class:`SynapseUnavailable` so a fail-closed caller can refuse the write
    instead of silently treating an outage as "all clear". A missing binary
    is never a failure here (returns `[]`): synapse-not-installed is a
    standalone-memo choice, not an outage.
    """
    binary = _executable()
    if binary is None:
        return []
    args: list[str] = ["conflicts"]
    if query:
        args.append(query)
    args.extend(["--k", str(max(1, int(k))), "--json"])
    try:
        payload = _probe_json(
            binary,
            args,
            op="list_conflicts",
            query=query,
            trace_id=trace_id,
            timeout=timeout or _timeout(),
        )
    except SynapseUnavailable:
        if strict:
            raise
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
    fall back to a memo-only briefing — this reader is always best-effort.

    The packet shape (top-level keys): ``schema``, ``trace_id``,
    ``status`` (ready/partial/degraded), ``present_state`` (memflow
    items), ``deep_memory`` (memo items), ``reality_conflicts``,
    ``attention`` / ``attention_state``, ``executive_summary``,
    ``backends``.
    """
    binary = _executable()
    if binary is None:
        return None
    args: list[str] = ["packet"]
    if query:
        args.extend(["--query", query])
    args.extend(["-k", str(max(1, int(k))), "--json"])
    try:
        payload = _probe_json(
            binary,
            args,
            op="get_packet",
            query=query,
            trace_id=trace_id,
            timeout=timeout or _timeout(),
        )
    except SynapseUnavailable:
        return None
    return payload if isinstance(payload, dict) else None


__all__ = [
    "SynapseUnavailable",
    "get_packet",
    "has_blocking_freeze",
    "is_available",
    "list_conflicts",
]
