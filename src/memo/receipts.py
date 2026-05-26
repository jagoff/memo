"""Best-effort Memflow operational receipts for memo write ops.

Memo owns the corpus. Memflow owns continuity. Receipts are how memo
tells memflow "I just did X" without orchestrating anything — memflow
can replay the breadcrumb trail across Macs and agents.

This module is a thin extraction of the pattern that landed in commit
8052e38 for `memo repo index`, generalised so `Memory.save / update /
delete / reindex` can emit too.

Default OFF. Set `MEMO_EMIT_RECEIPTS=1` to enable for save/update/
delete/reindex. (`memo repo index` keeps its own
`MEMO_MEMFLOW_RECEIPT=1` knob — they're independent.)

Wire: shells out to `memflow write fact <text> --meta key=value ...`.
Always returns a status dict (`{"ok": True, "path": ...}` or
`{"ok": False, "skipped": True, "reason": "..."}` or
`{"ok": False, "error": "..."}`) — never raises. Subprocess timeout
is 5s.

Synapse-originated writes pass `disabled=True` so the same memoria
does not get receipted twice (synapse keeps its own ledger).
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)
_TIMEOUT_S = 5.0
_TRUTHY = frozenset({"1", "true", "yes", "on"})


def _is_enabled() -> bool:
    raw = os.environ.get("MEMO_EMIT_RECEIPTS", "")
    return raw.strip().lower() in _TRUTHY


def _project_root() -> Path | None:
    raw = os.environ.get("MEMFLOW_PROJECT_ROOT")
    if raw:
        return Path(raw).expanduser()
    try:
        start = Path.cwd().expanduser()
    except OSError:
        return None
    for candidate in (start, *start.parents):
        if (candidate / ".memflow").is_dir():
            return candidate
    return None


def _binary() -> str | None:
    raw = os.environ.get("MEMO_MEMFLOW_BIN")
    if raw:
        return raw
    return shutil.which("memflow")


def _coerce(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    text = str(value).replace("\n", " ").strip()
    return text[:500]


def emit_receipt(
    operation: str,
    *,
    text: str,
    meta: dict[str, Any] | None = None,
    disabled: bool = False,
) -> dict[str, Any]:
    """Emit a Memflow receipt for one memo operation. Best-effort.

    Args:
        operation: short verb, e.g. ``save`` / ``update`` / ``delete``
            / ``reindex``. Mirrored into the receipt as both
            ``operation`` and ``topic=memo-<operation>``.
        text: human-readable one-line summary. Memflow's `write fact`
            takes this as the body.
        meta: extra key=value pairs persisted alongside the receipt.
            Values are coerced to short strings (≤500 chars, newlines
            stripped).
        disabled: explicit caller-side off-switch. Used by
            ``MemoSynapseBackend.remember()`` so synapse-originated
            writes do not double-receipt.

    Returns a status dict. Never raises; subprocess timeout is 5s.
    """
    if disabled:
        return {"ok": False, "skipped": True, "reason": "disabled"}
    if not _is_enabled():
        return {"ok": False, "skipped": True, "reason": "MEMO_EMIT_RECEIPTS not set"}

    project_root = _project_root()
    if project_root is None:
        return {"ok": False, "skipped": True, "reason": "memflow project root not found"}
    memflow_bin = _binary()
    if memflow_bin is None:
        return {"ok": False, "skipped": True, "reason": "memflow binary not found"}

    full_meta: dict[str, Any] = {
        "client": "memo",
        "topic": f"memo-{operation}",
        "operation": operation,
    }
    if meta:
        for key, value in meta.items():
            if key in full_meta:
                continue
            full_meta[key] = value

    command: list[str] = [memflow_bin, "write", "fact", text]
    for key, value in full_meta.items():
        command.extend(["--meta", f"{key}={_coerce(value)}"])

    env = dict(os.environ)
    env["MEMFLOW_PROJECT_ROOT"] = str(project_root)

    try:
        proc = subprocess.run(
            command,
            cwd=str(project_root),
            env=env,
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_S,
            check=False,
        )
    except Exception as exc:
        _log.debug("receipts.emit_receipt: subprocess failed: %s", exc)
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or f"memflow exited {proc.returncode}").strip()
        return {"ok": False, "error": detail[:500]}

    path = ""
    if proc.stdout.strip():
        path = proc.stdout.strip().splitlines()[0]
    return {"ok": True, "path": path} if path else {"ok": True}


__all__ = ["emit_receipt"]
