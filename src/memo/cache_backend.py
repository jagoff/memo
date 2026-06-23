"""Backing-store clients for cache-tier mode.

When `MEMO_CACHE_MODE != off`, memo's local store is a cache in front of an
authoritative backing store. This module provides the concrete
`CacheBackend` implementations the `CacheManager` (see `cache.py`) and
`Memory` talk to:

  - `MemflowBackend` — shells out to the `memflow` CLI, mirroring the
    established integration pattern in `receipts.py` / `synapse_client.py`
    (memo does not embed an MCP client; it calls the binary). Push uses
    `memflow write fact`; read-through uses `memflow ask --json --no-capture`.
  - `NullBackend` — no-op, used when `MEMO_CACHE_BACKEND=none` or the
    `memflow` binary isn't found. Read-through returns nothing; push reports
    failure so dirty entries are kept (never silently dropped).

Field mapping in `fetch()` is best-effort against memflow's `ask --json`
schema (`matches_raw` / `citations`); keys are read defensively so a schema
drift degrades to "fewer fields" rather than a crash.

Subprocess calls are bounded by `MEMO_SYNAPSE_CLIENT_TIMEOUT` (default 5s)
and never raise — a backend hiccup degrades to a local-only result.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from memo.flags import flag_float

_log = logging.getLogger(__name__)
_DEFAULT_TIMEOUT_S = 5.0


def _timeout() -> float:
    return flag_float("MEMO_SYNAPSE_CLIENT_TIMEOUT") or _DEFAULT_TIMEOUT_S


def _binary() -> str | None:
    from memo.flags import flag_str

    raw = flag_str("MEMO_MEMFLOW_BIN")
    if raw:
        return raw
    return shutil.which("memflow")


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


def _coerce_meta(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value).replace("\n", " ").strip()[:500]


class NullBackend:
    """No-op backing store. push fails (so dirty entries are never dropped),
    fetch returns nothing."""

    def push(self, record: Any) -> bool:
        return False

    def fetch(self, query: str, *, limit: int = 10) -> list[dict[str, Any]]:
        return []

    def has_current(self, id_: str, body_hash: str) -> bool:
        return False


class MemflowBackend:
    """Memflow-backed authoritative store, reached via the `memflow` CLI."""

    def __init__(self) -> None:
        self._bin = _binary()
        self._root = _project_root()

    @property
    def available(self) -> bool:
        return self._bin is not None and self._root is not None

    def _run(self, args: list[str]) -> subprocess.CompletedProcess[str] | None:
        if not self.available:
            return None
        env = dict(os.environ)
        env["MEMFLOW_PROJECT_ROOT"] = str(self._root)
        try:
            return subprocess.run(
                [self._bin, *args],  # type: ignore[list-item]
                cwd=str(self._root),
                env=env,
                capture_output=True,
                text=True,
                timeout=_timeout(),
                check=False,
            )
        except Exception as exc:
            _log.debug("memflow backend subprocess failed: %s", exc)
            return None

    # -- write side --------------------------------------------------------

    def push(self, record: Any) -> bool:
        """Persist a memory to memflow via `memflow write fact`.

        Carries the memo id/type/title as meta so a later fetch can map the
        memflow memory back to its memo origin. Returns True on exit 0.
        """
        body = getattr(record, "body", "") or ""
        rid = getattr(record, "id", "") or ""
        rtype = getattr(record, "type", "") or "note"
        title = getattr(record, "title", "") or ""
        tags = getattr(record, "tags", None) or []
        text = body if body.strip() else title
        if not text.strip():
            return False
        args = ["write", "fact", text]
        for key, value in (
            ("memo_id", rid),
            ("memo_type", rtype),
            ("memo_title", title),
            ("memo_tags", ",".join(tags)),
            ("source", "memo-cache"),
        ):
            args.extend(["--meta", f"{key}={_coerce_meta(value)}"])
        proc = self._run(args)
        if proc is None:
            return False
        if proc.returncode != 0:
            _log.debug(
                "memflow push exit=%s: %s", proc.returncode, (proc.stderr or "").strip()[:200]
            )
            return False
        return True

    # -- read side ---------------------------------------------------------

    def fetch(self, query: str, *, limit: int = 10) -> list[dict[str, Any]]:
        """Read-through: pull candidate memories from memflow via
        `memflow ask --json --no-capture`.

        `--no-capture` keeps the read from being recorded as a memflow turn
        (a cache fill is not a user interaction). Returns a list of dicts
        with best-effort fields {id, title, type, body, score}; an empty
        list on miss / unavailable backend.
        """
        if not query.strip():
            return []
        proc = self._run(["ask", query, "-k", str(limit), "--json", "--no-capture"])
        if proc is None or proc.returncode != 0 or not proc.stdout.strip():
            return []
        try:
            data = json.loads(proc.stdout)
        except json.JSONDecodeError:
            return []
        if not data.get("found"):
            return []
        raw = data.get("matches_raw") or data.get("citations") or []
        out: list[dict[str, Any]] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            mapped = self._map_match(item)
            if mapped:
                out.append(mapped)
        return out

    @staticmethod
    def _map_match(item: dict[str, Any]) -> dict[str, Any] | None:
        """Best-effort map a memflow match/citation to a memo candidate.

        memflow's schema isn't guaranteed stable, so every field is read
        defensively across plausible spellings. Body is required (nothing to
        materialize without it). Meta carried by memo's own push (`memo_*`)
        is preferred when present so a round-tripped memory keeps its origin.
        """
        meta = item.get("meta") or item.get("metadata") or {}
        if not isinstance(meta, dict):
            meta = {}
        body = (
            item.get("text") or item.get("body") or item.get("content") or item.get("answer") or ""
        )
        if not str(body).strip():
            return None
        _body_stripped = str(body).strip()
        title = (
            meta.get("memo_title")
            or item.get("title")
            or (_body_stripped.splitlines()[0][:80] if _body_stripped else "Untitled")
        )
        rtype = meta.get("memo_type") or item.get("type") or item.get("kind") or "note"
        # Prefer memo's own id so a round-trip is idempotent; else derive a
        # stable id from the body so repeated fetches don't duplicate.
        rid = (
            meta.get("memo_id")
            or item.get("memo_id")
            or item.get("id")
            or hashlib.sha256(str(body).encode("utf-8")).hexdigest()[:32]
        )
        tags_raw = meta.get("memo_tags") or ""
        tags = [t for t in str(tags_raw).split(",") if t.strip()]
        score = item.get("score") or item.get("confidence")
        return {
            "id": str(rid),
            "title": str(title),
            "type": str(rtype),
            "body": str(body),
            "tags": tags,
            "score": float(score) if isinstance(score, (int, float)) else None,
            "from_backend": True,
        }

    def has_current(self, id_: str, body_hash: str) -> bool:
        # Coherence revalidation against memflow is a follow-up; conservatively
        # report "not verified present" so callers never assume a flush is
        # unnecessary based on this. (Eviction already keys off the local
        # dirty flag, not this method.)
        return False


def make_backend(backend: str) -> Any:
    """Factory: build the configured backend, or NullBackend when the
    backend is `none` / unknown / unavailable."""
    name = (backend or "none").strip().lower()
    if name == "memflow":
        mf = MemflowBackend()
        if mf.available:
            return mf
        _log.warning(
            "cache backend 'memflow' selected but binary/project root "
            "not found; falling back to NullBackend (local-only)."
        )
        return NullBackend()
    if name == "vault":
        # Remote-vault backend reuses sync.py's SyncManager; not yet wired as
        # a CacheBackend. Until then, no-op rather than silently mis-routing.
        _log.warning("cache backend 'vault' not yet implemented; using NullBackend.")
        return NullBackend()
    return NullBackend()
