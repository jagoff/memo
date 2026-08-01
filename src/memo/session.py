"""Session snapshots — checkpoint where you left off so you can resume.

Companion to `capture.py`. Where capture extracts long-lived insights
into the memory archive, this module persists *short-lived* "what was
I working on" state: cwd, branch, last user prompt, last todo, last
plan. Survives a crashed/closed Claude Code session so the next
SessionStart can show a picker of recent work.

## Why a sidecar JSON store, not a memory

A session snapshot is ephemeral by nature: 90% of the value lives in
the first 24h, decaying to ~zero past a week. Pulling them through
the embedder pipeline (cost: ~200ms per Stop, plus a vec row that
becomes noise in unrelated queries) buys nothing — sessions are
looked up by *recency*, not semantic similarity. So we keep them in
their own dir under `state_dir/sessions/`, one JSON per session,
mtime-sorted on read.

## Schema

```
{
  "session_id": str,              # canonical Claude Code session id
  "cwd": str,                     # absolute path of the project
  "project": str,                 # basename of cwd, for the picker
  "branch": str | null,           # current git branch
  "head_commit": str | null,      # short oneline of HEAD
  "modified_files": list[str],    # `git status --porcelain` paths (truncated)
  "transcript_path": str | null,  # ~/.claude/projects/.../<sid>.jsonl
  "last_user_msg": str | null,    # first 240 chars of the most recent user msg
  "last_assistant_tail": str | null,  # last 200 chars of last assistant message
  "prompt_trail": list[str],      # ring buffer of last 5 user prompts (100 chars each)
                                  # updated on UserPromptSubmit — survives crashes
  "running_summary": str | null,  # LLM-generated arc summary (what/decided/pending)
                                  # updated on Stop every ≥3 turns
  "summary_turn": int,            # turn_count when running_summary was last written
  "summary": str | null,          # human-friendly label; defaults to last_user_msg[:80]
  "created": str,                 # ISO-8601 first checkpoint
  "updated": str,                 # ISO-8601 last checkpoint
  "turn_count": int               # incremented on every Stop
}
```

## Idempotence

`checkpoint()` is keyed by `session_id`. If a file already exists for
that id, fields are *merged* (created stays, turn_count++, the rest
overwritten). New session → new file.

## LRU cap

`prune_lru(cap=50)` deletes the oldest sessions by `updated` once
there are more than `cap` files. Called from `checkpoint()` so the
dir size is self-bounding without a separate daemon.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import re
import threading
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from memo.atomic_io import atomic_write_text, authority_write_lock
from memo.flags import flag_int
from memo.session_sources import (
    _COMMAND_WRAPPER_PREFIXES,
    _extract_text,
    _strip_command_wrappers,
    gather_git_state,
    is_command_noise,
    read_last_assistant_tail,
    read_last_user_msg,
)

if TYPE_CHECKING:
    from memo.identity import PrincipalIdentity
    from memo.operational_sessions import OperationalSession, OperationalSessionService


def _instant_sort_key(value: str | None) -> tuple[int, float, str]:
    raw = (value or "").strip()
    if not raw:
        return (0, 0.0, "")
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return (0, 0.0, raw)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return (1, parsed.timestamp(), raw)


_log = logging.getLogger(__name__)

_LRU_CAP_DEFAULT = 250
_SUMMARY_FALLBACK_CHARS = 80
_PROMPT_TRAIL_MAX = 5
_PROMPT_TRAIL_CHARS = 100
_RUNNING_SUMMARY_CHARS = 400
_SUMMARY_MIN_NEW_TURNS = 3
_SESSION_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z")
_CANONICAL_SESSION_KEYS = frozenset(
    {
        "session_id",
        "principal_id",
        "project",
        "workspace",
        "cwd",
        "status",
        "branch",
        "head",
        "head_commit",
        "summary",
        "checkpointed_at",
        "source_event_id",
        "recoverable_at",
        "terminated_at",
        "recoverable_reason",
        "updated_event_id",
    }
)


@dataclass(frozen=True)
class _OperationalSessionRuntime:
    service: OperationalSessionService
    identity_factory: Callable[[], PrincipalIdentity]


_OPERATIONAL_SESSION_RUNTIMES: dict[Path, _OperationalSessionRuntime] = {}
_OPERATIONAL_SESSION_RUNTIMES_LOCK = threading.RLock()


def _runtime_key(state_dir: Path) -> Path:
    return Path(state_dir).expanduser().resolve()


def install_operational_session_runtime(
    state_dir: Path,
    *,
    service: OperationalSessionService,
    identity_factory: Callable[[], PrincipalIdentity],
) -> None:
    """Install an explicitly constructed v2 session runtime for one state root."""
    if not callable(identity_factory):
        raise TypeError("identity_factory must be callable")
    key = _runtime_key(state_dir)
    with _OPERATIONAL_SESSION_RUNTIMES_LOCK:
        existing = _OPERATIONAL_SESSION_RUNTIMES.get(key)
        binding = _OperationalSessionRuntime(
            service=service,
            identity_factory=identity_factory,
        )
        if existing is not None and existing != binding:
            raise RuntimeError(f"operational session runtime already installed: {key}")
        _OPERATIONAL_SESSION_RUNTIMES[key] = binding


def remove_operational_session_runtime(state_dir: Path) -> None:
    """Remove the process-local v2 binding without mutating durable state."""
    key = _runtime_key(state_dir)
    with _OPERATIONAL_SESSION_RUNTIMES_LOCK:
        _OPERATIONAL_SESSION_RUNTIMES.pop(key, None)


def _operational_session_runtime(
    state_dir: Path,
) -> _OperationalSessionRuntime | None:
    key = _runtime_key(state_dir)
    with _OPERATIONAL_SESSION_RUNTIMES_LOCK:
        return _OPERATIONAL_SESSION_RUNTIMES.get(key)


def _canonical_session_snapshot(
    session: OperationalSession,
    local_artifacts: Mapping[str, object],
) -> dict[str, Any]:
    portable = session.to_dict()
    local = {
        key: value
        for key, value in local_artifacts.items()
        if isinstance(key, str) and key not in _CANONICAL_SESSION_KEYS
    }
    updated = session.terminated_at or session.recoverable_at or session.checkpointed_at
    return {
        **local,
        **portable,
        "cwd": session.workspace,
        "branch": session.branch or None,
        "head_commit": session.head or None,
        "summary": session.summary or None,
        "created": local.get("created") or session.checkpointed_at,
        "updated": updated,
    }


def _service_snapshot(
    service: OperationalSessionService,
    session: OperationalSession,
) -> dict[str, Any]:
    local = service.views.session_local_artifacts(session.session_id)
    return _canonical_session_snapshot(session, local)


def _list_canonical_sessions(
    service: OperationalSessionService,
    *,
    limit: int,
    project: str | None,
    workspace: str | None = None,
) -> list[dict[str, Any]]:
    sessions = service.list(
        limit=max(0, limit),
        project=project,
        workspace=workspace,
    )
    return [_service_snapshot(service, session) for session in sessions]


def _get_canonical_session(
    service: OperationalSessionService,
    session_id_or_prefix: str,
) -> dict[str, Any] | None:
    if not session_id_or_prefix or len(session_id_or_prefix) < 4:
        return None
    try:
        session_id_or_prefix = validate_session_id(session_id_or_prefix)
    except ValueError:
        return None
    canonical = service.get(session_id_or_prefix)
    if canonical is not None:
        return _service_snapshot(service, canonical)
    matches = [
        session
        for session in service.list(limit=10_000)
        if session.session_id.startswith(session_id_or_prefix)
    ]
    return _service_snapshot(service, matches[0]) if matches else None


def validate_session_id(session_id: str) -> str:
    """Return a filename-safe session id or raise ``ValueError``.

    Session ids cross MCP/hook trust boundaries and are used as sidecar
    filenames.  Keep the accepted form compatible with Claude/Codex UUIDs and
    the short slugs used by other clients while rejecting separators, glob
    metacharacters, dot segments, control characters, and oversized names.
    """
    if not isinstance(session_id, str) or not _SESSION_ID_RE.fullmatch(session_id):
        raise ValueError("session_id must be 1-128 ASCII letters, digits, underscores, or hyphens")
    return session_id


def sessions_dir(state_dir: Path) -> Path:
    """Where per-session JSON lives. Created on first use."""
    d = state_dir / "sessions"
    if d.is_symlink():
        raise ValueError(f"unsafe sessions directory: {d}")
    d.mkdir(parents=True, exist_ok=True)
    if d.is_symlink() or not d.resolve().is_relative_to(state_dir.resolve()):
        raise ValueError(f"unsafe sessions directory: {d}")
    return d


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _session_path(state_dir: Path, session_id: str) -> Path:
    safe_id = validate_session_id(session_id)
    path = sessions_dir(state_dir) / f"{safe_id}.json"
    if path.is_symlink():
        raise ValueError("session_id resolves to an unsafe session path")
    return path


def find_transcript_path(session_id: str) -> str | None:
    """Recover a transcript path by session id when a hook payload omits
    `transcript_path` (observed 2026-06-27 onward: some hook events stop
    carrying it, starving autosave/checkpoint/capture-stop/grounding of the
    one field they all key on). Claude Code names transcripts deterministically
    (`~/.claude/projects/<project>/<session_id>.jsonl`), so a glob recovers it
    without needing the payload. Best-effort, never raises."""
    if not session_id:
        return None
    try:
        session_id = validate_session_id(session_id)
    except ValueError:
        return None
    try:
        matches = list((Path.home() / ".claude" / "projects").glob(f"*/{session_id}.jsonl"))
    except OSError:
        return None
    return str(matches[0]) if matches else None


def _load(state_dir: Path, session_id: str) -> dict[str, Any] | None:
    try:
        p = _session_path(state_dir, session_id)
    except ValueError:
        return None
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _write(state_dir: Path, session_id: str, data: dict[str, Any]) -> Path:
    p = _session_path(state_dir, session_id)
    # ``None`` and ``False`` are intentionally equivalent for this stdlib bool
    # parameter, so mutating only that literal cannot produce a useful test.
    # pragma: no mutate start
    unicode_options: dict[str, Any] = {"ensure_ascii": False}
    # pragma: no mutate end
    atomic_write_text(
        p,
        json.dumps(data, indent=2, **unicode_options),
    )
    return p


@contextmanager
def _session_write_lock(state_dir: Path, session_id: str) -> Iterator[None]:
    """Cross-process write lock keyed to one validated session sidecar."""
    with authority_write_lock(_session_path(state_dir, session_id)):
        yield


def checkpoint(
    state_dir: Path,
    *,
    session_id: str,
    cwd: str,
    transcript_path: str | None = None,
    prompt: str | None = None,
    lru_cap: int | None = None,
    source_event_id: str | None = None,
    checkpointed_at: str | None = None,
    idempotency_key: str | None = None,
) -> dict[str, Any]:
    """Idempotent upsert keyed by `session_id`. Returns the persisted
    snapshot dict.

    First call for a `session_id` creates the JSON file. Subsequent
    calls preserve `created` + bump `turn_count`. Git state and last
    user msg are refreshed every call.

    `prompt` (optional): the user message from a UserPromptSubmit event.
    Appended to the `prompt_trail` ring buffer (last 5, 100 chars each)
    so the trail survives even if the session crashes before Stop fires.
    """
    if not session_id:
        raise ValueError("session_id required")

    cwd_path = Path(cwd).expanduser().resolve()
    runtime = _operational_session_runtime(state_dir)
    runtime_identity: PrincipalIdentity | None = None
    replayed_checkpoint: OperationalSession | None = None
    if runtime is not None:
        runtime_identity = runtime.identity_factory()
    git_state = gather_git_state(cwd_path)

    transcript_fields: dict[str, str | None] = {}
    if transcript_path:
        tp = Path(transcript_path).expanduser()
        transcript_fields = {
            "last_user_msg": read_last_user_msg(tp),
            "last_assistant_tail": read_last_assistant_tail(tp),
        }

    # Git/transcript inspection above can be slow. Re-load only after acquiring
    # the per-session lock, then merge onto the latest snapshot so stamps made
    # while that work ran are not clobbered.
    with _session_write_lock(state_dir, session_id):
        existing = _load(state_dir, session_id) or {}
        turn_count = int(existing.get("turn_count") or 0) + 1
        operation_key = idempotency_key or (
            f"session-checkpoint/{session_id}/{turn_count}"
        )
        if (
            runtime is not None
            and runtime_identity is not None
            and callable(
                getattr(type(runtime.service), "replay_checkpoint", None)
            )
        ):
            replayed_checkpoint = runtime.service.replay_checkpoint(
                identity=runtime_identity,
                session_id=session_id,
                project=cwd_path.name,
                workspace=str(cwd_path),
                source_event_id=source_event_id,
                checkpointed_at=checkpointed_at,
                idempotency_key=operation_key,
            )

        # prompt_trail: ring buffer of last N user prompts, crash-resilient
        # because it's updated on UserPromptSubmit (not just Stop).
        trail = list(existing.get("prompt_trail") or [])
        if prompt:
            clean_prompt = _strip_command_wrappers(prompt.strip())
            if clean_prompt and not clean_prompt.startswith(_COMMAND_WRAPPER_PREFIXES):
                trail.append(clean_prompt[:_PROMPT_TRAIL_CHARS])
                trail = trail[-_PROMPT_TRAIL_MAX:]

        now = checkpointed_at or (_now_iso() if runtime is None else "")
        snapshot: dict[str, Any] = {
            **existing,
            "session_id": session_id,
            "cwd": str(cwd_path),
            "project": cwd_path.name,
            "branch": git_state["branch"],
            "head_commit": git_state["head_commit"],
            "modified_files": git_state["modified_files"],
            "transcript_path": str(transcript_path)
            if transcript_path
            else existing.get("transcript_path"),
            "last_user_msg": transcript_fields.get("last_user_msg")
            or existing.get("last_user_msg"),
            "last_assistant_tail": transcript_fields.get("last_assistant_tail")
            or existing.get("last_assistant_tail"),
            "prompt_trail": trail,
            "running_summary": existing.get("running_summary"),
            "summary_turn": int(existing.get("summary_turn") or 0),
            # Default summary to last user msg head; an external enricher
            # (e.g. capture-stop with MLX warm) may overwrite later.
            "summary": (
                existing.get("summary")
                if existing.get("summary") and not is_command_noise(existing.get("summary"))
                else (
                    (transcript_fields.get("last_user_msg") or "")[:_SUMMARY_FALLBACK_CHARS] or None
                )
            ),
            "created": existing.get("created") or now,
            "updated": now,
            "turn_count": turn_count,
            "last_recall_turn": existing.get("last_recall_turn"),
            "last_recap_turn": existing.get("last_recap_turn"),
        }
        if runtime is not None:
            source_id = source_event_id or (
                "local-session/"
                + hashlib.sha256(
                    json.dumps(
                        {
                            "session_id": session_id,
                            "turn_count": turn_count,
                            "cwd": str(cwd_path),
                            "branch": snapshot["branch"],
                            "head": snapshot["head_commit"],
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest()
            )
            canonical = replayed_checkpoint
            if canonical is None:
                if runtime_identity is None:
                    raise RuntimeError("operational session identity is unavailable")
                canonical = runtime.service.checkpoint(
                    identity=runtime_identity,
                    session_id=session_id,
                    project=cwd_path.name,
                    workspace=str(cwd_path),
                    summary=str(snapshot.get("summary") or ""),
                    branch=str(snapshot.get("branch") or ""),
                    head=str(snapshot.get("head_commit") or ""),
                    source_event_id=source_id,
                    checkpointed_at=checkpointed_at,
                    idempotency_key=operation_key,
                )
            snapshot["created"] = existing.get("created") or canonical.checkpointed_at
            snapshot["updated"] = canonical.checkpointed_at
            local_artifacts = {
                key: value for key, value in snapshot.items() if key not in _CANONICAL_SESSION_KEYS
            }
            runtime.service.views.replace_session_local_artifacts(
                session_id,
                local_artifacts,
            )
            snapshot = _canonical_session_snapshot(
                canonical,
                local_artifacts,
            )
        _write(state_dir, session_id, snapshot)
    cap = lru_cap if lru_cap is not None else (flag_int("MEMO_SESSION_LRU_CAP") or _LRU_CAP_DEFAULT)
    prune_lru(state_dir, cap=cap)
    return snapshot


def next_turn(state_dir: Path, session_id: str) -> int:
    """The in-flight turn label for an exchange: (existing turn_count) + 1.

    The recall-hook computes this at UserPromptSubmit and writes the SAME value
    into both recall.log (as `turn`) and the session snapshot (`last_recall_turn`
    via stamp_recall_turn), so the Stop-hook grounding detector can join the
    answer back to its recall with zero race — the label is whatever the
    recall-hook stamped, regardless of how turn_count later evolves.
    """
    if not session_id:
        return 1
    with _session_write_lock(state_dir, session_id):
        existing = _load(state_dir, session_id) or {}
        return int(existing.get("turn_count") or 0) + 1


def stamp_recall_turn(state_dir: Path, session_id: str, turn: int) -> None:
    """Merge-write `last_recall_turn` into the session snapshot. Best-effort,
    never raises — telemetry must not break the recall hook. Creates a minimal
    snapshot if the session file doesn't exist yet (recall can fire before the
    autosave checkpoint)."""
    if not session_id:
        return
    try:
        with _session_write_lock(state_dir, session_id):
            existing = _load(state_dir, session_id) or {}
            existing["session_id"] = session_id
            existing["last_recall_turn"] = int(turn)
            # Refresh `updated` — the session GC evicts oldest-by-updated.
            existing["updated"] = _now_iso()
            _write(state_dir, session_id, existing)
    except (OSError, ValueError, TypeError) as exc:
        _log.debug("session: failed to checkpoint recall turn: %s", exc)


def stamp_recap_turn(state_dir: Path, session_id: str, turn: int) -> None:
    """Merge-write `last_recap_turn` into the session snapshot. Best-effort,
    never raises — recap must not break the recall hook it rides on. Mirrors
    `stamp_recall_turn`; used by `cli_recap.maybe_write_recap` to remember the
    turn a `※ memo recap:` line last fired on, so the cadence check
    (`due_for_recap`) doesn't re-fire every turn once due."""
    if not session_id:
        return
    try:
        with _session_write_lock(state_dir, session_id):
            existing = _load(state_dir, session_id) or {}
            existing["session_id"] = session_id
            existing["last_recap_turn"] = int(turn)
            _write(state_dir, session_id, existing)
    except (OSError, ValueError, TypeError) as exc:
        _log.debug("session: failed to checkpoint recap turn: %s", exc)


def recent_prompts(state_dir: Path, session_id: str, n: int) -> list[str]:
    """Last `n` user prompts from the session `prompt_trail` ring buffer.

    Used by the recall-hook to re-anchor a short follow-up prompt with recent
    conversation context before bailing. Returns the most recent `n` (oldest
    first), or `[]` if the session is missing/has no trail. Best-effort, never
    raises — recall must not break on a stale/absent session file.
    """
    if not session_id or n <= 0:
        return []
    try:
        existing = _load(state_dir, session_id) or {}
        trail = [
            p for p in (existing.get("prompt_trail") or []) if isinstance(p, str) and p.strip()
        ]
        return trail[-n:]
    except (OSError, ValueError, TypeError) as exc:
        _log.debug("session: failed to read prompt_trail: %s", exc)
        return []


def _resolve_session_cwd(value: str) -> str:
    try:
        return str(Path(value).expanduser().resolve())
    except OSError:
        return value


def list_sessions(
    state_dir: Path,
    *,
    limit: int = 10,
    project: str | None = None,
    cwd: str | None = None,
) -> list[dict[str, Any]]:
    """Recent sessions sorted by `updated` desc.

    Filters compose with AND semantics:

    - `project`: basename match. Cheap, but collides if two clones of
      the same repo live under different parents (`~/work/memo` vs
      `~/sandbox/memo`).
    - `cwd`: full-path match. Both the stored snapshot's cwd and the
      filter value are run through `Path(...).resolve()` first so
      `/tmp/x` vs `/private/tmp/x` (the macOS symlink dance) and
      `~/foo` vs `/Users/.../foo` compare equal. This is the filter
      the shell wrapper uses since it's the only one that uniquely
      identifies a working tree across siblings.
    """
    cwd_resolved = _resolve_session_cwd(cwd) if cwd else None
    runtime = _operational_session_runtime(state_dir)
    if runtime is not None:
        return _list_canonical_sessions(
            runtime.service,
            limit=limit,
            project=project,
            workspace=cwd_resolved,
        )

    out: list[dict[str, Any]] = []
    d = sessions_dir(state_dir)
    for p in d.glob("*.json"):
        if p.is_symlink():
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if project and data.get("project") != project:
            continue
        if cwd_resolved is not None:
            stored = data.get("cwd") or ""
            if not stored:
                continue
            if _resolve_session_cwd(stored) != cwd_resolved:
                continue
        out.append(data)
    out.sort(key=lambda x: _instant_sort_key(x.get("updated")), reverse=True)
    return out[:limit]


def get_session(state_dir: Path, session_id_or_prefix: str) -> dict[str, Any] | None:
    """Look up a session by full id or unique prefix (≥4 chars).
    Returns None if not found. Returns the *first* match on tie —
    sessions are LRU-capped at 250 so collision surface is tiny, and
    paying for a full prefix-resolution dance like `Memory.resolve_id`
    isn't worth it here."""
    if not session_id_or_prefix or len(session_id_or_prefix) < 4:
        return None
    try:
        session_id_or_prefix = validate_session_id(session_id_or_prefix)
    except ValueError:
        return None
    runtime = _operational_session_runtime(state_dir)
    if runtime is not None:
        return _get_canonical_session(runtime.service, session_id_or_prefix)
    d = sessions_dir(state_dir)
    # Fast path: exact filename hit.
    exact = _session_path(state_dir, session_id_or_prefix)
    if exact.is_file():
        try:
            return json.loads(exact.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
    # Prefix scan.
    for p in d.glob(f"{session_id_or_prefix}*.json"):
        if p.is_symlink():
            continue
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
    return None


def prune_lru(state_dir: Path, *, cap: int = _LRU_CAP_DEFAULT) -> int:
    """Delete oldest sessions by `updated` so the dir holds at most
    `cap` files. Returns the number deleted. Called from
    `checkpoint()` so the cleanup is automatic."""
    if cap < 0:
        raise ValueError("cap must be non-negative")
    d = sessions_dir(state_dir)
    files = list(d.glob("*.json"))
    pairs: list[tuple[Path, tuple[int, float, str]]] = []
    for p in files:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            sort_key = _instant_sort_key(data.get("updated"))
        except (json.JSONDecodeError, OSError):
            sort_key = (0, 0.0, "")
        pairs.append((p, sort_key))
    # Sort ascending — oldest first — and trim the head past `cap`.
    pairs.sort(key=lambda x: x[1])
    to_delete = pairs[: max(len(pairs) - cap, 0)]
    deleted = 0
    for p, _ in to_delete:
        try:
            p.unlink()
            deleted += 1
        except OSError:
            pass
    return deleted


def format_relative(updated_iso: str | None, now: datetime | None = None) -> str:
    """`updated` → `"5m ago"` / `"2h ago"` / `"3d ago"` for the picker.
    Falls back to `"—"` on parse failure."""
    if not updated_iso:
        return "—"
    try:
        ts = datetime.fromisoformat(updated_iso)
    except ValueError:
        return "—"
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    now = now or datetime.now(UTC)
    delta = now - ts
    secs = int(delta.total_seconds())
    if secs < 60:
        return "<1m ago"
    if secs < 3600:
        return f"{secs // 60}m ago"
    if secs < 86400:
        return f"{secs // 3600}h ago"
    return f"{secs // 86400}d ago"


_AUTOSAVE_THRESHOLD_KB_DEFAULT = 1024  # ~1MB -> roughly 50-200 turns
_AUTOSAVE_COOLDOWN_DEFAULT = 300  # 5 min between consecutive autosaves


def check_autosave(
    state_dir: Path,
    *,
    session_id: str,
    transcript_path: str | None,
    threshold_kb: int = _AUTOSAVE_THRESHOLD_KB_DEFAULT,
    cooldown_secs: int = _AUTOSAVE_COOLDOWN_DEFAULT,
) -> tuple[bool, int]:
    """Return (should_autosave, transcript_size_kb).

    Fast path: one os.stat() call. JSON read only if threshold is breached.
    Never raises.
    """
    if not transcript_path:
        return False, 0
    try:
        size_kb = Path(transcript_path).expanduser().stat().st_size // 1024
    except OSError:
        return False, 0

    if size_kb < threshold_kb:
        return False, size_kb

    existing = _load(state_dir, session_id)
    if existing:
        last = existing.get("last_autosave_at")
        if last:
            try:
                elapsed = (datetime.now(UTC) - datetime.fromisoformat(last)).total_seconds()
                if elapsed < cooldown_secs:
                    return False, size_kb
            except (ValueError, TypeError):
                _log.debug("session: unparseable transcript line, skipping")

    return True, size_kb


def mark_autosaved(state_dir: Path, session_id: str) -> None:
    """Stamp last_autosave_at in the session JSON (best-effort, silent)."""
    with (
        contextlib.suppress(OSError, ValueError),
        _session_write_lock(state_dir, session_id),
    ):
        existing = _load(state_dir, session_id)
        if existing is None:
            return
        existing["last_autosave_at"] = _now_iso()
        _write(state_dir, session_id, existing)


def _recent_summary_exchanges(transcript_path: Path) -> list[str]:
    """Read the compact user/assistant lines used by summary refresh."""

    try:
        lines = transcript_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    exchanges: list[str] = []
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        role = obj.get("type") or obj.get("role")
        if role not in ("user", "assistant"):
            continue
        msg = obj.get("message", obj)
        content = msg.get("content") if isinstance(msg, dict) else None
        text = _extract_text(content)
        if text:
            label = "User" if role == "user" else "Assistant"
            exchanges.append(f"[{label}] {text[:300]}")
    return exchanges


def refresh_summary(
    state_dir: Path,
    session_id: str,
    *,
    helper_model: str = "mlx-community/Qwen3-4B-4bit",
    min_new_turns: int = _SUMMARY_MIN_NEW_TURNS,
) -> bool:
    """Generate/update `running_summary` from the transcript using the helper model.

    Called from the Stop hook. Throttled by `min_new_turns` so we don't
    pay LLM cost on every single turn. Returns True if the summary was
    written, False if skipped or failed.

    Idempotent: if `turn_count - summary_turn < min_new_turns`, skip.
    On crash the last written `running_summary` is preserved in the snapshot.
    """
    with _session_write_lock(state_dir, session_id):
        existing = _load(state_dir, session_id)
        if existing is None:
            return False

        turn_count = int(existing.get("turn_count") or 0)
        summary_turn = int(existing.get("summary_turn") or 0)
        if turn_count - summary_turn < min_new_turns:
            return False
        transcript_path_str = existing.get("transcript_path")
    if not transcript_path_str:
        return False
    transcript_path = Path(transcript_path_str).expanduser()
    if not transcript_path.is_file():
        return False

    # Collect last ~10 user+assistant exchanges for the LLM prompt.
    exchanges = _recent_summary_exchanges(transcript_path)
    if not exchanges:
        return False

    recent = "\n\n".join(exchanges[-10:])
    llm_prompt = (
        "Based on this work session, write ONE brief PARAGRAPH (2-3 sentences) "
        "in English that summarizes: (1) what was being worked on, (2) what decisions or "
        "progress were made, (3) what was left pending or was the next step.\n\n"
        f"Session:\n{recent}\n\n"
        "Summary (2-3 sentences, no bullets or headings):"
    )

    try:
        from memo.llm import MLXChat

        chat = MLXChat()
        result = chat.chat(
            helper_model,
            [{"role": "user", "content": llm_prompt}],
            options={"temperature": 0.0, "num_predict": 150},
        )
        summary = (result.get("message") or {}).get("content") or ""
        summary = summary.strip()
    except Exception as exc:
        _log.debug("session: reflect summary LLM call failed: %s", exc)
        return False

    if not summary:
        return False

    with _session_write_lock(state_dir, session_id):
        latest = _load(state_dir, session_id)
        if latest is None:
            return False
        # Another concurrent refresh may already have summarized this turn.
        if int(latest.get("summary_turn") or 0) >= turn_count:
            return False
        latest["running_summary"] = summary[:_RUNNING_SUMMARY_CHARS]
        latest["summary_turn"] = turn_count
        _write(state_dir, session_id, latest)
        return True


def mark_reflected(state_dir: Path, session_id: str) -> bool:
    """Stamp `reflected_at` in the session JSON so `memo reflect --if-due`
    can skip already-processed sessions. Best-effort, never raises."""
    with (
        contextlib.suppress(OSError, ValueError),
        _session_write_lock(state_dir, session_id),
    ):
        existing = _load(state_dir, session_id)
        if existing is None:
            return False
        existing["reflected_at"] = _now_iso()
        _write(state_dir, session_id, existing)
        return True
    return False


def get_recalled_ids(state_dir: Path, session_id: str) -> dict[str, int]:
    """Return mapping of memory_id -> first_seen_turn for this session.
    Returns {} if session doesn't exist or has no recalled_ids."""
    existing = _load(state_dir, session_id) or {}
    return dict(existing.get("recalled_ids", {}))


def mark_ids_recalled(
    state_dir: Path,
    session_id: str,
    new_ids: dict[str, int],
) -> None:
    """Merge new {id: turn} entries into the session's recalled_ids.
    Best-effort: swallows all exceptions (non-critical path)."""
    if not new_ids:
        return
    try:
        with _session_write_lock(state_dir, session_id):
            existing = _load(state_dir, session_id) or {}
            recalled = dict(existing.get("recalled_ids", {}))
            # Only record first-seen turn; don't overwrite if already present
            for mid, turn in new_ids.items():
                if mid not in recalled:
                    recalled[mid] = turn
            existing["recalled_ids"] = recalled
            existing["updated"] = _now_iso()
            _write(state_dir, session_id, existing)
    except Exception:  # noqa: S110
        pass


def _clean_snapshot_summary(snapshot: dict[str, Any], width: int) -> str:
    """Pick the most useful short summary from a session snapshot."""
    for cand in (
        snapshot.get("running_summary"),
        snapshot.get("summary"),
        snapshot.get("last_user_msg"),
    ):
        if not is_command_noise(cand):
            return _strip_command_wrappers(str(cand)).replace("\n", " ")[:width]
    return "—"


def render_active_memory(snapshot: dict[str, Any]) -> list[str]:
    """Render a compact session-memory block from one snapshot."""
    if not snapshot:
        return []

    lines = ["### Active memory", ""]
    lines.append(f"- **In progress**: {_clean_snapshot_summary(snapshot, 140)}")

    project = snapshot.get("project") or "—"
    branch = snapshot.get("branch") or "—"
    turns = snapshot.get("turn_count") or 0
    lines.append(f"- **Context**: `{project}` · `{branch}` · {turns} turns")

    modified_files = [
        str(path).strip()
        for path in (snapshot.get("modified_files") or [])
        if isinstance(path, str) and path.strip()
    ][:4]
    if modified_files:
        lines.append("- **Files touched**: " + ", ".join(f"`{path}`" for path in modified_files))

    last_assistant_tail = snapshot.get("last_assistant_tail")
    if last_assistant_tail and not is_command_noise(last_assistant_tail):
        tail = _strip_command_wrappers(str(last_assistant_tail)).replace("\n", " ")[:160]
        if tail:
            lines.append(f"- **Last reply**: {tail}")

    trail = [
        str(prompt).strip()
        for prompt in (snapshot.get("prompt_trail") or [])
        if isinstance(prompt, str) and prompt.strip()
    ]
    if trail:
        lines.append("- **Open loops (session)**:")
        for i, prompt in enumerate(reversed(trail[-_PROMPT_TRAIL_MAX:]), 1):
            lines.append(f"  {i}. {prompt[:120]}")

    return lines


def render_continuity(rows: list[dict[str, Any]], cwd: str) -> str:
    """Render "what was I working on?" for the latest session in `cwd`.

    Memo-native continuity built on memo's own session snapshots (cwd / branch /
    running_summary / open-loop prompt_trail).
    Pure: takes the session rows + cwd, returns markdown. Returns a short
    "no prior session" line when none match.
    """
    same = [r for r in rows if (r.get("cwd") or "") == cwd]
    if not same:
        return "No previous session in this directory."
    top = same[0]

    lines = [
        f"## What you were doing ({format_relative(top.get('updated'))})",
        "",
        f"- **Summary**: {_clean_snapshot_summary(top, 160)}",
        f"- **Branch**: `{top.get('branch') or '—'}`  |  **Turns**: {top.get('turn_count') or 0}",
        f"- **Resume**: `claude --resume {top.get('session_id') or ''}`",
    ]
    active_memory = render_active_memory(top)
    if active_memory:
        lines += ["", *active_memory]
    return "\n".join(lines)
