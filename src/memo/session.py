"""Session snapshots — checkpoint where you left off so you can resume.

Companion to `capture.py`. Where capture extracts long-lived insights
into the memory archive, this module persists *short-lived* "what was
I working on" state: cwd, branch, last user prompt, last todo, last
plan. Survives a crashed/closed Claude Code session so the next
SessionStart can show a picker of recent work.

## Why a sidecar JSON store, not a memoria

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
import json
import logging
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)

_LRU_CAP_DEFAULT = 50
_LAST_USER_MSG_CHARS = 240
_SUMMARY_FALLBACK_CHARS = 80
_MODIFIED_FILES_CAP = 30
_PROMPT_TRAIL_MAX = 5
_PROMPT_TRAIL_CHARS = 100
_ASSISTANT_TAIL_CHARS = 200
_RUNNING_SUMMARY_CHARS = 400
_SUMMARY_MIN_NEW_TURNS = 3


def sessions_dir(state_dir: Path) -> Path:
    """Where per-session JSON lives. Created on first use."""
    d = state_dir / "sessions"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _git(cwd: Path, args: list[str], *, strip: bool = True) -> str | None:
    """Run a git command in `cwd`. Returns stdout (stripped by default),
    or None on any failure (not a repo, git not installed, command
    failed). Never raises — checkpoint must succeed even outside a
    git context.

    `strip=False` is critical for `git status --porcelain`: its lines
    start with a 2-char status code that may begin with a space
    (e.g. ` M file`). A blanket `.strip()` on the whole output eats
    the first line's leading space, shifting `line[3:]` by one — the
    first reported file ends up missing its first character.
    """
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    raw = out.stdout
    # Spelled out (not a ternary) so the comment explaining why
    # porcelain output needs `rstrip("\n")` instead of `.strip()`
    # stays attached to the right branch.
    if strip:  # noqa: SIM108
        raw = raw.strip()
    else:
        # Trim only the trailing newline git always appends; leading
        # whitespace on the first line is meaningful.
        raw = raw.rstrip("\n")
    return raw or None


def gather_git_state(cwd: Path) -> dict[str, Any]:
    """Best-effort git introspection. All fields nullable."""
    branch = _git(cwd, ["rev-parse", "--abbrev-ref", "HEAD"])
    head = _git(cwd, ["log", "-1", "--oneline", "--no-decorate"])
    status = _git(cwd, ["status", "--porcelain"], strip=False)
    modified: list[str] = []
    if status:
        for line in status.splitlines()[: _MODIFIED_FILES_CAP]:
            # Porcelain v1 line shape: `XY filename` — 2 status chars
            # then a single space then the path. Slice past the 3-char
            # header; renames (`R  old -> new`) are kept as-is to surface
            # both paths in the snapshot.
            modified.append(line[3:].strip())
    return {
        "branch": branch,
        "head_commit": head,
        "modified_files": modified,
    }


def read_last_user_msg(transcript_path: Path) -> str | None:
    """Walk the JSONL transcript backwards, return the latest user
    message text. Mirrors `capture._read_last_exchange` but only
    needs the user side, so we can stop scanning earlier.
    """
    if not transcript_path.is_file():
        return None
    try:
        lines = transcript_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        role = obj.get("type") or obj.get("role")
        if role != "user":
            continue
        msg = obj.get("message", obj)
        content = msg.get("content") if isinstance(msg, dict) else None
        text = _extract_text(content)
        if text:
            return text[:_LAST_USER_MSG_CHARS]
    return None


def read_last_assistant_tail(transcript_path: Path) -> str | None:
    """Walk the JSONL transcript backwards, return the tail of the latest
    assistant message (last ~200 chars). Used to give context about what
    Claude was doing when the session was interrupted.
    """
    if not transcript_path.is_file():
        return None
    try:
        lines = transcript_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        role = obj.get("type") or obj.get("role")
        if role != "assistant":
            continue
        msg = obj.get("message", obj)
        content = msg.get("content") if isinstance(msg, dict) else None
        text = _extract_text(content)
        if text:
            return text[-_ASSISTANT_TAIL_CHARS:] if len(text) > _ASSISTANT_TAIL_CHARS else text
    return None


def _extract_text(content: Any) -> str:
    """Same shape as capture._extract_text — Claude Code message content
    is either a plain string or a list of blocks. Skip tool blocks.
    Kept local so this module stays import-cheap (no `memo.capture`)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        chunks: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text":
                t = block.get("text") or ""
                if t.strip():
                    chunks.append(t.strip())
        return "\n\n".join(chunks).strip()
    return ""


def _session_path(state_dir: Path, session_id: str) -> Path:
    return sessions_dir(state_dir) / f"{session_id}.json"


def _load(state_dir: Path, session_id: str) -> dict[str, Any] | None:
    p = _session_path(state_dir, session_id)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _write(state_dir: Path, session_id: str, data: dict[str, Any]) -> Path:
    p = _session_path(state_dir, session_id)
    # Atomic-ish: write to .tmp, replace. Avoids a torn read if the
    # process is killed mid-write (Stop hook racing a SIGTERM).
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, p)
    return p


def checkpoint(
    state_dir: Path,
    *,
    session_id: str,
    cwd: str,
    transcript_path: str | None = None,
    prompt: str | None = None,
    lru_cap: int = _LRU_CAP_DEFAULT,
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
    existing = _load(state_dir, session_id) or {}

    git_state = gather_git_state(cwd_path)

    last_user_msg: str | None = None
    last_assistant_tail: str | None = None
    if transcript_path:
        tp = Path(transcript_path).expanduser()
        last_user_msg = read_last_user_msg(tp)
        last_assistant_tail = read_last_assistant_tail(tp)

    # prompt_trail: ring buffer of last N user prompts, crash-resilient
    # because it's updated on UserPromptSubmit (not just Stop).
    trail = list(existing.get("prompt_trail") or [])
    if prompt:
        trail.append(prompt.strip()[:_PROMPT_TRAIL_CHARS])
        trail = trail[-_PROMPT_TRAIL_MAX:]

    now = _now_iso()
    snapshot: dict[str, Any] = {
        "session_id": session_id,
        "cwd": str(cwd_path),
        "project": cwd_path.name,
        "branch": git_state["branch"],
        "head_commit": git_state["head_commit"],
        "modified_files": git_state["modified_files"],
        "transcript_path": str(transcript_path) if transcript_path else existing.get("transcript_path"),
        "last_user_msg": last_user_msg or existing.get("last_user_msg"),
        "last_assistant_tail": last_assistant_tail or existing.get("last_assistant_tail"),
        "prompt_trail": trail,
        "running_summary": existing.get("running_summary"),
        "summary_turn": int(existing.get("summary_turn") or 0),
        # Default summary to last user msg head; an external enricher
        # (e.g. capture-stop with MLX warm) may overwrite later.
        "summary": existing.get("summary") or (
            (last_user_msg or "")[:_SUMMARY_FALLBACK_CHARS] or None
        ),
        "created": existing.get("created") or now,
        "updated": now,
        "turn_count": int(existing.get("turn_count") or 0) + 1,
        # Correlation stamp written by the recall-hook (next_turn/stamp_recall_turn).
        # Preserved across checkpoints so the Stop-hook grounding detector can read
        # back the turn label the recall used. See grounding.score_turn.
        "last_recall_turn": existing.get("last_recall_turn"),
    }

    _write(state_dir, session_id, snapshot)
    prune_lru(state_dir, cap=lru_cap)
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
        existing = _load(state_dir, session_id) or {}
        existing["session_id"] = session_id
        existing["last_recall_turn"] = int(turn)
        _write(state_dir, session_id, existing)
    except (OSError, ValueError, TypeError) as exc:
        _log.debug("session: failed to checkpoint recall turn: %s", exc)


def list_sessions(
    state_dir: Path, *, limit: int = 10,
    project: str | None = None, cwd: str | None = None,
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
    cwd_resolved: str | None = None
    if cwd:
        try:
            cwd_resolved = str(Path(cwd).expanduser().resolve())
        except OSError:
            cwd_resolved = cwd

    out: list[dict[str, Any]] = []
    d = sessions_dir(state_dir)
    for p in d.glob("*.json"):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if project and data.get("project") != project:
            continue
        if cwd_resolved is not None:
            stored = data.get("cwd") or ""
            try:
                stored_resolved = str(Path(stored).expanduser().resolve())
            except OSError:
                stored_resolved = stored
            if stored_resolved != cwd_resolved:
                continue
        out.append(data)
    out.sort(key=lambda x: x.get("updated") or "", reverse=True)
    return out[:limit]


def get_session(state_dir: Path, session_id_or_prefix: str) -> dict[str, Any] | None:
    """Look up a session by full id or unique prefix (≥4 chars).
    Returns None if not found. Returns the *first* match on tie —
    sessions are LRU-capped at 50 so collision surface is tiny, and
    paying for a full prefix-resolution dance like `Memory.resolve_id`
    isn't worth it here."""
    if not session_id_or_prefix or len(session_id_or_prefix) < 4:
        return None
    d = sessions_dir(state_dir)
    # Fast path: exact filename hit.
    exact = d / f"{session_id_or_prefix}.json"
    if exact.is_file():
        try:
            return json.loads(exact.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
    # Prefix scan.
    for p in d.glob(f"{session_id_or_prefix}*.json"):
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
    return None


def prune_lru(state_dir: Path, *, cap: int = _LRU_CAP_DEFAULT) -> int:
    """Delete oldest sessions by `updated` so the dir holds at most
    `cap` files. Returns the number deleted. Called from
    `checkpoint()` so the cleanup is automatic."""
    d = sessions_dir(state_dir)
    files = list(d.glob("*.json"))
    if len(files) <= cap:
        return 0
    pairs: list[tuple[str, Path]] = []
    for p in files:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            updated = data.get("updated") or ""
        except (json.JSONDecodeError, OSError):
            updated = ""
        pairs.append((updated, p))
    # Sort ascending — oldest first — and trim the head past `cap`.
    pairs.sort(key=lambda x: x[0])
    to_delete = pairs[: len(pairs) - cap]
    deleted = 0
    for _, p in to_delete:
        try:
            p.unlink()
            deleted += 1
        except OSError:
            pass
    return deleted


def format_relative(updated_iso: str | None, now: datetime | None = None) -> str:
    """`updated` → `"hace 5m"` / `"hace 2h"` / `"hace 3d"` for the picker.
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
        return "hace <1m"
    if secs < 3600:
        return f"hace {secs // 60}m"
    if secs < 86400:
        return f"hace {secs // 3600}h"
    return f"hace {secs // 86400}d"


_AUTOSAVE_THRESHOLD_KB_DEFAULT = 1024  # ~1MB -> roughly 50-200 turns
_AUTOSAVE_COOLDOWN_DEFAULT = 300       # 5 min between consecutive autosaves


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
                pass

    return True, size_kb


def mark_autosaved(state_dir: Path, session_id: str) -> None:
    """Stamp last_autosave_at in the session JSON (best-effort, silent)."""
    existing = _load(state_dir, session_id)
    if existing is None:
        return
    existing["last_autosave_at"] = _now_iso()
    with contextlib.suppress(OSError):
        _write(state_dir, session_id, existing)


def refresh_summary(
    state_dir: Path,
    session_id: str,
    *,
    helper_model: str = "mlx-community/Qwen2.5-3B-Instruct-4bit",
    min_new_turns: int = _SUMMARY_MIN_NEW_TURNS,
) -> bool:
    """Generate/update `running_summary` from the transcript using the 3B helper.

    Called from the Stop hook. Throttled by `min_new_turns` so we don't
    pay LLM cost on every single turn. Returns True if the summary was
    written, False if skipped or failed.

    Idempotent: if `turn_count - summary_turn < min_new_turns`, skip.
    On crash the last written `running_summary` is preserved in the snapshot.
    """
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

    try:
        lines = transcript_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return False

    # Collect last ~10 user+assistant exchanges for the LLM prompt.
    exchanges: list[str] = []
    for line in lines:
        line = line.strip()
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
            label = "Usuario" if role == "user" else "Asistente"
            exchanges.append(f"[{label}] {text[:300]}")
    if not exchanges:
        return False

    recent = "\n\n".join(exchanges[-10:])
    llm_prompt = (
        "Basado en esta sesión de trabajo, escribe UN PÁRRAFO breve (2-3 oraciones) "
        "en español que resuma: (1) qué se estaba trabajando, (2) qué decisiones o "
        "progreso hubo, (3) qué quedó pendiente o era el siguiente paso.\n\n"
        f"Sesión:\n{recent}\n\n"
        "Resumen (2-3 oraciones, sin viñetas ni encabezados):"
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

    existing["running_summary"] = summary[:_RUNNING_SUMMARY_CHARS]
    existing["summary_turn"] = turn_count
    _write(state_dir, session_id, existing)
    return True


def mark_reflected(state_dir: Path, session_id: str) -> bool:
    """Stamp `reflected_at` in the session JSON so `memo reflect --if-due`
    can skip already-processed sessions. Best-effort, never raises."""
    existing = _load(state_dir, session_id)
    if existing is None:
        return False
    existing["reflected_at"] = _now_iso()
    with contextlib.suppress(OSError):
        _write(state_dir, session_id, existing)
        return True
    return False


def update_summary(
    state_dir: Path, session_id: str, summary: str,
) -> bool:
    """Patch the `summary` field of an existing session. Used by the
    capture-stop pipeline (which already has MLXChat warm) to enrich
    the cheap heuristic summary set by `checkpoint()`. Returns True
    on success, False if the session doesn't exist."""
    existing = _load(state_dir, session_id)
    if existing is None:
        return False
    summary = (summary or "").strip()
    if not summary:
        return False
    existing["summary"] = summary[:200]
    _write(state_dir, session_id, existing)
    return True
