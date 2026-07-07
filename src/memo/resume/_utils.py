from __future__ import annotations

import contextlib
import contextvars
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from ._types import (
    _ANSI_RE,
    _KNOWN_AGENTS,
    ResumeAgent,
    ResumeCandidate,
)

# Agent display-name → canonical-slug aliases. Inlined (was
# ``synapse.agents.AGENT_PROFILE_ALIASES``) so resume has no synapse dependency.
AGENT_PROFILE_ALIASES: dict[str, str] = {
    "claude code": "claude",
    "claude-code": "claude",
    "claude_code": "claude",
    "cursor agent": "cursor",
    "cursor-agent": "cursor",
    "devin ai": "devin",
    "devin desktop": "devin-desktop",
    "devin-desktop": "devin-desktop",
    "gemini cli": "gemini",
    "gemini-cli": "gemini",
    "opencode ai": "opencode",
    "opencode-ai": "opencode",
}

_DEFAULT_ACTIVE_WINDOW_SECONDS = 120
_RECENT_WINDOW_SECONDS = 3600
_DEFAULT_SCAN_CAP = 150

# Lets the episode backfill lift the per-provider parse cap (it must enumerate the
# WHOLE history, unlike the picker) without mutating the MEMO_RESUME_SCAN_CAP env —
# memo's architecture forbids reading/writing MEMO_* flags through os.environ.
_scan_cap_override: contextvars.ContextVar[int | None] = contextvars.ContextVar(
    "_scan_cap_override", default=None
)


def _scan_cap() -> int:
    override = _scan_cap_override.get()
    if override is not None:
        return max(0, override)
    from memo.flags import flag_int

    flag_value = flag_int("MEMO_RESUME_SCAN_CAP")
    return _DEFAULT_SCAN_CAP if flag_value is None else max(0, flag_value)


def _mtime_capped(paths: Iterable[Path], cap: int | None = None) -> list[Path]:
    """Newest-first by mtime, capped to `cap` files.

    Providers `stat()` every candidate (cheap) but only fully-parse the most
    recent `cap` — the content read (`_jsonl_latest_user_text` scans up to 4000
    lines/file) is the expensive part, and on a machine with thousands of
    transcripts an uncapped parse made `memo resume` scale linearly with disk.
    A picker never needs more than the most-recent few hundred sessions.
    """
    limit = cap if cap is not None else _scan_cap()
    stated: list[tuple[float, Path]] = []
    for path in paths:
        try:
            stated.append((path.stat().st_mtime, path))
        except OSError:
            continue
    stated.sort(key=lambda item: item[0], reverse=True)
    return [path for _, path in stated[:limit]]
_STATUS_RANK = {"active": 3, "stale": 2, "recent": 1}
_STATUS_BADGES = {
    "active": ("● ", "\x1b[32m"),  # green
    "stale": ("◌ ", "\x1b[33m"),  # yellow (was active, stopped without clean exit)
    "recent": ("◐ ", "\x1b[90m"),  # dim
}


def _active_window_seconds() -> int:
    from memo.flags import flag_int

    flag_value = flag_int("MEMO_RESUME_ACTIVE_WINDOW_S")
    return (
        _DEFAULT_ACTIVE_WINDOW_SECONDS if flag_value is None else max(0, flag_value)
    )


def _age_seconds(updated_at: str, now: datetime) -> float | None:
    raw = (updated_at or "").strip()
    if not raw:
        return None
    try:
        parsed = (
            datetime.fromisoformat(raw[:-1] + "+00:00")
            if raw.endswith("Z")
            else datetime.fromisoformat(raw)
        )
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return max(0.0, (now.astimezone(UTC) - parsed.astimezone(UTC)).total_seconds())


def _status_from_updated_at(updated_at: str, now: datetime) -> str:
    age = _age_seconds(updated_at, now)
    if age is None:
        return ""
    if age <= _active_window_seconds():
        return "active"
    if age < _RECENT_WINDOW_SECONDS:
        return "recent"
    return ""


def _with_run_status(candidate: ResumeCandidate, now: datetime) -> ResumeCandidate:
    # A status set by the checkpoint projection (e.g. "stale") is authoritative and kept.
    if candidate.status:
        return candidate
    computed = _status_from_updated_at(candidate.updated_at, now)
    if not computed:
        return candidate
    from dataclasses import replace
    return replace(candidate, status=computed)


def _prefer_status(left: str, right: str) -> str:
    return left if _STATUS_RANK.get(left, 0) >= _STATUS_RANK.get(right, 0) else right


def _normalize_agent_filter(agent: str) -> ResumeAgent:
    normalized = _normalize_agent_name(agent)
    if normalized == "all":
        return "all"
    if normalized in _KNOWN_AGENTS:
        return cast(ResumeAgent, normalized)
    return "generic"


def _normalize_agent_name(agent: str) -> str:
    normalized = " ".join(str(agent or "").strip().lower().replace("_", "-").split())
    normalized = AGENT_PROFILE_ALIASES.get(normalized, normalized)
    if normalized in _KNOWN_AGENTS or normalized == "all":
        return normalized
    return "generic"


def _agent_matches(candidate_agent: str, requested: ResumeAgent) -> bool:
    if requested == "all":
        return True
    return _normalize_agent_name(candidate_agent) == requested


def _resolve_cwd(value: str) -> str:
    if not value:
        return ""
    with contextlib.suppress(OSError):
        return str(Path(value).expanduser().resolve())
    return str(Path(value).expanduser())


def _same_cwd(left: str, right: str) -> bool:
    if not left or not right:
        return False
    return _resolve_cwd(left) == _resolve_cwd(right)


def _sort_key(value: str) -> str:
    return value or ""


def _file_updated_at(path: Path) -> str:
    with contextlib.suppress(OSError):
        return datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat(timespec="seconds").replace(
            "+00:00",
            "Z",
        )
    return ""


def _relative_path(root: Path, path: Path) -> str:
    with contextlib.suppress(ValueError):
        return path.relative_to(root).as_posix()
    return str(path)


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


def _clip(text: str, limit: int = 120) -> str:
    clean = " ".join(str(text or "").split())
    if len(clean) <= limit:
        return clean
    if limit <= 3:
        return clean[:limit]
    return clean[: max(0, limit - 3)].rstrip() + "..."


def _jsonable(value: object) -> object:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return str(value)


def _format_relative_time(value: str, *, now: datetime | None = None) -> str:
    raw = (value or "").strip()
    if not raw:
        return ""
    try:
        parsed = datetime.fromisoformat(raw[:-1] + "+00:00") if raw.endswith("Z") else datetime.fromisoformat(raw)
    except ValueError:
        return ""
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    reference = (now or datetime.now(UTC)).astimezone(UTC)
    delta = int((reference - parsed).total_seconds())
    if delta < 0:
        delta = 0
    if delta < 60:
        return "now"
    minutes = delta // 60
    if minutes < 60:
        return f"{minutes}m ago"
    hours = minutes // 60
    if hours < 24:
        return f"{hours}h ago"
    days = hours // 24
    if days < 30:
        return f"{days}d ago"
    months = days // 30
    if months < 12:
        return f"{months}mo ago"
    years = days // 365
    return f"{years}y ago"
