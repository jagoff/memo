"""External session state — git + transcript readers for `session.py`.

Lower-level helpers that read the *outside* world to build a session
snapshot: git status/branch/HEAD and the last user / assistant message
from the Claude Code transcript jsonl, plus the command-wrapper noise
filters. Split out of `session.py` to keep both files under the repo's
800-line limit; `session.py` re-imports what its CRUD path needs, so
`from memo.session import read_last_user_msg` etc. keep resolving.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)

_LAST_USER_MSG_CHARS = 240
_MODIFIED_FILES_CAP = 30
_ASSISTANT_TAIL_CHARS = 200


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
        for line in status.splitlines()[:_MODIFIED_FILES_CAP]:
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


# User "turns" in a Claude Code transcript include slash-command plumbing
# (the wrapper tags below), tool_result echoes, and harness-injected blocks
# (task notifications, system reminders) — none of which are a real typed
# prompt. Surfacing them as the session "summary" produces garbage like
# `<local-command-stdout>Enabled plan mode</local-command-stdout>`.
_COMMAND_WRAPPER_PREFIXES = (
    "<command-name>",
    "<command-message>",
    "<command-args>",
    "<local-command-stdout>",
    "<local-command-stderr>",
    "<bash-stdout>",
    "<bash-stderr>",
    "<user-prompt-submit-hook>",
    "<task-notification>",
    "<task-output>",
    "<system-reminder>",
)
_COMMAND_WRAPPER_TAG_RE = re.compile(
    r"<(command-[a-z]+|local-command-[a-z]+|bash-std[a-z]+|user-prompt-submit-hook)>"
    r".*?</\1>",
    re.DOTALL,
)


def _strip_command_wrappers(text: str) -> str:
    """Remove slash-command / local-command / bash wrapper tag pairs so a turn
    that mixes a wrapper with a real prompt still yields the real text."""
    if not text:
        return ""
    return _COMMAND_WRAPPER_TAG_RE.sub("", text).strip()


def is_command_noise(text: str | None) -> bool:
    """True when `text` is slash-command / local-command plumbing rather than a
    real prompt — including a value truncated mid-tag (e.g. an old stored
    summary like `<local-command-stdout>Enabled plan mode</local-command-`),
    which the tag-pair stripper can't repair. Used to heal persisted junk
    summaries at display time and to stop them sticking across checkpoints."""
    if not text:
        return True
    stripped = text.lstrip()
    if stripped.startswith(_COMMAND_WRAPPER_PREFIXES):
        return True
    return not _strip_command_wrappers(stripped)


def read_last_user_msg(transcript_path: Path) -> str | None:
    """Walk the JSONL transcript backwards, return the latest real user
    prompt. Mirrors `capture._read_last_exchange` but only needs the user
    side, so we can stop scanning earlier. Skips meta/compact-summary records
    and tool_result echoes; slash-command wrappers are stripped, and a turn
    that is *only* plumbing (empty after stripping) is skipped.
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
        if obj.get("isMeta") is True or obj.get("isCompactSummary") is True:
            continue
        msg = obj.get("message", obj)
        content = msg.get("content") if isinstance(msg, dict) else None
        text = _extract_text(content)
        if not text:
            continue
        cleaned = _strip_command_wrappers(text)
        if cleaned and not cleaned.startswith(_COMMAND_WRAPPER_PREFIXES):
            return cleaned[:_LAST_USER_MSG_CHARS]
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
