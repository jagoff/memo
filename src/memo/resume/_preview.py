"""Repo-delta + open-loops preview for a resume candidate (Phase 2).

When you highlight a session in the picker (Ctrl+T), this answers "what changed
here since I last worked on it" — commits in the session's cwd since the session,
the current uncommitted state, and the session's open loops (the prompt_trail).
All best-effort: a non-git cwd, a missing snapshot, or a slow git call degrades to
fewer lines, never an error. Kept out of `_tui.py` so the TUI carries no git /
session-store dependency; the picker calls this through an injected callback.
"""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING

from ._types import ResumeCandidate

if TYPE_CHECKING:
    from memo.config import Config


def _git(cwd: str, args: list[str], *, timeout: float = 2.0) -> str:
    try:
        proc = subprocess.run(
            ["git", "-C", cwd, *args],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return proc.stdout if proc.returncode == 0 else ""


def _open_loops(cfg: Config, candidate: ResumeCandidate) -> list[str]:
    """Last few user prompts from the memo session snapshot (memo/claude only)."""
    try:
        from memo.session import get_session

        snap = get_session(cfg.state_dir, candidate.session_id)
        if not snap:
            return []
        trail = [
            str(p).strip()
            for p in (snap.get("prompt_trail") or [])
            if isinstance(p, str) and p.strip()
        ]
        return [p[:100] for p in reversed(trail[-4:])]
    except Exception:
        return []


def session_preview(cfg: Config, candidate: ResumeCandidate) -> list[str]:
    """Preview lines for one candidate: repo delta since the session + open loops.

    Best-effort and bounded (git calls time-limited). Returns ``[]`` when there's
    nothing to show (non-git cwd, no commits, no snapshot).
    """
    lines: list[str] = []
    cwd = candidate.cwd
    if cwd and _git(cwd, ["rev-parse", "--git-dir"]):
        since = candidate.updated_at
        if since:
            log = _git(cwd, ["log", "--oneline", "-n", "30", f"--since={since}"])
            commits = [ln for ln in log.splitlines() if ln.strip()]
            if commits:
                lines.append(f"↑ {len(commits)} commit(s) here since this session:")
                lines.extend(f"    {c}" for c in commits[:5])
                if len(commits) > 5:
                    lines.append(f"    …(+{len(commits) - 5} more)")
        dirty = [ln for ln in _git(cwd, ["status", "--porcelain"]).splitlines() if ln.strip()]
        if dirty:
            lines.append(f"~ {len(dirty)} uncommitted file(s) in the working tree now")

    loops = _open_loops(cfg, candidate)
    if loops:
        if lines:
            lines.append("")
        lines.append("Open loops (recent prompts):")
        lines.extend(f"    {i}. {p}" for i, p in enumerate(loops, 1))
    return lines
