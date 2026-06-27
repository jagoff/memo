from __future__ import annotations

import os
import shutil
import sys
from collections.abc import Sequence

from ._types import (
    _ANSI_RESET,
    _RESUME_AGENT_COLORS,
    ResumeCandidate,
    ResumeDiscoveryReport,
)
from ._utils import (
    _STATUS_BADGES,
    _clip,
    _strip_ansi,
)


def _ansi_color(text: str, color: str) -> str:
    return f"{color}{text}{_ANSI_RESET}"


def _status_badge(status: str, *, color: bool = True) -> str:
    entry = _STATUS_BADGES.get(status)
    if entry is None:
        return ""
    glyph, ansi = entry
    return _ansi_color(glyph, ansi) if color else glyph


def _display_title(candidate: ResumeCandidate) -> str:
    return candidate.summary or candidate.title or candidate.session_id


def _candidate_line(
    index: int,
    candidate: ResumeCandidate,
    *,
    width: int,
    selected: bool = False,
    interactive: bool = False,
    include_cwd: bool = False,
) -> str:
    marker = ">" if selected else " "
    mode = "native" if candidate.resume_mode == "native_resume" else "context"
    if not interactive:
        mode = candidate.resume_mode.replace("_", "-")

    # Build agent tag with color
    agent_tag = f"[{candidate.agent}]"
    color = _RESUME_AGENT_COLORS.get(candidate.agent.lower())
    if color:
        agent_tag = _ansi_color(agent_tag, color)
    badge = _status_badge(candidate.status)

    prefix = f"{marker} {index}. {badge}{agent_tag} {mode} "
    suffix = "" if interactive else f" ({candidate.session_id[:8]})"
    if not interactive and include_cwd and candidate.cwd:
        suffix += f" cwd={candidate.cwd}"
    title = _display_title(candidate)
    max_width = max(1, width - 1)
    # Calculate width without ANSI codes for proper layout
    prefix_display_width = len(_strip_ansi(prefix))
    suffix_display_width = len(suffix)
    with_suffix_budget = max_width - prefix_display_width - suffix_display_width
    if suffix and len(title) <= with_suffix_budget:
        return f"{prefix}{title}{suffix}"
    without_suffix_budget = max_width - prefix_display_width
    if len(title) <= without_suffix_budget:
        return f"{prefix}{title}"
    clipped = _clip(title, max(4, without_suffix_budget))
    # Return without slicing to preserve ANSI codes in prefix
    return f"{prefix}{clipped}"


def format_resume_candidates(report: ResumeDiscoveryReport) -> str:
    width = shutil.get_terminal_size((100, 24)).columns
    lines = ["memo resume candidates"]
    if not report.candidates:
        lines.append("No sessions found.")
    for index, item in enumerate(report.candidates, start=1):
        lines.append(_candidate_line(index, item, width=width, include_cwd=report.include_all_cwd))
    for error in report.provider_errors:
        lines.append(f"note: {error.provider}: {error.detail}")
    return "\n".join(lines)


def format_context_resume(candidate: ResumeCandidate) -> str:
    lines = [
        "Native resume is not available for this candidate.",
        f"agent: {candidate.agent}",
        f"mode: {candidate.resume_mode}",
        f"uri: {candidate.uri}",
    ]
    if candidate.cwd:
        lines.append(f"cwd: {candidate.cwd}")
    if candidate.summary:
        lines.extend(["", candidate.summary])
    if candidate.provenance:
        lines.append("")
        lines.append("provenance:")
        lines.extend(f"- {item}" for item in candidate.provenance)
    lines.append("")
    lines.append("Open the agent in the cwd above and include this context.")
    return "\n".join(lines)


def resolve_resume_candidate(
    candidates: Sequence[ResumeCandidate],
    selector: str,
) -> ResumeCandidate | None:
    needle = selector.strip()
    if not needle:
        return None
    for candidate in candidates:
        if (
            candidate.session_id == needle
            or candidate.session_id.startswith(needle)
            or candidate.uri == needle
            or candidate.uri.endswith(f"/{needle}")
        ):
            return candidate
    return None


def execute_resume_candidate(candidate: ResumeCandidate) -> int:
    if candidate.resume_mode != "native_resume" or not candidate.resume_command:
        print(format_context_resume(candidate))
        return 0
    executable = shutil.which(candidate.resume_command[0])
    if not executable:
        print(f"resume executable not found: {candidate.resume_command[0]}", file=sys.stderr)
        return 1
    argv = [executable, *candidate.resume_command[1:]]
    # Native resume: replace this process with the agent CLI. No shell (execvp,
    # not a shell string), executable is PATH-resolved via shutil.which above.
    os.execvp(executable, argv)  # noqa: S606
    return 1
