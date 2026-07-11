"""Render memo's real cross-agent resume picker with seeded sample sessions.

Exercises the actual `pick_resume_candidate_interactive` TUI (real layout,
colors, agent tags, status badges) so the README screenshot shows the true
surface — just with illustrative, public sample data instead of the dev's
private sessions. VHS drives the keys and captures a frame.
"""
from __future__ import annotations

import os
import sys
from datetime import UTC, datetime, timedelta

from memo.resume import pick_resume_candidate_interactive
from memo.resume._types import ResumeCandidate

_NOW = datetime.now(UTC)


def _ago(**kw: int) -> str:
    return (_NOW - timedelta(**kw)).isoformat()


CWD = os.path.expanduser("~/repos/memo")

SAMPLES: list[ResumeCandidate] = [
    ResumeCandidate(
        agent="claude", provider="claude-native", uri="claude://s/a1",
        session_id="aa7b84c0", title="", summary="fix save-gate preset parsing",
        cwd=CWD, updated_at=_ago(minutes=2), status="active",
        resume_command=["claude", "--resume", "aa7b84c0"],
    ),
    ResumeCandidate(
        agent="codex", provider="codex-native", uri="codex://s/c2",
        session_id="019f51e9", title="", summary="port BM25 Spanish tokenizer tests",
        cwd=CWD, updated_at=_ago(minutes=41), status="recent",
        resume_command=["codex", "resume", "019f51e9"],
    ),
    ResumeCandidate(
        agent="gemini", provider="gemini-native", uri="gemini://s/g3",
        session_id="7f3c9d10", title="", summary="trace federation receipt bug",
        cwd=os.path.expanduser("~/repos/synapse"), updated_at=_ago(hours=1),
        status="recent", resume_command=["gemini", "--resume", "7f3c9d10"],
    ),
    ResumeCandidate(
        agent="devin", provider="devin-native", uri="devin://s/d4",
        session_id="04c1172d", title="", summary="cross-Mac sync rebase loop",
        cwd=os.path.expanduser("~/repos/memflow"), updated_at=_ago(hours=3),
        status="stale", resume_command=[],
    ),
    ResumeCandidate(
        agent="opencode", provider="opencode-native", uri="opencode://s/o5",
        session_id="2dd1b1d8", title="", summary="docker CPU-backend smoke test",
        cwd=CWD, updated_at=_ago(hours=5), status="recent",
        resume_command=["opencode", "--session", "2dd1b1d8"],
    ),
    ResumeCandidate(
        agent="claude", provider="memo", uri="memo://session/e6",
        session_id="15c89574", title="", summary="git-sync ahead/behind self-heal",
        cwd=os.path.expanduser("~/repos/consciousness-contracts"),
        updated_at=_ago(hours=9), status="recent",
        resume_command=["claude", "--resume", "15c89574"],
    ),
    ResumeCandidate(
        agent="codex", provider="codex-native", uri="codex://s/c7",
        session_id="8b132f30", title="", summary="reranker head-slice perf win",
        cwd=CWD, updated_at=_ago(days=1), status="recent",
        resume_command=["codex", "resume", "8b132f30"],
    ),
    ResumeCandidate(
        agent="claude", provider="memo", uri="memo://session/e8",
        session_id="6fab8c90", title="", summary="eval regression labels update",
        cwd=CWD, updated_at=_ago(days=2), status="recent",
        resume_command=["claude", "--resume", "6fab8c90"],
    ),
]


def main() -> int:
    chosen = pick_resume_candidate_interactive(
        SAMPLES, current_cwd=CWD, start_filter="all",
    )
    # The screenshot is captured before Enter; this only runs if a key selects.
    if chosen is not None:
        sys.stdout.write(f"would resume: {' '.join(chosen.resume_command)}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
