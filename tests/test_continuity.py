"""memo continuity — native flow_continuity parity over session snapshots."""

from __future__ import annotations

from memo.session import render_continuity


def _rows():
    return [
        {
            "cwd": "/repos/memo",
            "session_id": "abc12345",
            "branch": "master",
            "turn_count": 7,
            "summary": "arreglando el sync replay",
            "running_summary": "Se trabajó en el fix de sync y el instalador MCP.",
            "modified_files": ["src/memo/session.py", "src/memo/cli_session.py"],
            "last_assistant_tail": "Cerré el ajuste y dejé el resumen listo.",
            "prompt_trail": ["primer loop", "segundo loop"],
            "updated": "2026-06-16T10:00:00+00:00",
        },
        {"cwd": "/other", "session_id": "zzz", "branch": "x"},
    ]


def test_render_continuity_for_matching_cwd():
    out = render_continuity(_rows(), "/repos/memo")
    assert "What you were doing" in out
    assert "- **Summary**: Se trabajó en el fix de sync y el instalador MCP." in out
    assert "`master`" in out and "Turns**: 7" in out
    assert "claude --resume abc12345" in out
    assert "Active memory" in out
    assert "Files touched" in out and "cli_session.py" in out
    assert "Last reply" in out and "Cerré el ajuste" in out
    assert "Open loops (session)" in out and "segundo loop" in out


def test_render_continuity_no_prior_session_for_cwd():
    assert render_continuity(_rows(), "/nowhere") == "No previous session in this directory."


def test_render_continuity_empty_rows():
    assert "No previous session" in render_continuity([], "/repos/memo")
