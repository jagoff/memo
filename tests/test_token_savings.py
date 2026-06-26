"""Tests for `memo token-savings` command."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from click.testing import CliRunner

from memo.cli_token_savings import token_savings_cmd
from memo.dashboard_logs import append_context_cost_log, recall_log_path


def _run(state_dir: Path, data_dir: Path) -> tuple[int, str]:
    runner = CliRunner()
    result = runner.invoke(
        token_savings_cmd,
        [],
        env={
            "MEMO_NONINTERACTIVE": "1",
            "MEMO_STATE_DIR": str(state_dir),
            "MEMO_DATA_DIR": str(data_dir),
        },
        catch_exceptions=False,
    )
    return result.exit_code, result.output


def test_token_savings_empty_log(tmp_path: Path) -> None:
    """Command exits 0 with no log entries and prints a helpful message."""
    state = tmp_path / "state"
    state.mkdir()
    data = tmp_path / "data"
    data.mkdir()

    exit_code, output = _run(state, data)

    assert exit_code == 0
    # Either the header or the "no entries" message
    assert "memo token savings" in output or "No recall injections" in output


def test_token_savings_header_no_entries(tmp_path: Path) -> None:
    """With no log file, the no-entries message is shown."""
    state = tmp_path / "state"
    state.mkdir()
    data = tmp_path / "data"
    data.mkdir()

    exit_code, output = _run(state, data)

    assert exit_code == 0
    assert "No recall injections" in output


def test_token_savings_with_recall_entries(tmp_path: Path) -> None:
    """With 3 recall log entries, output contains '3 prompts'."""
    state = tmp_path / "state"
    state.mkdir()
    data = tmp_path / "data"
    data.mkdir()

    for _ in range(3):
        append_context_cost_log(state, kind="recall", chars=1200)

    exit_code, output = _run(state, data)

    assert exit_code == 0
    assert "3" in output
    assert "prompts" in output


def test_token_savings_header_with_entries(tmp_path: Path) -> None:
    """With recall entries, output shows the 'memo token savings' header."""
    state = tmp_path / "state"
    state.mkdir()
    data = tmp_path / "data"
    data.mkdir()

    append_context_cost_log(state, kind="recall", chars=800)

    exit_code, output = _run(state, data)

    assert exit_code == 0
    assert "memo token savings" in output


def test_token_savings_ignores_non_recall_kinds(tmp_path: Path) -> None:
    """Entries with kind != 'recall' are not counted."""
    state = tmp_path / "state"
    state.mkdir()
    data = tmp_path / "data"
    data.mkdir()

    # Add a non-recall entry
    append_context_cost_log(state, kind="briefing", chars=500)

    exit_code, output = _run(state, data)

    assert exit_code == 0
    assert "No recall injections" in output


def _append_bail_entry(state: Path, reason: str = "trivial prompt") -> None:
    """Write a bail entry to the recall log (no session_id → recall.log only)."""
    path = recall_log_path(state)
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "ts": datetime.now(UTC).isoformat(timespec="seconds"),
        "via": "bail",
        "reason": reason,
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def test_token_savings_trivial_bails_counted(tmp_path: Path) -> None:
    """With 3 recall entries + 2 trivial bail entries, output shows 2 prompts skipped."""
    state = tmp_path / "state"
    state.mkdir()
    data = tmp_path / "data"
    data.mkdir()

    for _ in range(3):
        append_context_cost_log(state, kind="recall", chars=1200)
    for _ in range(2):
        _append_bail_entry(state, reason="trivial prompt")

    exit_code, output = _run(state, data)

    assert exit_code == 0
    assert "Trivial bails:" in output
    assert "2" in output
    assert "prompts skipped" in output


def test_token_savings_estimated_total_shown(tmp_path: Path) -> None:
    """With recall entries, output contains 'Estimated total:' line."""
    state = tmp_path / "state"
    state.mkdir()
    data = tmp_path / "data"
    data.mkdir()

    for _ in range(2):
        append_context_cost_log(state, kind="recall", chars=1600)

    exit_code, output = _run(state, data)

    assert exit_code == 0
    assert "Estimated total:" in output


def test_token_savings_compact_savings_shows_potential_when_format_not_set(
    tmp_path: Path,
) -> None:
    """When MEMO_RECALL_FORMAT is unset (default 'full'), compact savings line shows 'potential'."""
    state = tmp_path / "state"
    state.mkdir()
    data = tmp_path / "data"
    data.mkdir()

    append_context_cost_log(state, kind="recall", chars=1200)

    runner = CliRunner()
    result = runner.invoke(
        token_savings_cmd,
        [],
        env={
            "MEMO_NONINTERACTIVE": "1",
            "MEMO_STATE_DIR": str(state),
            "MEMO_DATA_DIR": str(data),
            "MEMO_RECALL_FORMAT": "",  # unset → default 'full'
        },
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    assert "potential" in result.output
