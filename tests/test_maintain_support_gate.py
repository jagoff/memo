"""C2 supersede gate: a high-support contradiction loser is flagged for
manual triage (pair stays open) instead of auto-archived."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from memo.cli import cli


def _env(tmp_path: Path) -> dict[str, str]:
    return {
        "MEMO_CONFIG_FILE": str(tmp_path / "memo-config.toml"),
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_DATA_DIR": str(tmp_path / "data"),
        "MEMO_STATE_DIR": str(tmp_path / "state"),
    }


def _receipt(output: str) -> dict:
    return json.loads(output[output.index("{") :])


def _seed_contradiction(mock_memory):
    old = mock_memory.save(content="El puerto del dashboard es 8080", title="Puerto (viejo)")
    new = mock_memory.save(content="El puerto del dashboard es 8765", title="Puerto (nuevo)")
    mock_memory.contradict_store.upsert_open(
        memory_id_a=old.id,
        memory_id_b=new.id,
        relationship="contradiction",
        confidence=0.95,
        rationale="ports differ",
    )
    return old, new


def _run_maintain(mock_memory, env):
    with patch("memo.cli_maintain._get_memory", return_value=mock_memory):
        return CliRunner().invoke(
            cli,
            ["maintain", "--skip-consolidate", "--skip-stale", "--skip-synthesize", "--json"],
            env=env,
        )


def test_high_support_loser_flagged_not_archived(mock_memory, tmp_path):
    old, new = _seed_contradiction(mock_memory)
    # heavily corroborated on BOTH sides so the test is independent of the
    # timestamp tie-break inside _older_id
    mock_memory.store.bump_support_batch([old.id] * 5 + [new.id] * 5)

    env = {**_env(tmp_path), "MEMO_SUPERSEDE_SUPPORT_GATE": "3"}
    result = _run_maintain(mock_memory, env)
    assert result.exit_code == 0, result.output

    receipt = _receipt(result.output)
    assert receipt["flagged_for_review"], receipt
    assert receipt["superseded"] == []
    # nothing archived
    assert mock_memory.get(old.id) is not None
    assert mock_memory.get(new.id) is not None
    # pair stays open for triage
    open_pairs = mock_memory.contradict_store.list_open(min_confidence=0.9)
    assert any(
        {p.memory_id_a, p.memory_id_b} == {old.id, new.id} for p in open_pairs
    )


def test_gate_off_archives_older_side(mock_memory, tmp_path):
    old, new = _seed_contradiction(mock_memory)
    mock_memory.store.bump_support_batch([old.id] * 5 + [new.id] * 5)

    result = _run_maintain(mock_memory, _env(tmp_path))  # no gate env → off
    assert result.exit_code == 0, result.output
    receipt = _receipt(result.output)
    assert receipt["superseded"], receipt
    # exactly one side got archived
    gone = [i for i in (old.id, new.id) if mock_memory.get(i) is None]
    assert len(gone) == 1


def test_maintain_marks_competing_within_margin(mock_memory, monkeypatch):
    from unittest.mock import patch
    from click.testing import CliRunner
    from memo.cli import cli

    monkeypatch.setenv("MEMO_BELIEF_COMPETING", "1")
    monkeypatch.setenv("MEMO_SUPERSEDE_MARGIN", "0.5")
    _seed_contradiction(mock_memory)

    with patch("memo.cli_maintain._get_memory", return_value=mock_memory):
        res = CliRunner().invoke(
            cli,
            ["maintain", "--skip-consolidate", "--skip-stale", "--skip-synthesize", "--json"],
        )
    receipt = _receipt(res.output)
    assert receipt.get("competing"), "expected a competing pair in the maintain receipt"
