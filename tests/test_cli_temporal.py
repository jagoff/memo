"""CLI-plumbing tests for `memo temporal timeline`'s plain-text output.

Business logic (`TemporalAnalyzer.build_entity_timeline`) is covered in
test_temporal.py; this only exercises the CLI's rendering of a timeline it's
handed, using a mocked Memory.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from memo.cli import cli
from memo.temporal import EntityTimeline, TimelineEvent


def _env(tmp_path) -> dict[str, str]:
    return {
        "MEMO_DATA_DIR": str(tmp_path / "data"),
        "MEMO_STATE_DIR": str(tmp_path / "state"),
        "MEMO_VAULT_PATH": str(tmp_path / "vault"),
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_EMBEDDER_VIA_DAEMON": "0",
    }


def test_timeline_plain_text_renders_letter_leading_memory_id(tmp_path) -> None:
    """Regression: `[dim][{id}][/dim]` swallowed the id when it started with
    a-f, since the literal brackets around it went unescaped.
    """
    timeline = EntityTimeline(
        entity_name="mlx",
        entity_type="technology",
        first_seen="2026-01-01",
        last_seen="2026-01-02",
        events=[
            TimelineEvent(
                memory_id="ad957d80cccc",
                title="Event with a [bracketed] title",
                date="2026-01-01",
                type="fact",
                snippet="Snippet mentioning [scope] too.",
            )
        ],
    )
    mock_memory = MagicMock()
    mock_memory.temporal.build_entity_timeline.return_value = timeline

    with patch("memo.memory.Memory", return_value=mock_memory):
        result = CliRunner().invoke(cli, ["temporal", "timeline", "mlx"], env=_env(tmp_path))

    assert result.exit_code == 0, result.output
    assert "ad957d80" in result.output
    assert "[bracketed]" in result.output
    assert "[scope]" in result.output


def test_timeline_no_memories_found(tmp_path) -> None:
    mock_memory = MagicMock()
    mock_memory.temporal.build_entity_timeline.return_value = None

    with patch("memo.memory.Memory", return_value=mock_memory):
        result = CliRunner().invoke(cli, ["temporal", "timeline", "nope"], env=_env(tmp_path))

    assert result.exit_code == 0, result.output
    assert "No memories found for entity" in result.output
