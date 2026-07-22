"""presence_today.json — today's activity counters for the statusline (F2)."""

import json
from datetime import date
from pathlib import Path

from memo import presence


def test_read_missing_file_returns_zeros(tmp_path: Path) -> None:
    data = presence.read_today(tmp_path)
    assert data == {"date": date.today().isoformat(), "recalls": 0, "saves": 0, "tokens_saved": 0}


def test_bump_increments_and_persists(tmp_path: Path) -> None:
    presence.bump(tmp_path, recalls=3)
    presence.bump(tmp_path, recalls=2, saves=1)
    data = presence.read_today(tmp_path)
    assert data["recalls"] == 5
    assert data["saves"] == 1
    on_disk = json.loads(presence.presence_path(tmp_path).read_text(encoding="utf-8"))
    assert on_disk["recalls"] == 5


def test_set_tokens_overwrites_not_increments(tmp_path: Path) -> None:
    presence.set_tokens(tmp_path, 8000)
    presence.set_tokens(tmp_path, 9500)
    assert presence.read_today(tmp_path)["tokens_saved"] == 9500


def test_stale_date_rolls_over_to_zero(tmp_path: Path) -> None:
    presence.presence_path(tmp_path).write_text(
        json.dumps({"date": "2020-01-01", "recalls": 99, "saves": 9, "tokens_saved": 999}),
        encoding="utf-8",
    )
    data = presence.read_today(tmp_path)
    assert data["recalls"] == 0 and data["date"] == date.today().isoformat()


def test_corrupt_file_regenerates_and_never_raises(tmp_path: Path) -> None:
    presence.presence_path(tmp_path).write_text("{not json", encoding="utf-8")
    assert presence.read_today(tmp_path)["recalls"] == 0
    presence.bump(tmp_path, recalls=1)  # must not raise
    assert presence.read_today(tmp_path)["recalls"] == 1


def test_writers_never_raise_on_unwritable_dir(tmp_path: Path) -> None:
    target = tmp_path / "nope"
    target.write_text("file, not dir", encoding="utf-8")  # mkdir will fail under it
    presence.bump(target / "sub", recalls=1)  # must swallow
    presence.set_tokens(target / "sub", 5)  # must swallow


def test_summary_line_empty_when_no_activity() -> None:
    assert presence.summary_line({"recalls": 0, "saves": 0, "tokens_saved": 0}) == ""


def test_summary_line_renders_nonzero_segments() -> None:
    line = presence.summary_line({"recalls": 3, "saves": 1, "tokens_saved": 2500})
    assert line.startswith("※ memo today · ")
    assert "🧠 3 recalled" in line
    assert "💾 1 saved" in line
    assert "~2k tok" in line


def test_summary_line_omits_zero_segments() -> None:
    assert presence.summary_line({"recalls": 2, "saves": 0, "tokens_saved": 0}) == (
        "※ memo today · 🧠 2 recalled"
    )


def test_summary_line_tokens_below_1k_shown_raw() -> None:
    assert "~500 tok" in presence.summary_line({"recalls": 0, "saves": 0, "tokens_saved": 500})


def test_rollover_then_bump_resets_counter(tmp_path: Path) -> None:
    """Stale-date file is discarded; new bump starts from 0, not stale+new."""
    presence.presence_path(tmp_path).write_text(
        json.dumps({"date": "2020-01-01", "recalls": 99, "saves": 9, "tokens_saved": 999}),
        encoding="utf-8",
    )
    presence.bump(tmp_path, recalls=1)
    data = presence.read_today(tmp_path)
    assert data["date"] == date.today().isoformat()
    assert data["recalls"] == 1  # not 100 (rollover discarded the stale data)


def test_save_bumps_presence_counter(mock_memory) -> None:
    """Memory.save() → saves counter +1 (choke point for CLI/MCP/capture)."""
    mock_memory.save(content="presence counter smoke test — durable fact", title="presence smoke")
    assert presence.read_today(mock_memory.cfg.state_dir)["saves"] == 1
