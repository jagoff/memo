"""Install seed — one real memory saved at install, surfaced once (F4)."""

import json
from datetime import date, timedelta
from pathlib import Path

from memo.briefing import install_seed_lines


def _write_stamp(state_dir: Path, *, ts: str, shown: bool = False) -> None:
    (state_dir / ".install_seed.json").write_text(
        json.dumps({"id": "a1b2c3d4e5f60789", "ts": ts, "shown": shown}),
        encoding="utf-8",
    )


def test_fresh_seed_renders_once(tmp_path: Path) -> None:
    _write_stamp(tmp_path, ts=date.today().isoformat())
    lines = install_seed_lines(tmp_path)
    joined = "\n".join(lines)
    assert "memo remembers" in joined
    assert "a1b2c3d4" in joined
    assert install_seed_lines(tmp_path) == []  # shown flipped — one-shot


def test_already_shown_is_empty(tmp_path: Path) -> None:
    _write_stamp(tmp_path, ts=date.today().isoformat(), shown=True)
    assert install_seed_lines(tmp_path) == []


def test_old_seed_is_skipped(tmp_path: Path) -> None:
    _write_stamp(tmp_path, ts=(date.today() - timedelta(days=30)).isoformat())
    assert install_seed_lines(tmp_path) == []


def test_missing_or_corrupt_stamp_is_empty(tmp_path: Path) -> None:
    assert install_seed_lines(tmp_path) == []
    (tmp_path / ".install_seed.json").write_text("{corrupt", encoding="utf-8")
    assert install_seed_lines(tmp_path) == []


def test_seed_helper_is_idempotent(tmp_path: Path, monkeypatch) -> None:
    """_seed_install_memory: existing stamp → no second save."""
    from memo import cli_install_mcp

    _write_stamp(tmp_path, ts=date.today().isoformat())
    calls: list[str] = []

    class _FakeCfg:
        state_dir = tmp_path

    monkeypatch.setattr(
        "memo.config.Config.from_env", classmethod(lambda cls: _FakeCfg())
    )
    # Memory must never be constructed when the stamp exists.
    monkeypatch.setattr(
        "memo.memory.Memory", lambda cfg: calls.append("constructed") or None
    )
    cli_install_mcp._seed_install_memory()
    assert calls == []
