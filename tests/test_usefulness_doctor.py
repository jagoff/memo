"""Trust + adoption doctor report derivation."""

from __future__ import annotations

from pathlib import Path

from memo.config import Config
from memo.dashboard import append_recall_log
from memo.usefulness_doctor import build_report, format_text_report


def _cfg(tmp_path: Path) -> Config:
    data_dir = tmp_path / "data"
    state_dir = tmp_path / "state"
    cfg = Config(data_dir=data_dir, state_dir=state_dir)
    cfg.ensure_dirs()
    return cfg


def test_doctor_reports_silent_consumers(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    append_recall_log(
        cfg.state_dir,
        prompt="what did we decide about memo",
        hits=[{"id": "a" * 8, "score": 0.91, "title": "Memo decision"}],
        via="daemon",
        client="codex",
    )

    report = build_report(cfg, limit=100)

    assert report["verdict"] == "degraded"
    silent = [i for i in report["adoption"] if i["id"] == "silent_consumers"]
    assert silent
    assert "memflow" in silent[0]["evidence"]["silent"]
    assert any("source" in a["command"] for a in report["actions"])


def test_doctor_reports_unattributed_mcp_consults(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    append_recall_log(
        cfg.state_dir,
        prompt="search from an mcp client",
        hits=[{"id": "b" * 8, "score": 0.82, "title": "Search result"}],
        via="mcp:search",
    )

    report = build_report(cfg, limit=100)

    item = next(i for i in report["adoption"] if i["id"] == "unattributed_consults")
    assert item["severity"] == "warning"
    assert item["evidence"]["count"] == 1
    assert item["action"] == 'Pass source="<client>" on memo read tool calls.'


def test_doctor_text_report_is_action_oriented(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    append_recall_log(
        cfg.state_dir,
        prompt="what did we decide about retrieval",
        hits=[{"id": "c" * 8, "score": 0.86, "title": "Retrieval decision"}],
        via="daemon",
        client="codex",
    )

    text = format_text_report(build_report(cfg, limit=100))

    assert "memo trust + adoption doctor" in text
    assert "verdict:" in text
    assert "adoption" in text
    assert "action:" in text
