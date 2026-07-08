"""Trust + adoption doctor report derivation."""

from __future__ import annotations

from pathlib import Path

from memo.config import Config
from memo.dashboard import append_grounding_log, append_recall_log
from memo.memory import Memory
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


def test_doctor_reports_support_count_starvation(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    mem = Memory(cfg)
    ids: list[str] = []
    for n in range(25):
        rec = mem.save(content=f"fact {n}", title=f"Fact {n}", defer_embed=True)
        ids.append(rec.id)
    mem.store.set_confidence_batch([(id_, 1.0) for id_ in ids])

    report = build_report(cfg, limit=100)

    item = next(i for i in report["trust"] if i["id"] == "support_count_starvation")
    assert item["severity"] == "warning"
    assert item["evidence"]["memory_health_rows"] == 25
    assert item["evidence"]["support_count_positive"] == 0


def test_doctor_reports_invalidated_grounded_memory(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    mem = Memory(cfg)
    rec = mem.save(
        content="usamos webpack",
        title="Bundler",
        tags=["_invalidated"],
        extra={"invalidated_reason": "migramos a vite"},
        defer_embed=True,
    )
    append_grounding_log(
        cfg.state_dir,
        session_id="s1",
        turn=1,
        recall_id=rec.id[:8],
        used_score=0.9,
        method="test",
    )

    report = build_report(cfg, limit=100)

    item = next(i for i in report["trust"] if i["id"] == "untrusted_memories_grounded")
    assert item["severity"] == "critical"
    assert item["evidence"]["count"] == 1
    assert item["evidence"]["memories"][0]["id"] == rec.id[:8]
