"""Trust + adoption doctor report derivation."""

from __future__ import annotations

import ast
import json
import sqlite3
from contextlib import closing
from pathlib import Path

from memo.config import Config
from memo.dashboard import append_grounding_log, append_recall_log, recall_log_path
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
    assert "claude-code" in silent[0]["evidence"]["silent"]
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


def test_doctor_module_does_not_import_memo_memory() -> None:
    src = Path("src/memo/usefulness_doctor.py").read_text(encoding="utf-8")
    tree = ast.parse(src)

    forbidden = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "memo.memory":
            forbidden.append(node.module)
        if isinstance(node, ast.Import):
            forbidden.extend(alias.name for alias in node.names if alias.name == "memo.memory")

    assert forbidden == []


def test_doctor_missing_db_stays_read_only(tmp_path: Path) -> None:
    cfg = Config(data_dir=tmp_path / "data", state_dir=tmp_path / "state")

    report = build_report(cfg, limit=25)

    item = next(i for i in report["trust"] if i["id"] == "store_unavailable")
    assert item["status"] == "unknown"
    assert item["evidence"]["db_path"] == str(cfg.db_path)
    assert not cfg.state_dir.exists()
    assert not cfg.db_path.exists()


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
    ids: list[str] = []
    with closing(Memory(cfg)) as mem:
        for n in range(25):
            rec = mem.save(content=f"fact {n}", title=f"Fact {n}", defer_embed=True)
            ids.append(rec.id)
        mem.store.set_confidence_batch([(id_, 1.0) for id_ in ids])

    report = build_report(cfg, limit=100)

    item = next(i for i in report["trust"] if i["id"] == "support_count_starvation")
    assert item["severity"] == "warning"
    assert item["evidence"]["memory_health_rows"] == 25
    assert item["evidence"]["support_count_positive"] == 0


def test_doctor_reports_missing_support_count_schema(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    cfg.state_dir.mkdir(parents=True, exist_ok=True)
    cfg.db_path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(cfg.db_path)) as conn:
        conn.execute(
            "CREATE TABLE memory_health("
            "id TEXT PRIMARY KEY, confidence REAL, roi_score REAL, updated_at TEXT)"
        )
        conn.execute(
            "INSERT INTO memory_health(id, confidence, roi_score, updated_at) "
            "VALUES('abc12345', 1.0, 1.0, '2026-07-08T00:00:00')"
        )
        conn.commit()

    report = build_report(cfg, limit=100)

    item = next(i for i in report["trust"] if i["id"] == "schema_missing")
    assert item["status"] == "unknown"
    assert item["evidence"] == {"table": "memory_health", "missing_column": "support_count"}


def test_doctor_reports_trusted_memories_not_used(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    with closing(Memory(cfg)) as mem:
        used = mem.save(content="used supported fact", title="Used", defer_embed=True)
        unused = mem.save(content="unused supported fact", title="Unused", defer_embed=True)
        mem.store.bump_support_batch([used.id, used.id, used.id, unused.id, unused.id, unused.id])
    append_grounding_log(
        cfg.state_dir,
        session_id="s-trusted",
        turn=1,
        recall_id=used.id[:8],
        used_score=0.9,
        method="test",
    )

    report = build_report(cfg, limit=100)

    item = next(i for i in report["trust"] if i["id"] == "trusted_memories_not_used")
    assert item["severity"] == "warning"
    assert item["evidence"]["count"] == 1
    assert item["evidence"]["memories"][0]["id"] == unused.id[:8]


def test_doctor_reports_invalidated_grounded_memory(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    with closing(Memory(cfg)) as mem:
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


def test_doctor_parses_grounded_memory_json_defensively(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    with closing(Memory(cfg)) as mem:
        rec = mem.save(
            content="seguimos en vite",
            title="Bundler",
            extra={"superseded_by": "vite"},
            defer_embed=True,
        )
    append_grounding_log(
        cfg.state_dir,
        session_id="s2",
        turn=1,
        recall_id=rec.id[:8],
        used_score=0.9,
        method="test",
    )

    with closing(sqlite3.connect(cfg.db_path)) as conn:
        conn.execute("UPDATE meta SET tags = ?, extra_json = ? WHERE id = ?", ("{}", "[]", rec.id))
        conn.commit()

    report = build_report(cfg, limit=100)

    assert all(item["id"] != "untrusted_memories_grounded" for item in report["trust"])


def test_doctor_skips_malformed_recall_rows(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    path = recall_log_path(cfg.state_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"via": "mcp:search"}\nnot-json\n', encoding="utf-8")

    report = build_report(cfg, limit=100)

    assert sorted(report) == ["actions", "adoption", "summary", "trust", "verdict"]
    assert report["summary"]["malformed_rows"] == 1


def test_doctor_malformed_rows_respects_limit(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    path = recall_log_path(cfg.state_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('not-json-old\n{"via": "daemon"}\nnot-json-new\n', encoding="utf-8")

    report = build_report(cfg, limit=2)

    assert report["summary"]["malformed_rows"] == 1


def test_doctor_json_is_serializable_under_empty_state(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)

    payload = json.dumps(build_report(cfg, limit=100), ensure_ascii=False)

    assert '"verdict"' in payload
    assert '"actions"' in payload
