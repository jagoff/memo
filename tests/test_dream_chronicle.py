"""Tests for the nightly chronicle dream pass."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path


class _Cfg:
    """Minimal cfg fake — same shape test_dream_profile.py uses."""

    def __init__(self, tmp_path):
        self.memory_dir = tmp_path / "memories"
        self.state_dir = tmp_path / "state"
        self.helper_model = "stub-model"


def _mk_cfg(tmp_path):
    cfg = _Cfg(tmp_path)
    cfg.state_dir.mkdir(parents=True, exist_ok=True)
    return cfg


def test_chronicle_flags_registered_default_off():
    from memo.flags import REGISTRY

    for name in ("MEMO_DREAM_CHRONICLE_ENABLED", "MEMO_CHRONICLE_WEEKLY"):
        assert name in REGISTRY
        assert REGISTRY[name].default is False


def test_chronicle_path_lives_in_underscore_bucket(tmp_path):
    from memo import dream_chronicle as dc

    cfg = _mk_cfg(tmp_path)
    p = dc.chronicle_path(cfg, "2026-07-13")
    assert p == Path(cfg.memory_dir) / "_chronicle" / "2026-07-13.md"


def test_default_day_is_previous_day_before_6am():
    from memo import dream_chronicle as dc

    # dream corre 03:00 — la crónica es del día que acaba de terminar
    assert dc.default_day(datetime(2026, 7, 14, 3, 0)) == "2026-07-13"
    assert dc.default_day(datetime(2026, 7, 14, 15, 0)) == "2026-07-14"


def test_filter_cited_drops_uncited_and_fabricated_bullets():
    from memo import dream_chronicle as dc

    text = (
        "## Trabajo\n"
        "- fixed the sync race [aaaaaaaa]\n"
        "- invented claim with no citation\n"
        "- claim citing unknown id [ffffffff]\n"
        "- two real ids [aaaaaaaa] [bbbbbbbb]\n"
    )
    out, ratio = dc.filter_cited(text, {"aaaaaaaa", "bbbbbbbb"})
    assert "sync race" in out
    assert "two real ids" in out
    assert "invented claim" not in out
    assert "unknown id" not in out
    assert "## Trabajo" in out  # headings siempre pasan
    assert ratio == 0.5  # 2 de 4 bullets sobrevivieron


def test_filter_cited_no_bullets_is_ratio_one():
    from memo import dream_chronicle as dc

    out, ratio = dc.filter_cited("just prose\n", {"aaaaaaaa"})
    assert out == "just prose\n" or out == "just prose"
    assert ratio == 1.0


def _write_memory_md(root: Path, mid: str, day: str, title: str, mtype: str = "decision"):
    p = root / f"{title.replace(' ', '-')}.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        f"---\nid: {mid}\ntype: {mtype}\ncreated: {day}T10:00:00\n---\n# {title}\nbody\n",
        encoding="utf-8",
    )
    return p


def test_memories_created_on_filters_by_day_and_skips_buckets(tmp_path):
    from memo import dream_chronicle as dc

    cfg = _mk_cfg(tmp_path)
    cfg.memory_dir.mkdir(parents=True)
    _write_memory_md(cfg.memory_dir, "a" * 32, "2026-07-13", "hit today")
    _write_memory_md(cfg.memory_dir, "b" * 32, "2026-07-12", "old one")
    # bucket files (_profile/_chronicle) are never memories
    _write_memory_md(cfg.memory_dir / "_profile", "c" * 32, "2026-07-13", "profile doc")

    out = dc._memories_created_on(cfg, "2026-07-13")
    assert [m["id"][:8] for m in out] == ["aaaaaaaa"]
    assert out[0]["title"] == "hit today"
    assert out[0]["type"] == "decision"


def test_collect_facts_and_fact_lines(tmp_path, monkeypatch):
    import json as _json

    from memo import dream_chronicle as dc

    cfg = _mk_cfg(tmp_path)
    cfg.memory_dir.mkdir(parents=True)
    day = "2026-07-13"
    _write_memory_md(cfg.memory_dir, "a" * 32, day, "decided X")

    class _FakeStore:
        def recent(self, limit=200):
            return [
                {"agent": "claude-code", "session_id": "d" * 32, "cwd": "/x",
                 "updated_at": f"{day}T20:00:00", "summary": "fixed sync race", "turn_count": 12},
                {"agent": "claude-code", "session_id": "e" * 32, "cwd": "/x",
                 "updated_at": "2026-07-11T20:00:00", "summary": "other day", "turn_count": 3},
            ]

    monkeypatch.setattr("memo.resume._index.open_store", lambda cfg: _FakeStore())

    # receipt previo con actividad
    d = cfg.state_dir / "dream"
    d.mkdir(parents=True)
    (d / "last.json").write_text(_json.dumps({"superseded": ["x"], "merged": [], "errors": []}))

    facts = dc.collect_facts(cfg, day)
    assert len(facts["episodes"]) == 1
    assert facts["episodes"][0]["summary"] == "fixed sync race"
    assert [m["id"][:8] for m in facts["new_memories"]] == ["aaaaaaaa"]
    assert facts["receipt_events"] == {"superseded": 1}
    assert facts["grounded"] == 0  # no grounding.log en tmp

    lines, allowed = dc.fact_lines(facts)
    assert allowed == {"dddddddd", "aaaaaaaa"}
    assert any("[dddddddd]" in ln for ln in lines)
    assert any("[aaaaaaaa]" in ln for ln in lines)


def test_memories_created_on_rich_frontmatter_and_yaml_title(tmp_path):
    from memo import dream_chronicle as dc

    cfg = _mk_cfg(tmp_path)
    cfg.memory_dir.mkdir(parents=True)
    extra_lines = "\n".join(f"  k{i}: v{i}" for i in range(18))
    p = cfg.memory_dir / "rich.md"
    p.write_text(
        "---\n"
        f"id: {'f' * 32}\n"
        "type: decision\n"
        "title: 'titulo desde yaml'\n"
        "extra:\n"
        f"{extra_lines}\n"
        "created: '2026-07-13T10:00:00-03:00'\n"
        "---\n"
        "body sin heading\n",
        encoding="utf-8",
    )
    out = dc._memories_created_on(cfg, "2026-07-13")
    assert [m["id"][:8] for m in out] == ["ffffffff"]
    assert out[0]["title"] == "titulo desde yaml"
