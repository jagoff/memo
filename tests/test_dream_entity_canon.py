"""Tests for dream_entity_canon — MinHash-blocked LLM entity canonicalization."""

from __future__ import annotations

from memo import dream_entity_canon
from memo.dream_entity_canon import run_entity_canon


class _Graph:
    def __init__(self, rows):
        self.rows = rows
        self.canon_called = 0
        self.merges: list[tuple[int, int, str]] = []

    def canonicalize_existing(self):
        self.canon_called += 1
        return 0

    def list_entities(self, *, min_mentions=1):
        return [dict(r) for r in self.rows]

    def merge_entity_pair(self, canonical_id, dup_id, dup_name):
        self.merges.append((canonical_id, dup_id, dup_name))


class _Mem:
    def __init__(self, rows):
        self.graph = _Graph(rows)


_ROWS = [
    {"id": 1, "name": "memo recall daemon", "type": "technology", "mention_count": 5},
    {"id": 2, "name": "memo recall daemons", "type": "technology", "mention_count": 1},
    {"id": 3, "name": "kubernetes networking", "type": "technology", "mention_count": 2},
]


def test_disabled_by_default(monkeypatch):
    monkeypatch.delenv("MEMO_DREAM_ENTITY_CANON_ENABLED", raising=False)
    assert run_entity_canon(None, None)["status"] == "disabled"


def test_blocking_cuts_llm_calls_and_merges(monkeypatch):
    monkeypatch.setenv("MEMO_DREAM_ENTITY_CANON_ENABLED", "1")
    monkeypatch.setattr(dream_entity_canon, "_llm_same_entity", lambda mem, a, b: True)
    mem = _Mem(_ROWS)
    res = run_entity_canon(None, mem)
    assert res["status"] == "done"
    assert mem.graph.canon_called == 1  # exact fold_key pass ran first
    assert res["pairs_naive"] == 3  # 3 entities → 3 all-pairs LLM calls without blocking
    assert res["pairs_blocked"] == 1  # only the daemon variants share an LSH bucket
    assert res["llm_calls"] == 1  # the measured saving: 1 call instead of 3
    assert mem.graph.merges == [(1, 2, "memo recall daemons")]  # keep = more mentions


def test_llm_no_means_no_merge(monkeypatch):
    monkeypatch.setenv("MEMO_DREAM_ENTITY_CANON_ENABLED", "1")
    monkeypatch.setattr(dream_entity_canon, "_llm_same_entity", lambda mem, a, b: False)
    mem = _Mem(_ROWS)
    res = run_entity_canon(None, mem)
    assert res["llm_calls"] == 1
    assert mem.graph.merges == []
    assert res["merged"] == []


def test_dry_run_merges_nothing(monkeypatch):
    monkeypatch.setenv("MEMO_DREAM_ENTITY_CANON_ENABLED", "1")
    monkeypatch.setattr(dream_entity_canon, "_llm_same_entity", lambda mem, a, b: True)
    mem = _Mem(_ROWS)
    res = run_entity_canon(None, mem, dry_run=True)
    assert mem.graph.merges == []
    assert res["merged"] and res["merged"][0]["dry_run"] is True
    # Regression: canonicalize_existing() must NOT be called in dry_run —
    # it commits five categories of destructive mutations to graph.db.
    assert mem.graph.canon_called == 0, (
        "dry_run must not call canonicalize_existing() — it writes to graph.db"
    )


def test_max_pairs_caps_llm_calls(monkeypatch):
    monkeypatch.setenv("MEMO_DREAM_ENTITY_CANON_ENABLED", "1")
    monkeypatch.setattr(dream_entity_canon, "_llm_same_entity", lambda mem, a, b: False)
    mem = _Mem(_ROWS)
    res = run_entity_canon(None, mem, max_pairs=0)
    assert res["llm_calls"] == 0


def test_cli_entity_canon_disabled_smoke(tmp_path, monkeypatch):
    from click.testing import CliRunner

    from memo.cli_dream import dream_cmd

    monkeypatch.setenv("MEMO_NONINTERACTIVE", "1")
    monkeypatch.setenv("MEMO_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MEMO_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.delenv("MEMO_DREAM_ENTITY_CANON_ENABLED", raising=False)
    res = CliRunner().invoke(dream_cmd, ["entity-canon", "--json"])
    assert res.exit_code == 0, res.output
    assert '"disabled"' in res.output
