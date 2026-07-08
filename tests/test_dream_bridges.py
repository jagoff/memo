"""Tests for bridge-link synthesis (spec 3, phase 3)."""

from __future__ import annotations

from memo.dream_bridges import bridge_insights, decide_bridges, run_synthesize_bridges


class _Graph:
    def all_weighted_edges(self):
        return [
            ("j", "a1", 1.0),
            ("j", "a2", 1.0),
            ("a1", "a2", 1.0),
            ("j", "b1", 1.0),
            ("j", "b2", 1.0),
            ("b1", "b2", 1.0),
        ]

    def entity_memories(self, name, type_=None):
        return {"j": ["m1"], "a1": ["m2"], "b1": ["m3"]}.get(name, [])


class _Mem:
    graph = _Graph()

    def search(self, *a, **k):
        return []

    def __init__(self):
        self.saved: list[dict] = []

    def save(self, **kwargs):
        self.saved.append(kwargs)


def test_bridge_insights_maps_reps_and_source_memories():
    out = bridge_insights(_Mem(), min_side=2, max_bridges=5)
    assert len(out) == 1
    br = out[0]
    assert br["bridge"] == "j"
    assert set(br["left"]) | set(br["right"]) == {"a1", "a2", "b1", "b2"}
    # all sides tie on degree -> smallest name wins as representative
    assert br["left_rep"] == "a1"
    assert br["right_rep"] == "b1"
    assert set(br["memory_ids"]) == {"m1", "m2", "m3"}


def test_is_junk_anchor():
    from memo.dream_bridges import _is_junk_anchor

    assert _is_junk_anchor("2026-06-23")  # date
    assert _is_junk_anchor("2026")  # bare year
    assert _is_junk_anchor("12345")  # pure number
    assert _is_junk_anchor("archivos")  # generic
    assert _is_junk_anchor("files")
    assert not _is_junk_anchor("synapse")
    assert not _is_junk_anchor("recall hook")


class _JunkGraph:
    def all_weighted_edges(self):
        d = "2026-06-23"  # the joining node is a date -> a junk anchor
        return [
            (d, "a1", 1.0),
            (d, "a2", 1.0),
            ("a1", "a2", 1.0),
            (d, "b1", 1.0),
            (d, "b2", 1.0),
            ("b1", "b2", 1.0),
        ]

    def entity_memories(self, name, type_=None):
        return []


class _JunkMem:
    graph = _JunkGraph()

    def search(self, *a, **k):
        return []


def test_bridge_insights_filters_date_anchor():
    # The only bridge is a date entity -> co-mention artifact, not a real link.
    assert bridge_insights(_JunkMem(), min_side=2) == []


def test_decide_bridges_dedup_dryrun_save_and_fail():
    bridges = [
        {
            "bridge": "j",
            "left": ["a1", "a2"],
            "right": ["b1", "b2"],
            "left_rep": "a1",
            "right_rep": "b1",
            "memory_ids": ["m1"],
        }
    ]

    skip = decide_bridges(
        bridges, synthesize_fn=lambda b: {"title": "T", "body": "B"}, exists_fn=lambda h: True
    )
    assert skip[0]["status"] == "skip_exists"

    dry = decide_bridges(
        bridges,
        synthesize_fn=lambda b: {"title": "T", "body": "B"},
        exists_fn=lambda h: False,
        dry_run=True,
    )
    assert dry[0]["status"] == "would_save"
    assert dry[0]["bridge"] == "j"

    save = decide_bridges(
        bridges, synthesize_fn=lambda b: {"title": "T", "body": "B"}, exists_fn=lambda h: False
    )
    assert save[0]["status"] == "save"
    assert save[0]["provenance"] == ["j", "a1", "b1"]

    fail = decide_bridges(bridges, synthesize_fn=lambda b: None, exists_fn=lambda h: False)
    assert fail[0]["status"] == "synth_failed"


def test_run_synthesize_bridges_disabled_by_default(monkeypatch):
    monkeypatch.delenv("MEMO_DREAM_BRIDGES_ENABLED", raising=False)
    res = run_synthesize_bridges(None, None)  # mem untouched when the flag is off
    assert res["status"] == "disabled"


def test_run_synthesize_bridges_saves_bounded_and_cites(monkeypatch):
    monkeypatch.setenv("MEMO_DREAM_BRIDGES_ENABLED", "1")
    mem = _Mem()
    res = run_synthesize_bridges(None, mem)
    assert res["status"] == "done"
    assert len(mem.saved) == 1  # one durable insight per bridge
    rec = mem.saved[0]
    assert rec["type_"] == "synthesis"
    assert rec["extra"]["synthesis_kind"] == "bridge"
    # names the link + carries provenance + cited source memories
    assert "via j" in rec["content"]
    assert rec["extra"]["synthesis_sources"] == ["j", "a1", "b1"]
    assert set(rec["extra"]["synthesis_source_memories"]) == {"m1", "m2", "m3"}
    # provenance hash is embedded for dedup
    assert "[bridge " in rec["content"]


def test_run_synthesize_bridges_dedups_on_existing(monkeypatch):
    monkeypatch.setenv("MEMO_DREAM_BRIDGES_ENABLED", "1")

    class _Hit:
        def __init__(self, body):
            self.body = body

    class _DedupMem(_Mem):
        def search(self, query, **k):
            # echo the phash embedded in the query back as an existing memory
            phash = query.split()[-1]
            return [_Hit(f"prior insight [bridge {phash}]")]

    mem = _DedupMem()
    res = run_synthesize_bridges(None, mem)
    assert res["status"] == "done"
    assert mem.saved == []  # already exists -> skipped
    assert all(d["status"] == "skip_exists" for d in res["synthesized"])
