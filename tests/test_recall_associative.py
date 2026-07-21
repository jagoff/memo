from datetime import UTC

from memo.recall_assoc import build_nudge


class _Rec:
    def __init__(self, id, title):
        self.id = id
        self.title = title


class _Store:
    def memory_entities(self, mid):
        return {"s1": [{"name": "memory"}], "a1": [{"name": "memory"}]}.get(mid, [])

    def entity_memories(self, name, type_=None):
        return ["s1", "a1"] if name == "memory" else []

    def co_recall_counts(self, anchor, cands):
        return {}


class _Mem:
    """Minimal Memory stand-in exposing .graph and a title lookup by id."""

    def __init__(self):
        self.graph = _Store()

    def get(self, mid):  # title resolver
        return _Rec(mid, f"title-{mid}")


def test_build_nudge_returns_associated_titles(monkeypatch):
    import memo.recall_assoc as ra

    monkeypatch.setattr(ra, "_codegraph_adj", lambda: None)
    monkeypatch.setenv("MEMO_RECALL_ASSOCIATIVE", "1")

    nudge = build_nudge(_Mem(), [_Rec("s1", "seed")])
    ids = {h.id for h in nudge}
    assert "a1" in ids  # associated via shared entity 'memory'
    assert "s1" not in ids  # seed excluded
    assert all(hasattr(h, "title") for h in nudge)


def test_build_nudge_disabled_returns_empty(monkeypatch):
    monkeypatch.setenv("MEMO_RECALL_ASSOCIATIVE", "0")
    assert build_nudge(_Mem(), [_Rec("s1", "seed")]) == []


def test_memo_related_returns_hits_with_via():
    from memo.server_related import related_for

    hits = related_for(_Mem(), "s1", hops=2, limit=5)
    assert any(h["id"] == "a1" and h["via"] == "memory" for h in hits)
    assert all({"id", "title", "via", "activation"} <= set(h) for h in hits)


def test_render_associative_line_appends_nudge():
    """Balanced/compact output can receive the associative line via render_associative_line."""
    from memo.recall_assoc import NudgeItem, render_associative_line

    context = "<memo-recall readonly>\nsome content\n</memo-recall>"
    nudge = [NudgeItem(id="abc12345xyz", title="Test Title", via="memory")]
    result = render_associative_line(context, nudge, token_budget=1000)
    assert "🔗" in result
    assert "[abc12345]" in result
    assert "Test Title" in result
    assert "via memory" in result


def test_render_associative_line_includes_via():
    """The via field is included in the nudge line (unlike the old recall_logic rendering)."""
    from memo.recall_assoc import NudgeItem, render_associative_line

    context = "ctx"
    nudge = [
        NudgeItem(id="aaa11111bbb", title="Alpha", via="entity-foo"),
        NudgeItem(id="bbb22222ccc", title="Beta", via="co-recall"),
    ]
    result = render_associative_line(context, nudge, token_budget=0)
    assert "via entity-foo" in result
    assert "via co-recall" in result
    # Both items joined by "; "
    assert "[aaa11111]" in result
    assert "[bbb22222]" in result


def test_render_associative_line_empty_nudge_unchanged():
    """Empty nudge returns context unchanged."""
    from memo.recall_assoc import render_associative_line

    context = "some context"
    assert render_associative_line(context, [], token_budget=1000) == context


def test_render_associative_line_skips_when_over_budget():
    """Line is dropped when it does not fit the token budget."""
    from memo.recall_assoc import NudgeItem, render_associative_line

    context = "x" * 100  # 100 chars
    nudge = [NudgeItem(id="abc12345", title="Title", via="via")]
    # token_budget=10 → max_chars=40; context already exceeds that
    result = render_associative_line(context, nudge, token_budget=10)
    assert result == context


def test_render_associative_line_no_budget_cap_always_appends():
    """token_budget <= 0 means no cap — always append."""
    from memo.recall_assoc import NudgeItem, render_associative_line

    context = "x" * 10000
    nudge = [NudgeItem(id="abc12345", title="T", via="v")]
    result = render_associative_line(context, nudge, token_budget=0)
    assert "🔗" in result


def test_build_nudge_time_guard_returns_empty(monkeypatch):
    """When the time guard fires before associate(), return []."""
    import time

    import memo.recall_assoc as ra

    monkeypatch.setattr(ra, "_codegraph_adj", lambda: None)
    monkeypatch.setenv("MEMO_RECALL_ASSOCIATIVE", "1")
    # First call sets deadline (returns 0.0), all subsequent calls are well past it
    times = iter([0.0, 1000.0, 1000.0, 1000.0])
    monkeypatch.setattr(time, "monotonic", lambda: next(times))
    nudge = ra.build_nudge(_Mem(), [_Rec("s1", "seed")])
    assert nudge == []


def test_build_nudge_skips_none_records(monkeypatch):
    """Hits whose memory.get() returns None are silently excluded."""
    import memo.recall_assoc as ra

    monkeypatch.setattr(ra, "_codegraph_adj", lambda: None)
    monkeypatch.setenv("MEMO_RECALL_ASSOCIATIVE", "1")

    mem = _Mem()
    original_get = mem.get
    mem.get = lambda mid: None if mid == "a1" else original_get(mid)

    nudge = ra.build_nudge(mem, [_Rec("s1", "seed")])
    assert all(h.id != "a1" for h in nudge)


def test_build_nudge_exception_returns_empty(monkeypatch):
    """Any exception inside build_nudge returns [] instead of raising."""
    import memo.recall_assoc as ra

    monkeypatch.setattr(ra, "_codegraph_adj", lambda: None)
    monkeypatch.setenv("MEMO_RECALL_ASSOCIATIVE", "1")

    def _boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(ra, "associate", _boom)
    nudge = ra.build_nudge(_Mem(), [_Rec("s1", "seed")])
    assert nudge == []


def test_cli_related_json(tmp_path, monkeypatch):
    import sqlite3

    import pytest
    from click.testing import CliRunner

    from memo.cli import cli
    from memo.cli_common import get_memory

    connections = []

    def tracked_get_memory(cfg):
        memory = get_memory(cfg)
        connections.append(memory.store.connection)
        return memory

    monkeypatch.setattr("memo.cli_related._get_memory", tracked_get_memory)

    env = {
        "MEMO_DATA_DIR": str(tmp_path / "d"),
        "MEMO_STATE_DIR": str(tmp_path / "s"),
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_EMBEDDER_VIA_DAEMON": "0",
        "TQDM_DISABLE": "1",
    }
    for p in ("d", "s"):
        (tmp_path / p).mkdir()
    res = CliRunner().invoke(cli, ["related", "nonexistent-x", "--json"], env=env)
    assert res.exit_code == 0
    assert res.output.strip().startswith("[")  # JSON list (empty on no data)
    assert len(connections) == 1
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        connections[0].execute("SELECT 1")


def test_build_nudge_skips_forgotten(monkeypatch):
    import memo.recall_assoc as ra
    from memo.lifecycle import IS_FORGOTTEN_KEY

    monkeypatch.setattr(ra, "_codegraph_adj", lambda: None)
    monkeypatch.setenv("MEMO_RECALL_ASSOCIATIVE", "1")

    class _Store:
        def memory_entities(self, mid):
            return {
                "s1": [{"name": "memory"}],
                "a1": [{"name": "memory"}],
                "a2": [{"name": "memory"}],
            }.get(mid, [])

        def entity_memories(self, name, type_=None):
            return ["s1", "a1", "a2"] if name == "memory" else []

        def co_recall_counts(self, anchor, cands):
            return {}

    class _Rec:
        def __init__(self, id, forgotten=False):
            self.id = id
            self.title = f"t-{id}"
            self.extra = {IS_FORGOTTEN_KEY: True} if forgotten else {}

    class _Mem:
        graph = _Store()

        def get(self, mid):
            return _Rec(mid, forgotten=(mid == "a1"))

    ids = {h.id for h in ra.build_nudge(_Mem(), [_Rec("s1")])}
    assert "a1" not in ids  # soft-forgotten hit dropped
    assert "a2" in ids  # backfilled past the forgotten one


def test_recency_weight_prefers_recent():
    from datetime import datetime

    from memo.recall_assoc import _recency_weight

    today = datetime.now(UTC).isoformat()
    old = datetime(2020, 1, 1, tzinfo=UTC).isoformat()
    assert _recency_weight(today) > _recency_weight(old)
    assert _recency_weight("") == 1.0  # unknown -> neutral
    assert 0.0 < _recency_weight(old) <= 1.0


# --- verified edges drop the "· unverified" framing (dream_edge_verify loop) --


def test_render_associative_line_verified_drops_unverified():
    from memo.recall_assoc import NudgeItem, render_associative_line

    nudge = [NudgeItem(id="aaa11111bbb", title="Alpha", via="x", verified=True)]
    result = render_associative_line("ctx", nudge, token_budget=0)
    assert "via graph" in result
    assert "unverified" not in result


def test_render_associative_line_mixed_or_default_keeps_unverified():
    from memo.recall_assoc import NudgeItem, render_associative_line

    mixed = [
        NudgeItem(id="aaa11111bbb", title="Alpha", via="x", verified=True),
        NudgeItem(id="bbb22222ccc", title="Beta", via="y"),  # default False
    ]
    assert "via graph · unverified" in render_associative_line("ctx", mixed, token_budget=0)
    default = [NudgeItem(id="aaa11111bbb", title="Alpha", via="x")]
    assert "via graph · unverified" in render_associative_line("ctx", default, token_budget=0)


class _VerifiedConn:
    """Fake sqlite conn: one memory↔memory edge (a1, s1) above threshold."""

    def __init__(self):
        self.params = None

    def execute(self, sql, params=()):
        self.params = params

        class _Cur:
            def fetchall(_self):
                return [("a1", "s1")]

        return _Cur()


def test_build_nudge_marks_verified_from_graph_edge(monkeypatch):
    import memo.recall_assoc as ra
    from memo.dream_edge_verify import VERIFIED_CONFIDENCE

    monkeypatch.setattr(ra, "_codegraph_adj", lambda: None)
    monkeypatch.setenv("MEMO_RECALL_ASSOCIATIVE", "1")

    mem = _Mem()
    conn = _VerifiedConn()
    mem.graph._conn = conn
    nudge = ra.build_nudge(mem, [_Rec("s1", "seed")])
    assert any(h.id == "a1" and h.verified for h in nudge)
    # the batch query filters on the SAME shared threshold the nightly pass uses
    assert conn.params == (VERIFIED_CONFIDENCE,)


def test_build_nudge_without_conn_stays_unverified(monkeypatch):
    """Graph stores without a raw _conn (or query errors) degrade to the
    conservative unverified framing."""
    import memo.recall_assoc as ra

    monkeypatch.setattr(ra, "_codegraph_adj", lambda: None)
    monkeypatch.setenv("MEMO_RECALL_ASSOCIATIVE", "1")

    nudge = ra.build_nudge(_Mem(), [_Rec("s1", "seed")])
    assert nudge and all(not h.verified for h in nudge)
