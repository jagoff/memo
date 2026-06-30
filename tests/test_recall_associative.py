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
    assert "a1" in ids          # associated via shared entity 'memory'
    assert "s1" not in ids      # seed excluded
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
    assert "vía memory" in result


def test_render_associative_line_includes_via():
    """The via field is included in the nudge line (unlike the old recall_logic rendering)."""
    from memo.recall_assoc import NudgeItem, render_associative_line

    context = "ctx"
    nudge = [
        NudgeItem(id="aaa11111bbb", title="Alpha", via="entity-foo"),
        NudgeItem(id="bbb22222ccc", title="Beta", via="co-recall"),
    ]
    result = render_associative_line(context, nudge, token_budget=0)
    assert "vía entity-foo" in result
    assert "vía co-recall" in result
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


def test_cli_related_json(tmp_path):
    from click.testing import CliRunner

    from memo.cli import cli

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
