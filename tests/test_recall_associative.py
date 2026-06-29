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
