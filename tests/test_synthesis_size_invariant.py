"""Per-topic size invariant + body cap in cross-cluster synthesis (K2)."""

from __future__ import annotations

from memo.memory.consolidate_ops import _ConsolidateOpsMixin


def _unit(dim: int, i: int) -> list[float]:
    v = [0.0] * dim
    v[i] = 1.0
    return v


def _items(embs):
    return [
        {
            "id": f"m{i}",
            "title": f"t{i}",
            "type": "note",
            "tags": [],
            "path": f"p{i}.md",
            "updated": "2026-07-01",
            "emb": e,
        }
        for i, e in enumerate(embs)
    ]


def test_split_disabled_passthrough():
    items = _items([_unit(4, 0)] * 6)
    clusters = [[0, 1, 2, 3, 4, 5]]
    out = _ConsolidateOpsMixin._split_oversized_clusters(
        items, clusters, max_members=0, threshold=0.78
    )
    assert out == clusters


def test_split_reclusters_glued_subtopics():
    # 3 identical on axis 0 + 3 on axis 1, hand-glued into ONE oversized cluster:
    # the tighter re-cluster must separate the two subtopics.
    items = _items([_unit(4, 0)] * 3 + [_unit(4, 1)] * 3)
    clusters = [[0, 1, 2, 3, 4, 5]]
    out = _ConsolidateOpsMixin._split_oversized_clusters(
        items, clusters, max_members=4, threshold=0.78
    )
    assert sorted(sorted(c) for c in out) == [[0, 1, 2], [3, 4, 5]]


def test_split_guarantees_invariant_on_unsplittable_cluster():
    # All identical: a tighter threshold cannot split → hard slicing must bound size.
    items = _items([_unit(4, 0)] * 7)
    clusters = [[0, 1, 2, 3, 4, 5, 6]]
    out = _ConsolidateOpsMixin._split_oversized_clusters(
        items, clusters, max_members=3, threshold=0.78
    )
    assert all(len(c) <= 3 for c in out)
    assert sorted(i for c in out for i in c) == list(range(7))


def test_split_within_bounds_untouched():
    items = _items([_unit(4, 0)] * 3)
    clusters = [[0, 1, 2]]
    out = _ConsolidateOpsMixin._split_oversized_clusters(
        items, clusters, max_members=5, threshold=0.78
    )
    assert out == clusters


def test_resummarize_body_falls_back_to_truncation(mock_memory, monkeypatch):
    import memo.memory.consolidate_ops as ops

    monkeypatch.setattr(ops, "chat_with_timeout", lambda *a, **k: None)  # LLM timeout
    body = "x" * 500
    out = mock_memory._resummarize_body(object(), body, cap=100)
    assert out == body[:100]


def test_resummarize_body_uses_llm_result_when_under_cap(mock_memory, monkeypatch):
    import memo.memory.consolidate_ops as ops

    monkeypatch.setattr(
        ops, "chat_with_timeout", lambda *a, **k: {"message": {"content": "short summary"}}
    )
    out = mock_memory._resummarize_body(object(), "x" * 500, cap=100)
    assert out == "short summary"


def test_synthesize_applies_size_invariant_when_flag_set(mock_memory, monkeypatch):
    monkeypatch.setenv("MEMO_SYNTHESIS_MAX_MEMBERS", "2")
    calls: dict = {}
    orig = type(mock_memory)._split_oversized_clusters

    def _spy(items, clusters, *, max_members, threshold):
        calls["max_members"] = max_members
        return orig(items, clusters, max_members=max_members, threshold=threshold)

    monkeypatch.setattr(type(mock_memory), "_split_oversized_clusters", staticmethod(_spy))
    for i in range(3):
        mock_memory.save(content=f"same topic body {i}", title=f"n{i}", type_="note")
    mock_memory.synthesize_cross_cluster(dry_run=True, min_cluster_size=2)
    assert calls.get("max_members") == 2
