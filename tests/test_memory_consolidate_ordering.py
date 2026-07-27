"""Regression tests for consolidation ranking with timezone-aware timestamps."""


def test_consolidate_ranks_candidate_clusters_by_utc_instant(mock_memory, monkeypatch):
    """The newest cluster should win even when its timestamp string sorts earlier."""
    mem = mock_memory

    # Two duplicate clusters with the same size. The first cluster has a
    # lexicographically later timestamp string, but an earlier UTC instant.
    items = []
    for idx in range(5):
        items.append(
            {
                "id": f"a{idx + 1}",
                "title": "Older cluster",
                "type": "note",
                "tags": [],
                "path": f"a{idx + 1}.md",
                "updated": "2026-01-01T22:00:00+14:00",
                "emb": [1.0, 0.0, 0.0, 0.0],
            }
        )
    for idx in range(5):
        items.append(
            {
                "id": f"b{idx + 1}",
                "title": "Newer cluster",
                "type": "note",
                "tags": [],
                "path": f"b{idx + 1}.md",
                "updated": "2026-01-01T23:30:00-10:00",
                "emb": [0.0, 1.0, 0.0, 0.0],
            }
        )

    monkeypatch.setattr(mem, "_pull_embeddings", lambda **kwargs: items)
    monkeypatch.setattr(mem, "_read_body", lambda path: f"body:{path}")

    result = mem.consolidate(threshold=0.85, max_clusters=1, skip_llm=True)

    assert len(result) == 1
    assert result[0]["members"][0]["id"] == "b1"


def test_greedy_cluster_pure_python_fallback_assigns_best_match(monkeypatch):
    """F5: the numpy-less fallback must join the CLOSEST representative (argmax),
    matching the numpy path — not merely the first representative over threshold."""
    import sys

    from memo.memory.consolidate_ops import _ConsolidateOpsMixin

    # Force the pure-Python branch by making `import numpy` raise ImportError.
    monkeypatch.setitem(sys.modules, "numpy", None)

    # v2 (35°) is over threshold to BOTH v0 (0°, cos35≈0.819) and v1 (60°,
    # cos25≈0.906) but is CLOSER to v1. First-match would wrongly join v0's
    # cluster; best-match joins v1's.
    items = [
        {"emb": [1.0, 0.0]},  # 0°
        {"emb": [0.5, 0.8660254]},  # 60° (cos to v0 = 0.5 < threshold → own cluster)
        {"emb": [0.81915204, 0.57357644]},  # 35°
    ]
    clusters = _ConsolidateOpsMixin._greedy_cluster(items, threshold=0.78)

    assert clusters == [[0], [1, 2]]
