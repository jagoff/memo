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
