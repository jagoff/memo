"""dream_consolidate — cross-session clustering + consolidation decisions (pure core)."""

from __future__ import annotations

from memo import dream_consolidate as dc


def _ep(sid, cwd, summary="did work"):
    return {"session_id": sid, "cwd": cwd, "summary": summary}


def test_project_key_is_basename():
    assert dc._project_key("/Users/fer/repos/memo/") == "memo"
    assert dc._project_key("") == ""


def test_provenance_hash_is_order_independent():
    assert dc.provenance_hash(["a", "b"]) == dc.provenance_hash(["b", "a"])


def test_cluster_requires_min_distinct_sessions():
    eps = [
        _ep("s1", "/r/memo"),
        _ep("s2", "/r/memo"),
        _ep("s3", "/r/synapse"),  # only 1 session → dropped
        _ep("s1", "/r/memo"),  # duplicate session id, not a new session
    ]
    clusters = dc.cluster_by_project(eps, min_sessions=2)
    assert len(clusters) == 1
    assert clusters[0]["project"] == "memo"
    assert clusters[0]["session_ids"] == ["s1", "s2"]


def test_consolidate_skips_existing_and_synthesizes_new():
    clusters = [
        {"project": "memo", "episodes": [_ep("s1", "/r/memo")], "session_ids": ["s1", "s2"]},
        {"project": "synapse", "episodes": [_ep("s3", "/r/synapse")], "session_ids": ["s3", "s4"]},
    ]
    # synapse already consolidated; memo is new
    synapse_hash = dc.provenance_hash(["s3", "s4"])
    decisions = dc.consolidate_clusters(
        clusters,
        synthesize_fn=lambda cl: {"title": f"theme {cl['project']}", "body": "recurring insight"},
        exists_fn=lambda h: h == synapse_hash,
        dry_run=False,
    )
    by_proj = {d["project"]: d for d in decisions}
    assert by_proj["synapse"]["status"] == "exists"
    assert by_proj["memo"]["status"] == "save"
    assert by_proj["memo"]["provenance"] == ["s1", "s2"]


def test_consolidate_dry_run_marks_would_save():
    clusters = [
        {"project": "memo", "episodes": [_ep("s1", "/r/memo")], "session_ids": ["s1", "s2"]}
    ]
    decisions = dc.consolidate_clusters(
        clusters,
        synthesize_fn=lambda cl: {"title": "t", "body": "b"},
        exists_fn=lambda h: False,
        dry_run=True,
    )
    assert decisions[0]["status"] == "would_save"


def test_consolidate_skips_when_synthesize_returns_none():
    clusters = [
        {"project": "memo", "episodes": [_ep("s1", "/r/memo")], "session_ids": ["s1", "s2"]}
    ]
    decisions = dc.consolidate_clusters(
        clusters, synthesize_fn=lambda cl: None, exists_fn=lambda h: False, dry_run=False
    )
    assert decisions[0]["status"] == "skipped"


def test_run_consolidate_saves_with_keyword_content(monkeypatch):
    """Orchestrator must call mem.save(content=...) — the real save is keyword-only.
    A positional body would TypeError (the v2.3.11 bug this regression-guards)."""
    saved: list[dict] = []

    class _Store:
        def count(self):
            return 2

        def recent(self, limit=50):
            return [_ep("s1", "/r/memo", "a"), _ep("s2", "/r/memo", "b")]

    class _Mem:
        def search(self, q, limit=1, disable_reranker=True):
            return []

        def save(self, *, content, type, title, extra):  # keyword-only, like the real facade
            saved.append({"content": content, "type": type, "title": title})

    class _Cfg:
        state_dir = "/tmp/unused"

    monkeypatch.setattr("memo.resume._index.open_store", lambda cfg: _Store())
    monkeypatch.setattr(dc, "_llm_synthesize", lambda mem, cl: {"title": "T", "body": "insight"})

    res = dc.run_consolidate_episodes(_Cfg(), _Mem(), min_sessions=2, dry_run=False)
    assert res["status"] == "done"
    assert res["consolidated"][0]["status"] == "saved"
    assert len(saved) == 1
    assert saved[0]["type"] == "synthesis"
    assert "insight" in saved[0]["content"]
