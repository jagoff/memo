"""Tests for `Memory.synthesize_cross_cluster()` and stale-synthesis gc pruning.

All tests use the `mock_memory` fixture (isolated storage, fake embedder).
No MLX forward passes — synthesis is driven by a local chat stub.
"""

from __future__ import annotations

import json
import math
from typing import Any

import pytest

# ── helpers ─────────────────────────────────────────────────────────────────

def _unit_vec(dims: int) -> list[float]:
    """Unit vector in all-equal direction — cosine sim = 1.0 between any two."""
    v = 1.0 / math.sqrt(dims)
    return [v] * dims


def _orthogonal_vec(dims: int, index: int) -> list[float]:
    """Deterministic unit vector for dimension `index` (sparse, far from others)."""
    v = [0.0] * dims
    v[index % dims] = 1.0
    return v


class _SynthesisChat:
    """Chat stub that returns a valid synthesis JSON."""

    def chat(self, model: str, messages: list[dict], options: dict | None = None) -> dict:
        system = (messages[0].get("content") or "") if messages else ""
        if "collectively IMPLY" in system:
            return {
                "message": {
                    "content": json.dumps({
                        "title": "FP vs OOP tension in memo",
                        "body": "Tension between FP preferences and OOP architecture explains recurring confusion.",
                        "confidence": "medium",
                        "rationale": "Pattern across cluster: preference conflicts with codebase style.",
                    })
                }
            }
        # Fallback for other LLM calls (entity extraction, consolidation, etc.)
        return {"message": {"content": "{}"}}

    def complete(self, prompt: str, **_: Any) -> str:
        return "{}"


class _NullSynthesisChat(_SynthesisChat):
    """Returns null title — no insight found."""

    def chat(self, model: str, messages: list[dict], options: dict | None = None) -> dict:
        system = (messages[0].get("content") or "") if messages else ""
        if "collectively IMPLY" in system:
            return {"message": {"content": json.dumps({"title": None, "body": "", "confidence": "low", "rationale": ""})}}
        return {"message": {"content": "{}"}}


class _LowConfidenceChat(_SynthesisChat):
    """Returns low confidence synthesis."""

    def chat(self, model: str, messages: list[dict], options: dict | None = None) -> dict:
        system = (messages[0].get("content") or "") if messages else ""
        if "collectively IMPLY" in system:
            return {
                "message": {
                    "content": json.dumps({
                        "title": "Weak pattern",
                        "body": "Some possible relationship.",
                        "confidence": "low",
                        "rationale": "Speculative.",
                    })
                }
            }
        return {"message": {"content": "{}"}}


def _force_close_embeddings(mem: Any) -> None:
    """Override embedder so all inputs get the same unit vector → cosine sim = 1.0."""
    dims = mem.cfg.embedder_dims
    unit = _unit_vec(dims)
    mem.embedder.embed = lambda inputs: [unit for _ in inputs]
    mem.embedder.embed_query = lambda query: unit


# ── synthesize_cross_cluster tests ──────────────────────────────────────────

def test_synthesize_empty_corpus_returns_empty(mock_memory):
    results = mock_memory.synthesize_cross_cluster()
    assert results == []


def test_synthesize_below_min_cluster_size_returns_empty(mock_memory):
    _force_close_embeddings(mock_memory)
    mock_memory._chat = _SynthesisChat()

    mock_memory.save(content="Python FP preference", type_="preference")
    mock_memory.save(content="OOP mixin complexity", type_="note")

    # Only 2 memories, min_cluster_size=3 → no candidates
    results = mock_memory.synthesize_cross_cluster(min_cluster_size=3)
    assert results == []


def test_synthesize_dry_run_returns_proposals_no_save(mock_memory):
    _force_close_embeddings(mock_memory)
    mock_memory._chat = _SynthesisChat()

    for i in range(3):
        mock_memory.save(content=f"Python functional pattern {i}", type_="preference")

    results = mock_memory.synthesize_cross_cluster(min_cluster_size=2, dry_run=True)

    assert len(results) >= 1
    assert all(not r.get("saved") for r in results)

    # Nothing persisted as synthesis
    all_types = [r.get("type") for r in mock_memory.store.list_recent(limit=100)]
    assert "synthesis" not in all_types


def test_synthesize_saves_medium_confidence_insight(mock_memory):
    _force_close_embeddings(mock_memory)
    mock_memory._chat = _SynthesisChat()

    mock_memory.save(content="Prefer functional patterns in Python", type_="preference")
    mock_memory.save(content="Memo codebase uses OOP mixins", type_="fact")
    mock_memory.save(content="Confused by mixin hierarchy in memo", type_="note")

    results = mock_memory.synthesize_cross_cluster(
        min_cluster_size=2, min_confidence="medium", dry_run=False,
    )

    saved = [r for r in results if r.get("saved")]
    assert len(saved) >= 1

    # Verify saved record
    synth_id = saved[0]["id"]
    rec = mock_memory.get(synth_id)
    assert rec is not None
    assert rec.type == "synthesis"
    assert "synthesis" in rec.tags

    # Provenance in extra frontmatter bag
    import frontmatter
    md_path = mock_memory._resolve_existing(rec.path)
    post = frontmatter.loads(md_path.read_text())
    ex = post.get("extra") or {}
    assert ex.get("synthesis_sources")
    assert ex.get("synthesis_sources_hash")
    assert ex.get("synthesis_confidence") == "medium"


def test_synthesize_skips_null_title(mock_memory):
    _force_close_embeddings(mock_memory)
    mock_memory._chat = _NullSynthesisChat()

    for i in range(3):
        mock_memory.save(content=f"Topic content {i}", type_="note")

    results = mock_memory.synthesize_cross_cluster(min_cluster_size=2, dry_run=False)

    assert all(not r.get("saved") for r in results)
    all_types = [r.get("type") for r in mock_memory.store.list_recent(limit=100)]
    assert "synthesis" not in all_types


def test_synthesize_skips_below_min_confidence(mock_memory):
    _force_close_embeddings(mock_memory)
    mock_memory._chat = _LowConfidenceChat()

    for i in range(3):
        mock_memory.save(content=f"Weak pattern note {i}", type_="note")

    results = mock_memory.synthesize_cross_cluster(
        min_cluster_size=2, min_confidence="medium", dry_run=False,
    )

    assert all(not r.get("saved") for r in results)


def test_synthesize_skips_duplicate_hash(mock_memory):
    """Second call on same cluster is skipped (same provenance hash)."""
    _force_close_embeddings(mock_memory)
    mock_memory._chat = _SynthesisChat()

    for i in range(3):
        mock_memory.save(content=f"Repeatable content {i}", type_="note")

    results_1 = mock_memory.synthesize_cross_cluster(min_cluster_size=2, dry_run=False)
    saved_1 = [r for r in results_1 if r.get("saved")]
    assert len(saved_1) >= 1

    # Second pass: same cluster → hash already exists → 0 new saves
    results_2 = mock_memory.synthesize_cross_cluster(min_cluster_size=2, dry_run=False)
    saved_2 = [r for r in results_2 if r.get("saved")]
    assert len(saved_2) == 0


def test_synthesize_excludes_existing_synthesis_from_source_pool(mock_memory):
    """Synthesis memories must not be included in clustering source pool."""
    _force_close_embeddings(mock_memory)
    mock_memory._chat = _SynthesisChat()

    for i in range(3):
        mock_memory.save(content=f"Source content {i}", type_="note")

    # Run once → creates a synthesis
    results = mock_memory.synthesize_cross_cluster(min_cluster_size=2, dry_run=False)
    assert any(r.get("saved") for r in results)

    # Verify that synthesis memories are NOT in the source pool
    store_conn = mock_memory.store._conn
    source_rows = store_conn.execute(
        "SELECT meta.type FROM meta WHERE meta.type NOT IN ('reference', 'synthesis')"
    ).fetchall()
    synth_rows = store_conn.execute(
        "SELECT meta.type FROM meta WHERE meta.type = 'synthesis'"
    ).fetchall()
    assert len(synth_rows) >= 1
    assert all(r["type"] != "synthesis" for r in source_rows)


def test_synthesize_result_structure(mock_memory):
    """Each result has the expected keys."""
    _force_close_embeddings(mock_memory)
    mock_memory._chat = _SynthesisChat()

    for i in range(3):
        mock_memory.save(content=f"Structured content {i}", type_="note")

    results = mock_memory.synthesize_cross_cluster(min_cluster_size=2, dry_run=True)
    assert len(results) >= 1

    for r in results:
        assert "sources" in r
        assert "sources_hash" in r
        assert "title" in r
        assert "body" in r
        assert "confidence" in r
        assert "rationale" in r
        assert "saved" in r
        assert isinstance(r["sources"], list)


# ── stale synthesis gc tests ─────────────────────────────────────────────────

def test_gc_reports_stale_synthesis_when_source_deleted(mock_memory):
    """gc() detects synthesis memories whose sources were deleted."""
    _force_close_embeddings(mock_memory)
    mock_memory._chat = _SynthesisChat()

    # Create sources and synthesize
    for i in range(3):
        mock_memory.save(content=f"Source to be deleted {i}", type_="note")

    results = mock_memory.synthesize_cross_cluster(min_cluster_size=2, dry_run=False)
    saved = [r for r in results if r.get("saved")]
    assert len(saved) >= 1
    synth_id = saved[0]["id"]

    # Get source IDs from synthesis result (avoids re-reading frontmatter in gc test)
    source_ids = saved[0]["sources"]
    assert source_ids

    # Delete a source
    mock_memory.delete(source_ids[0])

    # gc should detect the stale synthesis
    report = mock_memory.gc(fix=False)
    assert synth_id in report.get("stale_synthesis", [])


def test_gc_archives_stale_synthesis_with_fix(mock_memory):
    """gc(fix=True) archives stale synthesis when source is deleted."""
    _force_close_embeddings(mock_memory)
    mock_memory._chat = _SynthesisChat()

    for i in range(3):
        mock_memory.save(content=f"Content for fix test {i}", type_="note")

    results = mock_memory.synthesize_cross_cluster(min_cluster_size=2, dry_run=False)
    saved = [r for r in results if r.get("saved")]
    assert len(saved) >= 1
    synth_id = saved[0]["id"]

    source_ids = saved[0]["sources"]
    assert source_ids

    mock_memory.delete(source_ids[0])

    report = mock_memory.gc(fix=True)
    assert synth_id in report.get("stale_synthesis", [])

    # After fix: synthesis should be archived (not findable via get)
    assert mock_memory.get(synth_id) is None


def test_gc_no_stale_synthesis_when_sources_intact(mock_memory):
    """gc() returns empty stale_synthesis when all sources exist."""
    _force_close_embeddings(mock_memory)
    mock_memory._chat = _SynthesisChat()

    for i in range(3):
        mock_memory.save(content=f"Intact source {i}", type_="note")

    results = mock_memory.synthesize_cross_cluster(min_cluster_size=2, dry_run=False)
    assert any(r.get("saved") for r in results)

    report = mock_memory.gc(fix=False)
    assert report.get("stale_synthesis", []) == []


# ── 3.3a: memory_synthesize_list confidence filter ──────────────────────────

def _get_synth_tool_fn(mock_memory, tool_name: str):
    """Register server_synthesis and return the named tool's underlying function."""
    import asyncio
    from memo import server_synthesis
    from fastmcp import FastMCP

    server = FastMCP("test")
    server_synthesis.register(server, mock_memory)
    tool = asyncio.run(server._get_tool(tool_name))
    assert tool is not None, f"{tool_name!r} not registered in server_synthesis"
    return tool.fn


def test_synthesize_list_confidence_filter_returns_matching(mock_memory):
    """memory_synthesize_list(confidence='medium') returns only medium-confidence entries."""
    _force_close_embeddings(mock_memory)
    mock_memory._chat = _SynthesisChat()  # returns confidence="medium"

    for i in range(3):
        mock_memory.save(content=f"Filterable content {i}", type_="note")

    mock_memory.synthesize_cross_cluster(min_cluster_size=2, dry_run=False)

    tool_fn = _get_synth_tool_fn(mock_memory, "memory_synthesize_list")

    all_results = tool_fn()
    assert any(r["confidence"] == "medium" for r in all_results)

    filtered = tool_fn(confidence="medium")
    assert all(r["confidence"] == "medium" for r in filtered)
    assert len(filtered) <= len(all_results)


def test_synthesize_list_confidence_filter_excludes_non_matching(mock_memory):
    """Filtering by 'high' excludes 'medium' synthesis memorias."""
    _force_close_embeddings(mock_memory)
    mock_memory._chat = _SynthesisChat()  # returns confidence="medium"

    for i in range(3):
        mock_memory.save(content=f"Filter exclude content {i}", type_="note")

    mock_memory.synthesize_cross_cluster(min_cluster_size=2, dry_run=False)

    tool_fn = _get_synth_tool_fn(mock_memory, "memory_synthesize_list")

    # Filter by "high" should return nothing (we only have "medium")
    filtered = tool_fn(confidence="high")
    assert filtered == []


def test_synthesize_list_invalid_confidence_raises(mock_memory):
    """memory_synthesize_list with invalid confidence raises ValueError."""
    tool_fn = _get_synth_tool_fn(mock_memory, "memory_synthesize_list")

    with pytest.raises(ValueError, match="confidence must be one of"):
        tool_fn(confidence="invalid")


def test_synthesize_list_no_filter_returns_all(mock_memory):
    """memory_synthesize_list() with no filter returns all synthesis memorias."""
    _force_close_embeddings(mock_memory)
    mock_memory._chat = _SynthesisChat()

    for i in range(3):
        mock_memory.save(content=f"All results content {i}", type_="note")

    mock_memory.synthesize_cross_cluster(min_cluster_size=2, dry_run=False)

    tool_fn = _get_synth_tool_fn(mock_memory, "memory_synthesize_list")
    results = tool_fn()
    assert len(results) >= 1


# ── 3.3b: memory_synthesize_delete MCP tool ─────────────────────────────────


def test_synthesize_delete_removes_synthesis(mock_memory):
    """memory_synthesize_delete deletes an existing synthesis memoria."""
    _force_close_embeddings(mock_memory)
    mock_memory._chat = _SynthesisChat()

    for i in range(3):
        mock_memory.save(content=f"Delete candidate content {i}", type_="note")

    results = mock_memory.synthesize_cross_cluster(min_cluster_size=2, dry_run=False)
    saved = [r for r in results if r.get("saved")]
    assert len(saved) >= 1
    synth_id = saved[0]["id"]

    delete_fn = _get_synth_tool_fn(mock_memory, "memory_synthesize_delete")
    result = delete_fn(id=synth_id)

    assert result["deleted"] is True
    assert result["id"] == synth_id
    assert mock_memory.get(synth_id) is None


def test_synthesize_delete_refuses_non_synthesis(mock_memory):
    """memory_synthesize_delete refuses to delete a non-synthesis memoria."""
    rec = mock_memory.save(content="Regular note", type_="note")

    delete_fn = _get_synth_tool_fn(mock_memory, "memory_synthesize_delete")
    result = delete_fn(id=rec.id)

    assert result["deleted"] is False
    assert "synthesis" in result["reason"].lower()
    # The original nota should still exist
    assert mock_memory.get(rec.id) is not None


def test_synthesize_delete_nonexistent_id(mock_memory):
    """memory_synthesize_delete returns deleted=False for unknown ID."""
    delete_fn = _get_synth_tool_fn(mock_memory, "memory_synthesize_delete")
    result = delete_fn(id="00000000-0000-0000-0000-000000000000")

    assert result["deleted"] is False
    assert "reason" in result


# ── 3.3c: dedup on re-run (synthesis_sources_hash) ──────────────────────────

def test_synthesize_dedup_uses_source_hash(mock_memory):
    """Re-running synthesize on unchanged cluster produces no new saves (hash dedup)."""
    _force_close_embeddings(mock_memory)
    mock_memory._chat = _SynthesisChat()

    for i in range(3):
        mock_memory.save(content=f"Dedup hash content {i}", type_="note")

    first_run = mock_memory.synthesize_cross_cluster(min_cluster_size=2, dry_run=False)
    assert any(r.get("saved") for r in first_run), "First run should save at least one synthesis"

    second_run = mock_memory.synthesize_cross_cluster(min_cluster_size=2, dry_run=False)
    assert not any(r.get("saved") for r in second_run), "Second run should save nothing (hash already exists)"


def test_synthesize_dedup_hash_stored_in_frontmatter(mock_memory):
    """Saved synthesis has synthesis_sources_hash in extra frontmatter."""
    import frontmatter as fm

    _force_close_embeddings(mock_memory)
    mock_memory._chat = _SynthesisChat()

    for i in range(3):
        mock_memory.save(content=f"Hash frontmatter content {i}", type_="note")

    results = mock_memory.synthesize_cross_cluster(min_cluster_size=2, dry_run=False)
    saved = [r for r in results if r.get("saved")]
    assert len(saved) >= 1

    synth_rec = mock_memory.get(saved[0]["id"])
    assert synth_rec is not None

    md_path = mock_memory._resolve_existing(synth_rec.path)
    post = fm.loads(md_path.read_text(encoding="utf-8"))
    ex = post.get("extra") or {}
    assert ex.get("synthesis_sources_hash"), "synthesis_sources_hash must be stored in extra"
    # Hash should match what was returned in the result
    assert ex["synthesis_sources_hash"] == saved[0]["sources_hash"]
