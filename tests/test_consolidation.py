"""Tests for advanced consolidation module."""

import pytest

from memo.consolidation import (
    AdvancedConsolidator,
    ConsolidationResult,
    MergeProposal,
)


@pytest.fixture
def consolidator(mock_memory):
    """Fixture providing AdvancedConsolidator instance."""
    return AdvancedConsolidator(mock_memory)


def test_consolidator_init(consolidator):
    """Test AdvancedConsolidator initialization."""
    assert consolidator.memory is not None
    assert consolidator._chat is None  # Lazy


def test_propose_merge_unrelated_cluster(consolidator):
    """Test merge proposal for unrelated cluster (should return None)."""
    cluster = {
        "cluster_id": 1,
        "relationship": "unrelated",
        "members": [],
        "summary": "Unrelated items",
    }
    proposal = consolidator.propose_merge(cluster)
    assert proposal is None


def test_propose_merge_evolution_cluster(consolidator, mock_memory):
    """Test merge proposal for evolution cluster (keep_latest strategy)."""
    # Create test memorias with different dates
    rec1 = mock_memory.save(
        content="Old version about MLX",
        title="MLX old",
        tags=["mlx"],
    )
    rec2 = mock_memory.save(
        content="New version about MLX",
        title="MLX new",
        tags=["mlx"],
    )

    cluster = {
        "cluster_id": 1,
        "relationship": "evolution",
        "members": [
            {"id": rec1.id, "title": rec1.title, "updated": rec1.updated, "body_preview": "Old"},
            {"id": rec2.id, "title": rec2.title, "updated": rec2.updated, "body_preview": "New"},
        ],
        "summary": "Evolution of MLX usage",
    }

    proposal = consolidator.propose_merge(cluster)
    assert proposal is not None
    assert proposal.merge_strategy == "keep_latest"
    assert proposal.cluster_id == 1
    assert len(proposal.memory_ids) == 2


def test_apply_merge_dry_run(consolidator):
    """Test merge application in dry-run mode."""
    proposal = MergeProposal(
        cluster_id=1,
        memory_ids=["a", "b"],
        merged_title="Merged",
        merged_body="Merged content",
        merge_strategy="synthesis",
        rationale="Test",
        archived_ids=["a"],
    )

    result = consolidator.apply_merge(proposal, dry_run=True)
    assert result.merged_id is None
    assert result.archived_ids == []
    assert "Dry run" in result.summary


def test_apply_merge_real(consolidator, mock_memory):
    """Test real merge application."""
    # Create test memorias
    rec1 = mock_memory.save(
        content="Content 1",
        title="Title 1",
        tags=["test"],
    )
    rec2 = mock_memory.save(
        content="Content 2",
        title="Title 2",
        tags=["test"],
    )

    proposal = MergeProposal(
        cluster_id=1,
        memory_ids=[rec1.id, rec2.id],
        merged_title="Merged Title",
        merged_body="Merged content",
        merge_strategy="synthesis",
        rationale="Test merge",
        archived_ids=[rec1.id],
    )

    result = consolidator.apply_merge(proposal, dry_run=False)
    assert result.merged_id is not None
    assert len(result.archived_ids) == 1
    assert rec1.id in result.archived_ids


def test_archive_memoria(consolidator, mock_memory):
    """Test archival of a memoria."""
    rec = mock_memory.save(
        content="To be archived",
        title="Archive me",
        tags=["test"],
    )

    replacement_id = "new123"
    success = consolidator._archive_memoria(rec.id, replacement_id)

    assert success is True
    # Check that the original is deleted
    assert mock_memory.get(rec.id) is None
    # Check that archival directory exists
    assert consolidator._archival_dir.is_dir()


def test_consolidate_all_empty_corpus(consolidator):
    """Test full consolidation pipeline with empty corpus."""
    result = consolidator.consolidate_all(
        threshold=0.85,
        max_clusters=20,
        auto_apply=False,
        dry_run=True,
    )

    assert "clusters" in result
    assert "proposals" in result
    assert "results" in result
    assert result["clusters"] == []
    assert result["proposals"] == []


def test_consolidate_all_with_data(consolidator, mock_memory):
    """Test full consolidation pipeline with actual memorias."""
    # Create test memorias
    mock_memory.save(
        content="Test content 1",
        title="Test 1",
        tags=["test"],
    )
    mock_memory.save(
        content="Test content 2",
        title="Test 2",
        tags=["test"],
    )

    result = consolidator.consolidate_all(
        threshold=0.85,
        max_clusters=20,
        auto_apply=False,
        dry_run=True,
    )

    assert "clusters" in result
    assert "proposals" in result


def test_merge_proposal_dataclass():
    """Test MergeProposal dataclass structure."""
    p = MergeProposal(
        cluster_id=1,
        memory_ids=["a", "b"],
        merged_title="Merged",
        merged_body="Content",
        merge_strategy="synthesis",
        rationale="Test",
        archived_ids=["a"],
    )
    assert p.cluster_id == 1
    assert p.merge_strategy == "synthesis"
    assert len(p.memory_ids) == 2


def test_consolidation_result_dataclass():
    """Test ConsolidationResult dataclass structure."""
    r = ConsolidationResult(
        merged_id="new123",
        archived_ids=["old1", "old2"],
        skipped_ids=[],
        summary="Merged successfully",
    )
    assert r.merged_id == "new123"
    assert len(r.archived_ids) == 2
    assert r.summary == "Merged successfully"
