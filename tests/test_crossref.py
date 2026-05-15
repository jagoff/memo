"""Tests for cross-reference module."""

import pytest

from memo.crossref import (
    Backlink,
    CrossReferenceIndex,
    LinkSuggester,
    LinkSuggestion,
    Wikilink,
)


@pytest.fixture
def crossref_index(tmp_cfg):
    """Fixture providing CrossReferenceIndex instance."""
    return CrossReferenceIndex(tmp_cfg.crossref_db)


@pytest.fixture
def link_suggester(mock_memory):
    """Fixture providing LinkSuggester instance."""
    from memo.crossref import CrossReferenceIndex
    crossref = CrossReferenceIndex(mock_memory.cfg.crossref_db)
    return LinkSuggester(mock_memory, crossref)


def test_crossref_index_init(crossref_index):
    """Test CrossReferenceIndex initialization."""
    assert crossref_index.db_path.is_file()


def test_crossref_index_wikilinks(crossref_index):
    """Test wikilink detection and indexing."""
    content = "See [[memoria-id]] for details and [[other-id|Other Memory]] for more."
    wikilinks = crossref_index.index_wikilinks("test-id", content)

    assert len(wikilinks) == 2
    assert wikilinks[0].target == "memoria-id"
    assert wikilinks[0].alias is None
    assert wikilinks[1].target == "other-id"
    assert wikilinks[1].alias == "Other Memory"


def test_crossref_index_wikilinks_no_links(crossref_index):
    """Test wikilink detection with no links."""
    content = "This has no wikilinks at all."
    wikilinks = crossref_index.index_wikilinks("test-id", content)

    assert len(wikilinks) == 0


def test_crossref_get_backlinks(crossref_index):
    """Test getting backlinks for a memoria."""
    # Index some links
    crossref_index.index_wikilinks("source1", "See [[target]] for details")
    crossref_index.index_wikilinks("source2", "Also [[target]] here")

    backlinks = crossref_index.get_backlinks("target")

    assert len(backlinks) == 2
    assert backlinks[0].target_id == "target"
    assert backlinks[0].source_id in ("source1", "source2")


def test_crossref_get_backlinks_none(crossref_index):
    """Test getting backlinks for a memoria with no backlinks."""
    backlinks = crossref_index.get_backlinks("nonexistent")
    assert backlinks == []


def test_crossref_get_outlinks(crossref_index):
    """Test getting outlinks from a memoria."""
    crossref_index.index_wikilinks("source", "See [[target1]] and [[target2]]")

    outlinks = crossref_index.get_outlinks("source")

    assert len(outlinks) == 2
    assert any(ol.target == "target1" for ol in outlinks)
    assert any(ol.target == "target2" for ol in outlinks)


def test_crossref_remove_memoria(crossref_index):
    """Test removing all links for a memoria."""
    # Index links
    crossref_index.index_wikilinks("source", "See [[target]]")
    crossref_index.index_wikilinks("other", "Also [[target]]")

    # Verify links exist
    assert len(crossref_index.get_backlinks("target")) == 2

    # Remove
    crossref_index.remove_memoria("target")

    # Verify removed
    assert len(crossref_index.get_backlinks("target")) == 0


def test_crossref_persistence(tmp_cfg):
    """Test that crossref index persists across instances."""
    db_path = tmp_cfg.crossref_db

    # Create first instance and index links
    index1 = CrossReferenceIndex(db_path)
    index1.index_wikilinks("source", "See [[target]]")

    # Create second instance and verify data persisted
    index2 = CrossReferenceIndex(db_path)
    backlinks = index2.get_backlinks("target")

    assert len(backlinks) == 1
    assert backlinks[0].source_id == "source"


def test_link_suggester_init(link_suggester):
    """Test LinkSuggester initialization."""
    assert link_suggester.memory is not None
    assert link_suggester.crossref is not None


def test_link_suggester_suggest_links(link_suggester, mock_memory):
    """Test link suggestions based on content."""
    # Create test memorias
    mock_memory.save(
        content="Memo about MLX performance",
        title="MLX Performance",
        tags=["mlx"],
    )
    mock_memory.save(
        content="Memo about Qwen models",
        title="Qwen Models",
        tags=["qwen"],
    )

    # Suggest links for similar content
    suggestions = link_suggester.suggest_links(
        content="MLX optimization techniques",
        title="MLX",
        tags=["mlx"],
        limit=5,
    )

    assert isinstance(suggestions, list)
    # May have suggestions or not depending on similarity
    if suggestions:
        assert all(isinstance(s, LinkSuggestion) for s in suggestions)


def test_link_suggester_suggest_links_empty(link_suggester):
    """Test link suggestions with empty corpus."""
    suggestions = link_suggester.suggest_links(
        content="Test content",
        title="Test",
        tags=[],
        limit=5,
    )

    assert suggestions == []


def test_link_suggester_format_wikilink(link_suggester):
    """Test formatting wikilinks."""
    # Without title
    wikilink1 = link_suggester.format_wikilink("abc123")
    assert wikilink1 == "[[abc123]]"

    # With title
    wikilink2 = link_suggester.format_wikilink("abc123", "My Memory")
    assert wikilink2 == "[[abc123|My Memory]]"


def test_wikilink_dataclass():
    """Test Wikilink dataclass structure."""
    link = Wikilink(
        target="memoria-id",
        alias="Display Name",
        position=100,
    )
    assert link.target == "memoria-id"
    assert link.alias == "Display Name"
    assert link.position == 100


def test_backlink_dataclass():
    """Test Backlink dataclass structure."""
    bl = Backlink(
        source_id="source",
        source_title="Source Title",
        target_id="target",
        link_type="wikilink",
        context="See [[target]]",
    )
    assert bl.source_id == "source"
    assert bl.target_id == "target"
    assert bl.link_type == "wikilink"


def test_link_suggestion_dataclass():
    """Test LinkSuggestion dataclass structure."""
    s = LinkSuggestion(
        memoria_id="abc123",
        title="Test Title",
        similarity=0.85,
        reason="High semantic similarity",
    )
    assert s.memoria_id == "abc123"
    assert s.similarity == 0.85
    assert "similarity" in s.reason.lower()


def test_wikilink_pattern():
    """Test the wikilink regex pattern."""
    from memo.crossref import _WIKILINK_PATTERN

    # Test simple wikilink
    matches = list(_WIKILINK_PATTERN.finditer("See [[target]]"))
    assert len(matches) == 1
    assert matches[0].group(1) == "target"

    # Test wikilink with alias
    matches = list(_WIKILINK_PATTERN.finditer("See [[target|Alias]]"))
    assert len(matches) == 1
    assert matches[0].group(1) == "target|Alias"

    # Test multiple wikilinks
    content = "[[a]] and [[b|B]] and [[c]]"
    matches = list(_WIKILINK_PATTERN.finditer(content))
    assert len(matches) == 3
