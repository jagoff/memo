"""Tests for temporal reasoning module."""

import pytest

from memo.temporal import (
    Contradiction,
    EntityTimeline,
    TemporalAnalyzer,
    TimelineEvent,
)


@pytest.fixture
def mock_memory(tmp_cfg):
    """Fixture providing a mock Memory instance for temporal tests."""
    from memo.memory import Memory
    return Memory(tmp_cfg)


@pytest.fixture
def temporal_analyzer(mock_memory):
    """Fixture providing TemporalAnalyzer instance."""
    return TemporalAnalyzer(mock_memory)


def test_temporal_analyzer_init(temporal_analyzer):
    """Test TemporalAnalyzer initialization."""
    assert temporal_analyzer.memory is not None
    assert temporal_analyzer._chat is None  # Lazy


def test_detect_entity_contradictions_no_entities(temporal_analyzer):
    """Test contradiction detection when entity has no memorias."""
    contradictions = temporal_analyzer.detect_entity_contradictions(
        entity_name="nonexistent",
        entity_type="technology",
        confidence_threshold=0.7,
        max_pairs=20,
    )
    assert contradictions == []


def test_detect_entity_contradictions_single_memoria(temporal_analyzer, mock_memory):
    """Test contradiction detection when entity has only one memoria."""
    # Save a test memoria
    mock_memory.save(
        content="Test content about MLX",
        title="MLX test",
        tags=["mlx", "test"],
    )

    contradictions = temporal_analyzer.detect_entity_contradictions(
        entity_name="mlx",
        confidence_threshold=0.7,
        max_pairs=20,
    )
    assert contradictions == []


def test_build_entity_timeline_no_memorias(temporal_analyzer):
    """Test timeline building when entity has no memorias."""
    timeline = temporal_analyzer.build_entity_timeline(
        entity_name="nonexistent",
    )
    assert timeline is None


def test_build_entity_timeline_with_memorias(temporal_analyzer, mock_memory):
    """Test timeline building with actual memorias."""
    # Save test memorias
    rec1 = mock_memory.save(
        content="First memoria about MLX",
        title="MLX first",
        tags=["mlx"],
    )
    rec2 = mock_memory.save(
        content="Second memoria about MLX",
        title="MLX second",
        tags=["mlx"],
    )

    # Extract entities first (required for graph lookup)
    mock_memory.extract_entities(ids=[rec1.id, rec2.id])

    timeline = temporal_analyzer.build_entity_timeline(
        entity_name="mlx",
    )

    assert timeline is not None
    assert timeline.entity_name == "mlx"
    assert len(timeline.events) >= 2
    assert all(isinstance(e, TimelineEvent) for e in timeline.events)


def test_detect_stale_memorias_empty_corpus(temporal_analyzer, mock_memory):
    """Test stale detection with empty corpus."""
    stale = temporal_analyzer.detect_stale_memorias(
        days_threshold=180,
        min_access_count=0,
    )
    assert stale == []


def test_detect_stale_memorias_with_recent(temporal_analyzer, mock_memory):
    """Test stale detection with recent memorias (should be empty)."""
    mock_memory.save(
        content="Recent memoria",
        title="Recent",
        tags=["test"],
    )

    stale = temporal_analyzer.detect_stale_memorias(
        days_threshold=180,
        min_access_count=0,
    )
    assert stale == []


def test_detect_temporal_patterns_empty(temporal_analyzer):
    """Test temporal patterns with empty corpus."""
    patterns = temporal_analyzer.detect_temporal_patterns()
    assert "memorias_per_month" in patterns
    assert "type_distribution_over_time" in patterns
    assert "most_active_entities" in patterns
    assert patterns["memorias_per_month"] == {}
    assert patterns["most_active_entities"] == {}


def test_detect_temporal_patterns_with_data(temporal_analyzer, mock_memory):
    """Test temporal patterns with actual memorias."""
    mock_memory.save(
        content="Test memoria 1",
        title="Test 1",
        tags=["test"],
    )
    mock_memory.save(
        content="Test memoria 2",
        title="Test 2",
        tags=["test"],
    )

    patterns = temporal_analyzer.detect_temporal_patterns()
    assert "memorias_per_month" in patterns
    assert len(patterns["memorias_per_month"]) > 0


def test_contradiction_dataclass():
    """Test Contradiction dataclass structure."""
    c = Contradiction(
        memoria_id_a="aaa",
        memoria_id_b="bbb",
        title_a="Title A",
        title_b="Title B",
        date_a="2026-01-01",
        date_b="2026-02-01",
        relationship="contradiction",
        rationale="Test rationale",
        confidence=0.9,
    )
    assert c.memoria_id_a == "aaa"
    assert c.relationship == "contradiction"
    assert c.confidence == 0.9


def test_timeline_event_dataclass():
    """Test TimelineEvent dataclass structure."""
    e = TimelineEvent(
        memoria_id="abc",
        title="Test",
        date="2026-01-01",
        type="note",
        snippet="Test snippet",
    )
    assert e.memoria_id == "abc"
    assert e.type == "note"


def test_entity_timeline_dataclass():
    """Test EntityTimeline dataclass structure."""
    events = [
        TimelineEvent(
            memoria_id="abc",
            title="Test",
            date="2026-01-01",
            type="note",
            snippet="Test",
        )
    ]
    t = EntityTimeline(
        entity_name="test",
        entity_type="technology",
        events=events,
        first_seen="2026-01-01",
        last_seen="2026-01-01",
    )
    assert t.entity_name == "test"
    assert len(t.events) == 1
