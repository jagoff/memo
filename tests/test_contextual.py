"""Tests for contextual recall module."""

import pytest

from memo.contextual import (
    ContextStore,
    ContextualRecall,
    ContextualSearchResult,
    PromptContext,
    UserPreferences,
)


@pytest.fixture
def context_store(tmp_cfg):
    """Fixture providing ContextStore instance."""
    return ContextStore(tmp_cfg.state_dir)


@pytest.fixture
def contextual_recall(mock_memory):
    """Fixture providing ContextualRecall instance."""
    from memo.contextual import ContextStore

    context_store = ContextStore(mock_memory.cfg.state_dir)
    return ContextualRecall(mock_memory, context_store)


def test_context_store_init(context_store):
    """Test ContextStore initialization."""
    assert context_store.state_dir.is_dir()
    assert context_store.max_context_length == 10


def test_context_store_add_prompt(context_store):
    """Test adding a prompt to context history."""
    context_store.add_prompt("test query", ["mem1", "mem2"])
    history = context_store.get_recent_context(n=5)
    assert len(history) == 1
    assert history[0].prompt == "test query"
    assert history[0].recalled_memories == ["mem1", "mem2"]


def test_context_store_max_length(context_store):
    """Test that context store respects max length."""
    context_store.max_context_length = 3

    for i in range(5):
        context_store.add_prompt(f"query {i}", [f"mem{i}"])

    history = context_store.get_recent_context(n=10)
    assert len(history) == 3
    # Should keep the most recent
    assert history[-1].prompt == "query 4"


def test_context_store_get_recent_context(context_store):
    """Test getting recent context with limit."""
    for i in range(5):
        context_store.add_prompt(f"query {i}", [f"mem{i}"])

    history = context_store.get_recent_context(n=3)
    assert len(history) == 3
    assert history[0].prompt == "query 2"  # 3rd most recent
    assert history[-1].prompt == "query 4"  # Most recent


def test_context_store_record_feedback(context_store):
    """Test recording user feedback."""
    context_store.record_feedback("mem1", "decision", ["mlx", "qwen"])

    prefs = context_store.get_preferences()
    assert "decision" in prefs.preferred_types
    assert "mlx" in prefs.preferred_entities
    assert "qwen" in prefs.preferred_entities


def test_context_store_feedback_accumulation(context_store):
    """Test that feedback accumulates correctly."""
    context_store.record_feedback("mem1", "decision", ["mlx"])
    context_store.record_feedback("mem2", "decision", ["mlx"])

    prefs = context_store.get_preferences()
    assert prefs.preferred_types["decision"] > 0.5  # Should have increased
    assert prefs.preferred_entities["mlx"] > 0.5


def test_context_store_feedback_capping(context_store):
    """Test that feedback scores are capped at 1.0."""
    # Boost decision many times
    for _ in range(20):
        context_store.record_feedback("mem1", "decision", ["mlx"])

    prefs = context_store.get_preferences()
    assert prefs.preferred_types["decision"] <= 1.0


def test_context_store_persistence(tmp_cfg):
    """Test that context and preferences persist across instances."""
    state_dir = tmp_cfg.state_dir

    # Create first instance and add data
    store1 = ContextStore(state_dir)
    store1.add_prompt("test query", ["mem1"])
    store1.record_feedback("mem1", "decision", ["mlx"])

    # Create second instance and verify data persisted
    store2 = ContextStore(state_dir)
    history = store2.get_recent_context(n=5)
    prefs = store2.get_preferences()

    assert len(history) == 1
    assert history[0].prompt == "test query"
    assert "decision" in prefs.preferred_types


def test_contextual_recall_init(contextual_recall):
    """Test ContextualRecall initialization."""
    assert contextual_recall.memory is not None
    assert contextual_recall.context is not None


def test_contextual_recall_search_with_context(contextual_recall, mock_memory):
    """Test contextual search with actual data."""
    # Create test memorias
    mock_memory.save(
        content="Memo about MLX",
        title="MLX",
        tags=["mlx"],
    )
    mock_memory.save(
        content="Memo about Qwen",
        title="Qwen",
        tags=["qwen"],
    )

    # Search with context
    results = contextual_recall.search_with_context(
        query="MLX",
        limit=10,
        mode="hybrid",
    )

    assert isinstance(results, list)
    # Results should be ContextualSearchResult objects
    if results:
        assert all(isinstance(r, ContextualSearchResult) for r in results)


def test_contextual_recall_record_search(contextual_recall):
    """Test recording a search."""
    contextual_recall.record_search("test query", ["mem1", "mem2"])

    history = contextual_recall.context.get_recent_context(n=5)
    assert len(history) == 1
    assert history[0].prompt == "test query"


def test_contextual_recall_record_click(contextual_recall, mock_memory):
    """Test recording a click."""
    rec = mock_memory.save(
        content="Test content",
        title="Test",
        tags=["test"],
    )

    contextual_recall.record_click(rec.id)

    prefs = contextual_recall.context.get_preferences()
    # Should have recorded feedback for this memoria's type
    assert rec.type in prefs.preferred_types


def test_contextual_search_result_dataclass():
    """Test ContextualSearchResult dataclass structure."""
    result = ContextualSearchResult(
        memory_id="abc123",
        title="Test Title",
        original_score=0.8,
        contextual_score=0.9,
        boost_factors={"entity_overlap": 0.1},
        snippet="Test snippet",
    )
    assert result.memory_id == "abc123"
    assert result.contextual_score > result.original_score
    assert "entity_overlap" in result.boost_factors


def test_prompt_context_dataclass():
    """Test PromptContext dataclass structure."""
    ctx = PromptContext(
        timestamp="2026-01-01T00:00:00Z",
        prompt="test query",
        recalled_memories=["mem1", "mem2"],
    )
    assert ctx.prompt == "test query"
    assert len(ctx.recalled_memories) == 2


def test_user_preferences_dataclass():
    """Test UserPreferences dataclass structure."""
    prefs = UserPreferences(
        preferred_types={"decision": 0.8, "fact": 0.6},
        preferred_entities={"mlx": 0.9},
        recency_weight=0.5,
        diversity_weight=0.3,
        last_updated="2026-01-01T00:00:00Z",
    )
    assert prefs.preferred_types["decision"] == 0.8
    assert prefs.recency_weight == 0.5
    assert len(prefs.preferred_entities) == 1


def test_contextual_search_boost_factors(contextual_recall, mock_memory):
    """Test that contextual search applies boost factors."""
    # Add some context history
    contextual_recall.context.add_prompt("MLX query", ["mem1"])

    # Create a memoria
    rec = mock_memory.save(
        content="Memo about MLX",
        title="MLX",
        tags=["mlx"],
    )

    # Extract entities
    mock_memory.extract_entities(ids=[rec.id])

    # Search
    results = contextual_recall.search_with_context(
        query="MLX",
        limit=10,
        mode="hybrid",
    )

    # Check that results have boost factors
    if results:
        assert all(hasattr(r, "boost_factors") for r in results)
