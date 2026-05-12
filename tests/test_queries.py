"""Tests for query composition module."""

import pytest

from memo.queries import (
    Query,
    QueryComposer,
    QueryResult,
    QueryStore,
)


@pytest.fixture
def query_store(tmp_cfg):
    """Fixture providing QueryStore instance."""
    return QueryStore(tmp_cfg.state_dir)


@pytest.fixture
def query_composer(mock_memory):
    """Fixture providing QueryComposer instance."""
    return QueryComposer(mock_memory, QueryStore(mock_memory.cfg.state_dir))


def test_query_store_init(query_store):
    """Test QueryStore initialization."""
    assert query_store.state_dir.is_dir()


def test_query_store_save_query(query_store):
    """Test saving a query."""
    query_store.save_query(
        name="test-query",
        query_text="MLX",
        type_filter="decision",
        tags_filter=["mlx"],
        search_mode="hybrid",
        limit=10,
        description="Test description",
    )

    query = query_store.get_query("test-query")
    assert query is not None
    assert query.name == "test-query"
    assert query.query_text == "MLX"
    assert query.type_filter == "decision"


def test_query_store_get_query(query_store):
    """Test getting a query."""
    query_store.save_query(
        name="test-query",
        query_text="Qwen",
        search_mode="hybrid",
    )

    query = query_store.get_query("test-query")
    assert query is not None
    assert query.name == "test-query"


def test_query_store_get_query_not_found(query_store):
    """Test getting a non-existent query."""
    query = query_store.get_query("nonexistent")
    assert query is None


def test_query_store_list_queries(query_store):
    """Test listing all queries."""
    query_store.save_query(name="q1", query_text="test1")
    query_store.save_query(name="q2", query_text="test2")

    queries = query_store.list_queries()
    assert len(queries) == 2
    assert all(isinstance(q, Query) for q in queries)


def test_query_store_delete_query(query_store):
    """Test deleting a query."""
    query_store.save_query(name="test", query_text="test")

    assert query_store.get_query("test") is not None

    success = query_store.delete_query("test")
    assert success is True

    assert query_store.get_query("test") is None


def test_query_store_persistence(tmp_cfg):
    """Test that queries persist across instances."""
    state_dir = tmp_cfg.state_dir

    # Create first instance and save query
    store1 = QueryStore(state_dir)
    store1.save_query(name="test", query_text="MLX")

    # Create second instance and verify persistence
    store2 = QueryStore(state_dir)
    query = store2.get_query("test")

    assert query is not None
    assert query.query_text == "MLX"


def test_query_composer_init(query_composer):
    """Test QueryComposer initialization."""
    assert query_composer.memory is not None
    assert query_composer.query_store is not None


def test_query_composer_execute_query(query_composer, mock_memory):
    """Test executing a query."""
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

    query = Query(
        name="test",
        query_text="MLX",
        type_filter=None,
        tags_filter=None,
        date_from=None,
        date_to=None,
        search_mode="hybrid",
        limit=10,
        description=None,
        created_at="2026-01-01T00:00:00Z",
    )

    result = query_composer.execute_query(query)

    assert result.query_name == "test"
    assert isinstance(result.results, list)
    assert result.count >= 0


def test_query_composer_execute_query_with_type_filter(query_composer, mock_memory):
    """Test executing a query with type filter."""
    mock_memory.save(
        content="Decision 1",
        title="Decision 1",
        tags=["test"],
        type="decision",
    )
    mock_memory.save(
        content="Note 1",
        title="Note 1",
        tags=["test"],
        type="note",
    )

    query = Query(
        name="test",
        query_text="test",
        type_filter="decision",
        tags_filter=None,
        date_from=None,
        date_to=None,
        search_mode="hybrid",
        limit=10,
        description=None,
        created_at="2026-01-01T00:00:00Z",
    )

    result = query_composer.execute_query(query)

    # Should only return decision type
    for r in result.results:
        assert r.type == "decision"


def test_query_composer_execute_query_with_tag_filter(query_composer, mock_memory):
    """Test executing a query with tag filter."""
    mock_memory.save(
        content="MLX content",
        title="MLX",
        tags=["mlx"],
    )
    mock_memory.save(
        content="Qwen content",
        title="Qwen",
        tags=["qwen"],
    )

    query = Query(
        name="test",
        query_text="content",
        type_filter=None,
        tags_filter=["mlx"],
        date_from=None,
        date_to=None,
        search_mode="hybrid",
        limit=10,
        description=None,
        created_at="2026-01-01T00:00:00Z",
    )

    result = query_composer.execute_query(query)

    # Should only return mlx-tagged memorias
    for r in result.results:
        assert "mlx" in r.tags


def test_query_composer_compose_and_save(query_composer, mock_memory):
    """Test compose_and_save method."""
    mock_memory.save(
        content="Test content",
        title="Test",
        tags=["test"],
    )

    result = query_composer.compose_and_save(
        name="test-query",
        query_text="test",
        type_filter=None,
        tags_filter=None,
        search_mode="hybrid",
        limit=10,
        description=None,
    )

    assert result.query_name == "test-query"
    assert isinstance(result.results, list)

    # Verify query was saved
    query = query_composer.query_store.get_query("test-query")
    assert query is not None


def test_query_dataclass():
    """Test Query dataclass structure."""
    q = Query(
        name="test",
        query_text="MLX",
        type_filter="decision",
        tags_filter=["mlx"],
        date_from="2026-01-01",
        date_to="2026-12-31",
        search_mode="hybrid",
        limit=10,
        description="Test",
        created_at="2026-01-01T00:00:00Z",
    )
    assert q.name == "test"
    assert q.query_text == "MLX"
    assert q.type_filter == "decision"
    assert len(q.tags_filter) == 1


def test_query_result_dataclass():
    """Test QueryResult dataclass structure."""
    result = QueryResult(
        query_name="test",
        results=[],
        count=0,
        executed_at="2026-01-01T00:00:00Z",
    )
    assert result.query_name == "test"
    assert result.count == 0
    assert result.results == []
