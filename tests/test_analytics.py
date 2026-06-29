"""Tests for analytics module."""

import pytest

from memo.analytics import (
    AccessPattern,
    AnalyticsEngine,
    CorpusMetrics,
    Dashboard,
    GrowthData,
)


@pytest.fixture
def analytics_engine(mock_memory):
    """Fixture providing AnalyticsEngine instance."""
    return AnalyticsEngine(mock_memory)


@pytest.fixture
def dashboard(analytics_engine):
    """Fixture providing Dashboard instance."""
    return Dashboard(analytics_engine)


def test_analytics_engine_init(analytics_engine):
    """Test AnalyticsEngine initialization."""
    assert analytics_engine.memory is not None


def test_analytics_engine_compute_corpus_metrics(analytics_engine, mock_memory):
    """Test computing corpus metrics."""
    # Create test memorias
    mock_memory.save(
        content="Test content 1",
        title="Test 1",
        tags=["test", "tag1"],
    )
    mock_memory.save(
        content="Test content 2",
        title="Test 2",
        tags=["test", "tag2"],
    )

    metrics = analytics_engine.compute_corpus_metrics()

    assert metrics.total_memories >= 2
    assert isinstance(metrics.type_distribution, dict)
    assert isinstance(metrics.tag_frequency, dict)
    assert isinstance(metrics.entity_frequency, dict)
    assert metrics.growth_rate >= 0


def test_analytics_engine_compute_growth_data(analytics_engine, mock_memory):
    """Test computing growth data."""
    mock_memory.save(
        content="Test content",
        title="Test",
        tags=["test"],
    )

    growth = analytics_engine.compute_growth_data(days=30)

    assert isinstance(growth.dates, list)
    assert isinstance(growth.counts, list)
    assert len(growth.dates) == len(growth.counts)


def test_analytics_engine_compute_access_patterns(analytics_engine):
    """Test computing access patterns."""
    patterns = analytics_engine.compute_access_patterns()

    assert isinstance(patterns.day_of_week, dict)
    assert isinstance(patterns.hour_of_day, dict)
    assert len(patterns.day_of_week) == 7
    assert len(patterns.hour_of_day) == 24


def test_analytics_engine_export_metrics_json(tmp_path, analytics_engine):
    """Test exporting metrics to JSON."""
    output_path = tmp_path / "analytics.json"

    analytics_engine.export_metrics_json(output_path)

    assert output_path.is_file()

    import json
    data = json.loads(output_path.read_text(encoding="utf-8"))
    assert "metrics" in data
    assert "growth" in data
    assert "access" in data


def test_analytics_engine_export_metrics_csv(tmp_path, analytics_engine):
    """Test exporting metrics to CSV."""
    output_path = tmp_path / "analytics.csv"

    analytics_engine.export_metrics_csv(output_path)

    assert output_path.is_file()

    content = output_path.read_text(encoding="utf-8")
    assert "Metric,Value" in content
    assert "Total Memories" in content


def test_dashboard_init(dashboard):
    """Test Dashboard initialization."""
    assert dashboard.analytics is not None


def test_dashboard_generate_summary(dashboard):
    """Test generating summary."""
    summary = dashboard.generate_summary()

    assert "Memory Analytics Dashboard" in summary
    assert "Total Memories" in summary
    assert "Type Distribution" in summary


def test_dashboard_generate_html_dashboard(tmp_path, dashboard):
    """Test generating HTML dashboard."""
    output_path = tmp_path / "dashboard.html"

    dashboard.generate_html_dashboard(output_path)

    assert output_path.is_file()

    content = output_path.read_text(encoding="utf-8")
    assert "<html>" in content
    assert "Memory Analytics Dashboard" in content


def test_corpus_metrics_dataclass():
    """Test CorpusMetrics dataclass structure."""
    metrics = CorpusMetrics(
        total_memories=100,
        total_entities=50,
        type_distribution={"note": 80, "decision": 20},
        tag_frequency={"test": 10},
        entity_frequency={"entity1": 5},
        growth_rate=1.5,
        average_access_count=3.0,
    )
    assert metrics.total_memories == 100
    assert metrics.growth_rate == 1.5


def test_growth_data_dataclass():
    """Test GrowthData dataclass structure."""
    growth = GrowthData(
        dates=["2026-01-01", "2026-01-02"],
        counts=[5, 10],
    )
    assert len(growth.dates) == 2
    assert growth.counts[0] == 5


def test_access_pattern_dataclass():
    """Test AccessPattern dataclass structure."""
    pattern = AccessPattern(
        day_of_week={"Monday": 10, "Tuesday": 5},
        hour_of_day={0: 0, 1: 0, 12: 5},
    )
    assert pattern.day_of_week["Monday"] == 10
    assert pattern.hour_of_day[12] == 5
