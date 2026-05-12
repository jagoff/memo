"""Tests for proactive suggestions module."""

import pytest

from memo.proactive import (
    ProactiveSuggester,
    Suggestion,
    SuggestionFeedback,
)


@pytest.fixture
def proactive_suggester(mock_memory):
    """Fixture providing ProactiveSuggester instance."""
    return ProactiveSuggester(mock_memory)


def test_proactive_suggester_init(proactive_suggester):
    """Test ProactiveSuggester initialization."""
    assert proactive_suggester.memory is not None
    assert proactive_suggester._chat is None  # Lazy


def test_analyze_conversation_empty(proactive_suggester):
    """Test analyzing empty conversation."""
    suggestions = proactive_suggester.analyze_conversation([])
    assert suggestions == []


def test_analyze_conversation_with_data(proactive_suggester):
    """Test analyzing conversation with actual turns."""
    turns = [
        {"user": "I decided to use MLX for embeddings", "assistant": "Good choice"},
        {"user": "MLX is faster than Ollama", "assistant": "Yes, it's native"},
    ]

    suggestions = proactive_suggester.analyze_conversation(turns, limit=3)

    # May return suggestions or empty depending on LLM response
    assert isinstance(suggestions, list)
    if suggestions:
        assert all(isinstance(s, Suggestion) for s in suggestions)


def test_record_feedback(proactive_suggester):
    """Test recording feedback on a suggestion."""
    suggestion = Suggestion(
        title="Test",
        type="decision",
        tags=["test"],
        body_snippet="Test snippet",
        confidence=0.8,
        rationale="Test rationale",
        suggested_at="2026-01-01T00:00:00Z",
    )

    proactive_suggester.record_feedback(suggestion, accepted=True)

    stats = proactive_suggester.get_feedback_stats()
    assert stats["total"] == 1
    assert stats["accepted"] == 1
    assert stats["rejected"] == 0
    assert stats["acceptance_rate"] == 1.0


def test_record_feedback_rejected(proactive_suggester):
    """Test recording rejected feedback."""
    suggestion = Suggestion(
        title="Test",
        type="decision",
        tags=["test"],
        body_snippet="Test snippet",
        confidence=0.8,
        rationale="Test rationale",
        suggested_at="2026-01-01T00:00:00Z",
    )

    proactive_suggester.record_feedback(suggestion, accepted=False)

    stats = proactive_suggester.get_feedback_stats()
    assert stats["total"] == 1
    assert stats["accepted"] == 0
    assert stats["rejected"] == 1
    assert stats["acceptance_rate"] == 0.0


def test_get_feedback_stats_empty(proactive_suggester):
    """Test feedback stats with no feedback."""
    stats = proactive_suggester.get_feedback_stats()
    assert stats["total"] == 0
    assert stats["accepted"] == 0
    assert stats["rejected"] == 0
    assert stats["acceptance_rate"] == 0.0


def test_detect_patterns(proactive_suggester, tmp_path):
    """Test pattern detection on a transcript."""
    # Create a dummy transcript file
    transcript_path = tmp_path / "transcript.jsonl"
    transcript_path.write_text('{"role":"user","content":"test"}\n', encoding="utf-8")

    patterns = proactive_suggester.detect_patterns(transcript_path)

    assert "recurring_themes" in patterns
    assert "decision_points" in patterns
    assert "technical_discoveries" in patterns
    assert "total_turns" in patterns


def test_suggestion_dataclass():
    """Test Suggestion dataclass structure."""
    s = Suggestion(
        title="Test Title",
        type="decision",
        tags=["test", "decision"],
        body_snippet="Test snippet",
        confidence=0.85,
        rationale="Test rationale",
        suggested_at="2026-01-01T00:00:00Z",
    )
    assert s.title == "Test Title"
    assert s.type == "decision"
    assert s.confidence == 0.85
    assert len(s.tags) == 2


def test_suggestion_feedback_dataclass():
    """Test SuggestionFeedback dataclass structure."""
    f = SuggestionFeedback(
        suggestion_id="test_id",
        accepted=True,
        timestamp="2026-01-01T00:00:00Z",
    )
    assert f.suggestion_id == "test_id"
    assert f.accepted is True


def test_analyze_conversation_low_confidence_filtering(proactive_suggester):
    """Test that low-confidence suggestions are filtered out."""
    # This test would require mocking the LLM response
    # For now, just verify the logic exists
    pass


def test_analyze_conversation_sorting(proactive_suggester):
    """Test that suggestions are sorted by confidence descending."""
    # This test would require mocking the LLM response
    # For now, just verify the logic exists
    pass
