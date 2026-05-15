"""Tests for cognitive module."""

import pytest

from memo.cognitive import (
    CognitiveManager,
    CognitiveState,
    CognitiveStateTracker,
    ContextAwareRetrieval,
    ContextType,
    ContextualSuggestion,
    MentalState,
    ProactiveGuidance,
)


@pytest.fixture
def cognitive_tracker(tmp_cfg):
    """Fixture providing CognitiveStateTracker instance."""
    return CognitiveStateTracker(tmp_cfg.state_dir)


@pytest.fixture
def context_aware_retrieval(cognitive_tracker):
    """Fixture providing ContextAwareRetrieval instance."""
    return ContextAwareRetrieval(cognitive_tracker)


@pytest.fixture
def proactive_guidance(cognitive_tracker):
    """Fixture providing ProactiveGuidance instance."""
    return ProactiveGuidance(cognitive_tracker)


@pytest.fixture
def cognitive_manager(cognitive_tracker, context_aware_retrieval, proactive_guidance):
    """Fixture providing CognitiveManager instance."""
    return CognitiveManager(cognitive_tracker, context_aware_retrieval, proactive_guidance)


def test_cognitive_tracker_init(cognitive_tracker):
    """Test CognitiveStateTracker initialization."""
    assert cognitive_tracker.state_dir.is_dir()


def test_cognitive_tracker_update_state(cognitive_tracker):
    """Test updating cognitive state."""
    state = cognitive_tracker.update_state(
        mental_state=MentalState.FOCUSED.value,
        context_type=ContextType.WORK.value,
        current_goal="Finish MLX integration",
        focus_area="MLX",
        energy_level=80,
        stress_level=20,
    )

    assert state.mental_state == MentalState.FOCUSED.value
    assert state.context_type == ContextType.WORK.value
    assert state.current_goal == "Finish MLX integration"
    assert state.focus_area == "MLX"


def test_cognitive_tracker_get_current_state(cognitive_tracker):
    """Test getting current state."""
    cognitive_tracker.update_state(
        mental_state=MentalState.FOCUSED.value,
        context_type=ContextType.WORK.value,
    )

    state = cognitive_tracker.get_current_state()

    assert state is not None
    assert state.mental_state == MentalState.FOCUSED.value


def test_cognitive_tracker_get_history(cognitive_tracker):
    """Test getting state history."""
    cognitive_tracker.update_state(
        mental_state=MentalState.FOCUSED.value,
        context_type=ContextType.WORK.value,
    )
    cognitive_tracker.update_state(
        mental_state=MentalState.RELAXED.value,
        context_type=ContextType.PERSONAL.value,
    )

    history = cognitive_tracker.get_history(limit=10)

    assert len(history) >= 2


def test_cognitive_tracker_persistence(tmp_cfg):
    """Test state persistence across instances."""
    state_dir = tmp_cfg.state_dir

    # Create first instance and update state
    tracker1 = CognitiveStateTracker(state_dir)
    tracker1.update_state(
        mental_state=MentalState.FOCUSED.value,
        context_type=ContextType.WORK.value,
        current_goal="Test goal",
    )

    # Create second instance and verify persistence
    tracker2 = CognitiveStateTracker(state_dir)
    state = tracker2.get_current_state()

    assert state is not None
    assert state.current_goal == "Test goal"


def test_context_aware_retrieval_init(context_aware_retrieval):
    """Test ContextAwareRetrieval initialization."""
    assert context_aware_retrieval.tracker is not None


def test_context_aware_retrieval_retrieve_with_context(context_aware_retrieval):
    """Test retrieving with context awareness."""
    # Mock search function
    def search_func(query: str, limit: int) -> list:
        return [f"result_{i}" for i in range(limit)]

    # Without state, should use normal search
    results = context_aware_retrieval.retrieve_with_context("test", search_func, limit=5)

    assert len(results) == 5


def test_context_aware_retrieval_adapt_query_to_state(cognitive_tracker, context_aware_retrieval):
    """Test query adaptation to mental state."""
    cognitive_tracker.update_state(
        mental_state=MentalState.PROBLEM_SOLVING.value,
        context_type=ContextType.WORK.value,
    )

    adapted = context_aware_retrieval._adapt_query_to_state("MLX", cognitive_tracker.get_current_state())

    assert "solution" in adapted.lower() or "fix" in adapted.lower()


def test_proactive_guidance_init(proactive_guidance):
    """Test ProactiveGuidance initialization."""
    assert proactive_guidance.tracker is not None


def test_proactive_guidance_generate_suggestions(cognitive_tracker, proactive_guidance):
    """Test generating proactive suggestions."""
    cognitive_tracker.update_state(
        mental_state=MentalState.FOCUSED.value,
        context_type=ContextType.WORK.value,
        focus_area="MLX",
    )

    # Mock search function
    def search_func(query: str, limit: int) -> list:
        return [f"memoria_{i}" for i in range(limit)]

    suggestions = proactive_guidance.generate_suggestions(search_func, limit=3)

    assert isinstance(suggestions, list)


def test_proactive_guidance_get_recent_suggestions(proactive_guidance):
    """Test getting recent suggestions."""
    suggestions = proactive_guidance.get_recent_suggestions(limit=10)

    assert isinstance(suggestions, list)


def test_cognitive_manager_init(cognitive_manager):
    """Test CognitiveManager initialization."""
    assert cognitive_manager.tracker is not None
    assert cognitive_manager.retrieval is not None
    assert cognitive_manager.guidance is not None


def test_cognitive_manager_update_mental_state(cognitive_manager):
    """Test updating mental state."""
    state = cognitive_manager.update_mental_state(
        mental_state=MentalState.FOCUSED.value,
        context_type=ContextType.WORK.value,
        current_goal="Test goal",
    )

    assert state.mental_state == MentalState.FOCUSED.value


def test_cognitive_manager_get_mental_state(cognitive_manager):
    """Test getting mental state."""
    cognitive_manager.update_mental_state(
        mental_state=MentalState.FOCUSED.value,
        context_type=ContextType.WORK.value,
    )

    state = cognitive_manager.get_mental_state()

    assert state is not None
    assert state.mental_state == MentalState.FOCUSED.value


def test_cognitive_manager_retrieve_aware(cognitive_manager):
    """Test retrieving with awareness."""
    def search_func(query: str, limit: int) -> list:
        return [f"result_{i}" for i in range(limit)]

    results = cognitive_manager.retrieve_aware("test", search_func, limit=5)

    assert len(results) == 5


def test_cognitive_manager_get_proactive_suggestions(cognitive_manager):
    """Test getting proactive suggestions."""
    cognitive_manager.update_mental_state(
        mental_state=MentalState.FOCUSED.value,
        context_type=ContextType.WORK.value,
        focus_area="MLX",
    )

    def search_func(query: str, limit: int) -> list:
        return [f"memoria_{i}" for i in range(limit)]

    suggestions = cognitive_manager.get_proactive_suggestions(search_func, limit=3)

    assert isinstance(suggestions, list)


def test_mental_state_enum():
    """Test MentalState enum values."""
    assert MentalState.FOCUSED.value == "focused"
    assert MentalState.RELAXED.value == "relaxed"
    assert MentalState.STRESSED.value == "stressed"
    assert MentalState.EXPLORING.value == "exploring"
    assert MentalState.PROBLEM_SOLVING.value == "problem_solving"
    assert MentalState.LEARNING.value == "learning"


def test_context_type_enum():
    """Test ContextType enum values."""
    assert ContextType.WORK.value == "work"
    assert ContextType.PERSONAL.value == "personal"
    assert ContextType.RESEARCH.value == "research"
    assert ContextType.CREATIVE.value == "creative"
    assert ContextType.ROUTINE.value == "routine"


def test_cognitive_state_dataclass():
    """Test CognitiveState dataclass structure."""
    state = CognitiveState(
        timestamp="2026-01-01T00:00:00Z",
        mental_state="focused",
        context_type="work",
        current_goal="Test goal",
        focus_area="MLX",
        energy_level=80,
        stress_level=20,
    )
    assert state.mental_state == "focused"
    assert state.energy_level == 80


def test_contextual_suggestion_dataclass():
    """Test ContextualSuggestion dataclass structure."""
    suggestion = ContextualSuggestion(
        suggestion_id="sugg-1",
        memoria_id="mem-123",
        relevance_reason="Related to your focus",
        confidence=0.85,
        suggested_at="2026-01-01T00:00:00Z",
    )
    assert suggestion.memoria_id == "mem-123"
    assert suggestion.confidence == 0.85
