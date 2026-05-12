"""Tests for autonomous agent module."""

import pytest

from memo.agent import (
    AgentThought,
    AutonomousAgent,
    InvestigationPlan,
    ReasoningStep,
    SynthesisResult,
)


@pytest.fixture
def autonomous_agent(mock_memory):
    """Fixture providing AutonomousAgent instance."""
    return AutonomousAgent(mock_memory, mock_memory._chat)


def test_autonomous_agent_init(autonomous_agent):
    """Test AutonomousAgent initialization."""
    assert autonomous_agent.memory is not None
    assert autonomous_agent.chat is not None
    assert autonomous_agent._thoughts == []
    assert autonomous_agent._synthesis_history == []


def test_autonomous_agent_think(autonomous_agent):
    """Test registering a thought (meta-cognition)."""
    thought = autonomous_agent.think("Test hypothesis", "hypothesis")

    assert thought.thought_type == "hypothesis"
    assert thought.content == "Test hypothesis"
    assert len(autonomous_agent._thoughts) == 1


def test_autonomous_agent_get_thoughts(autonomous_agent):
    """Test getting thoughts."""
    autonomous_agent.think("Hypothesis 1", "hypothesis")
    autonomous_agent.think("Reflection 1", "reflection")

    thoughts = autonomous_agent.get_thoughts()
    assert len(thoughts) == 2

    hypothesis_thoughts = autonomous_agent.get_thoughts("hypothesis")
    assert len(hypothesis_thoughts) == 1


def test_autonomous_agent_plan_investigation(autonomous_agent):
    """Test planning an investigation."""
    plan = autonomous_agent.plan_investigation("Explore MLX edge computing")

    assert plan.goal == "Explore MLX edge computing"
    assert len(plan.steps) >= 1
    assert plan.estimated_complexity >= 1
    assert plan.estimated_insight_value >= 1


def test_autonomous_agent_execute_step(autonomous_agent):
    """Test executing a step."""
    step = autonomous_agent.execute_step("search for MLX")

    assert step.action in ["search", "analyze", "unknown"]
    assert step.query


def test_autonomous_agent_reason_causally(autonomous_agent, mock_memory):
    """Test causal reasoning."""
    # Create test memorias
    m1 = mock_memory.save(content="MLX is fast", title="MLX", tags=["mlx"])
    m2 = mock_memory.save(content="Edge devices need speed", title="Edge", tags=["edge"])

    reasoning = autonomous_agent.reason_causally("MLX edge relationship", [m1.id, m2.id])

    assert reasoning  # Should return some explanation
    assert isinstance(reasoning, str)


def test_autonomous_agent_synthesize_knowledge(autonomous_agent, mock_memory):
    """Test knowledge synthesis (THE GAMECHANGER)."""
    # Create test memorias
    mock_memory.save(content="MLX is optimized for Apple Silicon", title="MLX", tags=["mlx"])
    mock_memory.save(content="Apple Silicon is efficient", title="Apple", tags=["apple"])

    synthesis = autonomous_agent.synthesize_knowledge("MLX Apple Silicon")

    assert synthesis.new_insight
    assert synthesis.confidence >= 0
    assert synthesis.novelty_score >= 0
    assert len(synthesis.reasoning_chain) >= 1
    assert len(autonomous_agent._synthesis_history) == 1


def test_autonomous_agent_proactive_discovery(autonomous_agent, mock_memory):
    """Test proactive discovery."""
    # Create test memorias with tags
    mock_memory.save(content="Test 1", title="T1", tags=["test"])
    mock_memory.save(content="Test 2", title="T2", tags=["test"])

    discoveries = autonomous_agent.proactive_discovery()

    # May return empty list if corpus is small or synthesis fails
    assert isinstance(discoveries, list)


def test_autonomous_agent_get_synthesis_history(autonomous_agent):
    """Test getting synthesis history."""
    synthesis = SynthesisResult(
        new_insight="Test insight",
        supporting_memorias=["id1"],
        reasoning_chain=[],
        confidence=0.7,
        novelty_score=0.8,
    )
    autonomous_agent._synthesis_history.append(synthesis)

    history = autonomous_agent.get_synthesis_history()
    assert len(history) == 1
    assert history[0].new_insight == "Test insight"


def test_reasoning_step_dataclass():
    """Test ReasoningStep dataclass structure."""
    step = ReasoningStep(
        step_number=1,
        action="search",
        query="test",
        results=["id1"],
        reasoning="Test reasoning",
        confidence=0.8,
    )
    assert step.step_number == 1
    assert step.action == "search"


def test_investigation_plan_dataclass():
    """Test InvestigationPlan dataclass structure."""
    plan = InvestigationPlan(
        goal="Test goal",
        steps=["step 1", "step 2"],
        estimated_complexity=5,
        estimated_insight_value=8,
    )
    assert plan.goal == "Test goal"
    assert len(plan.steps) == 2


def test_synthesis_result_dataclass():
    """Test SynthesisResult dataclass structure."""
    synthesis = SynthesisResult(
        new_insight="New insight",
        supporting_memorias=["id1", "id2"],
        reasoning_chain=[],
        confidence=0.9,
        novelty_score=0.7,
    )
    assert synthesis.new_insight == "New insight"
    assert synthesis.novelty_score == 0.7


def test_agent_thought_dataclass():
    """Test AgentThought dataclass structure."""
    thought = AgentThought(
        timestamp="2026-01-01T00:00:00Z",
        thought_type="hypothesis",
        content="Test content",
        related_memorias=["id1"],
    )
    assert thought.thought_type == "hypothesis"
    assert thought.content == "Test content"
