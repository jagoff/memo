from memo.chat.feedback import ChatFeedback
from memo.chat.insight import InsightCandidate, detect, insight_threshold, is_duplicate

_SOURCES = [{"id": "s1"}, {"id": "s2"}]


def _fb(query: str, rating: str) -> ChatFeedback:
    return ChatFeedback(
        feedback_id="f",
        created_at="2026-08-01T00:00:00",
        chat_session_id="sess",
        turn_id="turn",
        query=query,
        answer="answer",
        source_ids=["s1"],
        rating=rating,
    )


def test_goal_fast_path_short_circuits_gates() -> None:
    # Short answer, no citations, no sources — the goal fast-path bypasses
    # every heuristic gate below it.
    candidate = detect(
        "que vamos a hacer",
        "Quiero implementar un mejor sistema de recall.",
        [],
        threshold=90,
    )
    assert candidate is not None
    assert candidate.suggested_type == "note"
    assert candidate.tags == ["goal"]
    assert candidate.score == 55
    assert candidate.confidence == 0.6


def test_gate_answer_too_short() -> None:
    assert detect("q", "short answer", _SOURCES, threshold=50) is None


def test_gate_too_few_sources() -> None:
    long_answer = "x" * 250
    assert detect("q", long_answer, [{"id": "s1"}], threshold=50) is None


def test_gate_negative_phrase() -> None:
    answer = "No encontré información suficiente sobre eso. " + "x" * 200
    assert detect("q", answer, _SOURCES, threshold=50) is None


def test_gate_self_reference() -> None:
    answer = "Este sistema no puede responder eso realmente bien. " + "x" * 200
    assert detect("q", answer, _SOURCES, threshold=50) is None


def test_gate_mostly_bullets() -> None:
    lines = "\n".join(f"- item {i} con texto suficiente para no ser corto" for i in range(10))
    assert detect("q", lines, _SOURCES, threshold=50) is None


# 2 citations (+20) + decision verb (+10) + 2 unique capitalized entities
# (min(15, 2*5)=+10) => heuristic 40, final = 40*2 = 80. Deliberately a single
# lowercase-led sentence (no sentence-start capitals) so the naive
# capitalized-word regex only picks up the two proper nouns.
_SCORED_ANSWER = (
    "optamos por usar Kubernetes junto con Terraform para el despliegue [1][2] "
    "porque esa decision mejora bastante la infraestructura del equipo entero "
    "y simplifica el pipeline de entrega continua para todos los proyectos "
    "internos de la organizacion sin generar demasiada complejidad adicional"
)


def test_score_composition_exact_arithmetic() -> None:
    assert len(_SCORED_ANSWER) >= 200

    candidate = detect("cual fue la decision", _SCORED_ANSWER, _SOURCES, threshold=75)
    assert candidate is not None
    assert candidate.score == 80

    assert detect("cual fue la decision", _SCORED_ANSWER, _SOURCES, threshold=90) is None


def test_score_below_threshold_returns_none() -> None:
    answer = "x" * 250  # no citations, no verb, no entities, no date => score 0
    assert detect("q", answer, _SOURCES, threshold=1) is None


def test_candidate_title_and_body_derivation() -> None:
    candidate = detect("Cual fue la decision tomada?", _SCORED_ANSWER, _SOURCES, threshold=75)
    assert candidate is not None
    assert candidate.title == "Cual fue la decision tomada"
    assert "decision" in candidate.tags
    assert candidate.suggested_type == "decision"
    assert candidate.confidence == 0.8


def test_to_dict_roundtrips_fields() -> None:
    candidate = InsightCandidate(
        title="t",
        body="b",
        tags=["decision"],
        confidence=0.8,
        score=80,
        suggested_type="decision",
        chat_session_id="sess",
        chat_turn_id="turn",
    )
    d = candidate.to_dict()
    assert d["title"] == "t"
    assert d["schema"] == "memo.chat.insight_candidate.v1"


def test_insight_threshold_default_no_domain_match() -> None:
    assert insight_threshold("random unrelated query", []) == 90


def test_insight_threshold_domain_match_but_few_ups() -> None:
    events = [_fb("python bug in memo", "up") for _ in range(4)]
    assert insight_threshold("python api question", events) == 90


def test_insight_threshold_adapts_with_five_technical_ups() -> None:
    events = [_fb("python bug in memo backend", "up") for _ in range(5)]
    assert insight_threshold("python api question", events) == 75


def test_insight_threshold_ignores_down_votes() -> None:
    events = [_fb("python bug in memo backend", "down") for _ in range(5)]
    assert insight_threshold("python api question", events) == 90


class _Rec:
    def __init__(self, title: str) -> None:
        self.title = title


class _FakeMemory:
    def __init__(self, results: list[_Rec]) -> None:
        self._results = results

    def search(self, query: str, limit: int = 3) -> list[_Rec]:
        return self._results


def test_is_duplicate_true_on_substring_match() -> None:
    candidate = InsightCandidate(
        title="decision sobre despliegue",
        body="b",
        tags=[],
        confidence=0.8,
        score=80,
        suggested_type="decision",
        chat_session_id="",
        chat_turn_id="",
    )
    memory = _FakeMemory([_Rec("La decision sobre despliegue y su contexto")])
    assert is_duplicate(memory, candidate) is True


def test_is_duplicate_false_on_no_results() -> None:
    candidate = InsightCandidate(
        title="algo nuevo",
        body="b",
        tags=[],
        confidence=0.8,
        score=80,
        suggested_type="note",
        chat_session_id="",
        chat_turn_id="",
    )
    memory = _FakeMemory([])
    assert is_duplicate(memory, candidate) is False


def test_is_duplicate_false_on_unrelated_title() -> None:
    candidate = InsightCandidate(
        title="algo completamente distinto",
        body="b",
        tags=[],
        confidence=0.8,
        score=80,
        suggested_type="note",
        chat_session_id="",
        chat_turn_id="",
    )
    memory = _FakeMemory([_Rec("otro tema sin relacion")])
    assert is_duplicate(memory, candidate) is False
