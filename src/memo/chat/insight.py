"""Heuristic insight detector: chat answer -> proposed Memo memoria.

Faithful to the heuristic path of the archived synapse `insight.py:240-383`
(the LLM judge was off in production and is NOT ported here — this module is
pure heuristics, no LLM) and to the adaptive threshold in
`user_model.py:636-659` (lean: ups-only, no accept/reject tracking since chat
feedback here is a single 👍/👎, not an explicit insight decision).

memo has no `goal` memory type, so the goal fast-path proposes a `note`
tagged `goal` instead of a distinct type.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

from memo.chat.feedback import ChatFeedback

_GOAL_RE = re.compile(
    r"\b(decidimos|vamos a|quiero|planeo|el objetivo es|we will|i want to|the goal is)\b",
    re.IGNORECASE,
)
_GOAL_SCORE = 55
_GOAL_CONFIDENCE = 0.6

_NEGATIVE_RE = re.compile(
    r"\b(no encontré|no encuentro|no sé|sin resultados)\b",
    re.IGNORECASE,
)
_SELF_REF_RE = re.compile(
    r"\b(el chat|este sistema|esta respuesta)\b",
    re.IGNORECASE,
)
_CITATION_RE = re.compile(r"\[\d+\]")
_DECISION_VERB_RE = re.compile(
    r"\b(decidimos|decidí|optamos|elegimos|acordamos|resolvimos|definimos)\b",
    re.IGNORECASE,
)
_DATE_HINT_RE = re.compile(r"\b(hoy|ayer|mañana|20\d{2})\b", re.IGNORECASE)
_CAPS_PHRASE_RE = re.compile(r"\b[A-ZÁÉÍÓÚÑ][A-Za-zÁÉÍÓÚÑáéíóúñ0-9]{2,}\b")
_SENTENCE_END_RE = re.compile(r"[.!?]\s")

# First match wins — order is significant.
_DOMAIN_PATTERNS: dict[str, re.Pattern[str]] = {
    "decision": re.compile(
        r"\b(decision|decisión|decidimos|decid[ií]|elegimos|acordamos|optamos)\b",
        re.IGNORECASE,
    ),
    "personal": re.compile(
        r"\b(familia|amigo|coach|cliente|coaching|scrum|agile|personal)\b",
        re.IGNORECASE,
    ),
    "technical": re.compile(
        r"\b(python|code|bug|test|api|deploy|backend|memo)\b",
        re.IGNORECASE,
    ),
}
_DEFAULT_THRESHOLD = 90
_ADAPTED_THRESHOLD = 75
_MIN_UPS_FOR_ADAPTATION = 5


@dataclass(frozen=True)
class InsightCandidate:
    """Proposed Memo memoria derived from a chat answer. Not yet persisted."""

    title: str
    body: str
    tags: list[str]
    confidence: float
    score: int
    suggested_type: str
    chat_session_id: str
    chat_turn_id: str
    schema: str = "memo.chat.insight_candidate.v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _is_mostly_bullets(answer: str) -> bool:
    lines = [line.strip() for line in answer.splitlines() if line.strip()]
    if not lines:
        return True
    bullet_lines = sum(1 for line in lines if line.startswith(("-", "*", "•")))
    return bullet_lines >= max(2, int(len(lines) * 0.7))


def _first_paragraph(answer: str, *, limit: int = 600) -> str:
    para = answer.strip().split("\n\n", 1)[0].strip()
    if len(para) <= limit:
        return para
    truncated = para[:limit]
    matches = list(_SENTENCE_END_RE.finditer(truncated))
    if matches:
        end = matches[-1].end()
        return truncated[:end].strip()
    return truncated.rstrip() + "…"


def _derive_title(question: str, answer: str) -> str:
    q = question.strip().rstrip("?").strip()
    if q:
        return q[:80]
    return _first_paragraph(answer, limit=80)


def _entities(answer: str) -> list[str]:
    return sorted({m for m in _CAPS_PHRASE_RE.findall(answer) if len(m) >= 3})


def _heuristic_score(answer: str) -> int:
    score = 0
    if len(_CITATION_RE.findall(answer)) >= 2:
        score += 20
    if _DATE_HINT_RE.search(answer):
        score += 10
    if _DECISION_VERB_RE.search(answer):
        score += 10
    score += min(15, len(_entities(answer)) * 5)
    return min(50, score)


def detect(
    question: str,
    answer: str,
    sources: list[dict[str, Any]],
    *,
    threshold: int,
    chat_session_id: str = "",
    chat_turn_id: str = "",
) -> InsightCandidate | None:
    """Heuristic-only detection of a chat answer worth proposing as a memoria."""
    if _GOAL_RE.search(f"{question} {answer}"):
        return InsightCandidate(
            title=_derive_title(question, answer),
            body=_first_paragraph(answer),
            tags=["goal"],
            confidence=_GOAL_CONFIDENCE,
            score=_GOAL_SCORE,
            suggested_type="note",
            chat_session_id=chat_session_id,
            chat_turn_id=chat_turn_id,
        )

    if len(answer) < 200:
        return None
    if len(sources) < 2:
        return None
    if _NEGATIVE_RE.search(answer):
        return None
    if _SELF_REF_RE.search(answer):
        return None
    if _is_mostly_bullets(answer):
        return None

    final_score = _heuristic_score(answer) * 2
    if final_score < threshold:
        return None

    has_verb = bool(_DECISION_VERB_RE.search(answer))
    has_date = bool(_DATE_HINT_RE.search(answer))
    entities = _entities(answer)

    tags: list[str] = []
    if has_verb:
        tags.append("decision")
    if has_date:
        tags.append("temporal")
    tags.extend(e.lower() for e in entities[:3])

    if has_verb:
        suggested_type = "decision"
    elif has_date:
        suggested_type = "fact"
    else:
        suggested_type = "note"

    return InsightCandidate(
        title=_derive_title(question, answer),
        body=_first_paragraph(answer),
        tags=tags,
        confidence=round(final_score / 100.0, 3),
        score=final_score,
        suggested_type=suggested_type,
        chat_session_id=chat_session_id,
        chat_turn_id=chat_turn_id,
    )


def _detect_domain(text: str) -> str | None:
    for domain, pattern in _DOMAIN_PATTERNS.items():
        if pattern.search(text):
            return domain
    return None


def insight_threshold(query: str, feedback_events: list[ChatFeedback]) -> int:
    """Adaptive insight detection threshold.

    Default 90. Lowers to 75 when the query's domain has >= 5 👍 feedback
    events whose own query also matches that domain. Lean version of
    synapse's accept-rate-based adaptation — this codebase only collects
    👍/👎, not explicit insight accept/reject decisions.
    """
    domain = _detect_domain(query)
    if domain is None:
        return _DEFAULT_THRESHOLD
    pattern = _DOMAIN_PATTERNS[domain]
    ups = sum(1 for fb in feedback_events if fb.rating == "up" and pattern.search(fb.query))
    if ups >= _MIN_UPS_FOR_ADAPTATION:
        return _ADAPTED_THRESHOLD
    return _DEFAULT_THRESHOLD


def is_duplicate(memory: Any, candidate: InsightCandidate) -> bool:
    """True if `candidate.title` already matches an existing memoria's title.

    Uses `memory.search(title, limit=3)` and checks case-insensitive
    bidirectional substring overlap against the top hit's title.
    """
    title = candidate.title.strip()
    if not title:
        return False
    results = memory.search(title, limit=3)
    if not results:
        return False
    existing_title = str(getattr(results[0], "title", "") or "").strip()
    if not existing_title:
        return False
    a, b = title.lower(), existing_title.lower()
    return a in b or b in a
