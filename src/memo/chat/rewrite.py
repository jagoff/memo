"""Rules-only follow-up rewrite (the LLM paraphrase path was not rescued)."""

from __future__ import annotations

import re

_SUMMARY_FOLLOWUP_RE = re.compile(
    r"^\s*(resum[ií](me)?(lo)?|ampli[aá]|expand[ií]|m[aá]s detalles?|contame m[aá]s"
    r"|y de eso|tell me more|summar(y|ize))\b",
    re.IGNORECASE,
)
_INFO_QUESTION_RE = re.compile(
    r"^\s*(qu[eé]\s+(sab[eé]s|sabes|conoc[eé]s)\s+(de|sobre|del?)"
    r"|tell me about|what do you know about)\s+(?P<topic>.+?)[?\s]*$",
    re.IGNORECASE,
)
_PRONOUN_PREFIX_RE = re.compile(
    r"^\s*(y\s+(él|ella|eso|esa|ese|esto)|and\s+(he|she|it|that))\b", re.IGNORECASE
)
_FILLERS = {
    "que",
    "qué",
    "del",
    "de",
    "la",
    "el",
    "los",
    "las",
    "un",
    "una",
    "sobre",
    "sabes",
    "sabés",
    "conocés",
    "proyecto",
    "the",
    "about",
    "tell",
    "what",
}
_WORD_RE = re.compile(r"[\wáéíóúñü\-]{3,}", re.IGNORECASE)


def _history_topic(history: list[dict[str, str]] | None) -> str | None:
    if not history:
        return None
    for turn in reversed(history):
        if turn.get("role") != "user":
            continue
        words = [
            w for w in _WORD_RE.findall(str(turn.get("content", ""))) if w.lower() not in _FILLERS
        ]
        if words:
            return " ".join(words[:8])
    return None


def rewrite_query(question: str, history: list[dict[str, str]] | None) -> str:
    q = (question or "").strip()
    info = _INFO_QUESTION_RE.match(q)
    if info:
        return info.group("topic").strip()
    if _SUMMARY_FOLLOWUP_RE.match(q):
        topic = _history_topic(history)
        if topic:
            return topic
    if _PRONOUN_PREFIX_RE.match(q):
        topic = _history_topic(history)
        if topic:
            return f"{topic} {q}"
    return q
