"""Lightweight entity extraction for entity-aware retrieval.

Extracts named entity mentions from text to power entity_match_score
boosts in the search pipeline. Dependency-free: uses regex patterns
that work well for personal knowledge bases (person names, technology
names, project names, quoted identifiers).

Optional GLiNER upgrade: if `gliner` is installed and
MEMO_ENTITY_GLINER=1, uses the zero-shot NER model for higher recall
at the cost of ~200ms cold-load. Falls back silently to regex.

Feature flag: MEMO_ENTITY_RETRIEVAL_ENABLED (default 0).
"""

from __future__ import annotations

import logging
import re

_log = logging.getLogger("memo.entity_extractor")

__all__ = [
    "entity_match_score",
    "entity_retrieval_enabled",
    "extract_entities",
]

# -- Regex patterns for dependency-free entity extraction -------------------

# Sequences of ≥2 consecutive capitalized words (proper noun phrases).
# "Fernando Ferrari", "React Native", "Apple Silicon"
_PROPER_NOUN_RE = re.compile(
    r"\b[A-ZÁÉÍÓÚÜÑ][a-záéíóúüñA-ZÁÉÍÓÚÜÑ]+(?:\s+[A-ZÁÉÍÓÚÜÑ][a-záéíóúüñA-ZÁÉÍÓÚÜÑ]+)+"
)

# Single capitalized words that are likely proper nouns (≥4 chars, not at
# sentence start — heuristic: preceded by a non-sentence-boundary char).
_SINGLE_PROPER_RE = re.compile(r"(?<=\w\s)[A-ZÁÉÍÓÚÜÑ][a-záéíóúüñ]{3,}")

# CamelCase identifiers: FastAPI, SQLite, MLXEmbedder, Qwen3
_CAMEL_RE = re.compile(r"\b[A-Z][a-z]+(?:[A-Z][a-z0-9]+)+\b")

# ALL_CAPS acronyms (≥2 chars): BM25, API, MCP, RAG, LLM
_ACRONYM_RE = re.compile(r"\b[A-Z]{2,}\d*\b")

# Backtick/quote-delimited identifiers: `memo-mcp`, "obsidian-rag"
_BACKTICK_RE = re.compile(r"`([^`]+)`")
_QUOTED_RE = re.compile(r'"([^"]{3,40})"')

# Spanish/English stopwords to filter single-word proper nouns.
_STOPWORDS = frozenset(
    {
        "the",
        "The",
        "Esta",
        "Este",
        "Esto",
        "Aquí",
        "Hay",
        "Pero",
        "Para",
        "Como",
        "Cuando",
        "Donde",
        "Algo",
        "Cada",
        "Cuándo",
        "Cómo",
        "Qué",
        "Quién",
        "Cuál",
        "Ningún",
        "Todos",
        "Todas",
        "Desde",
        "Hasta",
        "Según",
        "También",
        "Además",
        "Aunque",
        "Sin embargo",
        "Por lo tanto",
        "En cambio",
    }
)

# Min length for entity inclusion.
_MIN_LEN = 3


def entity_retrieval_enabled() -> bool:
    from memo.flags import flag_bool

    return flag_bool("MEMO_ENTITY_RETRIEVAL_ENABLED")


def _gliner_enabled() -> bool:
    from memo.flags import flag_bool

    return flag_bool("MEMO_ENTITY_GLINER")


def _extract_regex(text: str) -> list[str]:
    """Regex-based entity extraction (dependency-free)."""
    found: list[str] = []
    seen: set[str] = set()

    def _add(s: str) -> None:
        s = s.strip()
        if len(s) >= _MIN_LEN and s not in seen and s not in _STOPWORDS:
            seen.add(s)
            found.append(s)

    for m in _PROPER_NOUN_RE.finditer(text):
        _add(m.group())
    for m in _SINGLE_PROPER_RE.finditer(text):
        _add(m.group())
    for m in _CAMEL_RE.finditer(text):
        _add(m.group())
    for m in _ACRONYM_RE.finditer(text):
        if len(m.group()) >= 2:
            _add(m.group())
    for m in _BACKTICK_RE.finditer(text):
        _add(m.group(1))
    for m in _QUOTED_RE.finditer(text):
        _add(m.group(1))
    return found


def _extract_gliner(text: str, labels: list[str]) -> list[str]:
    """GLiNER-based extraction (requires `pip install gliner`)."""
    try:
        import gliner  # type: ignore[import]
    except ImportError:
        # Expected when the optional dependency isn't installed — regex fallback.
        return _extract_regex(text)
    try:
        from memo.flags import flag_str

        model_name = flag_str("MEMO_ENTITY_GLINER_MODEL")
        # Module-level singleton to avoid repeated model loads.
        _model = getattr(_extract_gliner, "_model", None)
        if _model is None or getattr(_model, "_model_name", "") != model_name:
            _model = gliner.GLiNER.from_pretrained(model_name)
            _model._model_name = model_name  # type: ignore[attr-defined]
            _extract_gliner._model = _model  # type: ignore[attr-defined]
        entities = _model.predict_entities(text[:2000], labels, threshold=0.5)
        return list({e["text"] for e in entities if e.get("text")})
    except Exception as exc:
        # Model load / prediction failure (OOM, shape mismatch, bad model name):
        # surface it as a warning rather than silently degrading to regex.
        _log.warning("GLiNER extraction failed, falling back to regex: %s", exc)
        return _extract_regex(text)


def extract_entities(
    text: str,
    *,
    labels: list[str] | None = None,
) -> list[str]:
    """Extract entity mentions from ``text``.

    Returns a deduplicated list of entity strings. Falls back to regex
    extraction when GLiNER is unavailable.

    Args:
        text: The text to extract entities from.
        labels: Entity type labels for GLiNER (default: person, org,
                location, technology, project). Ignored in regex mode.
    """
    if not text or not text.strip():
        return []
    if _gliner_enabled():
        _labels = labels or ["person", "organization", "location", "technology", "project"]
        return _extract_gliner(text, _labels)
    return _extract_regex(text)


def entity_match_score(
    query_entities: list[str],
    doc_entities: list[str],
    *,
    boost_per_match: float = 0.05,
    max_boost: float = 0.2,
) -> float:
    """Return a score boost based on entity overlap between query and doc.

    Case-insensitive intersection. Each matching entity contributes
    ``boost_per_match``; total is capped at ``max_boost``.
    """
    if not query_entities or not doc_entities:
        return 0.0
    q_set = {e.lower() for e in query_entities}
    d_set = {e.lower() for e in doc_entities}
    overlap = len(q_set & d_set)
    return min(max_boost, boost_per_match * overlap)
