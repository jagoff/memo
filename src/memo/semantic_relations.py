"""Deterministic semantic-relation extraction for the memory graph.

Semantic relations are rebuildable graph state. This module intentionally keeps
the first extractor cheap and deterministic: it looks for explicit relationship
language in one memory and links it to another memory whose title/id is named in
that text. LLM extraction can be layered on later behind the same dataclass.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

RelationType = Literal[
    "causes",
    "contradicts",
    "depends_on",
    "extends",
    "implements",
    "supports",
    "supersedes",
]

DETERMINISTIC_DERIVED_FROM = "memo.semantic_relations.deterministic.v1"

_RELATION_PATTERNS: tuple[tuple[RelationType, tuple[str, ...], float], ...] = (
    ("supersedes", (r"\bsupersedes?\b", r"\breplaces?\b", r"\bdeprecated?\b"), 0.82),
    (
        "contradicts",
        (r"\bcontradicts?\b", r"\bconflicts? with\b", r"\bnot true\b", r"\brejected?\b"),
        0.78,
    ),
    ("depends_on", (r"\bdepends on\b", r"\brequires?\b", r"\bneeds?\b", r"\bblocked by\b"), 0.74),
    ("implements", (r"\bimplements?\b", r"\bships?\b", r"\badds?\b", r"\bintroduces?\b"), 0.72),
    ("supports", (r"\bsupports?\b", r"\benables?\b", r"\bfix(?:es|ed)?\b", r"\bimproves?\b"), 0.70),
    ("causes", (r"\bcauses?\b", r"\bleads to\b", r"\btriggers?\b"), 0.70),
    ("extends", (r"\bextends?\b", r"\bbuilds on\b", r"\bfollows up\b", r"\bcontinues?\b"), 0.68),
)

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_.:/-]{2,}", re.IGNORECASE)


@dataclass(frozen=True)
class RelationMemory:
    id: str
    title: str = ""
    body: str = ""


@dataclass(frozen=True)
class SemanticRelation:
    """A semantic relation between two memories."""

    source_id: str
    target_id: str
    relation_type: RelationType
    confidence: float
    evidence_id: str | None = None
    derived_from: str = DETERMINISTIC_DERIVED_FROM


def _field(obj: Any, name: str, default: str = "") -> str:
    value = obj.get(name, default) if isinstance(obj, Mapping) else getattr(obj, name, default)
    return "" if value is None else str(value)


def coerce_memory(obj: Any) -> RelationMemory:
    """Coerce MemoryRecord/dict/test fixtures into a small extractor record."""

    if isinstance(obj, str):
        return RelationMemory(id=obj, title=obj)
    return RelationMemory(
        id=_field(obj, "id"),
        title=_field(obj, "title"),
        body=_field(obj, "body"),
    )


def _tokens(text: str) -> set[str]:
    return {m.group(0).lower() for m in _TOKEN_RE.finditer(text)}


def _names_target(source_text: str, target: RelationMemory) -> bool:
    source = source_text.lower()
    target_id = target.id.strip().lower()
    if target_id and target_id in source:
        return True
    title = target.title.strip().lower()
    if title and title in source:
        return True
    target_terms = _tokens(f"{target.title} {target.body}")
    if not target_terms:
        return False
    source_terms = _tokens(source_text)
    return len(source_terms & target_terms) >= min(2, len(target_terms))


def _detect_relation(source: RelationMemory, target: RelationMemory) -> SemanticRelation | None:
    if not source.id or not target.id or source.id == target.id:
        return None
    text = f"{source.title}\n{source.body}".strip()
    if not text or not _names_target(text, target):
        return None
    for relation, patterns, confidence in _RELATION_PATTERNS:
        if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns):
            return SemanticRelation(
                source_id=source.id,
                target_id=target.id,
                relation_type=relation,
                confidence=confidence,
                evidence_id=source.id,
            )
    return None


def extract_relations_batch(
    memory_pairs: Iterable[tuple[Any, Any]], model: str = "deterministic"
) -> list[SemanticRelation]:
    """Extract explicit semantic relations from a batch of memory pairs.

    ``model`` is accepted for API compatibility; the current implementation is
    deterministic and ignores remote model configuration.
    """

    del model
    out: dict[tuple[str, str, str], SemanticRelation] = {}
    for source_obj, target_obj in memory_pairs:
        relation = _detect_relation(coerce_memory(source_obj), coerce_memory(target_obj))
        if relation is None:
            continue
        key = (relation.source_id, relation.target_id, relation.relation_type)
        out[key] = relation
    return list(out.values())


def store_relations(graph: Any, relations: Iterable[SemanticRelation]) -> int:
    """Write extracted relations to ``GraphStore`` and return the row count."""

    n = 0
    for rel in relations:
        graph.upsert_semantic_relation(
            source_kind="memory",
            source_id=rel.source_id,
            target_kind="memory",
            target_id=rel.target_id,
            relation=rel.relation_type,
            weight=rel.confidence,
            confidence=rel.confidence,
            evidence_id=rel.evidence_id,
            derived_from=rel.derived_from,
        )
        n += 1
    return n


__all__ = [
    "DETERMINISTIC_DERIVED_FROM",
    "RelationMemory",
    "RelationType",
    "SemanticRelation",
    "coerce_memory",
    "extract_relations_batch",
    "store_relations",
]
