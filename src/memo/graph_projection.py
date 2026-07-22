"""Curated, deterministic serving projection for memo's raw knowledge graph.

The raw graph remains evidence.  This module decides which evidence is safe to
serve and gives accepted nodes stable namespaced identifiers.  Projection
storage and retrieval live here too so hot paths never need raw graph tables.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import quote

from memo.graph_canonical import fold_key

_SCALAR_RE = re.compile(
    r"^(?:true|false|null|none|nil|[-+]?\d+(?:\.\d+)?|\d{4}[-/]\d{1,2}[-/]\d{1,2})$",
    re.IGNORECASE,
)
_CODE_SHAPE_RE = re.compile(
    r"(?:^test(?:[_\s-]|$)|^assert\b|\w+\([^)]*\)|(?:==|!=|=>|::))",
    re.IGNORECASE,
)
_REFERENCE_TYPES = frozenset({"reference", "test", "repo", "repository"})
_EXTRACTOR_WEIGHT = {
    "legacy": 0.2,
    "regex": 0.45,
    "llm": 0.85,
    "explicit": 1.0,
}


@dataclass(frozen=True)
class ProjectionMemoryState:
    id: str
    type: str
    forgotten: bool = False


@dataclass(frozen=True)
class RawEntityEvidence:
    entity_id: int
    name: str
    entity_type: str
    extractors: tuple[str, ...]
    confidences: tuple[float, ...]
    memories: tuple[ProjectionMemoryState, ...]
    alias_count: int = 0


@dataclass(frozen=True)
class ProjectionBuildConfig:
    min_quality: float = 0.45
    hub_max_doc_freq_ratio: float = 0.25
    evidence_limit: int = 8


@dataclass(frozen=True)
class ProjectionDecision:
    eligible: bool
    quality: float
    reason: str | None
    uri: str


def entity_uri(entity_type: str, name: str) -> str:
    """Return a stable URI for a normalized entity identity."""
    normalized_type = entity_type.strip().lower() or "concept"
    return f"entity://{quote(normalized_type, safe='')}/{quote(fold_key(name), safe='')}"


def memory_uri(memory_id: str) -> str:
    return f"memory://{quote(memory_id.strip(), safe='')}"


def fact_uri(fact_id: str) -> str:
    return f"fact://{quote(fact_id.strip(), safe='')}"


def _is_scalar_or_date(name: str) -> bool:
    return bool(_SCALAR_RE.fullmatch(name.strip()))


def _is_code_shape(name: str) -> bool:
    return bool(_CODE_SHAPE_RE.search(name.strip()))


def evaluate_entity(
    evidence: RawEntityEvidence,
    config: ProjectionBuildConfig,
) -> ProjectionDecision:
    """Apply hard rejection rules and a bounded, deterministic quality score."""
    live = tuple(memory for memory in evidence.memories if not memory.forgotten)
    key = fold_key(evidence.name)
    uri = entity_uri(evidence.entity_type, evidence.name)
    explicit = "explicit" in evidence.extractors

    if not live:
        return ProjectionDecision(False, 0.0, "no_live_memory", uri)
    if not key:
        return ProjectionDecision(False, 0.0, "empty_key", uri)
    if _is_scalar_or_date(evidence.name):
        return ProjectionDecision(False, 0.0, "scalar_or_date", uri)
    if _is_code_shape(evidence.name) and not explicit:
        return ProjectionDecision(False, 0.0, "code_shape", uri)

    confidences = tuple(max(0.0, min(1.0, value)) for value in evidence.confidences)
    average_confidence = sum(confidences) / max(1, len(confidences))
    durable_ratio = (
        sum(memory.type.strip().lower() not in _REFERENCE_TYPES for memory in live)
        / len(live)
    )
    provenance = max(
        (_EXTRACTOR_WEIGHT.get(value, 0.0) for value in evidence.extractors),
        default=0.0,
    )
    quality = min(
        1.0,
        0.35 * average_confidence
        + 0.25 * provenance
        + 0.15 * durable_ratio
        + min(0.2, 0.05 * len(live))
        + (0.05 if evidence.entity_type != "concept" else 0.0),
    )
    if quality < config.min_quality:
        return ProjectionDecision(False, quality, "quality_below_threshold", uri)
    return ProjectionDecision(True, quality, None, uri)


__all__ = [
    "ProjectionBuildConfig",
    "ProjectionDecision",
    "ProjectionMemoryState",
    "RawEntityEvidence",
    "entity_uri",
    "evaluate_entity",
    "fact_uri",
    "memory_uri",
]
