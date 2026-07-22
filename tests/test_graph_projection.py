from __future__ import annotations

from dataclasses import replace

import pytest

from memo.graph_projection import (
    ProjectionBuildConfig,
    ProjectionMemoryState,
    RawEntityEvidence,
    entity_uri,
    evaluate_entity,
    fact_uri,
    memory_uri,
)


def _memory(id_: str = "m1", *, type_: str = "decision") -> ProjectionMemoryState:
    return ProjectionMemoryState(id=id_, type=type_)


def _evidence(
    name: str,
    *,
    entity_type: str = "concept",
    extractor: str = "regex",
    confidence: float = 0.45,
    memory: ProjectionMemoryState | None = None,
) -> RawEntityEvidence:
    return RawEntityEvidence(
        entity_id=1,
        name=name,
        entity_type=entity_type,
        extractors=(extractor,),
        confidences=(confidence,),
        memories=(memory or _memory(),),
    )


@pytest.mark.parametrize(
    ("type_", "name", "expected"),
    [
        ("technology", "Fast API", "entity://technology/fastapi"),
        ("project", "Postgres", "entity://project/postgresql"),
        ("person", "José Núñez", "entity://person/josenunez"),
    ],
)
def test_entity_uri_is_stable(type_: str, name: str, expected: str) -> None:
    assert entity_uri(type_, name) == expected


def test_evidence_uris_are_namespaced_and_encoded() -> None:
    assert memory_uri("memo 1") == "memory://memo%201"
    assert fact_uri("fact/1") == "fact://fact%2F1"


@pytest.mark.parametrize(
    "name",
    ["", "true", "42", "2026-07-22", "test_rank_hits", "assert foo == bar"],
)
def test_projection_rejects_non_knowledge_shapes(name: str) -> None:
    decision = evaluate_entity(_evidence(name), ProjectionBuildConfig())

    assert decision.eligible is False
    assert decision.reason


def test_explicit_non_code_evidence_can_keep_test_shaped_real_name() -> None:
    evidence = _evidence("Test Kitchen", extractor="explicit", confidence=0.95)

    decision = evaluate_entity(evidence, ProjectionBuildConfig())

    assert decision.eligible is True


def test_reference_only_regex_concept_falls_below_quality_floor() -> None:
    evidence = _evidence(
        "fixture helper",
        memory=_memory("r1", type_="reference"),
    )

    decision = evaluate_entity(evidence, ProjectionBuildConfig(min_quality=0.45))

    assert decision.eligible is False
    assert decision.reason == "quality_below_threshold"


def test_forgotten_evidence_is_not_live() -> None:
    evidence = _evidence("memo", extractor="llm", confidence=0.85)
    evidence = replace(
        evidence,
        memories=(replace(evidence.memories[0], forgotten=True),),
    )

    decision = evaluate_entity(evidence, ProjectionBuildConfig())

    assert decision.reason == "no_live_memory"
