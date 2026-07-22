from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from memo.code_traceability import CodeReference, codegraph_uri
from memo.graph import GraphStore
from memo.graph_projection import (
    ProjectionBuildConfig,
    ProjectionBuildError,
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


def _states(*ids: str) -> dict[str, ProjectionMemoryState]:
    return {id_: ProjectionMemoryState(id=id_, type="decision") for id_ in ids}


def _connected_graph(tmp_path: Path) -> GraphStore:
    graph = GraphStore(tmp_path / "graph.db")
    for memory_id in ("m1", "m2"):
        graph.record_extraction(
            memory_id=memory_id,
            memory_date="2026-07-20",
            entities=[
                {"name": "MLX", "type": "technology"},
                {"name": "recall daemon", "type": "project"},
            ],
            extracted_at="2026-07-20T00:00:00+00:00",
            extractor="explicit",
        )
    graph.rebuild_edges()
    return graph


def test_projection_rebuild_activates_complete_version(tmp_path: Path) -> None:
    graph = _connected_graph(tmp_path)

    result = graph.projection.rebuild(_states("m1", "m2"), ProjectionBuildConfig())
    model = graph.projection.read_model(max_age_hours=36)

    assert result.activated is True
    assert model.version == result.version
    assert model.memory_nodes("m1")
    edge = next(iter(model.neighbors(entity_uri("technology", "mlx"))))
    assert edge.evidence_ids == ("memory://m1", "memory://m2")


def test_projection_materializes_bidirectional_memory_code_links(tmp_path: Path) -> None:
    graph = _connected_graph(tmp_path)
    uri = codegraph_uri("memo-repo", "file:src/memo/graph.py")
    ref = CodeReference(
        uri=uri,
        repo_id="memo-repo",
        stable_symbol_id="file:src/memo/graph.py",
        kind="file",
        label="graph.py",
        qualified_name="src/memo/graph.py",
        file_path="src/memo/graph.py",
        start_line=1,
        end_line=900,
        relation="modified",
        confidence=0.95,
    )
    states = _states("m1", "m2")
    states["m1"] = replace(states["m1"], code_refs=(ref,))

    result = graph.projection.rebuild(states, ProjectionBuildConfig())
    model = graph.projection.read_model(max_age_hours=36)

    assert result.code_node_count == 1
    assert result.code_link_count == 1
    assert model.code_links_for_memory("m1")[0].uri == uri
    assert model.code_links_for_uri(uri)[0].memory_id == "m1"
    assert model.resolve_code("src/memo/graph.py")[0].uri == uri
    edge = next(edge for edge in model.neighbors(uri) if edge.relation == "contextualizes_code")
    assert edge.evidence_ids == ("memory://m1",)
    health = graph.projection.health()
    assert health["code_node_count"] == 1
    assert health["code_link_count"] == 1


def test_failed_projection_validation_preserves_previous_active_version(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _connected_graph(tmp_path)
    first = graph.projection.rebuild(_states("m1", "m2"), ProjectionBuildConfig())

    def _fail_validation(*_args: object) -> None:
        raise ProjectionBuildError("invalid")

    monkeypatch.setattr(graph.projection, "_validate_version", _fail_validation)

    with pytest.raises(ProjectionBuildError):
        graph.projection.rebuild(_states("m1", "m2"), ProjectionBuildConfig())

    health = graph.projection.health()
    assert health["active_version"] == first.version
    assert "invalid" in health["last_error"]


def test_stale_projection_returns_unavailable_read_model(tmp_path: Path) -> None:
    graph = _connected_graph(tmp_path)
    built = datetime(2026, 7, 20, tzinfo=UTC)
    graph.projection.rebuild(
        _states("m1", "m2"),
        ProjectionBuildConfig(),
        now=built,
    )

    model = graph.projection.read_model(
        max_age_hours=24,
        now=built + timedelta(hours=25),
    )

    assert model.available is False
    assert model.skip_reason == "projection_stale"


def test_rebuild_quarantines_rejections_without_deleting_raw_rows(
    tmp_path: Path,
) -> None:
    graph = GraphStore(tmp_path / "graph.db")
    graph.record_extraction(
        memory_id="m1",
        memory_date="2026-07-20",
        entities=[{"name": "Memo", "type": "project"}],
        extracted_at="2026-07-20T00:00:00+00:00",
        extractor="explicit",
    )
    graph.record_extraction(
        memory_id="m2",
        memory_date="2026-07-20",
        entities=[{"name": "test_rank_hits", "type": "concept"}],
        extracted_at="2026-07-20T00:00:00+00:00",
        extractor="regex",
        confidence=0.45,
    )

    result = graph.projection.rebuild(_states("m1", "m2"), ProjectionBuildConfig())

    assert result.rejected_count == 1
    assert graph.count_entities() == 2
    assert graph.projection.health()["rejection_reasons"]["code_shape"] == 1


def test_memory_rebuild_graph_prunes_orphans_and_activates_projection(
    mem_with_stub,
) -> None:
    rec = mem_with_stub.save(
        content="MLX daemon knowledge",
        title="MLX runtime",
        type_="decision",
    )
    mem_with_stub.graph.record_extraction(
        memory_id=rec.id,
        memory_date="2026-07-20",
        entities=[{"name": "MLX", "type": "technology"}],
        extracted_at="2026-07-20T00:00:00+00:00",
        extractor="explicit",
    )
    mem_with_stub.graph.record_extraction(
        memory_id="gone",
        memory_date="2026-07-20",
        entities=[{"name": "orphan", "type": "concept"}],
        extracted_at="2026-07-20T00:00:00+00:00",
        extractor="explicit",
    )

    result = mem_with_stub.rebuild_graph()

    assert result.orphan_links_pruned == 1
    assert result.projection.activated is True
    assert mem_with_stub.graph.entity_memories("orphan") == []
    assert mem_with_stub.graph.projection.read_model(36).memory_nodes(rec.id)
