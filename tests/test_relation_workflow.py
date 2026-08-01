from __future__ import annotations

from dataclasses import replace

import pytest

from memo.errors import NotFoundError


def test_post_save_candidates_are_capped_namespace_safe_and_llm_free(
    mock_memory, monkeypatch
) -> None:
    alpha = [
        mock_memory.save(
            content=f"alpha architecture decision {index}",
            title=f"alpha-{index}",
            type_="decision",
            tags=["project:alpha"],
        )
        for index in range(4)
    ]
    beta = mock_memory.save(
        content="beta architecture decision",
        title="beta",
        type_="decision",
        tags=["project:beta"],
    )
    hits = [
        replace(record, score=0.9 - index * 0.01) for index, record in enumerate([*alpha, beta])
    ]
    monkeypatch.setattr(mock_memory, "search", lambda *_a, **_k: hits)
    monkeypatch.setattr(
        mock_memory,
        "_ensure_chat",
        lambda: (_ for _ in ()).throw(AssertionError("candidate generation called an LLM")),
    )

    source = mock_memory.save(
        content="alpha replacement architecture decision",
        title="alpha-source",
        type_="decision",
        tags=["project:alpha"],
    )

    assert source.relation_detection == "ok"
    assert len(source.relation_candidates) == 3
    assert beta.id not in {row["target_id"] for row in source.relation_candidates}


def test_post_save_candidate_failure_never_fails_canonical_save(mock_memory, monkeypatch) -> None:
    monkeypatch.setattr(
        mock_memory,
        "detect_relation_candidates",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("index unavailable")),
    )

    saved = mock_memory.save(
        content="save must commit despite candidate failure",
        title="candidate failure isolation",
        type_="decision",
    )

    assert saved.action == "created"
    assert saved.relation_detection == "unavailable"
    assert mock_memory.get(saved.id) is not None


def test_deferred_embed_skips_relation_candidate_search(mock_memory, monkeypatch) -> None:
    monkeypatch.setattr(
        mock_memory,
        "detect_relation_candidates",
        lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("deferred save attempted semantic relation detection")
        ),
    )

    saved = mock_memory.save(
        content="save without a resident embedding model",
        title="deferred relation detection",
        type_="decision",
        defer_embed=True,
    )

    assert saved.extra["_memo_embed_pending"] is True
    assert saved.relation_candidates == []
    assert saved.relation_detection == "deferred"


def test_judged_relation_annotation_excludes_pending(mock_memory, monkeypatch) -> None:
    first = mock_memory.save(content="choice one", title="one", type_="decision")
    second = mock_memory.save(content="choice two", title="two", type_="decision")
    third = mock_memory.save(content="choice three", title="three", type_="decision")
    mock_memory.compare_memories(first.id, second.id, "compatible", reason="same scope")
    mock_memory.store.create_relation_candidate(source_id=first.id, target_id=third.id)

    annotated = mock_memory.annotate_relations([first])[0]
    relations = annotated.extra["memory_relations"]

    assert len(relations) == 1
    assert relations[0]["relation"] == "compatible"
    assert relations[0]["other_id"] == second.id


def test_supersedes_judgment_closes_validity_and_supports_as_of(mock_memory) -> None:
    old = mock_memory.save(
        content="use the legacy orange backend",
        title="backend before",
        type_="decision",
    )
    new = mock_memory.save(
        content="use the current violet backend",
        title="backend after",
        type_="decision",
    )

    mock_memory.compare_memories(new.id, old.id, "supersedes", reason="backend migration")

    closed = mock_memory.get(old.id)
    current_ids = {row.id for row in mock_memory.search("legacy orange backend", limit=10)}
    historical_ids = {
        row.id for row in mock_memory.search("legacy orange backend", limit=10, as_of=old.created)
    }
    assert closed is not None and closed.invalid_at is not None
    assert old.id not in current_ids
    assert old.id in historical_ids


def test_explicit_compare_rejects_missing_endpoints_without_ghost_row(mock_memory) -> None:
    existing = mock_memory.save(content="real endpoint", title="real", type_="decision")

    with pytest.raises(NotFoundError, match="target"):
        mock_memory.compare_memories(existing.id, "missing", "related")

    assert mock_memory.store.relation_stats() == {}


def test_failed_supersede_judgment_restores_validity(mock_memory, monkeypatch) -> None:
    old = mock_memory.save(content="old rollback value", title="old", type_="decision")
    new = mock_memory.save(content="new rollback value", title="new", type_="decision")
    candidate = mock_memory.store.create_relation_candidate(source_id=new.id, target_id=old.id)

    monkeypatch.setattr(
        mock_memory.store,
        "commit_relation_judgment",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("judgment store down")),
    )

    with pytest.raises(RuntimeError, match="judgment store down"):
        mock_memory.judge_relation(candidate["id"], "supersedes", reason="test rollback")

    restored = mock_memory.get(old.id)
    assert restored is not None and restored.invalid_at is None


def test_relation_candidate_search_does_not_record_user_usage(mock_memory, monkeypatch) -> None:
    mock_memory.save(
        content="the original internal search policy",
        title="original policy",
        type_="decision",
    )
    monkeypatch.setattr(
        mock_memory,
        "_record_access",
        lambda _ids: (_ for _ in ()).throw(
            AssertionError("internal candidate search recorded user access")
        ),
    )

    saved = mock_memory.save(
        content="the revised internal search policy",
        title="revised policy",
        type_="decision",
    )

    assert saved.relation_detection == "ok"


def test_relation_capabilities_support_explicit_opt_out(mock_memory, monkeypatch) -> None:
    monkeypatch.setenv("MEMO_RELATION_CANDIDATES_ENABLED", "0")
    saved = mock_memory.save(
        content="opted-out relation candidate generation",
        title="candidate opt-out",
        type_="decision",
    )
    assert saved.relation_detection == "disabled"

    other = mock_memory.save(content="annotation peer", title="peer", type_="note")
    mock_memory.compare_memories(saved.id, other.id, "related", reason="opt-out proof")
    monkeypatch.setenv("MEMO_RELATION_ANNOTATIONS_ENABLED", "0")
    annotated = mock_memory.annotate_relations([saved])[0]
    assert "memory_relations" not in annotated.extra
