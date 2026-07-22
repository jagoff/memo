"""Focused contracts for post-retrieval scoring helpers.

These tests keep the ranking hot path deterministic and exercise its failure
isolation without constructing a full ``Memory`` instance for every branch.
"""

from __future__ import annotations

import sqlite3
from unittest.mock import MagicMock

import pytest

from memo.memory.record import MemoryRecord
from memo.memory.search_scoring_ops import _SearchScoringMixin


class _Harness(_SearchScoringMixin):
    pass


def _record(
    id_: str,
    score: float | None = 1.0,
    *,
    type_: str = "note",
    extra: dict | None = None,
) -> MemoryRecord:
    return MemoryRecord(
        id=id_,
        path=f"{id_}.md",
        title=id_,
        type=type_,
        tags=["test"],
        created="2026-01-01T00:00:00+00:00",
        updated="2026-01-01T00:00:00+00:00",
        body=f"body for {id_}",
        extra=extra or {},
        score=score,
    )


def test_entity_boost_copies_records_and_resorts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("memo.entity_extractor.extract_entities", lambda _query: ["memo"])
    monkeypatch.setattr(
        "memo.entity_extractor.entity_match_score",
        lambda _query, document: 0.4 if "memo" in document else 0.0,
    )
    plain = _record("plain", 0.5, extra={"entities": []})
    matching = _record("matching", 0.2, extra={"entities": ["memo"]})

    out = _Harness()._apply_entity_boost("memo", [plain, matching])

    assert [hit.id for hit in out] == ["matching", "plain"]
    assert out[0].score == pytest.approx(0.6)
    assert matching.score == 0.2


def test_co_recall_boost_scales_by_strongest_edge_and_resorts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MEMO_CO_RECALL_BOOST_WEIGHT", "0.2")
    harness = _Harness()
    harness.graph = MagicMock()
    harness.graph.co_recall_counts.return_value = {"b": 2, "c": 4}

    out = harness._apply_co_recall_boost(
        [_record("anchor", 0.8), _record("b", 0.75), _record("c", 0.7)]
    )

    assert [hit.id for hit in out] == ["c", "b", "anchor"]
    assert {hit.id: hit.score for hit in out} == {
        "c": pytest.approx(0.9),
        "b": pytest.approx(0.85),
        "anchor": pytest.approx(0.8),
    }


def test_co_recall_boost_is_identity_for_small_or_broken_inputs() -> None:
    harness = _Harness()
    harness.graph = MagicMock()
    small = [_record("a"), _record("b")]
    assert harness._apply_co_recall_boost(small) is small

    hits = [_record("a"), _record("b"), _record("c")]
    harness.graph.co_recall_counts.side_effect = RuntimeError("graph unavailable")
    assert harness._apply_co_recall_boost(hits) is hits


def test_retrieval_boost_ignores_untrusted_metadata_and_forwards_curated_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict] = []

    def _boost_for(**kwargs) -> float:
        calls.append(kwargs)
        return 4.0

    monkeypatch.setattr("memo.retrieval_boost.boost_for", _boost_for)
    harness = _Harness()
    harness.store = MagicMock()
    harness.store.get_health_batch.return_value = {
        "untrusted": {"confidence": 0.4, "roi_score": 1.0}
    }
    untrusted = _record("untrusted", 0.9, extra={"chunk_heading": "generated"})
    curated = _record("curated", 0.3, extra={"chunk_heading": "Install"})

    out = harness._apply_retrieval_boost("install memo", [untrusted, curated])

    assert [hit.id for hit in out] == ["curated", "untrusted"]
    assert out[0].score == pytest.approx(1.2)
    assert len(calls) == 1
    assert calls[0]["headings"] == ["Install"]
    assert calls[0]["tags"] == ["test"]


def test_health_scores_apply_only_known_non_neutral_multipliers() -> None:
    harness = _Harness()
    harness.store = MagicMock()
    harness.store.get_health_batch.return_value = {
        "high": {"confidence": 0.5, "roi_score": 1.0},
        "low": {"confidence": 1.0, "roi_score": 1.5},
    }

    out = harness._apply_health_scores(
        [_record("high", 0.9), _record("missing", 0.6), _record("low", 0.5)]
    )

    assert [hit.id for hit in out] == ["low", "missing", "high"]
    assert {hit.id: hit.score for hit in out} == {
        "low": pytest.approx(0.75),
        "missing": pytest.approx(0.6),
        "high": pytest.approx(0.45),
    }


@pytest.mark.parametrize(
    ("health_disabled", "outcome_enabled", "expect_roi_boost"),
    [("0", "0", True), ("1", "0", False), ("0", "1", False)],
)
def test_record_access_keeps_touch_authoritative_and_gates_legacy_roi(
    monkeypatch: pytest.MonkeyPatch,
    health_disabled: str,
    outcome_enabled: str,
    expect_roi_boost: bool,
) -> None:
    monkeypatch.setenv("MEMO_HEALTH_SCORES_DISABLED", health_disabled)
    monkeypatch.setenv("MEMO_OUTCOME_RANKING_ENABLED", outcome_enabled)
    harness = _Harness()
    harness.store = MagicMock()

    harness._record_access(["a", "b"])

    harness.store.touch.assert_called_once_with(["a", "b"])
    assert harness.store.boost_roi_batch.called is expect_roi_boost


def test_record_access_ignores_empty_ids_and_sqlite_lock_errors() -> None:
    harness = _Harness()
    harness.store = MagicMock()
    harness._record_access([])
    harness.store.touch.assert_not_called()

    harness.store.touch.side_effect = sqlite3.OperationalError("database is locked")
    harness._record_access(["a"])


def test_cache_read_through_filters_bad_candidates_and_survives_one_save_failure() -> None:
    harness = _Harness()
    existing = [_record("local", 0.8)]
    backend = MagicMock()
    backend.fetch.return_value = [
        {"id": "local", "body": "duplicate"},
        {"id": "blank", "body": "   "},
        {"id": "broken", "body": "cannot save"},
        {"id": "remote", "body": "remote body", "score": 0.7},
    ]
    harness._cache_backend = lambda: backend  # type: ignore[method-assign]

    def _save(*, content: str, **_kwargs) -> MemoryRecord:
        if content == "cannot save":
            raise OSError("disk full")
        return _record("materialized", None)

    harness.save = _save  # type: ignore[method-assign]

    out = harness._cache_read_through("query", existing, limit=3)

    assert [hit.id for hit in out] == ["local", "materialized"]
    assert out[1].score == pytest.approx(0.7)


def test_cache_read_through_does_not_fetch_when_limit_is_already_satisfied() -> None:
    harness = _Harness()
    backend = MagicMock()
    harness._cache_backend = lambda: backend  # type: ignore[method-assign]
    existing = [_record("first", 0.9), _record("second", 0.8)]

    out = harness._cache_read_through("query", existing, limit=1)

    assert out == [existing[0]]
    backend.fetch.assert_not_called()


def test_cache_read_through_deduplicates_repeated_backend_candidates() -> None:
    """A noisy backend must not produce duplicate hits or duplicate saves."""

    harness = _Harness()
    backend = MagicMock()
    candidate = {"id": "remote-1", "body": "same remote body", "score": 0.9}
    backend.fetch.return_value = [dict(candidate), dict(candidate)]
    harness._cache_backend = lambda: backend  # type: ignore[method-assign]
    harness.save = MagicMock(return_value=_record("local-content-id", None))  # type: ignore[method-assign]

    out = harness._cache_read_through("query", [], limit=5)

    assert [hit.id for hit in out] == ["local-content-id"]
    harness.save.assert_called_once()


def test_cache_read_through_persists_backend_identity_on_local_id_collision() -> None:
    """A content-address collision must remain deduplicated on the next read."""

    harness = _Harness()
    existing = _record("local-content-id", 0.8, extra={"origin": "local"})
    backend = MagicMock()
    backend.fetch.return_value = [{"id": "remote-1", "body": "same remote body", "score": 0.9}]
    harness._cache_backend = lambda: backend  # type: ignore[method-assign]
    harness.save = MagicMock(return_value=existing)  # type: ignore[method-assign]

    def _update(id_: str, *, extra: dict) -> MemoryRecord:
        assert id_ == existing.id
        # Real metadata-only updates return a fresh record without a retrieval
        # score; cache identity persistence must not erase the live hit score.
        return _record(existing.id, None, extra=extra)

    harness.update = MagicMock(side_effect=_update)  # type: ignore[method-assign]

    first = harness._cache_read_through("query", [existing], limit=5)
    second = harness._cache_read_through("query", first, limit=5)

    assert first[0].extra == {"origin": "local", "cache_backend_id": "remote-1"}
    assert first[0].score == pytest.approx(0.8)
    assert second == first
    harness.save.assert_called_once()
    harness.update.assert_called_once_with(
        existing.id,
        extra={"origin": "local", "cache_backend_id": "remote-1"},
    )


def test_cache_read_through_stops_after_remaining_slots_are_filled() -> None:
    """Failed saves do not consume slots, and excess candidates are not materialized."""

    harness = _Harness()
    existing = [_record("local", 0.8)]
    backend = MagicMock()
    backend.fetch.return_value = [
        {"id": "broken", "body": "broken", "score": 0.7},
        {"id": "remote-1", "body": "one", "score": 0.6},
        {"id": "remote-2", "body": "two", "score": 0.5},
        {"id": "excess", "body": "must not save", "score": 0.4},
    ]
    harness._cache_backend = lambda: backend  # type: ignore[method-assign]

    def _save(*, content: str, **_kwargs) -> MemoryRecord:
        if content == "broken":
            raise OSError("disk full")
        return _record(f"local-{content}", None)

    harness.save = MagicMock(side_effect=_save)  # type: ignore[method-assign]

    out = harness._cache_read_through("query", existing, limit=3)

    assert [hit.id for hit in out] == ["local", "local-one", "local-two"]
    assert [call.kwargs["content"] for call in harness.save.call_args_list] == [
        "broken",
        "one",
        "two",
    ]
