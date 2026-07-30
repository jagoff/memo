"""Focused contracts for post-retrieval scoring helpers.

These tests keep the ranking hot path deterministic and exercise its failure
isolation without constructing a full ``Memory`` instance for every branch.
"""

from __future__ import annotations

import sqlite3
from dataclasses import replace
from unittest.mock import MagicMock, call

import pytest

from memo.memory.record import MemoryRecord
from memo.memory.search_scoring_ops import _older_id, _SearchScoringMixin


class _Pair:
    """Minimal stand-in for contradict.PairRecord (only fields the penalty reads)."""

    def __init__(self, a: str, b: str, relationship: str | None = "contradiction") -> None:
        self.memory_id_a = a
        self.memory_id_b = b
        self.relationship = relationship


def _contradict_harness(pair: _Pair) -> _Harness:
    harness = _Harness()
    harness.contradict_store = MagicMock()

    def _pairs(_ids: list[str], status: str | None = None) -> list[_Pair]:
        # The default (no status) fetch carries the contradiction pair; the
        # follow-up status="evolved" fetch is empty here.
        return [pair] if status is None else []

    harness.contradict_store.pairs_for_ids.side_effect = _pairs
    return harness


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


def test_contradict_penalty_demotes_truly_older_side_across_utc_offsets() -> None:
    """F1: the older side must be chosen by UTC instant, not raw ISO-string order.

    ``a`` sorts lexicographically BEFORE ``b`` yet is NEWER in real time:
      a: 2026-06-04T23:30:00-05:00  ->  2026-06-05T04:30:00Z  (newer)
      b: 2026-06-05T01:00:00+00:00  ->  2026-06-05T01:00:00Z  (older)
    A raw ``a_ts < b_ts`` compare would demote ``a`` (the newer side) — the bug.
    """
    a = replace(_record("a" * 32, 0.5), updated="2026-06-04T23:30:00-05:00")
    b = replace(_record("b" * 32, 0.5), updated="2026-06-05T01:00:00+00:00")
    harness = _contradict_harness(_Pair("a" * 32, "b" * 32))

    out = harness._apply_contradict_penalty([a, b])
    scores = {r.id: r.score for r in out}

    # The truly-older side (b, earlier UTC instant) is demoted; the newer (a) is not.
    assert scores["b" * 32] == pytest.approx(0.2)  # 0.5 * 0.4 default penalty
    assert scores["a" * 32] == pytest.approx(0.5)
    assert [r.id for r in out] == ["a" * 32, "b" * 32]


def test_contradict_penalty_honors_explicit_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """F2: an explicitly configured 0 penalty must not be replaced by the 0.4 default."""
    monkeypatch.setenv("MEMO_CONTRADICT_PENALTY", "0")
    newer = replace(_record("a" * 32, 0.5), updated="2026-06-05T00:00:00+00:00")
    older = replace(_record("b" * 32, 0.5), updated="2026-01-01T00:00:00+00:00")
    harness = _contradict_harness(_Pair("a" * 32, "b" * 32))

    out = harness._apply_contradict_penalty([newer, older])
    scores = {r.id: r.score for r in out}

    # Older side * 0.0 == 0.0 (explicit zero honored); with `or 0.4` it'd be 0.2.
    assert scores["b" * 32] == pytest.approx(0.0)
    assert scores["a" * 32] == pytest.approx(0.5)


def test_older_id_handles_naive_invalid_and_equal_instants() -> None:
    assert (
        _older_id(
            "naive",
            "2026-01-01T00:00:00",
            "aware",
            "2026-01-01T01:00:00+00:00",
        )
        == "naive"
    )
    assert _older_id("a", "invalid", "b", "2026-01-01T00:00:00+00:00") is None
    assert _older_id("a", "2026-01-01T00:00:00+00:00", "b", "invalid") is None
    assert (
        _older_id(
            "a",
            "2026-01-01T00:00:00+00:00",
            "b",
            "2025-12-31T19:00:00-05:00",
        )
        == "b"
    )


def test_contradict_penalty_queries_exact_statuses_and_applies_evolution_default() -> None:
    a = replace(_record("a", 0.8), updated="2025-01-01T00:00:00+00:00")
    b = replace(_record("b", 0.7), updated="2026-01-01T00:00:00+00:00")
    c = replace(_record("c", 0.6), updated="2025-02-01T00:00:00+00:00")
    d = replace(_record("d", 0.5), updated="2026-02-01T00:00:00+00:00")
    harness = _Harness()
    harness.contradict_store = MagicMock()

    def pairs(ids: list[str], status: str | None = None) -> list[_Pair]:
        assert ids == ["a", "b", "c", "d"]
        if status is None:
            return [_Pair("a", "b", "contradiction")]
        assert status == "evolved"
        return [_Pair("c", "d", "evolution")]

    harness.contradict_store.pairs_for_ids.side_effect = pairs

    out = harness._apply_contradict_penalty([a, b, c, d])

    harness.contradict_store.pairs_for_ids.assert_has_calls(
        [
            call(["a", "b", "c", "d"]),
            call(["a", "b", "c", "d"], status="evolved"),
        ]
    )
    assert {hit.id: hit.score for hit in out} == {
        "a": pytest.approx(0.32),
        "b": pytest.approx(0.7),
        "c": pytest.approx(0.42),
        "d": pytest.approx(0.5),
    }
    assert [hit.id for hit in out] == ["b", "d", "c", "a"]


def test_evolution_demotes_b_side_with_none_score() -> None:
    newer = replace(_record("newer", 0.5), updated="2026-01-01T00:00:00+00:00")
    older = replace(_record("older", None), updated="2025-01-01T00:00:00+00:00")
    harness = _Harness()
    harness.contradict_store = MagicMock()
    harness.contradict_store.pairs_for_ids.side_effect = [
        [],
        [_Pair("newer", "older", "evolution")],
    ]

    out = harness._apply_contradict_penalty([newer, older])

    assert [hit.id for hit in out] == ["newer", "older"]
    assert out[0] is newer
    assert out[1].id == "older"
    assert out[1].score == 0.0


def test_unknown_relationship_does_not_hide_later_evolution() -> None:
    unknown_a = replace(_record("unknown-a", 0.9), updated="2024-01-01T00:00:00+00:00")
    unknown_b = replace(_record("unknown-b", 0.8), updated="2025-01-01T00:00:00+00:00")
    older = replace(_record("older", 0.7), updated="2025-02-01T00:00:00+00:00")
    newer = replace(_record("newer", 0.6), updated="2026-02-01T00:00:00+00:00")
    harness = _Harness()
    harness.contradict_store = MagicMock()
    harness.contradict_store.pairs_for_ids.side_effect = [
        [_Pair("unknown-a", "unknown-b", "related")],
        [_Pair("older", "newer", "evolution")],
    ]

    out = harness._apply_contradict_penalty([unknown_a, unknown_b, older, newer])

    assert {hit.id: hit.score for hit in out} == {
        "unknown-a": pytest.approx(0.9),
        "unknown-b": pytest.approx(0.8),
        "older": pytest.approx(0.49),
        "newer": pytest.approx(0.6),
    }


def test_declared_open_pair_queries_exact_ids_and_preserves_evolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MEMO_DECLARE_DISPUTES", "1")
    older = replace(_record("older", 0.5), updated="2025-01-01T00:00:00+00:00")
    newer = replace(_record("newer", 0.4), updated="2026-01-01T00:00:00+00:00")
    pair = _Pair("older", "newer", "evolution")
    harness = _Harness()
    harness.contradict_store = MagicMock()

    def pairs(ids: list[str], status: str | None = None) -> list[_Pair]:
        assert ids == ["older", "newer"]
        if status == "evolved":
            return [pair]
        if status == "open":
            return [pair]
        if status in {None, "competing"}:
            return []
        raise AssertionError(f"unexpected contradiction status: {status}")

    harness.contradict_store.pairs_for_ids.side_effect = pairs
    original = [older, newer]

    assert harness._apply_contradict_penalty(original) is original
    assert harness.contradict_store.pairs_for_ids.call_args_list == [
        call(["older", "newer"]),
        call(["older", "newer"], status="evolved"),
        call(["older", "newer"], status="competing"),
        call(["older", "newer"], status="open"),
    ]


def test_contradict_penalty_honors_registered_boundaries_and_keeps_strongest_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "memo.memory.search_scoring_ops.flag_float",
        lambda name: 1.0 if name == "MEMO_CONTRADICT_PENALTY" else 0.0,
    )
    old = replace(_record("old", 0.3), updated="2025-01-01T00:00:00+00:00")
    new = replace(_record("new", 0.2), updated="2026-01-01T00:00:00+00:00")
    newest = replace(_record("newest", 0.1), updated="2027-01-01T00:00:00+00:00")
    harness = _Harness()
    harness.contradict_store = MagicMock()
    harness.contradict_store.pairs_for_ids.side_effect = [
        [
            _Pair("old", "new", "contradiction"),
            _Pair("old", "newest", "contradiction"),
        ],
        [_Pair("new", "newest", "evolution")],
    ]

    out = harness._apply_contradict_penalty([old, new, newest])

    assert {hit.id: hit.score for hit in out} == {
        "old": pytest.approx(0.3),
        "new": pytest.approx(0.0),
        "newest": pytest.approx(0.1),
    }
    assert [hit.id for hit in out] == ["old", "newest", "new"]


def test_contradict_penalty_skips_missing_or_unparseable_timestamps() -> None:
    a = replace(_record("a", 0.3), updated="")
    b = replace(_record("b", 0.2), updated="2026-01-01T00:00:00+00:00")
    harness = _Harness()
    harness.contradict_store = MagicMock()
    harness.contradict_store.pairs_for_ids.side_effect = [
        [_Pair("a", "b", None), _Pair("missing", "b")],
        [_Pair("a", "b", "evolution")],
    ]

    original = [a, b]
    assert harness._apply_contradict_penalty(original) is original


def test_entity_boost_forwards_exact_entities_and_preserves_zero_boost_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extract = MagicMock(return_value=["Memo", "Qwen"])
    match = MagicMock(return_value=0.0)
    monkeypatch.setattr("memo.entity_extractor.extract_entities", extract)
    monkeypatch.setattr("memo.entity_extractor.entity_match_score", match)
    record = _record("zero", None, extra={"entities": ["Memo"]})
    original = [record]

    out = _Harness()._apply_entity_boost("exact query", original)

    extract.assert_called_once_with("exact query")
    match.assert_called_once_with(["Memo", "Qwen"], ["Memo"])
    assert out == original
    assert out[0] is record


def test_entity_boost_rounds_none_scores_to_six_places(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("memo.entity_extractor.extract_entities", lambda query: [query])
    monkeypatch.setattr(
        "memo.entity_extractor.entity_match_score",
        lambda _query, entities: 0.12345678 if entities else 0.0,
    )
    boosted = _record("boosted", None, extra={"entities": ["x"]})
    plain = _record("plain", 0.12, extra={"entities": []})

    out = _Harness()._apply_entity_boost("x", [plain, boosted])

    assert [hit.id for hit in out] == ["boosted", "plain"]
    assert out[0].score == 0.123457


def test_co_recall_contract_covers_empty_zero_missing_and_default_weight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _Harness()
    harness.graph = MagicMock()
    hits = [_record("anchor", 0.2), _record("one", None), _record("missing", 0.05)]
    harness.graph.co_recall_counts.return_value = {"one": 1, "zero": 0}

    out = harness._apply_co_recall_boost(hits)

    harness.graph.co_recall_counts.assert_called_once_with("anchor", ["one", "missing"])
    assert [hit.id for hit in out] == ["anchor", "one", "missing"]
    assert {hit.id: hit.score for hit in out} == {
        "anchor": pytest.approx(0.2),
        "one": pytest.approx(0.1),
        "missing": pytest.approx(0.05),
    }

    harness.graph.co_recall_counts.reset_mock()
    harness.graph.co_recall_counts.return_value = {}
    log_exception = MagicMock()
    monkeypatch.setattr("memo.memory.search_scoring_ops._log.exception", log_exception)
    original = list(hits)
    assert harness._apply_co_recall_boost(original) is original
    log_exception.assert_not_called()

    harness.graph.co_recall_counts.return_value = {"zero": 0}
    assert harness._apply_co_recall_boost(original) is original
    log_exception.assert_not_called()


def test_co_recall_failure_has_stable_diagnostic(caplog: pytest.LogCaptureFixture) -> None:
    harness = _Harness()
    harness.graph = MagicMock()
    harness.graph.co_recall_counts.side_effect = RuntimeError("graph unavailable")
    caplog.set_level("ERROR")

    hits = [_record("a"), _record("b"), _record("c")]
    assert harness._apply_co_recall_boost(hits) is hits

    assert "co-recall boost failed, returning unboosted results" in caplog.messages


def test_retrieval_boost_forwards_all_fields_and_obeys_confidence_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict] = []

    def boost_for(**kwargs) -> float:
        calls.append(kwargs)
        return 1.23456789

    monkeypatch.setattr("memo.retrieval_boost.boost_for", boost_for)
    harness = _Harness()
    harness.store = MagicMock()
    harness.store.get_health_batch.return_value = {
        "trusted": {"confidence": 0.9},
        "untrusted": {"confidence": 0.899999},
    }
    trusted = replace(
        _record("trusted", None, extra={"chunk_heading": ""}),
        path="",
        title="",
        tags=[],
    )
    untrusted = _record("untrusted", 0.9, extra={"chunk_heading": "ignore"})

    out = harness._apply_retrieval_boost("query", [trusted, untrusted])

    harness.store.get_health_batch.assert_called_once_with(["trusted", "untrusted"])
    assert calls == [
        {
            "query": "query",
            "filename": "",
            "title": "",
            "headings": None,
            "tags": None,
        }
    ]
    assert out[1].id == "trusted"
    assert out[1].score == 0.0
    assert out[0] is untrusted


def test_health_scores_exact_ids_neutral_boundary_none_score_and_rounding() -> None:
    harness = _Harness()
    harness.store = MagicMock()
    harness.store.get_health_batch.return_value = {
        "neutral": {"confidence": 1.0, "roi_score": 1.0},
        "changed": {"confidence": 1.23456789, "roi_score": 1.0},
    }
    neutral = _record("neutral", 0.3)
    changed = _record("changed", 0.12345678)

    out = harness._apply_health_scores([neutral, changed])

    harness.store.get_health_batch.assert_called_once_with(["neutral", "changed"])
    assert out[0] is neutral
    assert out[1].score == 0.152416


def test_record_access_forwards_roi_ids_and_logs_sqlite_failure(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("MEMO_HEALTH_SCORES_DISABLED", "0")
    monkeypatch.setenv("MEMO_OUTCOME_RANKING_ENABLED", "0")
    harness = _Harness()
    harness.store = MagicMock()
    ids = ["a", "b"]
    harness._record_access(ids)
    harness.store.boost_roi_batch.assert_called_once_with(ids)

    harness.store.touch.side_effect = sqlite3.OperationalError("database is locked")
    caplog.set_level("DEBUG")
    harness._record_access(["locked"])
    assert "access tracking skipped: database is locked" in caplog.messages


def test_cache_read_through_exact_limit_and_fetch_contract() -> None:
    harness = _Harness()
    backend = MagicMock()
    harness._cache_backend = lambda: backend  # type: ignore[method-assign]
    one = _record("one")

    assert harness._cache_read_through("q", [one], limit=1) == [one]
    assert harness._cache_read_through("q", [one], limit=0) == []
    backend.fetch.assert_not_called()

    backend.fetch.return_value = []
    existing: list[MemoryRecord] = []
    assert harness._cache_read_through("exact query", existing, limit=3) is existing
    backend.fetch.assert_called_once_with("exact query", limit=3)


@pytest.mark.parametrize(
    ("candidate", "expected"),
    [
        (
            {
                "id": "remote",
                "body": "body",
                "title": "Title",
                "type": "decision",
                "tags": ["one"],
                "score": 0.7654321,
            },
            {
                "content": "body",
                "title": "Title",
                "type_": "decision",
                "tags": ["one"],
                "extra": {
                    "source": "memo-cache-fill",
                    "cache_backend_id": "remote",
                },
                "auto_derive": False,
            },
        ),
        (
            {"body": "body"},
            {
                "content": "body",
                "title": "",
                "type_": "note",
                "tags": [],
                "extra": {"source": "memo-cache-fill"},
                "auto_derive": False,
            },
        ),
    ],
)
def test_materialize_cache_candidate_persists_exact_contract(
    candidate: dict,
    expected: dict,
) -> None:
    harness = _Harness()
    saved = _record("saved", None)
    harness.save = MagicMock(return_value=saved)  # type: ignore[method-assign]
    have: set[str] = set()
    have_backend_ids: set[str] = set()

    existing, materialized = harness._materialize_cache_candidate(
        candidate,
        existing=[],
        have=have,
        have_backend_ids=have_backend_ids,
    )

    assert existing == []
    harness.save.assert_called_once_with(**expected)
    assert materialized == replace(saved, score=candidate.get("score"))
    assert have == {"saved"}
    assert have_backend_ids == ({"remote"} if candidate.get("id") else set())


def test_materialize_cache_candidate_rejects_blank_and_logs_failure(
    caplog: pytest.LogCaptureFixture,
) -> None:
    harness = _Harness()
    harness.save = MagicMock(side_effect=OSError("disk full"))  # type: ignore[method-assign]
    existing = [_record("existing")]
    have = {"existing"}
    backend_ids: set[str] = set()

    assert harness._materialize_cache_candidate(
        {"id": "", "body": "  "},
        existing=existing,
        have=have,
        have_backend_ids=backend_ids,
    ) == (existing, None)
    harness.save.assert_not_called()

    caplog.set_level("DEBUG")
    assert harness._materialize_cache_candidate(
        {"id": "remote", "body": "body"},
        existing=existing,
        have=have,
        have_backend_ids=backend_ids,
    ) == (existing, None)
    assert "cache read-through materialize failed: disk full" in caplog.messages
    assert have == {"existing"}
    assert backend_ids == set()


def test_entity_boost_neutral_path_preserves_order_and_positive_path_sorts_none_last(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("memo.entity_extractor.extract_entities", lambda _query: ["entity"])
    monkeypatch.setattr("memo.entity_extractor.entity_match_score", lambda *_args: 0.0)
    low = _record("low", 0.1)
    high = _record("high", 0.5)
    neutral = _Harness()._apply_entity_boost("query", [low, high])
    assert [hit.id for hit in neutral] == ["low", "high"]
    assert neutral[0] is low
    assert neutral[1] is high

    monkeypatch.setattr(
        "memo.entity_extractor.entity_match_score",
        lambda _query, entities: 0.12345678 if entities else 0.0,
    )
    none_score = _record("none", None, extra={"entities": []})
    boosted = _record("boosted", 0.1, extra={"entities": ["entity"]})
    changed = _Harness()._apply_entity_boost("query", [none_score, boosted])
    assert [hit.id for hit in changed] == ["boosted", "none"]
    assert changed[0].score == 0.223457


def test_co_recall_zero_edges_are_identity_and_rounding_is_six_places(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _Harness()
    harness.graph = MagicMock()
    low = _record("anchor", 0.1)
    none_score = _record("none", None)
    high = _record("high", 0.5)
    harness.graph.co_recall_counts.return_value = {}
    original = [low, none_score, high]
    assert harness._apply_co_recall_boost(original) is original
    assert [hit.id for hit in original] == ["anchor", "none", "high"]

    monkeypatch.setenv("MEMO_CO_RECALL_BOOST_WEIGHT", "0.1")
    harness.graph.co_recall_counts.return_value = {"boosted": 1, "zero": 0}
    boosted = _record("boosted", 0.12345678)
    zero = _record("zero", None)
    out = harness._apply_co_recall_boost([low, boosted, zero])
    assert [hit.id for hit in out] == ["boosted", "anchor", "zero"]
    assert out[0].score == 0.223457
    assert out[2] is zero


def test_retrieval_boost_neutral_path_and_nonempty_metadata_are_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict] = []

    def neutral_boost(**kwargs) -> float:
        calls.append(kwargs)
        return 1.0

    monkeypatch.setattr("memo.retrieval_boost.boost_for", neutral_boost)
    harness = _Harness()
    harness.store = MagicMock()
    harness.store.get_health_batch.return_value = {}
    low = replace(_record("low", 0.1, extra={}), path="low.md", title="Low")
    high = replace(
        _record("high", 0.5, extra={"chunk_heading": "Heading"}),
        path="high.md",
        title="High",
    )

    out = harness._apply_retrieval_boost("query", [low, high])

    assert [hit.id for hit in out] == ["low", "high"]
    assert out[0] is low
    assert out[1] is high
    assert calls == [
        {
            "query": "query",
            "filename": "low.md",
            "title": "Low",
            "headings": None,
            "tags": ["test"],
        },
        {
            "query": "query",
            "filename": "high.md",
            "title": "High",
            "headings": ["Heading"],
            "tags": ["test"],
        },
    ]


def test_retrieval_boost_rounds_and_sorts_untrusted_none_score_last(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("memo.retrieval_boost.boost_for", lambda **_kwargs: 1.23456789)
    harness = _Harness()
    harness.store = MagicMock()
    harness.store.get_health_batch.return_value = {
        "untrusted": {"confidence": 0.1},
    }
    trusted = _record("trusted", 0.12345678)
    untrusted = _record("untrusted", None)

    out = harness._apply_retrieval_boost("query", [untrusted, trusted])

    assert [hit.id for hit in out] == ["trusted", "untrusted"]
    assert out[0].score == 0.152416
    assert out[1] is untrusted


def test_health_scores_neutral_path_preserves_order_and_none_score_sorts_last() -> None:
    harness = _Harness()
    harness.store = MagicMock()
    low = _record("low", 0.1)
    high = _record("high", 0.5)
    harness.store.get_health_batch.return_value = {
        "low": {"confidence": 1.0, "roi_score": 1.0},
        "high": {"confidence": 1.0, "roi_score": 1.0},
    }
    neutral = harness._apply_health_scores([low, high])
    assert [hit.id for hit in neutral] == ["low", "high"]
    assert neutral[0] is low
    assert neutral[1] is high

    none_score = _record("none", None)
    changed = _record("changed", 0.12345678)
    harness.store.get_health_batch.return_value = {
        "none": {"confidence": 0.5, "roi_score": 1.0},
        "changed": {"confidence": 1.23456789, "roi_score": 1.0},
    }
    out = harness._apply_health_scores([none_score, changed])
    assert [hit.id for hit in out] == ["changed", "none"]
    assert out[0].score == 0.152416
    assert out[1].score == 0.0


def test_materialize_cache_candidate_missing_body_is_rejected() -> None:
    harness = _Harness()
    harness.save = MagicMock()  # type: ignore[method-assign]
    existing = [_record("existing")]

    assert harness._materialize_cache_candidate(
        {"id": "remote"},
        existing=existing,
        have={"existing"},
        have_backend_ids=set(),
    ) == (existing, None)
    harness.save.assert_not_called()


def test_materialize_collision_records_backend_identity_in_caller_set() -> None:
    harness = _Harness()
    saved = _record("existing", None, extra={"origin": "local"})
    persisted = _record(
        "existing",
        None,
        extra={"origin": "local", "cache_backend_id": "remote"},
    )
    harness.save = MagicMock(return_value=saved)  # type: ignore[method-assign]
    harness.update = MagicMock(return_value=persisted)  # type: ignore[method-assign]
    existing = [_record("existing", 0.7, extra={"origin": "local"})]
    backend_ids: set[str] = set()

    merged, materialized = harness._materialize_cache_candidate(
        {"id": "remote", "body": "body"},
        existing=existing,
        have={"existing"},
        have_backend_ids=backend_ids,
    )

    assert materialized is None
    assert merged[0].extra == {"origin": "local", "cache_backend_id": "remote"}
    assert merged[0].score == 0.7
    assert backend_ids == {"remote"}


def test_declared_disputes_require_both_present_sides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MEMO_DECLARE_DISPUTES", "1")
    older = replace(_record("a", 0.5), updated="2025-01-01T00:00:00+00:00")
    newer = replace(_record("b", 0.4), updated="2026-01-01T00:00:00+00:00")
    harness = _Harness()
    harness.contradict_store = MagicMock()

    def pairs(ids: list[str], status: str | None = None) -> list[_Pair]:
        assert ids == ["a", "b"]
        if status is None:
            return [_Pair("a", "b")]
        if status == "evolved":
            return []
        if status == "competing":
            return [
                _Pair("a", "missing"),
                _Pair("missing", "b"),
            ]
        assert status == "open"
        return []

    harness.contradict_store.pairs_for_ids.side_effect = pairs

    out = harness._apply_contradict_penalty([older, newer])

    assert {hit.id: hit.score for hit in out} == {
        "a": pytest.approx(0.2),
        "b": pytest.approx(0.4),
    }


@pytest.mark.parametrize("older_id", ["a", "b"])
def test_declared_dispute_preserves_each_possible_older_side(
    monkeypatch: pytest.MonkeyPatch,
    older_id: str,
) -> None:
    monkeypatch.setenv("MEMO_DECLARE_DISPUTES", "1")
    a_updated = "2025-01-01T00:00:00+00:00" if older_id == "a" else "2026-01-01T00:00:00+00:00"
    b_updated = "2025-01-01T00:00:00+00:00" if older_id == "b" else "2026-01-01T00:00:00+00:00"
    a = replace(_record("a", 0.5), updated=a_updated)
    b = replace(_record("b", 0.4), updated=b_updated)
    harness = _Harness()
    harness.contradict_store = MagicMock()

    def pairs(_ids: list[str], status: str | None = None) -> list[_Pair]:
        if status is None:
            return [_Pair("a", "b")]
        if status == "competing":
            return [_Pair("a", "b")]
        return []

    harness.contradict_store.pairs_for_ids.side_effect = pairs
    original = [a, b]

    assert harness._apply_contradict_penalty(original) is original


def test_quality_rerank_gate_success_and_failure(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    harness = _Harness()
    original = [_record("original")]
    monkeypatch.setenv("MEMO_QUALITY_RERANK", "0")
    assert harness._apply_quality_rerank(original) is original

    monkeypatch.setenv("MEMO_QUALITY_RERANK", "1")
    reranked = [_record("reranked")]
    apply = MagicMock(return_value=reranked)
    monkeypatch.setattr("memo.quality.apply_quality_rerank", apply)
    assert harness._apply_quality_rerank(original) is reranked
    apply.assert_called_once_with(original)

    apply.side_effect = RuntimeError("quality unavailable")
    caplog.set_level("DEBUG")
    assert harness._apply_quality_rerank(original) is original
    assert "quality_rerank failed: quality unavailable" in caplog.messages


def test_verification_decay_neutral_rounding_sort_and_failure(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    factors = {"neutral": 1.0, "changed": 1.23456789, "none": 0.5}
    monkeypatch.setattr(
        "memo.memory.search_scoring_ops._state_decay_factor",
        lambda record: factors[record.id],
    )
    neutral = _record("neutral", 0.1)
    changed = _record("changed", 0.12345678)
    none_score = _record("none", None)

    out = _Harness()._apply_verification_decay([neutral, none_score, changed])

    assert [hit.id for hit in out] == ["changed", "neutral", "none"]
    assert out[0].score == 0.152416
    assert out[1] is neutral
    assert out[2].score == 0.0

    monkeypatch.setattr(
        "memo.memory.search_scoring_ops._state_decay_factor",
        MagicMock(side_effect=RuntimeError("state unavailable")),
    )
    caplog.set_level("DEBUG")
    original = [neutral]
    assert _Harness()._apply_verification_decay(original) is original
    assert "verification_decay failed: state unavailable" in caplog.messages
