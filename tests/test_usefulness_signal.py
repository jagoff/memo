"""Honest usefulness metrics over the consult/usage logs.

Pins three additions that turn "memo fired" into "memo was useful":
  - dedup_double_fire: the same prompt logged twice (subprocess+daemon) is one
    consult, not two — totals/rates must not double-count it.
  - strong_hit_rate: share of fired recalls with a high-confidence (>0.85) top
    hit — the honest relevance number, separate from "returned anything".
  - referenced_rate: share of surfaced memorias later fetched (usage.log) — a
    lower bound on "used", not just "shown".
"""

from __future__ import annotations

from memo.dashboard import (
    append_recall_log,
    append_usage_log,
    consult_breakdown,
    dedup_double_fire,
    recall_health,
    referenced_rate,
    usage_log_path,
)


def test_dedup_double_fire_collapses_subprocess_daemon_pair() -> None:
    # read_recall_log returns newest-first; a double-fire pair is the same
    # prompt seconds apart via two paths.
    rows = [
        {"ts": "2026-05-30T06:11:10+00:00", "prompt": "pushea primero", "via": "daemon",
         "hits": [{"id": "8d34571e", "score": 0.97}]},
        {"ts": "2026-05-30T06:11:07+00:00", "prompt": "pushea primero", "via": "subprocess",
         "hits": [{"id": "976266c7", "score": 0.95}]},
        {"ts": "2026-05-30T06:10:01+00:00", "prompt": "que queda pendiente", "via": "daemon",
         "hits": [{"id": "4ac31e77", "score": 1.0}]},
    ]
    out = dedup_double_fire(rows, window_s=15.0)
    # The pair collapses to one (the higher-quality daemon row), the distinct
    # prompt survives.
    assert len(out) == 2
    pushea = [r for r in out if r["prompt"] == "pushea primero"]
    assert len(pushea) == 1
    assert pushea[0]["hits"][0]["id"] == "8d34571e"  # higher top score kept


def test_dedup_keeps_same_prompt_outside_window() -> None:
    rows = [
        {"ts": "2026-05-30T07:00:00+00:00", "prompt": "seguimos", "via": "daemon", "hits": []},
        {"ts": "2026-05-30T06:00:00+00:00", "prompt": "seguimos", "via": "daemon", "hits": []},
    ]
    # An hour apart → genuinely two consults, not a double-fire.
    assert len(dedup_double_fire(rows, window_s=15.0)) == 2


def test_dedup_never_merges_empty_prompt_bails() -> None:
    rows = [
        {"ts": "2026-05-30T07:00:01+00:00", "prompt": "", "via": "bail"},
        {"ts": "2026-05-30T07:00:00+00:00", "prompt": "", "via": "bail"},
    ]
    assert len(dedup_double_fire(rows)) == 2


def test_strong_hit_rate_splits_confident_from_weak(tmp_path) -> None:
    # Two strong (>0.85), one weak, one bail. Distinct prompts so dedup leaves
    # them all.
    append_recall_log(tmp_path, prompt="strong one alpha", hits=[{"id": "a1", "score": 0.91, "title": "A"}], via="daemon")
    append_recall_log(tmp_path, prompt="strong two beta", hits=[{"id": "b2", "score": 0.88, "title": "B"}], via="daemon")
    append_recall_log(tmp_path, prompt="weak three gamma", hits=[{"id": "c3", "score": 0.62, "title": "C"}], via="daemon")
    append_recall_log(tmp_path, prompt="", hits=[], via="bail", reason="slash")

    health = recall_health(tmp_path, limit=500)
    assert health["fired"] == 3
    assert health["bailed"] == 1
    assert health["hit_rate"] == 1.0          # all fired returned something
    assert health["strong_hit_rate"] == round(2 / 3, 3)  # only 2 were confident


def test_referenced_rate_counts_only_post_surfacing_fetches(tmp_path) -> None:
    # Surface two memorias; fetch one of them AFTER it was surfaced.
    append_recall_log(
        tmp_path, prompt="what did we decide about caching",
        hits=[{"id": "dec00001", "score": 0.9, "title": "Cache decision"},
              {"id": "fact0001", "score": 0.87, "title": "A fact"}],
        via="daemon",
    )
    append_usage_log(tmp_path, "dec00001")  # acted on the surfaced decision

    ref = referenced_rate(tmp_path, [
        {"ts": "2026-05-30T06:00:00+00:00",
         "hits": [{"id": "dec00001"}, {"id": "fact0001"}]},
    ])
    # usage.log ids are 8-char prefixes, same as recall.log hit ids.
    assert ref["surfaced"] == 2
    assert ref["referenced"] == 1
    assert ref["referenced_rate"] == 0.5


def test_referenced_rate_none_when_nothing_surfaced(tmp_path) -> None:
    ref = referenced_rate(tmp_path, [{"ts": "2026-05-30T06:00:00+00:00", "hits": []}])
    assert ref["referenced_rate"] is None
    assert ref["surfaced"] == 0


def test_usage_log_round_trip_truncates_id(tmp_path) -> None:
    append_usage_log(tmp_path, "abcdefgh-1234-long-id")
    assert usage_log_path(tmp_path).is_file()
    from memo.dashboard import read_usage_log
    rows = read_usage_log(tmp_path)
    assert rows[-1]["id"] == "abcdefgh"  # 8-char prefix, matches recall.log


def test_consult_breakdown_exposes_strong_hit_rate(tmp_path) -> None:
    append_recall_log(tmp_path, prompt="alpha distinct prompt", hits=[{"id": "a1", "score": 0.9, "title": "A"}], via="daemon")
    append_recall_log(tmp_path, prompt="beta distinct prompt", hits=[{"id": "b2", "score": 0.5, "title": "B"}], via="subprocess")
    bd = consult_breakdown(tmp_path, limit=500)
    cc = next(c for c in bd["consumers"] if c["consumer"] == "claude-code")
    assert cc["fired"] == 2
    assert cc["hit_rate"] == 1.0
    assert cc["strong_hit_rate"] == 0.5
