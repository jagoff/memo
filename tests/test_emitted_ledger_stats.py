"""Tests for `emitted_ledger.bump` / `emitted_ledger.stats` -- the counters
`memo_cache_stats` surfaces under its `emit_ledger` key.

Unlike the plan brief's own snippet, `bump`/`stats` accumulate TOKENS, not
chars: every call site that has real text (`apply_ledger`'s digest hits, the
extra-payload JSON, memo_get's recovered body) converts it with
`mcp_budget.est_tokens` before handing a delta to `bump`. That keeps this
module's stdlib-only leaf-module contract (see its module docstring) intact
-- it never re-derives the 4-chars-per-token estimate itself, callers do.
"""

from __future__ import annotations

from memo import emitted_ledger as el


def test_stats_on_a_cold_session_are_zero(tmp_path):
    s = el.stats(tmp_path, "cold")
    assert s == {
        "entries": 0,
        "digests_served": 0,
        "tokens_suppressed": 0,
        "tokens_digest": 0,
        "memo_get_after_digest": 0,
        "net_saved_est": 0,
    }


def test_suppression_moves_the_counters(tmp_path):
    body = "x" * 400
    el.append(
        tmp_path,
        "s",
        [el.Entry(id="a", h=el.emitted_hash(body), n=len(body), ref="memo-r/a", t=1, src="mcp")],
    )
    el.bump(tmp_path, "s", digests_served=1, tokens_suppressed=100, tokens_digest=15)
    s = el.stats(tmp_path, "s")
    assert s["entries"] == 1
    assert s["digests_served"] == 1
    assert s["tokens_suppressed"] == 100
    assert s["tokens_digest"] == 15
    assert s["net_saved_est"] == 85


def test_memo_get_after_digest_reduces_net(tmp_path):
    el.bump(tmp_path, "s", digests_served=1, tokens_suppressed=100, tokens_digest=15)
    el.bump(tmp_path, "s", get_after_digest=1, tokens_recovered=115)
    s = el.stats(tmp_path, "s")
    assert s["memo_get_after_digest"] == 1
    # 85 was "saved" by the digest; recovering it back out via memo_get costs
    # 115 tokens -- more than the saving, so the net goes negative. That is
    # the whole point of subtracting recovery cost: a session that re-fetches
    # everything it was handed a pointer for must show the feature LOSING.
    assert s["net_saved_est"] == 85 - 115
    assert s["net_saved_est"] < 0


def test_bump_accumulates_across_multiple_calls(tmp_path):
    el.bump(tmp_path, "s", digests_served=1, tokens_suppressed=10)
    el.bump(tmp_path, "s", digests_served=2, tokens_suppressed=20)
    s = el.stats(tmp_path, "s")
    assert s["digests_served"] == 3
    assert s["tokens_suppressed"] == 30


def test_stats_is_per_session(tmp_path):
    el.bump(tmp_path, "s1", digests_served=1, tokens_suppressed=10)
    el.bump(tmp_path, "s2", digests_served=5, tokens_suppressed=50)
    assert el.stats(tmp_path, "s1")["tokens_suppressed"] == 10
    assert el.stats(tmp_path, "s2")["tokens_suppressed"] == 50


def test_bump_is_fail_open_on_unwritable_dir(tmp_path):
    ro = tmp_path / "ro"
    ro.mkdir()
    ro.chmod(0o500)
    try:
        el.bump(ro, "s", digests_served=1)  # must not raise
        assert el.stats(ro, "s")["digests_served"] == 0
    finally:
        ro.chmod(0o700)


def test_stats_is_fail_open_on_a_corrupt_counters_file(tmp_path):
    path = tmp_path / "emitted" / "s.counters.json"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"\xff\xfe not valid json")
    s = el.stats(tmp_path, "s")
    assert s["tokens_suppressed"] == 0
    assert s["net_saved_est"] == 0


def test_bump_with_no_deltas_is_a_harmless_noop(tmp_path):
    el.bump(tmp_path, "s")
    assert el.stats(tmp_path, "s")["digests_served"] == 0
