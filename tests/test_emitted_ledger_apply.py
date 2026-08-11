import json

import pytest

from memo import emitted_ledger as el
from memo import server_common as sc
from memo import server_session_patterns as ssp
from memo.mcp_budget import est_tokens


class _Cfg:
    def __init__(self, state_dir):
        self.state_dir = state_dir


class _Mem:
    def __init__(self, state_dir):
        self.cfg = _Cfg(state_dir)


def _hits():
    return [
        {"id": "mem_a", "title": "A", "body": "body a"},
        {"id": "mem_b", "title": "B", "body": "body b"},
    ]


@pytest.fixture
def mem(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMO_EMITTED_LEDGER", "1")
    monkeypatch.setenv("MEMO_SESSION_ID", "sess-apply")
    return _Mem(tmp_path)


def test_flag_off_is_a_passthrough(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMO_EMITTED_LEDGER", "0")
    monkeypatch.setenv("MEMO_SESSION_ID", "sess-off")
    hits = _hits()
    out, extra = sc.apply_ledger(_Mem(tmp_path), "memo_search", hits)
    assert out == hits and extra == {}
    assert el.read(tmp_path, "sess-off") == {}


def test_tool_not_in_allowlist_is_a_passthrough(mem, tmp_path):
    hits = _hits()
    out, extra = sc.apply_ledger(mem, "memo_get", hits)
    assert out == hits and extra == {}
    assert el.read(tmp_path, "sess-apply") == {}


def test_allowlist_follows_the_flag_value(tmp_path, monkeypatch):
    """F6.5: replacing the flag_str-driven allowlist with a hardcoded set of
    the default five tool names would pass every other test unchanged. Pin
    the wiring directly: point MEMO_EMITTED_LEDGER_TOOLS at a tool NOT in the
    default list, and confirm the default-list membership flips accordingly
    in both directions."""
    monkeypatch.setenv("MEMO_EMITTED_LEDGER", "1")
    monkeypatch.setenv("MEMO_SESSION_ID", "sess-allow")
    monkeypatch.setenv("MEMO_EMITTED_LEDGER_TOOLS", "custom_tool")
    mem_ = _Mem(tmp_path)
    hits = _hits()

    # memo_search is in the DEFAULT allowlist but not in this custom one.
    out, extra = sc.apply_ledger(mem_, "memo_search", hits)
    assert out == hits and extra == {}
    assert el.read(tmp_path, "sess-allow") == {}

    # custom_tool is NOT in the default allowlist but is in this custom one.
    out2, _ = sc.apply_ledger(mem_, "custom_tool", hits)
    assert [h["id"] for h in out2] == ["mem_a", "mem_b"]
    assert set(el.read(tmp_path, "sess-allow")) == {"mem_a", "mem_b"}


def test_first_call_emits_all_and_records(mem, tmp_path):
    out, extra = sc.apply_ledger(mem, "memo_search", _hits())
    assert [h["id"] for h in out] == ["mem_a", "mem_b"]
    assert extra == {}  # nothing suppressed on a cold ledger
    assert set(el.read(tmp_path, "sess-apply")) == {"mem_a", "mem_b"}


def test_second_identical_call_digests(mem, tmp_path):
    sc.apply_ledger(mem, "memo_search", _hits())
    # F4: source the expected ref from the LEDGER (what the first call
    # actually recorded), not from the first call's return (which is `{}` on
    # a cold ledger — there is no first-call cache_ref to compare against).
    recorded_ref_a = el.read(tmp_path, "sess-apply")["mem_a"].ref

    out, extra = sc.apply_ledger(mem, "memo_search", _hits())
    assert out == []
    assert [e["id"] for e in extra["already_in_context"]] == ["mem_a", "mem_b"]
    assert extra["already_in_context"][0]["title"] == "A"
    # F4: pin the ACTUAL value, not just the prefix — a mutation that swaps
    # in this call's freshly minted ref (which would name an empty batch,
    # since nothing was sent full this call) must fail this assertion.
    assert extra["already_in_context"][0]["ref"] == recorded_ref_a
    # F6.2: everything was digested, nothing sent full -> no batch for
    # cache_ref to name. It must be absent, not a token pointing at nothing.
    assert "cache_ref" not in extra
    assert "memo_get" in extra["hint"]


def test_partial_overlap_across_tools(mem, tmp_path):
    sc.apply_ledger(mem, "memo_search", _hits())
    recorded_ref_a = el.read(tmp_path, "sess-apply")["mem_a"].ref

    later = [*_hits(), {"id": "mem_c", "title": "C", "body": "body c"}]
    out, extra = sc.apply_ledger(mem, "memo_ask", later)
    assert [h["id"] for h in out] == ["mem_c"]
    assert [e["id"] for e in extra["already_in_context"]] == ["mem_a", "mem_b"]
    # F4: the digest ref must name the EARLIER call that actually emitted
    # mem_a/mem_b, not this call's batch (which only contains mem_c).
    assert extra["already_in_context"][0]["ref"] == recorded_ref_a
    assert extra["cache_ref"] != recorded_ref_a


def test_changed_body_is_reemitted(mem):
    sc.apply_ledger(mem, "memo_search", _hits())
    edited = [{"id": "mem_a", "title": "A", "body": "body a, edited"}]
    out, extra = sc.apply_ledger(mem, "memo_search", edited)
    assert [h["id"] for h in out] == ["mem_a"]
    assert extra == {}


def test_truncated_body_over_prefix_threshold_is_digested_via_hp(mem):
    """F5: pins Entry.for_text actually being used to build ledger entries.
    A manual Entry(...) construction that drops `hp` would leave the length
    arm of partition() permanently dead (hp is None -> never digest-by-length),
    so this body — over el._PREFIX_CHARS, truncated but prefix-preserving on
    the second call — must be digested. This is the first test anywhere that
    exercises that arm through the real apply_ledger path (test_emitted_ledger_
    partition.py exercises it directly against `partition()`, not through here)."""
    assert el._PREFIX_CHARS < 220
    long_body = "x" * 500
    sc.apply_ledger(mem, "memo_search", [{"id": "mem_a", "title": "A", "body": long_body}])
    truncated = long_body[:220]

    out, extra = sc.apply_ledger(
        mem, "memo_search", [{"id": "mem_a", "title": "A", "body": truncated}]
    )
    assert out == []
    assert [e["id"] for e in extra["already_in_context"]] == ["mem_a"]


def test_bodyless_hit_is_always_full_and_never_recorded(mem, tmp_path):
    """F2: a hit with no body key at all must never be digested — an empty
    string hashes to a fixed value that self-matches itself regardless of
    what the real content later becomes (context_surface.py's memo_context
    rows carry no body at all — id/title/score/section only — which is
    exactly why Task 5 left memo_context out of MEMO_EMITTED_LEDGER_TOOLS's
    default rather than wiring it against a bodyless row)."""
    hit = {"id": "mem_a", "title": "A"}
    sc.apply_ledger(mem, "memo_search", [hit])
    assert el.read(tmp_path, "sess-apply") == {}  # nothing recorded for it

    out, extra = sc.apply_ledger(mem, "memo_search", [hit])
    assert out == [hit]
    assert extra == {}
    assert el.read(tmp_path, "sess-apply") == {}


def test_idless_hit_is_always_full_and_never_recorded(mem, tmp_path):
    """F2: a hit with no id key at all must never be digested or recorded —
    otherwise every id-less hit collapses onto one shared ledger key `""`,
    letting an unrelated memory get digested against it and handed a ref that
    names something else entirely."""
    hit = {"title": "A", "body": "some body"}
    sc.apply_ledger(mem, "memo_search", [hit])
    assert el.read(tmp_path, "sess-apply") == {}

    out, extra = sc.apply_ledger(mem, "memo_search", [hit])
    assert out == [hit]
    assert extra == {}
    assert el.read(tmp_path, "sess-apply") == {}


def test_idless_hit_does_not_poison_or_borrow_another_ids_entry(mem):
    """F2's second reproduction: two DIFFERENT id-less hits (distinct real
    memories whose rows carry no id) collapse onto the shared "" ledger key
    when id_of defaults to "" on a missing key. Using IDENTICAL emitted text
    for both — e.g. two rows sharing the same boilerplate — is what actually
    exercises the hazard: without the guard, the second hit's text hashes
    equal to the first's recorded entry and gets digested against it, handed
    a ref that in reality names an unrelated memory. (Two hits with
    DIFFERENT text would fail the hash/length checks anyway regardless of
    the id collision, so that alone would not catch a regression here — the
    text must coincide for this to be a meaningful reproduction.) With both
    kept out of partition()'s view entirely, each id-less hit is always sent
    full no matter what its text is."""
    shared_body = "identical boilerplate emitted by two distinct memories"
    first = {"title": "A", "body": shared_body}
    sc.apply_ledger(mem, "memo_search", [first])

    second = {"title": "B", "body": shared_body}  # a different, unrelated memory
    out, extra = sc.apply_ledger(mem, "memo_search", [second])
    assert out == [second]
    assert extra == {}


def test_custom_accessors_drive_the_digest_decision(mem):
    """F3: a call site whose rows put text under a different key (e.g.
    context_pack.py's `snippet`) must be able to make the digest decision
    follow that key instead of the hardcoded `body` default."""

    def text_of(hit):
        return str(hit.get("snippet") or "")

    hits = [{"id": "mem_a", "title": "A", "snippet": "snippet body"}]
    sc.apply_ledger(mem, "memo_search", hits, text_of=text_of)
    out, extra = sc.apply_ledger(mem, "memo_search", hits, text_of=text_of)
    assert out == []
    assert [e["id"] for e in extra["already_in_context"]] == ["mem_a"]

    # Without the custom accessor, the default `body` reader sees nothing
    # under these rows, so the F2 guard keeps it full forever instead.
    out_default, extra_default = sc.apply_ledger(mem, "memo_search", hits)
    assert out_default == hits
    assert extra_default == {}


def test_unwritable_state_dir_is_a_passthrough_via_leaf_module_fail_open(tmp_path, monkeypatch):
    """F6.3, renamed from test_unwritable_state_dir_degrades_to_passthrough:
    this does NOT exercise apply_ledger's own try/except — el.read/el.append
    are independently fail-open (see test_emitted_ledger.py), so this passes
    even with apply_ledger's try/except removed entirely. It proves the
    leaf-module boundary holds when apply_ledger is layered on top, which is
    still worth having; see test_corrupt_config_flag_read_degrades_to_
    passthrough (F1) and the two fault-injection tests below for coverage of
    apply_ledger's OWN boundary."""
    monkeypatch.setenv("MEMO_EMITTED_LEDGER", "1")
    monkeypatch.setenv("MEMO_SESSION_ID", "sess-ro")
    ro = tmp_path / "ro"
    ro.mkdir()
    ro.chmod(0o500)
    try:
        hits = _hits()
        out, extra = sc.apply_ledger(_Mem(ro), "memo_search", hits)
        assert out == hits and extra == {}
    finally:
        ro.chmod(0o700)


def test_corrupt_config_flag_read_degrades_to_passthrough(tmp_path, monkeypatch):
    """F1 CRITICAL: flag_bool/flag_str for MEMO_EMITTED_LEDGER* resolve via
    memo's on-disk Markdown config when no env var is set — the normal
    deployment path (`memo config set`), not the env-var shortcut every other
    test in this file uses. config_md._read_uncached's `path.read_text(
    encoding="utf-8")` only catches OSError, so a non-UTF-8 byte in the
    user's config raises UnicodeDecodeError. Before this fix that escaped
    apply_ledger entirely, because both flag reads sat before the try — and
    it fired even with the feature never explicitly enabled, since the flag
    read is the very first statement."""
    monkeypatch.delenv("MEMO_EMITTED_LEDGER", raising=False)
    monkeypatch.delenv("MEMO_EMITTED_LEDGER_TOOLS", raising=False)
    monkeypatch.setenv("MEMO_SESSION_ID", "sess-corrupt-cfg")
    config_home = tmp_path / "config-home"
    (config_home / "config").mkdir(parents=True)
    (config_home / "config" / "advanced-config.md").write_bytes(
        b"\xff\xfe not valid utf-8 \x80\x81"
    )
    monkeypatch.setenv("MEMO_CONFIG_DIR", str(config_home))

    hits = _hits()
    out, extra = sc.apply_ledger(_Mem(tmp_path), "memo_search", hits)
    assert out == hits and extra == {}


def test_partition_raising_degrades_to_passthrough(mem, monkeypatch):
    """Fault-injection on apply_ledger's OWN exception boundary: partition()
    is a pure function that "should never" raise, but apply_ledger must not
    trust that. F6.4: assert the injected fault was actually reached — the
    prior version of this test asserted only `out == hits and extra == {}`,
    which is exactly what the healthy cold-ledger path also returns, so it
    could not distinguish "exception caught" from "no exception raised"."""
    calls = []

    def boom(*args, **kwargs):
        calls.append((args, kwargs))
        raise RuntimeError("boom")

    monkeypatch.setattr(el, "partition", boom)
    hits = _hits()
    out, extra = sc.apply_ledger(mem, "memo_search", hits)
    assert calls, "expected partition() to be called (and raise) before the assertion"
    assert out == hits and extra == {}


# -- Task 8: counters -------------------------------------------------------


def test_first_call_never_bumps_the_suppression_counters(mem, tmp_path):
    """A cold ledger emits everything in full -- nothing was suppressed, so
    the counters `stats()` reports must stay at zero."""
    sc.apply_ledger(mem, "memo_search", _hits())
    stats = el.stats(tmp_path, "sess-apply")
    assert stats["digests_served"] == 0
    assert stats["tokens_suppressed"] == 0
    assert stats["tokens_digest"] == 0
    assert stats["net_saved_est"] == 0


def test_digest_call_bumps_the_suppression_counters(mem, tmp_path):
    sc.apply_ledger(mem, "memo_search", _hits())
    _, extra = sc.apply_ledger(mem, "memo_search", _hits())

    stats = el.stats(tmp_path, "sess-apply")
    assert stats["digests_served"] == 2  # mem_a, mem_b
    expected_suppressed = est_tokens("body a") + est_tokens("body b")
    assert stats["tokens_suppressed"] == expected_suppressed
    expected_digest_cost = est_tokens(json.dumps(extra, separators=(",", ":"), default=str))
    assert stats["tokens_digest"] == expected_digest_cost
    assert stats["net_saved_est"] == expected_suppressed - expected_digest_cost
    # NOT asserted positive: these fixture bodies are 6 chars each, so the
    # fixed {id, title, ref} + "hint" text overhead of the digest stub
    # legitimately outweighs the saving here -- a real finding about the
    # stub's fixed cost, not a test bug. See task-8-report.md: the same is
    # true even for the ~49-char bodies in memory_with_memories's fixture
    # data, so this is not just a synthetic-fixture artifact -- only bodies
    # long enough for the per-byte saving to clear the stub's fixed overhead
    # make the digest pay off, and this repo's regression corpus should be
    # checked against real body-length distributions before promotion.


def test_partial_digest_call_only_counts_the_digested_hits(mem, tmp_path):
    """test_partial_overlap_across_tools's shape: mem_c is sent full (not
    suppressed), mem_a/mem_b are digested. Only the digested pair may count
    toward tokens_suppressed/digests_served."""
    sc.apply_ledger(mem, "memo_search", _hits())
    later = [*_hits(), {"id": "mem_c", "title": "C", "body": "body c"}]
    sc.apply_ledger(mem, "memo_ask", later)

    stats = el.stats(tmp_path, "sess-apply")
    assert stats["digests_served"] == 2
    assert stats["tokens_suppressed"] == est_tokens("body a") + est_tokens("body b")


def test_flag_off_never_writes_counters(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMO_EMITTED_LEDGER", "0")
    monkeypatch.setenv("MEMO_SESSION_ID", "sess-off-counters")
    m = _Mem(tmp_path)
    sc.apply_ledger(m, "memo_search", _hits())
    sc.apply_ledger(m, "memo_search", _hits())
    assert el.stats(tmp_path, "sess-off-counters") == {
        "entries": 0,
        "digests_served": 0,
        "tokens_suppressed": 0,
        "tokens_digest": 0,
        "memo_get_after_digest": 0,
        "net_saved_est": 0,
    }


def test_counter_bump_failure_does_not_break_the_actual_suppression(mem, tmp_path, monkeypatch):
    """Deliberate design point: a counter is measurement, not correctness.
    Unlike a ledger/partition failure (which degrades apply_ledger to a full
    passthrough -- see test_partition_raising_degrades_to_passthrough above),
    a broken counter must NOT undo the suppression this call already
    computed correctly. If it did, a bug in the measurement code would make
    memo re-emit content the model has already seen -- exactly the tokens
    this feature exists to stop spending."""
    sc.apply_ledger(mem, "memo_search", _hits())

    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(el, "bump", boom)
    out, extra = sc.apply_ledger(mem, "memo_search", _hits())
    assert out == []
    assert [e["id"] for e in extra["already_in_context"]] == ["mem_a", "mem_b"]


def test_session_id_lookup_raising_degrades_to_passthrough(mem, monkeypatch):
    """Fault-injection on session-id resolution: `_effective_session_id`
    currently never raises in practice (it falls back to a generated id), but
    apply_ledger's contract is "no session id resolvable -> passthrough", so
    this must hold even if that assumption ever breaks. F6.4: assert the
    injected fault was actually reached, not just that the healthy-path
    return value happens to match."""
    calls = []

    def boom() -> str:
        calls.append(True)
        raise RuntimeError("boom")

    monkeypatch.setattr(ssp, "_effective_session_id", boom)
    hits = _hits()
    out, extra = sc.apply_ledger(mem, "memo_search", hits)
    assert calls, "expected _effective_session_id() to be called (and raise)"
    assert out == hits and extra == {}
