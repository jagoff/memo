from memo import emitted_ledger as el


def _known(mid: str, text: str) -> el.Entry:
    # hp mirrors what Entry.for_text computes: the prefix hash of the text
    # ACTUALLY emitted. Constructing it any other way here would let a test
    # assert on a code path that real callers never exercise.
    return el.Entry(
        id=mid,
        h=el.emitted_hash(text),
        n=len(text),
        ref="memo-r/aaaaaa",
        t=1,
        src="mcp",
        hp=el.emitted_hash(text[: el._PREFIX_CHARS]),
    )


def _hit(mid: str, body: str) -> dict[str, str]:
    return {"id": mid, "title": f"title of {mid}", "body": body}


def _partition(hits, known):
    return el.partition(hits, known, text_of=lambda h: h["body"], id_of=lambda h: h["id"])


def test_empty_ledger_emits_everything_full():
    hits = [_hit("a", "body a"), _hit("b", "body b")]
    out = _partition(hits, {})
    assert out.full == hits
    assert out.digest == []
    assert out.suppressed_chars == 0


def test_identical_reemission_is_digested():
    hits = [_hit("a", "body a")]
    out = _partition(hits, {"a": _known("a", "body a")})
    assert out.full == []
    assert [h["id"] for h in out.digest] == ["a"]
    assert out.suppressed_chars == len("body a")


def test_changed_body_same_length_is_reemitted_full():
    # Same length as the recorded entry (6 chars) but genuinely different
    # content — not a truncation/edit-preserving-prefix case. Distinct from
    # test_longer_emission_wins_over_recorded_shorter_one (which covers a
    # longer rendering): the old length-only rule would have wrongly digested
    # this because len(text) <= prior.n regardless of content.
    hits = [_hit("a", "cody a")]
    out = _partition(hits, {"a": _known("a", "body a")})
    assert out.full == hits
    assert out.digest == []


def test_partial_overlap_splits():
    hits = [_hit("a", "body a"), _hit("b", "body b")]
    out = _partition(hits, {"a": _known("a", "body a")})
    assert [h["id"] for h in out.full] == ["b"]
    assert [h["id"] for h in out.digest] == ["a"]


def test_longer_emission_wins_over_recorded_shorter_one():
    # The hook emitted 400 chars at turn 2; memo_ask now has room for 900.
    # Digesting here would suppress 500 chars the model has never seen.
    short = "x" * 400
    longer = "x" * 900
    out = _partition([_hit("a", longer)], {"a": _known("a", short)})
    assert out.full and out.full[0]["body"] == longer
    assert out.digest == []


def test_shorter_emission_of_same_prefix_is_digested():
    longer = "x" * 900
    shorter = "x" * 400
    out = _partition([_hit("a", shorter)], {"a": _known("a", longer)})
    assert out.full == []
    assert [h["id"] for h in out.digest] == ["a"]
    # Pin to the CHARS ACTUALLY SUPPRESSED (len(shorter)), not prior.n (900).
    # The two only diverge in this truncation scenario; a suppressed_chars that
    # reported prior.n here would over-claim what this call actually digested,
    # even though nothing outside this test module currently reads the field
    # (see Partition's docstring -- the real gate numerator is computed
    # independently in server_common.apply_ledger).
    assert out.suppressed_chars == 400


def test_body_edited_at_start_and_shortened_is_reemitted_full():
    # The hole F2 closes: length alone (18 <= 26) would have wrongly let this
    # through under the pre-F2 rule, even though the hash also mismatches —
    # the OR made either check sufficient. The prefix hash now catches it: the
    # start of the body changed, so the model was never shown this text.
    known_text = "the quick brown fox jumps"
    edited_and_shorter = "A quick brown fox"
    assert len(edited_and_shorter) <= len(known_text)  # exercises the length arm
    out = _partition([_hit("a", edited_and_shorter)], {"a": _known("a", known_text)})
    assert out.full and out.full[0]["body"] == edited_and_shorter
    assert out.digest == []


def test_body_shortened_by_trailing_deletion_prefix_intact_is_digested():
    # The case F2 must not break: a real (>200 char) body truncated at the
    # end. The first _PREFIX_CHARS characters are untouched, so the prefix
    # hash still matches even though the overall hash does not.
    known_text = (
        "The incident began at 09:00 UTC when the primary database node "
        "became unresponsive under sustained load from a runaway batch job. "
    ) * 2
    assert len(known_text) > el._PREFIX_CHARS
    trailing_deletion = known_text[:220]
    out = _partition([_hit("a", trailing_deletion)], {"a": _known("a", known_text)})
    assert out.full == []
    assert [h["id"] for h in out.digest] == ["a"]


def test_entry_with_no_prefix_hash_is_reemitted_full():
    # Simulates an entry read from a ledger file written before `hp` existed
    # (read() yields hp=None for those lines). Length alone would pass
    # (5 <= 30), but an unknown prefix hash must never be treated as a match.
    known = el.Entry(
        id="a",
        h=el.emitted_hash("something else entirely here"),
        n=len("something else entirely here"),
        ref="memo-r/aaaaaa",
        t=1,
        src="mcp",
        hp=None,
    )
    hit = _hit("a", "short")
    out = _partition([hit], {"a": known})
    assert out.full == [hit]
    assert out.digest == []


def test_title_only_prior_emission_does_not_suppress_a_body():
    # render_recall_compact emits no body; n == 0 must never digest real text.
    out = _partition([_hit("a", "a real body")], {"a": _known("a", "")})
    assert out.full and out.digest == []


def test_hit_order_is_preserved_within_each_bucket():
    hits = [_hit(x, f"body {x}") for x in ("a", "b", "c", "d")]
    known = {"b": _known("b", "body b"), "d": _known("d", "body d")}
    out = _partition(hits, known)
    assert [h["id"] for h in out.full] == ["a", "c"]
    assert [h["id"] for h in out.digest] == ["b", "d"]
