from memo import emitted_ledger as el


def _known(mid: str, text: str) -> el.Entry:
    return el.Entry(
        id=mid, h=el.emitted_hash(text), n=len(text), ref="memo-r/aaaaaa", t=1, src="mcp"
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


def test_changed_body_is_reemitted_full():
    hits = [_hit("a", "body a, edited")]
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
