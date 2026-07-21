"""claim_support: pure evidence-ref detector for outcome claims."""

from __future__ import annotations

from memo import claim_support


def test_claim_with_no_ref_is_unsupported():
    r = claim_support.check_claim_support("Fixed the recall timeout bug.")
    assert r.unsupported is True
    assert r.claim_kind


def test_hedged_claim_is_exempt():
    r = claim_support.check_claim_support("Maybe this fix works.")
    assert r.unsupported is False


def test_hedge_propagates_across_coordinated_outcome_list():
    r = claim_support.check_claim_support("Maybe the fix works, passes tests, and is secure.")
    assert r.unsupported is False


def test_first_person_hedge_propagates_across_coordinated_outcome_list():
    r = claim_support.check_claim_support("I think the parser works, passes, and is secure.")
    assert r.unsupported is False


def test_hedge_in_prior_sentence_does_not_exempt_fixed_claim():
    r = claim_support.check_claim_support("Maybe investigate logs. Fixed the production timeout.")
    assert r.unsupported is True
    assert r.claim_kind == "fixed"


def test_todo_in_prior_sentence_does_not_exempt_shipped_claim():
    r = claim_support.check_claim_support("TODO document rollout. Shipped the production fix.")
    assert r.unsupported is True
    assert r.claim_kind == "shipped"


def test_hedge_before_comma_does_not_exempt_fixed_claim():
    r = claim_support.check_claim_support("Maybe investigate logs, fixed the production timeout.")
    assert r.unsupported is True
    assert r.claim_kind == "fixed"


def test_todo_before_em_dash_does_not_exempt_shipped_claim():
    r = claim_support.check_claim_support("TODO document rollout — shipped the production fix.")
    assert r.unsupported is True
    assert r.claim_kind == "shipped"


def test_hedge_before_colon_does_not_exempt_fixed_claim():
    r = claim_support.check_claim_support("Maybe investigate logs: fixed the production timeout.")
    assert r.unsupported is True
    assert r.claim_kind == "fixed"


def test_first_person_intent_is_exempt():
    r = claim_support.check_claim_support("I'm going to make the search faster.")
    assert r.unsupported is False


def test_claim_with_real_commit_ref_is_supported(monkeypatch):
    monkeypatch.setattr(claim_support, "_commit_exists", lambda sha, root: True)
    r = claim_support.check_claim_support("Shipped the fix in commit:a1b2c3d4.")
    assert r.unsupported is False


def test_colon_evidence_refs_remain_supported(monkeypatch):
    monkeypatch.setattr(claim_support, "_commit_exists", lambda sha, root: True)
    for text in (
        "Shipped the fix in commit:a1b2c3d4.",
        "Shipped the fix in pr:#123.",
        "Fixed the build in ci:main.",
    ):
        assert claim_support.check_claim_support(text).unsupported is False


def test_claim_with_fabricated_commit_ref_is_unsupported(monkeypatch):
    monkeypatch.setattr(claim_support, "_commit_exists", lambda sha, root: False)
    r = claim_support.check_claim_support("Shipped the fix in commit:deadbeef.")
    assert r.unsupported is True
    assert "commit" in r.reason.lower()


def test_test_green_counts_as_support():
    r = claim_support.check_claim_support("The parser works now — tests green.")
    assert r.unsupported is False


def test_non_claim_text_is_supported():
    r = claim_support.check_claim_support("The dashboard runs on port 8765.")
    assert r.unsupported is False
    assert r.claim_kind == ""
