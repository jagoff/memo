"""claim_support: pure evidence-ref detector for outcome claims."""
from __future__ import annotations

from memo import claim_support


def test_claim_with_no_ref_is_unsupported():
    r = claim_support.check_claim_support("Fixed the recall timeout bug.")
    assert r.unsupported is True
    assert r.claim_kind


def test_hedged_claim_is_exempt():
    r = claim_support.check_claim_support("I think this probably fixes the timeout.")
    assert r.unsupported is False


def test_first_person_intent_is_exempt():
    r = claim_support.check_claim_support("I'm going to make the search faster.")
    assert r.unsupported is False


def test_claim_with_real_commit_ref_is_supported(monkeypatch):
    monkeypatch.setattr(claim_support, "_commit_exists", lambda sha, root: True)
    r = claim_support.check_claim_support("Shipped the fix in commit:a1b2c3d4.")
    assert r.unsupported is False


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
