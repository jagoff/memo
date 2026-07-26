"""Secret redaction + <private> span stripping — pure module, no MLX, no env."""

from __future__ import annotations

import time

import pytest

from memo.errors import ValidationError
from memo.redact import (
    redact_secrets,
    sanitize_memory_input,
    sanitize_persisted_text,
    scan_secrets,
    strip_private_spans,
)


def test_redacts_aws_access_key():
    res = redact_secrets("creds: AKIAIOSFODNN7EXAMPLE done")
    assert "AKIAIOSFODNN7EXAMPLE" not in res.text
    assert "****MPLE" in res.text
    assert "aws-key" in res.found


def test_redacts_github_token():
    tok = "ghp_" + "a" * 32 + "WXYZ"
    res = redact_secrets(f"push failed with {tok} in remote url")
    assert tok not in res.text
    assert "****WXYZ" in res.text
    assert "github-token" in res.found


def test_redacts_openai_and_anthropic_keys_distinctly():
    openai = "sk-proj-" + "b" * 24 + "1234"
    anthropic = "sk-ant-" + "c" * 24 + "5678"
    res = redact_secrets(f"{openai} vs {anthropic}")
    assert openai not in res.text and anthropic not in res.text
    assert "openai-key" in res.found and "anthropic-key" in res.found


def test_redacts_private_key_block():
    pem = "-----BEGIN OPENSSH PRIVATE KEY-----\nabc\ndef\n-----END OPENSSH PRIVATE KEY-----"
    res = redact_secrets(f"key:\n{pem}\nend")
    assert "BEGIN OPENSSH" not in res.text
    assert "****[private-key]" in res.text
    assert "private-key" in res.found


def test_pem_requires_joined_text_not_per_line():
    """Regression guard for the sync gate: _PEM_RE only matches when BEGIN
    and END are in the SAME string — a per-line scan finds nothing. Any
    caller feeding line-oriented input (staged diffs) must join first."""
    pem = "-----BEGIN OPENSSH PRIVATE KEY-----\nabc\ndef\n-----END OPENSSH PRIVATE KEY-----"
    per_line = [f for line in pem.splitlines() for f in scan_secrets(line)]
    assert per_line == []  # per-line: invisible
    assert ("private-key", "****[private-key]") in scan_secrets(pem)  # joined: caught


@pytest.mark.parametrize("label", ["RSA ", "EC ", "ENCRYPTED ", "OPENSSH ", ""])
def test_redacts_private_key_block_across_label_forms(label):
    """Every real PEM key-type label ("RSA ", "EC ", "ENCRYPTED ", "OPENSSH ",
    plain) stays redacted under the bounded ``[A-Z ]{0,40}`` quantifier."""
    pem = f"-----BEGIN {label}PRIVATE KEY-----\nabc\ndef\n-----END {label}PRIVATE KEY-----"
    res = redact_secrets(f"key:\n{pem}\nend")
    assert f"BEGIN {label}PRIVATE KEY" not in res.text
    assert "****[private-key]" in res.text
    assert "private-key" in res.found


def test_pem_near_miss_without_close_marker_is_linear_and_unmatched():
    # Pathological shape for the former unbounded ``[A-Z ]*``: a BEGIN marker
    # followed by 10k uppercase chars that never complete ``PRIVATE KEY-----``.
    # The bounded ``{0,40}`` quantifier makes matching linear — it returns
    # promptly and leaves the fragment untouched (guards py/polynomial-redos).
    fragment = "-----BEGIN " + "A" * 10_000
    start = time.perf_counter()
    res = redact_secrets(fragment)
    assert time.perf_counter() - start < 1.0
    assert res.text == fragment
    assert res.found == ()


def test_clean_text_untouched():
    text = "we decided to use sqlite-vec because BM25 alone missed diacritics"
    res = redact_secrets(text)
    assert res.text == text
    assert res.found == ()


def test_hex_hashes_never_masked_even_with_entropy():
    sha = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    res = redact_secrets(f"commit {sha}", entropy=True)
    assert sha in res.text


def test_entropy_tier_masks_random_token_only_when_enabled():
    tok = "9fXk2Lq8Rz3Vw7Yb1Nd5Mh0Pg4St6Uj8Ca"  # 34 chars, mixed case + digits
    assert tok in redact_secrets(f"token {tok}").text  # entropy tier off by default
    res = redact_secrets(f"token {tok}", entropy=True)
    assert tok not in res.text
    assert "high-entropy" in res.found


def test_scan_secrets_reports_without_rewriting():
    tok = "ghp_" + "a" * 32 + "WXYZ"
    assert scan_secrets(f"body {tok}") == [("github-token", "****WXYZ")]


def test_strip_private_spans_removes_span():
    out = strip_private_spans("keep this <private>secret stuff</private> and this")
    assert "secret stuff" not in out
    assert "keep this" in out and "and this" in out


def test_strip_private_spans_multiline_and_unclosed_drops_to_end():
    out = strip_private_spans("public\n<private>\nline1\nline2\n")
    assert out == "public"


def test_strip_private_spans_multiple_spans_removed():
    out = strip_private_spans("a<private>x</private>b<private>y</private>c")
    assert out == "abc"


def test_strip_private_spans_many_unclosed_markers_is_linear():
    # Pathological shape for the former ``<private>.*?</private>`` regex: many
    # opens, no close. Linear scan drops from the first open to EOT and returns
    # promptly (guards the py/polynomial-redos fix).
    text = "keep " + "<private>" * 50_000 + "tail-no-close"
    assert strip_private_spans(text) == "keep"


def test_privacy_flags_registered_with_defaults(monkeypatch):
    from memo.flags import flag_bool

    monkeypatch.delenv("MEMO_REDACT_SECRETS", raising=False)
    monkeypatch.delenv("MEMO_REDACT_ENTROPY", raising=False)
    monkeypatch.delenv("MEMO_PRIVATE_MARKERS", raising=False)
    assert flag_bool("MEMO_REDACT_SECRETS") is True
    assert flag_bool("MEMO_REDACT_ENTROPY") is False
    assert flag_bool("MEMO_PRIVATE_MARKERS") is True


def test_persistence_sanitizer_covers_nested_record_fields() -> None:
    token = "ghp_" + "a" * 32 + "WXYZ"
    result = sanitize_memory_input(
        content=f"public <private>never store</private> {token}",
        title=f"title {token}",
        tags=[f"tag-{token}"],
        topic_key=f"topic-{token}",
        normalized_hash=f"legacy-{token}",
        extra={f"key-{token}": {"items": [token, 7, True]}},
    )
    serialized = repr(result)
    assert token not in serialized
    assert "never store" not in serialized
    assert result.changed is True
    assert result.tags.count("_redacted") == 1


def test_persistence_sanitizer_rejects_empty_or_colliding_metadata_keys() -> None:
    with pytest.raises(ValidationError, match="empty key"):
        sanitize_memory_input(content="safe", extra={"<private>x</private>": 1})

    token_a = "ghp_" + "a" * 32 + "WXYZ"
    token_b = "ghp_" + "b" * 32 + "WXYZ"
    with pytest.raises(ValidationError, match="colliding keys"):
        sanitize_memory_input(content="safe", extra={token_a: 1, token_b: 2})


def test_persistence_sanitizer_rejects_private_only_content_and_topic() -> None:
    with pytest.raises(ValidationError, match="content is empty"):
        sanitize_memory_input(content="<private>all private</private>")
    with pytest.raises(ValidationError, match="topic_key is empty"):
        sanitize_memory_input(content="safe", topic_key="<private>x</private>")


def test_persistence_sanitizer_preserves_benign_hashes_and_scalars() -> None:
    sha = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    result = sanitize_memory_input(
        content=f"commit {sha}",
        extra={"sha": sha, "count": 2, "enabled": False},
        entropy=True,
    )
    assert sha in result.content
    assert result.extra == {"sha": sha, "count": 2, "enabled": False}
    assert result.changed is False


def test_sanitize_persisted_text_reports_private_span() -> None:
    result = sanitize_persisted_text("keep <private>drop</private>")
    assert result.text == "keep"
    assert result.found == ("private-span",)
