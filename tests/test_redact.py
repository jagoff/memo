"""Secret redaction + <private> span stripping — pure module, no MLX, no env."""

from __future__ import annotations

from memo.redact import redact_secrets, scan_secrets, strip_private_spans


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


def test_privacy_flags_registered_with_defaults(monkeypatch):
    from memo.flags import flag_bool

    monkeypatch.delenv("MEMO_REDACT_SECRETS", raising=False)
    monkeypatch.delenv("MEMO_REDACT_ENTROPY", raising=False)
    monkeypatch.delenv("MEMO_PRIVATE_MARKERS", raising=False)
    assert flag_bool("MEMO_REDACT_SECRETS") is True
    assert flag_bool("MEMO_REDACT_ENTROPY") is False
    assert flag_bool("MEMO_PRIVATE_MARKERS") is True
