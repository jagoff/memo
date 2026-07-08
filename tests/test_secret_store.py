"""Tests for secret storage (tier, flags, encryption, detection)."""

import secrets

import pytest
from cryptography.exceptions import InvalidTag

from memo.secret_store import (
    decrypt_secret,
    derive_secret_key,
    detect_secrets_heuristic,
    detect_secrets_llm,
    encrypt_secret,
)
from memo.tiers import DURABLE_TYPES, SECRET_KINDS


def test_secret_in_durable_types():
    """Secret tier should be in durable types."""
    assert "secret" in DURABLE_TYPES


def test_secret_kinds_defined():
    """All secret kinds should be defined."""
    expected_kinds = {"api_token", "password", "ssh_key", "db_credential", "certificate", "generic"}
    assert expected_kinds == SECRET_KINDS


# Encryption tests
def test_encrypt_decrypt_roundtrip():
    """Secret should survive encrypt/decrypt cycle."""
    value = "sk_test_1234567890abcdef"
    ciphertext, nonce = encrypt_secret(value)
    decrypted = decrypt_secret(ciphertext, nonce)
    assert decrypted == value


def test_key_derivation_deterministic():
    """Same machine should derive same key twice."""
    key1 = derive_secret_key()
    key2 = derive_secret_key()
    assert key1 == key2


def test_different_values_different_ciphertexts():
    """Different values should produce different ciphertexts."""
    value1 = "secret1"
    value2 = "secret2"
    ct1, _nonce1 = encrypt_secret(value1)
    ct2, _nonce2 = encrypt_secret(value2)
    # Even though nonce is random, ciphertexts should differ
    assert ct1 != ct2


def test_invalid_nonce_raises():
    """Decrypting with wrong nonce should raise."""
    value = "secret"
    ct, _nonce = encrypt_secret(value)
    wrong_nonce = secrets.token_bytes(12)

    with pytest.raises(InvalidTag):
        decrypt_secret(ct, wrong_nonce)


# Detection tests
def test_detect_api_token_regex():
    """Should detect OpenAI-style API tokens."""
    content = "My OpenAI key is sk_test_1234567890abcdefghij"
    matches = detect_secrets_heuristic(content)
    assert any(kind == "api_token" for kind, _ in matches)


def test_detect_password_regex():
    """Should detect password patterns."""
    content = "password=mysecret123"
    matches = detect_secrets_heuristic(content)
    assert any(kind == "password" for kind, _ in matches)


def test_detect_ssh_key_regex():
    """Should detect SSH private key headers."""
    content = "-----BEGIN RSA PRIVATE KEY-----\nMIIEpA..."
    matches = detect_secrets_heuristic(content)
    assert any(kind == "ssh_key" for kind, _ in matches)


def test_detect_db_credential_regex():
    """Should detect database connection strings."""
    content = "postgres://user:pass@localhost/db"
    matches = detect_secrets_heuristic(content)
    assert any(kind == "db_credential" for kind, _ in matches)


def test_no_false_positives_on_normal_text():
    """Should not detect secrets in normal text."""
    content = "The password to the kingdom is not here."
    matches = detect_secrets_heuristic(content)
    assert len(matches) == 0


def test_llm_detection_skipped_if_no_heuristic_matches():
    """LLM detection should skip if heuristic found nothing."""
    content = "This is normal text."
    mock_llm = object()  # Would fail if called
    matches = detect_secrets_llm(content, [], mock_llm)
    assert matches == []


def test_llm_detection_with_mock():
    """LLM detection should parse and return LLM response."""

    class MockLLM:
        def chat(self, messages):
            return '[{"kind": "api_token", "confidence": 0.95}]'

    heuristic_matches = [("api_token", 0.7)]
    result = detect_secrets_llm("some content", heuristic_matches, MockLLM())
    assert result == [("api_token", 0.95)]
