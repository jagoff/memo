"""Tests for secret storage (flags, encryption, key custody, detection)."""

import os
import secrets
import stat

import pytest
from cryptography.exceptions import InvalidTag

from memo.flags import REGISTRY
from memo.secret_store import (
    SecretKeyError,
    decrypt_secret,
    derive_secret_key,
    detect_secrets_heuristic,
    encrypt_secret,
)
from memo.tiers import DURABLE_TYPES, SECRET_KINDS


def test_secret_storage_is_opt_in_and_not_a_recall_tier():
    """Credential storage must not silently join searchable durable memory."""
    assert REGISTRY["MEMO_SECRET_STORAGE_ENABLED"].default is False
    assert "secret" not in DURABLE_TYPES


def test_secret_kinds_defined():
    """All secret kinds should be defined."""
    expected_kinds = {"api_token", "password", "ssh_key", "db_credential", "certificate", "generic"}
    assert expected_kinds == SECRET_KINDS


# Encryption tests
def test_encrypt_decrypt_roundtrip(tmp_path):
    """Secret should survive encrypt/decrypt cycle."""
    value = "sk_test_1234567890abcdef"
    ciphertext, nonce = encrypt_secret(value, state_dir=tmp_path)
    decrypted = decrypt_secret(ciphertext, nonce, state_dir=tmp_path)
    assert decrypted == value
    assert value.encode() not in ciphertext


def test_master_key_is_random_private_and_reused(tmp_path):
    first, nonce = encrypt_secret("first", state_dir=tmp_path)
    key_path = tmp_path / "secret-master.key"
    key = key_path.read_bytes()

    assert len(key) == 32
    assert stat.S_IMODE(key_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(tmp_path.stat().st_mode) == 0o700
    assert decrypt_secret(first, nonce, state_dir=tmp_path) == "first"

    encrypt_secret("second", state_dir=tmp_path)
    assert key_path.read_bytes() == key


def test_master_key_refuses_permissive_or_symlinked_files(tmp_path):
    key_path = tmp_path / "secret-master.key"
    key_path.write_bytes(os.urandom(32))
    key_path.chmod(0o644)

    with pytest.raises(SecretKeyError, match="permissions"):
        encrypt_secret("blocked", state_dir=tmp_path)

    key_path.unlink()
    target = tmp_path / "elsewhere"
    target.write_bytes(os.urandom(32))
    key_path.symlink_to(target)
    with pytest.raises(SecretKeyError, match="symlink"):
        encrypt_secret("blocked", state_dir=tmp_path)


def test_key_derivation_deterministic(tmp_path):
    """Same machine should derive same key twice."""
    key1 = derive_secret_key(state_dir=tmp_path)
    key2 = derive_secret_key(state_dir=tmp_path)
    assert key1 == key2


def test_different_values_different_ciphertexts(tmp_path):
    """Different values should produce different ciphertexts."""
    value1 = "secret1"
    value2 = "secret2"
    ct1, _nonce1 = encrypt_secret(value1, state_dir=tmp_path)
    ct2, _nonce2 = encrypt_secret(value2, state_dir=tmp_path)
    # Even though nonce is random, ciphertexts should differ
    assert ct1 != ct2


def test_invalid_nonce_raises(tmp_path):
    """Decrypting with wrong nonce should raise."""
    value = "secret"
    ct, _nonce = encrypt_secret(value, state_dir=tmp_path)
    wrong_nonce = secrets.token_bytes(12)

    with pytest.raises(InvalidTag):
        decrypt_secret(ct, wrong_nonce, state_dir=tmp_path)


def test_authenticated_metadata_prevents_row_swaps(tmp_path):
    ciphertext, nonce = encrypt_secret(
        "credential",
        state_dir=tmp_path,
        associated_data=b"memo-secret:primary:api_token",
    )

    with pytest.raises(InvalidTag):
        decrypt_secret(
            ciphertext,
            nonce,
            state_dir=tmp_path,
            associated_data=b"memo-secret:other:api_token",
        )


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
