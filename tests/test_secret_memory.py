"""Tests for isolated secret-storage Memory integration."""

import secrets
import sqlite3
import stat
from unittest.mock import MagicMock, patch

import frontmatter
import pytest
from click.testing import CliRunner
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

import memo.secret_store as secret_crypto
from memo.cli import cli
from memo.errors import MemoError


# Memory mixin tests (placeholder)
# Full tests require working Memory + conftest fixtures
# Smoke test: verify Memory has secret methods
def test_memory_has_secret_methods():
    """Memory should have secret operation methods."""
    from memo.memory import Memory

    assert hasattr(Memory, "save_secret")
    assert hasattr(Memory, "get_secret")
    assert hasattr(Memory, "list_secrets")
    assert hasattr(Memory, "forget_secret")


def test_secret_operations_require_explicit_opt_in(mem_with_stub, monkeypatch):
    monkeypatch.delenv("MEMO_SECRET_STORAGE_ENABLED", raising=False)

    with pytest.raises(MemoError, match="MEMO_SECRET_STORAGE_ENABLED=1"):
        mem_with_stub.save_secret(value="credential", name="primary", interactive=False)


def test_general_memory_api_refuses_secret_type(mem_with_stub):
    with pytest.raises(ValueError, match="type_='secret'"):
        mem_with_stub.save(content="plaintext", title="must not index", type_="secret")


def test_secret_roundtrip_never_creates_searchable_markdown(mem_with_stub, monkeypatch):
    monkeypatch.setenv("MEMO_SECRET_STORAGE_ENABLED", "1")
    value = "sk_test_plaintext_must_never_leak_123456"

    record = mem_with_stub.save_secret(
        value=value,
        name="primary-api-key",
        kind="api_token",
        interactive=False,
    )

    assert record.path.startswith("secret://")
    assert list(mem_with_stub.cfg.memory_dir.rglob("*.md")) == []
    assert mem_with_stub.search("plaintext must never leak") == []
    assert mem_with_stub.get_secret("primary-api-key") == value
    assert stat.S_IMODE(mem_with_stub.cfg.db_path.stat().st_mode) == 0o600
    for path in mem_with_stub.cfg.state_dir.rglob("*"):
        if path.is_file():
            assert value.encode() not in path.read_bytes()

    mem_with_stub.forget_secret("primary-api-key")
    assert mem_with_stub.list_secrets() == []


def test_failed_duplicate_save_preserves_existing_secret(mem_with_stub, monkeypatch):
    monkeypatch.setenv("MEMO_SECRET_STORAGE_ENABLED", "1")
    mem_with_stub.save_secret(
        value="original",
        name="same-name",
        kind="generic",
        interactive=False,
    )

    with pytest.raises(sqlite3.IntegrityError):
        mem_with_stub.save_secret(
            value="replacement",
            name="same-name",
            kind="generic",
            interactive=False,
        )

    assert mem_with_stub.get_secret("same-name") == "original"


def test_post_insert_security_failure_rolls_back_row(mem_with_stub, monkeypatch):
    monkeypatch.setenv("MEMO_SECRET_STORAGE_ENABLED", "1")

    def fail_to_secure() -> None:
        raise PermissionError("cannot secure database")

    monkeypatch.setattr(mem_with_stub, "_secure_secret_state_files", fail_to_secure)

    with pytest.raises(PermissionError, match="cannot secure"):
        mem_with_stub.save_secret(
            value="credential",
            name="rolled-back",
            kind="generic",
            interactive=False,
        )

    assert mem_with_stub.list_secrets() == []


def test_secret_names_reject_log_injection(mem_with_stub, monkeypatch):
    monkeypatch.setenv("MEMO_SECRET_STORAGE_ENABLED", "1")

    with pytest.raises(ValueError, match="control whitespace"):
        mem_with_stub.save_secret(
            value="credential",
            name="safe\nop=forged",
            kind="generic",
            interactive=False,
        )


def test_legacy_ciphertext_rotates_to_random_master_key_on_read(mem_with_stub, monkeypatch):
    monkeypatch.setenv("MEMO_SECRET_STORAGE_ENABLED", "1")
    legacy_key = b"l" * 32
    nonce = secrets.token_bytes(12)
    legacy_ciphertext = AESGCM(legacy_key).encrypt(nonce, b"legacy-value", None)
    monkeypatch.setattr(secret_crypto, "_derive_legacy_secret_key", lambda: legacy_key)
    mem_with_stub.store.secret_store_insert(
        id="sec_legacy_crypto",
        name="legacy-crypto",
        kind="generic",
        encrypted_blob=legacy_ciphertext,
        nonce=nonce,
        created_at="2026-07-14T00:00:00+00:00",
        detection_method="manual",
    )

    assert mem_with_stub.get_secret("legacy-crypto") == "legacy-value"

    rotated = mem_with_stub.store.secret_store_get("legacy-crypto")
    assert rotated is not None
    assert not secret_crypto.secret_ciphertext_needs_rotation(rotated["encrypted_blob"])


def test_forget_removes_legacy_secret_marker(mem_with_stub, monkeypatch):
    monkeypatch.setenv("MEMO_SECRET_STORAGE_ENABLED", "1")
    record = mem_with_stub.save_secret(
        value="credential",
        name="legacy-key",
        kind="generic",
        interactive=False,
    )
    marker = mem_with_stub.cfg.memory_dir / "secrets" / "2026" / "07" / "legacy-key.md"
    marker.parent.mkdir(parents=True)
    marker.write_text(
        frontmatter.dumps(
            frontmatter.Post(
                "[ENCRYPTED: AES-256-GCM]",
                id=record.id,
                type="secret",
                name="legacy-key",
            )
        ),
        encoding="utf-8",
    )

    mem_with_stub.forget_secret("legacy-key")

    assert not marker.exists()


def test_reindex_purges_legacy_secret_from_search_index(mem_with_stub, monkeypatch):
    marker = mem_with_stub.cfg.memory_dir / "secrets" / "2026" / "07" / "legacy.md"
    marker.parent.mkdir(parents=True)
    marker.write_text(
        frontmatter.dumps(
            frontmatter.Post(
                "[ENCRYPTED: AES-256-GCM]",
                id="sec_legacy",
                title="legacy-key",
                type="secret",
                created="2026-07-14T00:00:00+00:00",
            )
        ),
        encoding="utf-8",
    )
    mem_with_stub.store.upsert(
        id_="sec_legacy",
        path=str(marker.relative_to(mem_with_stub.cfg.memory_dir)),
        title="legacy-key",
        type_="secret",
        tags=[],
        created="2026-07-14T00:00:00+00:00",
        updated="2026-07-14T00:00:00+00:00",
        body_hash="legacy",
        embedding=[1.0, 0.0, 0.0, 0.0],
        body_text="[ENCRYPTED: AES-256-GCM]",
    )

    counts = mem_with_stub.reindex(force=True)

    assert counts["skipped"] == 1
    assert mem_with_stub.store.get("sec_legacy") is None


def test_secret_list_closes_memory(tmp_path):
    mock_memory = MagicMock()
    mock_memory.list_secrets.return_value = []

    with (
        patch("memo.cli_secret.Config.from_env", return_value=MagicMock()),
        patch("memo.cli_secret.Memory", return_value=mock_memory),
    ):
        result = CliRunner().invoke(cli, ["secret", "list"], env={"MEMO_NONINTERACTIVE": "1"})

    assert result.exit_code == 0, result.output
    mock_memory.close.assert_called_once_with()


def test_secret_get_closes_memory_on_error(tmp_path):
    mock_memory = MagicMock()
    mock_memory.get_secret.side_effect = RuntimeError("missing")

    with (
        patch("memo.cli_secret.Config.from_env", return_value=MagicMock()),
        patch("memo.cli_secret.Memory", return_value=mock_memory),
    ):
        result = CliRunner().invoke(
            cli,
            ["secret", "get", "--name", "missing"],
            env={"MEMO_NONINTERACTIVE": "1"},
        )

    assert result.exit_code != 0
    mock_memory.close.assert_called_once_with()
