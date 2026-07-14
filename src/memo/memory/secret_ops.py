"""Secret operations mixin for Memory facade."""

from __future__ import annotations

import hashlib
import logging
import os
import secrets
import sqlite3
import stat
from pathlib import Path
from typing import Any

import frontmatter

from memo.errors import MemoError
from memo.flags import flag_bool
from memo.memory._base import _MemoryBase
from memo.memory.record import MemoryRecord, _now_iso
from memo.secret_store import (
    decrypt_secret,
    detect_secrets_heuristic,
    encrypt_secret,
    secret_ciphertext_needs_rotation,
)

_log = logging.getLogger(__name__)


def _require_secret_storage_enabled() -> None:
    if not flag_bool("MEMO_SECRET_STORAGE_ENABLED"):
        raise MemoError(
            "Secret storage is disabled by default; explicitly set "
            "MEMO_SECRET_STORAGE_ENABLED=1 to opt in"
        )


def _validate_secret_name(name: str) -> str:
    clean = name.strip()
    if clean != name or not clean or len(clean) > 128 or any(ord(char) < 32 for char in clean):
        raise ValueError(
            "Secret name must be 1-128 characters with no surrounding/control whitespace"
        )
    return clean


def _validate_secret_value(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("Secret value must be a non-empty string")
    if len(value.encode()) > 1024 * 1024:
        raise ValueError("Secret value exceeds the 1 MiB storage limit")
    return value


def _secret_aad(name: str, kind: str) -> bytes:
    return f"memo-secret:{name}:{kind}".encode()


class _SecretOpsMixin(_MemoryBase):
    """Operations for managing encrypted secrets."""

    def save_secret(
        self,
        *,
        value: str,
        name: str,
        kind: str | None = None,
        tags: list[str] | None = None,
        interactive: bool = True,
    ) -> MemoryRecord:
        """
        Save an encrypted secret.

        1. Auto-detect kind if not provided
        2. Encrypt with AES-256-GCM
        3. Save only to the isolated sqlite secret table
        4. Log access
        """
        from memo.tiers import SECRET_KINDS

        _require_secret_storage_enabled()
        name = _validate_secret_name(name)
        value = _validate_secret_value(value)
        if kind and kind not in SECRET_KINDS:
            raise ValueError(f"Invalid kind: {kind}. Choices: {SECRET_KINDS}")

        # Auto-detect if kind not provided
        if not kind:
            matches = detect_secrets_heuristic(value[:200])
            if matches:
                kind = max(matches, key=lambda x: x[1])[0]  # Pick highest-confidence
        kind = kind or "generic"

        # User confirmation (if interactive)
        if interactive:
            import click

            prompt = f"Found possible {kind} secret '{name}'. Save encrypted? [Y/n] "
            response = click.prompt(prompt, default="y", show_default=False).strip().lower()
            if response == "n":
                raise MemoError("Secret save cancelled by user")

        # Encrypt
        ciphertext, nonce = encrypt_secret(
            value,
            state_dir=self.cfg.state_dir,
            associated_data=_secret_aad(name, kind),
        )

        # Create record
        now_iso = _now_iso()
        record_id = f"sec_{secrets.token_hex(8)}"

        markdown_content = "[ENCRYPTED: AES-256-GCM]"
        inserted = False
        try:
            self.store.secret_store_insert(
                id=record_id,
                name=name,
                kind=kind,
                encrypted_blob=ciphertext,
                nonce=nonce,
                created_at=now_iso,
                detection_method="manual",
            )
            inserted = True
            self._secure_secret_state_files()
        except (OSError, sqlite3.Error):
            # The database row is the only durable write. Roll it back if a
            # post-insert security invariant (notably permissions) fails.
            if inserted:
                self.store.secret_store_delete(name)
            raise

        # Log access
        self._log_secret_access(record_id, "save", name)

        return MemoryRecord(
            id=record_id,
            path=f"secret://{record_id}",
            type="secret",
            title=name,
            tags=tags or [],
            created=now_iso,
            updated=now_iso,
            body=markdown_content,
            extra={"kind": kind, "name": name},
        )

    def get_secret(self, name: str) -> str:
        """Retrieve and decrypt a secret by name."""
        _require_secret_storage_enabled()
        name = _validate_secret_name(name)
        secret_row = self.store.secret_store_get(name)
        if not secret_row:
            raise MemoError(f"Secret not found: {name}")

        ciphertext = secret_row["encrypted_blob"]
        nonce = secret_row["nonce"]

        try:
            plaintext = decrypt_secret(
                ciphertext,
                nonce,
                state_dir=self.cfg.state_dir,
                associated_data=_secret_aad(name, secret_row["kind"]),
            )
        except Exception as exc:
            raise MemoError(f"Failed to decrypt secret '{name}': {exc}") from exc

        if secret_ciphertext_needs_rotation(ciphertext):
            rotated, rotated_nonce = encrypt_secret(
                plaintext,
                state_dir=self.cfg.state_dir,
                associated_data=_secret_aad(name, secret_row["kind"]),
            )
            self.store.secret_store_update_encrypted(name, rotated, rotated_nonce)

        # Update access metadata
        self.store.secret_store_increment_access(name)
        self._log_secret_access(secret_row["id"], "get", name)

        return plaintext

    def list_secrets(self, kind: str | None = None) -> list[dict[str, Any]]:
        """List secret metadata (names and kinds, not values)."""
        from memo.tiers import SECRET_KINDS

        _require_secret_storage_enabled()
        if kind is not None and kind not in SECRET_KINDS:
            raise ValueError(f"Invalid kind: {kind}. Choices: {SECRET_KINDS}")
        return self.store.secret_store_list(kind=kind)

    def forget_secret(self, name: str) -> None:
        """Delete a secret."""
        _require_secret_storage_enabled()
        name = _validate_secret_name(name)
        secret_row = self.store.secret_store_get(name)
        if not secret_row:
            raise MemoError(f"Secret not found: {name}")

        deleted = self.store.secret_store_delete(name)
        if not deleted:
            raise MemoError(f"Failed to delete secret: {name}")

        failures: list[str] = []
        for marker in self._legacy_secret_markers(secret_row["id"], name):
            try:
                marker.unlink()
            except OSError as exc:
                failures.append(f"{marker}: {exc}")
        if failures:
            raise MemoError(
                "Secret value was deleted, but legacy metadata could not be removed: "
                + "; ".join(failures)
            )

        self._log_secret_access(secret_row["id"], "delete", name)

    def _legacy_secret_markers(self, secret_id: str, name: str) -> list[Path]:
        """Find old metadata-only files without following symlinks."""
        root = self.cfg.memory_dir / "secrets"
        if not root.is_dir() or root.is_symlink():
            return []
        matches: list[Path] = []
        for path in root.rglob("*.md"):
            if path.is_symlink() or not path.is_file():
                continue
            try:
                if path.stat().st_size > 64 * 1024:
                    continue
                post = frontmatter.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, ValueError):
                continue
            if post.get("type") != "secret":
                continue
            if post.get("id") == secret_id or post.get("name") == name:
                matches.append(path)
        return matches

    def _secure_secret_state_files(self) -> None:
        """Ciphertext and SQLite journals must not be readable by other users."""
        for path in self.cfg.state_dir.glob(f"{self.cfg.db_path.name}*"):
            if path.is_file() and not path.is_symlink():
                path.chmod(0o600)

    def _log_secret_access(self, secret_id: str, op: str, name: str) -> None:
        """Log secret access to grounding.log for audit trail."""
        try:
            grounding_log_path = self.cfg.state_dir / "grounding.log"
            name_hash = hashlib.sha256(name.encode("utf-8")).hexdigest()[:12]
            entry = (
                f"{_now_iso()} | secret_id={secret_id[:8]} | op={op} | name_sha256={name_hash}\n"
            )
            flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
            fd = os.open(grounding_log_path, flags, 0o600)
            try:
                info = os.fstat(fd)
                if not stat.S_ISREG(info.st_mode):
                    raise OSError("secret audit log is not a regular file")
                os.fchmod(fd, 0o600)
                payload = entry.encode()
                while payload:
                    written = os.write(fd, payload)
                    if written <= 0:
                        raise OSError("short write to secret audit log")
                    payload = payload[written:]
            finally:
                os.close(fd)
        except Exception as exc:
            _log.debug("Failed to log secret access: %s", exc)
