"""Secret operations mixin for Memory facade."""

from __future__ import annotations

import logging
import secrets
from typing import Any

import frontmatter

from memo.errors import MemoError
from memo.memory._base import _MemoryBase
from memo.memory.record import MemoryRecord, _now_iso, _slugify
from memo.secret_store import decrypt_secret, detect_secrets_heuristic, encrypt_secret

_log = logging.getLogger(__name__)


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
        3. Save markdown + index in sqlite
        4. Log access
        """
        from memo.tiers import SECRET_KINDS

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
        ciphertext, nonce = encrypt_secret(value)

        # Create record
        now_iso = _now_iso()
        record_id = f"sec_{secrets.token_hex(8)}"

        frontmatter_data = {
            "id": record_id,
            "type": "secret",
            "kind": kind,
            "name": name,
            "created": now_iso,
            "tags": tags or [],
        }

        markdown_content = "[ENCRYPTED: AES-256-GCM]"
        doc = frontmatter.Post(markdown_content, handler=None, **frontmatter_data)

        # Write markdown
        file_path = (
            self.cfg.memory_dir / "secrets" / now_iso[:7].replace("-", "/") / f"{_slugify(name)}.md"
        )
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(frontmatter.dumps(doc), encoding="utf-8")
        file_path.chmod(0o600)  # Restrict permissions

        # Index in sqlite
        self.store.secret_store_insert(
            id=record_id,
            name=name,
            kind=kind,
            encrypted_blob=ciphertext,
            nonce=nonce,
            created_at=now_iso,
            detection_method="manual",
        )

        # Log access
        self._log_secret_access(record_id, "save", name)

        return MemoryRecord(
            id=record_id,
            path=str(file_path.relative_to(self.cfg.memory_dir)),
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
        secret_row = self.store.secret_store_get(name)
        if not secret_row:
            raise MemoError(f"Secret not found: {name}")

        ciphertext = secret_row["encrypted_blob"]
        nonce = secret_row["nonce"]

        try:
            plaintext = decrypt_secret(ciphertext, nonce)
        except Exception as exc:
            raise MemoError(f"Failed to decrypt secret '{name}': {exc}") from exc

        # Update access metadata
        self.store.secret_store_increment_access(name)
        self._log_secret_access(secret_row["id"], "get", name)

        return plaintext

    def list_secrets(self, kind: str | None = None) -> list[dict[str, Any]]:
        """List secret metadata (names and kinds, not values)."""
        return self.store.secret_store_list(kind=kind)

    def forget_secret(self, name: str) -> None:
        """Delete a secret."""
        secret_row = self.store.secret_store_get(name)
        if not secret_row:
            raise MemoError(f"Secret not found: {name}")

        deleted = self.store.secret_store_delete(name)
        if not deleted:
            raise MemoError(f"Failed to delete secret: {name}")

        self._log_secret_access(secret_row["id"], "delete", name)

    def _log_secret_access(self, secret_id: str, op: str, name: str) -> None:
        """Log secret access to grounding.log for audit trail."""
        try:
            grounding_log_path = self.cfg.state_dir / "grounding.log"
            entry = f"{_now_iso()} | secret_id={secret_id[:8]} | op={op} | name={name}\n"
            grounding_log_path.open("a", encoding="utf-8").write(entry)
        except Exception as exc:
            _log.debug("Failed to log secret access: %s", exc)
