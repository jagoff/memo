"""EXPERIMENTAL — file-level encryption primitives and in-process key manager.

The primitives are tested and the CLI/MCP lock/status surface is gated behind
``MEMO_ENCRYPTION_ENABLED``. Automatic save/search encryption is not wired into
the memory pipeline yet; keep this vertical opt-in until that integration is
complete.

Enables:
- Encrypt sensitive memorias with AES-256-GCM
- Per-memoria encryption (tag-based) or global encryption
- Key derivation from password (PBKDF2)
- Encrypt .md files on disk
- Search over encrypted content (decrypt in memory)

## Encryption Mode

Two encryption modes:
- Per-memoria: Only memorias with `encrypted` tag are encrypted
- Global: All memorias are encrypted

## Key Management

Keys are derived from user password using PBKDF2:
- Salt stored in key derivation file
- Key derivation parameters (iterations, hash)
- Master key used for all encryptions

## File Encryption

Encrypted .md files on disk:
- Original content encrypted with AES-256-GCM
- Auth tag for integrity verification
- Nonce for each encryption operation
- Header with metadata (version, nonce, auth tag)

## Search

Search over encrypted content:
- Decrypt in memory during search
- Re-encrypt after search (optional)
- Or keep decrypted in memory cache
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


@dataclass
class EncryptionMetadata:
    """Metadata for encrypted content."""

    version: str
    nonce: str
    auth_tag: str
    algorithm: str = "AES-256-GCM"


@dataclass
class KeyDerivation:
    """Key derivation parameters."""

    salt: str
    iterations: int
    hash_algorithm: str
    key_length: int


class KeyManager:
    """Manages encryption key derivation and storage.

    Args:
        state_dir: Directory to store key derivation data.
    """

    def __init__(self, state_dir: Path) -> None:
        self.state_dir = state_dir
        self.key_file = state_dir / "encryption_keys.json"
        self._master_key: bytes | None = None

    def derive_key(
        self,
        password: str,
        salt: bytes | None = None,
        iterations: int = 100000,
    ) -> tuple[bytes, bytes, KeyDerivation]:
        """Derive encryption key from password using PBKDF2.

        Args:
            password: User password.
            salt: Optional salt (generated if None).
            iterations: PBKDF2 iterations.

        Returns:
            (key, salt, key_derivation_params).
        """
        if salt is None:
            salt = os.urandom(32)

        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=iterations,
            backend=default_backend(),
        )

        key = kdf.derive(password.encode("utf-8"))

        kd = KeyDerivation(
            salt=salt.hex(),
            iterations=iterations,
            hash_algorithm="SHA256",
            key_length=32,
        )

        return key, salt, kd

    def save_key_derivation(self, kd: KeyDerivation) -> None:
        """Save key derivation parameters to disk.

        Args:
            kd: KeyDerivation to save.
        """
        data = {
            "salt": kd.salt,
            "iterations": kd.iterations,
            "hash_algorithm": kd.hash_algorithm,
            "key_length": kd.key_length,
        }
        self.key_file.write_text(json.dumps(data), encoding="utf-8")

    def load_key_derivation(self) -> KeyDerivation | None:
        """Load key derivation parameters from disk.

        Returns:
            KeyDerivation or None if not found.
        """
        if not self.key_file.is_file():
            return None

        try:
            data = json.loads(self.key_file.read_text(encoding="utf-8"))
            return KeyDerivation(**data)
        except Exception:
            return None

    def set_master_key(self, key: bytes) -> None:
        """Set the master encryption key in memory.

        Args:
            key: The master encryption key.
        """
        self._master_key = key

    def get_master_key(self) -> bytes | None:
        """Get the master encryption key.

        Returns:
            Master key or None if not set.
        """
        return self._master_key

    def clear_master_key(self) -> None:
        """Clear the master key from memory."""
        self._master_key = None


class Encryptor:
    """Encrypts and decrypts memoria content.

    Args:
        key_manager: KeyManager for key management.
    """

    def __init__(self, key_manager: KeyManager) -> None:
        self.key_manager = key_manager

    def encrypt(self, plaintext: str) -> tuple[str, EncryptionMetadata]:
        """Encrypt plaintext content.

        Args:
            plaintext: The content to encrypt.

        Returns:
            (ciphertext_hex, metadata).
        """
        key = self.key_manager.get_master_key()
        if not key:
            raise ValueError("Master key not set. Call unlock() first.")

        aesgcm = AESGCM(key)
        nonce = os.urandom(12)

        ciphertext = aesgcm.encrypt(nonce, plaintext.encode("utf-8"), None)

        # Extract auth tag (last 16 bytes for GCM)
        auth_tag = ciphertext[-16:].hex()
        encrypted_data = ciphertext[:-16].hex()

        metadata = EncryptionMetadata(
            version="1.0",
            nonce=nonce.hex(),
            auth_tag=auth_tag,
        )

        return encrypted_data, metadata

    def decrypt(self, ciphertext_hex: str, metadata: EncryptionMetadata) -> str:
        """Decrypt ciphertext content.

        Args:
            ciphertext_hex: Hex-encoded ciphertext.
            metadata: Encryption metadata.

        Returns:
            Decrypted plaintext.
        """
        key = self.key_manager.get_master_key()
        if not key:
            raise ValueError("Master key not set. Call unlock() first.")

        aesgcm = AESGCM(key)
        nonce = bytes.fromhex(metadata.nonce)
        auth_tag = bytes.fromhex(metadata.auth_tag)
        ciphertext = bytes.fromhex(ciphertext_hex) + auth_tag

        plaintext = aesgcm.decrypt(nonce, ciphertext, None)
        return plaintext.decode("utf-8")

    def encrypt_file(
        self,
        file_path: Path,
        metadata: EncryptionMetadata | None = None,
    ) -> EncryptionMetadata:
        """Encrypt a file in place.

        Args:
            file_path: Path to file to encrypt.
            metadata: Optional metadata (generated if None).

        Returns:
            Encryption metadata.
        """
        plaintext = file_path.read_text(encoding="utf-8")
        ciphertext, meta = self.encrypt(plaintext)

        # Write encrypted file with header
        header = f"ENC::{meta.version}::{meta.nonce}::{meta.auth_tag}::\n"
        file_path.write_text(header + ciphertext, encoding="utf-8")

        return meta

    def decrypt_file(self, file_path: Path) -> tuple[str, EncryptionMetadata]:
        """Decrypt a file in place.

        Args:
            file_path: Path to encrypted file.

        Returns:
            (decrypted_content, metadata).
        """
        content = file_path.read_text(encoding="utf-8")

        if not content.startswith("ENC::"):
            raise ValueError("Not an encrypted file")

        # Parse header
        parts = content.split("::", maxsplit=4)
        if len(parts) < 5:
            raise ValueError("Invalid encrypted file format")

        _, version, nonce, auth_tag, ciphertext = parts

        metadata = EncryptionMetadata(
            version=version,
            nonce=nonce,
            auth_tag=auth_tag,
        )

        plaintext = self.decrypt(ciphertext, metadata)

        # Write decrypted content
        file_path.write_text(plaintext, encoding="utf-8")

        return plaintext, metadata


class EncryptionManager:
    """Manages encryption for the memory vault.

    Args:
        key_manager: KeyManager for key management.
        encryptor: Encryptor for encryption/decryption.
    """

    def __init__(self, key_manager: KeyManager, encryptor: Encryptor) -> None:
        self.key_manager = key_manager
        self.encryptor = encryptor
        self._is_unlocked = False

    def unlock(self, password: str) -> bool:
        """Unlock the vault with password.

        Derives master key from password and stores it in memory.

        Args:
            password: User password.

        Returns:
            True if successful.
        """
        # Load or create key derivation
        kd = self.key_manager.load_key_derivation()
        if kd:
            salt = bytes.fromhex(kd.salt)
            iterations = kd.iterations
        else:
            salt = None
            iterations = 100000

        key, salt, kd = self.key_manager.derive_key(password, salt, iterations)

        # Save key derivation if new
        if not self.key_manager.load_key_derivation():
            self.key_manager.save_key_derivation(kd)

        self.key_manager.set_master_key(key)
        self._is_unlocked = True

        return True

    def lock(self) -> None:
        """Lock the vault by clearing the master key."""
        self.key_manager.clear_master_key()
        self._is_unlocked = False

    def is_unlocked(self) -> bool:
        """Check if the vault is unlocked."""
        return self._is_unlocked

    def encrypt_memoria(self, memoria_id: str, content: str) -> tuple[str, EncryptionMetadata]:
        """Encrypt a memoria's content.

        Args:
            memoria_id: The memoria ID.
            content: The content to encrypt.

        Returns:
            (ciphertext, metadata).
        """
        if not self._is_unlocked:
            raise ValueError("Vault is locked. Call unlock() first.")

        return self.encryptor.encrypt(content)

    def decrypt_memoria(self, ciphertext: str, metadata: EncryptionMetadata) -> str:
        """Decrypt a memoria's content.

        Args:
            ciphertext: Hex-encoded ciphertext.
            metadata: Encryption metadata.

        Returns:
            Decrypted content.
        """
        if not self._is_unlocked:
            raise ValueError("Vault is locked. Call unlock() first.")

        return self.encryptor.decrypt(ciphertext, metadata)


__all__ = [
    "EncryptionManager",
    "EncryptionMetadata",
    "Encryptor",
    "KeyDerivation",
    "KeyManager",
]
