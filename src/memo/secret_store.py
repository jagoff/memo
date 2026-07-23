"""Encryption and key derivation for secret storage."""

from __future__ import annotations

import hashlib
import logging
import os
import platform
import re
import secrets
import socket
import stat
from pathlib import Path

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from memo.errors import MemoError

_log = logging.getLogger(__name__)

_MASTER_KEY_FILENAME = "secret-master.key"
_MASTER_KEY_BYTES = 32
_CIPHERTEXT_V2_PREFIX = b"memo-secret:v2\0"


class SecretKeyError(MemoError):
    """The local secret master key cannot be loaded safely."""


def _resolve_state_dir(state_dir: Path | None) -> Path:
    if state_dir is not None:
        return Path(state_dir)
    from memo.config import Config

    return Config.from_env().state_dir


def _secure_state_dir(state_dir: Path) -> None:
    """Create the local key directory and keep it private to this user."""
    if state_dir.is_symlink():
        raise SecretKeyError(f"Secret state directory must not be a symlink: {state_dir}")
    state_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        state_dir.chmod(0o700)
    except OSError as exc:
        raise SecretKeyError(f"Cannot secure secret state directory {state_dir}: {exc}") from exc


def _read_master_key(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        if path.is_symlink():
            raise SecretKeyError(f"Secret master key must not be a symlink: {path}") from exc
        raise SecretKeyError(f"Cannot open secret master key {path}: {exc}") from exc
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise SecretKeyError(f"Secret master key is not a regular file: {path}")
        if stat.S_IMODE(info.st_mode) & 0o077:
            raise SecretKeyError(
                f"Secret master key permissions are too open at {path}; run chmod 600"
            )
        key = os.read(fd, _MASTER_KEY_BYTES + 1)
    finally:
        os.close(fd)
    if len(key) != _MASTER_KEY_BYTES:
        raise SecretKeyError(
            f"Secret master key at {path} has invalid length; expected {_MASTER_KEY_BYTES} bytes"
        )
    return key


def _write_all(fd: int, payload: bytes) -> None:
    remaining = memoryview(payload)
    while remaining:
        written = os.write(fd, remaining)
        if written <= 0:
            raise OSError("short write while creating secret master key")
        remaining = remaining[written:]


def _load_or_create_master_key(state_dir: Path | None = None) -> bytes:
    root = _resolve_state_dir(state_dir)
    _secure_state_dir(root)
    path = root / _MASTER_KEY_FILENAME
    if path.exists() or path.is_symlink():
        return _read_master_key(path)

    key = secrets.token_bytes(_MASTER_KEY_BYTES)
    scratch = root / (f".{_MASTER_KEY_FILENAME}.{os.getpid()}.{secrets.token_hex(8)}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(scratch, flags, 0o600)
    except OSError as exc:
        raise SecretKeyError(f"Cannot stage secret master key in {root}: {exc}") from exc
    try:
        _write_all(fd, key)
        os.fsync(fd)
    except OSError:
        os.close(fd)
        scratch.unlink(missing_ok=True)
        raise
    else:
        os.close(fd)
    try:
        # Hard-link publication is atomic and never overwrites an existing file
        # or symlink. Concurrent creators either publish once or read the winner.
        os.link(scratch, path, follow_symlinks=False)
    except FileExistsError:
        return _read_master_key(path)
    except OSError as exc:
        raise SecretKeyError(f"Cannot publish secret master key {path}: {exc}") from exc
    finally:
        scratch.unlink(missing_ok=True)
    return _read_master_key(path)


def _load_or_create_machine_salt() -> str:
    """Persist a random salt per machine at ~/.memo/machine.salt."""
    salt_path = Path.home() / ".memo" / "machine.salt"
    if salt_path.exists():
        return salt_path.read_text(encoding="utf-8").strip()

    salt = secrets.token_hex(16)  # 32 hex chars = 128 bits
    salt_path.parent.mkdir(parents=True, exist_ok=True)
    salt_path.write_text(salt, encoding="utf-8")
    salt_path.chmod(0o600)
    _log.debug("Created machine salt at %s", salt_path)
    return salt


def _derive_legacy_secret_key() -> bytes:
    """Derive the pre-v3.5.1 key so existing ciphertext can be migrated."""
    device_id = hashlib.sha256(platform.node().encode()).hexdigest()[:16]
    hostname = socket.gethostname()
    machine_salt = _load_or_create_machine_salt()

    material = f"{hostname}:{device_id}:{machine_salt}".encode()

    # PBKDF2 key derivation using hashlib
    return hashlib.pbkdf2_hmac(
        "sha256",
        material,
        b"memo_secret_v1",
        iterations=100000,
        dklen=32,
    )


def derive_secret_key(*, state_dir: Path | None = None) -> bytes:
    """Load a random 256-bit master key from memo's private local state."""
    return _load_or_create_master_key(state_dir)


def encrypt_secret(
    value: str,
    *,
    state_dir: Path | None = None,
    associated_data: bytes | None = None,
) -> tuple[bytes, bytes]:
    """
    Encrypt a secret value with AES-256-GCM.
    Returns: (ciphertext, nonce)
    """
    key = derive_secret_key(state_dir=state_dir)
    nonce = secrets.token_bytes(12)  # 96 bits for GCM

    cipher = AESGCM(key)
    ciphertext = cipher.encrypt(nonce, value.encode("utf-8"), associated_data)

    return _CIPHERTEXT_V2_PREFIX + ciphertext, nonce


def decrypt_secret(
    ciphertext: bytes,
    nonce: bytes,
    *,
    state_dir: Path | None = None,
    associated_data: bytes | None = None,
) -> str:
    """
    Decrypt a secret value.
    Raises cryptography.hazmat.primitives.ciphers.aead.InvalidTag if nonce/ciphertext invalid.
    """
    payload = bytes(ciphertext)
    if payload.startswith(_CIPHERTEXT_V2_PREFIX):
        key = derive_secret_key(state_dir=state_dir)
        payload = payload[len(_CIPHERTEXT_V2_PREFIX) :]
        aad = associated_data
    else:
        # Compatibility path for records created by the old predictable
        # hostname/device/salt derivation. SecretOps rotates these on access.
        key = _derive_legacy_secret_key()
        aad = None
    cipher = AESGCM(key)
    plaintext = cipher.decrypt(nonce, payload, aad)
    return plaintext.decode("utf-8")


def secret_ciphertext_needs_rotation(ciphertext: bytes) -> bool:
    """Return True for ciphertext written by the legacy derivation scheme."""
    return not bytes(ciphertext).startswith(_CIPHERTEXT_V2_PREFIX)


# Detection patterns
HEURISTIC_PATTERNS: dict[str, re.Pattern] = {
    "api_token": re.compile(
        r"(sk_[a-z0-9_-]{20,}|token[=:]\s*[a-z0-9]{32,}|"
        r"api[_-]?key[=:]\s*\S+|ghp_[a-z0-9]{36,})",
        re.IGNORECASE,
    ),
    "password": re.compile(
        r"(password[=:]\s*\S+|passwd\s*:\s*\S+|pwd[=:]\s*\S+)",
        re.IGNORECASE,
    ),
    "ssh_key": re.compile(r"(-----BEGIN [A-Z]+ PRIVATE KEY|-----BEGIN RSA PRIVATE KEY)"),
    "db_credential": re.compile(
        r"(postgres://|mysql://|mongodb://|postgresql://|user[=:]\s*\w+.*password[=:]\s*\S+)"
    ),
}


def detect_secrets_heuristic(content: str) -> list[tuple[str, float]]:
    """
    Fast regex-based detection.
    Returns: [(kind, confidence), ...] with confidence 0.7 for heuristic matches.
    """
    found = []
    for kind, pattern in HEURISTIC_PATTERNS.items():
        if pattern.search(content):
            found.append((kind, 0.7))
    return found
