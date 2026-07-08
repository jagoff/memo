"""Encryption and key derivation for secret storage."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import secrets
import socket
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

_log = logging.getLogger(__name__)


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


def derive_secret_key() -> bytes:
    """
    Device-bound key derivation.
    Same machine → same key (deterministic, no user input).
    """
    try:
        from consciousness_contracts.uri import device_id as get_device_id

        device_id = get_device_id()
    except Exception:
        device_id = "unknown"

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


def encrypt_secret(value: str) -> tuple[bytes, bytes]:
    """
    Encrypt a secret value with AES-256-GCM.
    Returns: (ciphertext, nonce)
    """
    key = derive_secret_key()
    nonce = secrets.token_bytes(12)  # 96 bits for GCM

    cipher = AESGCM(key)
    ciphertext = cipher.encrypt(nonce, value.encode("utf-8"), None)

    return ciphertext, nonce


def decrypt_secret(ciphertext: bytes, nonce: bytes) -> str:
    """
    Decrypt a secret value.
    Raises cryptography.hazmat.primitives.ciphers.aead.InvalidTag if nonce/ciphertext invalid.
    """
    key = derive_secret_key()
    cipher = AESGCM(key)
    plaintext = cipher.decrypt(nonce, ciphertext, None)
    return plaintext.decode("utf-8")


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


def detect_secrets_llm(
    content: str,
    heuristic_matches: list[tuple[str, float]],
    llm_instance: Any,
) -> list[tuple[str, float]]:
    """
    Ask LLM to confirm/refine heuristic matches.
    Skipped if no heuristic matches or MEMO_DETECT_SECRETS_LLM=False.
    Returns: [(kind, confidence), ...]
    """
    from memo.flags import flag_bool

    if not heuristic_matches or not flag_bool("MEMO_DETECT_SECRETS_LLM"):
        return heuristic_matches

    prompt = f"""
    Analyze the following text snippet for secrets (API keys, passwords, tokens, credentials).

    For each potential secret found, respond with JSON:
    [
        {{"kind": "api_token|password|ssh_key|db_credential|certificate|generic", "confidence": 0.95}},
        ...
    ]

    Text:
    {content[:600]}

    Return only valid JSON array, no other text.
    """

    try:
        response = llm_instance.chat([{"role": "user", "content": prompt}])
        parsed = json.loads(response)
        if isinstance(parsed, list) and all(
            isinstance(item, dict) and "kind" in item and "confidence" in item for item in parsed
        ):
            return [(item["kind"], item["confidence"]) for item in parsed]
    except (json.JSONDecodeError, KeyError, TypeError, AttributeError) as exc:
        _log.debug("LLM detection failed: %s, falling back to heuristic", exc)

    return heuristic_matches
