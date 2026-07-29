"""Private Ed25519 key isolation for operational signatures."""

from __future__ import annotations

import base64
import hashlib
import json
import re
import subprocess
import uuid
from dataclasses import dataclass, replace
from typing import Protocol

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

_DEVICE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


def _b64url(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _decode_b64url(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


class KeyStoreError(RuntimeError):
    """A private operational key is missing or unavailable."""


@dataclass(frozen=True)
class PublicKeyRecord:
    device_id: str
    key_id: str
    fingerprint: str
    public_key: str
    roles: tuple[str, ...]
    enrollment_sequence: int
    revocation_sequence: int | None = None
    proof_of_possession: str = ""

    def proof_payload(self) -> bytes:
        body = {
            "device_id": self.device_id,
            "enrollment_sequence": self.enrollment_sequence,
            "fingerprint": self.fingerprint,
            "key_id": self.key_id,
            "public_key": self.public_key,
            "revocation_sequence": self.revocation_sequence,
            "roles": list(self.roles),
        }
        encoded = json.dumps(
            body,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        return b"memo-key-enrollment-v1\0" + encoded


class _PrivateKeyProvider(Protocol):
    def put(self, key_id: str, private_key: bytes) -> None: ...

    def get(self, key_id: str) -> bytes: ...

    def destroy(self, key_id: str) -> None: ...


class InMemoryKeyProvider:
    """Ephemeral provider used only by tests and isolated rehearsal."""

    def __init__(self) -> None:
        self._keys: dict[str, bytes] = {}

    def put(self, key_id: str, private_key: bytes) -> None:
        if key_id in self._keys:
            raise KeyStoreError(f"duplicate private key id: {key_id}")
        self._keys[key_id] = bytes(private_key)

    def get(self, key_id: str) -> bytes:
        try:
            return self._keys[key_id]
        except KeyError as exc:
            raise KeyStoreError(f"unknown private key id: {key_id}") from exc

    def destroy(self, key_id: str) -> None:
        if self._keys.pop(key_id, None) is None:
            raise KeyStoreError(f"unknown private key id: {key_id}")


class MacOSKeychainProvider:
    """macOS Keychain-backed application-secret provider."""

    def __init__(self, *, service: str = "com.memo.operational-signing") -> None:
        self.service = service

    def put(self, key_id: str, private_key: bytes) -> None:
        encoded = _b64url(private_key)
        command = [
            "security",
            "add-generic-password",
            "-U",
            "-s",
            self.service,
            "-a",
            key_id,
            "-w",
            encoded,
        ]
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
        except (OSError, subprocess.CalledProcessError) as exc:
            raise KeyStoreError("unable to store operational key in Keychain") from exc

    def get(self, key_id: str) -> bytes:
        command = [
            "security",
            "find-generic-password",
            "-s",
            self.service,
            "-a",
            key_id,
            "-w",
        ]
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
        except (OSError, subprocess.CalledProcessError) as exc:
            raise KeyStoreError(f"unknown private key id: {key_id}") from exc
        return _decode_b64url(result.stdout.strip())

    def destroy(self, key_id: str) -> None:
        command = [
            "security",
            "delete-generic-password",
            "-s",
            self.service,
            "-a",
            key_id,
        ]
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
        except (OSError, subprocess.CalledProcessError) as exc:
            raise KeyStoreError(f"unknown private key id: {key_id}") from exc


class DeviceKeyStore:
    """Creates public records while keeping private seeds behind a provider."""

    def __init__(self, provider: _PrivateKeyProvider | None = None) -> None:
        self._provider = provider or MacOSKeychainProvider()

    @classmethod
    def in_memory(cls) -> DeviceKeyStore:
        return cls(InMemoryKeyProvider())

    def generate(
        self,
        *,
        device_id: str,
        roles: tuple[str, ...] = ("origin", "migration_attestor"),
        enrollment_sequence: int = 1,
    ) -> PublicKeyRecord:
        if not _DEVICE_ID_RE.fullmatch(device_id):
            raise ValueError("device_id is unsafe")
        allowed_roles = {"origin", "migration_attestor"}
        if not roles or len(roles) != len(set(roles)) or not set(roles).issubset(
            allowed_roles
        ):
            raise ValueError("roles must be unique operational signing roles")
        if enrollment_sequence < 1:
            raise ValueError("enrollment_sequence must be positive")
        private_key = Ed25519PrivateKey.generate()
        public_bytes = private_key.public_key().public_bytes_raw()
        fingerprint = hashlib.sha256(public_bytes).hexdigest()
        key_id = f"ed25519-{fingerprint[:16]}-{uuid.uuid4().hex[:8]}"
        record = PublicKeyRecord(
            device_id=device_id,
            key_id=key_id,
            fingerprint=fingerprint,
            public_key=_b64url(public_bytes),
            roles=roles,
            enrollment_sequence=enrollment_sequence,
        )
        private_bytes = private_key.private_bytes_raw()
        self._provider.put(key_id, private_bytes)
        proof = private_key.sign(record.proof_payload())
        return replace(record, proof_of_possession=_b64url(proof))

    def sign(self, *, key_id: str, payload: bytes) -> bytes:
        private_bytes = self._provider.get(key_id)
        try:
            key = Ed25519PrivateKey.from_private_bytes(private_bytes)
            return key.sign(bytes(payload))
        except ValueError as exc:
            raise KeyStoreError(f"invalid private key material for {key_id}") from exc

    def destroy(self, *, key_id: str) -> None:
        self._provider.destroy(key_id)


__all__ = [
    "DeviceKeyStore",
    "InMemoryKeyProvider",
    "KeyStoreError",
    "MacOSKeychainProvider",
    "PublicKeyRecord",
]
