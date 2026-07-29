"""Opaque private-key operations for operational Ed25519 signatures."""

from __future__ import annotations

import base64
import hashlib
import json
import re
import uuid
from dataclasses import dataclass, field, replace
from typing import Protocol

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

_DEVICE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_SIGNATURE_PREFIX = b"memo-signature-v1\0"
_BOOTSTRAP_DOMAIN = "memo.operational.roster.bootstrap.v1"


def _b64url(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


class KeyStoreError(RuntimeError):
    """An opaque operational key operation is unavailable."""


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
    _bootstrap_signature: str = field(default="", repr=False, compare=False)

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
        return b"memo-key-enrollment-v1\0" + _canonical(body)

    def to_dict(self) -> dict[str, object]:
        return {
            "device_id": self.device_id,
            "key_id": self.key_id,
            "fingerprint": self.fingerprint,
            "public_key": self.public_key,
            "roles": list(self.roles),
            "enrollment_sequence": self.enrollment_sequence,
            "revocation_sequence": self.revocation_sequence,
            "proof_of_possession": self.proof_of_possession,
        }


def _bootstrap_roster_bodies(
    key: PublicKeyRecord,
) -> tuple[dict[str, object], bytes]:
    body: dict[str, object] = {
        "schema": "memo.operational_roster.v1",
        "version": 1,
        "peers": [key.device_id],
        "keys": [key.to_dict()],
        "local_device_id": key.device_id,
        "created_at": "",
        "previous_roster_hash": "",
        "roster_hash": "",
        "signature": "",
    }
    digest = hashlib.sha256(_canonical(body)).hexdigest()
    body["roster_hash"] = digest
    return body, _canonical(body)


class _PrivateKeyProvider(Protocol):
    """Only opaque handle operations cross this boundary."""

    def generate(self, key_id: str) -> bytes: ...

    def sign(self, key_id: str, payload: bytes) -> bytes: ...

    def destroy(self, key_id: str) -> None: ...


class InMemoryKeyProvider:
    """Ephemeral test provider retaining key objects, never serialized seeds."""

    def __init__(self) -> None:
        self._keys: dict[str, Ed25519PrivateKey] = {}

    def generate(self, key_id: str) -> bytes:
        if key_id in self._keys:
            raise KeyStoreError(f"duplicate private key id: {key_id}")
        key = Ed25519PrivateKey.generate()
        self._keys[key_id] = key
        return key.public_key().public_bytes_raw()

    def sign(self, key_id: str, payload: bytes) -> bytes:
        try:
            key = self._keys[key_id]
        except KeyError as exc:
            raise KeyStoreError(f"unknown private key id: {key_id}") from exc
        return key.sign(bytes(payload))

    def destroy(self, key_id: str) -> None:
        if self._keys.pop(key_id, None) is None:
            raise KeyStoreError(f"unknown private key id: {key_id}")


class MacOSKeychainProvider:
    """Fail-closed placeholder for a non-exportable Ed25519 Keychain backend.

    The ``security`` generic-password CLI cannot satisfy this contract: it
    exports secret bytes and exposes them through argv/stdout. Production setup
    therefore fails until Memo has a native SecKey-backed implementation.
    """

    _MESSAGE = (
        "non-exportable Ed25519 Keychain operations are unavailable; "
        "refusing exportable generic-password fallback"
    )

    def __init__(self, *, service: str = "com.memo.operational-signing") -> None:
        self.service = service

    def generate(self, key_id: str) -> bytes:
        del key_id
        raise KeyStoreError(self._MESSAGE)

    def sign(self, key_id: str, payload: bytes) -> bytes:
        del key_id, payload
        raise KeyStoreError(self._MESSAGE)

    def destroy(self, key_id: str) -> None:
        del key_id
        raise KeyStoreError(self._MESSAGE)


class DeviceKeyStore:
    """Creates public records while delegating all private operations."""

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
        key_id = f"ed25519-{uuid.uuid4().hex}"
        public_bytes = self._provider.generate(key_id)
        if len(public_bytes) != 32:
            raise KeyStoreError("provider returned an invalid public key")
        fingerprint = hashlib.sha256(public_bytes).hexdigest()
        record = PublicKeyRecord(
            device_id=device_id,
            key_id=key_id,
            fingerprint=fingerprint,
            public_key=_b64url(public_bytes),
            roles=roles,
            enrollment_sequence=enrollment_sequence,
        )
        proof = self._provider.sign(key_id, record.proof_payload())
        record = replace(record, proof_of_possession=_b64url(proof))
        _, roster_payload = _bootstrap_roster_bodies(record)
        bootstrap_signed = (
            _SIGNATURE_PREFIX
            + _BOOTSTRAP_DOMAIN.encode("ascii")
            + b"\0"
            + roster_payload
        )
        bootstrap_signature = self._provider.sign(key_id, bootstrap_signed)
        return replace(record, _bootstrap_signature=_b64url(bootstrap_signature))

    def sign(self, *, key_id: str, payload: bytes) -> bytes:
        try:
            return self._provider.sign(key_id, bytes(payload))
        except KeyStoreError:
            raise
        except Exception:
            raise KeyStoreError("opaque signing operation failed") from None

    def destroy(self, *, key_id: str) -> None:
        try:
            self._provider.destroy(key_id)
        except KeyStoreError:
            raise
        except Exception:
            raise KeyStoreError("opaque key destruction failed") from None


__all__ = [
    "DeviceKeyStore",
    "InMemoryKeyProvider",
    "KeyStoreError",
    "MacOSKeychainProvider",
    "PublicKeyRecord",
]
