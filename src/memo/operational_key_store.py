"""Opaque private-key operations for operational Ed25519 signatures."""

from __future__ import annotations

import base64
import hashlib
import json
import re
import shutil
import subprocess
import sys
import threading
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal, Protocol

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from memo.atomic_io import authority_write_lock

_DEVICE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_SIGNATURE_PREFIX = b"memo-signature-v1\0"
_BOOTSTRAP_DOMAIN = "memo.operational.roster.bootstrap.v1"
_AUTHORITY_PIN_SCHEMA = "memo.operational_authority_pin.v1"
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


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
class AuthorityPinState:
    """Non-secret monotonic authority metadata stored outside the state root."""

    revision: int = 0
    roster_version: int = 0
    roster_hash: str = ""
    epoch: int | None = None
    authorization_sha256: str = ""
    bootstrap_state: Literal["absent", "pending", "epoch_pending", "consumed"] = "absent"
    pending_roster_version: int | None = None
    pending_roster_hash: str = ""
    pending_epoch: int | None = None
    pending_authorization_sha256: str = ""
    schema: Literal["memo.operational_authority_pin.v1"] = "memo.operational_authority_pin.v1"

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "revision": self.revision,
            "roster_version": self.roster_version,
            "roster_hash": self.roster_hash,
            "epoch": self.epoch,
            "authorization_sha256": self.authorization_sha256,
            "bootstrap_state": self.bootstrap_state,
            "pending_roster_version": self.pending_roster_version,
            "pending_roster_hash": self.pending_roster_hash,
            "pending_epoch": self.pending_epoch,
            "pending_authorization_sha256": self.pending_authorization_sha256,
        }


class _AuthorityPinProvider(Protocol):
    def read(self, authority_id: str) -> bytes | None: ...

    def write(self, authority_id: str, value: bytes) -> None: ...


class InMemoryAuthorityPinProvider:
    """Shared persistent test backend for cross-instance authority probes."""

    def __init__(self) -> None:
        self._values: dict[str, bytes] = {}
        self._lock = threading.RLock()

    def read(self, authority_id: str) -> bytes | None:
        with self._lock:
            value = self._values.get(authority_id)
            return bytes(value) if value is not None else None

    def write(self, authority_id: str, value: bytes) -> None:
        with self._lock:
            self._values[authority_id] = bytes(value)


class MacOSAuthorityPinProvider:
    """Keychain-backed provider for non-secret monotonic authority metadata."""

    def __init__(self, *, service: str = "com.memo.operational-authority-pin") -> None:
        self.service = service

    def _security(self) -> str:
        executable = shutil.which("security") if sys.platform == "darwin" else None
        if executable is None:
            raise KeyStoreError("macOS Keychain authority metadata is unavailable; failing closed")
        return executable

    def read(self, authority_id: str) -> bytes | None:
        command = [
            self._security(),
            "find-generic-password",
            "-s",
            self.service,
            "-a",
            authority_id,
            "-w",
        ]
        try:
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            raise KeyStoreError("Keychain authority metadata read failed") from None
        if result.returncode == 44 or b"could not be found" in result.stderr:
            return None
        if result.returncode != 0:
            raise KeyStoreError("Keychain authority metadata read failed")
        return result.stdout.rstrip(b"\n")

    def write(self, authority_id: str, value: bytes) -> None:
        # The value is signed-hash metadata, not private key material.
        command = [
            self._security(),
            "add-generic-password",
            "-U",
            "-s",
            self.service,
            "-a",
            authority_id,
            "-w",
            value.decode("ascii"),
        ]
        try:
            subprocess.run(
                command,
                check=True,
                capture_output=True,
                timeout=10,
            )
        except (OSError, UnicodeDecodeError, subprocess.SubprocessError):
            raise KeyStoreError("Keychain authority metadata write failed") from None


class AuthorityPinStore:
    """Crash-recoverable monotonic pins for roster and authority epochs."""

    def __init__(
        self,
        *,
        authority_id: str,
        provider: _AuthorityPinProvider | None = None,
    ) -> None:
        if not authority_id or len(authority_id) > 512:
            raise ValueError("authority_id is invalid")
        self.authority_id = authority_id
        self._provider = provider or MacOSAuthorityPinProvider()
        lock_name = hashlib.sha256(authority_id.encode("utf-8")).hexdigest()
        self._lock_path = (
            Path.home()
            / "Library"
            / "Application Support"
            / "Memo"
            / "authority-pin-locks"
            / lock_name
        )

    @classmethod
    def in_memory(
        cls,
        *,
        authority_id: str,
        provider: InMemoryAuthorityPinProvider | None = None,
    ) -> AuthorityPinStore:
        return cls(
            authority_id=authority_id,
            provider=provider or InMemoryAuthorityPinProvider(),
        )

    def _decode(self, raw: bytes | None) -> AuthorityPinState:
        if raw is None:
            return AuthorityPinState()
        try:
            body = json.loads(raw.decode("utf-8"))
            expected = set(AuthorityPinState().to_dict())
            if not isinstance(body, dict) or set(body) != expected:
                raise ValueError
            if body["schema"] != _AUTHORITY_PIN_SCHEMA:
                raise ValueError
            state = AuthorityPinState(
                revision=int(body["revision"]),
                roster_version=int(body["roster_version"]),
                roster_hash=str(body["roster_hash"]),
                epoch=(int(body["epoch"]) if body["epoch"] is not None else None),
                authorization_sha256=str(body["authorization_sha256"]),
                bootstrap_state=str(body["bootstrap_state"]),  # type: ignore[arg-type]
                pending_roster_version=(
                    int(body["pending_roster_version"])
                    if body["pending_roster_version"] is not None
                    else None
                ),
                pending_roster_hash=str(body["pending_roster_hash"]),
                pending_epoch=(
                    int(body["pending_epoch"]) if body["pending_epoch"] is not None else None
                ),
                pending_authorization_sha256=str(body["pending_authorization_sha256"]),
            )
        except (KeyError, TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError):
            raise KeyStoreError("authority pin metadata is invalid") from None
        if (
            state.schema != _AUTHORITY_PIN_SCHEMA
            or state.revision < 0
            or state.roster_version < 0
            or (state.roster_version == 0) != (state.roster_hash == "")
            or (state.roster_hash != "" and _SHA256_RE.fullmatch(state.roster_hash) is None)
            or (state.epoch is None) != (state.authorization_sha256 == "")
            or (
                state.authorization_sha256
                and _SHA256_RE.fullmatch(state.authorization_sha256) is None
            )
            or state.bootstrap_state not in {"absent", "pending", "epoch_pending", "consumed"}
            or (state.pending_roster_version is None) != (state.pending_roster_hash == "")
            or (
                state.pending_roster_hash
                and _SHA256_RE.fullmatch(state.pending_roster_hash) is None
            )
            or (state.pending_epoch is None) != (state.pending_authorization_sha256 == "")
            or (
                state.pending_authorization_sha256
                and _SHA256_RE.fullmatch(state.pending_authorization_sha256) is None
            )
        ):
            raise KeyStoreError("authority pin metadata is invalid")
        return state

    def _read_unlocked(self) -> AuthorityPinState:
        try:
            return self._decode(self._provider.read(self.authority_id))
        except KeyStoreError:
            raise
        except Exception:
            raise KeyStoreError("authority pin metadata read failed") from None

    def read(self) -> AuthorityPinState:
        with authority_write_lock(self._lock_path):
            return self._read_unlocked()

    def _write_unlocked(self, state: AuthorityPinState) -> AuthorityPinState:
        updated = replace(state, revision=state.revision + 1)
        encoded = _canonical(updated.to_dict())
        try:
            self._provider.write(self.authority_id, encoded)
            persisted = self._provider.read(self.authority_id)
        except KeyStoreError:
            raise
        except Exception:
            raise KeyStoreError("authority pin metadata write failed") from None
        if persisted != encoded:
            raise KeyStoreError("authority pin metadata write was not durable")
        return updated

    def prepare_roster(self, *, version: int, roster_hash: str) -> None:
        if version < 1 or _SHA256_RE.fullmatch(roster_hash) is None:
            raise KeyStoreError("roster pin is invalid")
        with authority_write_lock(self._lock_path):
            state = self._read_unlocked()
            if state.pending_roster_version == version and state.pending_roster_hash == roster_hash:
                return
            if state.pending_roster_version is not None:
                raise KeyStoreError("a different roster pin update is pending")
            if version != state.roster_version + 1:
                raise KeyStoreError("roster pin must advance exactly once")
            if state.roster_version == 0 and state.bootstrap_state != "absent":
                raise KeyStoreError("authority bootstrap pin state is invalid")
            self._write_unlocked(
                replace(
                    state,
                    pending_roster_version=version,
                    pending_roster_hash=roster_hash,
                    bootstrap_state=(
                        "pending" if state.roster_version == 0 else state.bootstrap_state
                    ),
                )
            )

    def commit_roster(self, *, version: int, roster_hash: str) -> None:
        with authority_write_lock(self._lock_path):
            state = self._read_unlocked()
            if state.roster_version == version and state.roster_hash == roster_hash:
                return
            if state.pending_roster_version != version or state.pending_roster_hash != roster_hash:
                raise KeyStoreError("roster commit does not match its pending pin")
            self._write_unlocked(
                replace(
                    state,
                    roster_version=version,
                    roster_hash=roster_hash,
                    pending_roster_version=None,
                    pending_roster_hash="",
                )
            )

    def verify_roster(self, *, version: int, roster_hash: str) -> None:
        with authority_write_lock(self._lock_path):
            state = self._read_unlocked()
            if state.pending_roster_version is not None:
                if (
                    state.pending_roster_version != version
                    or state.pending_roster_hash != roster_hash
                ):
                    raise KeyStoreError("roster authority update is incomplete")
                self._write_unlocked(
                    replace(
                        state,
                        roster_version=version,
                        roster_hash=roster_hash,
                        pending_roster_version=None,
                        pending_roster_hash="",
                    )
                )
                return
            if state.roster_version != version or state.roster_hash != roster_hash:
                raise KeyStoreError("roster rollback or trust-root substitution detected")

    def prepare_epoch(
        self,
        *,
        epoch: int,
        authorization_sha256: str,
        bootstrap: bool,
    ) -> None:
        if epoch < 0 or _SHA256_RE.fullmatch(authorization_sha256) is None:
            raise KeyStoreError("epoch pin is invalid")
        with authority_write_lock(self._lock_path):
            state = self._read_unlocked()
            if (
                state.pending_epoch == epoch
                and state.pending_authorization_sha256 == authorization_sha256
            ):
                return
            if state.pending_epoch is not None:
                raise KeyStoreError("a different epoch pin update is pending")
            if bootstrap:
                if epoch != 0 or state.epoch is not None or state.bootstrap_state != "pending":
                    raise KeyStoreError("authority epoch bootstrap is not pending")
            elif state.epoch is None or epoch <= state.epoch or state.bootstrap_state != "consumed":
                raise KeyStoreError("authority epoch pin must increase")
            self._write_unlocked(
                replace(
                    state,
                    pending_epoch=epoch,
                    pending_authorization_sha256=authorization_sha256,
                    bootstrap_state=("epoch_pending" if bootstrap else state.bootstrap_state),
                )
            )

    def commit_epoch(
        self,
        *,
        epoch: int,
        authorization_sha256: str,
        bootstrap: bool,
    ) -> None:
        with authority_write_lock(self._lock_path):
            state = self._read_unlocked()
            if state.epoch == epoch and state.authorization_sha256 == authorization_sha256:
                return
            if (
                state.pending_epoch != epoch
                or state.pending_authorization_sha256 != authorization_sha256
            ):
                raise KeyStoreError("epoch commit does not match its pending pin")
            if bootstrap and state.bootstrap_state != "epoch_pending":
                raise KeyStoreError("authority bootstrap consumption is invalid")
            self._write_unlocked(
                replace(
                    state,
                    epoch=epoch,
                    authorization_sha256=authorization_sha256,
                    pending_epoch=None,
                    pending_authorization_sha256="",
                    bootstrap_state=("consumed" if bootstrap else state.bootstrap_state),
                )
            )

    def verify_epoch(self, *, epoch: int, authorization_sha256: str) -> None:
        with authority_write_lock(self._lock_path):
            state = self._read_unlocked()
            if state.pending_epoch is not None:
                if (
                    state.pending_epoch != epoch
                    or state.pending_authorization_sha256 != authorization_sha256
                ):
                    raise KeyStoreError("authority epoch update is incomplete")
                bootstrap = state.bootstrap_state == "epoch_pending"
                self._write_unlocked(
                    replace(
                        state,
                        epoch=epoch,
                        authorization_sha256=authorization_sha256,
                        pending_epoch=None,
                        pending_authorization_sha256="",
                        bootstrap_state=("consumed" if bootstrap else state.bootstrap_state),
                    )
                )
                return
            if state.epoch != epoch or state.authorization_sha256 != authorization_sha256:
                raise KeyStoreError("authority epoch rollback detected")


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
        if not roles or len(roles) != len(set(roles)) or not set(roles).issubset(allowed_roles):
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
            _SIGNATURE_PREFIX + _BOOTSTRAP_DOMAIN.encode("ascii") + b"\0" + roster_payload
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
    "AuthorityPinState",
    "AuthorityPinStore",
    "DeviceKeyStore",
    "InMemoryAuthorityPinProvider",
    "InMemoryKeyProvider",
    "KeyStoreError",
    "MacOSAuthorityPinProvider",
    "MacOSKeychainProvider",
    "PublicKeyRecord",
]
