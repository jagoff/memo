"""Opaque private-key operations for operational Ed25519 signatures."""

from __future__ import annotations

import base64
import hashlib
import json
import os
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
    pending_roster_record: str = ""
    pending_epoch: int | None = None
    pending_authorization_sha256: str = ""
    pending_epoch_authorization: str = ""
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
            "pending_roster_record": self.pending_roster_record,
            "pending_epoch": self.pending_epoch,
            "pending_authorization_sha256": self.pending_authorization_sha256,
            "pending_epoch_authorization": self.pending_epoch_authorization,
        }


class _AuthorityPinProvider(Protocol):
    def _resolve_installation(self, location_binding: str) -> str: ...

    def _read_pin(self, installation_id: str) -> bytes | None: ...

    def _write_pin(self, installation_id: str, value: bytes) -> None: ...


class InMemoryAuthorityPinProvider:
    """Shared persistent test backend for cross-instance authority probes."""

    def __init__(self) -> None:
        self._installations: dict[str, str] = {}
        self._values: dict[str, bytes] = {}
        self._lock = threading.RLock()

    def _resolve_installation(self, location_binding: str) -> str:
        with self._lock:
            installation_id = self._installations.get(location_binding)
            if installation_id is None:
                installation_id = str(uuid.uuid4())
                self._installations[location_binding] = installation_id
            return installation_id

    def _read_pin(self, installation_id: str) -> bytes | None:
        with self._lock:
            value = self._values.get(installation_id)
            return bytes(value) if value is not None else None

    def _write_pin(self, installation_id: str, value: bytes) -> None:
        with self._lock:
            self._values[installation_id] = bytes(value)


class MacOSAuthorityPinProvider:
    """Keychain-backed provider for non-secret monotonic authority metadata."""

    def __init__(self, *, service: str = "com.memo.operational-authority-pin") -> None:
        self.service = service

    def _security(self) -> str:
        executable = shutil.which("security") if sys.platform == "darwin" else None
        if executable is None:
            raise KeyStoreError("macOS Keychain authority metadata is unavailable; failing closed")
        return executable

    def _read_account(self, account: str) -> bytes | None:
        command = [
            self._security(),
            "find-generic-password",
            "-s",
            self.service,
            "-a",
            account,
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

    def _write_account(self, account: str, value: bytes) -> None:
        # The value is authority metadata, not private key material.
        try:
            text = value.decode("utf-8")
        except UnicodeDecodeError:
            raise KeyStoreError("Keychain authority metadata write failed") from None
        command = [
            self._security(),
            "add-generic-password",
            "-U",
            "-s",
            self.service,
            "-a",
            account,
            "-w",
            text,
        ]
        try:
            subprocess.run(
                command,
                check=True,
                capture_output=True,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            raise KeyStoreError("Keychain authority metadata write failed") from None

    def _resolve_installation(self, location_binding: str) -> str:
        binding_digest = hashlib.sha256(location_binding.encode("utf-8")).hexdigest()
        account = f"binding:{binding_digest}"
        lock_path = (
            Path.home()
            / "Library"
            / "Application Support"
            / "Memo"
            / "authority-pin-locks"
            / f"binding-{binding_digest}"
        )
        with authority_write_lock(lock_path):
            raw = self._read_account(account)
            if raw is None:
                raw = str(uuid.uuid4()).encode("ascii")
                self._write_account(account, raw)
                if self._read_account(account) != raw:
                    raise KeyStoreError("Keychain authority binding was not durable")
        try:
            installation_id = str(uuid.UUID(raw.decode("ascii")))
        except (UnicodeDecodeError, ValueError):
            raise KeyStoreError("Keychain authority binding is invalid") from None
        return installation_id

    def _read_pin(self, installation_id: str) -> bytes | None:
        return self._read_account(f"pin:{installation_id}")

    def _write_pin(self, installation_id: str, value: bytes) -> None:
        self._write_account(f"pin:{installation_id}", value)


class AuthorityPinStore:
    """Root-bound crash-recoverable pins for roster and authority epochs."""

    _root: Path
    _provider: _AuthorityPinProvider
    _installation_id: str
    _lock_path: Path

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise TypeError("AuthorityPinStore is created from a canonical authority root")

    @classmethod
    def _create(
        cls,
        root: Path,
        *,
        provider: _AuthorityPinProvider,
    ) -> AuthorityPinStore:
        canonical_root = Path(root).expanduser().resolve()
        location_binding = hashlib.sha256(
            b"memo-authority-root-v1\0" + os.fsencode(canonical_root)
        ).hexdigest()
        installation_id = provider._resolve_installation(location_binding)
        try:
            installation_id = str(uuid.UUID(installation_id))
        except ValueError:
            raise KeyStoreError("authority installation binding is invalid") from None
        instance = object.__new__(cls)
        instance._root = canonical_root
        instance._provider = provider
        instance._installation_id = installation_id
        instance._lock_path = (
            Path.home()
            / "Library"
            / "Application Support"
            / "Memo"
            / "authority-pin-locks"
            / f"pin-{installation_id}"
        )
        return instance

    @classmethod
    def for_root(cls, root: Path) -> AuthorityPinStore:
        """Open the productive authority pin bound to ``root``."""
        return cls._create(root, provider=MacOSAuthorityPinProvider())

    @classmethod
    def _for_test(
        cls,
        root: Path,
        *,
        provider: InMemoryAuthorityPinProvider | None = None,
    ) -> AuthorityPinStore:
        return cls._create(
            root,
            provider=provider or InMemoryAuthorityPinProvider(),
        )

    def _installation_id_for_test(self) -> str:
        return self._installation_id

    def _assert_bound(self, root: Path) -> None:
        if Path(root).expanduser().resolve() != self._root:
            raise KeyStoreError("authority pin store is bound to a different root")

    @staticmethod
    def _roster_metadata(record: bytes) -> tuple[int, str]:
        try:
            body = json.loads(record.decode("utf-8"))
            if (
                not isinstance(body, dict)
                or _canonical(body) != record
                or isinstance(body.get("version"), bool)
                or not isinstance(body.get("version"), int)
                or not isinstance(body.get("roster_hash"), str)
            ):
                raise ValueError
            version = body["version"]
            roster_hash = body["roster_hash"]
        except (UnicodeDecodeError, ValueError, json.JSONDecodeError):
            raise KeyStoreError("prepared roster record is invalid") from None
        if version < 1 or _SHA256_RE.fullmatch(roster_hash) is None:
            raise KeyStoreError("prepared roster record is invalid")
        return version, roster_hash

    @staticmethod
    def _epoch_metadata(record: bytes) -> tuple[int, str]:
        try:
            body = json.loads(record.decode("utf-8"))
            if (
                not isinstance(body, dict)
                or _canonical(body) != record
                or isinstance(body.get("epoch"), bool)
                or not isinstance(body.get("epoch"), int)
            ):
                raise ValueError
            epoch = body["epoch"]
        except (UnicodeDecodeError, ValueError, json.JSONDecodeError):
            raise KeyStoreError("prepared epoch authorization is invalid") from None
        if epoch < 0:
            raise KeyStoreError("prepared epoch authorization is invalid")
        return epoch, hashlib.sha256(record).hexdigest()

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
                pending_roster_record=str(body["pending_roster_record"]),
                pending_epoch=(
                    int(body["pending_epoch"]) if body["pending_epoch"] is not None else None
                ),
                pending_authorization_sha256=str(body["pending_authorization_sha256"]),
                pending_epoch_authorization=str(body["pending_epoch_authorization"]),
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
            or (state.pending_roster_version is None) != (state.pending_roster_record == "")
            or (
                state.pending_roster_hash
                and _SHA256_RE.fullmatch(state.pending_roster_hash) is None
            )
            or (state.pending_epoch is None) != (state.pending_authorization_sha256 == "")
            or (state.pending_epoch is None) != (state.pending_epoch_authorization == "")
            or (
                state.pending_authorization_sha256
                and _SHA256_RE.fullmatch(state.pending_authorization_sha256) is None
            )
        ):
            raise KeyStoreError("authority pin metadata is invalid")
        if state.pending_roster_record:
            version, roster_hash = self._roster_metadata(
                state.pending_roster_record.encode("utf-8")
            )
            if (
                version != state.pending_roster_version
                or roster_hash != state.pending_roster_hash
            ):
                raise KeyStoreError("authority pin metadata is invalid")
        if state.pending_epoch_authorization:
            epoch, authorization_sha256 = self._epoch_metadata(
                state.pending_epoch_authorization.encode("utf-8")
            )
            if (
                epoch != state.pending_epoch
                or authorization_sha256 != state.pending_authorization_sha256
            ):
                raise KeyStoreError("authority pin metadata is invalid")
        return state

    def _read_unlocked(self) -> AuthorityPinState:
        try:
            return self._decode(self._provider._read_pin(self._installation_id))
        except KeyStoreError:
            raise
        except Exception:
            raise KeyStoreError("authority pin metadata read failed") from None

    def _read(self, root: Path) -> AuthorityPinState:
        self._assert_bound(root)
        with authority_write_lock(self._lock_path):
            return self._read_unlocked()

    def _snapshot_for_test(self) -> AuthorityPinState:
        return self._read(self._root)

    def _write_unlocked(self, state: AuthorityPinState) -> AuthorityPinState:
        updated = replace(state, revision=state.revision + 1)
        encoded = _canonical(updated.to_dict())
        try:
            self._provider._write_pin(self._installation_id, encoded)
            persisted = self._provider._read_pin(self._installation_id)
        except KeyStoreError:
            raise
        except Exception:
            raise KeyStoreError("authority pin metadata write failed") from None
        if persisted != encoded:
            raise KeyStoreError("authority pin metadata write was not durable")
        return updated

    def _stage_roster(self, root: Path, record: bytes) -> None:
        self._assert_bound(root)
        version, roster_hash = self._roster_metadata(record)
        record_text = record.decode("utf-8")
        with authority_write_lock(self._lock_path):
            state = self._read_unlocked()
            if (
                state.pending_roster_version == version
                and state.pending_roster_hash == roster_hash
                and state.pending_roster_record == record_text
            ):
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
                    pending_roster_record=record_text,
                    bootstrap_state=(
                        "pending" if state.roster_version == 0 else state.bootstrap_state
                    ),
                )
            )

    def _finish_roster(self, root: Path, record: bytes) -> None:
        self._assert_bound(root)
        version, roster_hash = self._roster_metadata(record)
        record_text = record.decode("utf-8")
        with authority_write_lock(self._lock_path):
            state = self._read_unlocked()
            if state.roster_version == version and state.roster_hash == roster_hash:
                return
            if (
                state.pending_roster_version != version
                or state.pending_roster_hash != roster_hash
                or state.pending_roster_record != record_text
            ):
                raise KeyStoreError("roster commit does not match its pending pin")
            self._write_unlocked(
                replace(
                    state,
                    roster_version=version,
                    roster_hash=roster_hash,
                    pending_roster_version=None,
                    pending_roster_hash="",
                    pending_roster_record="",
                )
            )

    def _verify_roster(self, root: Path, *, version: int, roster_hash: str) -> None:
        self._assert_bound(root)
        with authority_write_lock(self._lock_path):
            state = self._read_unlocked()
            if state.pending_roster_version is not None:
                raise KeyStoreError("roster authority update is incomplete")
            if state.roster_version != version or state.roster_hash != roster_hash:
                raise KeyStoreError("roster rollback or trust-root substitution detected")

    def _prepared_roster(self, root: Path) -> bytes | None:
        state = self._read(root)
        if not state.pending_roster_record:
            return None
        return state.pending_roster_record.encode("utf-8")

    def _stage_epoch(
        self,
        root: Path,
        authorization: bytes,
        *,
        bootstrap: bool,
    ) -> None:
        self._assert_bound(root)
        epoch, authorization_sha256 = self._epoch_metadata(authorization)
        authorization_text = authorization.decode("utf-8")
        with authority_write_lock(self._lock_path):
            state = self._read_unlocked()
            if (
                state.pending_epoch == epoch
                and state.pending_authorization_sha256 == authorization_sha256
                and state.pending_epoch_authorization == authorization_text
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
                    pending_epoch_authorization=authorization_text,
                    bootstrap_state=("epoch_pending" if bootstrap else state.bootstrap_state),
                )
            )

    def _finish_epoch(
        self,
        root: Path,
        authorization: bytes,
        *,
        bootstrap: bool,
    ) -> None:
        self._assert_bound(root)
        epoch, authorization_sha256 = self._epoch_metadata(authorization)
        authorization_text = authorization.decode("utf-8")
        with authority_write_lock(self._lock_path):
            state = self._read_unlocked()
            if state.epoch == epoch and state.authorization_sha256 == authorization_sha256:
                return
            if (
                state.pending_epoch != epoch
                or state.pending_authorization_sha256 != authorization_sha256
                or state.pending_epoch_authorization != authorization_text
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
                    pending_epoch_authorization="",
                    bootstrap_state=("consumed" if bootstrap else state.bootstrap_state),
                )
            )

    def _verify_epoch(
        self,
        root: Path,
        *,
        epoch: int,
        authorization_sha256: str,
    ) -> None:
        self._assert_bound(root)
        with authority_write_lock(self._lock_path):
            state = self._read_unlocked()
            if state.pending_epoch is not None:
                raise KeyStoreError("authority epoch update is incomplete")
            if state.epoch != epoch or state.authorization_sha256 != authorization_sha256:
                raise KeyStoreError("authority epoch rollback detected")

    def _prepared_epoch(self, root: Path) -> tuple[bytes, bool] | None:
        state = self._read(root)
        if not state.pending_epoch_authorization:
            return None
        return (
            state.pending_epoch_authorization.encode("utf-8"),
            state.bootstrap_state == "epoch_pending",
        )


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
