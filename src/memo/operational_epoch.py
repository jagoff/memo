"""Durable, authenticated authority epochs and request-level fencing."""

from __future__ import annotations

import hashlib
import json
import os
import weakref
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import RLock
from typing import Any, Literal, NoReturn, SupportsIndex

from memo.atomic_io import (
    authority_admission_lock,
    authority_write_lock,
    open_secure_directory,
)
from memo.errors import AuthorityEpochError, OperationalError
from memo.identity import PrincipalIdentity
from memo.operational_event import (
    EpochMarkerAuthorization,
    MigrationOrigin,
    canonical_json_bytes,
    canonical_signed_bytes,
)
from memo.operational_key_store import AuthorityPinStore, KeyStoreError
from memo.operational_roster import RosterError, VerificationRoster
from memo.operational_signing import (
    OperationalSigner,
    OperationalVerifier,
    SignatureEnvelope,
)

_MARKER_SCHEMA = "memo.operational_authority_epoch.v1"
_HIGH_WATERMARK_SCHEMA = "memo.operational_authority_epoch_high_watermark.v1"
_AUTHORIZATION_SCHEMA = "memo.operational_epoch_authorization.v1"
_SYSTEM_CAPABILITY_SCHEMA = "memo.operational_system_capability.v1"
_SYSTEM_CAPABILITY_DOMAIN = "memo.operational.system_capability.v1"
_SYSTEM_CAPABILITY_FIELDS = frozenset(
    {
        "schema",
        "authority_id",
        "authority_root_sha256",
        "process_nonce",
        "fence_nonce",
        "system_role",
        "device_id",
        "roster_version",
        "roster_hash",
        "key_id",
    }
)
_SYSTEM_ROLES = frozenset({"daemon", "migration"})
# This is a freshness challenge, not a bearer secret. Every proof is also
# signed by an enrolled operational key and bound to one concrete fence.
_PROCESS_NONCE = os.urandom(32)


@dataclass(frozen=True)
class CommitContext:
    identity: PrincipalIdentity
    authority_epoch: int
    control_oid: str
    origin_device: str
    migration_origin: MigrationOrigin | None = None


class SystemCapability:
    """A signed fence-local proof created only while invoking a bound operation."""

    __slots__ = ("__operation", "__payload", "__signature")

    def __new__(cls) -> SystemCapability:
        del cls
        raise TypeError("SystemCapability is created only by authenticated composition")


SystemContextOperation = Callable[[PrincipalIdentity], CommitContext]
SystemRole = Literal["daemon", "migration"]


class _BoundSystemContext:
    """Opaque callable carrying immutable proof bytes, never a raw capability."""

    __slots__ = ("__fence_ref", "__payload", "__signature")

    def __init__(
        self,
        fence: EpochFence,
        payload: bytes,
        signature: SignatureEnvelope,
    ) -> None:
        self.__fence_ref = weakref.ref(fence)
        self.__payload = bytes(payload)
        self.__signature = signature

    def __call__(self, identity: PrincipalIdentity) -> CommitContext:
        fence = self.__fence_ref()
        if fence is None:
            raise AuthorityEpochError("bound system capability fence no longer exists")
        capability = object.__new__(SystemCapability)
        object.__setattr__(
            capability,
            "_SystemCapability__operation",
            self,
        )
        object.__setattr__(
            capability,
            "_SystemCapability__payload",
            self.__payload,
        )
        object.__setattr__(
            capability,
            "_SystemCapability__signature",
            self.__signature,
        )
        token = _ACTIVE_SYSTEM_OPERATION.set(self)
        try:
            return fence.system_context(identity, capability=capability)
        finally:
            _ACTIVE_SYSTEM_OPERATION.reset(token)

    def _authorizes(
        self,
        fence: EpochFence,
        payload: bytes,
        signature: SignatureEnvelope,
    ) -> bool:
        return (
            self.__fence_ref() is fence
            and self.__payload == payload
            and self.__signature == signature
        )


_ACTIVE_SYSTEM_OPERATION: ContextVar[_BoundSystemContext | None] = ContextVar(
    "memo_active_system_operation",
    default=None,
)


def _fsync_directory(path: Path) -> None:
    with open_secure_directory(path) as directory:
        os.fsync(directory.descriptor)


def atomic_write_text(destination: Path, text: str) -> None:
    """Write one epoch authority document through the strict no-follow path."""
    destination = Path(destination)
    with open_secure_directory(destination.parent, create=True) as directory:
        directory.atomic_write_bytes(
            destination.name,
            text.encode("utf-8"),
        )


def _authority_relative(root: Path, path: Path) -> Path:
    try:
        relative = Path(path).absolute().relative_to(Path(root).absolute())
    except ValueError as exc:
        raise AuthorityEpochError(f"authority epoch path escapes root: {path}") from exc
    if not relative.parts:
        raise AuthorityEpochError("authority epoch file path is required")
    return relative


def _canonical_text(value: object) -> str:
    return canonical_json_bytes(value).decode("utf-8")


def _sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _authority_root_sha256(fence: EpochFence) -> str:
    canonical_root = fence.root.expanduser().absolute()
    return hashlib.sha256(
        b"memo-operational-authority-root-v1\0" + os.fsencode(canonical_root)
    ).hexdigest()


def _authority_id(fence: EpochFence) -> str:
    try:
        authority_id = object.__getattribute__(
            fence.pin_store,
            "_installation_id",
        )
    except (AttributeError, TypeError):
        raise AuthorityEpochError("authority installation identity is unavailable") from None
    if not isinstance(authority_id, str) or not authority_id:
        raise AuthorityEpochError("authority installation identity is invalid")
    return authority_id


class _FenceInstanceNonceStore:
    """Assign one immutable CSPRNG nonce to each live fence instance."""

    __slots__ = ("__lock", "__nonces")

    def __init__(self) -> None:
        self.__lock = RLock()
        self.__nonces: weakref.WeakKeyDictionary[object, bytes] = weakref.WeakKeyDictionary()

    def allocate(self, fence: EpochFence) -> None:
        with self.__lock:
            if fence in self.__nonces:
                raise RuntimeError("EpochFence instance nonce is already allocated")
            self.__nonces[fence] = os.urandom(32)

    def read(self, fence: EpochFence) -> bytes | None:
        with self.__lock:
            return self.__nonces.get(fence)

    def __get__(
        self,
        instance: EpochFence | None,
        owner: type[EpochFence] | None = None,
    ) -> bytes | _FenceInstanceNonceStore:
        del owner
        if instance is None:
            return self
        nonce = self.read(instance)
        if nonce is None:
            raise AttributeError("EpochFence instance nonce is unavailable")
        return nonce

    def __set__(self, instance: EpochFence, value: object) -> NoReturn:
        del instance, value
        raise AttributeError("EpochFence instance nonce is immutable")

    def __delete__(self, instance: EpochFence) -> NoReturn:
        del instance
        raise AttributeError("EpochFence instance nonce is immutable")


_FENCE_INSTANCE_NONCES = _FenceInstanceNonceStore()


def _fence_nonce(fence: EpochFence) -> str:
    nonce = _FENCE_INSTANCE_NONCES.read(fence)
    if nonce is None:
        raise AuthorityEpochError("fence instance nonce is unavailable") from None
    if type(nonce) is not bytes or len(nonce) != 32:
        raise AuthorityEpochError("fence instance nonce is invalid")
    return nonce.hex()


def _system_capability_claims(
    fence: EpochFence,
    *,
    roster: VerificationRoster,
    key_id: str,
    system_role: str,
) -> dict[str, object]:
    return {
        "schema": _SYSTEM_CAPABILITY_SCHEMA,
        "authority_id": _authority_id(fence),
        "authority_root_sha256": _authority_root_sha256(fence),
        "process_nonce": _PROCESS_NONCE.hex(),
        "fence_nonce": _fence_nonce(fence),
        "system_role": system_role,
        "device_id": roster.local_device_id,
        "roster_version": roster.version,
        "roster_hash": roster.roster_hash,
        "key_id": key_id,
    }


def _decode_system_capability(payload: bytes) -> dict[str, object]:
    try:
        value = json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError, TypeError) as exc:
        raise AuthorityEpochError("system capability proof is invalid") from exc
    if (
        not isinstance(value, dict)
        or set(value) != _SYSTEM_CAPABILITY_FIELDS
        or canonical_json_bytes(value) != payload
    ):
        raise AuthorityEpochError("system capability proof is not canonical")
    roster_version = value.get("roster_version")
    if (
        value.get("schema") != _SYSTEM_CAPABILITY_SCHEMA
        or not isinstance(value.get("authority_id"), str)
        or not value.get("authority_id")
        or not _is_sha256(value.get("authority_root_sha256"))
        or not _is_sha256(value.get("process_nonce"))
        or not _is_sha256(value.get("fence_nonce"))
        or value.get("system_role") not in _SYSTEM_ROLES
        or not isinstance(value.get("device_id"), str)
        or not value.get("device_id")
        or isinstance(roster_version, bool)
        or not isinstance(roster_version, int)
        or roster_version < 1
        or not _is_sha256(value.get("roster_hash"))
        or not isinstance(value.get("key_id"), str)
        or not value.get("key_id")
    ):
        raise AuthorityEpochError("system capability claims are invalid")
    return value


def _require_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise AuthorityEpochError(f"authority epoch {field} is invalid")
    return value


def _require_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AuthorityEpochError(f"authority epoch {field} is invalid")
    return value


def _decode_authorization(value: object) -> EpochMarkerAuthorization:
    if not isinstance(value, dict):
        raise AuthorityEpochError("authority epoch authorization is invalid")
    if set(value) != {
        "schema",
        "attempt_id",
        "device_id",
        "epoch",
        "control_oid",
        "artifact_digests",
        "roster_version",
        "key_id",
        "signature",
    }:
        raise AuthorityEpochError("authority epoch authorization fields are invalid")
    signature_value = value.get("signature")
    artifact_value = value.get("artifact_digests")
    if not isinstance(signature_value, dict) or not isinstance(artifact_value, dict):
        raise AuthorityEpochError("authority epoch authorization is invalid")
    if set(signature_value) != {
        "algorithm",
        "key_id",
        "roster_version",
        "signature",
    }:
        raise AuthorityEpochError("authority epoch authorization signature fields are invalid")
    if not all(
        isinstance(name, str)
        and name
        and isinstance(digest, str)
        and len(digest) == 64
        and all(character in "0123456789abcdef" for character in digest)
        for name, digest in artifact_value.items()
    ):
        raise AuthorityEpochError("authority epoch artifact digests are invalid")
    try:
        signature = SignatureEnvelope(
            algorithm=_require_string(signature_value.get("algorithm"), "algorithm"),  # type: ignore[arg-type]
            key_id=_require_string(signature_value.get("key_id"), "signature key"),
            roster_version=_require_int(
                signature_value.get("roster_version"), "signature roster version"
            ),
            signature=_require_string(signature_value.get("signature"), "authorization signature"),
        )
        return EpochMarkerAuthorization(
            schema=_require_string(value.get("schema"), "authorization schema"),  # type: ignore[arg-type]
            attempt_id=_require_string(value.get("attempt_id"), "attempt id"),
            device_id=_require_string(value.get("device_id"), "device id"),
            epoch=_require_int(value.get("epoch"), "epoch"),
            control_oid=_require_string(value.get("control_oid"), "control OID"),
            artifact_digests=dict(artifact_value),
            roster_version=_require_int(value.get("roster_version"), "roster version"),
            key_id=_require_string(value.get("key_id"), "key id"),
            signature=signature,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise AuthorityEpochError("authority epoch authorization is invalid") from exc


def _authority_documents(
    authorization: EpochMarkerAuthorization,
) -> tuple[dict[str, object], dict[str, object]]:
    authorization_body = asdict(authorization)
    authorization_sha256 = _sha256(authorization_body)
    marker: dict[str, object] = {
        "schema": _MARKER_SCHEMA,
        "epoch": authorization.epoch,
        "control_oid": authorization.control_oid,
        "artifact_digests": dict(authorization.artifact_digests),
        "authorization": authorization_body,
        "authorization_sha256": authorization_sha256,
        "attempt_id": authorization.attempt_id,
        "device_id": authorization.device_id,
        "roster_version": authorization.roster_version,
        "key_id": authorization.key_id,
    }
    high_watermark: dict[str, object] = {
        "schema": _HIGH_WATERMARK_SCHEMA,
        "highest_epoch": authorization.epoch,
        "marker_sha256": _sha256(marker),
        "authorization": authorization_body,
        "authorization_sha256": authorization_sha256,
    }
    return marker, high_watermark


class EpochFence:
    __instance_nonce = _FENCE_INSTANCE_NONCES

    __slots__ = (
        "__weakref__",
        "high_watermark_path",
        "marker_path",
        "pin_store",
        "root",
        "roster",
    )

    def __new__(
        cls,
        *_args: object,
        **_kwargs: object,
    ) -> EpochFence:
        instance = super().__new__(cls)
        _FENCE_INSTANCE_NONCES.allocate(instance)
        return instance

    def __setattr__(self, name: str, value: object) -> None:
        if name == "_EpochFence__instance_nonce":
            raise AttributeError("EpochFence instance nonce is immutable")
        object.__setattr__(self, name, value)

    def __copy__(self) -> NoReturn:
        raise TypeError("EpochFence instances cannot be copied")

    def __deepcopy__(self, memo: object) -> NoReturn:
        del memo
        raise TypeError("EpochFence instances cannot be copied")

    def __reduce__(self) -> NoReturn:
        raise TypeError("EpochFence instances cannot be serialized")

    def __reduce_ex__(self, protocol: SupportsIndex) -> NoReturn:
        del protocol
        raise TypeError("EpochFence instances cannot be serialized")

    def __init__(
        self,
        root: Path,
        *,
        roster: VerificationRoster,
        verifier: OperationalVerifier | None = None,
        pin_store: AuthorityPinStore | None = None,
    ) -> None:
        if verifier is None:
            verifier = OperationalVerifier()
        if type(verifier) is not OperationalVerifier:
            raise TypeError("EpochFence requires the stateless operational verifier")
        self.root = Path(root)
        self.marker_path = self.root / "authority-epoch.json"
        self.high_watermark_path = self.root / "authority-epoch-high-watermark.json"
        if pin_store is None:
            try:
                pin_store = AuthorityPinStore.for_root(self.root)
            except KeyStoreError as exc:
                raise AuthorityEpochError(
                    "verification roster authority pin is unavailable"
                ) from exc
        self.pin_store = pin_store
        try:
            canonical_roster = VerificationRoster.load(
                self.root,
                pin_store=self.pin_store,
            )
        except RosterError as exc:
            raise AuthorityEpochError(
                "verification roster does not match the pinned roster"
            ) from exc
        if canonical_roster != roster:
            raise AuthorityEpochError(
                "supplied verification roster does not match the pinned roster"
            )
        self.roster = canonical_roster
        self._recover_prepared_authority()
        self._read_authority(required=False)

    def _load_latest_roster(self) -> VerificationRoster:
        try:
            return VerificationRoster.load(
                self.root,
                pin_store=self.pin_store,
            )
        except RosterError as exc:
            raise AuthorityEpochError("verification roster is unavailable or invalid") from exc

    def _load_roster_version(self, version: int) -> VerificationRoster:
        try:
            return VerificationRoster.load_version(
                self.root,
                version=version,
                pin_store=self.pin_store,
            )
        except RosterError as exc:
            raise AuthorityEpochError(
                "epoch authorization roster history is unavailable or invalid"
            ) from exc

    def _assert_latest_roster(
        self,
        expected: VerificationRoster,
    ) -> None:
        if self._load_latest_roster() != expected:
            raise AuthorityEpochError("verification roster changed concurrently")

    def _verify_authorization(
        self,
        authorization: EpochMarkerAuthorization,
        observed_artifact_digests: Mapping[str, str],
        *,
        roster: VerificationRoster,
    ) -> None:
        if authorization.schema != _AUTHORIZATION_SCHEMA:
            raise AuthorityEpochError("unknown epoch authorization schema")
        if authorization.device_id != roster.local_device_id:
            raise AuthorityEpochError("epoch authorization device mismatch")
        if authorization.roster_version != roster.version:
            raise AuthorityEpochError("epoch authorization roster mismatch")
        if authorization.key_id != authorization.signature.key_id:
            raise AuthorityEpochError("epoch authorization key mismatch")
        if dict(authorization.artifact_digests) != dict(observed_artifact_digests):
            raise AuthorityEpochError("epoch authorization artifact digest mismatch")
        try:
            OperationalVerifier().verify(
                domain=_AUTHORIZATION_SCHEMA,
                payload=canonical_signed_bytes(authorization),
                envelope=authorization.signature,
                roster=roster,
            )
        except OperationalError as exc:
            raise AuthorityEpochError("invalid epoch marker authorization") from exc

    def bootstrap(
        self,
        *,
        authorization: EpochMarkerAuthorization,
        observed_artifact_digests: Mapping[str, str],
    ) -> None:
        if authorization.epoch != 0:
            raise AuthorityEpochError("bootstrap authority epoch must be zero")
        required = {"bootstrap_roster", "empty_anchor"}
        if set(observed_artifact_digests) != required:
            raise AuthorityEpochError("bootstrap artifact digests are incomplete")
        self._persist(
            authorization=authorization,
            observed_artifact_digests=observed_artifact_digests,
            bootstrap=True,
        )

    def activate(
        self,
        *,
        authorization: EpochMarkerAuthorization,
        observed_artifact_digests: Mapping[str, str],
    ) -> None:
        self._persist(
            authorization=authorization,
            observed_artifact_digests=observed_artifact_digests,
            bootstrap=False,
        )

    def _persist(
        self,
        *,
        authorization: EpochMarkerAuthorization,
        observed_artifact_digests: Mapping[str, str],
        bootstrap: bool,
    ) -> None:
        with authority_admission_lock(self.root):
            latest_roster = self._load_latest_roster()
            self._verify_authorization(
                authorization,
                observed_artifact_digests,
                roster=latest_roster,
            )
            authorization_record = canonical_json_bytes(asdict(authorization))
            with authority_write_lock(self.root):
                current = self._read_authority(required=False)
                if bootstrap:
                    if current is not None:
                        raise AuthorityEpochError("authority epoch already bootstrapped")
                else:
                    if current is None:
                        raise AuthorityEpochError(
                            "authority epoch activation requires an existing marker"
                        )
                    if authorization.epoch <= current.epoch:
                        raise AuthorityEpochError("authority epoch must increase monotonically")
                self._assert_latest_roster(latest_roster)
                try:
                    self.pin_store._stage_epoch(
                        self.root,
                        authorization_record,
                        bootstrap=bootstrap,
                    )
                except KeyStoreError as exc:
                    raise AuthorityEpochError("authority epoch pin rejected update") from exc
                self._write_authority(authorization)
                try:
                    self.pin_store._finish_epoch(
                        self.root,
                        authorization_record,
                        bootstrap=bootstrap,
                    )
                except KeyStoreError as exc:
                    raise AuthorityEpochError("authority epoch pin commit failed") from exc

    def _write_authority(self, authorization: EpochMarkerAuthorization) -> None:
        marker, high_watermark = _authority_documents(authorization)
        atomic_write_text(self.marker_path, _canonical_text(marker))
        _fsync_directory(self.root)
        atomic_write_text(
            self.high_watermark_path,
            _canonical_text(high_watermark),
        )
        _fsync_directory(self.root)

    def _read_json(self, path: Path, description: str) -> dict[str, Any]:
        try:
            with open_secure_directory(self.root) as directory:
                text = directory.read_bytes(
                    _authority_relative(self.root, path),
                ).decode("utf-8")
            value = json.loads(text)
        except FileNotFoundError:
            raise
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            raise AuthorityEpochError(f"authority epoch {description} is invalid") from exc
        if not isinstance(value, dict) or _canonical_text(value) != text:
            raise AuthorityEpochError(f"authority epoch {description} must be canonical")
        return value

    def _authority_exists(self, path: Path) -> bool:
        try:
            with open_secure_directory(self.root) as directory:
                return directory.exists(_authority_relative(self.root, path))
        except FileNotFoundError:
            return False

    def _recover_prepared_authority(self) -> None:
        try:
            prepared = self.pin_store._prepared_epoch(self.root)
            state = self.pin_store._read(self.root)
        except KeyStoreError as exc:
            raise AuthorityEpochError("authority epoch pin is unavailable") from exc
        if prepared is None:
            return
        authorization_record, bootstrap = prepared
        try:
            authorization_value = json.loads(authorization_record.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise AuthorityEpochError("prepared authority epoch is invalid") from exc
        authorization = _decode_authorization(authorization_value)
        if canonical_json_bytes(asdict(authorization)) != authorization_record:
            raise AuthorityEpochError("prepared authority epoch is not canonical")
        authorization_roster = self._load_roster_version(authorization.roster_version)
        self._verify_authorization(
            authorization,
            authorization.artifact_digests,
            roster=authorization_roster,
        )
        authorization_sha256 = hashlib.sha256(authorization_record).hexdigest()
        if (
            state.pending_epoch != authorization.epoch
            or state.pending_authorization_sha256 != authorization_sha256
        ):
            raise AuthorityEpochError("prepared authority epoch does not match its pin")
        if bootstrap:
            if state.epoch is not None or authorization.epoch != 0:
                raise AuthorityEpochError("prepared bootstrap authority epoch is invalid")
        elif state.epoch is None or authorization.epoch <= state.epoch:
            raise AuthorityEpochError("prepared authority epoch predecessor is invalid")

        pending_documents = _authority_documents(authorization)
        with authority_write_lock(self.root):
            existing: list[tuple[Path, str, int]] = [
                (self.marker_path, "marker", 0),
                (self.high_watermark_path, "high-watermark", 1),
            ]
            present = 0
            for path, description, document_index in existing:
                if not self._authority_exists(path):
                    continue
                present += 1
                body = self._read_json(path, description)
                existing_authorization = _decode_authorization(body.get("authorization"))
                existing_roster = self._load_roster_version(existing_authorization.roster_version)
                self._verify_authorization(
                    existing_authorization,
                    existing_authorization.artifact_digests,
                    roster=existing_roster,
                )
                existing_record = canonical_json_bytes(asdict(existing_authorization))
                existing_sha256 = hashlib.sha256(existing_record).hexdigest()
                if existing_sha256 == authorization_sha256:
                    expected = pending_documents[document_index]
                elif (
                    not bootstrap
                    and existing_sha256 == state.authorization_sha256
                    and existing_authorization.epoch == state.epoch
                ):
                    expected = _authority_documents(existing_authorization)[document_index]
                else:
                    raise AuthorityEpochError(
                        "prepared authority epoch destination has an invalid predecessor"
                    )
                if body != expected:
                    raise AuthorityEpochError(
                        "prepared authority epoch destination is inconsistent"
                    )
            if not bootstrap and present == 0:
                raise AuthorityEpochError("prepared authority epoch predecessor root is missing")
            self._write_authority(authorization)
            try:
                self.pin_store._finish_epoch(
                    self.root,
                    authorization_record,
                    bootstrap=bootstrap,
                )
            except KeyStoreError as exc:
                raise AuthorityEpochError("authority epoch pin commit failed") from exc

    def _read_authority(self, *, required: bool = True) -> EpochMarkerAuthorization | None:
        marker_exists = self._authority_exists(self.marker_path)
        high_watermark_exists = self._authority_exists(self.high_watermark_path)
        if not marker_exists and not high_watermark_exists:
            try:
                pin = self.pin_store._read(self.root)
            except KeyStoreError as exc:
                raise AuthorityEpochError("authority epoch pin is unavailable") from exc
            if pin.epoch is not None or pin.pending_epoch is not None:
                raise AuthorityEpochError(
                    "authority epoch root rollback or incomplete update detected"
                )
            if required:
                raise AuthorityEpochError("authority epoch marker is missing")
            return None
        if not marker_exists or not high_watermark_exists:
            raise AuthorityEpochError("authority epoch marker or high-watermark is missing")
        try:
            marker = self._read_json(self.marker_path, "marker")
            high_watermark = self._read_json(self.high_watermark_path, "high-watermark")
        except FileNotFoundError:
            raise AuthorityEpochError(
                "authority epoch marker or high-watermark is missing"
            ) from None
        if marker.get("schema") != _MARKER_SCHEMA:
            raise AuthorityEpochError("authority epoch marker schema is invalid")
        if high_watermark.get("schema") != _HIGH_WATERMARK_SCHEMA:
            raise AuthorityEpochError("authority epoch high-watermark schema is invalid")
        if set(marker) != {
            "schema",
            "epoch",
            "control_oid",
            "artifact_digests",
            "authorization",
            "authorization_sha256",
            "attempt_id",
            "device_id",
            "roster_version",
            "key_id",
        }:
            raise AuthorityEpochError("authority epoch marker fields are invalid")
        if set(high_watermark) != {
            "schema",
            "highest_epoch",
            "marker_sha256",
            "authorization",
            "authorization_sha256",
        }:
            raise AuthorityEpochError("authority epoch high-watermark fields are invalid")

        authorization = _decode_authorization(marker.get("authorization"))
        high_authorization = _decode_authorization(high_watermark.get("authorization"))
        authorization_sha256 = _sha256(asdict(authorization))
        if (
            marker.get("authorization_sha256") != authorization_sha256
            or high_watermark.get("authorization_sha256") != authorization_sha256
            or canonical_json_bytes(asdict(high_authorization))
            != canonical_json_bytes(asdict(authorization))
        ):
            raise AuthorityEpochError("authority epoch authorization digest mismatch")
        marker_sha256 = _sha256(marker)
        if high_watermark.get("marker_sha256") != marker_sha256:
            raise AuthorityEpochError("authority epoch marker rollback detected")
        if (
            high_watermark.get("highest_epoch") != authorization.epoch
            or marker.get("epoch") != authorization.epoch
            or marker.get("control_oid") != authorization.control_oid
            or marker.get("artifact_digests") != dict(authorization.artifact_digests)
            or marker.get("attempt_id") != authorization.attempt_id
            or marker.get("device_id") != authorization.device_id
            or marker.get("roster_version") != authorization.roster_version
            or marker.get("key_id") != authorization.key_id
        ):
            raise AuthorityEpochError("authority epoch marker claims mismatch")
        authorization_roster = self._load_roster_version(authorization.roster_version)
        self._verify_authorization(
            authorization,
            authorization.artifact_digests,
            roster=authorization_roster,
        )
        try:
            self.pin_store._verify_epoch(
                self.root,
                epoch=authorization.epoch,
                authorization_sha256=authorization_sha256,
            )
        except KeyStoreError as exc:
            raise AuthorityEpochError("authority epoch rollback detected") from exc
        return authorization

    def _verify_system_capability(
        self,
        payload: bytes,
        signature: SignatureEnvelope,
        *,
        roster: VerificationRoster,
    ) -> None:
        if not isinstance(signature, SignatureEnvelope):
            raise AuthorityEpochError("system capability signature is invalid")
        claims = _decode_system_capability(payload)
        key_id = claims["key_id"]
        system_role = claims["system_role"]
        assert isinstance(key_id, str)
        assert isinstance(system_role, str)
        if claims != _system_capability_claims(
            self,
            roster=roster,
            key_id=key_id,
            system_role=system_role,
        ):
            raise AuthorityEpochError(
                "system capability is bound to a different authority or process"
            )
        try:
            OperationalVerifier().verify(
                domain=_SYSTEM_CAPABILITY_DOMAIN,
                payload=payload,
                envelope=signature,
                roster=roster,
            )
        except OperationalError as exc:
            raise AuthorityEpochError("invalid system capability signature") from exc

    def context(
        self,
        identity: PrincipalIdentity,
        *,
        request_epoch: int,
        request_control_oid: str,
    ) -> CommitContext:
        latest_roster = self._load_latest_roster()
        authorization = self._read_authority()
        assert authorization is not None
        if request_epoch != authorization.epoch:
            raise AuthorityEpochError(
                "request authority epoch is stale or future",
                details={
                    "request_epoch": request_epoch,
                    "epoch": authorization.epoch,
                },
            )
        if request_control_oid != authorization.control_oid:
            raise AuthorityEpochError("request control OID does not match authority")
        self._assert_latest_roster(latest_roster)
        return CommitContext(
            identity=identity,
            authority_epoch=request_epoch,
            control_oid=request_control_oid,
            origin_device=latest_roster.local_device_id,
        )

    def system_context(
        self,
        identity: PrincipalIdentity,
        *,
        capability: SystemCapability,
    ) -> CommitContext:
        if type(capability) is not SystemCapability:
            raise AuthorityEpochError("invalid system capability")
        try:
            operation = object.__getattribute__(
                capability,
                "_SystemCapability__operation",
            )
            payload = object.__getattribute__(
                capability,
                "_SystemCapability__payload",
            )
            signature = object.__getattribute__(
                capability,
                "_SystemCapability__signature",
            )
        except (AttributeError, TypeError):
            raise AuthorityEpochError("invalid system capability") from None
        if (
            type(operation) is not _BoundSystemContext
            or _ACTIVE_SYSTEM_OPERATION.get() is not operation
            or not isinstance(payload, bytes)
            or not operation._authorizes(self, payload, signature)
        ):
            raise AuthorityEpochError("invalid system capability")
        latest_roster = self._load_latest_roster()
        self._verify_system_capability(
            payload,
            signature,
            roster=latest_roster,
        )
        authorization = self._read_authority()
        assert authorization is not None
        self._assert_latest_roster(latest_roster)
        return CommitContext(
            identity=identity,
            authority_epoch=authorization.epoch,
            control_oid=authorization.control_oid,
            origin_device=latest_roster.local_device_id,
        )

    @contextmanager
    def verified(self, context: CommitContext) -> Iterator[None]:
        """Hold the durable epoch fence for the complete guarded mutation.

        Callers that persist authority-bound state must keep this context open
        until their own durable commit point.  This closes the race where an
        activation could advance the marker after a one-shot verification but
        before the caller fsynced its record.
        """
        if not isinstance(context, CommitContext):
            raise AuthorityEpochError("commit context is required")
        with authority_admission_lock(self.root), authority_write_lock(self.root):
            latest_roster = self._load_latest_roster()
            authorization = self._read_authority()
            assert authorization is not None
            if context.authority_epoch != authorization.epoch:
                raise AuthorityEpochError("commit context authority epoch mismatch")
            if context.control_oid != authorization.control_oid:
                raise AuthorityEpochError("commit context control OID mismatch")
            if context.origin_device != latest_roster.local_device_id:
                raise AuthorityEpochError("commit context origin device mismatch")
            if context.identity.device_id != latest_roster.local_device_id:
                raise AuthorityEpochError("commit context principal device mismatch")
            self._assert_latest_roster(latest_roster)
            yield

    def verify(self, context: CommitContext) -> None:
        """Verify one context without retaining the mutation fence."""
        with self.verified(context):
            return


def bind_system_context(
    fence: EpochFence,
    *,
    signer: OperationalSigner,
    key_id: str,
    system_role: SystemRole,
) -> SystemContextOperation:
    """Authenticate one fence-local daemon/migration operation.

    Possessing or constructing an ``EpochFence`` is intentionally insufficient:
    the caller must prove control of a key enrolled for the requested internal
    role. The returned callable contains only immutable signed proof bytes and a
    weak fence reference; no signer or raw ``SystemCapability`` escapes.
    """

    if type(fence) is not EpochFence:
        raise AuthorityEpochError("system context binding requires an EpochFence")
    if type(signer) is not OperationalSigner:
        raise AuthorityEpochError("system context binding requires an operational signer")
    if not isinstance(key_id, str) or not key_id:
        raise AuthorityEpochError("system context binding key is invalid")
    if system_role not in _SYSTEM_ROLES:
        raise AuthorityEpochError("system context binding role is invalid")
    latest_roster = fence._load_latest_roster()
    claims = _system_capability_claims(
        fence,
        roster=latest_roster,
        key_id=key_id,
        system_role=system_role,
    )
    payload = canonical_json_bytes(claims)
    try:
        signature = signer.sign(
            domain=_SYSTEM_CAPABILITY_DOMAIN,
            payload=payload,
            key_id=key_id,
        )
    except OperationalError as exc:
        raise AuthorityEpochError("system context binding signature failed") from exc
    fence._verify_system_capability(
        payload,
        signature,
        roster=latest_roster,
    )
    fence._assert_latest_roster(latest_roster)
    return _BoundSystemContext(fence, payload, signature)


__all__ = [
    "AuthorityEpochError",
    "CommitContext",
    "EpochFence",
    "SystemCapability",
    "SystemContextOperation",
    "SystemRole",
    "bind_system_context",
]
