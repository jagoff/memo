"""Durable, authenticated authority epochs and request-level fencing."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from memo.atomic_io import atomic_write_text, authority_write_lock
from memo.errors import AuthorityEpochError, OperationalError
from memo.identity import PrincipalIdentity
from memo.operational_event import (
    EpochMarkerAuthorization,
    MigrationOrigin,
    canonical_json_bytes,
    canonical_signed_bytes,
)
from memo.operational_key_store import AuthorityPinStore, KeyStoreError
from memo.operational_roster import VerificationRoster
from memo.operational_signing import (
    OperationalVerifier,
    SignatureEnvelope,
)

_MARKER_SCHEMA = "memo.operational_authority_epoch.v1"
_HIGH_WATERMARK_SCHEMA = "memo.operational_authority_epoch_high_watermark.v1"
_AUTHORIZATION_SCHEMA = "memo.operational_epoch_authorization.v1"


@dataclass(frozen=True)
class CommitContext:
    identity: PrincipalIdentity
    authority_epoch: int
    control_oid: str
    origin_device: str
    migration_origin: MigrationOrigin | None = None


class SystemCapability:
    """Nominal type whose instances exist only inside trusted composition."""

    __slots__ = ("__owner",)

    def __new__(cls) -> SystemCapability:
        del cls
        raise TypeError("SystemCapability is created only inside a trusted composition closure")


SystemContextOperation = Callable[[PrincipalIdentity], CommitContext]
SystemContextSink = Callable[[SystemContextOperation], None]


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _canonical_text(value: object) -> str:
    return canonical_json_bytes(value).decode("utf-8")


def _sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


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
    def __init__(
        self,
        root: Path,
        *,
        roster: VerificationRoster,
        verifier: OperationalVerifier,
        pin_store: AuthorityPinStore,
        _system_context_sink: SystemContextSink | None = None,
    ) -> None:
        self.root = Path(root)
        self.marker_path = self.root / "authority-epoch.json"
        self.high_watermark_path = self.root / "authority-epoch-high-watermark.json"
        self.roster = roster
        self.verifier = verifier
        self.pin_store = pin_store
        self._recover_prepared_authority()
        self._read_authority(required=False)
        if _system_context_sink is not None:
            # Task 3's composition root owns this one-shot handoff. It retains
            # only the bound operation in the trusted daemon/migration closure;
            # adapters receive neither the capability nor construction authority.
            capability = object.__new__(SystemCapability)
            object.__setattr__(capability, "_SystemCapability__owner", self)

            def bound_system_context(identity: PrincipalIdentity) -> CommitContext:
                return self.system_context(identity, capability=capability)

            _system_context_sink(bound_system_context)

    def _verify_authorization(
        self,
        authorization: EpochMarkerAuthorization,
        observed_artifact_digests: Mapping[str, str],
    ) -> None:
        if authorization.schema != _AUTHORIZATION_SCHEMA:
            raise AuthorityEpochError("unknown epoch authorization schema")
        if authorization.device_id != self.roster.local_device_id:
            raise AuthorityEpochError("epoch authorization device mismatch")
        if authorization.roster_version != self.roster.version:
            raise AuthorityEpochError("epoch authorization roster mismatch")
        if authorization.key_id != authorization.signature.key_id:
            raise AuthorityEpochError("epoch authorization key mismatch")
        if dict(authorization.artifact_digests) != dict(observed_artifact_digests):
            raise AuthorityEpochError("epoch authorization artifact digest mismatch")
        try:
            self.verifier.verify(
                domain=_AUTHORIZATION_SCHEMA,
                payload=canonical_signed_bytes(authorization),
                envelope=authorization.signature,
                roster=self.roster,
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
        self._verify_authorization(authorization, observed_artifact_digests)
        authorization_record = canonical_json_bytes(asdict(authorization))
        with authority_write_lock(self.marker_path):
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
        self.root.mkdir(parents=True, exist_ok=True)
        atomic_write_text(self.marker_path, _canonical_text(marker))
        _fsync_directory(self.root)
        atomic_write_text(
            self.high_watermark_path,
            _canonical_text(high_watermark),
        )
        _fsync_directory(self.root)

    def _read_json(self, path: Path, description: str) -> dict[str, Any]:
        try:
            text = path.read_text(encoding="utf-8")
            value = json.loads(text)
        except FileNotFoundError:
            raise
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            raise AuthorityEpochError(f"authority epoch {description} is invalid") from exc
        if not isinstance(value, dict) or _canonical_text(value) != text:
            raise AuthorityEpochError(f"authority epoch {description} must be canonical")
        return value

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
        self._verify_authorization(authorization, authorization.artifact_digests)
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
        with authority_write_lock(self.marker_path):
            existing: list[tuple[Path, str, int]] = [
                (self.marker_path, "marker", 0),
                (self.high_watermark_path, "high-watermark", 1),
            ]
            present = 0
            for path, description, document_index in existing:
                if not path.exists():
                    continue
                present += 1
                body = self._read_json(path, description)
                existing_authorization = _decode_authorization(body.get("authorization"))
                self._verify_authorization(
                    existing_authorization,
                    existing_authorization.artifact_digests,
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
                raise AuthorityEpochError(
                    "prepared authority epoch predecessor root is missing"
                )
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
        marker_exists = self.marker_path.exists()
        high_watermark_exists = self.high_watermark_path.exists()
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
            raise AuthorityEpochError(
                "authority epoch marker or high-watermark is missing"
            )
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
        self._verify_authorization(authorization, authorization.artifact_digests)
        try:
            self.pin_store._verify_epoch(
                self.root,
                epoch=authorization.epoch,
                authorization_sha256=authorization_sha256,
            )
        except KeyStoreError as exc:
            raise AuthorityEpochError("authority epoch rollback detected") from exc
        return authorization

    def context(
        self,
        identity: PrincipalIdentity,
        *,
        request_epoch: int,
        request_control_oid: str,
    ) -> CommitContext:
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
        return CommitContext(
            identity=identity,
            authority_epoch=request_epoch,
            control_oid=request_control_oid,
            origin_device=self.roster.local_device_id,
        )

    def system_context(
        self,
        identity: PrincipalIdentity,
        *,
        capability: SystemCapability,
    ) -> CommitContext:
        try:
            owner = object.__getattribute__(
                capability,
                "_SystemCapability__owner",
            )
        except (AttributeError, TypeError):
            owner = None
        if type(capability) is not SystemCapability or owner is not self:
            raise AuthorityEpochError("invalid system capability")
        authorization = self._read_authority()
        assert authorization is not None
        return CommitContext(
            identity=identity,
            authority_epoch=authorization.epoch,
            control_oid=authorization.control_oid,
            origin_device=self.roster.local_device_id,
        )

    def verify(self, context: CommitContext) -> None:
        if not isinstance(context, CommitContext):
            raise AuthorityEpochError("commit context is required")
        with authority_write_lock(self.marker_path):
            authorization = self._read_authority()
            assert authorization is not None
            if context.authority_epoch != authorization.epoch:
                raise AuthorityEpochError("commit context authority epoch mismatch")
            if context.control_oid != authorization.control_oid:
                raise AuthorityEpochError("commit context control OID mismatch")
            if context.origin_device != self.roster.local_device_id:
                raise AuthorityEpochError("commit context origin device mismatch")
            if context.identity.device_id != self.roster.local_device_id:
                raise AuthorityEpochError("commit context principal device mismatch")


__all__ = [
    "AuthorityEpochError",
    "CommitContext",
    "EpochFence",
    "SystemCapability",
    "SystemContextOperation",
    "SystemContextSink",
]
