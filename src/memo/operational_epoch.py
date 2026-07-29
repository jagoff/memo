"""Durable authority epoch marker and request-level fencing."""

from __future__ import annotations

import hashlib
import json
import secrets
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from memo.atomic_io import atomic_write_text, authority_write_lock
from memo.errors import AuthorityEpochError
from memo.identity import PrincipalIdentity
from memo.operational_event import (
    EpochMarkerAuthorization,
    MigrationOrigin,
    canonical_json_bytes,
    canonical_signed_bytes,
)
from memo.operational_roster import VerificationRoster
from memo.operational_signing import OperationalVerifier

_SYSTEM_CAPABILITY_TOKEN = secrets.token_bytes(32)


@dataclass(frozen=True)
class CommitContext:
    identity: PrincipalIdentity
    authority_epoch: int
    control_oid: str
    origin_device: str
    migration_origin: MigrationOrigin | None = None


class SystemCapability:
    """Opaque process-local authorization for trusted internal callers."""

    __slots__ = ("_token",)

    def __init__(self, token: bytes) -> None:
        if token is not _SYSTEM_CAPABILITY_TOKEN:
            raise TypeError("SystemCapability cannot be constructed externally")
        self._token = token


class EpochFence:
    def __init__(
        self,
        root: Path,
        *,
        roster: VerificationRoster,
        verifier: OperationalVerifier,
    ) -> None:
        self.root = Path(root)
        self.marker_path = self.root / "authority-epoch.json"
        self.roster = roster
        self.verifier = verifier
        self._marker_observed = self.marker_path.exists()
        self._highest_epoch_seen: int | None = None

    def system_capability(self) -> SystemCapability:
        return SystemCapability(_SYSTEM_CAPABILITY_TOKEN)

    def _verify_authorization(
        self,
        authorization: EpochMarkerAuthorization,
        observed_artifact_digests: Mapping[str, str],
    ) -> None:
        if authorization.schema != "memo.operational_epoch_authorization.v1":
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
                domain="memo.operational_epoch_authorization.v1",
                payload=canonical_signed_bytes(authorization),
                envelope=authorization.signature,
                roster=self.roster,
            )
        except Exception as exc:
            if isinstance(exc, AuthorityEpochError):
                raise
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
        with authority_write_lock(self.marker_path):
            current = self._read_marker(required=False)
            if bootstrap:
                if current is not None:
                    raise AuthorityEpochError("authority epoch already bootstrapped")
            elif current is not None and authorization.epoch <= int(current["epoch"]):
                raise AuthorityEpochError("authority epoch must increase monotonically")
            authorization_sha256 = hashlib.sha256(
                canonical_json_bytes(authorization)
            ).hexdigest()
            marker = {
                "schema": "memo.operational_authority_epoch.v1",
                "epoch": authorization.epoch,
                "control_oid": authorization.control_oid,
                "artifact_digests": dict(authorization.artifact_digests),
                "authorization_sha256": authorization_sha256,
                "attempt_id": authorization.attempt_id,
                "device_id": authorization.device_id,
                "roster_version": authorization.roster_version,
                "key_id": authorization.key_id,
            }
            atomic_write_text(
                self.marker_path,
                json.dumps(
                    marker,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                ),
            )
            self._marker_observed = True
            self._highest_epoch_seen = authorization.epoch

    def _read_marker(self, *, required: bool = True) -> dict[str, Any] | None:
        try:
            value = json.loads(self.marker_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            if required or self._marker_observed:
                raise AuthorityEpochError("authority epoch marker is missing") from None
            return None
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            raise AuthorityEpochError("authority epoch marker is invalid") from exc
        if not isinstance(value, dict):
            raise AuthorityEpochError("authority epoch marker must be an object")
        required_fields = {
            "schema",
            "epoch",
            "control_oid",
            "artifact_digests",
            "authorization_sha256",
        }
        if (
            value.get("schema") != "memo.operational_authority_epoch.v1"
            or not required_fields.issubset(value)
        ):
            raise AuthorityEpochError("authority epoch marker schema is invalid")
        try:
            epoch = int(value["epoch"])
        except (TypeError, ValueError) as exc:
            raise AuthorityEpochError("authority epoch marker epoch is invalid") from exc
        if self._highest_epoch_seen is not None and epoch < self._highest_epoch_seen:
            raise AuthorityEpochError("authority epoch marker rollback detected")
        self._highest_epoch_seen = epoch
        return value

    def context(
        self,
        identity: PrincipalIdentity,
        *,
        request_epoch: int,
        request_control_oid: str,
    ) -> CommitContext:
        marker = self._read_marker()
        assert marker is not None
        if request_epoch != int(marker["epoch"]):
            raise AuthorityEpochError(
                "request authority epoch is stale or future",
                details={"request_epoch": request_epoch, "epoch": marker["epoch"]},
            )
        if request_control_oid != str(marker["control_oid"]):
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
        if (
            not isinstance(capability, SystemCapability)
            or capability._token is not _SYSTEM_CAPABILITY_TOKEN
        ):
            raise AuthorityEpochError("invalid system capability")
        marker = self._read_marker()
        assert marker is not None
        return CommitContext(
            identity=identity,
            authority_epoch=int(marker["epoch"]),
            control_oid=str(marker["control_oid"]),
            origin_device=self.roster.local_device_id,
        )

    def verify(self, context: CommitContext) -> None:
        if not isinstance(context, CommitContext):
            raise AuthorityEpochError("commit context is required")
        with authority_write_lock(self.marker_path):
            marker = self._read_marker()
            assert marker is not None
            if context.authority_epoch != int(marker["epoch"]):
                raise AuthorityEpochError("commit context authority epoch mismatch")
            if context.control_oid != str(marker["control_oid"]):
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
]
