"""Domain-separated Ed25519 signing for operational authority records."""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from memo.errors import KeyRevokedError, SignatureError
from memo.operational_key_store import DeviceKeyStore, KeyStoreError, PublicKeyRecord

if TYPE_CHECKING:
    from memo.operational_roster import VerificationRoster

_SIGNATURE_PREFIX = b"memo-signature-v1\0"
ALLOWED_SIGNATURE_DOMAINS = frozenset(
    {
        "memo.operational.event.v2",
        "memo.operational.anchor.v1",
        "memo.operational.roster.v1",
        "memo.operational.roster.bootstrap.v1",
        "memo.operational.migration_origin.v1",
        "memo.operational.migration_prepared.v1",
        "memo.operational.system_capability.v1",
        "memo.operational_epoch_authorization.v1",
        "memo.cutover.vote.v1",
    }
)


def _b64url(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _decode_b64url(value: str) -> bytes:
    try:
        return base64.b64decode(
            value + "=" * (-len(value) % 4),
            altchars=b"-_",
            validate=True,
        )
    except (ValueError, TypeError) as exc:
        raise SignatureError("invalid base64url signature") from exc


def _validate_canonical_json(payload: bytes) -> dict[str, object]:
    try:
        body = json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError, TypeError) as exc:
        raise SignatureError("signature payload must be canonical JSON") from exc
    if not isinstance(body, dict):
        raise SignatureError("signature payload must be a JSON object")
    try:
        canonical = json.dumps(
            body,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise SignatureError("signature payload is not canonical JSON") from exc
    if canonical != payload:
        raise SignatureError("signature payload is not canonical JSON")
    return body


def signature_payload(domain: str, payload: bytes) -> bytes:
    if domain not in ALLOWED_SIGNATURE_DOMAINS:
        raise SignatureError(f"unsupported signature domain: {domain}")
    try:
        encoded_domain = domain.encode("ascii")
    except UnicodeEncodeError as exc:
        raise SignatureError("signature domain must be ASCII") from exc
    _validate_canonical_json(payload)
    return _SIGNATURE_PREFIX + encoded_domain + b"\0" + bytes(payload)


def _sequence(body: dict[str, object], field: str, fallback: int) -> int:
    value = body.get(field)
    if value is None:
        return fallback
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise SignatureError(f"invalid signed sequence field: {field}")
    try:
        sequence = int(value)
    except ValueError as exc:
        raise SignatureError(f"invalid signed sequence field: {field}") from exc
    if sequence < 0:
        raise SignatureError(f"invalid signed sequence field: {field}")
    return sequence


def _claim_string(
    body: dict[str, object],
    name: str,
    *,
    required: bool,
) -> str | None:
    value = body.get(name)
    if value is None and not required:
        return None
    if not isinstance(value, str) or not value:
        raise SignatureError(f"signed record has invalid {name}")
    return value


def _claim_int(
    body: dict[str, object],
    name: str,
    *,
    required: bool,
) -> int | None:
    value = body.get(name)
    if value is None and not required:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise SignatureError(f"signed record has invalid {name}")
    return value


@dataclass(frozen=True)
class SignatureEnvelope:
    algorithm: Literal["ed25519"]
    key_id: str
    roster_version: int
    signature: str


class OperationalSigner:
    def __init__(self, key_store: DeviceKeyStore, *, roster_version: int) -> None:
        self.key_store = key_store
        self.roster_version = int(roster_version)

    def sign(self, *, domain: str, payload: bytes, key_id: str) -> SignatureEnvelope:
        signed_payload = signature_payload(domain, payload)
        try:
            signature = self.key_store.sign(key_id=key_id, payload=signed_payload)
        except KeyStoreError as exc:
            raise SignatureError(str(exc)) from None
        return SignatureEnvelope(
            algorithm="ed25519",
            key_id=key_id,
            roster_version=self.roster_version,
            signature=_b64url(signature),
        )


class OperationalVerifier:
    def verify(
        self,
        *,
        domain: str,
        payload: bytes,
        envelope: SignatureEnvelope,
        roster: VerificationRoster,
    ) -> None:
        if envelope.algorithm != "ed25519":
            raise SignatureError(f"unsupported signature algorithm: {envelope.algorithm}")
        if envelope.roster_version != roster.version:
            raise SignatureError("signature roster version mismatch")
        body = _validate_canonical_json(payload)
        key = roster.key(envelope.key_id)
        self._validate_claims(domain, body, envelope, roster, key)
        signed_payload = signature_payload(domain, payload)
        try:
            public_key = Ed25519PublicKey.from_public_bytes(_decode_b64url(key.public_key))
            public_key.verify(_decode_b64url(envelope.signature), signed_payload)
        except (InvalidSignature, ValueError) as exc:
            raise SignatureError("operational signature verification failed") from exc

    @staticmethod
    def _validate_claims(
        domain: str,
        body: dict[str, object],
        envelope: SignatureEnvelope,
        roster: VerificationRoster,
        key: PublicKeyRecord,
    ) -> None:
        required_role = "origin"
        expected_device: str | None = None
        activation_sequence = roster.version
        record_key_field: str | None = None
        require_record_claims = False

        if domain == "memo.operational.migration_origin.v1":
            required_role = "migration_attestor"
            require_record_claims = True
            record_key_field = "attestor_key_id"
            expected_device = _claim_string(body, "attestor_device_id", required=True)
            if key.roles != ("migration_attestor",):
                raise SignatureError("migration attestor key must have an exclusive role")
        elif domain == "memo.operational.migration_prepared.v1":
            required_role = "migration_attestor"
            record_key_field = "attestor_key_id"
            if key.roles != ("migration_attestor",):
                raise SignatureError("migration attestor key must have an exclusive role")
        elif domain == "memo.operational.event.v2":
            record_key_field = "key_id"
            require_record_claims = body.get("schema") == "memo.operational_event.v2"
            expected_device = _claim_string(body, "origin_device", required=require_record_claims)
        elif domain == "memo.operational.anchor.v1":
            record_key_field = "key_id"
            require_record_claims = True
            signer_role = _claim_string(body, "signer_role", required=True)
            if signer_role == "migration_attestor":
                required_role = "migration_attestor"
                if key.roles != ("migration_attestor",):
                    raise SignatureError("migration attestor key must have an exclusive role")
            elif signer_role != "origin":
                raise SignatureError("anchor signer role is invalid")
            if signer_role == "origin":
                expected_device = _claim_string(body, "origin_device", required=True)
        elif domain == "memo.operational_epoch_authorization.v1":
            record_key_field = "key_id"
            require_record_claims = True
            expected_device = _claim_string(body, "device_id", required=True)
        elif domain == "memo.operational.system_capability.v1":
            if (
                _claim_string(body, "schema", required=True)
                != "memo.operational_system_capability.v1"
            ):
                raise SignatureError("system capability schema is invalid")
            record_key_field = "key_id"
            require_record_claims = True
            expected_device = _claim_string(body, "device_id", required=True)
            system_role = _claim_string(body, "system_role", required=True)
            if system_role == "migration":
                required_role = "migration_attestor"
                if key.roles != ("migration_attestor",):
                    raise SignatureError("system migration key must have an exclusive role")
            elif system_role != "daemon":
                raise SignatureError("system capability role is invalid")
            roster_hash = _claim_string(body, "roster_hash", required=True)
            if roster_hash != roster.roster_hash:
                raise SignatureError("system capability roster hash mismatch")
        elif domain in {
            "memo.operational.roster.v1",
            "memo.operational.roster.bootstrap.v1",
        }:
            required_role = "origin"

        if record_key_field is not None:
            declared_key = _claim_string(body, record_key_field, required=require_record_claims)
            if declared_key is not None and declared_key != envelope.key_id:
                raise SignatureError("signed record key id differs from envelope")
        declared_roster = _claim_int(body, "roster_version", required=require_record_claims)
        if declared_roster is not None and (
            declared_roster != envelope.roster_version or declared_roster != roster.version
        ):
            raise SignatureError("signed record roster version mismatch")
        if key.enrollment_sequence > activation_sequence:
            raise SignatureError("key is not active at this record sequence")
        if key.revocation_sequence is not None and activation_sequence >= key.revocation_sequence:
            raise KeyRevokedError("key is revoked")
        if required_role not in key.roles:
            raise SignatureError(f"key {key.key_id} is not authorized for role {required_role}")
        if expected_device is not None and key.device_id != expected_device:
            raise SignatureError("signature device role mismatch")


__all__ = [
    "ALLOWED_SIGNATURE_DOMAINS",
    "OperationalSigner",
    "OperationalVerifier",
    "SignatureEnvelope",
    "SignatureError",
    "signature_payload",
]
