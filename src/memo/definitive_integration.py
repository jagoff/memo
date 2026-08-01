"""Cryptographically verifiable receipts for empirical runtime integration."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal

from memo.atomic_io import atomic_write_text
from memo.errors import OperationalError, OperationalErrorCode
from memo.operational_event import canonical_json_bytes, canonical_signed_bytes
from memo.operational_key_store import PublicKeyRecord
from memo.operational_roster import VerificationRoster
from memo.operational_signing import (
    OperationalSigner,
    OperationalVerifier,
    SignatureEnvelope,
)

_DOMAIN = "memo.definitive.integration.v1"


@dataclass(frozen=True)
class DefinitiveIntegrationReceipt:
    schema: Literal["memo.definitive_integration.v1"]
    attempt_id: str
    topology: str
    checks: Mapping[str, bool]
    evidence: Mapping[str, str]
    signer_device_id: str
    signer_key_id: str
    roster_version: int
    roster_hash: str
    created_at: str
    receipt_sha256: str
    signature: SignatureEnvelope | None


def _receipt_digest(receipt: DefinitiveIntegrationReceipt) -> str:
    body = asdict(receipt)
    body["receipt_sha256"] = ""
    body["signature"] = ""
    return hashlib.sha256(canonical_json_bytes(body)).hexdigest()


def sign_integration_receipt(
    *,
    attempt_id: str,
    checks: Mapping[str, bool],
    evidence: Mapping[str, str],
    signer_device_id: str,
    signer_key_id: str,
    roster_version: int,
    roster_hash: str,
    created_at: str,
    signer: OperationalSigner,
) -> DefinitiveIntegrationReceipt:
    if (
        not attempt_id.strip()
        or not signer_device_id.strip()
        or not signer_key_id.strip()
        or not roster_hash.strip()
    ):
        raise ValueError("integration receipt identity fields must be non-empty")
    if not checks or not all(isinstance(value, bool) for value in checks.values()):
        raise ValueError("integration receipt checks must be non-empty booleans")
    if not all(str(key).strip() and str(value).strip() for key, value in evidence.items()):
        raise ValueError("integration receipt evidence must be non-empty strings")
    unsigned = DefinitiveIntegrationReceipt(
        schema="memo.definitive_integration.v1",
        attempt_id=attempt_id.strip(),
        topology="two-independent-peers+git+terminal+restart",
        checks=dict(sorted(checks.items())),
        evidence=dict(sorted(evidence.items())),
        signer_device_id=signer_device_id,
        signer_key_id=signer_key_id,
        roster_version=roster_version,
        roster_hash=roster_hash,
        created_at=created_at,
        receipt_sha256="",
        signature=None,
    )
    hashed = replace(unsigned, receipt_sha256=_receipt_digest(unsigned))
    return replace(
        hashed,
        signature=signer.sign(
            domain=_DOMAIN,
            payload=canonical_signed_bytes(hashed),
            key_id=signer_key_id,
        ),
    )


def verify_integration_receipt(
    receipt: DefinitiveIntegrationReceipt,
    *,
    roster: VerificationRoster,
    trusted_roster_hash: str,
    require_all_checks: bool = True,
) -> None:
    if receipt.schema != "memo.definitive_integration.v1":
        raise OperationalError(
            OperationalErrorCode.INVALID_EVENT,
            "unsupported definitive integration receipt schema",
            retryable=False,
        )
    if receipt.receipt_sha256 != _receipt_digest(receipt):
        raise OperationalError(
            OperationalErrorCode.INVALID_EVENT,
            "definitive integration receipt digest mismatch",
            retryable=False,
        )
    if receipt.signature is None:
        raise OperationalError(
            OperationalErrorCode.SIGNATURE_INVALID,
            "definitive integration receipt signature is missing",
            retryable=False,
        )
    if (
        not trusted_roster_hash
        or receipt.roster_hash != trusted_roster_hash
        or roster.roster_hash != trusted_roster_hash
        or receipt.roster_version != roster.version
    ):
        raise OperationalError(
            OperationalErrorCode.SIGNATURE_INVALID,
            "definitive integration receipt is not bound to the trusted roster",
            retryable=False,
        )
    try:
        signer_key = roster.key(receipt.signer_key_id)
    except Exception as exc:
        raise OperationalError(
            OperationalErrorCode.SIGNATURE_INVALID,
            "definitive integration signer key is absent from the trusted roster",
            retryable=False,
        ) from exc
    if (
        receipt.signature.key_id != receipt.signer_key_id
        or signer_key.device_id != receipt.signer_device_id
    ):
        raise OperationalError(
            OperationalErrorCode.SIGNATURE_INVALID,
            "definitive integration signer identity is not roster-bound",
            retryable=False,
        )
    OperationalVerifier().verify(
        domain=_DOMAIN,
        payload=canonical_signed_bytes(receipt),
        envelope=receipt.signature,
        roster=roster,
    )
    if require_all_checks and not all(receipt.checks.values()):
        raise OperationalError(
            OperationalErrorCode.INVALID_EVENT,
            "definitive integration receipt contains failed checks",
            retryable=False,
        )


def write_integration_receipt(
    receipt: DefinitiveIntegrationReceipt,
    path: Path,
) -> None:
    atomic_write_text(
        Path(path),
        json.dumps(asdict(receipt), sort_keys=True, indent=2, ensure_ascii=False) + "\n",
    )


def read_integration_receipt(path: Path) -> DefinitiveIntegrationReceipt:
    try:
        body = json.loads(Path(path).read_text(encoding="utf-8"))
        signature_body = body["signature"]
        if not isinstance(body, dict) or not isinstance(signature_body, dict):
            raise TypeError
        return DefinitiveIntegrationReceipt(
            **{
                **body,
                "checks": dict(body["checks"]),
                "evidence": dict(body["evidence"]),
                "signature": SignatureEnvelope(**signature_body),
            }
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise OperationalError(
            OperationalErrorCode.INVALID_EVENT,
            "definitive integration receipt is invalid",
            retryable=False,
        ) from exc


def read_trusted_roster_snapshot(path: Path, *, trusted_roster_hash: str) -> VerificationRoster:
    try:
        body = json.loads(Path(path).read_text(encoding="utf-8"))
        signature_body = body["signature"]
        if not isinstance(body, dict) or not isinstance(signature_body, dict):
            raise TypeError
        keys = tuple(
            PublicKeyRecord(
                device_id=str(item["device_id"]),
                key_id=str(item["key_id"]),
                fingerprint=str(item["fingerprint"]),
                public_key=str(item["public_key"]),
                roles=tuple(str(role) for role in item["roles"]),
                enrollment_sequence=int(item["enrollment_sequence"]),
                revocation_sequence=(
                    int(item["revocation_sequence"])
                    if item.get("revocation_sequence") is not None
                    else None
                ),
                proof_of_possession=str(item.get("proof_of_possession") or ""),
                algorithm=str(item.get("algorithm") or "ed25519"),  # type: ignore[arg-type]
            )
            for item in body["keys"]
        )
        roster = VerificationRoster(
            version=int(body["version"]),
            peers=tuple(str(peer) for peer in body["peers"]),
            keys=keys,
            local_device_id=str(body["local_device_id"]),
            schema=str(body["schema"]),  # type: ignore[arg-type]
            created_at=str(body.get("created_at") or ""),
            previous_roster_hash=str(body.get("previous_roster_hash") or ""),
            roster_hash=str(body["roster_hash"]),
            signature=SignatureEnvelope(**signature_body),
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise OperationalError(
            OperationalErrorCode.SIGNATURE_INVALID,
            "trusted definitive integration roster is invalid",
            retryable=False,
        ) from exc
    if not trusted_roster_hash or roster.roster_hash != trusted_roster_hash:
        raise OperationalError(
            OperationalErrorCode.SIGNATURE_INVALID,
            "definitive integration roster does not match the external trust root",
            retryable=False,
        )
    return roster


__all__ = [
    "DefinitiveIntegrationReceipt",
    "read_integration_receipt",
    "read_trusted_roster_snapshot",
    "sign_integration_receipt",
    "verify_integration_receipt",
    "write_integration_receipt",
]
