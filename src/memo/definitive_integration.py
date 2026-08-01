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
    created_at: str,
    signer: OperationalSigner,
) -> DefinitiveIntegrationReceipt:
    if not attempt_id.strip() or not signer_device_id.strip() or not signer_key_id.strip():
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


__all__ = [
    "DefinitiveIntegrationReceipt",
    "sign_integration_receipt",
    "verify_integration_receipt",
    "write_integration_receipt",
]
