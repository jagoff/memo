"""Signed, canonical Memflow source receipts (v2)."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from memo.errors import SignatureError
from memo.operational_event import canonical_json_bytes
from memo.operational_roster import VerificationRoster
from memo.operational_signing import OperationalSigner, OperationalVerifier, SignatureEnvelope

DOMAIN = "memo.cutover.source_receipt.v2"

def _ts(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise SignatureError("invalid source receipt timestamp") from exc
    if parsed.tzinfo is None:
        raise SignatureError("source receipt timestamps must be timezone-aware")
    return parsed.astimezone(UTC)

@dataclass(frozen=True)
class SourceBucket:
    start: str
    end: str
    count: int
    digest: str

    def to_dict(self) -> dict[str, Any]:
        return {"start": self.start, "end": self.end, "count": self.count, "digest": self.digest}

@dataclass(frozen=True)
class SourceReceiptV2:
    device_id: str
    key_id: str
    roster_id: str
    query: str
    extractor_version: str
    snapshot_commit: str
    raw_event_set_sha256: str
    window_start: str
    window_end: str
    issued_at: str
    collected_at: str
    cursor: str
    extraction_complete: bool
    hourly_buckets: tuple[SourceBucket, ...]
    frozen_at: str | None = None
    signature: SignatureEnvelope | None = None
    schema: str = "memo.cutover_source_receipt.v2"

    def to_dict(self, *, blank_signature: bool = False) -> dict[str, Any]:
        env = self.signature
        return {"schema": self.schema, "device_id": self.device_id, "key_id": self.key_id,
                "roster_id": self.roster_id, "query": self.query, "extractor_version": self.extractor_version,
                "snapshot_commit": self.snapshot_commit, "raw_event_set_sha256": self.raw_event_set_sha256,
                "window_start": self.window_start, "window_end": self.window_end, "issued_at": self.issued_at,
                "collected_at": self.collected_at, "cursor": self.cursor, "extraction_complete": self.extraction_complete,
                "hourly_buckets": [b.to_dict() for b in self.hourly_buckets], "frozen_at": self.frozen_at,
                "signature": "" if blank_signature else (env.signature if env else ""),
                "algorithm": env.algorithm if env else "", "roster_version": env.roster_version if env else 0}

    def signed_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict(blank_signature=True))

    def signature_envelope(self) -> SignatureEnvelope:
        if self.signature is None:
            raise SignatureError("source receipt is unsigned")
        return self.signature

def sign_source_receipt(receipt: SourceReceiptV2, signer: OperationalSigner) -> SourceReceiptV2:
    # Include envelope metadata in the canonical payload before signing; the
    # verifier reconstructs the same bytes from the persisted receipt.
    seeded = SignatureEnvelope(
        algorithm="ed25519", key_id=receipt.key_id,
        roster_version=signer.roster_version, signature="",
    )
    env = signer.sign(
        domain=DOMAIN,
        payload=SourceReceiptV2(**{**receipt.__dict__, "signature": seeded}).signed_bytes(),
        key_id=receipt.key_id,
    )
    return SourceReceiptV2(**{**receipt.__dict__, "signature": env})

def verify_source_receipt(receipt: SourceReceiptV2, *, roster: VerificationRoster, frozen_at: str | None = None,
                          window_start: str | None = None, window_end: str | None = None,
                          authoritative_events: list[dict[str, Any]] | None = None) -> None:
    if receipt.schema != "memo.cutover_source_receipt.v2" or receipt.signature is None:
        raise SignatureError("source receipt is unsigned or has invalid schema")
    key = roster.key(receipt.key_id)
    if key.device_id != receipt.device_id or receipt.roster_id not in {roster.roster_hash, str(roster.version)}:
        raise SignatureError("source receipt device/key/roster mismatch")
    if len(receipt.raw_event_set_sha256) != 64 or any(c not in "0123456789abcdef" for c in receipt.raw_event_set_sha256):
        raise SignatureError("invalid raw event digest")
    start, end = _ts(receipt.window_start), _ts(receipt.window_end)
    if window_start is not None and start != _ts(window_start):
        raise SignatureError("source receipt window start mismatch")
    if window_end is not None and end != _ts(window_end):
        raise SignatureError("source receipt window end mismatch")
    issued = _ts(receipt.issued_at)
    if end < start or _ts(receipt.collected_at) > issued or end > issued:
        raise SignatureError("invalid source receipt time window")
    if frozen_at is not None and _ts(receipt.collected_at) > _ts(frozen_at):
        raise SignatureError("source receipt is newer than frozen_at")
    if not receipt.hourly_buckets:
        raise SignatureError("hourly buckets are required")
    previous_end = start
    for bucket in receipt.hourly_buckets:
        if isinstance(bucket.count, bool) or not isinstance(bucket.count, int) or bucket.count < 0:
            raise SignatureError("hourly bucket count is invalid")
        if len(bucket.digest) != 64 or any(c not in "0123456789abcdef" for c in bucket.digest):
            raise SignatureError("hourly bucket digest is invalid")
        bs, be = _ts(bucket.start), _ts(bucket.end)
        if bs != previous_end or be <= bs or (be - bs).total_seconds() != 3600 or be > end:
            raise SignatureError("hourly buckets are malformed or unordered")
        previous_end = be
    if previous_end != end:
        raise SignatureError("hourly buckets do not cover receipt window")
    if authoritative_events is not None:
        digest = hashlib.sha256(json.dumps(authoritative_events, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
        if digest != receipt.raw_event_set_sha256 or sum(b.count for b in receipt.hourly_buckets) != len(authoritative_events):
            raise SignatureError("source receipt aggregate does not match authoritative events")
    if not receipt.cursor or not receipt.extraction_complete:
        raise SignatureError("source extraction is incomplete")
    OperationalVerifier().verify(domain=DOMAIN, payload=receipt.signed_bytes(), envelope=receipt.signature, roster=roster)

__all__ = ["SourceBucket", "SourceReceiptV2", "sign_source_receipt", "verify_source_receipt", "DOMAIN"]
