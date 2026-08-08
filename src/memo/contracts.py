"""Native wire contracts for memo.

The project used to borrow several small schemas from an optional private
package shared with Synapse and Memflow.  Memo 4 owns those contracts directly
so a clean installation has no external runtime or schema dependency.

The classes in this module are deliberately stdlib-only.  They are used at
storage, CLI, MCP, HTTP, and sync boundaries, so keeping them as frozen
dataclasses gives callers a stable shape without coupling the core to a
transport library.
"""

from __future__ import annotations

import hashlib
import json
import secrets
from dataclasses import asdict, dataclass, field, replace
from enum import StrEnum
from typing import Any, Literal

MEMO_BACKEND_SCHEMA = "memo.backend.v1"
MEMO_EVENT_SCHEMA = "memo.event.v1"
MEMO_RECEIPT_SCHEMA = "memo.write_receipt.v1"
MEMO_EVIDENCE_SCHEMA = "memo.evidence_pack.v1"
MEMO_OPERATIONAL_SCHEMA = "memo.operational_state.v1"
MEMO_SYNC_SCHEMA = "memo.sync_event.v1"
MEMO_FEDERATION_SCHEMA = "memo.federation_bundle.v1"

LEGACY_PROVENANCE_KEYS: frozenset[str] = frozenset(
    {
        "synapse_trace_id",
        "synapse_route_reason",
        "synapse_write_policy_schema",
        "synapse_write_target",
        "synapse_agent_id",
        "synapse_agent_signature",
        "synapse_evidence_paths",
    }
)

PROVENANCE_KEYS: frozenset[str] = frozenset(
    {
        "trace_id",
        "actor_id",
        "actor_signature",
        "source_client",
        "route_reason",
        "policy_version",
        "write_target",
        "evidence_uris",
        "idempotency_key",
    }
)

_LEGACY_TO_NATIVE = {
    "synapse_trace_id": "trace_id",
    "synapse_route_reason": "route_reason",
    "synapse_write_policy_schema": "policy_version",
    "synapse_write_target": "write_target",
    "synapse_agent_id": "actor_id",
    "synapse_agent_signature": "actor_signature",
    "synapse_evidence_paths": "evidence_uris",
}


class AnswerStatus(StrEnum):
    ANSWERED = "answered"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    CONFLICTED = "conflicted"
    ERROR = "error"


class TrustTier(StrEnum):
    HUMAN = "human"
    TOOL_OBSERVED = "tool_observed"
    AGENT_VERIFIED = "agent_verified"
    AGENT_INFERRED = "agent_inferred"
    EXTERNAL_UNTRUSTED = "external_untrusted"


class Visibility(StrEnum):
    LOCAL_ONLY = "local_only"
    OWNER = "owner"
    SHARED = "shared"


@dataclass(frozen=True)
class ActorIdentity:
    actor_id: str = "memo"
    actor_kind: Literal["human", "agent", "tool", "system", "device"] = "system"
    signature: str = ""
    source_client: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvidenceItem:
    id: str
    uri: str
    title: str
    snippet: str
    score: float | None = None
    source: str = "memory"
    type: str = "note"
    valid_at: str | None = None
    invalid_at: str | None = None
    trust_tier: TrustTier = TrustTier.AGENT_INFERRED
    supports: tuple[str, ...] = ()
    contradicts: tuple[str, ...] = ()
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["trust_tier"] = self.trust_tier.value
        return out


@dataclass(frozen=True)
class EvidencePack:
    question: str
    status: AnswerStatus
    items: tuple[EvidenceItem, ...] = ()
    queries: tuple[str, ...] = ()
    claims: tuple[dict[str, Any], ...] = ()
    confidence: float = 0.0
    coverage: float = 0.0
    token_estimate: int = 0
    abstention_reason: str = ""
    schema: str = MEMO_EVIDENCE_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "question": self.question,
            "status": self.status.value,
            "items": [item.to_dict() for item in self.items],
            "queries": list(self.queries),
            "claims": list(self.claims),
            "confidence": self.confidence,
            "coverage": self.coverage,
            "token_estimate": self.token_estimate,
            "abstention_reason": self.abstention_reason,
        }


@dataclass(frozen=True)
class MemoEvent:
    event_id: str
    ts: str
    device_id: str
    sequence: int
    op: str
    subject_uri: str
    actor: ActorIdentity = field(default_factory=ActorIdentity)
    trace_id: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    content_hash: str = ""
    previous_hash: str = ""
    event_hash: str = ""
    schema: str = MEMO_EVENT_SCHEMA

    def unsigned_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["actor"] = self.actor.to_dict()
        data.pop("event_hash", None)
        return data

    def with_hash(self) -> MemoEvent:
        digest = hashlib.sha256(
            json.dumps(
                self.unsigned_dict(),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        return replace(self, event_hash=digest)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["actor"] = self.actor.to_dict()
        return data


@dataclass(frozen=True)
class WriteReceipt:
    receipt_id: str
    operation: str
    subject_uri: str
    trace_id: str = ""
    actor_id: str = "memo"
    event_hash: str = ""
    generated_at: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    schema: str = MEMO_RECEIPT_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def new_event_id() -> str:
    return secrets.token_hex(16)


def normalize_provenance(
    extra: dict[str, Any] | None,
    *,
    preserve_legacy: bool = False,
) -> dict[str, Any]:
    """Return memo-native provenance from a record ``extra`` bag.

    Native values win over legacy aliases.  ``preserve_legacy`` keeps the
    original keys under ``legacy_provenance`` for one-way migration receipts;
    normal writes never emit the old vocabulary.
    """
    if not extra:
        return {}
    nested = extra.get("provenance")
    nested_values = dict(nested) if isinstance(nested, dict) else {}
    out = {key: value for key, value in nested_values.items() if key not in LEGACY_PROVENANCE_KEYS}
    legacy: dict[str, Any] = {}
    for old_key, native_key in _LEGACY_TO_NATIVE.items():
        if old_key not in nested_values:
            continue
        legacy[old_key] = nested_values[old_key]
        out.setdefault(native_key, nested_values[old_key])
    for key in PROVENANCE_KEYS:
        if key in extra and key not in out:
            out[key] = extra[key]
    for old_key, native_key in _LEGACY_TO_NATIVE.items():
        if old_key not in extra:
            continue
        legacy[old_key] = extra[old_key]
        out.setdefault(native_key, extra[old_key])
    if preserve_legacy and legacy:
        out["legacy_provenance"] = legacy
    return {key: value for key, value in out.items() if value not in (None, "", [], {})}


__all__ = [
    "LEGACY_PROVENANCE_KEYS",
    "MEMO_BACKEND_SCHEMA",
    "MEMO_EVENT_SCHEMA",
    "MEMO_OPERATIONAL_SCHEMA",
    "MEMO_RECEIPT_SCHEMA",
    "MEMO_SYNC_SCHEMA",
    "PROVENANCE_KEYS",
    "ActorIdentity",
    "AnswerStatus",
    "EvidenceItem",
    "EvidencePack",
    "MemoEvent",
    "TrustTier",
    "Visibility",
    "WriteReceipt",
    "new_event_id",
    "normalize_provenance",
]
