"""Memo-native write preflight.

Policies are local, deterministic, auditable, and fail closed when the
operational authority cannot be read.  This replaces the former subprocess
freeze check while keeping the public ``WriteRefused`` domain error.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

from memo.contracts import ActorIdentity, TrustTier, Visibility, normalize_provenance
from memo.errors import WriteRefused
from memo.operational import OperationalStore


@dataclass(frozen=True)
class WritePolicyDecision:
    allowed: bool
    policy_version: str
    reason: str
    conflicts: tuple[str, ...] = ()
    visibility: Visibility = Visibility.OWNER
    trust_tier: TrustTier = TrustTier.AGENT_INFERRED
    actor_id: str = "memo"
    override: bool = False

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["visibility"] = self.visibility.value
        out["trust_tier"] = self.trust_tier.value
        return out


def _topic(title: str | None, content: str, tags: list[str] | None) -> str:
    if title and title.strip():
        return title.strip()[:160]
    for tag in tags or ():
        if str(tag).strip():
            return str(tag).strip()[:160]
    for line in content.splitlines():
        if line.strip():
            return line.strip()[:160]
    return ""


def _coerce_visibility(extra: dict[str, Any]) -> Visibility:
    raw = str(extra.get("visibility") or Visibility.OWNER.value)
    try:
        return Visibility(raw)
    except ValueError as exc:
        raise ValueError("visibility must be local_only|owner|shared") from exc


_TRUST_RANK = {
    TrustTier.EXTERNAL_UNTRUSTED: 0,
    TrustTier.AGENT_INFERRED: 1,
    TrustTier.AGENT_VERIFIED: 2,
    TrustTier.TOOL_OBSERVED: 3,
    TrustTier.HUMAN: 4,
}


def _actor_trust_ceiling(actor: ActorIdentity) -> TrustTier:
    if actor.actor_kind == "human":
        return TrustTier.HUMAN
    if actor.actor_kind == "tool":
        return TrustTier.TOOL_OBSERVED
    if actor.actor_kind == "device":
        return TrustTier.AGENT_VERIFIED
    return TrustTier.AGENT_INFERRED


def _coerce_trust(extra: dict[str, Any], actor: ActorIdentity) -> TrustTier:
    ceiling = _actor_trust_ceiling(actor)
    raw = str(extra.get("trust_tier") or "")
    if raw:
        try:
            requested = TrustTier(raw)
        except ValueError as exc:
            raise ValueError(f"unknown trust_tier: {raw}") from exc
        if _TRUST_RANK[requested] > _TRUST_RANK[ceiling]:
            raise ValueError(
                f"trust_tier {requested.value!r} exceeds the "
                f"{actor.actor_kind} actor ceiling {ceiling.value!r}"
            )
        return requested
    return ceiling


def actor_for_existing_record(extra: dict[str, Any] | None) -> ActorIdentity:
    """Represent the ceiling already persisted on an authoritative record."""
    bag = dict(extra or {})
    provenance = normalize_provenance(bag)
    try:
        trust = TrustTier(str(bag.get("trust_tier") or TrustTier.AGENT_INFERRED.value))
    except ValueError:
        trust = TrustTier.EXTERNAL_UNTRUSTED
    actor_kind: Literal["human", "agent", "tool", "system", "device"] = (
        "human"
        if trust is TrustTier.HUMAN
        else "tool"
        if trust is TrustTier.TOOL_OBSERVED
        else "device"
        if trust is TrustTier.AGENT_VERIFIED
        else "agent"
    )
    return ActorIdentity(
        actor_id=str(provenance.get("actor_id") or "memo-existing"),
        actor_kind=actor_kind,
        signature=str(provenance.get("actor_signature") or ""),
        source_client=str(provenance.get("source_client") or ""),
    )


class WritePolicyEngine:
    VERSION = "memo.write_policy.v1"

    def __init__(self, operational: OperationalStore) -> None:
        self.operational = operational

    def preflight(
        self,
        *,
        title: str | None,
        content: str,
        tags: list[str] | None,
        extra: dict[str, Any] | None,
        actor: ActorIdentity | None = None,
        allow_conflict_override: bool = False,
        override_reason: str = "",
    ) -> WritePolicyDecision:
        bag = dict(extra or {})
        provenance = normalize_provenance(bag)
        resolved_actor = actor or ActorIdentity(
            actor_id=str(provenance.get("actor_id") or "memo"),
            actor_kind=("agent" if provenance.get("actor_id") else "system"),
            signature=str(provenance.get("actor_signature") or ""),
            source_client=str(provenance.get("source_client") or ""),
        )
        visibility = _coerce_visibility(bag)
        trust = _coerce_trust(bag, resolved_actor)
        principals = bag.get("principals") or []
        if visibility is Visibility.SHARED and not principals:
            return WritePolicyDecision(
                allowed=False,
                policy_version=self.VERSION,
                reason="shared memories require explicit principals",
                visibility=visibility,
                trust_tier=trust,
                actor_id=resolved_actor.actor_id,
            )
        topic = _topic(title, content, tags)
        conflicts = self.operational.active_conflicts(topic)
        blocking = [
            row
            for row in conflicts
            if row.get("freeze_write")
            and row.get("lifecycle_state") in {"detected", "acknowledged"}
        ]
        if blocking and not allow_conflict_override:
            return WritePolicyDecision(
                allowed=False,
                policy_version=self.VERSION,
                reason=f"write frozen by native conflict: {blocking[0].get('summary') or topic}",
                conflicts=tuple(str(row.get("id") or "") for row in blocking),
                visibility=visibility,
                trust_tier=trust,
                actor_id=resolved_actor.actor_id,
            )
        if blocking and allow_conflict_override and not override_reason.strip():
            return WritePolicyDecision(
                allowed=False,
                policy_version=self.VERSION,
                reason="conflict override requires a non-empty reason",
                conflicts=tuple(str(row.get("id") or "") for row in blocking),
                visibility=visibility,
                trust_tier=trust,
                actor_id=resolved_actor.actor_id,
            )
        if blocking and allow_conflict_override and resolved_actor.actor_kind != "human":
            return WritePolicyDecision(
                allowed=False,
                policy_version=self.VERSION,
                reason="conflict override requires authenticated human authority",
                conflicts=tuple(str(row.get("id") or "") for row in blocking),
                visibility=visibility,
                trust_tier=trust,
                actor_id=resolved_actor.actor_id,
            )
        return WritePolicyDecision(
            allowed=True,
            policy_version=self.VERSION,
            reason=(f"human override: {override_reason.strip()}" if blocking else "allowed"),
            conflicts=tuple(str(row.get("id") or "") for row in blocking),
            visibility=visibility,
            trust_tier=trust,
            actor_id=resolved_actor.actor_id,
            override=bool(blocking),
        )

    @staticmethod
    def enforce(decision: WritePolicyDecision) -> None:
        if decision.allowed:
            return
        raise WriteRefused(
            {
                "conflict_id": decision.conflicts[0] if decision.conflicts else "memo-policy",
                "summary": decision.reason,
                "freeze_write": True,
                "lifecycle_state": "detected",
                "policy_version": decision.policy_version,
            }
        )


__all__ = ["WritePolicyDecision", "WritePolicyEngine", "actor_for_existing_record"]
