"""Deterministic, prepared-only migration from operational v1 to v2."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

from memo.atomic_io import atomic_write_text, authority_write_lock, open_secure_directory
from memo.contracts import MemoEvent
from memo.errors import OperationalError, OperationalErrorCode
from memo.identity import PrincipalIdentity
from memo.operation_ledger_v1 import LedgerIntegrityError, LegacyOperationLedger
from memo.operation_ledger_v2 import OperationLedgerV2
from memo.operation_views import OperationalViewStore
from memo.operational import OperationalStore
from memo.operational_epoch import CommitContext, EpochFence
from memo.operational_event import (
    MigrationOrigin,
    MigrationPreparedStamp,
    OperationalCommand,
    SourceProof,
    authenticate_source_proofs,
    canonical_json_bytes,
    canonical_signed_bytes,
)
from memo.operational_event_types import (
    ATTENTION_ADDED,
    CONFLICT_OPENED,
    FOCUS_SET,
    HANDOFF_CREATED,
    OUTCOME_RECORDED,
)
from memo.operational_key_store import AuthorityPinStore
from memo.operational_roster import VerificationRoster
from memo.operational_signing import (
    OperationalSigner,
    OperationalVerifier,
    SignatureEnvelope,
)

_PLAN_SCHEMA = "memo.operational_v1_migration_plan.v1"
_PREPARED_SCHEMA = "memo.operational_migration_prepared.v1"
_PREPARED_FILE = "migration-v1.json"
_ACTIVATION_FILE = "operational-v2-activated.json"
_DOMAINS = ("focus", "handoffs", "attention", "conflicts", "outcomes")
_EVENT_TYPES = {
    "focus": FOCUS_SET,
    "handoffs": HANDOFF_CREATED,
    "attention": ATTENTION_ADDED,
    "conflicts": CONFLICT_OPENED,
    "outcomes": OUTCOME_RECORDED,
}


def _failure(
    message: str,
    *,
    code: OperationalErrorCode = OperationalErrorCode.INVALID_EVENT,
    details: dict[str, Any] | None = None,
) -> OperationalError:
    return OperationalError(
        code,
        message,
        retryable=False,
        details=details,
    )


def _sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_timestamp(value: str) -> str:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise _failure("migration authority timestamp must include a timezone")
    return (
        parsed.astimezone(UTC)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _frozen_json(value: object) -> object:
    """Return a detached canonical-JSON value suitable for a frozen plan."""
    return json.loads(canonical_json_bytes(value))


@dataclass(frozen=True)
class MigrationSeed:
    domain: str
    entity_id: str
    event_id: str
    event_type: str
    project: str
    workspace: str
    subject_uri: str
    payload: Mapping[str, object]
    source_proof: SourceProof


@dataclass(frozen=True)
class V1MigrationPlan:
    schema: str
    source_root: str
    local_origin: str
    workspace: str
    source_manifest_sha256: str
    source_state_sha256: str
    source_snapshot_state_sha256: str
    source_state: Mapping[str, object]
    source_heads: tuple[tuple[str, str], ...]
    source_proof_root_sha256: str
    source_proof_count: int
    attempt_id: str
    seeds: tuple[MigrationSeed, ...]


@dataclass(frozen=True)
class V1MigrationAuthority:
    """Explicit authority inputs required to build one prepared generation."""

    signer: OperationalSigner
    verifier: OperationalVerifier
    roster: VerificationRoster
    roster_root: Path
    pin_store: AuthorityPinStore
    epoch_fence: EpochFence
    attestor_key_id: str
    capability_manifest_sha256: str
    authority_epoch: int
    control_oid: str
    issued_at: str
    expires_at: str


@dataclass(frozen=True)
class MigrationParityReport:
    equal: bool
    v1_state_sha256: str
    v2_state_sha256: str
    diff: tuple[str, ...]


@dataclass(frozen=True)
class MigrationReport:
    source_manifest_sha256: str
    target_generation_sha256: str
    events_inserted: int
    v1_state_sha256: str
    v2_state_sha256: str
    seed_event_ids: tuple[str, ...]
    parity: MigrationParityReport
    prepared_stamp: MigrationPreparedStamp


def _domain_state(state: Mapping[str, object]) -> dict[str, object]:
    return {
        "schema": state.get("schema"),
        **{
            domain: _frozen_json(state.get(domain, {}))
            for domain in _DOMAINS
        },
    }


def _source_state_and_contributors(
    events: Sequence[MemoEvent],
) -> tuple[dict[str, object], dict[tuple[str, str], MemoEvent]]:
    state = OperationalStore._empty()
    contributors: dict[tuple[str, str], MemoEvent] = {}
    for candidate in events:
        op = candidate.op
        payload = dict(candidate.payload)
        OperationalStore._apply(state, op, payload)
        domain = ""
        entity_id = ""
        if op == "focus.set":
            domain, entity_id = "focus", str(payload.get("project") or "")
        elif op == "focus.clear":
            contributors.pop(("focus", str(payload.get("project") or "")), None)
        elif op in {"handoff.create", "handoff.consume"}:
            domain, entity_id = "handoffs", str(payload.get("id") or "")
        elif op in {"attention.add", "attention.ack"}:
            domain, entity_id = "attention", str(payload.get("id") or "")
        elif op in {"conflict.open", "conflict.resolve"}:
            domain, entity_id = "conflicts", str(payload.get("id") or "")
        elif op == "outcome.record":
            domain, entity_id = "outcomes", str(payload.get("task_id") or "")
        elif (
            op == "anomaly.record"
            and payload.get("kind") == "semantic_contradiction"
        ):
            domain, entity_id = "conflicts", str(payload.get("anomaly_id") or "")
        if domain and entity_id:
            rows = state.get(domain)
            if isinstance(rows, dict) and entity_id in rows:
                contributors[(domain, entity_id)] = candidate
    return _domain_state(state), contributors


def _snapshot_oracle_sha256(
    source_root: Path,
    *,
    heads: Mapping[str, str],
    source_state: Mapping[str, object],
) -> str:
    """Compare a current v1 snapshot without ever treating it as authority."""
    path = source_root / "operational-state.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, TypeError, json.JSONDecodeError):
        return ""
    if (
        not isinstance(value, dict)
        or value.get("journal_heads") != dict(heads)
        or value.get("schema") != source_state.get("schema")
    ):
        return ""
    oracle = _domain_state(value)
    diff = _dict_diff(source_state, oracle)
    if diff:
        raise _failure(
            "current operational-state.json disagrees with verified v1 replay",
            code=OperationalErrorCode.ANCHOR_CONFLICT,
            details={"diff": list(diff)},
        )
    return _sha256(oracle)


def _raw_source_proof(event: MemoEvent) -> SourceProof:
    return SourceProof(
        source_system="memo_v1",
        source_event_id=event.event_id,
        source_schema=event.schema,
        source_origin=event.device_id,
        source_sequence=event.sequence,
        source_previous_hash=event.previous_hash,
        source_event_hash=event.event_hash,
        source_content_hash=event.content_hash,
        source_actor=event.actor.to_dict(),
        source_subject_uri=event.subject_uri,
    )


def _seed_specs(
    *,
    manifest_sha256: str,
    workspace: str,
    state: Mapping[str, object],
    contributors: Mapping[tuple[str, str], MemoEvent],
) -> tuple[MigrationSeed, ...]:
    pending: list[MigrationSeed] = []
    proofs: list[SourceProof] = []
    for domain in _DOMAINS:
        rows = state.get(domain)
        if not isinstance(rows, Mapping):
            raise _failure(f"v1 domain state is invalid: {domain}")
        for raw_id in sorted(rows, key=str):
            entity_id = str(raw_id)
            row = rows[raw_id]
            if not isinstance(row, Mapping):
                raise _failure(f"v1 domain row is invalid: {domain}/{entity_id}")
            contributor = contributors.get((domain, entity_id))
            if contributor is None:
                raise _failure(
                    f"v1 domain row lacks a source event: {domain}/{entity_id}"
                )
            encoded_id = quote(entity_id, safe="")
            event_id = f"memo-v1/{manifest_sha256}/{domain}/{encoded_id}"
            payload = _frozen_json(row)
            assert isinstance(payload, dict)
            project = str(payload.get("project") or "")
            proof = _raw_source_proof(contributor)
            proofs.append(proof)
            pending.append(
                MigrationSeed(
                    domain=domain,
                    entity_id=entity_id,
                    event_id=event_id,
                    event_type=_EVENT_TYPES[domain],
                    project=project,
                    workspace=workspace,
                    subject_uri=f"memo://{domain}/{encoded_id}",
                    payload=payload,
                    source_proof=proof,
                )
            )
    if not pending:
        return ()
    proof_root, authenticated = authenticate_source_proofs(
        tuple(proofs),
        source_manifest_sha256=manifest_sha256,
    )
    del proof_root
    return tuple(
        replace(seed, source_proof=proof)
        for seed, proof in zip(pending, authenticated, strict=True)
    )


def plan_v1_migration(
    source: Path,
    *,
    device_id: str,
    workspace: str = "",
) -> V1MigrationPlan:
    """Verify and freeze a deterministic, side-effect-free v1 migration plan."""
    source_root = Path(source).expanduser().absolute()
    legacy = LegacyOperationLedger(source_root, device_id=device_id)
    try:
        manifest_before = OperationLedgerV2.legacy_manifest_sha256(legacy)
        events = legacy.validated_events()
        heads = legacy.head_hashes()
        manifest_sha256 = OperationLedgerV2.legacy_manifest_sha256(legacy)
    except OperationalError:
        raise
    except (LedgerIntegrityError, OSError, TypeError, ValueError) as exc:
        raise _failure(
            f"v1 verification failed: {exc}",
            code=OperationalErrorCode.ANCHOR_CONFLICT,
        ) from exc
    if manifest_before != manifest_sha256:
        raise _failure(
            "v1 source manifest changed while migration planning was in progress",
            code=OperationalErrorCode.ANCHOR_CONFLICT,
        )
    if events and device_id not in heads:
        raise _failure(
            "non-empty v1 migration requires the enrolled local source origin",
            code=OperationalErrorCode.ANCHOR_CONFLICT,
        )
    state, contributors = _source_state_and_contributors(events)
    state_sha256 = _sha256(state)
    snapshot_state_sha256 = _snapshot_oracle_sha256(
        source_root,
        heads=heads,
        source_state=state,
    )
    seeds = _seed_specs(
        manifest_sha256=manifest_sha256,
        workspace=workspace,
        state=state,
        contributors=contributors,
    )
    if seeds:
        source_proof_root_sha256, _ = authenticate_source_proofs(
            tuple(_raw_source_proof(contributors[(seed.domain, seed.entity_id)]) for seed in seeds),
            source_manifest_sha256=manifest_sha256,
        )
    else:
        source_proof_root_sha256 = ""
    return V1MigrationPlan(
        schema=_PLAN_SCHEMA,
        source_root=str(source_root),
        local_origin=device_id,
        workspace=workspace,
        source_manifest_sha256=manifest_sha256,
        source_state_sha256=state_sha256,
        source_snapshot_state_sha256=snapshot_state_sha256,
        source_state=state,
        source_heads=tuple(sorted(heads.items())),
        source_proof_root_sha256=source_proof_root_sha256,
        source_proof_count=len(seeds),
        attempt_id=f"memo-v1-{manifest_sha256}",
        seeds=seeds,
    )


def _validate_authority(
    plan: V1MigrationPlan,
    authority: V1MigrationAuthority,
) -> None:
    if authority.roster.local_device_id != plan.local_origin:
        raise _failure("migration authority local origin differs from the v1 plan")
    origins = {origin for origin, _ in plan.source_heads}
    missing = origins.difference(authority.roster.peers)
    if missing:
        raise _failure(
            "v1 origins are absent from the migration roster",
            code=OperationalErrorCode.SIGNATURE_INVALID,
            details={"origins": sorted(missing)},
        )
    if authority.signer.roster_version != authority.roster.version:
        raise _failure(
            "migration signer roster is stale",
            code=OperationalErrorCode.SIGNATURE_INVALID,
        )
    key = authority.roster.key(authority.attestor_key_id)
    if key.roles != ("migration_attestor",):
        raise _failure(
            "migration attestor must have the exclusive migration_attestor role",
            code=OperationalErrorCode.SIGNATURE_INVALID,
        )
    if not _is_sha256(authority.capability_manifest_sha256):
        raise _failure("migration capability manifest digest is invalid")


def _ledger(
    root: Path,
    *,
    plan: V1MigrationPlan,
    authority: V1MigrationAuthority,
) -> OperationLedgerV2:
    return OperationLedgerV2(
        root,
        device_id=plan.local_origin,
        clock=lambda: authority.issued_at,
        signer=authority.signer,
        verifier=authority.verifier,
        roster=authority.roster,
        roster_root=authority.roster_root,
        pin_store=authority.pin_store,
        epoch_fence=authority.epoch_fence,
    )


def _migration_origin(
    plan: V1MigrationPlan,
    authority: V1MigrationAuthority,
) -> MigrationOrigin:
    unsigned = MigrationOrigin(
        schema="memo.operational_migration_origin.v1",
        attempt_id=plan.attempt_id,
        migration_device_id=plan.local_origin,
        source_manifest_sha256=plan.source_manifest_sha256,
        capability_manifest_sha256=authority.capability_manifest_sha256,
        attestor_device_id=authority.roster.key(
            authority.attestor_key_id
        ).device_id,
        attestor_key_id=authority.attestor_key_id,
        roster_version=authority.roster.version,
        issued_at=authority.issued_at,
        expires_at=authority.expires_at,
        signature="",
        source_proof_root_sha256=plan.source_proof_root_sha256,
        source_proof_count=plan.source_proof_count,
    )
    envelope = authority.signer.sign(
        domain="memo.operational.migration_origin.v1",
        payload=canonical_signed_bytes(unsigned),
        key_id=authority.attestor_key_id,
    )
    return replace(unsigned, signature=envelope.signature)


def _commit_context(
    plan: V1MigrationPlan,
    authority: V1MigrationAuthority,
    migration_origin: MigrationOrigin,
) -> CommitContext:
    identity = _migration_identity(plan)
    context = authority.epoch_fence.context(
        identity,
        request_epoch=authority.authority_epoch,
        request_control_oid=authority.control_oid,
    )
    return replace(context, migration_origin=migration_origin)


def _migration_identity(plan: V1MigrationPlan) -> PrincipalIdentity:
    return PrincipalIdentity(
        principal_id=f"{plan.local_origin}:{plan.attempt_id}",
        actor_id="memo-v1-migration",
        kind="system",
        device_id=plan.local_origin,
        session_id=plan.attempt_id,
        source_client="memo-migration",
    )


def _command(
    seed: MigrationSeed,
    *,
    identity: PrincipalIdentity,
) -> OperationalCommand:
    return OperationalCommand(
        event_type=seed.event_type,
        actor=identity,
        target_id=seed.entity_id,
        project=seed.project,
        workspace=seed.workspace,
        expires_at=None,
        visibility="owner",
        idempotency_key=seed.event_id,
        caused_by=(seed.source_proof.source_event_id,),
        subject_uri=seed.subject_uri,
        trace_id=f"migration/{seed.source_proof.source_event_id}",
        payload=seed.payload,
        source_proof=seed.source_proof,
    )


def _dict_diff(left: object, right: object, prefix: str = "") -> tuple[str, ...]:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        paths: list[str] = []
        keys = sorted(set(left).union(right), key=str)
        for key in keys:
            child = f"{prefix}/{key}" if prefix else str(key)
            if key not in left or key not in right:
                paths.append(child)
            else:
                paths.extend(_dict_diff(left[key], right[key], child))
        return tuple(paths)
    if left != right:
        return (prefix or "<root>",)
    return ()


def _verify_v1_parity_locked(
    plan: V1MigrationPlan,
    target: Path,
    *,
    authority: V1MigrationAuthority,
) -> MigrationParityReport:
    """Compare canonical v1 domains with the current prepared v2 view."""
    _assert_source_unchanged(plan)
    target_root = Path(target).expanduser().absolute()
    ledger = _ledger(target_root, plan=plan, authority=authority)
    verification = ledger.verify()
    if not verification.ok:
        raise _failure(
            "prepared v2 ledger verification failed",
            code=OperationalErrorCode.ANCHOR_CONFLICT,
            details={"errors": list(verification.errors)},
        )
    views = OperationalViewStore(target_root / "operational.db")
    v1_state = _domain_state(plan.source_state)
    v2_state = _domain_state(views.state())
    v1_sha256 = _sha256(v1_state)
    v2_sha256 = _sha256(v2_state)
    diff = _dict_diff(v1_state, v2_state)
    return MigrationParityReport(
        equal=not diff and v1_sha256 == v2_sha256,
        v1_state_sha256=v1_sha256,
        v2_state_sha256=v2_sha256,
        diff=diff,
    )


def verify_v1_parity(
    plan: V1MigrationPlan,
    target: Path,
    *,
    authority: V1MigrationAuthority,
) -> MigrationParityReport:
    """Verify source and prepared state under one stable v1 journal fence."""
    with authority_write_lock(Path(plan.source_root) / "journal"):
        return _verify_v1_parity_locked(
            plan,
            target,
            authority=authority,
        )


def _generation_sha256(
    plan: V1MigrationPlan,
    ledger: OperationLedgerV2,
    parity: MigrationParityReport,
) -> str:
    bundles = ledger.export_bundles()
    return _sha256(
        {
            "schema": "memo.operational_prepared_generation.v1",
            "source_manifest_sha256": plan.source_manifest_sha256,
            "anchors": [
                {
                    "origin": bundle.anchor.origin_device,
                    "anchor_hash": bundle.anchor.anchor_hash,
                    "head_sequence": bundle.head_sequence,
                    "head_hash": bundle.head_hash,
                }
                for bundle in bundles
            ],
            "events": [
                {
                    "event_id": event.event_id,
                    "event_hash": event.event_hash,
                }
                for event in ledger.validated_events()
            ],
            "state_sha256": parity.v2_state_sha256,
        }
    )


def _verify_generation_matches_plan(
    plan: V1MigrationPlan,
    ledger: OperationLedgerV2,
    *,
    authority: V1MigrationAuthority,
) -> None:
    bundles = ledger.export_bundles()
    anchors = {
        bundle.anchor.origin_device: bundle.anchor
        for bundle in bundles
    }
    if set(anchors) != {origin for origin, _ in plan.source_heads}:
        raise _failure(
            "prepared generation origins do not match the migration plan",
            code=OperationalErrorCode.ANCHOR_CONFLICT,
        )
    for origin, source_head in plan.source_heads:
        anchor = anchors[origin]
        if (
            anchor.kind != "memo_v1"
            or anchor.base_event_hash != source_head
            or anchor.source_manifest_sha256 != plan.source_manifest_sha256
        ):
            raise _failure(
                f"prepared genesis anchor does not match the plan: {origin}",
                code=OperationalErrorCode.ANCHOR_CONFLICT,
            )
    events = ledger.validated_events()
    if tuple(event.event_id for event in events) != tuple(
        seed.event_id for seed in plan.seeds
    ):
        raise _failure(
            "prepared seed event identities do not match the migration plan",
            code=OperationalErrorCode.ANCHOR_CONFLICT,
        )
    if not plan.seeds:
        return
    identity = _migration_identity(plan)
    migration_origin = _migration_origin(plan, authority)
    migration_digest = _sha256(migration_origin)
    created_at = _canonical_timestamp(authority.issued_at)
    for seed, event in zip(plan.seeds, events, strict=True):
        command = _command(seed, identity=identity)
        content_hash = hashlib.sha256(
            canonical_json_bytes(asdict(command))
        ).hexdigest()
        checks = {
            "event_type": (event.event_type, seed.event_type),
            "actor": (event.actor, identity),
            "target_id": (event.target_id, seed.entity_id),
            "project": (event.project, seed.project),
            "workspace": (event.workspace, seed.workspace),
            "origin_device": (event.origin_device, plan.local_origin),
            "authority_epoch": (
                event.authority_epoch,
                authority.authority_epoch,
            ),
            "control_oid": (event.control_oid, authority.control_oid),
            "created_at": (event.created_at, created_at),
            "expires_at": (event.expires_at, None),
            "visibility": (event.visibility, "owner"),
            "idempotency_key": (event.idempotency_key, seed.event_id),
            "caused_by": (event.caused_by, command.caused_by),
            "subject_uri": (event.subject_uri, seed.subject_uri),
            "trace_id": (event.trace_id, command.trace_id),
            "payload": (
                canonical_json_bytes(event.payload),
                canonical_json_bytes(seed.payload),
            ),
            "content_hash": (event.content_hash, content_hash),
            "source_proof": (event.source_proof, seed.source_proof),
            "migration_origin": (event.migration_origin, migration_origin),
            "migration_origin_sha256": (
                event.migration_origin_sha256,
                migration_digest,
            ),
        }
        mismatches = tuple(
            field for field, (actual, expected) in checks.items() if actual != expected
        )
        if mismatches:
            raise _failure(
                (
                    "prepared seed event does not match the plan: "
                    f"{seed.event_id} ({', '.join(mismatches)})"
                ),
                code=OperationalErrorCode.ANCHOR_CONFLICT,
                details={"fields": list(mismatches)},
            )


def _prepared_stamp(
    plan: V1MigrationPlan,
    *,
    authority: V1MigrationAuthority,
    generation_sha256: str,
    parity: MigrationParityReport,
) -> MigrationPreparedStamp:
    unsigned = MigrationPreparedStamp(
        schema="memo.operational_migration_prepared.v1",
        source_manifest_sha256=plan.source_manifest_sha256,
        target_generation_sha256=generation_sha256,
        parity_report_sha256=_sha256(parity),
        attestor_key_id=authority.attestor_key_id,
        signature="",
    )
    envelope = authority.signer.sign(
        domain="memo.operational.migration_prepared.v1",
        payload=canonical_signed_bytes(unsigned),
        key_id=authority.attestor_key_id,
    )
    return replace(unsigned, signature=envelope.signature)


def _decode_stamp(path: Path) -> MigrationPreparedStamp:
    try:
        encoded = path.read_bytes()
        value = json.loads(encoded)
        stamp = MigrationPreparedStamp(**value)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise _failure(
            f"prepared migration stamp is invalid: {path}",
            code=OperationalErrorCode.ANCHOR_CONFLICT,
        ) from exc
    if canonical_json_bytes(stamp) != encoded or stamp.schema != _PREPARED_SCHEMA:
        raise _failure(
            f"prepared migration stamp is not canonical: {path}",
            code=OperationalErrorCode.ANCHOR_CONFLICT,
        )
    if (
        not _is_sha256(stamp.source_manifest_sha256)
        or not _is_sha256(stamp.target_generation_sha256)
        or not _is_sha256(stamp.parity_report_sha256)
        or not stamp.attestor_key_id
        or not stamp.signature
    ):
        raise _failure(
            f"prepared migration stamp fields are invalid: {path}",
            code=OperationalErrorCode.ANCHOR_CONFLICT,
        )
    return stamp


def _verify_stamp(
    stamp: MigrationPreparedStamp,
    *,
    plan: V1MigrationPlan,
    authority: V1MigrationAuthority,
    generation_sha256: str,
    parity: MigrationParityReport,
) -> None:
    if (
        stamp.source_manifest_sha256 != plan.source_manifest_sha256
        or stamp.target_generation_sha256 != generation_sha256
        or stamp.parity_report_sha256 != _sha256(parity)
        or stamp.attestor_key_id != authority.attestor_key_id
    ):
        raise _failure(
            "prepared migration stamp does not match the verified generation",
            code=OperationalErrorCode.ANCHOR_CONFLICT,
        )
    authority.verifier.verify(
        domain="memo.operational.migration_prepared.v1",
        payload=canonical_signed_bytes(stamp),
        envelope=SignatureEnvelope(
            algorithm="ed25519",
            key_id=stamp.attestor_key_id,
            roster_version=authority.roster.version,
            signature=stamp.signature,
        ),
        roster=authority.roster,
    )


def _report_existing(
    plan: V1MigrationPlan,
    target: Path,
    *,
    authority: V1MigrationAuthority,
) -> MigrationReport:
    if target.is_symlink() or not target.is_dir():
        raise _failure(
            "prepared migration target is not a safe directory",
            code=OperationalErrorCode.STORAGE_UNAVAILABLE,
        )
    if (target / _ACTIVATION_FILE).exists():
        raise _failure(
            "an activated v2 generation cannot be replayed as prepared migration",
            code=OperationalErrorCode.ANCHOR_CONFLICT,
        )
    stamp = _decode_stamp(target / _PREPARED_FILE)
    parity = verify_v1_parity(plan, target, authority=authority)
    if not parity.equal:
        raise _failure(
            f"v1/v2 state mismatch: {parity.diff}",
            details={"diff": list(parity.diff)},
        )
    ledger = _ledger(target, plan=plan, authority=authority)
    _verify_generation_matches_plan(
        plan,
        ledger,
        authority=authority,
    )
    generation_sha256 = _generation_sha256(plan, ledger, parity)
    _verify_stamp(
        stamp,
        plan=plan,
        authority=authority,
        generation_sha256=generation_sha256,
        parity=parity,
    )
    return MigrationReport(
        source_manifest_sha256=plan.source_manifest_sha256,
        target_generation_sha256=generation_sha256,
        events_inserted=0,
        v1_state_sha256=parity.v1_state_sha256,
        v2_state_sha256=parity.v2_state_sha256,
        seed_event_ids=tuple(seed.event_id for seed in plan.seeds),
        parity=parity,
        prepared_stamp=stamp,
    )


def _assert_source_unchanged(plan: V1MigrationPlan) -> None:
    observed = plan_v1_migration(
        Path(plan.source_root),
        device_id=plan.local_origin,
        workspace=plan.workspace,
    )
    if observed.source_manifest_sha256 != plan.source_manifest_sha256:
        raise _failure(
            "v1 source manifest changed after migration planning",
            code=OperationalErrorCode.ANCHOR_CONFLICT,
            details={
                "expected": plan.source_manifest_sha256,
                "observed": observed.source_manifest_sha256,
            },
        )
    comparable_observed = replace(
        observed,
        source_snapshot_state_sha256="",
    )
    comparable_plan = replace(
        plan,
        source_snapshot_state_sha256="",
    )
    if canonical_json_bytes(comparable_observed) != canonical_json_bytes(comparable_plan):
        raise _failure(
            "v1 migration plan does not match the verified source manifest",
            code=OperationalErrorCode.ANCHOR_CONFLICT,
        )


def _install_prepared(staging: Path, target: Path) -> None:
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    # The generation marker is the final authority file written in staging.
    # Persist its directory entry before publishing the directory itself so a
    # crash cannot make the target rename durable while losing the marker.
    with open_secure_directory(staging) as directory:
        os.fsync(directory.descriptor)
    os.rename(staging, target)
    with open_secure_directory(target.parent) as directory:
        os.fsync(directory.descriptor)


def apply_v1_migration(
    plan: V1MigrationPlan,
    target: Path,
    *,
    authority: V1MigrationAuthority,
) -> MigrationReport:
    """Build and atomically install one parity-proven, dormant v2 generation."""
    if not isinstance(plan, V1MigrationPlan) or plan.schema != _PLAN_SCHEMA:
        raise TypeError("a v1 migration plan is required")
    if not isinstance(authority, V1MigrationAuthority):
        raise TypeError("v1 migration authority is required")
    _validate_authority(plan, authority)
    _assert_source_unchanged(plan)

    target_root = Path(target).expanduser().absolute()
    source_root = Path(plan.source_root).expanduser().absolute()
    if target_root == source_root or source_root in target_root.parents:
        raise _failure("prepared target must be outside the v1 source root")
    if target_root.exists() or target_root.is_symlink():
        with authority_write_lock(Path(plan.source_root) / "journal"):
            return _report_existing(
                plan,
                target_root,
                authority=authority,
            )

    target_root.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{target_root.name}.staging-{plan.source_manifest_sha256[:12]}-",
            dir=target_root.parent,
        )
    )
    installed = False
    try:
        ledger = _ledger(staging, plan=plan, authority=authority)
        for origin, head_hash in plan.source_heads:
            legacy = LegacyOperationLedger(Path(plan.source_root), device_id=origin)
            ledger.ensure_anchor_from_v1(
                legacy,
                source_head_hash=head_hash,
                migration_attestor=authority.signer,
                attestor_key_id=authority.attestor_key_id,
            )
        if plan.seeds:
            migration_origin = _migration_origin(plan, authority)
            context = _commit_context(plan, authority, migration_origin)
            for seed in plan.seeds:
                ledger.append_migration_seed(
                    _command(seed, identity=context.identity),
                    context=context,
                    event_id=seed.event_id,
                )
        verification = ledger.verify()
        if not verification.ok:
            raise _failure(
                "prepared v2 ledger verification failed",
                code=OperationalErrorCode.ANCHOR_CONFLICT,
                details={"errors": list(verification.errors)},
            )
        _verify_generation_matches_plan(
            plan,
            ledger,
            authority=authority,
        )
        views = OperationalViewStore(staging / "operational.db")
        rebuild = views.rebuild(ledger.validated_events())
        if rebuild.quarantined:
            raise _failure(
                "prepared v2 view quarantined migration events",
                details={"quarantined": rebuild.quarantined},
            )
        with authority_write_lock(Path(plan.source_root) / "journal"):
            parity = verify_v1_parity(plan, staging, authority=authority)
            if not parity.equal:
                raise _failure(
                    f"v1/v2 state mismatch: {parity.diff}",
                    details={"diff": list(parity.diff)},
                )
            generation_sha256 = _generation_sha256(plan, ledger, parity)
            stamp = _prepared_stamp(
                plan,
                authority=authority,
                generation_sha256=generation_sha256,
                parity=parity,
            )
            atomic_write_text(
                staging / _PREPARED_FILE,
                canonical_json_bytes(stamp).decode("utf-8"),
            )
            if (staging / _ACTIVATION_FILE).exists():
                raise _failure("prepared migration created an activation marker")
            try:
                _install_prepared(staging, target_root)
                installed = True
            except FileExistsError:
                return _report_existing(
                    plan,
                    target_root,
                    authority=authority,
                )
        return MigrationReport(
            source_manifest_sha256=plan.source_manifest_sha256,
            target_generation_sha256=generation_sha256,
            events_inserted=len(plan.seeds),
            v1_state_sha256=parity.v1_state_sha256,
            v2_state_sha256=parity.v2_state_sha256,
            seed_event_ids=tuple(seed.event_id for seed in plan.seeds),
            parity=parity,
            prepared_stamp=stamp,
        )
    finally:
        if not installed and staging.exists() and not staging.is_symlink():
            shutil.rmtree(staging)


def migrate_v1(
    source: Path,
    target: Path,
    *,
    device_id: str,
    authority: V1MigrationAuthority,
    workspace: str = "",
) -> MigrationReport:
    """Plan, apply, verify, and install one dormant v2 generation."""
    plan = plan_v1_migration(
        source,
        device_id=device_id,
        workspace=workspace,
    )
    return apply_v1_migration(plan, target, authority=authority)


def inspect_prepared_migration(root: Path) -> dict[str, object]:
    """Return structural prepared evidence without selecting or activating v2."""
    target = Path(root)
    marker = target / _PREPARED_FILE
    activation = target / _ACTIVATION_FILE
    if not marker.exists():
        return {
            "present": False,
            "structurally_valid": False,
            "activated": activation.exists(),
        }
    try:
        stamp = _decode_stamp(marker)
    except OperationalError as exc:
        return {
            "present": True,
            "structurally_valid": False,
            "activated": activation.exists(),
            "error": str(exc),
        }
    return {
        "present": True,
        "structurally_valid": True,
        "activated": activation.exists(),
        "source_manifest_sha256": stamp.source_manifest_sha256,
        "target_generation_sha256": stamp.target_generation_sha256,
        "parity_report_sha256": stamp.parity_report_sha256,
        "attestor_key_id": stamp.attestor_key_id,
    }


__all__ = [
    "MigrationParityReport",
    "MigrationReport",
    "MigrationSeed",
    "V1MigrationAuthority",
    "V1MigrationPlan",
    "apply_v1_migration",
    "inspect_prepared_migration",
    "migrate_v1",
    "plan_v1_migration",
    "verify_v1_parity",
]
