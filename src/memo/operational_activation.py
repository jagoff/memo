"""Verified activation and reopening of Memo's operational ledger v2."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from memo.atomic_io import atomic_write_text, authority_write_lock
from memo.errors import OperationalError, OperationalErrorCode
from memo.flags import flag_bool
from memo.identity import PrincipalIdentity
from memo.operation_ledger_v2 import OperationLedgerV2
from memo.operation_views import OperationalViewStore
from memo.operational import OperationalStore
from memo.operational_epoch import EpochFence, bind_system_context
from memo.operational_event import (
    EpochMarkerAuthorization,
    canonical_json_bytes,
    canonical_signed_bytes,
)
from memo.operational_key_store import AuthorityPinStore, DeviceKeyStore
from memo.operational_roster import VerificationRoster
from memo.operational_signing import (
    OperationalSigner,
    OperationalVerifier,
    SignatureEnvelope,
)
from memo.util import utc_now_iso

if TYPE_CHECKING:
    from memo.config import Config

_ACTIVATION_SCHEMA = "memo.operational_activation.v1"
_ACTIVATION_DOMAIN = "memo.operational.activation.v1"
_ACTIVATION_FILE = "operational-v2-activated.json"


def _failure(message: str) -> OperationalError:
    return OperationalError(
        OperationalErrorCode.STORAGE_UNAVAILABLE,
        message,
        retryable=False,
    )


def _root_sha256(root: Path) -> str:
    return hashlib.sha256(b"memo-operational-root-v1\0" + os.fsencode(root.resolve())).hexdigest()


def _file_sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise _failure(f"operational activation artifact is unavailable: {path.name}") from exc


@dataclass(frozen=True)
class OperationalActivationStamp:
    schema: Literal["memo.operational_activation.v1"]
    backend_version: Literal[2]
    device_id: str
    operational_root_sha256: str
    roster_version: int
    roster_hash: str
    authority_epoch: int
    control_oid: str
    anchor_hash: str
    anchor_sha256: str
    reducer_version: int
    activated_at: str
    key_id: str
    signature: SignatureEnvelope


@dataclass(frozen=True)
class OperationalRuntimeAuthority:
    signer: OperationalSigner
    roster: VerificationRoster
    pin_store: AuthorityPinStore
    fence: EpochFence
    key_id: str


def _identity(device_id: str) -> PrincipalIdentity:
    return PrincipalIdentity(
        principal_id=f"memo:{device_id}",
        actor_id="memo-runtime",
        kind="system",
        device_id=device_id,
        session_id="operational-runtime",
        source_client="memo",
    )


def _store(
    cfg: Config,
    *,
    authority: OperationalRuntimeAuthority,
) -> OperationalStore:
    ledger = OperationLedgerV2(
        cfg.operational_root,
        device_id=cfg.device_id,
        signer=authority.signer,
        verifier=OperationalVerifier(),
        roster=authority.roster,
        roster_root=cfg.operational_root,
        pin_store=authority.pin_store,
        epoch_fence=authority.fence,
    )
    context_operation = bind_system_context(
        authority.fence,
        signer=authority.signer,
        key_id=authority.key_id,
        system_role="daemon",
    )
    identity = _identity(cfg.device_id)
    views = OperationalViewStore(cfg.operational_db)
    views.catch_up(ledger)
    return OperationalStore.for_v2(
        ledger=ledger,
        views=views,
        epoch_fence=authority.fence,
        transaction_root=cfg.operational_root / "transactions",
        context_provider=lambda: context_operation(identity),
    )


def _decode_stamp(path: Path) -> OperationalActivationStamp:
    try:
        encoded = path.read_bytes()
        body = json.loads(encoded)
        signature_body = body["signature"]
        if not isinstance(body, dict) or not isinstance(signature_body, dict):
            raise TypeError
        stamp = OperationalActivationStamp(
            **{
                **body,
                "signature": SignatureEnvelope(**signature_body),
            }
        )
    except (
        FileNotFoundError,
        OSError,
        KeyError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        raise _failure("operational v2 activation stamp is missing or invalid") from exc
    if canonical_json_bytes(stamp) != encoded:
        raise _failure("operational v2 activation stamp is not canonical")
    return stamp


def _verify_stamp(
    cfg: Config,
    stamp: OperationalActivationStamp,
    *,
    authority: OperationalRuntimeAuthority,
) -> OperationLedgerV2:
    if (
        stamp.schema != _ACTIVATION_SCHEMA
        or stamp.backend_version != 2
        or stamp.device_id != cfg.device_id
        or stamp.operational_root_sha256 != _root_sha256(cfg.operational_root)
        or stamp.roster_version != authority.roster.version
        or stamp.roster_hash != authority.roster.roster_hash
        or stamp.key_id != authority.key_id
    ):
        raise _failure("operational v2 activation stamp authority binding is invalid")
    try:
        OperationalVerifier().verify(
            domain=_ACTIVATION_DOMAIN,
            payload=canonical_signed_bytes(stamp),
            envelope=stamp.signature,
            roster=authority.roster,
        )
    except OperationalError as exc:
        raise _failure("operational v2 activation signature is invalid") from exc
    authority.fence.context(
        _identity(cfg.device_id),
        request_epoch=stamp.authority_epoch,
        request_control_oid=stamp.control_oid,
    )
    ledger = OperationLedgerV2(
        cfg.operational_root,
        device_id=cfg.device_id,
        signer=authority.signer,
        verifier=OperationalVerifier(),
        roster=authority.roster,
        roster_root=cfg.operational_root,
        pin_store=authority.pin_store,
        epoch_fence=authority.fence,
        reducer_version=stamp.reducer_version,
    )
    anchor = ledger.anchor(cfg.device_id)
    anchor_path = ledger.anchors_dir / f"{cfg.device_id}.json"
    if anchor.anchor_hash != stamp.anchor_hash or _file_sha256(anchor_path) != stamp.anchor_sha256:
        raise _failure("operational v2 activation anchor binding is invalid")
    report = ledger.verify()
    if not report.ok:
        raise _failure("activated operational v2 ledger verification failed")
    return ledger


def activate_fresh_operational_v2(
    cfg: Config,
    *,
    authority: OperationalRuntimeAuthority,
) -> OperationalStore:
    """Create, bind, and activate an empty signed v2 generation."""
    root = cfg.operational_root
    activation_path = root / _ACTIVATION_FILE
    if activation_path.exists():
        raise _failure("operational v2 is already activated")
    ledger = OperationLedgerV2(
        root,
        device_id=cfg.device_id,
        signer=authority.signer,
        verifier=OperationalVerifier(),
        roster=authority.roster,
        roster_root=root,
        pin_store=authority.pin_store,
        epoch_fence=authority.fence,
    )
    anchor = ledger.ensure_anchor()
    anchor_path = ledger.anchors_dir / f"{cfg.device_id}.json"
    roster_path = root / "verification-roster.json"
    digests = {
        "bootstrap_roster": _file_sha256(roster_path),
        "empty_anchor": _file_sha256(anchor_path),
    }
    unsigned_epoch = EpochMarkerAuthorization(
        schema="memo.operational_epoch_authorization.v1",
        attempt_id=f"fresh-{cfg.device_id}",
        device_id=cfg.device_id,
        epoch=0,
        control_oid=hashlib.sha256(
            canonical_json_bytes(
                {"anchor": anchor.anchor_hash, "roster": authority.roster.roster_hash}
            )
        ).hexdigest(),
        artifact_digests=digests,
        roster_version=authority.roster.version,
        key_id=authority.key_id,
        signature=None,  # type: ignore[arg-type]
    )
    epoch = replace(
        unsigned_epoch,
        signature=authority.signer.sign(
            domain="memo.operational_epoch_authorization.v1",
            payload=canonical_signed_bytes(unsigned_epoch),
            key_id=authority.key_id,
        ),
    )
    authority.fence.bootstrap(
        authorization=epoch,
        observed_artifact_digests=digests,
    )
    unsigned_stamp = OperationalActivationStamp(
        schema="memo.operational_activation.v1",
        backend_version=2,
        device_id=cfg.device_id,
        operational_root_sha256=_root_sha256(root),
        roster_version=authority.roster.version,
        roster_hash=authority.roster.roster_hash,
        authority_epoch=epoch.epoch,
        control_oid=epoch.control_oid,
        anchor_hash=anchor.anchor_hash,
        anchor_sha256=_file_sha256(anchor_path),
        reducer_version=ledger.reducer_version,
        activated_at=utc_now_iso(),
        key_id=authority.key_id,
        signature=None,  # type: ignore[arg-type]
    )
    stamp = replace(
        unsigned_stamp,
        signature=authority.signer.sign(
            domain=_ACTIVATION_DOMAIN,
            payload=canonical_signed_bytes(unsigned_stamp),
            key_id=authority.key_id,
        ),
    )
    with authority_write_lock(root):
        atomic_write_text(
            activation_path,
            canonical_json_bytes(stamp).decode("utf-8"),
        )
    OperationalViewStore(cfg.operational_db).rebuild(ledger.validated_events())
    return _store(cfg, authority=authority)


def open_activated_operational_v2(
    cfg: Config,
    *,
    authority: OperationalRuntimeAuthority,
) -> OperationalStore:
    """Open v2 only after verifying every signed activation binding."""
    stamp = _decode_stamp(cfg.operational_root / _ACTIVATION_FILE)
    _verify_stamp(cfg, stamp, authority=authority)
    return _store(cfg, authority=authority)


def build_fresh_productive_authority(cfg: Config) -> OperationalRuntimeAuthority:
    """Enroll the first productive Secure Enclave origin for a fresh install."""
    root = cfg.operational_root
    pin_store = AuthorityPinStore.for_root(root)
    keys = DeviceKeyStore()
    key = keys.generate(device_id=cfg.device_id, roles=("origin",))
    roster = VerificationRoster.bootstrap(
        device_id=cfg.device_id,
        key=key,
        root=root,
        pin_store=pin_store,
    )
    signer = OperationalSigner(keys, roster_version=roster.version)
    fence = EpochFence(
        root,
        roster=roster,
        verifier=OperationalVerifier(),
        pin_store=pin_store,
    )
    return OperationalRuntimeAuthority(
        signer=signer,
        roster=roster,
        pin_store=pin_store,
        fence=fence,
        key_id=key.key_id,
    )


def _active_origin_key(roster: VerificationRoster) -> str:
    matches = [
        key.key_id
        for key in roster.keys
        if key.device_id == roster.local_device_id
        and "origin" in key.roles
        and key.revocation_sequence is None
    ]
    if len(matches) != 1:
        raise _failure("operational v2 has no unique active local origin key")
    return matches[0]


def _authority_from_config(cfg: Config) -> OperationalRuntimeAuthority | None:
    signer = cfg.operational_signer
    fence = cfg.operational_epoch_fence
    if signer is None and fence is None:
        return None
    if signer is None and fence is not None and cfg.operational_context_provider is not None:
        # Explicitly composed legacy migration authority. It remains valid for
        # the v1 selector but is not sufficient to activate or open v2.
        return None
    if signer is None or fence is None:
        raise _failure("operational v2 runtime authority is only partially composed")
    roster = fence.roster
    return OperationalRuntimeAuthority(
        signer=signer,
        roster=roster,
        pin_store=fence.pin_store,
        fence=fence,
        key_id=_active_origin_key(roster),
    )


def _open_productive_authority(cfg: Config) -> OperationalRuntimeAuthority:
    root = cfg.operational_root
    pin_store = AuthorityPinStore.for_root(root)
    roster = VerificationRoster.load(root, pin_store=pin_store)
    keys = DeviceKeyStore()
    signer = OperationalSigner(keys, roster_version=roster.version)
    fence = EpochFence(
        root,
        roster=roster,
        verifier=OperationalVerifier(),
        pin_store=pin_store,
    )
    return OperationalRuntimeAuthority(
        signer=signer,
        roster=roster,
        pin_store=pin_store,
        fence=fence,
        key_id=_active_origin_key(roster),
    )


def select_operational_store(cfg: Config) -> OperationalStore:
    """Select legacy migration authority or a completely verified v2 runtime.

    Existing v1 installs remain on their byte-compatible backend until an
    explicit migration writes the signed activation stamp. Fresh productive
    installs start on v2. A partially created v2 root never falls back to v1.
    """
    activation = cfg.operational_root / _ACTIVATION_FILE
    injected = _authority_from_config(cfg)
    if activation.exists():
        authority = injected or _open_productive_authority(cfg)
        return open_activated_operational_v2(cfg, authority=authority)

    legacy_root = cfg.state_dir / "journal"
    if legacy_root.exists() or (cfg.operational_context_provider is not None and injected is None):
        return OperationalStore(
            cfg.state_dir,
            device_id=cfg.device_id,
            context_provider=cfg.operational_context_provider,
            epoch_fence=cfg.operational_epoch_fence,
        )

    if injected is not None:
        return activate_fresh_operational_v2(cfg, authority=injected)
    if cfg.operational_root.exists():
        raise _failure("partial operational v2 install requires explicit recovery")
    # Productive v2 enrollment currently relies on the macOS Secure Enclave
    # and Keychain providers. Linux has no equivalent provider yet; a fresh
    # CPU install must retain the byte-compatible v1 backend instead of
    # crashing during its first `memo save`.
    if not flag_bool("MEMO_OPERATIONAL_V2_AUTO_ACTIVATE") or sys.platform != "darwin":
        return OperationalStore(cfg.state_dir, device_id=cfg.device_id)
    authority = build_fresh_productive_authority(cfg)
    return activate_fresh_operational_v2(cfg, authority=authority)


__all__ = [
    "OperationalActivationStamp",
    "OperationalRuntimeAuthority",
    "activate_fresh_operational_v2",
    "build_fresh_productive_authority",
    "open_activated_operational_v2",
    "select_operational_store",
]
