"""Hermetic operational authority composition for tests.

The helpers deliberately build the real roster, signer, epoch marker, and
fence using only in-memory private-key and pin providers.  Tests therefore do
not bypass the production fail-closed write contract or touch Keychain.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import mkdtemp
from typing import TYPE_CHECKING

from memo.identity import PrincipalIdentity
from memo.operational_epoch import CommitContext, EpochFence
from memo.operational_event import EpochMarkerAuthorization, canonical_signed_bytes
from memo.operational_key_store import (
    AuthorityPinStore,
    DeviceKeyStore,
    InMemoryAuthorityPinProvider,
)
from memo.operational_roster import VerificationRoster
from memo.operational_signing import OperationalSigner, OperationalVerifier

if TYPE_CHECKING:
    from memo.config import Config
    from memo.operational import OperationalStore


@dataclass(frozen=True)
class TestOperationalAuthority:
    fence: EpochFence
    context: CommitContext

    def context_provider(self) -> CommitContext:
        return self.context


@dataclass(frozen=True)
class TestFreshV2Authority:
    signer: OperationalSigner
    roster: VerificationRoster
    pin_store: AuthorityPinStore
    fence: EpochFence
    key_id: str

    def runtime_authority(self):
        from memo.operational_activation import OperationalRuntimeAuthority

        return OperationalRuntimeAuthority(
            signer=self.signer,
            roster=self.roster,
            pin_store=self.pin_store,
            fence=self.fence,
            key_id=self.key_id,
        )


def build_test_fresh_v2_authority(
    root: Path,
    *,
    device_id: str,
) -> TestFreshV2Authority:
    authority_root = Path(root).resolve()
    keys = DeviceKeyStore.in_memory()
    origin_key = keys.generate(device_id=device_id, roles=("origin",))
    pin_store = AuthorityPinStore._for_test(
        authority_root,
        provider=InMemoryAuthorityPinProvider(),
    )
    roster = VerificationRoster.bootstrap(
        device_id=device_id,
        key=origin_key,
        root=authority_root,
        pin_store=pin_store,
    )
    signer = OperationalSigner(keys, roster_version=roster.version)
    fence = EpochFence(
        authority_root,
        roster=roster,
        verifier=OperationalVerifier(),
        pin_store=pin_store,
    )
    return TestFreshV2Authority(
        signer=signer,
        roster=roster,
        pin_store=pin_store,
        fence=fence,
        key_id=origin_key.key_id,
    )


def build_test_operational_authority(
    root: Path,
    *,
    device_id: str,
) -> TestOperationalAuthority:
    authority_root = Path(root).resolve()
    keys = DeviceKeyStore.in_memory()
    origin_key = keys.generate(device_id=device_id, roles=("origin",))
    pin_store = AuthorityPinStore._for_test(
        authority_root,
        provider=InMemoryAuthorityPinProvider(),
    )
    roster = VerificationRoster.bootstrap(
        device_id=device_id,
        key=origin_key,
        root=authority_root,
        pin_store=pin_store,
    )
    signer = OperationalSigner(keys, roster_version=roster.version)
    verifier = OperationalVerifier()
    fence = EpochFence(
        authority_root,
        roster=roster,
        verifier=verifier,
        pin_store=pin_store,
    )
    unsigned = EpochMarkerAuthorization(
        schema="memo.operational_epoch_authorization.v1",
        attempt_id="test-bootstrap-0",
        device_id=device_id,
        epoch=0,
        control_oid="test-control-0",
        artifact_digests={
            "bootstrap_roster": "a" * 64,
            "empty_anchor": "b" * 64,
        },
        roster_version=roster.version,
        key_id=origin_key.key_id,
        signature=None,  # type: ignore[arg-type]
    )
    authorization = replace(
        unsigned,
        signature=signer.sign(
            domain="memo.operational_epoch_authorization.v1",
            payload=canonical_signed_bytes(unsigned),
            key_id=origin_key.key_id,
        ),
    )
    fence.bootstrap(
        authorization=authorization,
        observed_artifact_digests=authorization.artifact_digests,
    )
    identity = PrincipalIdentity(
        principal_id=f"test:{device_id}",
        actor_id="memo-test",
        kind="agent",
        device_id=device_id,
        session_id="pytest",
        source_client="pytest",
    )
    context = fence.context(
        identity,
        request_epoch=authorization.epoch,
        request_control_oid=authorization.control_oid,
    )
    return TestOperationalAuthority(fence=fence, context=context)


def authorize_test_config(cfg: Config) -> Config:
    """Compose a real in-memory authority into one isolated test Config."""
    cfg.state_dir.mkdir(parents=True, exist_ok=True)
    # A reopened Memory may intentionally reuse its application state. Private
    # test keys are in-memory, so each composition needs its own authority root
    # while the operational journal itself remains shared.
    authority_root = Path(mkdtemp(prefix="memo-test-operational-authority-", dir=cfg.state_dir))
    authority = build_test_operational_authority(
        authority_root,
        device_id=cfg.device_id,
    )
    return cfg.model_copy(
        update={
            "operational_context_provider": authority.context_provider,
            "operational_epoch_fence": authority.fence,
        }
    )


def build_authorized_legacy_store(root: Path, *, device_id: str) -> OperationalStore:
    """Build a v1 store whose writes traverse a real authenticated fence."""
    from memo.operational import OperationalStore

    authority = build_test_operational_authority(
        Path(root) / "test-operational-authority",
        device_id=device_id,
    )
    return OperationalStore(
        root,
        device_id=device_id,
        context_provider=authority.context_provider,
        epoch_fence=authority.fence,
    )
