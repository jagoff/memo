"""Hermetic operational authority composition for tests.

The helpers deliberately build the real roster, signer, epoch marker, and
fence using only in-memory private-key and pin providers.  Tests therefore do
not bypass the production fail-closed write contract or touch Keychain.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

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
