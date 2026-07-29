from __future__ import annotations

from dataclasses import replace

import pytest

from memo.identity import PrincipalIdentity
from memo.operational_epoch import AuthorityEpochError, EpochFence
from memo.operational_event import EpochMarkerAuthorization, canonical_signed_bytes
from memo.operational_key_store import DeviceKeyStore
from memo.operational_roster import VerificationRoster
from memo.operational_signing import OperationalSigner, OperationalVerifier


def _authorization(
    signer: OperationalSigner,
    *,
    key_id: str,
    epoch: int,
    control_oid: str,
    digests: dict[str, str],
) -> EpochMarkerAuthorization:
    unsigned = EpochMarkerAuthorization(
        schema="memo.operational_epoch_authorization.v1",
        attempt_id=f"attempt-{epoch}",
        device_id="device-a",
        epoch=epoch,
        control_oid=control_oid,
        artifact_digests=digests,
        roster_version=1,
        key_id=key_id,
        signature=None,  # type: ignore[arg-type]
    )
    envelope = signer.sign(
        domain="memo.operational_epoch_authorization.v1",
        payload=canonical_signed_bytes(unsigned),
        key_id=key_id,
    )
    return replace(unsigned, signature=envelope)


def test_external_context_requires_explicit_epoch_and_control(tmp_path) -> None:
    keys = DeviceKeyStore.in_memory()
    key = keys.generate(device_id="device-a")
    roster = VerificationRoster.bootstrap(device_id="device-a", key=key, root=tmp_path)
    signer = OperationalSigner(keys, roster_version=roster.version)
    fence = EpochFence(tmp_path, roster=roster, verifier=OperationalVerifier())
    auth = _authorization(
        signer,
        key_id=key.key_id,
        epoch=0,
        control_oid="control-0",
        digests={"bootstrap_roster": "a" * 64, "empty_anchor": "b" * 64},
    )
    fence.bootstrap(
        authorization=auth,
        observed_artifact_digests=auth.artifact_digests,
    )
    identity = PrincipalIdentity(
        principal_id="p1",
        actor_id="agent-a",
        kind="agent",
        device_id="device-a",
        session_id="session-a",
        source_client="codex",
    )
    with pytest.raises(TypeError):
        fence.context(identity)  # type: ignore[call-arg]
    with pytest.raises(AuthorityEpochError):
        fence.context(identity, request_epoch=4, request_control_oid="stale-control")
    context = fence.context(
        identity, request_epoch=0, request_control_oid="control-0"
    )
    fence.verify(context)


def test_activation_is_signed_digest_bound_and_monotonic(tmp_path) -> None:
    keys = DeviceKeyStore.in_memory()
    key = keys.generate(device_id="device-a")
    roster = VerificationRoster.bootstrap(device_id="device-a", key=key, root=tmp_path)
    signer = OperationalSigner(keys, roster_version=1)
    fence = EpochFence(tmp_path, roster=roster, verifier=OperationalVerifier())
    digests = {"memo_generation": "a" * 64}
    auth = _authorization(
        signer, key_id=key.key_id, epoch=1, control_oid="control-1", digests=digests
    )
    with pytest.raises(AuthorityEpochError):
        fence.activate(
            authorization=auth,
            observed_artifact_digests={"memo_generation": "b" * 64},
        )
    fence.activate(authorization=auth, observed_artifact_digests=digests)
    with pytest.raises(AuthorityEpochError):
        fence.activate(
            authorization=replace(auth, epoch=0),
            observed_artifact_digests=digests,
        )
