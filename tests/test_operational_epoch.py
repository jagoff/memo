from __future__ import annotations

import json
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
    bootstrap = _authorization(
        signer,
        key_id=key.key_id,
        epoch=0,
        control_oid="control-0",
        digests={"bootstrap_roster": "c" * 64, "empty_anchor": "d" * 64},
    )
    fence.bootstrap(
        authorization=bootstrap,
        observed_artifact_digests=bootstrap.artifact_digests,
    )
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


def test_marker_tampering_is_rejected_on_every_read(tmp_path) -> None:
    keys = DeviceKeyStore.in_memory()
    key = keys.generate(device_id="device-a")
    roster = VerificationRoster.bootstrap(device_id="device-a", key=key, root=tmp_path)
    signer = OperationalSigner(keys, roster_version=1)
    fence = EpochFence(tmp_path, roster=roster, verifier=OperationalVerifier())
    auth = _authorization(
        signer,
        key_id=key.key_id,
        epoch=0,
        control_oid="control-0",
        digests={"bootstrap_roster": "a" * 64, "empty_anchor": "b" * 64},
    )
    fence.bootstrap(
        authorization=auth, observed_artifact_digests=auth.artifact_digests
    )
    marker_path = tmp_path / "authority-epoch.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker.update(
        {
            "epoch": 9,
            "control_oid": "attacker",
            "artifact_digests": {"attacker": "c" * 64},
            "authorization_sha256": "d" * 64,
        }
    )
    marker_path.write_text(json.dumps(marker), encoding="utf-8")
    with pytest.raises(AuthorityEpochError):
        EpochFence(tmp_path, roster=roster, verifier=OperationalVerifier())


def test_marker_rollback_and_loss_fail_closed_after_reinstantiation(tmp_path) -> None:
    keys = DeviceKeyStore.in_memory()
    key = keys.generate(device_id="device-a")
    roster = VerificationRoster.bootstrap(device_id="device-a", key=key, root=tmp_path)
    signer = OperationalSigner(keys, roster_version=1)
    verifier = OperationalVerifier()
    fence = EpochFence(tmp_path, roster=roster, verifier=verifier)
    bootstrap = _authorization(
        signer,
        key_id=key.key_id,
        epoch=0,
        control_oid="control-0",
        digests={"bootstrap_roster": "a" * 64, "empty_anchor": "b" * 64},
    )
    fence.bootstrap(
        authorization=bootstrap,
        observed_artifact_digests=bootstrap.artifact_digests,
    )
    marker_path = tmp_path / "authority-epoch.json"
    epoch_zero = marker_path.read_bytes()
    epoch_one = _authorization(
        signer,
        key_id=key.key_id,
        epoch=1,
        control_oid="control-1",
        digests={"memo_generation": "c" * 64},
    )
    fence.activate(
        authorization=epoch_one,
        observed_artifact_digests=epoch_one.artifact_digests,
    )
    marker_path.write_bytes(epoch_zero)
    with pytest.raises(AuthorityEpochError):
        EpochFence(tmp_path, roster=roster, verifier=verifier)

    marker_path.unlink()
    with pytest.raises(AuthorityEpochError):
        EpochFence(tmp_path, roster=roster, verifier=verifier)


def test_bootstrap_is_one_shot_and_adapters_cannot_mint_system_capability(
    tmp_path,
) -> None:
    keys = DeviceKeyStore.in_memory()
    key = keys.generate(device_id="device-a")
    roster = VerificationRoster.bootstrap(device_id="device-a", key=key, root=tmp_path)
    fence = EpochFence(tmp_path, roster=roster, verifier=OperationalVerifier())
    assert not hasattr(fence, "system_capability")

    loaded = VerificationRoster.load(tmp_path)
    other_root = tmp_path / "other"
    other_root.mkdir()
    copied_fence = EpochFence(
        other_root, roster=loaded, verifier=OperationalVerifier()
    )
    signer = OperationalSigner(keys, roster_version=1)
    auth = _authorization(
        signer,
        key_id=key.key_id,
        epoch=0,
        control_oid="control-0",
        digests={"bootstrap_roster": "a" * 64, "empty_anchor": "b" * 64},
    )
    with pytest.raises(AuthorityEpochError):
        copied_fence.bootstrap(
            authorization=auth,
            observed_artifact_digests=auth.artifact_digests,
        )


def test_authority_writes_fsync_parent_directory(tmp_path, monkeypatch) -> None:
    import memo.operational_epoch as epoch_module

    calls: list[object] = []
    original = epoch_module._fsync_directory

    def tracking(path: object) -> None:
        calls.append(path)
        original(path)

    monkeypatch.setattr(epoch_module, "_fsync_directory", tracking)
    keys = DeviceKeyStore.in_memory()
    key = keys.generate(device_id="device-a")
    roster = VerificationRoster.bootstrap(device_id="device-a", key=key, root=tmp_path)
    signer = OperationalSigner(keys, roster_version=1)
    auth = _authorization(
        signer,
        key_id=key.key_id,
        epoch=0,
        control_oid="control-0",
        digests={"bootstrap_roster": "a" * 64, "empty_anchor": "b" * 64},
    )
    EpochFence(tmp_path, roster=roster, verifier=OperationalVerifier()).bootstrap(
        authorization=auth, observed_artifact_digests=auth.artifact_digests
    )
    assert calls
