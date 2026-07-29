from __future__ import annotations

import json
from dataclasses import replace

import pytest

from memo.identity import PrincipalIdentity
from memo.operational_epoch import AuthorityEpochError, EpochFence
from memo.operational_event import EpochMarkerAuthorization, canonical_signed_bytes
from memo.operational_key_store import (
    AuthorityPinStore,
    DeviceKeyStore,
    InMemoryAuthorityPinProvider,
)
from memo.operational_roster import VerificationRoster
from memo.operational_signing import OperationalSigner, OperationalVerifier

_AUTHORITY_PINS = InMemoryAuthorityPinProvider()


def _pin_store(root: object) -> AuthorityPinStore:
    return AuthorityPinStore(authority_id=str(root), provider=_AUTHORITY_PINS)


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
    roster = VerificationRoster.bootstrap(
        device_id="device-a", key=key, root=tmp_path, pin_store=_pin_store(tmp_path)
    )
    signer = OperationalSigner(keys, roster_version=roster.version)
    fence = EpochFence(
        tmp_path, roster=roster, verifier=OperationalVerifier(), pin_store=_pin_store(tmp_path)
    )
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
    context = fence.context(identity, request_epoch=0, request_control_oid="control-0")
    fence.verify(context)


def test_activation_is_signed_digest_bound_and_monotonic(tmp_path) -> None:
    keys = DeviceKeyStore.in_memory()
    key = keys.generate(device_id="device-a")
    roster = VerificationRoster.bootstrap(
        device_id="device-a", key=key, root=tmp_path, pin_store=_pin_store(tmp_path)
    )
    signer = OperationalSigner(keys, roster_version=1)
    fence = EpochFence(
        tmp_path, roster=roster, verifier=OperationalVerifier(), pin_store=_pin_store(tmp_path)
    )
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
    roster = VerificationRoster.bootstrap(
        device_id="device-a", key=key, root=tmp_path, pin_store=_pin_store(tmp_path)
    )
    signer = OperationalSigner(keys, roster_version=1)
    fence = EpochFence(
        tmp_path, roster=roster, verifier=OperationalVerifier(), pin_store=_pin_store(tmp_path)
    )
    auth = _authorization(
        signer,
        key_id=key.key_id,
        epoch=0,
        control_oid="control-0",
        digests={"bootstrap_roster": "a" * 64, "empty_anchor": "b" * 64},
    )
    fence.bootstrap(authorization=auth, observed_artifact_digests=auth.artifact_digests)
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
        EpochFence(
            tmp_path, roster=roster, verifier=OperationalVerifier(), pin_store=_pin_store(tmp_path)
        )


def test_marker_rollback_and_loss_fail_closed_after_reinstantiation(tmp_path) -> None:
    keys = DeviceKeyStore.in_memory()
    key = keys.generate(device_id="device-a")
    roster = VerificationRoster.bootstrap(
        device_id="device-a", key=key, root=tmp_path, pin_store=_pin_store(tmp_path)
    )
    signer = OperationalSigner(keys, roster_version=1)
    verifier = OperationalVerifier()
    fence = EpochFence(tmp_path, roster=roster, verifier=verifier, pin_store=_pin_store(tmp_path))
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
        EpochFence(tmp_path, roster=roster, verifier=verifier, pin_store=_pin_store(tmp_path))

    marker_path.unlink()
    with pytest.raises(AuthorityEpochError):
        EpochFence(tmp_path, roster=roster, verifier=verifier, pin_store=_pin_store(tmp_path))


def test_paired_marker_and_watermark_rollback_fails_across_instances(
    tmp_path,
) -> None:
    keys = DeviceKeyStore.in_memory()
    key = keys.generate(device_id="device-a")
    roster = VerificationRoster.bootstrap(
        device_id="device-a", key=key, root=tmp_path, pin_store=_pin_store(tmp_path)
    )
    signer = OperationalSigner(keys, roster_version=1)
    verifier = OperationalVerifier()
    fence = EpochFence(tmp_path, roster=roster, verifier=verifier, pin_store=_pin_store(tmp_path))
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
    watermark_path = tmp_path / "authority-epoch-high-watermark.json"
    epoch_zero = (marker_path.read_bytes(), watermark_path.read_bytes())
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

    marker_path.write_bytes(epoch_zero[0])
    watermark_path.write_bytes(epoch_zero[1])

    with pytest.raises(AuthorityEpochError):
        EpochFence(tmp_path, roster=roster, verifier=verifier, pin_store=_pin_store(tmp_path))


def test_full_authority_root_rollback_fails_across_instances(tmp_path) -> None:
    keys = DeviceKeyStore.in_memory()
    key = keys.generate(device_id="device-a")
    roster = VerificationRoster.bootstrap(
        device_id="device-a",
        key=key,
        root=tmp_path,
        pin_store=_pin_store(tmp_path),
    )
    signer = OperationalSigner(keys, roster_version=1)
    verifier = OperationalVerifier()
    fence = EpochFence(
        tmp_path,
        roster=roster,
        verifier=verifier,
        pin_store=_pin_store(tmp_path),
    )
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
    snapshot = {
        path.relative_to(tmp_path): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
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

    for relative, contents in snapshot.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)

    with pytest.raises(AuthorityEpochError):
        EpochFence(
            tmp_path,
            roster=roster,
            verifier=verifier,
            pin_store=_pin_store(tmp_path),
        )


def test_restart_recovers_crash_between_marker_and_watermark(tmp_path, monkeypatch) -> None:
    import memo.operational_epoch as epoch_module

    keys = DeviceKeyStore.in_memory()
    key = keys.generate(device_id="device-a")
    roster = VerificationRoster.bootstrap(
        device_id="device-a",
        key=key,
        root=tmp_path,
        pin_store=_pin_store(tmp_path),
    )
    verifier = OperationalVerifier()
    fence = EpochFence(
        tmp_path,
        roster=roster,
        verifier=verifier,
        pin_store=_pin_store(tmp_path),
    )
    authorization = _authorization(
        OperationalSigner(keys, roster_version=1),
        key_id=key.key_id,
        epoch=0,
        control_oid="control-0",
        digests={"bootstrap_roster": "a" * 64, "empty_anchor": "b" * 64},
    )
    original = epoch_module.atomic_write_text

    def crash_on_watermark(path: object, text: str) -> None:
        if str(path).endswith("authority-epoch-high-watermark.json"):
            raise OSError("fault injection after marker replace")
        original(path, text)  # type: ignore[arg-type]

    monkeypatch.setattr(epoch_module, "atomic_write_text", crash_on_watermark)
    with pytest.raises(OSError, match="fault injection"):
        fence.bootstrap(
            authorization=authorization,
            observed_artifact_digests=authorization.artifact_digests,
        )
    monkeypatch.setattr(epoch_module, "atomic_write_text", original)

    restarted = EpochFence(
        tmp_path,
        roster=VerificationRoster.load(tmp_path, pin_store=_pin_store(tmp_path)),
        verifier=verifier,
        pin_store=_pin_store(tmp_path),
    )
    context = restarted.context(
        PrincipalIdentity(
            principal_id="p1",
            actor_id="agent-a",
            kind="agent",
            device_id="device-a",
            session_id="session-a",
            source_client="codex",
        ),
        request_epoch=0,
        request_control_oid="control-0",
    )
    restarted.verify(context)


def test_roster_restart_can_complete_epoch_zero_once(tmp_path) -> None:
    keys = DeviceKeyStore.in_memory()
    key = keys.generate(device_id="device-a")
    VerificationRoster.bootstrap(
        device_id="device-a", key=key, root=tmp_path, pin_store=_pin_store(tmp_path)
    )
    signer = OperationalSigner(keys, roster_version=1)
    verifier = OperationalVerifier()
    authorization = _authorization(
        signer,
        key_id=key.key_id,
        epoch=0,
        control_oid="control-0",
        digests={"bootstrap_roster": "a" * 64, "empty_anchor": "b" * 64},
    )

    restarted_roster = VerificationRoster.load(tmp_path, pin_store=_pin_store(tmp_path))
    restarted = EpochFence(
        tmp_path,
        roster=restarted_roster,
        verifier=verifier,
        pin_store=_pin_store(tmp_path),
    )
    restarted.bootstrap(
        authorization=authorization,
        observed_artifact_digests=authorization.artifact_digests,
    )

    reloaded = VerificationRoster.load(tmp_path, pin_store=_pin_store(tmp_path))
    with pytest.raises(AuthorityEpochError):
        EpochFence(
            tmp_path, roster=reloaded, verifier=verifier, pin_store=_pin_store(tmp_path)
        ).bootstrap(
            authorization=authorization,
            observed_artifact_digests=authorization.artifact_digests,
        )


def test_bootstrap_is_one_shot_and_adapters_cannot_mint_system_capability(
    tmp_path,
) -> None:
    import memo.operational_epoch as epoch_module

    assert not hasattr(epoch_module, "_issue_system_capability")
    with pytest.raises(TypeError):
        epoch_module.SystemCapability()

    keys = DeviceKeyStore.in_memory()
    key = keys.generate(device_id="device-a")
    roster = VerificationRoster.bootstrap(
        device_id="device-a", key=key, root=tmp_path, pin_store=_pin_store(tmp_path)
    )
    fence = EpochFence(
        tmp_path, roster=roster, verifier=OperationalVerifier(), pin_store=_pin_store(tmp_path)
    )
    assert not hasattr(fence, "system_capability")
    forged = object.__new__(epoch_module.SystemCapability)
    with pytest.raises(AuthorityEpochError):
        fence.system_context(
            PrincipalIdentity(
                principal_id="p1",
                actor_id="agent-a",
                kind="agent",
                device_id="device-a",
                session_id="session-a",
                source_client="codex",
            ),
            capability=forged,
        )

    loaded = VerificationRoster.load(tmp_path, pin_store=_pin_store(tmp_path))
    other_root = tmp_path / "other"
    other_root.mkdir()
    copied_fence = EpochFence(
        other_root,
        roster=loaded,
        verifier=OperationalVerifier(),
        pin_store=_pin_store(other_root),
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
    roster = VerificationRoster.bootstrap(
        device_id="device-a", key=key, root=tmp_path, pin_store=_pin_store(tmp_path)
    )
    signer = OperationalSigner(keys, roster_version=1)
    auth = _authorization(
        signer,
        key_id=key.key_id,
        epoch=0,
        control_oid="control-0",
        digests={"bootstrap_roster": "a" * 64, "empty_anchor": "b" * 64},
    )
    EpochFence(
        tmp_path, roster=roster, verifier=OperationalVerifier(), pin_store=_pin_store(tmp_path)
    ).bootstrap(authorization=auth, observed_artifact_digests=auth.artifact_digests)
    assert calls
