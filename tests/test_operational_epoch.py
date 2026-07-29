from __future__ import annotations

import base64
import copy
import gc
import inspect
import json
import pickle
import threading
import weakref
from contextlib import suppress
from dataclasses import replace
from pathlib import Path

import pytest

from memo.identity import PrincipalIdentity
from memo.operational_epoch import AuthorityEpochError, EpochFence
from memo.operational_event import EpochMarkerAuthorization, canonical_signed_bytes
from memo.operational_key_store import (
    AuthorityPinStore,
    DeviceKeyStore,
    InMemoryAuthorityPinProvider,
    KeyStoreError,
    PublicKeyRecord,
)
from memo.operational_roster import VerificationRoster
from memo.operational_signing import (
    OperationalSigner,
    OperationalVerifier,
    SignatureEnvelope,
)

_AUTHORITY_PINS = InMemoryAuthorityPinProvider()


def _pin_store(root: Path) -> AuthorityPinStore:
    return AuthorityPinStore._for_test(root, provider=_AUTHORITY_PINS)


def _use_in_memory_productive_pin_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> InMemoryAuthorityPinProvider:
    provider = InMemoryAuthorityPinProvider()

    def for_root(
        cls: type[AuthorityPinStore],
        root: Path,
    ) -> AuthorityPinStore:
        return cls._for_test(root, provider=provider)

    monkeypatch.setattr(AuthorityPinStore, "for_root", classmethod(for_root))
    return provider


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
        roster_version=signer.roster_version,
        key_id=key_id,
        signature=None,  # type: ignore[arg-type]
    )
    envelope = signer.sign(
        domain="memo.operational_epoch_authorization.v1",
        payload=canonical_signed_bytes(unsigned),
        key_id=key_id,
    )
    return replace(unsigned, signature=envelope)


def _system_identity() -> PrincipalIdentity:
    return PrincipalIdentity(
        principal_id="system-p1",
        actor_id="migration-daemon",
        kind="agent",
        device_id="device-a",
        session_id="system-session",
        source_client="memo",
    )


def _bootstrap_system_authority(
    root: Path,
) -> tuple[DeviceKeyStore, PublicKeyRecord, VerificationRoster, EpochFence]:
    keys = DeviceKeyStore.in_memory()
    key = keys.generate(device_id="device-a", roles=("origin",))
    roster = VerificationRoster.bootstrap(
        device_id="device-a",
        key=key,
        root=root,
        pin_store=_pin_store(root),
    )
    fence = EpochFence(
        root,
        roster=roster,
        verifier=OperationalVerifier(),
        pin_store=_pin_store(root),
    )
    authorization = _authorization(
        OperationalSigner(keys, roster_version=1),
        key_id=key.key_id,
        epoch=0,
        control_oid="control-0",
        digests={"bootstrap_roster": "a" * 64, "empty_anchor": "b" * 64},
    )
    fence.bootstrap(
        authorization=authorization,
        observed_artifact_digests=authorization.artifact_digests,
    )
    return keys, key, roster, fence


def _rotate_origin_key(
    root: Path,
    *,
    keys: DeviceKeyStore,
    old_key: PublicKeyRecord,
    roster: VerificationRoster,
) -> tuple[VerificationRoster, PublicKeyRecord]:
    new_key = keys.generate(
        device_id="device-a",
        roles=("origin",),
        enrollment_sequence=2,
    )
    revoked = replace(
        old_key,
        revocation_sequence=2,
        proof_of_possession="",
    )
    revoked = replace(
        revoked,
        proof_of_possession=base64.urlsafe_b64encode(
            keys.sign(
                key_id=revoked.key_id,
                payload=revoked.proof_payload(),
            )
        )
        .rstrip(b"=")
        .decode("ascii"),
    )
    updated = roster.with_keys(
        version=2,
        peers=("device-a",),
        keys=(revoked, new_key),
        signer=OperationalSigner(keys, roster_version=1),
        root=root,
        pin_store=_pin_store(root),
    )
    return updated, new_key


def test_public_epoch_fence_contract_uses_root_bound_pin_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _use_in_memory_productive_pin_factory(monkeypatch)
    keys = DeviceKeyStore.in_memory()
    key = keys.generate(device_id="device-a")
    roster = VerificationRoster.bootstrap(
        device_id="device-a",
        key=key,
        root=tmp_path,
        pin_store=AuthorityPinStore._for_test(tmp_path, provider=provider),
    )

    fence = EpochFence(
        tmp_path,
        roster=roster,
        verifier=OperationalVerifier(),
    )

    assert fence.roster == roster
    assert EpochFence(tmp_path, roster=roster).roster == roster


def test_epoch_fence_rejects_valid_roster_pinned_to_different_root(
    tmp_path: Path,
) -> None:
    keys = DeviceKeyStore.in_memory()
    enrolled = keys.generate(device_id="device-a")
    canonical = VerificationRoster.bootstrap(
        device_id="device-a",
        key=enrolled,
        root=tmp_path,
        pin_store=_pin_store(tmp_path),
    )
    attacker_root = tmp_path / "attacker"
    attacker = keys.generate(device_id="device-a")
    unpinned = VerificationRoster.bootstrap(
        device_id="device-a",
        key=attacker,
        root=attacker_root,
        pin_store=_pin_store(attacker_root),
    )
    assert unpinned != canonical

    with pytest.raises(AuthorityEpochError, match="pinned roster"):
        EpochFence(
            tmp_path,
            roster=unpinned,
            verifier=OperationalVerifier(),
            pin_store=_pin_store(tmp_path),
        )


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


def test_verified_context_holds_marker_lock_and_verify_does_not_relock(
    tmp_path: Path,
) -> None:
    keys, key, _roster, fence = _bootstrap_system_authority(tmp_path)
    context = fence.context(
        _system_identity(),
        request_epoch=0,
        request_control_oid="control-0",
    )
    epoch_one = _authorization(
        OperationalSigner(keys, roster_version=1),
        key_id=key.key_id,
        epoch=1,
        control_oid="control-1",
        digests={"memo_generation": "c" * 64},
    )
    holder_entered = threading.Event()
    release_holder = threading.Event()
    activation_started = threading.Event()
    activation_finished = threading.Event()
    failures: list[BaseException] = []

    def hold_verified_epoch() -> None:
        try:
            with fence.verified(context):
                holder_entered.set()
                assert release_holder.wait(timeout=2)
        except BaseException as exc:  # pragma: no cover - diagnostic capture
            failures.append(exc)

    def activate_next_epoch() -> None:
        try:
            activation_started.set()
            fence.activate(
                authorization=epoch_one,
                observed_artifact_digests=epoch_one.artifact_digests,
            )
            activation_finished.set()
        except BaseException as exc:  # pragma: no cover - diagnostic capture
            failures.append(exc)

    holder = threading.Thread(target=hold_verified_epoch)
    activator = threading.Thread(target=activate_next_epoch)
    holder.start()
    assert holder_entered.wait(timeout=2)
    activator.start()
    assert activation_started.wait(timeout=2)
    assert not activation_finished.wait(timeout=0.2)
    release_holder.set()
    holder.join(timeout=2)
    activator.join(timeout=2)

    assert not holder.is_alive()
    assert not activator.is_alive()
    assert activation_finished.is_set()
    assert failures == []

    fresh = fence.context(
        _system_identity(),
        request_epoch=1,
        request_control_oid="control-1",
    )
    verification_finished = threading.Event()
    verifier_thread = threading.Thread(
        target=lambda: (fence.verify(fresh), verification_finished.set())
    )
    verifier_thread.start()
    verifier_thread.join(timeout=2)
    assert not verifier_thread.is_alive()
    assert verification_finished.is_set()


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


def test_full_root_rollback_cannot_rebase_into_fresh_public_namespace(tmp_path) -> None:
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

    def attempt_public_rebase() -> None:
        fresh = AuthorityPinStore(
            authority_id=f"attacker-selected:{tmp_path}",
            provider=_AUTHORITY_PINS,
        )
        fresh.prepare_roster(version=1, roster_hash=roster.roster_hash)
        fresh.commit_roster(version=1, roster_hash=roster.roster_hash)
        marker = json.loads((tmp_path / "authority-epoch.json").read_text(encoding="utf-8"))
        authorization_sha256 = str(marker["authorization_sha256"])
        fresh.prepare_epoch(
            epoch=0,
            authorization_sha256=authorization_sha256,
            bootstrap=True,
        )
        fresh.commit_epoch(
            epoch=0,
            authorization_sha256=authorization_sha256,
            bootstrap=True,
        )
        EpochFence(
            tmp_path,
            roster=roster,
            verifier=verifier,
            pin_store=fresh,
        )

    with pytest.raises((TypeError, AttributeError, KeyStoreError)):
        attempt_public_rebase()


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


def test_restart_recovers_crash_before_first_epoch_root_write(
    tmp_path,
    monkeypatch,
) -> None:
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

    def crash_before_marker(path: object, text: str) -> None:
        del path, text
        raise OSError("fault injection before first epoch write")

    monkeypatch.setattr(epoch_module, "atomic_write_text", crash_before_marker)
    with pytest.raises(OSError, match="before first epoch write"):
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


def test_restart_recovers_activation_crash_before_first_root_write(
    tmp_path,
    monkeypatch,
) -> None:
    import memo.operational_epoch as epoch_module

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
    activation = _authorization(
        signer,
        key_id=key.key_id,
        epoch=1,
        control_oid="control-1",
        digests={"memo_generation": "c" * 64},
    )
    original = epoch_module.atomic_write_text

    def crash_before_activation_marker(path: object, text: str) -> None:
        del path, text
        raise OSError("fault injection before first activation write")

    monkeypatch.setattr(
        epoch_module,
        "atomic_write_text",
        crash_before_activation_marker,
    )
    with pytest.raises(OSError, match="before first activation write"):
        fence.activate(
            authorization=activation,
            observed_artifact_digests=activation.artifact_digests,
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
        request_epoch=1,
        request_control_oid="control-1",
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
    signer = OperationalSigner(keys, roster_version=1)
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
    assert not hasattr(fence, "system_capability")
    forged = object.__new__(epoch_module.SystemCapability)
    with suppress(AttributeError):
        object.__setattr__(forged, "_SystemCapability__owner", fence)
    with pytest.raises(AuthorityEpochError):
        fence.system_context(_system_identity(), capability=forged)

    loaded = VerificationRoster.load(tmp_path, pin_store=_pin_store(tmp_path))
    other_root = tmp_path / "other"
    other_root.mkdir()
    with pytest.raises(AuthorityEpochError, match="pinned roster"):
        EpochFence(
            other_root,
            roster=loaded,
            verifier=OperationalVerifier(),
            pin_store=_pin_store(other_root),
        )


def test_ordinary_fence_construction_has_no_privileged_binding_channel(
    tmp_path,
) -> None:
    keys = DeviceKeyStore.in_memory()
    key = keys.generate(device_id="device-a")
    roster = VerificationRoster.bootstrap(
        device_id="device-a",
        key=key,
        root=tmp_path,
        pin_store=_pin_store(tmp_path),
    )
    constructor_parameters = inspect.signature(EpochFence).parameters
    assert "_system_context_sink" not in constructor_parameters
    captured: list[object] = []
    for keyword in (
        "_system_context_sink",
        "system_context_sink",
        "callback",
        "capability",
    ):
        with pytest.raises(TypeError):
            EpochFence(
                tmp_path,
                roster=roster,
                verifier=OperationalVerifier(),
                pin_store=_pin_store(tmp_path),
                **{keyword: captured.append},
            )
    assert captured == []

    fences = [
        EpochFence(
            tmp_path,
            roster=roster,
            verifier=OperationalVerifier(),
            pin_store=_pin_store(tmp_path),
        )
        for _ in range(3)
    ]
    for fence in fences:
        assert not hasattr(fence, "system_capability")
        assert not hasattr(fence, "bound_system_context")
        assert not hasattr(fence, "system_context_operation")


def test_binding_requires_enrolled_signer_and_an_internal_role(tmp_path) -> None:
    import memo.operational_epoch as epoch_module

    keys = DeviceKeyStore.in_memory()
    key = keys.generate(device_id="device-a")
    roster = VerificationRoster.bootstrap(
        device_id="device-a",
        key=key,
        root=tmp_path,
        pin_store=_pin_store(tmp_path),
    )
    signer = OperationalSigner(keys, roster_version=1)
    fence = EpochFence(
        tmp_path,
        roster=roster,
        verifier=OperationalVerifier(),
        pin_store=_pin_store(tmp_path),
    )
    authorization = _authorization(
        signer,
        key_id=key.key_id,
        epoch=0,
        control_oid="control-0",
        digests={"bootstrap_roster": "a" * 64, "empty_anchor": "b" * 64},
    )
    fence.bootstrap(
        authorization=authorization,
        observed_artifact_digests=authorization.artifact_digests,
    )

    untrusted_keys = DeviceKeyStore.in_memory()
    untrusted_key = untrusted_keys.generate(device_id="device-a", roles=("origin",))
    with pytest.raises(AuthorityEpochError):
        epoch_module.bind_system_context(
            fence,
            signer=OperationalSigner(untrusted_keys, roster_version=1),
            key_id=untrusted_key.key_id,
            system_role="daemon",
        )
    with pytest.raises(AuthorityEpochError):
        epoch_module.bind_system_context(
            fence,
            signer=signer,
            key_id=key.key_id,
            system_role="adapter",
        )
    with pytest.raises(AuthorityEpochError):
        epoch_module.bind_system_context(
            fence,
            signer=signer,
            key_id=key.key_id,
            system_role="migration",
        )


def test_live_fence_rejects_binding_with_signer_revoked_by_latest_roster(
    tmp_path: Path,
) -> None:
    import memo.operational_epoch as epoch_module

    keys, old_key, roster, fence = _bootstrap_system_authority(tmp_path)
    _rotate_origin_key(
        tmp_path,
        keys=keys,
        old_key=old_key,
        roster=roster,
    )

    for signer_version in (1, 2):
        with pytest.raises(AuthorityEpochError):
            epoch_module.bind_system_context(
                fence,
                signer=OperationalSigner(keys, roster_version=signer_version),
                key_id=old_key.key_id,
                system_role="daemon",
            )


def test_bound_operation_is_revoked_when_latest_roster_advances(
    tmp_path: Path,
) -> None:
    import memo.operational_epoch as epoch_module

    keys, old_key, roster, fence = _bootstrap_system_authority(tmp_path)
    operation = epoch_module.bind_system_context(
        fence,
        signer=OperationalSigner(keys, roster_version=1),
        key_id=old_key.key_id,
        system_role="daemon",
    )
    operation(_system_identity())
    _rotate_origin_key(
        tmp_path,
        keys=keys,
        old_key=old_key,
        roster=roster,
    )

    with pytest.raises(AuthorityEpochError):
        operation(_system_identity())


def test_latest_enrolled_key_can_bind_and_use_live_fence(tmp_path: Path) -> None:
    import memo.operational_epoch as epoch_module

    keys, old_key, roster, fence = _bootstrap_system_authority(tmp_path)
    updated, new_key = _rotate_origin_key(
        tmp_path,
        keys=keys,
        old_key=old_key,
        roster=roster,
    )

    operation = epoch_module.bind_system_context(
        fence,
        signer=OperationalSigner(keys, roster_version=updated.version),
        key_id=new_key.key_id,
        system_role="daemon",
    )

    context = operation(_system_identity())
    assert (context.authority_epoch, context.control_oid) == (0, "control-0")


def test_marker_signed_by_historical_roster_survives_latest_rotation(
    tmp_path: Path,
) -> None:
    keys, old_key, roster, fence = _bootstrap_system_authority(tmp_path)
    updated, _new_key = _rotate_origin_key(
        tmp_path,
        keys=keys,
        old_key=old_key,
        roster=roster,
    )

    live_context = fence.context(
        _system_identity(),
        request_epoch=0,
        request_control_oid="control-0",
    )
    reopened = EpochFence(
        tmp_path,
        roster=updated,
        verifier=OperationalVerifier(),
        pin_store=_pin_store(tmp_path),
    )
    reopened_context = reopened.context(
        _system_identity(),
        request_epoch=0,
        request_control_oid="control-0",
    )

    assert (live_context.authority_epoch, live_context.control_oid) == (
        reopened_context.authority_epoch,
        reopened_context.control_oid,
    ) == (0, "control-0")


def test_latest_roster_activation_advances_historical_marker(
    tmp_path: Path,
) -> None:
    keys, old_key, roster, fence = _bootstrap_system_authority(tmp_path)
    updated, new_key = _rotate_origin_key(
        tmp_path,
        keys=keys,
        old_key=old_key,
        roster=roster,
    )
    authorization = _authorization(
        OperationalSigner(keys, roster_version=updated.version),
        key_id=new_key.key_id,
        epoch=1,
        control_oid="control-1",
        digests={"memo_generation": "c" * 64},
    )

    fence.activate(
        authorization=authorization,
        observed_artifact_digests=authorization.artifact_digests,
    )

    context = fence.context(
        _system_identity(),
        request_epoch=1,
        request_control_oid="control-1",
    )
    assert (context.authority_epoch, context.control_oid) == (1, "control-1")


@pytest.mark.parametrize("damage", ["corrupt", "truncate"])
def test_privileged_use_fails_closed_when_roster_history_is_invalid(
    tmp_path: Path,
    damage: str,
) -> None:
    import memo.operational_epoch as epoch_module

    keys = DeviceKeyStore.in_memory()
    old_key = keys.generate(device_id="device-a", roles=("origin",))
    roster = VerificationRoster.bootstrap(
        device_id="device-a",
        key=old_key,
        root=tmp_path,
        pin_store=_pin_store(tmp_path),
    )
    updated, new_key = _rotate_origin_key(
        tmp_path,
        keys=keys,
        old_key=old_key,
        roster=roster,
    )
    fence = EpochFence(
        tmp_path,
        roster=updated,
        verifier=OperationalVerifier(),
        pin_store=_pin_store(tmp_path),
    )
    authorization = _authorization(
        OperationalSigner(keys, roster_version=updated.version),
        key_id=new_key.key_id,
        epoch=0,
        control_oid="control-0",
        digests={"bootstrap_roster": "a" * 64, "empty_anchor": "b" * 64},
    )
    fence.bootstrap(
        authorization=authorization,
        observed_artifact_digests=authorization.artifact_digests,
    )
    operation = epoch_module.bind_system_context(
        fence,
        signer=OperationalSigner(keys, roster_version=updated.version),
        key_id=new_key.key_id,
        system_role="daemon",
    )
    operation(_system_identity())
    if damage == "corrupt":
        (tmp_path / "verification-rosters/00000001.json").write_text(
            "{}",
            encoding="utf-8",
        )
    else:
        (tmp_path / "verification-rosters/00000002.json").unlink()

    with pytest.raises(AuthorityEpochError):
        operation(_system_identity())


def test_privileged_use_fails_closed_on_concurrent_roster_update(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import memo.operational_epoch as epoch_module

    keys, old_key, roster, fence = _bootstrap_system_authority(tmp_path)
    operation = epoch_module.bind_system_context(
        fence,
        signer=OperationalSigner(keys, roster_version=1),
        key_id=old_key.key_id,
        system_role="daemon",
    )
    original = EpochFence._verify_system_capability
    rotated = False

    def verify_then_rotate(
        current_fence: EpochFence,
        payload: bytes,
        signature: SignatureEnvelope,
        *args: object,
        **kwargs: object,
    ) -> None:
        nonlocal rotated
        original(current_fence, payload, signature, *args, **kwargs)
        if current_fence is fence and not rotated:
            rotated = True
            _rotate_origin_key(
                tmp_path,
                keys=keys,
                old_key=old_key,
                roster=roster,
            )

    monkeypatch.setattr(
        EpochFence,
        "_verify_system_capability",
        verify_then_rotate,
    )

    with pytest.raises(AuthorityEpochError):
        operation(_system_identity())
    assert rotated is True


def test_migration_binding_accepts_an_exclusive_enrolled_attestor(tmp_path) -> None:
    import memo.operational_epoch as epoch_module

    keys = DeviceKeyStore.in_memory()
    origin_key = keys.generate(device_id="device-a", roles=("origin",))
    roster = VerificationRoster.bootstrap(
        device_id="device-a",
        key=origin_key,
        root=tmp_path,
        pin_store=_pin_store(tmp_path),
    )
    migration_key = keys.generate(
        device_id="device-a",
        roles=("migration_attestor",),
        enrollment_sequence=2,
    )
    roster = roster.with_keys(
        version=2,
        peers=("device-a",),
        keys=(origin_key, migration_key),
        signer=OperationalSigner(keys, roster_version=1),
        root=tmp_path,
        pin_store=_pin_store(tmp_path),
    )
    signer = OperationalSigner(keys, roster_version=2)
    fence = EpochFence(
        tmp_path,
        roster=roster,
        verifier=OperationalVerifier(),
        pin_store=_pin_store(tmp_path),
    )
    authorization = _authorization(
        signer,
        key_id=origin_key.key_id,
        epoch=0,
        control_oid="control-0",
        digests={"bootstrap_roster": "a" * 64, "empty_anchor": "b" * 64},
    )
    fence.bootstrap(
        authorization=authorization,
        observed_artifact_digests=authorization.artifact_digests,
    )

    operation = epoch_module.bind_system_context(
        fence,
        signer=signer,
        key_id=migration_key.key_id,
        system_role="migration",
    )
    context = operation(_system_identity())
    assert (context.authority_epoch, context.control_oid) == (0, "control-0")


def test_legitimate_binding_returns_only_an_opaque_bound_operation(tmp_path) -> None:
    import memo.operational_epoch as epoch_module

    keys = DeviceKeyStore.in_memory()
    key = keys.generate(device_id="device-a", roles=("origin",))
    roster = VerificationRoster.bootstrap(
        device_id="device-a",
        key=key,
        root=tmp_path,
        pin_store=_pin_store(tmp_path),
    )
    fence = EpochFence(
        tmp_path,
        roster=roster,
        verifier=OperationalVerifier(),
        pin_store=_pin_store(tmp_path),
    )
    assert hasattr(epoch_module, "SystemCapability")
    assert hasattr(epoch_module.EpochFence, "system_context")
    assert not hasattr(fence, "system_capability")
    assert not hasattr(epoch_module, "_issue_system_capability")
    with pytest.raises(TypeError):
        epoch_module.SystemCapability()
    authorization = _authorization(
        OperationalSigner(keys, roster_version=1),
        key_id=key.key_id,
        epoch=0,
        control_oid="control-0",
        digests={"bootstrap_roster": "a" * 64, "empty_anchor": "b" * 64},
    )
    fence.bootstrap(
        authorization=authorization,
        observed_artifact_digests=authorization.artifact_digests,
    )
    operation = epoch_module.bind_system_context(
        fence,
        signer=OperationalSigner(keys, roster_version=1),
        key_id=key.key_id,
        system_role="daemon",
    )
    assert callable(operation)
    assert not isinstance(operation, epoch_module.SystemCapability)
    assert getattr(operation, "__closure__", None) is None
    assert not hasattr(operation, "__dict__")
    assert not hasattr(operation, "capability")
    assert not hasattr(operation, "validator")
    direct_referents = gc.get_referents(operation)
    assert not any(
        isinstance(value, epoch_module.SystemCapability) for value in direct_referents
    )
    assert not any(
        isinstance(value, (OperationalSigner, OperationalVerifier))
        for value in direct_referents
    )
    assert not any(isinstance(value, (dict, list, set)) for value in direct_referents)

    context = operation(_system_identity())
    assert (context.authority_epoch, context.control_oid) == (0, "control-0")
    with pytest.raises(AuthorityEpochError):
        fence.system_context(
            _system_identity(),
            capability=object.__new__(epoch_module.SystemCapability),
        )


def test_bound_operation_proof_cannot_replay_to_another_fence(tmp_path) -> None:
    import memo.operational_epoch as epoch_module

    keys = DeviceKeyStore.in_memory()
    key = keys.generate(device_id="device-a", roles=("origin",))
    roster = VerificationRoster.bootstrap(
        device_id="device-a",
        key=key,
        root=tmp_path,
        pin_store=_pin_store(tmp_path),
    )
    signer = OperationalSigner(keys, roster_version=1)
    fence = EpochFence(
        tmp_path,
        roster=roster,
        verifier=OperationalVerifier(),
        pin_store=_pin_store(tmp_path),
    )
    authorization = _authorization(
        signer,
        key_id=key.key_id,
        epoch=0,
        control_oid="control-0",
        digests={"bootstrap_roster": "a" * 64, "empty_anchor": "b" * 64},
    )
    fence.bootstrap(
        authorization=authorization,
        observed_artifact_digests=authorization.artifact_digests,
    )
    operation = epoch_module.bind_system_context(
        fence,
        signer=signer,
        key_id=key.key_id,
        system_role="daemon",
    )
    referents = gc.get_referents(operation)
    payload = next(value for value in referents if isinstance(value, bytes))
    envelope = next(
        value for value in referents if isinstance(value, SignatureEnvelope)
    )
    forged = object.__new__(epoch_module.SystemCapability)
    object.__setattr__(forged, "_SystemCapability__operation", operation)
    object.__setattr__(forged, "_SystemCapability__payload", payload)
    object.__setattr__(forged, "_SystemCapability__signature", envelope)
    with pytest.raises(AuthorityEpochError):
        fence.system_context(_system_identity(), capability=forged)

    recreated = EpochFence(
        tmp_path,
        roster=VerificationRoster.load(tmp_path, pin_store=_pin_store(tmp_path)),
        verifier=OperationalVerifier(),
        pin_store=_pin_store(tmp_path),
    )
    with pytest.raises(AuthorityEpochError):
        recreated.system_context(_system_identity(), capability=forged)


def test_bound_operation_dies_with_its_fence_and_process_nonce(
    tmp_path,
    monkeypatch,
) -> None:
    import memo.operational_epoch as epoch_module

    keys = DeviceKeyStore.in_memory()
    key = keys.generate(device_id="device-a", roles=("origin",))
    roster = VerificationRoster.bootstrap(
        device_id="device-a",
        key=key,
        root=tmp_path,
        pin_store=_pin_store(tmp_path),
    )
    signer = OperationalSigner(keys, roster_version=1)
    fence = EpochFence(
        tmp_path,
        roster=roster,
        verifier=OperationalVerifier(),
        pin_store=_pin_store(tmp_path),
    )
    authorization = _authorization(
        signer,
        key_id=key.key_id,
        epoch=0,
        control_oid="control-0",
        digests={"bootstrap_roster": "a" * 64, "empty_anchor": "b" * 64},
    )
    fence.bootstrap(
        authorization=authorization,
        observed_artifact_digests=authorization.artifact_digests,
    )
    operation = epoch_module.bind_system_context(
        fence,
        signer=signer,
        key_id=key.key_id,
        system_role="daemon",
    )
    operation(_system_identity())

    monkeypatch.setattr(epoch_module, "_PROCESS_NONCE", b"recreated-process" * 2)
    with pytest.raises(AuthorityEpochError):
        operation(_system_identity())
    monkeypatch.undo()

    fence_reference = weakref.ref(fence)
    del fence
    gc.collect()
    assert fence_reference() is None
    EpochFence(
        tmp_path,
        roster=VerificationRoster.load(tmp_path, pin_store=_pin_store(tmp_path)),
        verifier=OperationalVerifier(),
        pin_store=_pin_store(tmp_path),
    )
    with pytest.raises(AuthorityEpochError):
        operation(_system_identity())


def test_retained_proof_is_rejected_after_fence_identifier_reuse(tmp_path) -> None:
    import memo.operational_epoch as epoch_module

    keys = DeviceKeyStore.in_memory()
    key = keys.generate(device_id="device-a", roles=("origin",))
    pin_store = _pin_store(tmp_path)
    roster = VerificationRoster.bootstrap(
        device_id="device-a",
        key=key,
        root=tmp_path,
        pin_store=pin_store,
    )
    signer = OperationalSigner(keys, roster_version=1)
    fence = EpochFence(
        tmp_path,
        roster=roster,
        verifier=OperationalVerifier(),
        pin_store=pin_store,
    )
    authorization = _authorization(
        signer,
        key_id=key.key_id,
        epoch=0,
        control_oid="control-0",
        digests={"bootstrap_roster": "a" * 64, "empty_anchor": "b" * 64},
    )
    fence.bootstrap(
        authorization=authorization,
        observed_artifact_digests=authorization.artifact_digests,
    )
    original_fences = [fence]
    original_fences.extend(
        EpochFence(
            tmp_path,
            roster=roster,
            verifier=OperationalVerifier(),
            pin_store=pin_store,
        )
        for _ in range(31)
    )
    retained_proofs: dict[int, tuple[bytes, SignatureEnvelope]] = {}
    fence_references: list[weakref.ReferenceType[EpochFence]] = []
    for original_fence in original_fences:
        operation = epoch_module.bind_system_context(
            original_fence,
            signer=signer,
            key_id=key.key_id,
            system_role="daemon",
        )
        referents = gc.get_referents(operation)
        payload = next(value for value in referents if isinstance(value, bytes))
        envelope = next(
            value for value in referents if isinstance(value, SignatureEnvelope)
        )
        retained_proofs[id(original_fence)] = (payload, envelope)
        fence_references.append(weakref.ref(original_fence))

    del operation
    del original_fence
    del fence
    del original_fences
    gc.collect()
    assert all(reference() is None for reference in fence_references)

    reused_identifier = False
    replacements: list[EpochFence] = []
    for _ in range(4096):
        replacement = EpochFence.__new__(EpochFence)
        replacements.append(replacement)
        retained_proof = retained_proofs.get(id(replacement))
        if retained_proof is not None:
            reused_identifier = True
            replacement.__init__(
                tmp_path,
                roster=roster,
                verifier=OperationalVerifier(),
                pin_store=pin_store,
            )
            payload, envelope = retained_proof
            replay = epoch_module._BoundSystemContext(  # type: ignore[attr-defined]
                replacement,
                payload,
                envelope,
            )
            with pytest.raises(AuthorityEpochError):
                replay(_system_identity())
            break

    assert reused_identifier, "test did not exercise runtime fence identifier reuse"


def test_fence_instance_nonce_is_unique_write_once_and_not_serializable(
    tmp_path,
) -> None:
    import memo.operational_epoch as epoch_module

    keys = DeviceKeyStore.in_memory()
    key = keys.generate(device_id="device-a", roles=("origin",))
    pin_store = _pin_store(tmp_path)
    roster = VerificationRoster.bootstrap(
        device_id="device-a",
        key=key,
        root=tmp_path,
        pin_store=pin_store,
    )
    fence = EpochFence(
        tmp_path,
        roster=roster,
        verifier=OperationalVerifier(),
        pin_store=pin_store,
    )
    initial_nonce = epoch_module._fence_nonce(fence)  # type: ignore[attr-defined]
    replacement = EpochFence(
        tmp_path,
        roster=roster,
        verifier=OperationalVerifier(),
        pin_store=pin_store,
    )
    assert epoch_module._fence_nonce(replacement) != initial_nonce  # type: ignore[attr-defined]

    with pytest.raises(AttributeError):
        fence._EpochFence__instance_nonce = b"reset" * 8  # type: ignore[attr-defined]
    with pytest.raises(AttributeError):
        object.__setattr__(
            fence,
            "_EpochFence__instance_nonce",
            bytes.fromhex(epoch_module._fence_nonce(replacement)),  # type: ignore[attr-defined]
        )
    fence.__init__(
        tmp_path,
        roster=roster,
        verifier=OperationalVerifier(),
        pin_store=pin_store,
    )
    assert epoch_module._fence_nonce(fence) == initial_nonce  # type: ignore[attr-defined]

    with pytest.raises(TypeError):
        copy.copy(fence)
    with pytest.raises(TypeError):
        copy.deepcopy(fence)
    with pytest.raises(TypeError):
        pickle.dumps(fence)

    reconstructed = object.__new__(EpochFence)
    reconstructed.__init__(
        tmp_path,
        roster=roster,
        verifier=OperationalVerifier(),
        pin_store=pin_store,
    )
    with pytest.raises(AuthorityEpochError):
        epoch_module._fence_nonce(reconstructed)  # type: ignore[attr-defined]


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
