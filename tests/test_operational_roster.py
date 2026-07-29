from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from memo.operational_key_store import (
    AuthorityPinStore,
    DeviceKeyStore,
    InMemoryAuthorityPinProvider,
)
from memo.operational_roster import (
    RosterError,
    VerificationRoster,
    verify_bootstrap,
)
from memo.operational_signing import OperationalSigner

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


def test_public_bootstrap_contract_uses_root_bound_pin_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _use_in_memory_productive_pin_factory(monkeypatch)
    store = DeviceKeyStore.in_memory()
    key = store.generate(device_id="device-a")

    roster = VerificationRoster.bootstrap(
        device_id="device-a",
        key=key,
        root=tmp_path,
    )

    assert roster.version == 1
    assert roster.peers == ("device-a",)


def test_public_load_contract_uses_root_bound_pin_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _use_in_memory_productive_pin_factory(monkeypatch)
    store = DeviceKeyStore.in_memory()
    key = store.generate(device_id="device-a")
    roster = VerificationRoster.bootstrap(
        device_id="device-a",
        key=key,
        root=tmp_path,
        pin_store=AuthorityPinStore._for_test(tmp_path, provider=provider),
    )

    assert VerificationRoster.load(tmp_path) == roster


def test_fresh_bootstrap_persists_one_peer_roster_before_epoch_zero(tmp_path) -> None:
    store = DeviceKeyStore.in_memory()
    key = store.generate(device_id="device-a")
    roster = VerificationRoster.bootstrap(
        device_id="device-a", key=key, root=tmp_path, pin_store=_pin_store(tmp_path)
    )
    assert roster.version == 1
    assert roster.peers == ("device-a",)
    assert roster.local_key_id == key.key_id
    assert verify_bootstrap(roster, key)
    assert (tmp_path / "verification-roster.json").exists()
    assert not (tmp_path / "authority-epoch.json").exists()


def test_bootstrap_recovers_crash_between_history_and_current(tmp_path, monkeypatch) -> None:
    import memo.operational_roster as roster_module

    store = DeviceKeyStore.in_memory()
    key = store.generate(device_id="device-a")
    original = roster_module._atomic_authority_write

    def crash_before_current(path: object, data: bytes) -> None:
        del path, data
        raise OSError("fault injection after roster history")

    monkeypatch.setattr(
        roster_module,
        "_atomic_authority_write",
        crash_before_current,
    )
    with pytest.raises(OSError, match="fault injection"):
        VerificationRoster.bootstrap(
            device_id="device-a",
            key=key,
            root=tmp_path,
            pin_store=_pin_store(tmp_path),
        )
    monkeypatch.setattr(roster_module, "_atomic_authority_write", original)

    recovered = VerificationRoster.bootstrap(
        device_id="device-a",
        key=key,
        root=tmp_path,
        pin_store=_pin_store(tmp_path),
    )
    assert (
        VerificationRoster.load(
            tmp_path,
            pin_store=_pin_store(tmp_path),
        )
        == recovered
    )


def test_load_recovers_bootstrap_crash_before_first_root_write(
    tmp_path,
    monkeypatch,
) -> None:
    import memo.operational_roster as roster_module

    store = DeviceKeyStore.in_memory()
    key = store.generate(device_id="device-a")
    original = roster_module._create_authority_file

    def crash_before_history(path: object, data: bytes) -> None:
        del path, data
        raise OSError("fault injection before first roster write")

    monkeypatch.setattr(roster_module, "_create_authority_file", crash_before_history)
    with pytest.raises(OSError, match="before first roster write"):
        VerificationRoster.bootstrap(
            device_id="device-a",
            key=key,
            root=tmp_path,
            pin_store=_pin_store(tmp_path),
        )
    monkeypatch.setattr(roster_module, "_create_authority_file", original)

    recovered = VerificationRoster.load(tmp_path, pin_store=_pin_store(tmp_path))
    assert recovered.version == 1
    assert recovered.roster_hash
    assert (tmp_path / "verification-rosters/00000001.json").is_file()
    assert (tmp_path / "verification-roster.json").is_file()


def test_roster_rejects_duplicate_ids_fingerprints_and_regression(tmp_path) -> None:
    store = DeviceKeyStore.in_memory()
    first = store.generate(device_id="device-a")
    second = store.generate(device_id="device-b")
    with pytest.raises(RosterError):
        VerificationRoster(
            version=1,
            peers=("device-a", "device-b"),
            keys=(first, replace(second, key_id=first.key_id)),
            local_device_id="device-a",
        )
    with pytest.raises(RosterError):
        VerificationRoster(
            version=1,
            peers=("device-a", "device-b"),
            keys=(first, replace(second, fingerprint=first.fingerprint)),
            local_device_id="device-a",
        )
    roster = VerificationRoster.bootstrap(
        device_id="device-a", key=first, root=tmp_path, pin_store=_pin_store(tmp_path)
    )
    with pytest.raises(RosterError):
        roster.with_keys(version=1, peers=("device-a",), keys=(first,))


def _rehash_roster(body: dict[str, object]) -> None:
    body["roster_hash"] = ""
    body["roster_hash"] = hashlib.sha256(
        json.dumps(
            body,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()


def test_bootstrap_roster_is_signed_and_reload_verifies_every_field(tmp_path) -> None:
    store = DeviceKeyStore.in_memory()
    key = store.generate(device_id="device-a")
    roster = VerificationRoster.bootstrap(
        device_id="device-a", key=key, root=tmp_path, pin_store=_pin_store(tmp_path)
    )
    loaded = VerificationRoster.load(tmp_path, pin_store=_pin_store(tmp_path))
    assert loaded == roster
    persisted = json.loads((tmp_path / "verification-roster.json").read_text(encoding="utf-8"))
    assert persisted["signature"]["algorithm"] == "ed25519"
    assert persisted["signature"]["key_id"] == key.key_id
    assert (tmp_path / "verification-rosters/00000001.json").is_file()

    persisted["previous_roster_hash"] = "attacker"
    _rehash_roster(persisted)
    (tmp_path / "verification-roster.json").write_text(json.dumps(persisted), encoding="utf-8")
    (tmp_path / "verification-rosters/00000001.json").write_text(
        json.dumps(persisted), encoding="utf-8"
    )
    with pytest.raises(RosterError):
        VerificationRoster.load(tmp_path, pin_store=_pin_store(tmp_path))


def test_roster_load_rejects_invalid_pop_and_trust_root_substitution(tmp_path) -> None:
    store = DeviceKeyStore.in_memory()
    key = store.generate(device_id="device-a")
    VerificationRoster.bootstrap(
        device_id="device-a", key=key, root=tmp_path, pin_store=_pin_store(tmp_path)
    )
    history = tmp_path / "verification-rosters/00000001.json"
    body = json.loads(history.read_text(encoding="utf-8"))
    body["keys"][0]["proof_of_possession"] = "invalid"
    _rehash_roster(body)
    history.write_text(json.dumps(body), encoding="utf-8")
    (tmp_path / "verification-roster.json").write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(RosterError):
        VerificationRoster.load(tmp_path, pin_store=_pin_store(tmp_path))


def test_signed_roster_updates_form_immutable_history(tmp_path) -> None:
    store = DeviceKeyStore.in_memory()
    first = store.generate(device_id="device-a")
    roster = VerificationRoster.bootstrap(
        device_id="device-a", key=first, root=tmp_path, pin_store=_pin_store(tmp_path)
    )
    second = store.generate(device_id="device-b", roles=("origin",), enrollment_sequence=2)
    signer = OperationalSigner(store, roster_version=roster.version)
    updated = roster.with_keys(
        version=2,
        peers=("device-a", "device-b"),
        keys=(first, second),
        signer=signer,
        root=tmp_path,
        pin_store=_pin_store(tmp_path),
    )
    assert updated.previous_roster_hash == roster.roster_hash
    assert VerificationRoster.load(tmp_path, pin_store=_pin_store(tmp_path)) == updated
    assert (tmp_path / "verification-rosters/00000001.json").is_file()
    assert (tmp_path / "verification-rosters/00000002.json").is_file()

    version_one = tmp_path / "verification-rosters/00000001.json"
    tampered = json.loads(version_one.read_text(encoding="utf-8"))
    tampered["local_device_id"] = "device-b"
    _rehash_roster(tampered)
    version_one.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(RosterError):
        VerificationRoster.load(tmp_path, pin_store=_pin_store(tmp_path))


def test_roster_update_rejects_noncanonical_unpinned_predecessor(tmp_path: Path) -> None:
    store = DeviceKeyStore.in_memory()
    first = store.generate(device_id="device-a")
    roster = VerificationRoster.bootstrap(
        device_id="device-a",
        key=first,
        root=tmp_path,
        pin_store=_pin_store(tmp_path),
    )
    assert roster.signature is not None
    detached = replace(
        roster,
        signature=replace(
            roster.signature,
            signature=f"{roster.signature.signature}==",
        ),
    )
    assert detached != roster
    assert detached.roster_hash == roster.roster_hash
    assert verify_bootstrap(detached, first)

    second = store.generate(
        device_id="device-b",
        roles=("origin",),
        enrollment_sequence=2,
    )
    with pytest.raises(RosterError, match="pinned predecessor"):
        detached.with_keys(
            version=2,
            peers=("device-a", "device-b"),
            keys=(first, second),
            signer=OperationalSigner(store, roster_version=1),
            root=tmp_path,
            pin_store=_pin_store(tmp_path),
        )


def test_roster_update_accepts_exact_pinned_predecessor(tmp_path: Path) -> None:
    store = DeviceKeyStore.in_memory()
    first = store.generate(device_id="device-a")
    roster = VerificationRoster.bootstrap(
        device_id="device-a",
        key=first,
        root=tmp_path,
        pin_store=_pin_store(tmp_path),
    )
    second = store.generate(
        device_id="device-b",
        roles=("origin",),
        enrollment_sequence=2,
    )

    updated = roster.with_keys(
        version=2,
        peers=("device-a", "device-b"),
        keys=(first, second),
        signer=OperationalSigner(store, roster_version=1),
        root=tmp_path,
        pin_store=_pin_store(tmp_path),
    )

    assert VerificationRoster.load(
        tmp_path,
        pin_store=_pin_store(tmp_path),
    ) == updated


def test_load_recovers_crash_between_roster_history_and_current(tmp_path, monkeypatch) -> None:
    import memo.operational_roster as roster_module

    store = DeviceKeyStore.in_memory()
    first = store.generate(device_id="device-a")
    roster = VerificationRoster.bootstrap(
        device_id="device-a",
        key=first,
        root=tmp_path,
        pin_store=_pin_store(tmp_path),
    )
    second = store.generate(
        device_id="device-b",
        roles=("origin",),
        enrollment_sequence=2,
    )
    original = roster_module._atomic_authority_write

    def crash_before_current(path: object, data: bytes) -> None:
        del path, data
        raise OSError("fault injection after roster update history")

    monkeypatch.setattr(
        roster_module,
        "_atomic_authority_write",
        crash_before_current,
    )
    with pytest.raises(OSError, match="fault injection"):
        roster.with_keys(
            version=2,
            peers=("device-a", "device-b"),
            keys=(first, second),
            signer=OperationalSigner(store, roster_version=1),
            root=tmp_path,
            pin_store=_pin_store(tmp_path),
        )
    monkeypatch.setattr(roster_module, "_atomic_authority_write", original)

    recovered = VerificationRoster.load(
        tmp_path,
        pin_store=_pin_store(tmp_path),
    )
    assert recovered.version == 2
    assert _pin_store(tmp_path)._snapshot_for_test().roster_version == 2


def test_load_recovers_update_crash_before_first_root_write(tmp_path, monkeypatch) -> None:
    import memo.operational_roster as roster_module

    store = DeviceKeyStore.in_memory()
    first = store.generate(device_id="device-a")
    roster = VerificationRoster.bootstrap(
        device_id="device-a",
        key=first,
        root=tmp_path,
        pin_store=_pin_store(tmp_path),
    )
    second = store.generate(
        device_id="device-b",
        roles=("origin",),
        enrollment_sequence=2,
    )
    original = roster_module._create_authority_file

    def crash_before_update_history(path: object, data: bytes) -> None:
        del path, data
        raise OSError("fault injection before first roster update write")

    monkeypatch.setattr(
        roster_module,
        "_create_authority_file",
        crash_before_update_history,
    )
    with pytest.raises(OSError, match="before first roster update write"):
        roster.with_keys(
            version=2,
            peers=("device-a", "device-b"),
            keys=(first, second),
            signer=OperationalSigner(store, roster_version=1),
            root=tmp_path,
            pin_store=_pin_store(tmp_path),
        )
    monkeypatch.setattr(roster_module, "_create_authority_file", original)

    recovered = VerificationRoster.load(tmp_path, pin_store=_pin_store(tmp_path))
    assert recovered.version == 2
    assert recovered.previous_roster_hash == roster.roster_hash


def test_roster_rejects_valid_history_truncation_and_current_rollback(
    tmp_path,
) -> None:
    store = DeviceKeyStore.in_memory()
    first = store.generate(device_id="device-a")
    roster = VerificationRoster.bootstrap(
        device_id="device-a", key=first, root=tmp_path, pin_store=_pin_store(tmp_path)
    )
    version_one = (tmp_path / "verification-rosters/00000001.json").read_bytes()
    second = store.generate(device_id="device-b", roles=("origin",), enrollment_sequence=2)
    roster.with_keys(
        version=2,
        peers=("device-a", "device-b"),
        keys=(first, second),
        signer=OperationalSigner(store, roster_version=1),
        root=tmp_path,
        pin_store=_pin_store(tmp_path),
    )

    (tmp_path / "verification-rosters/00000002.json").unlink()
    (tmp_path / "verification-roster.json").write_bytes(version_one)

    with pytest.raises(RosterError):
        VerificationRoster.load(tmp_path, pin_store=_pin_store(tmp_path))


def test_roster_rejects_coordinated_trust_root_substitution(tmp_path) -> None:
    store = DeviceKeyStore.in_memory()
    first = store.generate(device_id="device-a")
    VerificationRoster.bootstrap(
        device_id="device-a", key=first, root=tmp_path, pin_store=_pin_store(tmp_path)
    )

    attacker_root = tmp_path / "attacker"
    attacker = store.generate(device_id="device-a")
    VerificationRoster.bootstrap(
        device_id="device-a",
        key=attacker,
        root=attacker_root,
        pin_store=_pin_store(attacker_root),
    )
    substituted = (attacker_root / "verification-rosters/00000001.json").read_bytes()
    (tmp_path / "verification-rosters/00000001.json").write_bytes(substituted)
    (tmp_path / "verification-roster.json").write_bytes(substituted)

    with pytest.raises(RosterError):
        VerificationRoster.load(tmp_path, pin_store=_pin_store(tmp_path))
