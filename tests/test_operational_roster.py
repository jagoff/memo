from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest

from memo.operational_key_store import DeviceKeyStore
from memo.operational_roster import (
    RosterError,
    VerificationRoster,
    verify_bootstrap,
)
from memo.operational_signing import OperationalSigner


def test_fresh_bootstrap_persists_one_peer_roster_before_epoch_zero(tmp_path) -> None:
    store = DeviceKeyStore.in_memory()
    key = store.generate(device_id="device-a")
    roster = VerificationRoster.bootstrap(
        device_id="device-a", key=key, root=tmp_path
    )
    assert roster.version == 1
    assert roster.peers == ("device-a",)
    assert roster.local_key_id == key.key_id
    assert verify_bootstrap(roster, key)
    assert (tmp_path / "verification-roster.json").exists()
    assert not (tmp_path / "authority-epoch.json").exists()


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
        device_id="device-a", key=first, root=tmp_path
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
    roster = VerificationRoster.bootstrap(device_id="device-a", key=key, root=tmp_path)
    loaded = VerificationRoster.load(tmp_path)
    assert loaded == roster
    persisted = json.loads(
        (tmp_path / "verification-roster.json").read_text(encoding="utf-8")
    )
    assert persisted["signature"]["algorithm"] == "ed25519"
    assert persisted["signature"]["key_id"] == key.key_id
    assert (tmp_path / "verification-rosters/00000001.json").is_file()

    persisted["previous_roster_hash"] = "attacker"
    _rehash_roster(persisted)
    (tmp_path / "verification-roster.json").write_text(
        json.dumps(persisted), encoding="utf-8"
    )
    (tmp_path / "verification-rosters/00000001.json").write_text(
        json.dumps(persisted), encoding="utf-8"
    )
    with pytest.raises(RosterError):
        VerificationRoster.load(tmp_path)


def test_roster_load_rejects_invalid_pop_and_trust_root_substitution(tmp_path) -> None:
    store = DeviceKeyStore.in_memory()
    key = store.generate(device_id="device-a")
    VerificationRoster.bootstrap(device_id="device-a", key=key, root=tmp_path)
    history = tmp_path / "verification-rosters/00000001.json"
    body = json.loads(history.read_text(encoding="utf-8"))
    body["keys"][0]["proof_of_possession"] = "invalid"
    _rehash_roster(body)
    history.write_text(json.dumps(body), encoding="utf-8")
    (tmp_path / "verification-roster.json").write_text(
        json.dumps(body), encoding="utf-8"
    )
    with pytest.raises(RosterError):
        VerificationRoster.load(tmp_path)


def test_signed_roster_updates_form_immutable_history(tmp_path) -> None:
    store = DeviceKeyStore.in_memory()
    first = store.generate(device_id="device-a")
    roster = VerificationRoster.bootstrap(
        device_id="device-a", key=first, root=tmp_path
    )
    second = store.generate(
        device_id="device-b", roles=("origin",), enrollment_sequence=2
    )
    signer = OperationalSigner(store, roster_version=roster.version)
    updated = roster.with_keys(
        version=2,
        peers=("device-a", "device-b"),
        keys=(first, second),
        signer=signer,
        root=tmp_path,
    )
    assert updated.previous_roster_hash == roster.roster_hash
    assert VerificationRoster.load(tmp_path) == updated
    assert (tmp_path / "verification-rosters/00000001.json").is_file()
    assert (tmp_path / "verification-rosters/00000002.json").is_file()

    version_one = tmp_path / "verification-rosters/00000001.json"
    tampered = json.loads(version_one.read_text(encoding="utf-8"))
    tampered["local_device_id"] = "device-b"
    _rehash_roster(tampered)
    version_one.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(RosterError):
        VerificationRoster.load(tmp_path)
