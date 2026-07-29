from __future__ import annotations

from dataclasses import replace

import pytest

from memo.operational_key_store import DeviceKeyStore
from memo.operational_roster import (
    RosterError,
    VerificationRoster,
    verify_bootstrap,
)


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
