from __future__ import annotations

import pytest

from memo.operational_key_store import DeviceKeyStore, KeyStoreError


def test_in_memory_key_store_generates_signs_and_destroys() -> None:
    store = DeviceKeyStore.in_memory()
    record = store.generate(device_id="device-a")
    assert record.device_id == "device-a"
    assert record.key_id
    assert len(record.fingerprint) == 64
    assert record.public_key
    assert set(record.roles) == {"origin", "migration_attestor"}
    assert store.sign(key_id=record.key_id, payload=b"payload")
    store.destroy(key_id=record.key_id)
    with pytest.raises(KeyStoreError):
        store.sign(key_id=record.key_id, payload=b"payload")


def test_duplicate_device_generation_has_distinct_key_ids() -> None:
    store = DeviceKeyStore.in_memory()
    first = store.generate(device_id="device-a")
    second = store.generate(device_id="device-a")
    assert first.key_id != second.key_id
    assert first.fingerprint != second.fingerprint
