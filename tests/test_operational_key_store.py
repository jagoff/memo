from __future__ import annotations

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from memo.operational_key_store import (
    DeviceKeyStore,
    KeyStoreError,
    MacOSKeychainProvider,
)


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


class _OpaqueProvider:
    def __init__(self) -> None:
        self.keys: dict[str, Ed25519PrivateKey] = {}
        self.calls: list[tuple[str, str]] = []

    def generate(self, key_id: str) -> bytes:
        self.calls.append(("generate", key_id))
        key = Ed25519PrivateKey.generate()
        self.keys[key_id] = key
        return key.public_key().public_bytes_raw()

    def sign(self, key_id: str, payload: bytes) -> bytes:
        self.calls.append(("sign", key_id))
        return self.keys[key_id].sign(payload)

    def destroy(self, key_id: str) -> None:
        self.calls.append(("destroy", key_id))
        del self.keys[key_id]


def test_device_key_store_delegates_to_opaque_provider_without_seed_export() -> None:
    provider = _OpaqueProvider()
    store = DeviceKeyStore(provider)
    record = store.generate(device_id="device-a")
    signature = store.sign(key_id=record.key_id, payload=b"payload")
    assert signature
    store.destroy(key_id=record.key_id)
    assert [call[0] for call in provider.calls] == [
        "generate",
        "sign",
        "sign",
        "sign",
        "destroy",
    ]
    assert not provider.keys


def test_productive_provider_fails_closed_without_nonexportable_ed25519() -> None:
    provider = MacOSKeychainProvider()
    store = DeviceKeyStore(provider)
    with pytest.raises(KeyStoreError) as exc:
        store.generate(device_id="device-a")
    assert exc.value.__cause__ is None
    assert "non-exportable" in str(exc.value)
