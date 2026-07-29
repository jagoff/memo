from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from memo.operational_key_store import (
    AuthorityPinStore,
    DeviceKeyStore,
    InMemoryAuthorityPinProvider,
    KeyStoreError,
    MacOSAuthorityPinProvider,
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


def test_authority_pin_provider_persists_monotonic_state_across_instances(tmp_path) -> None:
    provider = InMemoryAuthorityPinProvider()
    first = AuthorityPinStore._for_test(tmp_path, provider=provider)
    second = AuthorityPinStore._for_test(tmp_path, provider=provider)
    roster_record = json.dumps(
        {"roster_hash": "a" * 64, "version": 1},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    epoch_record = b'{"epoch":0}'

    first._stage_roster(tmp_path, roster_record)
    second._finish_roster(tmp_path, roster_record)
    first._stage_epoch(tmp_path, epoch_record, bootstrap=True)
    second._finish_epoch(tmp_path, epoch_record, bootstrap=True)

    state = first._snapshot_for_test()
    assert (state.roster_version, state.roster_hash) == (1, "a" * 64)
    assert state.epoch == 0
    assert state.bootstrap_state == "consumed"
    conflicting = json.dumps(
        {"roster_hash": "c" * 64, "version": 1},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    with pytest.raises(KeyStoreError):
        second._stage_roster(tmp_path, conflicting)


def test_authority_binding_is_root_derived_stable_and_concurrency_safe(tmp_path) -> None:
    provider = InMemoryAuthorityPinProvider()

    def bind() -> AuthorityPinStore:
        return AuthorityPinStore._for_test(tmp_path, provider=provider)

    with ThreadPoolExecutor(max_workers=8) as executor:
        stores = list(executor.map(lambda _: bind(), range(32)))

    installation_ids = {store._installation_id_for_test() for store in stores}
    assert len(installation_ids) == 1
    other = AuthorityPinStore._for_test(tmp_path / "other", provider=provider)
    assert other._installation_id_for_test() not in installation_ids

    with pytest.raises(TypeError):
        AuthorityPinStore(authority_id="caller-selected", provider=provider)
    for public_mutator in (
        "prepare_roster",
        "commit_roster",
        "prepare_epoch",
        "commit_epoch",
    ):
        assert not hasattr(AuthorityPinStore, public_mutator)
    with pytest.raises(KeyStoreError):
        stores[0]._stage_roster(
            tmp_path / "other",
            json.dumps(
                {"roster_hash": "a" * 64, "version": 1},
                sort_keys=True,
                separators=(",", ":"),
            ).encode(),
        )


def test_productive_authority_pin_provider_fails_closed_off_macos(
    monkeypatch,
) -> None:
    import memo.operational_key_store as key_store_module

    monkeypatch.setattr(key_store_module.sys, "platform", "linux")
    provider = MacOSAuthorityPinProvider()
    with pytest.raises(KeyStoreError) as exc:
        provider._read_pin("00000000-0000-0000-0000-000000000000")
    assert exc.value.__cause__ is None
    assert "failing closed" in str(exc.value)


def test_productive_authority_provider_separates_bindings_and_pin_accounts(
    tmp_path,
    monkeypatch,
) -> None:
    provider = MacOSAuthorityPinProvider()
    accounts: dict[str, bytes] = {}
    account_lock = threading.Lock()

    def read_account(account: str) -> bytes | None:
        with account_lock:
            return accounts.get(account)

    def write_account(account: str, value: bytes) -> None:
        with account_lock:
            accounts[account] = bytes(value)

    monkeypatch.setattr(provider, "_read_account", read_account)
    monkeypatch.setattr(provider, "_write_account", write_account)
    with ThreadPoolExecutor(max_workers=8) as executor:
        first_ids = set(
            executor.map(
                lambda _: provider._resolve_installation("canonical-root-a"),
                range(32),
            )
        )
    assert len(first_ids) == 1
    first_id = next(iter(first_ids))
    second_id = provider._resolve_installation("canonical-root-b")
    assert second_id != first_id

    provider._write_pin(first_id, b"first-pin")
    provider._write_pin(second_id, b"second-pin")
    assert provider._read_pin(first_id) == b"first-pin"
    assert provider._read_pin(second_id) == b"second-pin"

    first_store = AuthorityPinStore._create(tmp_path, provider=provider)
    second_store = AuthorityPinStore._create(tmp_path, provider=provider)
    roster_record = json.dumps(
        {"roster_hash": "d" * 64, "version": 1},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    with ThreadPoolExecutor(max_workers=2) as executor:
        list(
            executor.map(
                lambda store: store._stage_roster(tmp_path, roster_record),
                (first_store, second_store),
            )
        )
        list(
            executor.map(
                lambda store: store._finish_roster(tmp_path, roster_record),
                (first_store, second_store),
            )
        )
    assert first_store._snapshot_for_test() == second_store._snapshot_for_test()
    other_store = AuthorityPinStore._create(tmp_path / "other", provider=provider)
    assert other_store._snapshot_for_test().roster_version == 0
