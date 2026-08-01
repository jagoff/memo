from __future__ import annotations

import base64
import contextlib
import hashlib
import json
import os
import subprocess
import sys
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.asymmetric.utils import (
    decode_dss_signature,
    encode_dss_signature,
)

from memo.errors import SignatureError
from memo.identity import PrincipalIdentity
from memo.operation_ledger_v2 import OperationLedgerV2
from memo.operational_epoch import EpochFence
from memo.operational_event import (
    EpochMarkerAuthorization,
    OperationalCommand,
    canonical_signed_bytes,
)
from memo.operational_event_types import FOCUS_SET
from memo.operational_key_store import (
    AuthorityPinStore,
    DeviceKeyStore,
    InMemoryAuthorityPinProvider,
    KeyStoreError,
    MacOSAuthorityPinProvider,
    MacOSKeychainProvider,
    SignatureAlgorithm,
)
from memo.operational_macos_secure_enclave import SecureEnclaveP256Backend
from memo.operational_roster import VerificationRoster
from memo.operational_signing import OperationalSigner, OperationalVerifier

_P256_ORDER = int(
    "FFFFFFFF00000000FFFFFFFFFFFFFFFFBCE6FAADA7179E84F3B9CAC2FC632551",
    16,
)


def test_in_memory_key_store_generates_signs_and_destroys() -> None:
    store = DeviceKeyStore.in_memory()
    record = store.generate(device_id="device-a")
    assert record.device_id == "device-a"
    assert record.key_id
    assert len(record.fingerprint) == 64
    assert record.public_key
    assert set(record.roles) == {"origin", "migration_attestor"}
    assert "algorithm" not in record.to_dict()
    assert b'"algorithm"' not in record.proof_payload()
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
    algorithm: SignatureAlgorithm = "ed25519"

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


class _P256OpaqueProvider:
    algorithm: SignatureAlgorithm = "ecdsa-p256-sha256"

    def __init__(self) -> None:
        self.keys: dict[str, ec.EllipticCurvePrivateKey] = {}

    def generate(self, key_id: str) -> bytes:
        if key_id in self.keys:
            raise KeyStoreError(f"duplicate private key id: {key_id}")
        key = ec.generate_private_key(ec.SECP256R1())
        self.keys[key_id] = key
        return key.public_key().public_bytes(
            serialization.Encoding.X962,
            serialization.PublicFormat.UncompressedPoint,
        )

    def sign(self, key_id: str, payload: bytes) -> bytes:
        try:
            key = self.keys[key_id]
        except KeyError as exc:
            raise KeyStoreError(f"unknown private key id: {key_id}") from exc
        return key.sign(bytes(payload), ec.ECDSA(hashes.SHA256()))

    def destroy(self, key_id: str) -> None:
        if self.keys.pop(key_id, None) is None:
            raise KeyStoreError(f"unknown private key id: {key_id}")


def test_productive_provider_delegates_nonexportable_p256_lifecycle(
    tmp_path,
) -> None:
    backend = _P256OpaqueProvider()
    provider = MacOSKeychainProvider(backend=backend)
    store = DeviceKeyStore(provider)
    record = store.generate(device_id="device-a")

    assert record.algorithm == "ecdsa-p256-sha256"
    assert record.key_id.startswith("p256-se-")
    assert record.to_dict()["algorithm"] == "ecdsa-p256-sha256"
    roster = VerificationRoster.bootstrap(
        device_id="device-a",
        key=record,
        root=tmp_path / "authority",
        pin_store=AuthorityPinStore._for_test(tmp_path / "authority"),
    )
    signer = OperationalSigner(store, roster_version=1)
    envelope = signer.sign(
        domain="memo.cutover.vote.v1",
        payload=b"{}",
        key_id=record.key_id,
    )
    assert envelope.algorithm == "ecdsa-p256-sha256"
    OperationalVerifier().verify(
        domain="memo.cutover.vote.v1",
        payload=b"{}",
        envelope=envelope,
        roster=roster,
    )
    raw_signature = base64.urlsafe_b64decode(
        envelope.signature + "=" * (-len(envelope.signature) % 4)
    )
    r, s = decode_dss_signature(raw_signature)
    high_s = encode_dss_signature(r, _P256_ORDER - s)
    with pytest.raises(SignatureError, match="verification failed"):
        OperationalVerifier().verify(
            domain="memo.cutover.vote.v1",
            payload=b"{}",
            envelope=type(envelope)(
                algorithm=envelope.algorithm,
                key_id=envelope.key_id,
                roster_version=envelope.roster_version,
                signature=base64.urlsafe_b64encode(high_s).rstrip(b"=").decode("ascii"),
            ),
            roster=roster,
        )

    store.destroy(key_id=record.key_id)
    with pytest.raises(KeyStoreError, match="unknown private key"):
        store.sign(key_id=record.key_id, payload=b"payload")


def test_p256_roster_drives_anchor_and_event_verification(tmp_path) -> None:
    authority_root = tmp_path / "authority"
    backend = _P256OpaqueProvider()
    store = DeviceKeyStore(MacOSKeychainProvider(backend=backend))
    key = store.generate(device_id="device-a", roles=("origin",))
    pin_store = AuthorityPinStore._for_test(authority_root)
    roster = VerificationRoster.bootstrap(
        device_id="device-a",
        key=key,
        root=authority_root,
        pin_store=pin_store,
    )
    signer = OperationalSigner(store, roster_version=roster.version)
    verifier = OperationalVerifier()
    fence = EpochFence(
        authority_root,
        roster=roster,
        verifier=verifier,
        pin_store=pin_store,
    )
    unsigned = EpochMarkerAuthorization(
        schema="memo.operational_epoch_authorization.v1",
        attempt_id="attempt-0",
        device_id="device-a",
        epoch=0,
        control_oid="control-0",
        artifact_digests={
            "bootstrap_roster": "a" * 64,
            "empty_anchor": "b" * 64,
        },
        roster_version=roster.version,
        key_id=key.key_id,
        signature=None,  # type: ignore[arg-type]
    )
    authorization = replace(
        unsigned,
        signature=signer.sign(
            domain="memo.operational_epoch_authorization.v1",
            payload=canonical_signed_bytes(unsigned),
            key_id=key.key_id,
        ),
    )
    fence.bootstrap(
        authorization=authorization,
        observed_artifact_digests=authorization.artifact_digests,
    )
    identity = PrincipalIdentity(
        principal_id="device-a:session-a",
        actor_id="agent-a",
        kind="agent",
        device_id="device-a",
        session_id="session-a",
        source_client="codex",
    )
    ledger = OperationLedgerV2(
        tmp_path / "operational",
        device_id="device-a",
        clock=lambda: "2026-07-30T12:00:00Z",
        signer=signer,
        verifier=verifier,
        roster=roster,
        roster_root=authority_root,
        pin_store=pin_store,
        epoch_fence=fence,
    )
    anchor = ledger.ensure_anchor()
    event = ledger.append(
        OperationalCommand(
            event_type=FOCUS_SET,
            actor=identity,
            target_id=None,
            project="demo",
            workspace="/tmp/demo",
            expires_at=None,
            visibility="owner",
            idempotency_key="idem-p256-1",
            caused_by=(),
            subject_uri="memo://focus/demo",
            trace_id="trace-p256-1",
            payload={"project": "demo", "summary": "P-256 authority"},
            source_proof=None,
        ),
        context=fence.context(
            identity,
            request_epoch=0,
            request_control_oid="control-0",
        ),
    )

    assert anchor.key_id == key.key_id
    assert event.key_id == key.key_id
    assert ledger.verify().ok is True


@pytest.mark.parametrize(
    "service",
    [
        "",
        "com.apple.login",
        "com.memo.operational-signing.",
        "com.memo.operational-signing.bad_name",
        "com.memo.operational-signing..test",
    ],
)
def test_productive_provider_rejects_unsafe_keychain_service(
    service: str,
) -> None:
    with pytest.raises(KeyStoreError, match="service is unsafe"):
        MacOSKeychainProvider(
            service=service,
            backend=_P256OpaqueProvider(),
        )


def test_productive_provider_rejects_algorithm_downgrade() -> None:
    with pytest.raises(KeyStoreError, match="algorithm mismatch"):
        MacOSKeychainProvider(backend=_OpaqueProvider())


def test_productive_provider_rejects_unsafe_key_id_before_backend() -> None:
    backend = _P256OpaqueProvider()
    provider = MacOSKeychainProvider(backend=backend)

    for operation in (
        lambda: provider.generate("ed25519-" + "0" * 32),
        lambda: provider.sign("p256-se-short", b"payload"),
        lambda: provider.destroy("p256-se-" + "A" * 32),
    ):
        with pytest.raises(KeyStoreError, match="key id is unsafe"):
            operation()
    assert not backend.keys


class _InvalidP256PublicProvider:
    algorithm: SignatureAlgorithm = "ecdsa-p256-sha256"

    def __init__(self) -> None:
        self.destroyed: list[str] = []

    def generate(self, key_id: str) -> bytes:
        return b"\x04" + b"\x00" * 64

    def sign(self, key_id: str, payload: bytes) -> bytes:
        raise AssertionError("invalid public key must not be signed")

    def destroy(self, key_id: str) -> None:
        self.destroyed.append(key_id)


def test_key_enrollment_destroys_invalid_generated_p256_key() -> None:
    backend = _InvalidP256PublicProvider()
    store = DeviceKeyStore(backend)

    with pytest.raises(KeyStoreError, match="enrollment failed"):
        store.generate(device_id="device-a")
    assert len(backend.destroyed) == 1
    assert backend.destroyed[0].startswith("p256-se-")


def test_secure_enclave_backend_fails_closed_off_macos(monkeypatch) -> None:
    import memo.operational_macos_secure_enclave as secure_enclave_module

    monkeypatch.setattr(secure_enclave_module.sys, "platform", "linux")
    with pytest.raises(KeyStoreError, match="unavailable"):
        SecureEnclaveP256Backend(service="com.memo.operational-signing")


def test_secure_enclave_helper_rejects_unsafe_mode(tmp_path) -> None:
    helper = tmp_path / "helper"
    helper.write_bytes(b"not executable code")
    helper.chmod(0o700)

    with pytest.raises(KeyStoreError, match="unsafe ownership or mode"):
        SecureEnclaveP256Backend._read_helper_snapshot(helper)


def test_secure_enclave_helper_detects_cached_binary_substitution(
    tmp_path,
    monkeypatch,
) -> None:
    helper = tmp_path / "helper"
    helper.write_bytes(b"trusted executable")
    helper.chmod(0o500)
    backend = object.__new__(SecureEnclaveP256Backend)
    backend._helper = helper
    backend._helper_sha256 = hashlib.sha256(helper.read_bytes()).hexdigest()
    monkeypatch.setattr(
        SecureEnclaveP256Backend,
        "_verify_code_signature",
        staticmethod(lambda _helper: None),
    )
    backend._verify_helper()

    helper.chmod(0o700)
    helper.write_bytes(b"substituted executable")
    helper.chmod(0o500)
    with pytest.raises(KeyStoreError, match="content-address verification"):
        backend._verify_helper()


@pytest.mark.skipif(
    sys.platform != "darwin" or os.getenv("MEMO_SECURE_ENCLAVE_TEST") != "1",
    reason="productive Secure Enclave smoke is opt-in",
)
def test_productive_secure_enclave_generates_reopens_signs_and_destroys() -> None:
    service = f"com.memo.operational-signing.test.{uuid.uuid4().hex}"
    provider = MacOSKeychainProvider(service=service)
    native_backend = provider._backend
    assert isinstance(native_backend, SecureEnclaveP256Backend)
    valid_key_id = "p256-se-" + "0" * 32
    invalid_service = subprocess.run(
        [
            str(native_backend._helper),
            "sign",
            "com.apple.login",
            valid_key_id,
        ],
        input=b"payload",
        check=False,
        capture_output=True,
        timeout=10,
    )
    assert invalid_service.returncode != 0
    assert invalid_service.stderr == b"invalid Keychain service"
    invalid_key_id = subprocess.run(
        [
            str(native_backend._helper),
            "sign",
            service,
            "p256-se-unsafe",
        ],
        input=b"payload",
        check=False,
        capture_output=True,
        timeout=10,
    )
    assert invalid_key_id.returncode != 0
    assert invalid_key_id.stderr == b"invalid private key id"

    first = DeviceKeyStore(provider)
    record = first.generate(device_id="device-a")
    try:
        reopened = DeviceKeyStore(MacOSKeychainProvider(service=service))
        signature = reopened.sign(key_id=record.key_id, payload=b"memo-smoke")
        public = ec.EllipticCurvePublicKey.from_encoded_point(
            ec.SECP256R1(),
            base64.urlsafe_b64decode(record.public_key + "=" * (-len(record.public_key) % 4)),
        )
        public.verify(signature, b"memo-smoke", ec.ECDSA(hashes.SHA256()))
        reopened.destroy(key_id=record.key_id)
        with pytest.raises(KeyStoreError, match="unknown private key"):
            first.sign(key_id=record.key_id, payload=b"memo-smoke")
    except BaseException:
        with contextlib.suppress(KeyStoreError):
            first.destroy(key_id=record.key_id)
        raise


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
