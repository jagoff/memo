from __future__ import annotations

import gc
import json
import multiprocessing
import os
import sqlite3
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from queue import Empty
from typing import Protocol

import pytest

from memo.errors import OperationalError
from memo.operational_signing import OperationalSigner, OperationalVerifier
from memo.terminal_bridge import (
    PresenterOutcome,
    TerminalBridge,
    TerminalPresentRequest,
    TerminalRegistration,
    sign_terminal_registration,
    verify_terminal_registration,
)
from tests.operational_authority import (
    TestFreshV2Authority as FreshV2Authority,
)
from tests.operational_authority import (
    build_test_fresh_v2_authority,
)


class _Queue(Protocol):
    def put(self, value: object) -> None: ...

    def get(self, *, timeout: float) -> object: ...

    def get_nowait(self) -> object: ...


class _Event(Protocol):
    def set(self) -> None: ...

    def wait(self, timeout: float | None = None) -> bool: ...


class FakePresenter:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.outcomes: list[PresenterOutcome | Exception] = []

    def present(
        self,
        *,
        terminal_id: str,
        mode: str,
        payload: str,
    ) -> PresenterOutcome:
        del terminal_id, mode
        self.calls.append(payload)
        if self.outcomes:
            outcome = self.outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome
        return PresenterOutcome(state="presented")


class _BlockingProcessPresenter:
    def __init__(self, presentations: _Queue, release: _Event) -> None:
        self.presentations = presentations
        self.release = release

    def present(
        self,
        *,
        terminal_id: str,
        mode: str,
        payload: str,
    ) -> PresenterOutcome:
        del terminal_id, mode
        self.presentations.put(payload)
        if not self.release.wait(10):
            raise RuntimeError("test presenter release timed out")
        return PresenterOutcome(state="presented")


def _present_in_process(
    database: Path,
    request: TerminalPresentRequest,
    now_text: str,
    ready: _Queue,
    start: _Event,
    results: _Queue,
    presentations: _Queue,
    release: _Event,
) -> None:
    now = datetime.fromisoformat(now_text)
    presenter = _BlockingProcessPresenter(presentations, release)
    with TerminalBridge(database, presenter=presenter, clock=lambda: now) as bridge:
        ready.put("ready")
        if not start.wait(10):
            raise RuntimeError("test process start timed out")
        receipt = bridge.present(request, peer_uid=os.getuid())
        results.put((receipt.state, receipt.attempt))


def _unsigned_registration(
    now: datetime,
    *,
    terminal_id: str = "term-1",
) -> TerminalRegistration:
    return TerminalRegistration(
        id=terminal_id,
        principal_id="principal-a",
        session_id="session-a",
        uid=os.getuid(),
        capabilities=("inject", "notify"),
        issued_at=now.isoformat(),
        expires_at=(now + timedelta(minutes=5)).isoformat(),
        nonce="nonce-a",
        signature="",
    )


def _registration(
    now: datetime,
    authority: FreshV2Authority,
    *,
    terminal_id: str = "term-1",
) -> TerminalRegistration:
    return sign_terminal_registration(
        _unsigned_registration(now, terminal_id=terminal_id),
        signer=authority.signer,
        key_id=authority.key_id,
        device_id="device-a",
    )


def _registration_verifier(
    authority: FreshV2Authority,
) -> Callable[[TerminalRegistration], bool]:
    verifier = OperationalVerifier()

    def verify(registration: TerminalRegistration) -> bool:
        return verify_terminal_registration(
            registration,
            verifier=verifier,
            roster=authority.roster,
        )

    return verify


def _request(
    now: datetime,
    *,
    payload: str = "handoff",
    event_id: str = "event-1",
    terminal_id: str = "term-1",
    mode: str = "notify",
) -> TerminalPresentRequest:
    return TerminalPresentRequest(
        event_id=event_id,
        message_id="message-1",
        delivery_id="delivery-1",
        terminal_id=terminal_id,
        mode=mode,
        payload=payload,
        sanitized_payload_hash="",
        deadline=(now + timedelta(minutes=1)).isoformat(),
        idempotency_key=f"present-{event_id}",
        principal_id="principal-a",
        session_id="session-a",
    )


def test_registration_verifier_defaults_to_fail_closed(tmp_path: Path) -> None:
    now = datetime(2026, 7, 31, 12, tzinfo=UTC)
    fake = replace(_unsigned_registration(now), signature="not-a-real-signature")

    with TerminalBridge(
        tmp_path / "terminal.sqlite",
        presenter=FakePresenter(),
        clock=lambda: now,
    ) as bridge:
        with pytest.raises(OperationalError, match="signature is invalid"):
            bridge.register(fake, peer_uid=os.getuid())


def test_registration_signature_binds_all_fields_device_and_key(tmp_path: Path) -> None:
    now = datetime(2026, 7, 31, 12, tzinfo=UTC)
    authority = build_test_fresh_v2_authority(
        tmp_path / "authority",
        device_id="device-a",
    )
    verifier = OperationalVerifier()
    signed = _registration(now, authority)

    assert verify_terminal_registration(signed, verifier=verifier, roster=authority.roster)
    tampered_registrations = (
        replace(signed, id="term-tampered"),
        replace(signed, principal_id="principal-tampered"),
        replace(signed, session_id="session-tampered"),
        replace(signed, uid=signed.uid + 1),
        replace(signed, capabilities=("notify",)),
        replace(signed, issued_at=(now + timedelta(seconds=1)).isoformat()),
        replace(signed, expires_at=(now + timedelta(minutes=6)).isoformat()),
        replace(signed, nonce="nonce-tampered"),
    )
    for tampered in tampered_registrations:
        assert not verify_terminal_registration(
            tampered,
            verifier=verifier,
            roster=authority.roster,
        )

    signature_body = json.loads(signed.signature)
    signature_body["signer_device_id"] = "device-b"
    wrong_device = replace(
        signed,
        signature=json.dumps(signature_body, sort_keys=True, separators=(",", ":")),
    )
    assert not verify_terminal_registration(
        wrong_device,
        verifier=verifier,
        roster=authority.roster,
    )

    signature_body = json.loads(signed.signature)
    signature_body["key_id"] = "ed25519-00000000000000000000000000000000"
    wrong_key = replace(
        signed,
        signature=json.dumps(signature_body, sort_keys=True, separators=(",", ":")),
    )
    assert not verify_terminal_registration(
        wrong_key,
        verifier=verifier,
        roster=authority.roster,
    )

    signature_body = json.loads(signed.signature)
    original_signature = signature_body["signature"]
    replacement_prefix = "A" if original_signature[0] != "A" else "B"
    signature_body["signature"] = f"{replacement_prefix}{original_signature[1:]}"
    false_signature = replace(
        signed,
        signature=json.dumps(signature_body, sort_keys=True, separators=(",", ":")),
    )
    assert not verify_terminal_registration(
        false_signature,
        verifier=verifier,
        roster=authority.roster,
    )
    correctly_signed_wrong_device = sign_terminal_registration(
        _unsigned_registration(now),
        signer=authority.signer,
        key_id=authority.key_id,
        device_id="device-b",
    )
    assert not verify_terminal_registration(
        correctly_signed_wrong_device,
        verifier=verifier,
        roster=authority.roster,
    )


def test_registration_nonce_cannot_be_replayed_for_another_terminal(tmp_path: Path) -> None:
    now = datetime(2026, 7, 31, 12, tzinfo=UTC)
    authority = build_test_fresh_v2_authority(tmp_path / "authority", device_id="device-a")
    first = _registration(now, authority, terminal_id="term-1")
    replay = sign_terminal_registration(
        replace(_unsigned_registration(now, terminal_id="term-2"), nonce=first.nonce),
        signer=authority.signer,
        key_id=authority.key_id,
        device_id="device-a",
    )
    with TerminalBridge(
        tmp_path / "terminal.sqlite",
        presenter=FakePresenter(),
        clock=lambda: now,
        registration_verifier=_registration_verifier(authority),
    ) as bridge:
        bridge.register(first, peer_uid=os.getuid())
        with pytest.raises(OperationalError, match="nonce has already been used"):
            bridge.register(replay, peer_uid=os.getuid())


def test_registration_signed_by_key_revoked_at_roster_version_is_rejected(
    tmp_path: Path,
) -> None:
    now = datetime(2026, 7, 31, 12, tzinfo=UTC)
    authority = build_test_fresh_v2_authority(tmp_path / "authority", device_id="device-a")
    signer = OperationalSigner(authority.signer.key_store, roster_version=2)
    signed = sign_terminal_registration(
        _unsigned_registration(now),
        signer=signer,
        key_id=authority.key_id,
        device_id="device-a",
    )
    revoked = replace(
        authority.roster,
        version=2,
        keys=(replace(authority.roster.keys[0], revocation_sequence=2),),
    )

    assert not verify_terminal_registration(
        signed,
        verifier=OperationalVerifier(),
        roster=revoked,
    )

def test_bridge_rejects_foreign_uid_and_sanitizes_payload(tmp_path: Path) -> None:
    now = datetime(2026, 7, 31, 12, tzinfo=UTC)
    authority = build_test_fresh_v2_authority(tmp_path / "authority", device_id="device-a")
    presenter = FakePresenter()
    with TerminalBridge(
        tmp_path / "terminal.sqlite",
        presenter=presenter,
        clock=lambda: now,
        registration_verifier=_registration_verifier(authority),
    ) as bridge:
        bridge.register(_registration(now, authority), peer_uid=os.getuid())

        with pytest.raises(OperationalError):
            bridge.present(_request(now, payload="hello"), peer_uid=os.getuid() + 1)

        receipt = bridge.present(
            _request(now, payload="\x1b]0;bad\x07safe\n"),
            peer_uid=os.getuid(),
        )

    assert presenter.calls == ["safe\n"]
    assert receipt.event_id == "event-1"
    assert receipt.state == "presented"
    assert receipt.receipt_hash


def test_duplicate_reserves_before_side_effect(tmp_path: Path) -> None:
    now = datetime(2026, 7, 31, 12, tzinfo=UTC)
    authority = build_test_fresh_v2_authority(tmp_path / "authority", device_id="device-a")
    presenter = FakePresenter()
    with TerminalBridge(
        tmp_path / "terminal.sqlite",
        presenter=presenter,
        clock=lambda: now,
        registration_verifier=_registration_verifier(authority),
    ) as bridge:
        bridge.register(_registration(now, authority), peer_uid=os.getuid())
        request = _request(now)

        first = bridge.present(request, peer_uid=os.getuid())
        duplicate = bridge.present(request, peer_uid=os.getuid())

    assert presenter.calls == ["handoff"]
    assert duplicate == first


def test_exactly_once_reservation_is_atomic_across_processes(tmp_path: Path) -> None:
    now = datetime(2026, 7, 31, 12, tzinfo=UTC)
    authority = build_test_fresh_v2_authority(tmp_path / "authority", device_id="device-a")
    database = tmp_path / "terminal.sqlite"
    with TerminalBridge(
        database,
        presenter=FakePresenter(),
        clock=lambda: now,
        registration_verifier=_registration_verifier(authority),
    ) as bridge:
        bridge.register(_registration(now, authority), peer_uid=os.getuid())

    context = multiprocessing.get_context("spawn")
    ready = context.Queue()
    start = context.Event()
    release = context.Event()
    results = context.Queue()
    presentations = context.Queue()
    request = _request(now)
    processes = [
        context.Process(
            target=_present_in_process,
            args=(
                database,
                request,
                now.isoformat(),
                ready,
                start,
                results,
                presentations,
                release,
            ),
        )
        for _ in range(2)
    ]
    for process in processes:
        process.start()
    try:
        assert ready.get(timeout=10) == "ready"
        assert ready.get(timeout=10) == "ready"
        start.set()
        assert presentations.get(timeout=10) == "handoff"
        assert results.get(timeout=10) == ("reserved", 1)
        release.set()
        assert results.get(timeout=10) == ("presented", 1)
    finally:
        release.set()
        for process in processes:
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)

    assert [process.exitcode for process in processes] == [0, 0]
    with pytest.raises(Empty):
        presentations.get_nowait()


def test_bridge_enforces_registration_binding_capability_and_payload_limit(
    tmp_path: Path,
) -> None:
    now = datetime(2026, 7, 31, 12, tzinfo=UTC)
    authority = build_test_fresh_v2_authority(tmp_path / "authority", device_id="device-a")
    presenter = FakePresenter()
    with TerminalBridge(
        tmp_path / "terminal.sqlite",
        presenter=presenter,
        clock=lambda: now,
        registration_verifier=_registration_verifier(authority),
    ) as bridge:
        registration = sign_terminal_registration(
            replace(_unsigned_registration(now), capabilities=("notify",)),
            signer=authority.signer,
            key_id=authority.key_id,
            device_id="device-a",
        )
        bridge.register(registration, peer_uid=os.getuid())

        with pytest.raises(OperationalError):
            bridge.present(_request(now, mode="inject"), peer_uid=os.getuid())
        with pytest.raises(OperationalError):
            bridge.present(
                replace(_request(now), session_id="foreign-session"),
                peer_uid=os.getuid(),
            )
        with pytest.raises(OperationalError):
            bridge.present(_request(now, payload="x" * 16_385), peer_uid=os.getuid())


def test_uncertain_requires_explicit_negative_reconciliation_before_retry(
    tmp_path: Path,
) -> None:
    current = datetime(2026, 7, 31, 12, tzinfo=UTC)
    authority = build_test_fresh_v2_authority(tmp_path / "authority", device_id="device-a")
    presenter = FakePresenter()
    presenter.outcomes = [RuntimeError("ambiguous presenter failure")]
    with TerminalBridge(
        tmp_path / "terminal.sqlite",
        presenter=presenter,
        clock=lambda: current,
        registration_verifier=_registration_verifier(authority),
    ) as bridge:
        bridge.register(_registration(current, authority), peer_uid=os.getuid())
        request = _request(current)

        uncertain = bridge.present(request, peer_uid=os.getuid())
        duplicate = bridge.present(request, peer_uid=os.getuid())
        assert uncertain.state == duplicate.state == "uncertain"
        assert len(presenter.calls) == 1

        reconciled = bridge.reconcile(
            terminal_id="term-1", event_id="event-1", observation="not_presented"
        )
        assert reconciled.state == "known_failed"

        retried = bridge.present(request, peer_uid=os.getuid())
        assert retried.state == "presented"
        assert retried.attempt == 2
        assert len(presenter.calls) == 2


def test_socket_path_is_owner_only_and_expired_registration_can_be_replaced(
    tmp_path: Path,
) -> None:
    current = datetime(2026, 7, 31, 12, tzinfo=UTC)
    authority = build_test_fresh_v2_authority(tmp_path / "authority", device_id="device-a")
    socket_path = tmp_path / "terminal.sock"
    socket_path.touch(mode=0o666)
    with TerminalBridge(
        tmp_path / "terminal.sqlite",
        presenter=FakePresenter(),
        clock=lambda: current,
        registration_verifier=_registration_verifier(authority),
        socket_path=socket_path,
    ) as bridge:
        first = sign_terminal_registration(
            replace(
                _unsigned_registration(current),
                expires_at=(current + timedelta(seconds=1)).isoformat(),
            ),
            signer=authority.signer,
            key_id=authority.key_id,
            device_id="device-a",
        )
        bridge.register(first, peer_uid=os.getuid())
        assert socket_path.stat().st_mode & 0o777 == 0o600

        current += timedelta(seconds=2)
        replacement = sign_terminal_registration(
            replace(_unsigned_registration(current), nonce="nonce-b"),
            signer=authority.signer,
            key_id=authority.key_id,
            device_id="device-a",
        )
        assert bridge.register(replacement, peer_uid=os.getuid()) == replacement


def test_close_context_manager_and_finalizer_release_database(tmp_path: Path) -> None:
    database = tmp_path / "terminal.sqlite"
    bridge = TerminalBridge(database, presenter=FakePresenter())
    connection = bridge._connection
    bridge.close()
    bridge.close()
    with pytest.raises(sqlite3.ProgrammingError):
        connection.execute("SELECT 1")

    finalized = TerminalBridge(database, presenter=FakePresenter())
    finalized_connection = finalized._connection
    del finalized
    gc.collect()
    with pytest.raises(sqlite3.ProgrammingError):
        finalized_connection.execute("SELECT 1")
