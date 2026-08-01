from __future__ import annotations

import os
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from memo.errors import OperationalError
from memo.terminal_bridge import (
    PresenterOutcome,
    TerminalBridge,
    TerminalPresentRequest,
    TerminalRegistration,
)


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


def _registration(now: datetime, *, terminal_id: str = "term-1") -> TerminalRegistration:
    return TerminalRegistration(
        id=terminal_id,
        principal_id="principal-a",
        session_id="session-a",
        uid=os.getuid(),
        capabilities=("inject", "notify"),
        issued_at=now.isoformat(),
        expires_at=(now + timedelta(minutes=5)).isoformat(),
        nonce="nonce-a",
        signature="signature-a",
    )


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


def test_bridge_rejects_foreign_uid_and_sanitizes_payload(tmp_path) -> None:
    now = datetime(2026, 7, 31, 12, tzinfo=UTC)
    presenter = FakePresenter()
    bridge = TerminalBridge(tmp_path / "terminal.sqlite", presenter=presenter, clock=lambda: now)
    bridge.register(_registration(now), peer_uid=os.getuid())

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


def test_duplicate_reserves_before_side_effect(tmp_path) -> None:
    now = datetime(2026, 7, 31, 12, tzinfo=UTC)
    presenter = FakePresenter()
    bridge = TerminalBridge(tmp_path / "terminal.sqlite", presenter=presenter, clock=lambda: now)
    bridge.register(_registration(now), peer_uid=os.getuid())
    request = _request(now)

    first = bridge.present(request, peer_uid=os.getuid())
    duplicate = bridge.present(request, peer_uid=os.getuid())

    assert presenter.calls == ["handoff"]
    assert duplicate == first


def test_bridge_enforces_registration_binding_capability_and_payload_limit(tmp_path) -> None:
    now = datetime(2026, 7, 31, 12, tzinfo=UTC)
    presenter = FakePresenter()
    bridge = TerminalBridge(tmp_path / "terminal.sqlite", presenter=presenter, clock=lambda: now)
    registration = replace(_registration(now), capabilities=("notify",))
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


def test_uncertain_requires_explicit_negative_reconciliation_before_retry(tmp_path) -> None:
    current = datetime(2026, 7, 31, 12, tzinfo=UTC)
    presenter = FakePresenter()
    presenter.outcomes = [RuntimeError("ambiguous presenter failure")]
    bridge = TerminalBridge(
        tmp_path / "terminal.sqlite", presenter=presenter, clock=lambda: current
    )
    bridge.register(_registration(current), peer_uid=os.getuid())
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


def test_socket_path_is_owner_only_and_expired_registration_can_be_replaced(tmp_path) -> None:
    current = datetime(2026, 7, 31, 12, tzinfo=UTC)
    socket_path = tmp_path / "terminal.sock"
    socket_path.touch(mode=0o666)
    bridge = TerminalBridge(
        tmp_path / "terminal.sqlite",
        presenter=FakePresenter(),
        clock=lambda: current,
        socket_path=socket_path,
    )
    bridge.register(
        replace(_registration(current), expires_at=(current + timedelta(seconds=1)).isoformat()),
        peer_uid=os.getuid(),
    )
    assert socket_path.stat().st_mode & 0o777 == 0o600

    current += timedelta(seconds=2)
    replacement = replace(
        _registration(current), nonce="nonce-b", signature="signature-b"
    )
    assert bridge.register(replacement, peer_uid=os.getuid()) == replacement
