"""Controlled, user-owned terminal presentation with durable receipts."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol

from memo.errors import OperationalError, OperationalErrorCode

TerminalMode = Literal["notify", "inject"]
PresentationState = Literal[
    "reserved", "presented", "known_failed", "uncertain"
]
ReconciliationObservation = Literal["presented", "not_presented"]
_CAPABILITIES = frozenset({"notify", "inject"})
_MAX_PAYLOAD_BYTES = 16_384


def _invalid(message: str) -> OperationalError:
    return OperationalError(
        OperationalErrorCode.INVALID_EVENT,
        message,
        retryable=False,
    )


def _canonical_time(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("terminal timestamps must include a timezone")
    return value.astimezone(UTC).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def _parse_time(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise _invalid("terminal timestamp is invalid") from exc
    if parsed.tzinfo is None:
        raise _invalid("terminal timestamp must include a timezone")
    return parsed.astimezone(UTC)


def _digest(value: object) -> str:
    wire = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(wire).hexdigest()


def sanitize_terminal_payload(payload: str) -> str:
    """Strip terminal control sequences while preserving tab and newline."""

    output: list[str] = []
    index = 0
    while index < len(payload):
        character = payload[index]
        codepoint = ord(character)
        if character == "\x1b":
            index += 1
            if index >= len(payload):
                break
            introducer = payload[index]
            if introducer == "]":
                index += 1
                while index < len(payload):
                    if payload[index] == "\x07":
                        index += 1
                        break
                    if (
                        payload[index] == "\x1b"
                        and index + 1 < len(payload)
                        and payload[index + 1] == "\\"
                    ):
                        index += 2
                        break
                    index += 1
                continue
            if introducer == "[":
                index += 1
                while index < len(payload):
                    value = ord(payload[index])
                    index += 1
                    if 0x40 <= value <= 0x7E:
                        break
                continue
            index += 1
            continue
        if (codepoint < 0x20 and character not in {"\n", "\t"}) or (
            0x7F <= codepoint <= 0x9F
        ):
            index += 1
            continue
        output.append(character)
        index += 1
    return "".join(output)


@dataclass(frozen=True)
class TerminalRegistration:
    id: str
    principal_id: str
    session_id: str
    uid: int
    capabilities: tuple[TerminalMode, ...]
    issued_at: str
    expires_at: str
    nonce: str
    signature: str


@dataclass(frozen=True)
class TerminalPresentRequest:
    event_id: str
    message_id: str
    delivery_id: str
    terminal_id: str
    mode: TerminalMode
    payload: str
    sanitized_payload_hash: str
    deadline: str
    idempotency_key: str
    principal_id: str
    session_id: str


@dataclass(frozen=True)
class PresenterOutcome:
    state: Literal["presented", "known_failed"]
    error_code: str = ""


@dataclass(frozen=True)
class PresenterReceipt:
    event_id: str
    message_id: str
    delivery_id: str
    terminal_id: str
    state: PresentationState
    attempt: int
    presenter_timestamp: str
    error_code: str
    receipt_hash: str


class TerminalPresenter(Protocol):
    def present(
        self,
        *,
        terminal_id: str,
        mode: str,
        payload: str,
    ) -> PresenterOutcome: ...


RegistrationVerifier = Callable[[TerminalRegistration], bool]


class TerminalBridge:
    """Authorize terminal presentation and reserve effects before execution."""

    def __init__(
        self,
        database: Path,
        *,
        presenter: TerminalPresenter,
        clock: Callable[[], datetime] | None = None,
        registration_verifier: RegistrationVerifier | None = None,
        socket_path: Path | None = None,
    ) -> None:
        self.database = Path(database)
        self.database.parent.mkdir(parents=True, exist_ok=True)
        self.presenter = presenter
        self.clock = clock or (lambda: datetime.now(UTC))
        self.registration_verifier = registration_verifier or (
            lambda registration: bool(registration.signature)
        )
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(
            str(self.database), timeout=10.0, check_same_thread=False
        )
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA foreign_keys=ON")
        self._create_schema()
        self._recover_abandoned_reservations()
        if socket_path is not None and socket_path.exists():
            os.chmod(socket_path, 0o600)

    def _create_schema(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS terminal_registration (
                terminal_id TEXT PRIMARY KEY,
                principal_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                uid INTEGER NOT NULL,
                capabilities TEXT NOT NULL,
                issued_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                nonce TEXT NOT NULL,
                signature TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS presenter_receipt (
                terminal_id TEXT NOT NULL,
                event_id TEXT NOT NULL,
                message_id TEXT NOT NULL,
                delivery_id TEXT NOT NULL,
                mode TEXT NOT NULL,
                payload_hash TEXT NOT NULL,
                deadline TEXT NOT NULL,
                idempotency_key TEXT NOT NULL,
                principal_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                request_hash TEXT NOT NULL,
                state TEXT NOT NULL,
                attempt INTEGER NOT NULL,
                presenter_timestamp TEXT NOT NULL,
                error_code TEXT NOT NULL,
                receipt_hash TEXT NOT NULL,
                PRIMARY KEY (terminal_id, event_id)
            );
            """
        )
        self._connection.commit()

    def _recover_abandoned_reservations(self) -> None:
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM presenter_receipt WHERE state = 'reserved'"
            ).fetchall()
            for row in rows:
                receipt = self._receipt(row)
                uncertain = self._with_receipt_hash(
                    replace(
                        receipt,
                        state="uncertain",
                        presenter_timestamp=_canonical_time(self.clock()),
                        error_code="presenter_outcome_unknown",
                    )
                )
                self._store_outcome(uncertain)
            self._connection.commit()

    def register(
        self,
        registration: TerminalRegistration,
        *,
        peer_uid: int,
    ) -> TerminalRegistration:
        now = self.clock().astimezone(UTC)
        if peer_uid != registration.uid:
            raise _invalid("terminal registration UID does not match socket peer UID")
        if not all(
            value.strip()
            for value in (
                registration.id,
                registration.principal_id,
                registration.session_id,
                registration.nonce,
                registration.signature,
            )
        ):
            raise _invalid("terminal registration fields must be non-empty")
        capabilities = tuple(sorted(set(registration.capabilities)))
        if not capabilities or not set(capabilities) <= _CAPABILITIES:
            raise _invalid("terminal registration capabilities are invalid")
        issued_at = _parse_time(registration.issued_at)
        expires_at = _parse_time(registration.expires_at)
        if expires_at <= issued_at or expires_at <= now:
            raise OperationalError(
                OperationalErrorCode.EXPIRED,
                "terminal registration is expired",
                retryable=False,
            )
        normalized = replace(registration, capabilities=capabilities)
        if not self.registration_verifier(normalized):
            raise OperationalError(
                OperationalErrorCode.SIGNATURE_INVALID,
                "terminal registration signature is invalid",
                retryable=False,
            )

        with self._lock:
            existing_row = self._connection.execute(
                "SELECT * FROM terminal_registration WHERE terminal_id = ?",
                (normalized.id,),
            ).fetchone()
            if existing_row is not None:
                existing = self._registration(existing_row)
                if _parse_time(existing.expires_at) > now:
                    if existing == normalized:
                        return existing
                    raise _invalid("active terminal registration cannot be replaced")
            self._connection.execute(
                """
                INSERT INTO terminal_registration (
                    terminal_id, principal_id, session_id, uid, capabilities,
                    issued_at, expires_at, nonce, signature
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(terminal_id) DO UPDATE SET
                    principal_id=excluded.principal_id,
                    session_id=excluded.session_id,
                    uid=excluded.uid,
                    capabilities=excluded.capabilities,
                    issued_at=excluded.issued_at,
                    expires_at=excluded.expires_at,
                    nonce=excluded.nonce,
                    signature=excluded.signature
                """,
                (
                    normalized.id,
                    normalized.principal_id,
                    normalized.session_id,
                    normalized.uid,
                    json.dumps(normalized.capabilities),
                    _canonical_time(issued_at),
                    _canonical_time(expires_at),
                    normalized.nonce,
                    normalized.signature,
                ),
            )
            self._connection.commit()
        return normalized

    def present(
        self,
        request: TerminalPresentRequest,
        *,
        peer_uid: int,
    ) -> PresenterReceipt:
        now = self.clock().astimezone(UTC)
        registration = self._authorized_registration(request, peer_uid=peer_uid, now=now)
        sanitized = sanitize_terminal_payload(request.payload)
        if len(request.payload.encode("utf-8")) > _MAX_PAYLOAD_BYTES:
            raise _invalid("terminal payload exceeds 16,384 UTF-8 bytes")
        if len(sanitized.encode("utf-8")) > _MAX_PAYLOAD_BYTES:
            raise _invalid("sanitized terminal payload exceeds 16,384 UTF-8 bytes")
        payload_hash = hashlib.sha256(sanitized.encode("utf-8")).hexdigest()
        if request.sanitized_payload_hash and request.sanitized_payload_hash != payload_hash:
            raise _invalid("sanitized terminal payload hash does not match request")
        deadline = _parse_time(request.deadline)
        if deadline <= now:
            raise OperationalError(
                OperationalErrorCode.EXPIRED,
                "terminal presentation deadline has passed",
                retryable=False,
            )
        request_hash = self._request_hash(request, payload_hash=payload_hash)

        with self._lock:
            existing_row = self._connection.execute(
                """
                SELECT * FROM presenter_receipt
                WHERE terminal_id = ? AND event_id = ?
                """,
                (request.terminal_id, request.event_id),
            ).fetchone()
            attempt = 1
            if existing_row is not None:
                if str(existing_row["request_hash"]) != request_hash:
                    raise OperationalError(
                        OperationalErrorCode.IDEMPOTENCY_CONFLICT,
                        "terminal event identifies a different presentation request",
                        retryable=False,
                    )
                existing = self._receipt(existing_row)
                if existing.state != "known_failed":
                    return existing
                attempt = existing.attempt + 1
            reserved = self._with_receipt_hash(
                PresenterReceipt(
                    event_id=request.event_id,
                    message_id=request.message_id,
                    delivery_id=request.delivery_id,
                    terminal_id=request.terminal_id,
                    state="reserved",
                    attempt=attempt,
                    presenter_timestamp=_canonical_time(now),
                    error_code="",
                    receipt_hash="",
                )
            )
            self._connection.execute("BEGIN IMMEDIATE")
            self._connection.execute(
                """
                INSERT INTO presenter_receipt (
                    terminal_id, event_id, message_id, delivery_id, mode,
                    payload_hash, deadline, idempotency_key, principal_id,
                    session_id, request_hash, state, attempt,
                    presenter_timestamp, error_code, receipt_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(terminal_id, event_id) DO UPDATE SET
                    state=excluded.state,
                    attempt=excluded.attempt,
                    presenter_timestamp=excluded.presenter_timestamp,
                    error_code=excluded.error_code,
                    receipt_hash=excluded.receipt_hash
                """,
                (
                    request.terminal_id,
                    request.event_id,
                    request.message_id,
                    request.delivery_id,
                    request.mode,
                    payload_hash,
                    _canonical_time(deadline),
                    request.idempotency_key,
                    request.principal_id,
                    request.session_id,
                    request_hash,
                    reserved.state,
                    reserved.attempt,
                    reserved.presenter_timestamp,
                    reserved.error_code,
                    reserved.receipt_hash,
                ),
            )
            self._connection.commit()

        try:
            outcome = self.presenter.present(
                terminal_id=registration.id,
                mode=request.mode,
                payload=sanitized,
            )
            if outcome.state not in {"presented", "known_failed"}:
                raise ValueError("presenter returned an invalid outcome")
            completed = replace(
                reserved,
                state=outcome.state,
                presenter_timestamp=_canonical_time(self.clock()),
                error_code=outcome.error_code,
            )
        except Exception as exc:
            completed = replace(
                reserved,
                state="uncertain",
                presenter_timestamp=_canonical_time(self.clock()),
                error_code=type(exc).__name__,
            )
        completed = self._with_receipt_hash(completed)
        with self._lock:
            self._store_outcome(completed)
            self._connection.commit()
        return completed

    def reconcile(
        self,
        *,
        terminal_id: str,
        event_id: str,
        observation: ReconciliationObservation,
    ) -> PresenterReceipt:
        if observation not in {"presented", "not_presented"}:
            raise _invalid("terminal reconciliation observation is invalid")
        with self._lock:
            row = self._connection.execute(
                """
                SELECT * FROM presenter_receipt
                WHERE terminal_id = ? AND event_id = ?
                """,
                (terminal_id, event_id),
            ).fetchone()
            if row is None:
                raise OperationalError(
                    OperationalErrorCode.NOT_FOUND,
                    "terminal presenter receipt does not exist",
                    retryable=False,
                )
            current = self._receipt(row)
            if current.state == "presented":
                if observation == "presented":
                    return current
                raise _invalid("presented terminal receipt cannot regress")
            if current.state == "known_failed" and observation == "not_presented":
                return current
            if current.state not in {"reserved", "uncertain"}:
                raise _invalid("terminal receipt cannot be reconciled")
            reconciled = self._with_receipt_hash(
                replace(
                    current,
                    state="presented" if observation == "presented" else "known_failed",
                    presenter_timestamp=_canonical_time(self.clock()),
                    error_code="" if observation == "presented" else "reconciled_not_presented",
                )
            )
            self._store_outcome(reconciled)
            self._connection.commit()
            return reconciled

    def _authorized_registration(
        self,
        request: TerminalPresentRequest,
        *,
        peer_uid: int,
        now: datetime,
    ) -> TerminalRegistration:
        if not all(
            value.strip()
            for value in (
                request.event_id,
                request.message_id,
                request.delivery_id,
                request.terminal_id,
                request.idempotency_key,
                request.principal_id,
                request.session_id,
            )
        ):
            raise _invalid("terminal presentation identifiers must be non-empty")
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM terminal_registration WHERE terminal_id = ?",
                (request.terminal_id,),
            ).fetchone()
        if row is None:
            raise OperationalError(
                OperationalErrorCode.NOT_FOUND,
                "terminal registration does not exist",
                retryable=False,
            )
        registration = self._registration(row)
        if peer_uid != registration.uid:
            raise _invalid("terminal socket peer UID is not authorized")
        if _parse_time(registration.expires_at) <= now:
            raise OperationalError(
                OperationalErrorCode.EXPIRED,
                "terminal registration is expired",
                retryable=False,
            )
        if (
            request.principal_id != registration.principal_id
            or request.session_id != registration.session_id
        ):
            raise _invalid("terminal principal or session binding does not match")
        if request.mode not in registration.capabilities:
            raise _invalid(f"terminal is not authorized for {request.mode!r}")
        return registration

    @staticmethod
    def _request_hash(request: TerminalPresentRequest, *, payload_hash: str) -> str:
        return _digest(
            {
                "event_id": request.event_id,
                "message_id": request.message_id,
                "delivery_id": request.delivery_id,
                "terminal_id": request.terminal_id,
                "mode": request.mode,
                "payload_hash": payload_hash,
                "deadline": _canonical_time(_parse_time(request.deadline)),
                "idempotency_key": request.idempotency_key,
                "principal_id": request.principal_id,
                "session_id": request.session_id,
            }
        )

    @staticmethod
    def _with_receipt_hash(receipt: PresenterReceipt) -> PresenterReceipt:
        values = asdict(receipt)
        values.pop("receipt_hash")
        return replace(receipt, receipt_hash=_digest(values))

    def _store_outcome(self, receipt: PresenterReceipt) -> None:
        self._connection.execute(
            """
            UPDATE presenter_receipt SET
                state = ?, attempt = ?, presenter_timestamp = ?,
                error_code = ?, receipt_hash = ?
            WHERE terminal_id = ? AND event_id = ?
            """,
            (
                receipt.state,
                receipt.attempt,
                receipt.presenter_timestamp,
                receipt.error_code,
                receipt.receipt_hash,
                receipt.terminal_id,
                receipt.event_id,
            ),
        )

    @staticmethod
    def _registration(row: sqlite3.Row) -> TerminalRegistration:
        return TerminalRegistration(
            id=str(row["terminal_id"]),
            principal_id=str(row["principal_id"]),
            session_id=str(row["session_id"]),
            uid=int(row["uid"]),
            capabilities=tuple(json.loads(str(row["capabilities"]))),
            issued_at=str(row["issued_at"]),
            expires_at=str(row["expires_at"]),
            nonce=str(row["nonce"]),
            signature=str(row["signature"]),
        )

    @staticmethod
    def _receipt(row: sqlite3.Row) -> PresenterReceipt:
        return PresenterReceipt(
            event_id=str(row["event_id"]),
            message_id=str(row["message_id"]),
            delivery_id=str(row["delivery_id"]),
            terminal_id=str(row["terminal_id"]),
            state=str(row["state"]),  # type: ignore[arg-type]
            attempt=int(row["attempt"]),
            presenter_timestamp=str(row["presenter_timestamp"]),
            error_code=str(row["error_code"]),
            receipt_hash=str(row["receipt_hash"]),
        )
