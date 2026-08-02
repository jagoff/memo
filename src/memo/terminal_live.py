"""Registered local terminal sessions for immediate agent-to-agent delivery."""

from __future__ import annotations

import builtins
import os
import re
import secrets
import sqlite3
import stat
import subprocess
import unicodedata
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from memo.errors import TerminalValidationError

if TYPE_CHECKING:
    from memo.config import Config

_AGENTS = frozenset({"blackbox", "claude", "codex", "devin", "gemini", "opencode"})
_TERMINAL_APPS = {
    "": "",
    "terminal": "Terminal",
    "iterm": "iTerm",
    "iterm2": "iTerm2",
    "ghostty": "Ghostty",
}


@dataclass(frozen=True)
class ProcessSnapshot:
    """Identity and foreground state of one local process."""

    pid: int
    uid: int
    tty: Path
    started_at: str
    pgid: int
    foreground_pgid: int
    command: str


@dataclass(frozen=True)
class TerminalRegistration:
    """One explicitly registered agent terminal."""

    id: str
    tty: str
    pid: int
    uid: int
    agent: str
    terminal_app: str
    project: str
    process_started_at: str
    created_at: str
    last_seen_at: str


@dataclass(frozen=True)
class TerminalReceipt:
    """Outcome of one attempted terminal delivery."""

    receipt_id: str
    message_id: str
    target_id: str
    sender_id: str
    kind: str
    status: str
    transport: str
    error: str
    created_at: str
    delivered_at: str


ProcessProbe = Callable[[int], ProcessSnapshot | None]
Presenter = Callable[..., str]

_MAX_MESSAGE_BYTES = 16 * 1024
_MESSAGE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
_REGISTRATION_ID_RE = re.compile(r"term-[0-9a-f]{16}\Z")


def _default_presenter(tty: Path, payload: bytes, *, terminal_app: str) -> str:
    from memo.terminal_presenter import deliver_input

    return deliver_input(tty, payload, terminal_app=terminal_app)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _canonical_tty(value: str | Path, *, uid: int | None = None) -> Path:
    raw = Path(value).expanduser()
    if not raw.is_absolute():
        raw = Path("/dev") / raw
    try:
        path = raw.resolve(strict=True)
        info = path.stat()
    except OSError as exc:
        raise TerminalValidationError("terminal target is unavailable") from exc
    if path.parent != Path("/dev") or not stat.S_ISCHR(info.st_mode):
        raise TerminalValidationError("terminal target is not a local TTY")
    expected_uid = os.getuid() if uid is None else uid
    if info.st_uid != expected_uid:
        raise TerminalValidationError("terminal target belongs to another user")
    return path


def _process_snapshot(pid: int) -> ProcessSnapshot | None:
    if pid <= 1:
        return None
    try:
        proc = subprocess.run(
            [
                "ps",
                "-p",
                str(pid),
                "-o",
                "uid=",
                "-o",
                "tty=",
                "-o",
                "lstart=",
                "-o",
                "pgid=",
                "-o",
                "tpgid=",
                "-o",
                "command=",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    parts = proc.stdout.strip().split(maxsplit=9)
    if len(parts) != 10 or parts[1] in {"?", "??", "-"}:
        return None
    try:
        uid = int(parts[0])
        pgid = int(parts[7])
        foreground_pgid = int(parts[8])
    except ValueError:
        return None
    tty = Path(parts[1])
    if not tty.is_absolute():
        tty = Path("/dev") / tty
    return ProcessSnapshot(
        pid=pid,
        uid=uid,
        tty=tty,
        started_at=" ".join(parts[2:7]),
        pgid=pgid,
        foreground_pgid=foreground_pgid,
        command=parts[9],
    )


def _command_matches_agent(command: str, agent: str) -> bool:
    lowered = command.lower()
    return any(part == agent or part.endswith(f"/{agent}") for part in lowered.split()) or (
        f"/{agent} " in lowered
    )


def _consume_csi(message: str, index: int) -> int:
    while index < len(message) and not ("@" <= message[index] <= "~"):
        index += 1
    return index + (index < len(message))


def _consume_osc(message: str, index: int) -> int:
    while index < len(message):
        if message[index] == "\x07":
            return index + 1
        if message[index : index + 2] == "\x1b\\":
            return index + 2
        index += 1
    return index


def _consume_escape(message: str, index: int) -> int:
    index += 1
    if index >= len(message):
        return index
    if message[index] == "[":
        return _consume_csi(message, index + 1)
    if message[index] == "]":
        return _consume_osc(message, index + 1)
    return index + 1


def _strip_terminal_controls(message: str) -> str:
    if len(message.encode("utf-8")) > _MAX_MESSAGE_BYTES:
        raise TerminalValidationError("terminal message exceeds 16 KiB")
    result: list[str] = []
    i = 0
    while i < len(message):
        char = message[i]
        if char == "\x1b":
            i = _consume_escape(message, i)
            continue
        if char in {"\n", "\t"} or unicodedata.category(char) != "Cc":
            result.append(char)
        i += 1
    sanitized = "".join(result)
    if not sanitized.strip():
        raise TerminalValidationError("terminal message is empty after sanitization")
    return sanitized


class TerminalBridge:
    """Persist and revalidate local terminal registrations."""

    def __init__(
        self,
        cfg: Config,
        *,
        process_probe: ProcessProbe | None = None,
        presenter: Presenter | None = None,
    ) -> None:
        self._path = cfg.state_dir / "terminal_live.db"
        self._probe = process_probe or _process_snapshot
        self._present = presenter or _default_presenter
        self._init_db()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self._path, timeout=5)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS terminal_registrations (
                    id TEXT PRIMARY KEY,
                    tty TEXT NOT NULL UNIQUE,
                    pid INTEGER NOT NULL,
                    uid INTEGER NOT NULL,
                    agent TEXT NOT NULL,
                    terminal_app TEXT NOT NULL,
                    project TEXT NOT NULL,
                    process_started_at TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    last_seen_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS terminal_receipts (
                    receipt_id TEXT PRIMARY KEY,
                    message_id TEXT NOT NULL UNIQUE,
                    target_id TEXT NOT NULL,
                    sender_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    status TEXT NOT NULL,
                    transport TEXT NOT NULL,
                    error TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    delivered_at TEXT NOT NULL
                )
                """
            )
        self._path.chmod(0o600)

    @staticmethod
    def _from_row(row: sqlite3.Row) -> TerminalRegistration:
        return TerminalRegistration(
            id=str(row["id"]),
            tty=str(row["tty"]),
            pid=int(row["pid"]),
            uid=int(row["uid"]),
            agent=str(row["agent"]),
            terminal_app=str(row["terminal_app"]),
            project=str(row["project"]),
            process_started_at=str(row["process_started_at"]),
            created_at=str(row["created_at"]),
            last_seen_at=str(row["last_seen_at"]),
        )

    def register(
        self,
        *,
        agent: str,
        tty: str | Path,
        pid: int,
        terminal_app: str = "",
        project: str | Path = "",
    ) -> TerminalRegistration:
        normalized_agent = agent.strip().lower()
        if normalized_agent not in _AGENTS:
            raise TerminalValidationError("unsupported agent registration")
        app_key = terminal_app.strip().lower()
        if app_key not in _TERMINAL_APPS:
            raise TerminalValidationError("unsupported terminal application")
        snapshot = self._probe(pid)
        if snapshot is None or snapshot.uid != os.getuid():
            raise TerminalValidationError("agent process is unavailable")
        canonical_tty = _canonical_tty(tty, uid=snapshot.uid)
        try:
            process_tty = _canonical_tty(snapshot.tty, uid=snapshot.uid)
        except TerminalValidationError as exc:
            raise TerminalValidationError("agent process has no valid TTY") from exc
        if process_tty != canonical_tty:
            raise TerminalValidationError("agent process is attached to a different TTY")

        now = _now()
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT id, created_at FROM terminal_registrations WHERE tty = ?",
                (str(canonical_tty),),
            ).fetchone()
            registration_id = str(existing["id"]) if existing else f"term-{secrets.token_hex(8)}"
            created_at = str(existing["created_at"]) if existing else now
            conn.execute(
                """
                INSERT INTO terminal_registrations (
                    id, tty, pid, uid, agent, terminal_app, project,
                    process_started_at, created_at, last_seen_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(tty) DO UPDATE SET
                    pid = excluded.pid,
                    uid = excluded.uid,
                    agent = excluded.agent,
                    terminal_app = excluded.terminal_app,
                    project = excluded.project,
                    process_started_at = excluded.process_started_at,
                    last_seen_at = excluded.last_seen_at
                """,
                (
                    registration_id,
                    str(canonical_tty),
                    pid,
                    snapshot.uid,
                    normalized_agent,
                    _TERMINAL_APPS[app_key],
                    str(project),
                    snapshot.started_at,
                    created_at,
                    now,
                ),
            )
            row = conn.execute(
                "SELECT * FROM terminal_registrations WHERE id = ?",
                (registration_id,),
            ).fetchone()
        if row is None:  # pragma: no cover - guarded by the transaction above
            raise TerminalValidationError("terminal registration was not persisted")
        return self._from_row(row)

    def _is_live(self, registration: TerminalRegistration) -> bool:
        snapshot = self._probe(registration.pid)
        if snapshot is None or snapshot.uid != registration.uid:
            return False
        try:
            process_tty = _canonical_tty(snapshot.tty, uid=registration.uid)
        except TerminalValidationError:
            return False
        return (
            str(process_tty) == registration.tty
            and snapshot.started_at == registration.process_started_at
            and _command_matches_agent(snapshot.command, registration.agent)
        )

    def list(self) -> list[TerminalRegistration]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM terminal_registrations ORDER BY created_at, id"
            ).fetchall()
            registrations = [self._from_row(row) for row in rows]
            live = [item for item in registrations if self._is_live(item)]
            stale_ids = [item.id for item in registrations if item not in live]
            if stale_ids:
                conn.executemany(
                    "DELETE FROM terminal_registrations WHERE id = ?",
                    [(item,) for item in stale_ids],
                )
        return live

    def registration_id(self, registration_id: str) -> str:
        """Return an exact live registration id, or an empty string."""
        return registration_id if any(item.id == registration_id for item in self.list()) else ""

    def registration_id_for_tty(self, tty: str | Path) -> str:
        """Resolve a live registration by exact canonical TTY."""
        try:
            canonical = str(_canonical_tty(tty))
        except TerminalValidationError:
            return ""
        return next((item.id for item in self.list() if item.tty == canonical), "")

    @staticmethod
    def _receipt_from_row(row: sqlite3.Row) -> TerminalReceipt:
        return TerminalReceipt(
            receipt_id=str(row["receipt_id"]),
            message_id=str(row["message_id"]),
            target_id=str(row["target_id"]),
            sender_id=str(row["sender_id"]),
            kind=str(row["kind"]),
            status=str(row["status"]),
            transport=str(row["transport"]),
            error=str(row["error"]),
            created_at=str(row["created_at"]),
            delivered_at=str(row["delivered_at"]),
        )

    def _existing_receipt(self, message_id: str) -> TerminalReceipt | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM terminal_receipts WHERE message_id = ?",
                (message_id,),
            ).fetchone()
        return self._receipt_from_row(row) if row else None

    def _live_target(self, target_id: str) -> TerminalRegistration:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM terminal_registrations WHERE id = ?",
                (target_id,),
            ).fetchone()
        if row is None:
            raise TerminalValidationError("registered terminal target was not found")
        registration = self._from_row(row)
        snapshot = self._probe(registration.pid)
        if (
            snapshot is None
            or snapshot.uid != registration.uid
            or snapshot.started_at != registration.process_started_at
            or not _command_matches_agent(snapshot.command, registration.agent)
        ):
            raise TerminalValidationError("registered terminal target is stale")
        try:
            process_tty = _canonical_tty(snapshot.tty, uid=registration.uid)
        except TerminalValidationError as exc:
            raise TerminalValidationError("registered terminal target is stale") from exc
        if str(process_tty) != registration.tty:
            raise TerminalValidationError("registered terminal target changed TTY")
        if snapshot.pgid != snapshot.foreground_pgid:
            raise TerminalValidationError("registered agent is not the foreground terminal process")
        return registration

    def _deliver(
        self,
        target_id: str,
        payload: bytes,
        *,
        sender: str | None,
        message_id: str | None,
        kind: str,
    ) -> TerminalReceipt:
        resolved_message_id = (message_id or f"msg-{secrets.token_hex(12)}").strip()
        if not _MESSAGE_ID_RE.fullmatch(resolved_message_id):
            raise TerminalValidationError("terminal message id is malformed or too long")
        sender_id = (sender or "").strip()
        if sender_id and not _REGISTRATION_ID_RE.fullmatch(sender_id):
            raise TerminalValidationError("terminal sender id is malformed")
        existing = self._existing_receipt(resolved_message_id)
        if existing is not None:
            return replace(existing, status="duplicate")

        registration = self._live_target(target_id)
        now = _now()
        receipt_id = f"rcpt-{secrets.token_hex(8)}"
        with self._connect() as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO terminal_receipts (
                        receipt_id, message_id, target_id, sender_id, kind,
                        status, transport, error, created_at, delivered_at
                    ) VALUES (?, ?, ?, ?, ?, 'pending', '', '', ?, '')
                    """,
                    (
                        receipt_id,
                        resolved_message_id,
                        target_id,
                        sender_id,
                        kind,
                        now,
                    ),
                )
            except sqlite3.IntegrityError:
                duplicate = self._existing_receipt(resolved_message_id)
                if duplicate is None:  # pragma: no cover - defensive race guard
                    raise TerminalValidationError("terminal receipt collision") from None
                return replace(duplicate, status="duplicate")

        try:
            transport = self._present(
                Path(registration.tty),
                payload,
                terminal_app=registration.terminal_app,
            )
        except (OSError, RuntimeError) as exc:
            with self._connect() as conn:
                conn.execute(
                    "UPDATE terminal_receipts SET status = 'failed', error = ? "
                    "WHERE receipt_id = ?",
                    (type(exc).__name__, receipt_id),
                )
            raise TerminalValidationError("terminal delivery failed") from exc

        delivered_at = _now()
        with self._connect() as conn:
            conn.execute(
                "UPDATE terminal_receipts SET status = 'delivered', transport = ?, "
                "delivered_at = ? WHERE receipt_id = ?",
                (transport, delivered_at, receipt_id),
            )
            row = conn.execute(
                "SELECT * FROM terminal_receipts WHERE receipt_id = ?",
                (receipt_id,),
            ).fetchone()
        if row is None:  # pragma: no cover - guarded by insert above
            raise TerminalValidationError("terminal receipt was not persisted")
        return self._receipt_from_row(row)

    def send(
        self,
        target_id: str,
        message: str,
        *,
        sender: str | None = None,
        submit: bool = True,
        message_id: str | None = None,
    ) -> TerminalReceipt:
        sanitized = _strip_terminal_controls(message)
        payload = sanitized.encode("utf-8") + (b"\r" if submit else b"")
        return self._deliver(
            target_id,
            payload,
            sender=sender,
            message_id=message_id,
            kind="message",
        )

    def enter(
        self,
        target_id: str,
        *,
        sender: str | None = None,
        message_id: str | None = None,
    ) -> TerminalReceipt:
        return self._deliver(
            target_id,
            b"\r",
            sender=sender,
            message_id=message_id,
            kind="enter",
        )

    def history(self, *, limit: int = 50) -> builtins.list[TerminalReceipt]:
        bounded = max(1, min(limit, 500))
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM terminal_receipts ORDER BY created_at DESC, receipt_id DESC LIMIT ?",
                (bounded,),
            ).fetchall()
        return [self._receipt_from_row(row) for row in rows]
