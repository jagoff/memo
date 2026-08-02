"""Registered local terminal sessions for immediate agent-to-agent delivery."""

from __future__ import annotations

import builtins
import errno
import fcntl
import hashlib
import hmac
import os
import re
import secrets
import sqlite3
import stat
import subprocess
import sys
import time
import unicodedata
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from memo.errors import TerminalDeliveryError, TerminalValidationError

if TYPE_CHECKING:
    from memo.config import Config

_AGENTS = frozenset({"blackbox", "claude", "codex", "devin", "gemini", "opencode"})
_TERMINAL_APPS = {
    "": "",
    "apple_terminal": "Terminal",
    "terminal": "Terminal",
    "iterm": "iTerm",
    "iterm.app": "iTerm2",
    "iterm2": "iTerm2",
    "ghostty": "Ghostty",
    "tmux": "tmux",
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


class _ProbeUnknown:
    """A transient probe failure that is not proof the process exited."""


_PROBE_UNKNOWN = _ProbeUnknown()
ProcessProbe = Callable[[int], ProcessSnapshot | _ProbeUnknown | None]
Presenter = Callable[..., str]
TransportProbe = Callable[[Path, str], bool]
ProcessBindingProbe = Callable[[], bool]
ReceiptOwnerProbe = Callable[[int, str], bool | None]
Failpoint = Callable[[str], None]

_MAX_MESSAGE_BYTES = 16 * 1024
_MAX_FINAL_RECEIPTS = 2_000
_MAX_RECEIPT_KEYS = 10_000
_LOCK_BUCKETS = 64
_LOCK_TIMEOUT_SECONDS = 10.0
_PENDING_LEASE = timedelta(seconds=30)
_PROBE_GRACE = timedelta(seconds=30)
_MESSAGE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
_REGISTRATION_ID_RE = re.compile(r"term-[0-9a-f]{16}\Z")
_DELIVERY_TRANSPORTS = frozenset({"ghostty-applescript", "iterm-applescript", "tiocsti", "tmux"})


def _default_presenter(tty: Path, payload: bytes, *, terminal_app: str) -> str:
    from memo.terminal_presenter import deliver_input

    return deliver_input(tty, payload, terminal_app=terminal_app)


def _default_transport_probe(tty: Path, terminal_app: str) -> bool:
    from memo.terminal_presenter import exact_tty_transport_supported

    return exact_tty_transport_supported(tty, terminal_app)


def _default_process_binding_probe() -> bool:
    # None of the legacy TTY transports binds an input operation atomically to
    # the registered PID/start identity. Production delivery therefore fails
    # closed until a cooperative receiver or native process-bound API exists.
    return False


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _is_local_tty_path(path: Path, mode: int) -> bool:
    return path.is_relative_to(Path("/dev")) and stat.S_ISCHR(mode)


def _path_is_tty(path: Path) -> bool:
    flags = os.O_RDWR | os.O_NONBLOCK | getattr(os, "O_NOCTTY", 0)
    try:
        fd = os.open(path, flags)
    except OSError:
        return False
    try:
        return os.isatty(fd)
    finally:
        os.close(fd)


def _canonical_tty(value: str | Path, *, uid: int | None = None) -> Path:
    raw = Path(value).expanduser()
    if not raw.is_absolute():
        raw = Path("/dev") / raw
    try:
        path = raw.resolve(strict=True)
        info = path.stat()
    except OSError as exc:
        raise TerminalValidationError("terminal target is unavailable") from exc
    if not _is_local_tty_path(path, info.st_mode):
        raise TerminalValidationError("terminal target is not a local TTY")
    expected_uid = os.getuid() if uid is None else uid
    if info.st_uid != expected_uid:
        raise TerminalValidationError("terminal target belongs to another user")
    if not _path_is_tty(path):
        raise TerminalValidationError("terminal target is not a local TTY")
    return path


def _pid_exists(pid: int) -> bool | None:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return None
    return True


def _linux_process_birth_identity(pid: int) -> str:
    try:
        stat_line = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
    except OSError:
        return ""
    closing_paren = stat_line.rfind(")")
    if closing_paren < 0:
        return ""
    fields = stat_line[closing_paren + 2 :].split()
    # fields[0] is proc stat field 3 (state); starttime is field 22.
    if len(fields) <= 19 or not fields[19].isdigit():
        return ""
    return f"linux-start-ticks:{fields[19]}"


def _darwin_process_birth_identity(pid: int) -> str:
    try:
        import ctypes

        class ProcBsdInfo(ctypes.Structure):
            _fields_ = [
                ("pbi_flags", ctypes.c_uint32),
                ("pbi_status", ctypes.c_uint32),
                ("pbi_xstatus", ctypes.c_uint32),
                ("pbi_pid", ctypes.c_uint32),
                ("pbi_ppid", ctypes.c_uint32),
                ("pbi_uid", ctypes.c_uint32),
                ("pbi_gid", ctypes.c_uint32),
                ("pbi_ruid", ctypes.c_uint32),
                ("pbi_rgid", ctypes.c_uint32),
                ("pbi_svuid", ctypes.c_uint32),
                ("pbi_svgid", ctypes.c_uint32),
                ("rfu_1", ctypes.c_uint32),
                ("pbi_comm", ctypes.c_char * 16),
                ("pbi_name", ctypes.c_char * 32),
                ("pbi_nfiles", ctypes.c_uint32),
                ("pbi_pgid", ctypes.c_uint32),
                ("pbi_pjobc", ctypes.c_uint32),
                ("e_tdev", ctypes.c_uint32),
                ("e_tpgid", ctypes.c_uint32),
                ("pbi_nice", ctypes.c_int32),
                ("pbi_start_tvsec", ctypes.c_uint64),
                ("pbi_start_tvusec", ctypes.c_uint64),
            ]

        info = ProcBsdInfo()
        libproc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        proc_pidinfo = libproc.proc_pidinfo
        proc_pidinfo.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_uint64,
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        proc_pidinfo.restype = ctypes.c_int
        size = ctypes.sizeof(info)
        if proc_pidinfo(pid, 3, 0, ctypes.byref(info), size) != size:
            return ""
        return f"darwin-start:{info.pbi_start_tvsec}:{info.pbi_start_tvusec}"
    except (AttributeError, OSError, TypeError, ValueError):
        return ""


def _process_birth_identity(pid: int) -> str:
    if sys.platform.startswith("linux"):
        return _linux_process_birth_identity(pid)
    if sys.platform == "darwin":
        return _darwin_process_birth_identity(pid)
    return ""


def _default_receipt_owner_probe(pid: int, started_at: str) -> bool | None:
    alive = _pid_exists(pid)
    if alive is not True:
        return alive
    identity = _process_birth_identity(pid)
    if not identity:
        return None
    return hmac.compare_digest(identity, started_at)


def _process_snapshot(pid: int) -> ProcessSnapshot | _ProbeUnknown | None:
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
        return _PROBE_UNKNOWN
    if proc.returncode != 0 or not proc.stdout.strip():
        return None if _pid_exists(pid) is False else _PROBE_UNKNOWN
    parts = proc.stdout.strip().split(maxsplit=9)
    if len(parts) != 10 or parts[1] in {"?", "??", "-"}:
        return _PROBE_UNKNOWN
    try:
        uid = int(parts[0])
        pgid = int(parts[7])
        foreground_pgid = int(parts[8])
    except ValueError:
        return _PROBE_UNKNOWN
    tty = Path(parts[1])
    if not tty.is_absolute():
        tty = Path("/dev") / tty
    birth_identity = _process_birth_identity(pid)
    if not birth_identity:
        return _PROBE_UNKNOWN
    return ProcessSnapshot(
        pid=pid,
        uid=uid,
        tty=tty,
        started_at=birth_identity,
        pgid=pgid,
        foreground_pgid=foreground_pgid,
        command=parts[9],
    )


def _command_matches_agent(command: str, agent: str) -> bool:
    lowered = command.lower()
    return any(part == agent or part.endswith(f"/{agent}") for part in lowered.split()) or (
        f"/{agent} " in lowered
    )


def _receipt_fingerprint(
    target_id: str,
    sender_id: str,
    kind: str,
    payload: bytes,
) -> str:
    digest = hashlib.sha256()
    for part in (target_id.encode(), sender_id.encode(), kind.encode(), payload):
        digest.update(len(part).to_bytes(8, "big"))
        digest.update(part)
    return digest.hexdigest()


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


def _utf8_bytes(message: str) -> bytes:
    try:
        encoded_message = message.encode("utf-8")
    except UnicodeEncodeError:
        encoded_message = None
    if encoded_message is None:
        # Raise after leaving the codec exception handler so neither the
        # surrogate-bearing body nor codec offsets survive in exception context.
        raise TerminalValidationError("terminal message is not valid UTF-8") from None
    return encoded_message


def _strip_terminal_controls(message: str) -> str:
    if len(_utf8_bytes(message)) > _MAX_MESSAGE_BYTES:
        raise TerminalValidationError("terminal message exceeds 16 KiB")
    result: list[str] = []
    i = 0
    while i < len(message):
        char = message[i]
        if char == "\x1b":
            i = _consume_escape(message, i)
            continue
        if char == "\n":
            result.append(r"\n")
        elif char == "\t":
            result.append(r"\t")
        elif unicodedata.category(char) != "Cc":
            result.append(char)
        i += 1
    sanitized = "".join(result)
    if not sanitized.strip():
        raise TerminalValidationError("terminal message is empty after sanitization")
    if len(sanitized.encode("utf-8")) > _MAX_MESSAGE_BYTES:
        raise TerminalValidationError("terminal message exceeds 16 KiB after sanitization")
    return sanitized


class TerminalBridge:
    """Persist and revalidate local terminal registrations."""

    def __init__(
        self,
        cfg: Config,
        *,
        process_probe: ProcessProbe | None = None,
        presenter: Presenter | None = None,
        transport_probe: TransportProbe | None = None,
        process_binding_probe: ProcessBindingProbe | None = None,
        receipt_owner_probe: ReceiptOwnerProbe | None = None,
        failpoint: Failpoint | None = None,
    ) -> None:
        self._path = cfg.state_dir / "terminal_live.db"
        self._lock_dir = cfg.state_dir / "terminal-live-locks"
        self._probe = process_probe or _process_snapshot
        self._present = presenter or _default_presenter
        self._transport_supported = transport_probe or _default_transport_probe
        self._process_binding_available = process_binding_probe or _default_process_binding_probe
        self._receipt_owner_probe = receipt_owner_probe or _default_receipt_owner_probe
        self._failpoint = failpoint
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
        self._lock_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._lock_dir.chmod(0o700)
        with self._connect() as conn:
            # Serialize additive migrations across processes. Without an early
            # write lock, two first-open callers can both observe a missing
            # column and race the same ALTER TABLE.
            conn.execute("BEGIN IMMEDIATE")
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
                    delivered_at TEXT NOT NULL,
                    fingerprint TEXT NOT NULL DEFAULT '',
                    owner_pid INTEGER NOT NULL DEFAULT 0,
                    owner_started_at TEXT NOT NULL DEFAULT ''
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS terminal_receipt_tombstones (
                    message_id TEXT PRIMARY KEY,
                    fingerprint TEXT NOT NULL,
                    receipt_id TEXT NOT NULL
                )
                """
            )
            receipt_columns = {
                str(row["name"]) for row in conn.execute("PRAGMA table_info(terminal_receipts)")
            }
            migrations = (
                ("fingerprint", "TEXT NOT NULL DEFAULT ''"),
                ("owner_pid", "INTEGER NOT NULL DEFAULT 0"),
                ("owner_started_at", "TEXT NOT NULL DEFAULT ''"),
            )
            for name, declaration in migrations:
                if name not in receipt_columns:
                    conn.execute(f"ALTER TABLE terminal_receipts ADD COLUMN {name} {declaration}")
            conn.execute(
                "UPDATE terminal_receipts SET status = 'unknown', "
                "error = 'DeliveryStateUnknown' WHERE status = 'pending' "
                "AND (owner_pid <= 1 OR owner_started_at = '')"
            )
            self._recover_pending_receipts(conn)
            self._prune_receipts(conn)
        self._path.chmod(0o600)

    def _recover_pending_receipts(self, conn: sqlite3.Connection) -> None:
        rows = conn.execute(
            "SELECT receipt_id, owner_pid, owner_started_at, created_at "
            "FROM terminal_receipts "
            "WHERE status = 'pending'"
        ).fetchall()
        unknown_ids = []
        for row in rows:
            owner_active = self._receipt_owner_probe(
                int(row["owner_pid"]),
                str(row["owner_started_at"]),
            )
            try:
                created_at = datetime.fromisoformat(str(row["created_at"]))
                if created_at.tzinfo is None:
                    created_at = created_at.replace(tzinfo=UTC)
                lease_expired = datetime.now(UTC) - created_at > _PENDING_LEASE
            except (TypeError, ValueError):
                lease_expired = True
            # An unverifiable owner is not evidence that delivery is active.
            # The bounded lease also recovers failures that leave the owning
            # process alive but no longer executing the delivery.
            if owner_active is not True or lease_expired:
                unknown_ids.append((str(row["receipt_id"]),))
        if unknown_ids:
            conn.executemany(
                "UPDATE terminal_receipts SET status = 'unknown', "
                "error = 'DeliveryStateUnknown' WHERE receipt_id = ?",
                unknown_ids,
            )

    @staticmethod
    def _prune_receipts(conn: sqlite3.Connection) -> None:
        tombstone_limit = max(0, _MAX_RECEIPT_KEYS - _MAX_FINAL_RECEIPTS)
        tombstone_count = int(
            conn.execute("SELECT COUNT(*) FROM terminal_receipt_tombstones").fetchone()[0]
        )
        available = max(0, tombstone_limit - tombstone_count)
        if not available:
            return
        rows = conn.execute(
            "SELECT receipt_id, message_id, fingerprint FROM terminal_receipts "
            "WHERE status IN ('delivered', 'failed') "
            "ORDER BY rowid DESC LIMIT ? OFFSET ?",
            (available, _MAX_FINAL_RECEIPTS),
        ).fetchall()
        if not rows:
            return
        conn.executemany(
            "INSERT OR IGNORE INTO terminal_receipt_tombstones "
            "(message_id, fingerprint, receipt_id) VALUES (?, ?, ?)",
            [
                (str(row["message_id"]), str(row["fingerprint"]), str(row["receipt_id"]))
                for row in rows
            ],
        )
        conn.executemany(
            "DELETE FROM terminal_receipts WHERE receipt_id = ?",
            [(str(row["receipt_id"]),) for row in rows],
        )

    @staticmethod
    def _ensure_receipt_capacity(conn: sqlite3.Connection) -> None:
        live_count = int(conn.execute("SELECT COUNT(*) FROM terminal_receipts").fetchone()[0])
        tombstone_count = int(
            conn.execute("SELECT COUNT(*) FROM terminal_receipt_tombstones").fetchone()[0]
        )
        if live_count + tombstone_count >= _MAX_RECEIPT_KEYS:
            raise TerminalValidationError(
                "terminal receipt idempotency capacity is exhausted; refusing new delivery"
            )

    @staticmethod
    def _lock_key(tty: Path | None) -> str:
        material = "missing-target"
        if tty is not None:
            try:
                info = tty.stat()
                material = f"tty:{info.st_dev}:{info.st_rdev}"
            except OSError:
                material = f"tty-path:{tty}"
        # A fixed bucket set bounds persistent lockfiles. Hash collisions only
        # serialize unrelated terminals; they cannot weaken mutual exclusion.
        digest = hashlib.sha256(material.encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:8], "big") % _LOCK_BUCKETS
        return f"{bucket:02x}"

    @contextmanager
    def _exclusive_lock(self, tty: Path | None) -> Iterator[None]:
        lock_path = self._lock_dir / f"{self._lock_key(tty)}.lock"
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(lock_path, flags, 0o600)
        try:
            info = os.fstat(fd)
            if info.st_uid != os.getuid() or not stat.S_ISREG(info.st_mode):
                raise TerminalValidationError("terminal delivery lock is unsafe")
            os.fchmod(fd, 0o600)
            deadline = time.monotonic() + _LOCK_TIMEOUT_SECONDS
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except OSError as exc:
                    if exc.errno not in {errno.EACCES, errno.EAGAIN}:
                        raise TerminalValidationError(
                            "terminal delivery lock could not be acquired"
                        ) from exc
                    if time.monotonic() >= deadline:
                        raise TerminalValidationError("terminal delivery lock timed out") from None
                    time.sleep(0.01)
            yield
        finally:
            with suppress(OSError):
                fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    @contextmanager
    def _target_lock(self, target_id: str) -> Iterator[None]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT tty FROM terminal_registrations WHERE id = ?",
                (target_id,),
            ).fetchone()
        tty = Path(str(row["tty"])) if row else None
        with self._exclusive_lock(tty):
            yield

    def _trip_failpoint(self, name: str) -> None:
        if self._failpoint is not None:
            self._failpoint(name)

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

    def _validated_registration_target(
        self,
        *,
        agent: str,
        tty: str | Path,
        pid: int,
        terminal_app: str,
    ) -> tuple[str, str, ProcessSnapshot, Path]:
        normalized_agent = agent.strip().lower()
        if normalized_agent not in _AGENTS:
            raise TerminalValidationError("unsupported agent registration")
        app_key = terminal_app.strip().lower()
        if app_key not in _TERMINAL_APPS:
            raise TerminalValidationError("unsupported terminal application")
        if not self._process_binding_available():
            raise TerminalValidationError(
                "terminal registration is disabled: no process-bound input transport is available"
            )
        snapshot = self._probe(pid)
        if isinstance(snapshot, _ProbeUnknown):
            raise TerminalValidationError("agent process identity could not be proven")
        if snapshot is None or snapshot.uid != os.getuid():
            raise TerminalValidationError("agent process is unavailable")
        if not _command_matches_agent(snapshot.command, normalized_agent):
            raise TerminalValidationError("agent process command does not match registration")
        canonical_tty = _canonical_tty(tty, uid=snapshot.uid)
        try:
            process_tty = _canonical_tty(snapshot.tty, uid=snapshot.uid)
        except TerminalValidationError as exc:
            raise TerminalValidationError("agent process has no valid TTY") from exc
        if process_tty != canonical_tty:
            raise TerminalValidationError("agent process is attached to a different TTY")
        normalized_app = _TERMINAL_APPS[app_key]
        if not self._transport_supported(canonical_tty, normalized_app):
            raise TerminalValidationError(
                "no safe exact-TTY terminal transport is available for this terminal"
            )
        return normalized_agent, normalized_app, snapshot, canonical_tty

    def register(
        self,
        *,
        agent: str,
        tty: str | Path,
        pid: int,
        terminal_app: str = "",
        project: str | Path = "",
    ) -> TerminalRegistration:
        normalized_agent, normalized_app, snapshot, canonical_tty = (
            self._validated_registration_target(
                agent=agent,
                tty=tty,
                pid=pid,
                terminal_app=terminal_app,
            )
        )

        now = _now()
        with self._exclusive_lock(canonical_tty), self._connect() as conn:
            existing = conn.execute(
                "SELECT id, pid, uid, agent, process_started_at, created_at "
                "FROM terminal_registrations WHERE tty = ?",
                (str(canonical_tty),),
            ).fetchone()
            same_process = bool(
                existing
                and int(existing["pid"]) == pid
                and int(existing["uid"]) == snapshot.uid
                and str(existing["agent"]) == normalized_agent
                and str(existing["process_started_at"]) == snapshot.started_at
            )
            registration_id = (
                str(existing["id"]) if same_process else f"term-{secrets.token_hex(8)}"
            )
            created_at = str(existing["created_at"]) if same_process else now
            conn.execute(
                """
                    INSERT INTO terminal_registrations (
                        id, tty, pid, uid, agent, terminal_app, project,
                        process_started_at, created_at, last_seen_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(tty) DO UPDATE SET
                        id = excluded.id,
                        pid = excluded.pid,
                        uid = excluded.uid,
                        agent = excluded.agent,
                        terminal_app = excluded.terminal_app,
                        project = excluded.project,
                        process_started_at = excluded.process_started_at,
                        created_at = excluded.created_at,
                        last_seen_at = excluded.last_seen_at
                    """,
                (
                    registration_id,
                    str(canonical_tty),
                    pid,
                    snapshot.uid,
                    normalized_agent,
                    normalized_app,
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

    def _registration_state(
        self,
        registration: TerminalRegistration,
        *,
        require_foreground: bool = False,
    ) -> tuple[Literal["live", "unknown", "stale", "changed", "background"], str]:
        snapshot = self._probe(registration.pid)
        if isinstance(snapshot, _ProbeUnknown):
            return "unknown", "process identity probe is temporarily unavailable"
        if (
            snapshot is None
            or snapshot.uid != registration.uid
            or snapshot.started_at != registration.process_started_at
            or not _command_matches_agent(snapshot.command, registration.agent)
        ):
            return "stale", "registered terminal process is stale"
        try:
            process_tty = _canonical_tty(snapshot.tty, uid=registration.uid)
        except TerminalValidationError:
            return "stale", "registered terminal process is stale"
        if str(process_tty) != registration.tty:
            return "changed", "registered terminal process changed TTY"
        if require_foreground and snapshot.pgid != snapshot.foreground_pgid:
            return "background", "registered agent is not the foreground terminal process"
        return "live", ""

    @staticmethod
    def _within_probe_grace(registration: TerminalRegistration) -> bool:
        try:
            last_seen = datetime.fromisoformat(registration.last_seen_at)
        except ValueError:
            return False
        return datetime.now(UTC) - last_seen <= _PROBE_GRACE

    def list(self) -> list[TerminalRegistration]:
        if not self._process_binding_available():
            with self._connect() as conn:
                conn.execute("DELETE FROM terminal_registrations")
            return []
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM terminal_registrations ORDER BY created_at, id"
            ).fetchall()
            registrations = [self._from_row(row) for row in rows]
            live: list[TerminalRegistration] = []
            stale_ids: list[str] = []
            refreshed_ids: list[tuple[str, str]] = []
            now = _now()
            for item in registrations:
                if not self._transport_supported(Path(item.tty), item.terminal_app):
                    stale_ids.append(item.id)
                    continue
                state, _reason = self._registration_state(item)
                if state == "live":
                    refreshed = replace(item, last_seen_at=now)
                    live.append(refreshed)
                    refreshed_ids.append((now, item.id))
                elif state == "unknown" and self._within_probe_grace(item):
                    live.append(item)
                elif state != "unknown":
                    stale_ids.append(item.id)
            if refreshed_ids:
                conn.executemany(
                    "UPDATE terminal_registrations SET last_seen_at = ? WHERE id = ?",
                    refreshed_ids,
                )
            if stale_ids:
                conn.executemany(
                    "DELETE FROM terminal_registrations WHERE id = ?",
                    [(item,) for item in stale_ids],
                )
        return live

    def registration_id(self, registration_id: str) -> str:
        """Preserve an explicit sender id for core live-validation on delivery."""
        return registration_id.strip()

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

    def _retry_receipt(
        self,
        conn: sqlite3.Connection,
        message_id: str,
        fingerprint: str,
    ) -> TerminalReceipt | None:
        row = conn.execute(
            "SELECT * FROM terminal_receipts WHERE message_id = ?",
            (message_id,),
        ).fetchone()
        if row is None:
            tombstone = conn.execute(
                "SELECT receipt_id, fingerprint FROM terminal_receipt_tombstones "
                "WHERE message_id = ?",
                (message_id,),
            ).fetchone()
            if tombstone is None:
                return None
            stored_fingerprint = str(tombstone["fingerprint"])
            if not stored_fingerprint or not hmac.compare_digest(
                stored_fingerprint,
                fingerprint,
            ):
                raise TerminalValidationError(
                    "terminal message id conflicts with a different delivery request"
                )
            return TerminalReceipt(
                receipt_id=str(tombstone["receipt_id"]),
                message_id=message_id,
                target_id="",
                sender_id="",
                kind="",
                status="duplicate",
                transport="",
                error="ReceiptPruned",
                created_at="",
                delivered_at="",
            )
        stored_fingerprint = str(row["fingerprint"])
        if not stored_fingerprint or not hmac.compare_digest(stored_fingerprint, fingerprint):
            raise TerminalValidationError(
                "terminal message id conflicts with a different delivery request"
            )
        receipt = self._receipt_from_row(row)
        if receipt.status == "pending":
            # This method runs while holding the target's interprocess lock. A
            # remaining pending row therefore has no active presenter and is
            # durably ambiguous; never attempt the body again.
            conn.execute(
                "UPDATE terminal_receipts SET status = 'unknown', "
                "error = 'DeliveryStateUnknown' WHERE receipt_id = ?",
                (receipt.receipt_id,),
            )
            return replace(receipt, status="unknown", error="DeliveryStateUnknown")
        if receipt.status == "unknown":
            return receipt
        return replace(receipt, status="duplicate")

    def _live_registration(
        self,
        registration_id: str,
        *,
        role: Literal["target", "sender"],
        require_foreground: bool,
    ) -> TerminalRegistration:
        if not self._process_binding_available():
            raise TerminalValidationError(
                "terminal delivery is disabled: no process-bound input transport is available"
            )
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM terminal_registrations WHERE id = ?",
                (registration_id,),
            ).fetchone()
        if row is None:
            raise TerminalValidationError(f"registered terminal {role} was not found")
        registration = self._from_row(row)
        if not self._transport_supported(Path(registration.tty), registration.terminal_app):
            raise TerminalValidationError(
                f"registered terminal {role} has no safe exact-TTY transport"
            )
        state, reason = self._registration_state(
            registration,
            require_foreground=require_foreground,
        )
        if state != "live":
            raise TerminalValidationError(f"registered terminal {role}: {reason}")
        now = _now()
        with self._connect() as conn:
            conn.execute(
                "UPDATE terminal_registrations SET last_seen_at = ? WHERE id = ?",
                (now, registration.id),
            )
        return replace(registration, last_seen_at=now)

    def _live_target(self, target_id: str) -> TerminalRegistration:
        return self._live_registration(target_id, role="target", require_foreground=True)

    def _live_sender(self, sender_id: str) -> TerminalRegistration:
        return self._live_registration(sender_id, role="sender", require_foreground=False)

    def _attempt_delivery(
        self,
        target_id: str,
        payload: bytes,
        sender_id: str,
    ) -> tuple[str, str, str]:
        """Return transport, stable failure code, and body-free public error."""
        try:
            registration = self._live_target(target_id)
            if sender_id:
                self._live_sender(sender_id)
            transport = self._present(
                Path(registration.tty),
                payload,
                terminal_app=registration.terminal_app,
            )
            if transport not in _DELIVERY_TRANSPORTS:
                return (
                    "",
                    "TerminalValidationError",
                    "terminal presenter returned an invalid transport",
                )
            return transport, "", ""
        except TerminalValidationError as exc:
            return "", type(exc).__name__, str(exc)
        except TerminalDeliveryError as exc:
            safe_error = str(exc)
            return "", safe_error, f"terminal delivery failed: {safe_error}"
        except subprocess.TimeoutExpired:
            return "", "OSError", "terminal delivery timed out"
        except OSError as exc:
            message = (
                "no safe exact-TTY terminal transport is available"
                if exc.errno == errno.ENOTSUP
                else "terminal delivery failed"
            )
            return "", type(exc).__name__, message
        except RuntimeError as exc:
            return "", type(exc).__name__, "terminal delivery failed"

    def _deliver_locked(
        self,
        target_id: str,
        payload: bytes,
        *,
        sender_id: str,
        message_id: str,
        kind: str,
        fingerprint: str,
    ) -> TerminalReceipt:
        now = _now()
        receipt_id = f"rcpt-{secrets.token_hex(8)}"
        owner_started_at = _process_birth_identity(os.getpid())
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            existing = self._retry_receipt(conn, message_id, fingerprint)
            if existing is not None:
                return existing
            self._ensure_receipt_capacity(conn)
            inserted = conn.execute(
                """
                INSERT OR IGNORE INTO terminal_receipts (
                    receipt_id, message_id, target_id, sender_id, kind,
                    status, transport, error, created_at, delivered_at,
                    fingerprint, owner_pid, owner_started_at
                ) VALUES (?, ?, ?, ?, ?, 'pending', '', '', ?, '', ?, ?, ?)
                """,
                (
                    receipt_id,
                    message_id,
                    target_id,
                    sender_id,
                    kind,
                    now,
                    fingerprint,
                    os.getpid(),
                    owner_started_at,
                ),
            )
            if inserted.rowcount == 0:
                duplicate = self._retry_receipt(conn, message_id, fingerprint)
                if duplicate is None:  # pragma: no cover - defensive race guard
                    raise TerminalValidationError("terminal receipt collision")
                return duplicate

        self._trip_failpoint("before_presenter")
        transport, failure_error, failure_message = self._attempt_delivery(
            target_id,
            payload,
            sender_id,
        )
        self._trip_failpoint("after_presenter")
        self._trip_failpoint("before_finalize")

        if failure_error:
            with self._connect() as conn:
                conn.execute(
                    "UPDATE terminal_receipts SET status = 'failed', error = ? "
                    "WHERE receipt_id = ?",
                    (failure_error, receipt_id),
                )
                self._prune_receipts(conn)
            # Raise after leaving the exception handler so presenter exceptions
            # cannot retain a body-bearing argv through chained context.
            raise TerminalValidationError(failure_message)

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
            self._prune_receipts(conn)
        if row is None:  # pragma: no cover - guarded by insert above
            raise TerminalValidationError("terminal receipt was not persisted")
        return self._receipt_from_row(row)

    def _deliver(
        self,
        target_id: str,
        payload: bytes,
        *,
        sender: str | None,
        message_id: str | None,
        kind: str,
    ) -> TerminalReceipt:
        resolved_target_id = target_id.strip()
        if not _REGISTRATION_ID_RE.fullmatch(resolved_target_id):
            raise TerminalValidationError("terminal target id is malformed")
        resolved_message_id = (message_id or f"msg-{secrets.token_hex(12)}").strip()
        if not _MESSAGE_ID_RE.fullmatch(resolved_message_id):
            raise TerminalValidationError("terminal message id is malformed or too long")
        sender_id = (sender or "").strip()
        if sender_id and not _REGISTRATION_ID_RE.fullmatch(sender_id):
            raise TerminalValidationError("terminal sender id is malformed")
        if not self._process_binding_available():
            # This is a capability hard fail, not a delivery attempt. Do not
            # create a misleading pending/failed receipt before rejecting it.
            raise TerminalValidationError(
                "terminal delivery is disabled: no process-bound input transport is available"
            )
        fingerprint = _receipt_fingerprint(resolved_target_id, sender_id, kind, payload)
        with self._target_lock(resolved_target_id):
            return self._deliver_locked(
                resolved_target_id,
                payload,
                sender_id=sender_id,
                message_id=resolved_message_id,
                kind=kind,
                fingerprint=fingerprint,
            )

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
            self._recover_pending_receipts(conn)
            rows = conn.execute(
                "SELECT * FROM terminal_receipts ORDER BY rowid DESC LIMIT ?",
                (bounded,),
            ).fetchall()
        return [self._receipt_from_row(row) for row in rows]
