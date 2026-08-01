"""Persistent append-only runtime events owned by Memo."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import sqlite3
import stat
import threading
from collections.abc import Iterator
from contextlib import closing, contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, BinaryIO, TypedDict

from .config import Config

SCHEMA = "memo.terminal_event.v1"
MAX_EVENT_PAGE_SIZE = 1_000
MAX_EVENT_SCAN_LINES = 1_000
MAX_EVENT_SCAN_BYTES = 4 * 1024 * 1024
MAX_EVENT_RECORD_BYTES = MAX_EVENT_SCAN_BYTES
_INDEX_SCHEMA_VERSION = "2"
_INGEST_THREAD_LOCK = threading.Lock()


class EventPage(TypedDict):
    """One bounded page from the append-only event journal."""

    events: list[dict[str, Any]]
    next_cursor: str
    has_more: bool


@dataclass(frozen=True)
class _EventCursor:
    offset: int
    boundary_digest: str | None
    continuation: bool = False


@dataclass
class _ChunkScan:
    events: list[dict[str, Any]]
    position: int
    scanned_lines: int
    next_offset: int
    next_continuation: bool = False
    partial_tail: bool = False


def _paths(state_dir: Path) -> tuple[Path, Path]:
    root = state_dir / "events"
    return root / "terminal-conversation.jsonl", root / "context.json"


def _index_path(data_path: Path) -> Path:
    return data_path.with_name("terminal-conversation-index.sqlite3")


def _lock_path(data_path: Path) -> Path:
    return data_path.with_name("terminal-conversation.lock")


def _context(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"epoch": 0, "context_id": None}
    except OSError as exc:
        raise RuntimeError("cannot read event context") from exc
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise RuntimeError("invalid event context") from exc
    if not isinstance(value, dict):
        raise RuntimeError("invalid event context")
    epoch = value.get("epoch", 0)
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise RuntimeError("invalid event context epoch")
    return value


def _require_regular_or_missing(path: Path, *, label: str) -> None:
    """Reject symlinks and special files at authority-owned state paths."""
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        return
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise ValueError(f"unsafe {label} path: {path}")


@contextmanager
def _ingest_lock(data_path: Path) -> Iterator[None]:
    """Serialize journal/index changes across threads and processes."""
    lock_path = _lock_path(data_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    _require_regular_or_missing(lock_path, label="event-lock")
    fd = os.open(
        lock_path,
        os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        lock_fh = os.fdopen(fd, "a+b")
    except BaseException:
        os.close(fd)
        raise
    with _INGEST_THREAD_LOCK, lock_fh:
        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)


def _open_index(data_path: Path) -> sqlite3.Connection:
    path = _index_path(data_path)
    _require_regular_or_missing(path, label="event-index")
    if not path.exists():
        try:
            fd = os.open(
                path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0),
                0o600,
            )
        except FileExistsError:
            pass
        else:
            os.close(fd)
    conn = sqlite3.connect(path, timeout=30.0)
    try:
        conn.execute("PRAGMA synchronous=FULL")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS event_receipts (
                event_id TEXT PRIMARY KEY,
                payload_json TEXT NOT NULL,
                end_offset INTEGER NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS event_index_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        conn.commit()
    except BaseException:
        conn.close()
        raise
    return conn


def _meta(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute(
        "SELECT value FROM event_index_meta WHERE key = ?",
        (key,),
    ).fetchone()
    return str(row[0]) if row is not None else None


def _set_meta(conn: sqlite3.Connection, key: str, value: str | int) -> None:
    conn.execute(
        """
        INSERT INTO event_index_meta(key, value) VALUES (?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """,
        (key, str(value)),
    )


def _reset_index(conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM event_receipts")
    conn.execute("DELETE FROM event_index_meta")


def _record_receipt(
    conn: sqlite3.Connection,
    record: dict[str, Any],
    *,
    end_offset: int,
) -> None:
    payload = json.dumps(record, sort_keys=True, ensure_ascii=True)
    row = conn.execute(
        "SELECT payload_json FROM event_receipts WHERE event_id = ?",
        (record["event_id"],),
    ).fetchone()
    if row is not None:
        old = json.loads(str(row[0]))
        if old != record:
            raise ValueError("event_id already exists with different payload")
        return
    conn.execute(
        """
        INSERT INTO event_receipts(event_id, payload_json, end_offset)
        VALUES (?, ?, ?)
        """,
        (record["event_id"], payload, end_offset),
    )


def _journal_identity(data_path: Path) -> tuple[str, str, int] | None:
    try:
        stat = data_path.stat()
    except FileNotFoundError:
        return None
    return str(stat.st_dev), str(stat.st_ino), stat.st_size


def _boundary_digest(data_path: Path, offset: int) -> str:
    """Fingerprint the indexed boundary to detect truncate+regrow in place."""
    if offset <= 0:
        return hashlib.sha256(b"").hexdigest()
    with data_path.open("rb") as fh:
        start = max(0, offset - 64)
        fh.seek(start)
        return hashlib.sha256(fh.read(offset - start)).hexdigest()


def _normalize_incomplete_tail(data_path: Path, start: int, tail: bytes) -> int:
    """Keep a complete newline-less JSON record or discard an interrupted tail."""
    try:
        item = json.loads(tail)
    except (json.JSONDecodeError, UnicodeDecodeError):
        item = None
    with data_path.open("r+b") as fh:
        if isinstance(item, dict) and isinstance(item.get("event_id"), str):
            fh.seek(0, os.SEEK_END)
            fh.write(b"\n")
            end_offset = fh.tell()
        else:
            fh.truncate(start)
            end_offset = start
        fh.flush()
        os.fsync(fh.fileno())
    return end_offset


def _discard_oversized_legacy_line(
    data_path: Path,
    fh: BinaryIO,
    *,
    line_start: int,
    chunk: bytes,
) -> int:
    """Skip one oversized legacy record without holding it all in memory."""
    while chunk and not chunk.endswith(b"\n"):
        chunk = fh.readline(MAX_EVENT_RECORD_BYTES + 1)
    if chunk:
        return fh.tell()
    with data_path.open("r+b") as writable:
        writable.truncate(line_start)
        writable.flush()
        os.fsync(writable.fileno())
    return line_start


def _scan_index_delta(
    conn: sqlite3.Connection,
    data_path: Path,
    *,
    start: int,
) -> int:
    """Index complete journal lines from ``start`` without retaining the prefix."""
    if not data_path.exists():
        return 0
    last_complete = start
    with data_path.open("rb") as fh:
        fh.seek(start)
        while True:
            line_start = fh.tell()
            line = fh.readline(MAX_EVENT_RECORD_BYTES + 1)
            if not line:
                break
            if len(line) > MAX_EVENT_RECORD_BYTES:
                last_complete = _discard_oversized_legacy_line(
                    data_path,
                    fh,
                    line_start=line_start,
                    chunk=line,
                )
                if last_complete == line_start:
                    break
                continue
            if not line.endswith(b"\n"):
                end_offset = _normalize_incomplete_tail(data_path, line_start, line)
                if end_offset == line_start:
                    break
                line += b"\n"
                last_complete = end_offset
            else:
                last_complete = fh.tell()
            try:
                item = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if not isinstance(item, dict) or not isinstance(item.get("event_id"), str):
                continue
            _record_receipt(conn, item, end_offset=last_complete)
    return last_complete


def _reconcile_index(conn: sqlite3.Connection, data_path: Path) -> int:
    """Rebuild once for legacy journals, then inspect only unindexed tail bytes."""
    identity = _journal_identity(data_path)
    if identity is None:
        _reset_index(conn)
        _set_meta(conn, "schema_version", _INDEX_SCHEMA_VERSION)
        _set_meta(conn, "indexed_bytes", 0)
        conn.commit()
        return 0

    device, inode, size = identity
    schema = _meta(conn, "schema_version")
    indexed_raw = _meta(conn, "indexed_bytes")
    stored_device = _meta(conn, "journal_device")
    stored_inode = _meta(conn, "journal_inode")
    stored_boundary = _meta(conn, "boundary_digest")
    try:
        indexed = int(indexed_raw or "0")
    except ValueError:
        indexed = -1
    if (
        schema != _INDEX_SCHEMA_VERSION
        or indexed < 0
        or indexed > size
        or (stored_device is not None and stored_device != device)
        or (stored_inode is not None and stored_inode != inode)
        or (
            stored_boundary is not None
            and indexed <= size
            and stored_boundary != _boundary_digest(data_path, indexed)
        )
    ):
        _reset_index(conn)
        indexed = 0

    indexed = _scan_index_delta(conn, data_path, start=indexed)
    identity = _journal_identity(data_path)
    if identity is not None:
        device, inode, _ = identity
        _set_meta(conn, "journal_device", device)
        _set_meta(conn, "journal_inode", inode)
    _set_meta(conn, "schema_version", _INDEX_SCHEMA_VERSION)
    _set_meta(conn, "indexed_bytes", indexed)
    _set_meta(conn, "boundary_digest", _boundary_digest(data_path, indexed))
    conn.commit()
    return indexed


def _append_record(data_path: Path, record: dict[str, Any]) -> int:
    payload = (json.dumps(record, sort_keys=True, ensure_ascii=True) + "\n").encode()
    if len(payload) > MAX_EVENT_RECORD_BYTES:
        raise ValueError(f"event record exceeds {MAX_EVENT_RECORD_BYTES} bytes")
    fd = os.open(
        data_path,
        os.O_APPEND | os.O_CREAT | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        view = memoryview(payload)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError("event journal append made no progress")
            view = view[written:]
        os.fsync(fd)
        return os.lseek(fd, 0, os.SEEK_END)
    finally:
        os.close(fd)


def ingest_event(
    event: dict[str, Any], *, state_dir: Path | None = None, expected_epoch: int | None = None
) -> dict[str, Any]:
    if (
        not isinstance(event, dict)
        or not isinstance(event.get("event_id"), str)
        or not event["event_id"]
    ):
        raise ValueError("event_id is required")
    kind = event.get("kind") or event.get("type")
    if not isinstance(kind, str) or kind not in {
        "terminal",
        "conversation",
        "agent",
        "signal",
    }:
        raise ValueError("kind must be terminal, conversation, agent, or signal")
    data_path, context_path = _paths(state_dir or Config.from_env().state_dir)
    record = dict(event)
    record.update(schema=SCHEMA, kind=kind)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    _require_regular_or_missing(data_path, label="event-journal")
    _require_regular_or_missing(context_path, label="event-context")
    with _ingest_lock(data_path):
        context = _context(context_path)
        if expected_epoch is not None and not context_path.exists():
            raise RuntimeError("missing context")
        if expected_epoch is not None and expected_epoch != int(context.get("epoch", 0)):
            raise RuntimeError("stale context epoch")
        with closing(_open_index(data_path)) as conn:
            _reconcile_index(conn, data_path)
            row = conn.execute(
                "SELECT payload_json FROM event_receipts WHERE event_id = ?",
                (record["event_id"],),
            ).fetchone()
            if row is not None:
                old = json.loads(str(row[0]))
                if old != record:
                    raise ValueError("event_id already exists with different payload")
                return {
                    "accepted": False,
                    "duplicate": True,
                    "event": old,
                    "epoch": context.get("epoch", 0),
                }

            end_offset = _append_record(data_path, record)
            _record_receipt(conn, record, end_offset=end_offset)
            identity = _journal_identity(data_path)
            if identity is not None:
                device, inode, _ = identity
                _set_meta(conn, "journal_device", device)
                _set_meta(conn, "journal_inode", inode)
            _set_meta(conn, "schema_version", _INDEX_SCHEMA_VERSION)
            _set_meta(conn, "indexed_bytes", end_offset)
            _set_meta(conn, "boundary_digest", _boundary_digest(data_path, end_offset))
            conn.commit()
    return {
        "accepted": True,
        "duplicate": False,
        "event": record,
        "epoch": context.get("epoch", 0),
    }


def list_events(
    *, state_dir: Path | None = None, kind: str | None = None, limit: int = 100
) -> list[dict[str, Any]]:
    """Return the legacy tail-shaped event list.

    This deliberately retains the original whole-file implementation and
    ``limit=0`` semantics because existing callers depend on that exact
    contract. New incremental consumers should use :func:`list_event_page`.
    """
    data_path, _ = _paths(state_dir or Config.from_env().state_dir)
    _require_regular_or_missing(data_path, label="event-journal")
    if not data_path.exists():
        return []
    out = []
    for line in data_path.read_text(encoding="utf-8").splitlines():
        try:
            item = json.loads(line)
            if kind is None or item.get("kind") == kind:
                out.append(item)
        except json.JSONDecodeError:
            continue
    return out[-max(0, limit) :]


def _parse_cursor(cursor: str | int) -> _EventCursor:
    raw = str(cursor).strip()
    if not raw:
        return _EventCursor(0, None)
    if raw.startswith("v1:"):
        parts = raw.split(":")
        if len(parts) != 4 or parts[3] not in {"0", "1"}:
            raise ValueError("cursor is invalid")
        try:
            offset = int(parts[1])
        except ValueError as exc:
            raise ValueError("cursor is invalid") from exc
        digest = parts[2]
        if (
            offset < 0
            or (offset == 0 and parts[3] == "1")
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
        ):
            raise ValueError("cursor is invalid")
        return _EventCursor(offset, digest, parts[3] == "1")
    try:
        offset = int(raw)
    except ValueError as exc:
        raise ValueError("cursor must be a non-negative byte offset") from exc
    if offset < 0:
        raise ValueError("cursor must be a non-negative byte offset")
    return _EventCursor(offset, None)


def _stream_boundary_digest(fh: BinaryIO, offset: int) -> str:
    current = fh.tell()
    try:
        start = max(0, offset - 64)
        fh.seek(start)
        return hashlib.sha256(fh.read(offset - start)).hexdigest()
    finally:
        fh.seek(current)


def _encode_cursor(fh: BinaryIO, offset: int, *, continuation: bool = False) -> str:
    digest = _stream_boundary_digest(fh, offset)
    return f"v1:{offset}:{digest}:{int(continuation)}"


def _zero_cursor() -> str:
    digest = hashlib.sha256(b"").hexdigest()
    return f"v1:0:{digest}:0"


def _timestamp(value: Any) -> datetime | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), tz=UTC)
        except (OSError, OverflowError, ValueError):
            return None
    if not isinstance(value, str) or not value.strip():
        return None
    raw = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _since_cutoff(since: str | None) -> datetime | None:
    if since is None:
        return None
    cutoff = _timestamp(since)
    if cutoff is None:
        raise ValueError("since must be an ISO-8601 timestamp")
    return cutoff


def _matches_since(item: dict[str, Any], cutoff: datetime | None) -> bool:
    if cutoff is None:
        return True
    value = item.get("timestamp") or item.get("created_at")
    observed = _timestamp(value)
    # Missing or malformed timestamps are retained: migration filters must not
    # silently discard an event that cannot prove it predates the cutoff.
    return observed is None or observed >= cutoff


def _cursor_start(
    fh: BinaryIO,
    cursor: _EventCursor,
    snapshot_size: int,
) -> tuple[int, bool]:
    """Resolve a cursor against one immutable journal-size snapshot."""
    if cursor.offset > snapshot_size:
        return 0, False
    if cursor.offset == 0:
        return 0, cursor.continuation

    digest_matches = (
        cursor.boundary_digest is None
        or cursor.boundary_digest == _stream_boundary_digest(fh, cursor.offset)
    )
    fh.seek(cursor.offset - 1)
    boundary_matches = cursor.continuation or fh.read(1) == b"\n"
    if not digest_matches or not boundary_matches:
        return 0, False
    return cursor.offset, cursor.continuation


def _initial_scan(
    chunk: bytes,
    *,
    offset: int,
    chunk_end: int,
    at_snapshot_end: bool,
    continuation: bool,
) -> _ChunkScan:
    state = _ChunkScan(events=[], position=0, scanned_lines=0, next_offset=offset)
    if not continuation:
        return state

    newline = chunk.find(b"\n")
    if newline >= 0:
        state.position = newline + 1
        state.scanned_lines = 1
        state.next_offset = offset + state.position
    elif at_snapshot_end:
        state.partial_tail = True
    else:
        state.next_offset = chunk_end
        state.next_continuation = True
    return state


def _finish_incomplete_line(
    state: _ChunkScan,
    *,
    offset: int,
    line_start: int,
    chunk_end: int,
    at_snapshot_end: bool,
) -> None:
    if at_snapshot_end:
        # Only newline-terminated records belong to this snapshot. Re-read the
        # tail after its writer finishes.
        state.next_offset = offset + line_start
        state.partial_tail = True
    elif line_start == 0:
        # One physical record exceeds the hard byte budget. Advance in bounded
        # chunks and discard it on reaching its newline.
        state.next_offset = chunk_end
        state.next_continuation = True
    else:
        # Give a record crossing the page boundary a fresh full-size page.
        state.next_offset = offset + line_start


def _matching_event(
    line: bytes,
    *,
    kind: str | None,
    cutoff: datetime | None,
) -> dict[str, Any] | None:
    try:
        item = json.loads(line)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(item, dict):
        return None
    if kind is not None and item.get("kind") != kind:
        return None
    return item if _matches_since(item, cutoff) else None


def _scan_event_chunk(
    chunk: bytes,
    *,
    offset: int,
    snapshot_size: int,
    continuation: bool,
    kind: str | None,
    cutoff: datetime | None,
    limit: int,
) -> _ChunkScan:
    chunk_end = offset + len(chunk)
    at_snapshot_end = chunk_end == snapshot_size
    state = _initial_scan(
        chunk,
        offset=offset,
        chunk_end=chunk_end,
        at_snapshot_end=at_snapshot_end,
        continuation=continuation,
    )
    if continuation and state.position == 0:
        return state

    while (
        state.position < len(chunk)
        and len(state.events) < limit
        and state.scanned_lines < MAX_EVENT_SCAN_LINES
    ):
        line_start = state.position
        newline = chunk.find(b"\n", line_start)
        if newline < 0:
            _finish_incomplete_line(
                state,
                offset=offset,
                line_start=line_start,
                chunk_end=chunk_end,
                at_snapshot_end=at_snapshot_end,
            )
            break

        state.position = newline + 1
        state.next_offset = offset + state.position
        state.scanned_lines += 1
        item = _matching_event(chunk[line_start : newline + 1], kind=kind, cutoff=cutoff)
        if item is not None:
            state.events.append(item)
    return state


def list_event_page(
    *,
    cursor: str | int,
    state_dir: Path | None = None,
    kind: str | None = None,
    limit: int = 100,
    since: str | None = None,
) -> EventPage:
    """Read a bounded event page using a stable opaque byte-offset cursor.

    The file size is snapshotted before reading, so events appended during a
    call are left for the next poll. Malformed lines advance the cursor and are
    skipped. The cursor carries a boundary fingerprint so truncate/rotate and
    regrowth are detected even when the old offset still lands on a newline.
    Legacy decimal cursors remain accepted.
    """
    if limit < 1 or limit > MAX_EVENT_PAGE_SIZE:
        raise ValueError(f"limit must be between 1 and {MAX_EVENT_PAGE_SIZE}")
    parsed_cursor = _parse_cursor(cursor)
    cutoff = _since_cutoff(since)
    data_path, _ = _paths(state_dir or Config.from_env().state_dir)
    _require_regular_or_missing(data_path, label="event-journal")
    if not data_path.exists():
        return {"events": [], "next_cursor": _zero_cursor(), "has_more": False}

    try:
        with data_path.open("rb") as fh:
            fh.seek(0, 2)
            snapshot_size = fh.tell()
            offset, continuation = _cursor_start(fh, parsed_cursor, snapshot_size)
            fh.seek(offset)
            read_size = min(MAX_EVENT_SCAN_BYTES, snapshot_size - offset)
            chunk = fh.read(read_size)
            scan = _scan_event_chunk(
                chunk,
                offset=offset,
                snapshot_size=snapshot_size,
                continuation=continuation,
                kind=kind,
                cutoff=cutoff,
                limit=limit,
            )
            next_cursor = _encode_cursor(
                fh,
                scan.next_offset,
                continuation=scan.next_continuation,
            )
    except FileNotFoundError:
        # Rotation can briefly leave no file between the existence check and
        # open. Resetting to zero lets the next poll consume the replacement.
        return {"events": [], "next_cursor": _zero_cursor(), "has_more": False}

    return {
        "events": scan.events,
        "next_cursor": next_cursor,
        "has_more": not scan.partial_tail and scan.next_offset < snapshot_size,
    }
