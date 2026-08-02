"""Per-session chat history as one JSONL file per session id."""

from __future__ import annotations

import json
import math
import re
import time
from collections import deque
from collections.abc import Iterator
from pathlib import Path
from typing import Any

_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_MAX_REASONABLE_TS = 32_503_680_000  # 3000-01-01 UTC
_MAX_RECENT_SCAN_BYTES = 4 * 1024 * 1024


def iso_ts(ts: float) -> str:
    """Format an epoch-seconds timestamp the way the rest of chat/ does."""
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(ts))


def _iter_reverse_lines(path: Path) -> Iterator[bytes]:
    """Yield bounded JSONL tail lines newest-first in linear time."""

    chunk_size = 8192
    pending: deque[bytes] = deque()
    with path.open("rb") as fh:
        fh.seek(0, 2)
        position = fh.tell()
        scanned = 0
        while position > 0 and scanned < _MAX_RECENT_SCAN_BYTES:
            read_size = min(chunk_size, position, _MAX_RECENT_SCAN_BYTES - scanned)
            position -= read_size
            fh.seek(position)
            block = fh.read(read_size)
            scanned += len(block)
            parts = block.split(b"\n")
            if len(parts) == 1:
                pending.appendleft(block)
                continue

            pending.appendleft(parts[-1])
            yield b"".join(pending)
            yield from reversed(parts[1:-1])
            pending = deque([parts[0]])

        # Only reaching the start proves the remaining fragment is complete.
        if position == 0 and pending:
            yield b"".join(pending)


class SessionStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    @staticmethod
    def validate_id(session_id: object) -> str:
        """Validate a session id without touching the filesystem."""
        if not isinstance(session_id, str) or _SESSION_ID_RE.fullmatch(session_id) is None:
            raise ValueError(f"invalid session id: {session_id!r}")
        return session_id

    def _path(self, session_id: str) -> Path:
        return self.root / f"{self.validate_id(session_id)}.jsonl"

    @staticmethod
    def validate_text(value: object, field: str) -> str:
        """Validate persisted text before creating or appending any file."""
        if not isinstance(value, str):
            raise ValueError(f"{field} must be a string")
        try:
            value.encode("utf-8", errors="strict")
        except UnicodeEncodeError as exc:
            raise ValueError(f"{field} must contain valid UTF-8 text") from exc
        return value

    @staticmethod
    def _parse_turn(raw_line: str | bytes) -> dict[str, Any] | None:
        try:
            parsed = json.loads(raw_line)
        except (ValueError, UnicodeDecodeError, RecursionError, OverflowError):
            return None
        if not isinstance(parsed, dict):
            return None
        role = parsed.get("role")
        text = parsed.get("text")
        ts = parsed.get("ts")
        if not isinstance(role, str) or not isinstance(text, str):
            return None
        if isinstance(ts, bool) or not isinstance(ts, int | float):
            return None
        if isinstance(ts, int):
            if not 0 <= ts <= _MAX_REASONABLE_TS:
                return None
        elif not math.isfinite(ts) or not 0 <= ts <= _MAX_REASONABLE_TS:
            return None
        try:
            role.encode("utf-8", errors="strict")
            text.encode("utf-8", errors="strict")
        except UnicodeEncodeError:
            return None
        return parsed

    def append_turn(self, session_id: str, role: str, text: str) -> None:
        path = self._path(session_id)
        role = self.validate_text(role, "role")
        text = self.validate_text(text, "text")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(
                json.dumps({"role": role, "text": text, "ts": time.time()}, ensure_ascii=False)
                + "\n"
            )

    def append_exchange(self, session_id: str, question: str, answer: str) -> None:
        """Append one complete user/assistant exchange with a single write."""
        path = self._path(session_id)
        question = self.validate_text(question, "question")
        answer = self.validate_text(answer, "answer")
        timestamp = time.time()
        payload = "".join(
            json.dumps(turn, ensure_ascii=False) + "\n"
            for turn in (
                {"role": "user", "text": question, "ts": timestamp},
                {"role": "assistant", "text": answer, "ts": timestamp},
            )
        ).encode("utf-8")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("ab") as fh:
            fh.write(payload)

    def get(self, session_id: str) -> list[dict[str, Any]]:
        path = self._path(session_id)
        try:
            fh = path.open("rb")
        except FileNotFoundError:
            return []
        turns = []
        with fh:
            for line in fh:
                parsed = self._parse_turn(line)
                if parsed is not None:
                    turns.append(parsed)
        return turns

    def get_recent(self, session_id: str, limit: int = 12) -> list[dict[str, Any]]:
        """Read only the tail needed for prompt history.

        Sessions can grow indefinitely. Reading backwards avoids loading and
        decoding an entire JSONL transcript for every follow-up request.
        """
        if type(limit) is not int or limit < 1:
            raise ValueError("limit must be a positive integer")
        path = self._path(session_id)
        newest_turns: list[dict[str, Any]] = []
        try:
            for raw_line in _iter_reverse_lines(path):
                parsed = self._parse_turn(raw_line)
                if parsed is not None:
                    newest_turns.append(parsed)
                    if len(newest_turns) >= limit:
                        break
        except FileNotFoundError:
            return []
        return list(reversed(newest_turns[:limit]))

    def list_sessions(self, limit: int = 50) -> list[dict[str, Any]]:
        if not self.root.exists():
            return []
        entries: list[tuple[float, dict[str, Any]]] = []
        for path in self.root.glob("*.jsonl"):
            try:
                self.validate_id(path.stem)
                turns = self.get(path.stem)
                first_user = next((t["text"] for t in turns if t.get("role") == "user"), "")
                if turns:
                    first_ts = float(turns[0]["ts"])
                    last_ts = float(turns[-1]["ts"])
                else:
                    first_ts = last_ts = path.stat().st_mtime
                entries.append(
                    (
                        last_ts,
                        {
                            "session_id": path.stem,
                            "first_ts": iso_ts(first_ts),
                            "last_ts": iso_ts(last_ts),
                            "turn_count": len(turns),
                            "label": first_user,
                        },
                    )
                )
            except (OSError, ValueError):
                continue
        entries.sort(key=lambda e: e[0], reverse=True)
        return [row for _, row in entries[:limit]]

    def delete(self, session_id: str) -> bool:
        path = self._path(session_id)
        if path.exists():
            path.unlink()
            return True
        return False

    def delete_all(self) -> int:
        count = 0
        if self.root.exists():
            for path in self.root.glob("*.jsonl"):
                path.unlink()
                count += 1
        return count

    def recent_queries(self, limit: int = 8) -> list[str]:
        queries: list[str] = []
        for row in self.list_sessions(limit=limit * 3):
            try:
                turns = self.get(row["session_id"])
            except (OSError, ValueError):
                continue
            for turn in reversed(turns):
                if turn.get("role") == "user" and turn.get("text"):
                    if turn["text"] not in queries:
                        queries.append(turn["text"])
                    break
        return queries[:limit]
