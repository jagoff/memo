"""Per-session chat history as one JSONL file per session id."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def iso_ts(ts: float) -> str:
    """Format an epoch-seconds timestamp the way the rest of chat/ does."""
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(ts))


class SessionStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def _path(self, session_id: str) -> Path:
        if not _SESSION_ID_RE.match(session_id or ""):
            raise ValueError(f"invalid session id: {session_id!r}")
        return self.root / f"{session_id}.jsonl"

    def append_turn(self, session_id: str, role: str, text: str) -> None:
        path = self._path(session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(
                json.dumps({"role": role, "text": text, "ts": time.time()}, ensure_ascii=False)
                + "\n"
            )

    def get(self, session_id: str) -> list[dict[str, Any]]:
        path = self._path(session_id)
        if not path.exists():
            return []
        turns = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                turns.append(parsed)
        return turns

    def list_sessions(self, limit: int = 50) -> list[dict[str, Any]]:
        if not self.root.exists():
            return []
        entries = []
        for path in self.root.glob("*.jsonl"):
            turns = self.get(path.stem)
            first_user = next((t["text"] for t in turns if t.get("role") == "user"), "")
            first_ts = turns[0]["ts"] if turns else path.stat().st_mtime
            last_ts = turns[-1]["ts"] if turns else path.stat().st_mtime
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
            for turn in reversed(self.get(row["session_id"])):
                if turn.get("role") == "user" and turn.get("text"):
                    if turn["text"] not in queries:
                        queries.append(turn["text"])
                    break
        return queries[:limit]
