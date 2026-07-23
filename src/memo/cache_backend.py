"""Memo-native backing store for the optional bounded-cache mode.

The durable Markdown vault remains Memo's normal source of truth.  Operators
who explicitly enable cache mode can use a separate, content-addressed archive
under Memo's own state directory.  It needs no daemon, binary, network, or
third-party schema.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from memo.atomic_io import atomic_write_text, authority_write_lock

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


class NullBackend:
    """Safe no-op backend: dirty entries are never considered flushed."""

    def push(self, record: Any) -> bool:
        return False

    def fetch(self, query: str, *, limit: int = 10) -> list[dict[str, Any]]:
        return []


class NativeVaultBackend:
    """A dependency-free, content-addressed archive owned by Memo."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def _path(self, record_id: str) -> Path:
        safe_id = "".join(ch for ch in record_id if ch.isalnum() or ch in {"-", "_"})
        if not safe_id:
            raise ValueError("cache archive record id cannot be empty")
        return self.root / safe_id[:2] / f"{safe_id}.json"

    def push(self, record: Any) -> bool:
        record_id = str(getattr(record, "id", "") or "")
        body = str(getattr(record, "body", "") or "")
        title = str(getattr(record, "title", "") or "")
        if not record_id or not (body.strip() or title.strip()):
            return False
        payload = {
            "id": record_id,
            "title": title,
            "type": str(getattr(record, "type", "") or "note"),
            "body": body,
            "tags": list(getattr(record, "tags", None) or ()),
            "created": str(getattr(record, "created", "") or ""),
            "updated": str(getattr(record, "updated", "") or ""),
            "extra": dict(getattr(record, "extra", None) or {}),
            "schema": "memo.cache_archive.v1",
        }
        path = self._path(record_id)
        with authority_write_lock(self.root):
            atomic_write_text(
                path,
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            )
        return True

    def fetch(self, query: str, *, limit: int = 10) -> list[dict[str, Any]]:
        query_tokens = {token.casefold() for token in _TOKEN_RE.findall(query)}
        if not query_tokens or not self.root.is_dir():
            return []
        matches: list[dict[str, Any]] = []
        for path in self.root.glob("*/*.json"):
            if path.is_symlink():
                continue
            try:
                row = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError, TypeError):
                continue
            haystack = " ".join(
                [
                    str(row.get("title") or ""),
                    str(row.get("body") or ""),
                    " ".join(str(tag) for tag in row.get("tags") or ()),
                ]
            )
            tokens = {token.casefold() for token in _TOKEN_RE.findall(haystack)}
            overlap = len(query_tokens & tokens)
            if not overlap:
                continue
            row["score"] = overlap / max(1, len(query_tokens))
            row["from_backend"] = True
            matches.append(row)
        matches.sort(
            key=lambda row: (float(row.get("score") or 0.0), str(row.get("updated") or "")),
            reverse=True,
        )
        return matches[: max(0, limit)]


def make_backend(backend: str, *, root: Path | None = None) -> Any:
    name = (backend or "none").strip().lower()
    if name in {"vault", "native"} and root is not None:
        return NativeVaultBackend(root)
    return NullBackend()


__all__ = ["NativeVaultBackend", "NullBackend", "make_backend"]
