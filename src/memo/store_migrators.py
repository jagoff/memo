"""Mem0/Zep dump → memo import-record mappers.

Adoption funnel: users migrating from hosted memory services get their
facts into memo's markdown-is-truth store via the existing
`Importer.import_records` machinery. Pure functions, no I/O beyond the
caller-parsed JSON.
"""

from __future__ import annotations

from typing import Any

__all__ = ["mem0_to_import_records", "zep_to_import_records"]


def _title_of(content: str) -> str:
    return content.splitlines()[0][:80]


def mem0_to_import_records(data: Any) -> list[dict[str, Any]]:
    """Map a Mem0 export / `get_all` dump to memo import records.

    Accepts a bare list or a {"results"|"memories": [...]} envelope. Each
    item's `memory` (fallback `text`/`data`) becomes content; `categories`
    become tags; items are typed `fact` (Mem0 stores extracted facts) and
    tagged `imported:mem0`.
    """
    items = (data.get("results") or data.get("memories")) if isinstance(data, dict) else data
    out: list[dict[str, Any]] = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        content = str(item.get("memory") or item.get("text") or item.get("data") or "").strip()
        if not content:
            continue
        tags = [str(c) for c in (item.get("categories") or []) if c] + ["imported:mem0"]
        out.append(
            {
                "content": content,
                "title": _title_of(content),
                "type": "fact",
                "tags": tags,
                "created": item.get("created_at"),
            }
        )
    return out


def zep_to_import_records(data: Any) -> list[dict[str, Any]]:
    """Map a Zep facts/edges dump to memo import records.

    Accepts {"facts": [...]}, {"edges": [...]} or a bare list. Items whose
    `invalid_at`/`expired_at` is set are skipped — Zep already invalidated
    them, and importing would resurrect superseded knowledge (memo's
    contradiction machinery treats imports as live facts).
    """
    items = (data.get("facts") or data.get("edges")) if isinstance(data, dict) else data
    out: list[dict[str, Any]] = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        if item.get("invalid_at") or item.get("expired_at"):
            continue
        content = str(item.get("fact") or item.get("content") or "").strip()
        if not content:
            continue
        out.append(
            {
                "content": content,
                "title": _title_of(content),
                "type": "fact",
                "tags": ["imported:zep"],
                "created": item.get("created_at") or item.get("valid_at"),
            }
        )
    return out
