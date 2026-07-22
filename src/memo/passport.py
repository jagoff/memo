"""GC-11 Memory Passport — a versioned, vendor-neutral snapshot of the durable
corpus that another memo (or another tool) can consume.

`import_export.py` already round-trips memo→memo via bare JSON/CSV/markdown.
The passport adds three things that make it a *portable brain* rather than a
dump: a stable ``schema`` header, validation on import, and semantic fidelity —
type, tags, and the ``extra`` bag (which carries provenance + verification
state, see ``record._PROVENANCE_KEYS``).

Fidelity contract (honest v1): the passport carries the **canonical** record —
what markdown is the source of truth for. Derived indexes (embeddings, graph
relations) are rebuilt by the receiving store on import + reindex; ids and
``updated`` are regenerated on the target. Everything a human authored —
content, title, type, tags, created, and the provenance/verification ``extra``
— round-trips. Embeddings are deliberately out of v1 (rebuildable; keep the
passport small and vendor-neutral).

Pure functions over plain dicts / attribute objects; the Exporter/Importer in
``import_export.py`` wire the real store.
"""

from __future__ import annotations

from typing import Any

from memo.errors import ValidationError

PASSPORT_SCHEMA = "memo.passport.v1"

_ENTRY_FIELDS = ("id", "type", "title", "body", "tags", "created", "updated")


def entry_from_record(rec: Any) -> dict[str, Any]:
    """Map a ``MemoryRecord`` (or any attribute object) to a passport entry."""
    entry: dict[str, Any] = {f: getattr(rec, f, None) for f in _ENTRY_FIELDS}
    entry["tags"] = list(entry.get("tags") or [])
    entry["extra"] = dict(getattr(rec, "extra", None) or {})
    return entry


def build_passport(
    entries: list[dict[str, Any]], *, generator: str, exported_at: str
) -> dict[str, Any]:
    """Wrap passport entries in the versioned envelope. Pure — no I/O, no clock."""
    return {
        "schema": PASSPORT_SCHEMA,
        "generator": generator,
        "exported_at": exported_at,
        "count": len(entries),
        "memories": list(entries),
    }


def validate_passport(obj: Any) -> None:
    """Raise :class:`ValidationError` unless ``obj`` is a v1 passport.

    Conservative on shape (schema + a list of dict entries) but lenient on the
    per-entry fields — a future minor may add keys, and import already tolerates
    missing optional fields.
    """
    if not isinstance(obj, dict):
        raise ValidationError(f"passport must be an object, got {type(obj).__name__}")
    schema = obj.get("schema")
    if schema != PASSPORT_SCHEMA:
        raise ValidationError(f"unsupported passport schema: {schema!r} (want {PASSPORT_SCHEMA!r})")
    memories = obj.get("memories")
    if not isinstance(memories, list):
        raise ValidationError("passport 'memories' must be a list")
    for i, entry in enumerate(memories):
        if not isinstance(entry, dict):
            raise ValidationError(f"passport memory[{i}] must be an object")


def normalize_for_import(entry: dict[str, Any]) -> dict[str, Any]:
    """Map a passport entry to the ``import_records`` / ``save`` kwargs shape.

    ``body`` becomes ``content``; ``extra`` (provenance + verification) is
    carried through so the receiving store preserves it verbatim.
    """
    return {
        "content": entry.get("body") or entry.get("content") or "",
        "title": entry.get("title") or "",
        "tags": list(entry.get("tags") or []),
        "type": entry.get("type") or "note",
        "created": entry.get("created"),
        "extra": dict(entry.get("extra") or {}) or None,
    }
