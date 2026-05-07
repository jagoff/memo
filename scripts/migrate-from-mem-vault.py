"""Migrate mem-vault frontmatter → memo schema (in-place).

mem-vault writes .md files with frontmatter shaped like:

    ---
    agent_id: web
    name: Title here
    description: One-line summary
    created: 2026-04-29T04:15:37-03:00
    last_used: 2026-04-30T20:35:58-03:00
    tags: [a, b, c]
    ---
    body

memo expects:

    ---
    id: <uuid4hex>
    title: Title here
    type: decision | fact | bug | feedback | preference | note | manual
    tags: [a, b, c]
    created: 2026-04-29T04:15:37-03:00
    updated: 2026-04-30T20:35:58-03:00
    extra:
      agent_id: web
      description: One-line summary
    ---
    body

Migration rules (idempotent — files already migrated are skipped):

- If `id` already present → skip.
- Generate `id` = uuid4 hex.
- `name` → `title`. If `name` missing, derive title from first body line.
- `last_used` → `updated`. Fallback to `created`. Fallback to now.
- Default `type` = "note" (mem-vault didn't have a type concept).
- Preserve `agent_id` + `description` + any unknown fields under `extra:`.
- Body untouched.

Usage:

    .venv/bin/python scripts/migrate-from-mem-vault.py [--dry-run]

After this, `memo reindex` absorbs all migrated files into the sqlite-vec
index.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import sys
import uuid
from pathlib import Path

import frontmatter

from memo.config import Config


# Fields memo writes natively. Anything else gets stashed in `extra`.
_NATIVE_FIELDS = frozenset({"id", "title", "type", "tags", "created", "updated", "extra"})

# Mapping for known mem-vault fields. Dest=None means "drop" (the field is
# subsumed by another). The order matters: we apply renames before stashing.
_RENAMES = {
    "name": "title",
    "last_used": "updated",
}


def _now_iso() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).astimezone().isoformat(timespec="seconds")


def _coerce_iso(value) -> str:
    """mem-vault stores datetimes as YAML datetime — frontmatter parses
    those back to `datetime` objects. memo expects ISO strings. Normalise."""
    if value is None:
        return _now_iso()
    if isinstance(value, str):
        return value
    if isinstance(value, _dt.datetime):
        if value.tzinfo is None:
            value = value.astimezone()
        return value.isoformat(timespec="seconds")
    if isinstance(value, _dt.date):
        return value.isoformat()
    return str(value)


def migrate_file(path: Path) -> str:
    """Returns one of: 'migrated', 'skipped' (already has id), 'error'."""
    try:
        post = frontmatter.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"  ✗ parse fail: {path.name} ({exc})")
        return "error"

    meta = dict(post.metadata)

    if meta.get("id"):
        return "skipped"

    # Step 1: rename known fields.
    for src, dst in _RENAMES.items():
        if src in meta and dst not in meta:
            meta[dst] = meta.pop(src)
        elif src in meta:
            # Both present — drop the legacy one to avoid duplication.
            del meta[src]

    # Step 2: ensure required memo fields exist.
    if not meta.get("title"):
        # Derive title from first non-empty body line.
        for raw in (post.content or "").splitlines():
            line = raw.strip().lstrip("#*->`\"' ")
            if line:
                meta["title"] = line[:80]
                break
        if not meta.get("title"):
            meta["title"] = path.stem.replace("-", " ")[:80]

    meta["id"] = uuid.uuid4().hex
    meta["created"] = _coerce_iso(meta.get("created"))
    meta["updated"] = _coerce_iso(meta.get("updated") or meta.get("created"))
    meta["type"] = meta.get("type") or "note"
    meta["tags"] = list(meta.get("tags") or [])

    # Step 3: stash unknown fields under `extra:`.
    extra = dict(meta.get("extra") or {})
    for k in list(meta.keys()):
        if k in _NATIVE_FIELDS:
            continue
        extra[k] = meta.pop(k)
    if extra:
        meta["extra"] = extra

    new_post = frontmatter.Post(post.content or "", **meta)
    path.write_text(frontmatter.dumps(new_post), encoding="utf-8")
    return "migrated"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="Walk + report counts but don't write any files.")
    args = ap.parse_args()

    cfg = Config.from_env()
    memory_dir = cfg.memory_dir
    if not memory_dir.is_dir():
        print(f"Memory dir does not exist: {memory_dir}", file=sys.stderr)
        return 1

    counts = {"migrated": 0, "skipped": 0, "error": 0}
    for md in sorted(memory_dir.rglob("*.md")):
        if args.dry_run:
            try:
                post = frontmatter.loads(md.read_text(encoding="utf-8"))
            except Exception:
                counts["error"] += 1
                continue
            counts["skipped" if post.get("id") else "migrated"] += 1
        else:
            counts[migrate_file(md)] += 1

    verb = "would migrate" if args.dry_run else "migrated"
    print(f"{verb}: {counts['migrated']}  skipped (already had id): {counts['skipped']}  errors: {counts['error']}")
    if not args.dry_run:
        print("\nRun `memo reindex` to absorb the migrated files into the index.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
