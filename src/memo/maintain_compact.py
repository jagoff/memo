"""Quality-compaction core for `memo maintain` — receipts, rollback, restore.

Pure-ish helpers over config/state paths and `Memory` handles: receipt
placement (fail-closed), run identity, undo/rollback targets, and index/archive
restoration. No click wiring — extracted from `cli_maintain.py` so the CLI
module keeps only command plumbing; `cli_maintain` re-exports these names so
importers (`tests`, `dream_profile`) work unchanged.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from memo.config import Config
from memo.errors import StorageError, ValidationError
from memo.util import safe_operation

_log = logging.getLogger(__name__)

_RUN_STAMP_RE = re.compile(r"\d{1,20}(?:-\d{1,20}-\d{1,30})?\Z")


def _state_path(cfg: Config):
    return cfg.state_dir / "maintain"

def _prepare_quality_compact_receipt_paths(cfg: Config) -> tuple[Path, Path, str]:
    """Fail closed if apply-mode receipt persistence cannot be set up."""

    d = _state_path(cfg)
    runs_dir = d / "runs"
    run_stamp = f"{int(time.time())}-{os.getpid()}-{time.time_ns()}"
    try:
        runs_dir.mkdir(parents=True, exist_ok=True)
        probe = d / ".quality_compact_receipt_probe"
        probe.write_text(run_stamp, encoding="utf-8")
        probe.unlink(missing_ok=True)
    except OSError as exc:
        raise ValidationError(f"quality compaction receipt setup failed: {exc}") from exc
    return d, runs_dir, run_stamp

def _persist_quality_compact_receipt(
    d: Path,
    runs_dir: Path,
    run_stamp: str,
    payload: str,
) -> None:
    """Publish run + last receipts without exposing a rolled-back last.json."""

    run_path = runs_dir / f"{run_stamp}.json"
    last_path = d / "last.json"
    staged: dict[Path, Path] = {}
    published_run = False
    try:
        for path in (run_path, last_path):
            tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
            tmp.write_text(payload, encoding="utf-8")
            staged[path] = tmp
        os.replace(staged[run_path], run_path)
        published_run = True
        os.replace(staged[last_path], last_path)
    except OSError:
        for tmp in staged.values():
            tmp.unlink(missing_ok=True)
        if published_run:
            run_path.unlink(missing_ok=True)
        raise

def _synthesis_state_path(cfg: Config) -> Path:
    return cfg.state_dir / "synthesis_state.json"


@safe_operation(
    fallback=None,
    log_level=logging.DEBUG,
    error_message="maintain: could not read synthesis_state.json",
)

def _read_synthesis_last_run(cfg: Config) -> str | None:
    """Return the ISO timestamp of the last synthesis run, or None."""
    p = _synthesis_state_path(cfg)
    data = json.loads(p.read_text(encoding="utf-8"))
    return data.get("last_run") or None

def _write_synthesis_last_run(cfg: Config, ts: str) -> None:
    """Persist the synthesis run timestamp to state_dir/synthesis_state.json."""
    p = _synthesis_state_path(cfg)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"last_run": ts}, ensure_ascii=False), encoding="utf-8")
    except Exception as exc:
        _log.warning("maintain: could not write synthesis_state.json: %s", exc)

def _older_id(mem: Any, id_a: str, id_b: str) -> tuple[str, str]:
    """Return (older_id, newer_id) by `updated` timestamp; falls back to the
    pair order (a, b) when a record or timestamp is missing."""
    ra, rb = mem.get(id_a), mem.get(id_b)
    ua = getattr(ra, "updated", "") or ""
    ub = getattr(rb, "updated", "") or ""
    if ua and ub:
        return (id_a, id_b) if ua <= ub else (id_b, id_a)
    return id_a, id_b

def _undo_targets(receipt: dict[str, Any]) -> tuple[list[str], list[str], list[str]]:
    """(archived_ids, soft_forgotten_ids, invalidated_ids) recorded in a receipt.

    `invalidated_ids` are contradiction losers closed in place by
    `invalidate_in_place` (action='invalidate') — the loser's .md was never
    moved to inactive/, so undo REOPENS its interval instead of restoring a
    file (see `_reopen_invalidated`)."""
    archived: list[str] = []
    invalidated: list[str] = []
    for s in receipt.get("superseded", []):
        if not (isinstance(s, dict) and s.get("older")):
            continue
        if s.get("action") == "archive":
            archived.append(s["older"])
        elif s.get("action") == "invalidate":
            invalidated.append(s["older"])
    for m in receipt.get("merged", []):
        if isinstance(m, dict):
            archived.extend(m.get("archived_ids") or [])
    for a in receipt.get("archived_stale", []):
        if isinstance(a, dict) and a.get("id"):
            archived.append(a["id"])
    for q in receipt.get("quality_compacted", []):
        if isinstance(q, dict):
            archived.extend(q.get("archived_ids") or [])
    forgotten: list[str] = [x for x in receipt.get("dead_archived", []) if isinstance(x, str)]
    for f in receipt.get("forgotten", []):
        if isinstance(f, dict) and f.get("id"):
            forgotten.append(f["id"])
    return archived, forgotten, invalidated

def _quality_compact_rollback_ids(receipt: dict[str, Any]) -> list[str]:
    """Conservatively include every attempted archive when rolling back apply."""

    rollback_ids: list[str] = []
    seen: set[str] = set()
    for item in receipt.get("quality_compacted", []):
        if not isinstance(item, dict):
            continue
        for value in item.get("attempted_ids") or item.get("archived_ids") or []:
            source_id = str(value or "")
            if not source_id or source_id in seen:
                continue
            seen.add(source_id)
            rollback_ids.append(source_id)
    return rollback_ids

def _rollback_quality_compaction(mem: Any, receipt: dict[str, Any]) -> tuple[list[str], list[str]]:
    rollback_ids = _quality_compact_rollback_ids(receipt)
    restored, missing = _restore_archived(mem, rollback_ids, dry_run=False)
    indexed, index_failed = _restore_quality_compact_indexes(mem, restored)
    missing_all = sorted(set(missing) | set(index_failed))
    return indexed, missing_all

def _restore_quality_compact_indexes(
    mem: Any,
    restored_ids: list[str],
) -> tuple[list[str], list[str]]:
    """Restore rollback records to the text index without loading a model.

    Receipt persistence is the last step of quality compaction.  If it fails,
    rollback must remain possible on a host that has no ML dependency or model
    cache.  Reindexing here used to make the rollback itself depend on an
    embedder and could leave the Markdown restored but the memory invisible.
    Text-only upserts make the record immediately available to CRUD/BM25 and
    mark it for a later semantic reindex.
    """

    import frontmatter

    from memo.util import sha256_short

    indexed: list[str] = []
    failed: list[str] = []
    for memory_id in restored_ids:
        path = mem.cfg.memory_dir / f"{memory_id}.md"
        try:
            post = frontmatter.loads(path.read_text(encoding="utf-8"))
            meta = post.metadata
            body = post.content or ""
            raw_extra = meta.get("extra")
            extra = dict(raw_extra) if isinstance(raw_extra, dict) else {}
            # archive_memory stamps these fields after moving the file.  A
            # rollback means that archival never committed, so the stamps must
            # not survive.
            extra.pop("superseded_by", None)
            extra.pop("superseded_at", None)
            extra["_memo_embed_pending"] = True
            post.metadata["extra"] = extra
            path.write_text(frontmatter.dumps(post), encoding="utf-8")

            raw_tags = meta.get("tags") or []
            if isinstance(raw_tags, str):
                tags = [tag.strip() for tag in raw_tags.split(",") if tag.strip()]
            elif isinstance(raw_tags, (list, tuple, set)):
                tags = [str(tag) for tag in raw_tags]
            else:
                tags = []

            now = datetime.now(UTC).isoformat()

            def _iso(value: Any, fallback: str) -> str:
                if value is None:
                    return fallback
                isoformat = getattr(value, "isoformat", None)
                return str(isoformat() if callable(isoformat) else value)

            rel = path.relative_to(mem.cfg.memory_dir).as_posix()
            mem.store.upsert_text_only(
                id_=memory_id,
                path=rel,
                title=str(meta.get("title") or memory_id[:8]),
                type_=str(meta.get("type") or "note"),
                tags=tags,
                created=_iso(meta.get("created"), now),
                updated=_iso(meta.get("updated"), now),
                body_hash=sha256_short(body),
                extra=extra,
                body_text=body,
            )
        except (OSError, StorageError, TypeError, ValueError) as exc:
            _log.warning(
                "quality compact rollback: could not restore index for %s: %s", memory_id, exc
            )
            failed.append(memory_id)
            continue
        indexed.append(memory_id)
    return indexed, failed

def _restore_archived(mem: Any, ids: list[str], *, dry_run: bool) -> tuple[list[str], list[str]]:
    """Move receipt-listed .md files back out of inactive/ (matched by
    frontmatter id — the receipt stores ids, not filenames). Returns
    (restored_ids, missing_ids). Caller runs `mem.reindex()` afterwards."""
    import shutil

    import frontmatter

    inactive_dir = mem.cfg.memory_dir / "inactive"
    wanted = set(ids)
    restored: list[str] = []
    if inactive_dir.is_dir():
        for p in sorted(inactive_dir.glob("*.md")):
            try:
                fid = str(frontmatter.load(p).get("id") or "")
            except Exception:  # noqa: S112 — a corrupt archived file must not sink the restore
                continue
            if fid not in wanted:
                continue
            dest = mem.cfg.memory_dir / p.name
            if dest.exists():
                _log.warning("maintain undo: %s exists; skipping restore of %s", dest, fid[:8])
                continue
            if not dry_run:
                shutil.move(str(p), str(dest))
            restored.append(fid)
    return restored, sorted(wanted - set(restored))
