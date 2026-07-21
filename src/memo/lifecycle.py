"""Memory lifecycle management — forgetting, archival, expiration.

Internal maintenance layer driven by `memo maintain` (NOT exposed on the
public CLI or MCP surface — see tests/test_architecture_boundaries.py;
memo is the memory store, orchestration lives in Synapse). The user-facing
primitives are `Memory.forget()` / `Memory.unforget()` (CRUD-like, exposed
as the `memo_forget` / `memo_unforget` MCP tools).

Manages the lifecycle of memories over time:
- Explicit forget with a TTL (`forget_after`) — soft, reversible
- Automatic archival of inactive memories
- Expiration of temporary/debug memories
- Promotion/demotion reporting based on access frequency

## Forgetting (soft, reversible) — `forget_after` TTL

A memory can carry `forget_after` (ISO date) and `forget_reason` in its
`extra` bag / frontmatter. Once the date passes, `apply_lifecycle_rules`
soft-forgets it: sets `extra.is_forgotten = True` so it drops out of
`search` / recall / `list` by default, WITHOUT moving or deleting the file.
Fully reversible via `Memory.unforget()`. This is the supermemory-style
`isForgotten` model — distinct from the harder archival below.

## Archival

Memories that haven't been accessed in N days are archived to an
`inactive/` subdirectory and removed from the index (recoverable from disk).

## Promotion/Demotion

Access counts (from the history store) drive a report of promotion/demotion
candidates. Advisory only — no priority flag is written today.

## Expiration

Temporary memories (type=temp, or tagged with `temp:...`) expire after a
window and are archived or deleted based on policy.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

_log = logging.getLogger(__name__)

# Keys carried in a memory's `extra` bag (persisted to meta.extra_json and to
# the on-disk frontmatter `extra:` block, so they survive a reindex for free).
FORGET_AFTER_KEY = "forget_after"  # ISO date/datetime; soft-forget once passed
FORGET_REASON_KEY = "forget_reason"  # free-text why
IS_FORGOTTEN_KEY = "is_forgotten"  # bool; excluded from search/recall/list


def _parse_iso_date(value: Any) -> datetime | None:
    """Best-effort parse of a forget_after value to an aware datetime.

    Accepts full ISO datetimes and bare `YYYY-MM-DD` dates (interpreted as
    end-of-day UTC so a same-day `forget_after` doesn't fire prematurely).
    Returns None on anything unparseable.
    """
    if not value:
        return None
    s = str(value).strip()
    try:
        if len(s) == 10:  # YYYY-MM-DD
            dt = datetime.fromisoformat(s).replace(
                hour=23,
                minute=59,
                second=59,
                tzinfo=UTC,
            )
        else:
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)
        return dt
    except (ValueError, TypeError):
        return None


@dataclass
class LifecyclePolicy:
    """Policy rules for memory lifecycle management."""

    archival_days: int = 180  # Days of inactivity before archival
    promotion_threshold: int = 5  # Access count to promote
    demotion_threshold: int = 2  # Access count below which to demote
    temp_expiration_days: int = 30  # Default expiration for temp memories
    delete_expired: bool = False  # If True, delete expired; if False, archive


@dataclass
class LifecycleAction:
    """A lifecycle action taken on a memory."""

    memory_id: str
    action: str  # archive, promote, demote, expire, delete
    reason: str
    timestamp: str


class LifecycleManager:
    """Manages memory lifecycle based on access patterns and policies.

    Args:
        memory: The Memory instance to operate on.
        policy: Optional LifecyclePolicy. Defaults to sensible defaults.
    """

    def __init__(self, memory: Any, policy: LifecyclePolicy | None = None) -> None:
        self.memory = memory
        self.policy = policy or LifecyclePolicy()
        self._actions_log: list[LifecycleAction] = []

    def get_access_count(self, memory_id: str) -> int:
        """Get the number of times a memory has been read/hit.

        Reads the `access` table (populated by `store.touch()` on every
        search/ask hit). Falls back to counting non-save history events when
        the store doesn't expose access tracking (older stores / test doubles).
        """
        store = getattr(self.memory, "store", None)
        if store is not None and hasattr(store, "get_access"):
            return int(store.get_access(memory_id).get("access_count", 0))
        events = self.memory.history.list_recent(record_id=memory_id, limit=1000)
        return sum(1 for e in events if e.get("op") != "save")

    def get_days_since_access(self, memory_id: str) -> int | None:
        """Get days since last read/hit, or None if never accessed."""
        last_ts_raw: str | None = None
        store = getattr(self.memory, "store", None)
        if store is not None and hasattr(store, "get_access"):
            last_ts_raw = store.get_access(memory_id).get("last_accessed")
        else:
            events = self.memory.history.list_recent(record_id=memory_id, limit=1000)
            access_events = [e for e in events if e.get("op") != "save"]
            if access_events:
                last_ts_raw = access_events[0].get("ts")
        if not last_ts_raw:
            return None
        try:
            last_ts = datetime.fromisoformat(last_ts_raw.replace("Z", "+00:00"))
            return (datetime.now(UTC) - last_ts).days
        except (ValueError, AttributeError, TypeError) as exc:
            _log.debug("lifecycle: bad access ts %r for %s: %s", last_ts_raw, memory_id, exc)
            return None

    def get_days_since_update(self, memory_id: str) -> int:
        """Get days since last update (created/updated timestamp)."""
        rec = self.memory.get(memory_id)
        if not rec:
            return 0

        try:
            updated = datetime.fromisoformat(rec.updated.replace("Z", "+00:00"))
            days = (datetime.now(UTC) - updated).days
            return days
        except (ValueError, AttributeError, TypeError) as exc:
            # Bad timestamp → treated as "0 days" (just-updated), which blocks
            # archival; log so the corrupt record is diagnosable.
            _log.debug("lifecycle: bad updated ts for %s: %s", memory_id, exc)
            return 0

    def should_archive(self, memory_id: str) -> tuple[bool, str]:
        """Determine if a memory should be archived.

        Returns:
            (should_archive, reason)
        """
        days_since_access = self.get_days_since_access(memory_id)

        # Never accessed: use update date
        if days_since_access is None:
            days_since_update = self.get_days_since_update(memory_id)
            if days_since_update > self.policy.archival_days:
                return True, f"Never accessed, {days_since_update} days since update"

        # Accessed but long ago
        if days_since_access and days_since_access > self.policy.archival_days:
            return True, f"Last accessed {days_since_access} days ago"

        return False, "Recently accessed"

    def should_promote(self, memory_id: str) -> tuple[bool, str]:
        """Determine if a memory should be promoted (high priority).

        Returns:
            (should_promote, reason)
        """
        access_count = self.get_access_count(memory_id)

        if access_count >= self.policy.promotion_threshold:
            return True, f"Accessed {access_count} times"

        return False, f"Accessed only {access_count} times"

    def should_demote(self, memory_id: str) -> tuple[bool, str]:
        """Determine if a memory should be demoted (low priority).

        Returns:
            (should_demote, reason)
        """
        access_count = self.get_access_count(memory_id)

        if access_count < self.policy.demotion_threshold:
            return True, f"Accessed only {access_count} times"

        return False, f"Accessed {access_count} times"

    def should_expire(self, memory_id: str) -> tuple[bool, str]:
        """Determine if a temporary memory should expire.

        Returns:
            (should_expire, reason)
        """
        rec = self.memory.get(memory_id)
        if not rec:
            return False, "Memory not found"

        # Check if it's a temporary type
        if rec.type == "temp":
            days_since_update = self.get_days_since_update(memory_id)
            if days_since_update > self.policy.temp_expiration_days:
                return True, f"Temp memory, {days_since_update} days old"

        # Check for temp:... tags
        for tag in rec.tags:
            if tag.startswith("temp:"):
                try:
                    days = int(tag.split(":")[1])
                    days_since_update = self.get_days_since_update(memory_id)
                    if days_since_update > days:
                        return True, f"Temp tag {tag}, {days_since_update} days old"
                except (ValueError, IndexError):
                    pass

        return False, "Not a temporary memory"

    def should_forget(self, memory_id: str) -> tuple[bool, str]:
        """Determine if a memory's `forget_after` TTL has elapsed.

        Returns (should_forget, reason). False for memories already forgotten
        or without a parseable `forget_after`.
        """
        rec = self.memory.get(memory_id)
        if rec is None:
            return False, "Memory not found"
        extra = rec.extra or {}
        if extra.get(IS_FORGOTTEN_KEY):
            return False, "Already forgotten"
        due = _parse_iso_date(extra.get(FORGET_AFTER_KEY))
        if due is None:
            return False, "No forget_after set"
        if datetime.now(UTC) >= due:
            return True, f"forget_after {due.date().isoformat()} elapsed"
        return False, f"forget_after {due.date().isoformat()} not yet reached"

    def enforce_forget_ttl(
        self,
        *,
        dry_run: bool = False,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        """Soft-forget every memory whose `forget_after` TTL has elapsed.

        Narrow counterpart to `apply_lifecycle_rules` — does ONLY the explicit
        forget pass (no inactivity-archival, no temp-expiration), so callers
        like `memo maintain` can enforce user-set TTLs without triggering the
        broader archival policy. Returns the list of `{id, reason}` acted on.
        """
        acted: list[dict[str, Any]] = []
        for rec in self.memory.list(limit=limit):
            should_forget, reason = self.should_forget(rec.id)
            if not should_forget:
                continue
            if not dry_run:
                self.memory.forget(rec.id, reason=reason)
            acted.append({"id": rec.id, "reason": reason})
        return acted

    def archive_memory(self, memory_id: str, *, superseded_by: str | None = None) -> bool:
        """Archive a memory by moving it to the inactive/ subdirectory.

        `superseded_by` (C6): stamp the WINNING memory id + a close-date into
        the archived file's extra bag, so the supersede provenance lives in
        the markdown (portable; the contradict sidecar DB is machine-local).

        Returns True if successful.
        """
        rec = self.memory.get(memory_id)
        if not rec:
            return False

        # Create inactive directory
        inactive_dir = self.memory.cfg.memory_dir / "inactive"
        inactive_dir.mkdir(parents=True, exist_ok=True)

        # Read the original file
        source_path = self.memory._resolve_existing(rec.path)
        if not source_path.is_file():
            return False

        import shutil

        # Step 1: Move the canonical .md to inactive/ FIRST. Until the move
        # succeeds nothing is touched, so a failure loses nothing — the file
        # stays in place and we abort. (Dropping the index first would let
        # Memory.delete() unlink the canonical .md; if the move then failed the
        # memory would be permanently destroyed instead of archived.)
        # Use memory_id as the archive filename (same pattern as consolidation.py)
        # so two different memories with the same title/date basename can never
        # silently overwrite each other in inactive/. source_path.name is NOT
        # safe: archive→recreate→archive on the same day produces the same
        # basename, and shutil.move on POSIX atomically replaces the destination.
        target_path = inactive_dir / f"{memory_id}.md"
        try:
            shutil.move(str(source_path), str(target_path))
        except OSError as exc:
            _log.warning("archive_memory: move failed for %s: %s", memory_id, exc)
            return False

        # Step 2: Drop the searchable index rows only (meta/vec/fts) via the
        # store, NOT the .md — which now lives in inactive/. After this get()
        # returns None (matches test expectations); the file is recoverable by
        # moving it back out of inactive/ and reindexing (reindex deliberately
        # skips inactive/ so archived memories don't auto-resurrect on the
        # post-sync-pull reindex). Going through Memory.delete() here would
        # unlink the moved file's original path (harmlessly) but is also
        # heavier than needed.
        self.memory.store.delete(memory_id)

        # C6: validity-window stamp — best-effort, never un-archives on failure.
        if superseded_by:
            try:
                import frontmatter

                post = frontmatter.loads(target_path.read_text(encoding="utf-8"))
                _raw = post.get("extra")
                extra: dict[str, object] = dict(_raw) if isinstance(_raw, dict) else {}
                extra["superseded_by"] = superseded_by
                extra["superseded_at"] = datetime.now(UTC).isoformat()
                post["extra"] = extra
                target_path.write_text(frontmatter.dumps(post), encoding="utf-8")
            except Exception as exc:
                _log.warning(
                    "archive_memory: superseded_by stamp failed for %s: %s", memory_id, exc
                )

        # Log action
        self._actions_log.append(
            LifecycleAction(
                memory_id=memory_id,
                action="archive",
                reason="Inactive",
                timestamp=datetime.now(UTC).isoformat(),
            )
        )

        return True

    def apply_lifecycle_rules(
        self,
        dry_run: bool = False,
        limit: int = 100,
    ) -> dict[str, Any]:
        """Apply lifecycle rules to all memories.

        Args:
            dry_run: If True, only report what would happen.
            limit: Maximum memories to process.

        Returns:
            Dict with counts of actions taken.
        """
        all_records = self.memory.list(limit=limit)

        actions = {
            "forgotten": 0,
            "archived": 0,
            "promoted": 0,
            "demoted": 0,
            "expired": 0,
            "deleted": 0,
            "skipped": 0,
        }

        for rec in all_records:
            # Explicit forget TTL first — soft, reversible, keeps the file.
            should_forget, forget_reason = self.should_forget(rec.id)
            if should_forget:
                if not dry_run:
                    self.memory.forget(rec.id, reason=forget_reason)
                actions["forgotten"] += 1
                continue

            # Check expiration next
            should_expire, _expire_reason = self.should_expire(rec.id)
            if should_expire:
                if self.policy.delete_expired:
                    if not dry_run:
                        self.memory.delete(rec.id)
                    actions["deleted"] += 1
                else:
                    if not dry_run:
                        self.archive_memory(rec.id)
                    actions["expired"] += 1
                continue

            # Check archival
            should_archive, _archive_reason = self.should_archive(rec.id)
            if should_archive:
                if not dry_run:
                    self.archive_memory(rec.id)
                actions["archived"] += 1
                continue

            # Check promotion/demotion
            should_promote, _promote_reason = self.should_promote(rec.id)
            should_demote, _demote_reason = self.should_demote(rec.id)

            if should_promote:
                actions["promoted"] += 1
                # In a full implementation, would set a priority flag in frontmatter
            elif should_demote:
                actions["demoted"] += 1
                # In a full implementation, would set a low priority flag
            else:
                actions["skipped"] += 1

        return actions

    def get_lifecycle_report(self, limit: int = 100) -> dict[str, Any]:
        """Generate a report on the lifecycle state of memories.

        Returns:
            Dict with statistics and recommendations.
        """
        all_records = self.memory.list(limit=limit)

        stats = {
            "total": len(all_records),
            "forget_candidates": 0,
            "archive_candidates": 0,
            "promotion_candidates": 0,
            "demotion_candidates": 0,
            "expiration_candidates": 0,
            "never_accessed": 0,
            "avg_access_count": 0.0,
        }

        total_access = 0
        for rec in all_records:
            access_count = self.get_access_count(rec.id)
            total_access += access_count

            if access_count == 0:
                stats["never_accessed"] += 1

            should_forget, _ = self.should_forget(rec.id)
            if should_forget:
                stats["forget_candidates"] += 1

            should_archive, _ = self.should_archive(rec.id)
            if should_archive:
                stats["archive_candidates"] += 1

            should_promote, _ = self.should_promote(rec.id)
            if should_promote:
                stats["promotion_candidates"] += 1

            should_demote, _ = self.should_demote(rec.id)
            if should_demote:
                stats["demotion_candidates"] += 1

            should_expire, _ = self.should_expire(rec.id)
            if should_expire:
                stats["expiration_candidates"] += 1

        if len(all_records) > 0:
            stats["avg_access_count"] = total_access / len(all_records)

        return stats


__all__ = [
    "LifecycleAction",
    "LifecycleManager",
    "LifecyclePolicy",
]
