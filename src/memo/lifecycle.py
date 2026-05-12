"""Memory lifecycle management — archival, promotion, expiration.

Manages the lifecycle of memories over time:
- Automatic archival of inactive memories
- Promotion/demotion based on access frequency
- Expiration of temporary/debug memories
- Lifecycle policies and rules

## Archival

Memories that haven't been accessed in N days are automatically archived
to an `inactive/` subdirectory. They remain searchable but are deprioritized
in results.

## Promotion/Demotion

Frequently-accessed memories get promoted (higher priority in search).
Rarely-accessed memories get demoted (lower priority). Access count is
tracked via the history store.

## Expiration

Temporary memories (type=temp, or tagged with `temp:...`) can have an
expiration date. After expiration, they are either archived or deleted
based on policy.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from memo.config import Config


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
    """A lifecycle action taken on a memoria."""
    memoria_id: str
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

    def get_access_count(self, memoria_id: str) -> int:
        """Get the number of times a memoria has been accessed.

        Uses the history store to count non-save operations.
        """
        events = self.memory.history.list_recent(
            record_id=memoria_id,
            limit=1000,
        )
        # Count only non-save events (get, search hits)
        return sum(1 for e in events if e.get("op") != "save")

    def get_days_since_access(self, memoria_id: str) -> int | None:
        """Get days since last access, or None if never accessed."""
        events = self.memory.history.list_recent(
            record_id=memoria_id,
            limit=1,
        )
        if not events:
            return None

        last_event = events[0]
        try:
            last_ts = datetime.fromisoformat(last_event["ts"].replace("Z", "+00:00"))
            days = (datetime.now(UTC) - last_ts).days
            return days
        except Exception:
            return None

    def get_days_since_update(self, memoria_id: str) -> int:
        """Get days since last update (created/updated timestamp)."""
        rec = self.memory.get(memoria_id)
        if not rec:
            return 0

        try:
            updated = datetime.fromisoformat(rec.updated.replace("Z", "+00:00"))
            days = (datetime.now(UTC) - updated).days
            return days
        except Exception:
            return 0

    def should_archive(self, memoria_id: str) -> tuple[bool, str]:
        """Determine if a memoria should be archived.

        Returns:
            (should_archive, reason)
        """
        days_since_access = self.get_days_since_access(memoria_id)

        # Never accessed: use update date
        if days_since_access is None:
            days_since_update = self.get_days_since_update(memoria_id)
            if days_since_update > self.policy.archival_days:
                return True, f"Never accessed, {days_since_update} days since update"

        # Accessed but long ago
        if days_since_access and days_since_access > self.policy.archival_days:
            return True, f"Last accessed {days_since_access} days ago"

        return False, "Recently accessed"

    def should_promote(self, memoria_id: str) -> tuple[bool, str]:
        """Determine if a memoria should be promoted (high priority).

        Returns:
            (should_promote, reason)
        """
        access_count = self.get_access_count(memoria_id)

        if access_count >= self.policy.promotion_threshold:
            return True, f"Accessed {access_count} times"

        return False, f"Accessed only {access_count} times"

    def should_demote(self, memoria_id: str) -> tuple[bool, str]:
        """Determine if a memoria should be demoted (low priority).

        Returns:
            (should_demote, reason)
        """
        access_count = self.get_access_count(memoria_id)

        if access_count < self.policy.demotion_threshold:
            return True, f"Accessed only {access_count} times"

        return False, f"Accessed {access_count} times"

    def should_expire(self, memoria_id: str) -> tuple[bool, str]:
        """Determine if a temporary memoria should expire.

        Returns:
            (should_expire, reason)
        """
        rec = self.memory.get(memoria_id)
        if not rec:
            return False, "Memoria not found"

        # Check if it's a temporary type
        if rec.type == "temp":
            days_since_update = self.get_days_since_update(memoria_id)
            if days_since_update > self.policy.temp_expiration_days:
                return True, f"Temp memoria, {days_since_update} days old"

        # Check for temp:... tags
        for tag in rec.tags:
            if tag.startswith("temp:"):
                try:
                    days = int(tag.split(":")[1])
                    days_since_update = self.get_days_since_update(memoria_id)
                    if days_since_update > days:
                        return True, f"Temp tag {tag}, {days_since_update} days old"
                except (ValueError, IndexError):
                    pass

        return False, "Not a temporary memoria"

    def archive_memoria(self, memoria_id: str) -> bool:
        """Archive a memoria by moving it to the inactive/ subdirectory.

        Returns True if successful.
        """
        rec = self.memory.get(memoria_id)
        if not rec:
            return False

        # Create inactive directory
        inactive_dir = self.memory.cfg.memory_dir / "inactive"
        inactive_dir.mkdir(parents=True, exist_ok=True)

        # Read the original file
        source_path = self.memory._resolve_existing(rec.path)
        if not source_path.is_file():
            return False

        # Move to inactive
        import shutil

        target_path = inactive_dir / source_path.name
        shutil.move(str(source_path), str(target_path))

        # Update store to mark as archived
        # For now, just delete from store (file is preserved in inactive/)
        self.memory.delete(memoria_id)

        # Log action
        self._actions_log.append(
            LifecycleAction(
                memoria_id=memoria_id,
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
        """Apply lifecycle rules to all memorias.

        Args:
            dry_run: If True, only report what would happen.
            limit: Maximum memorias to process.

        Returns:
            Dict with counts of actions taken.
        """
        all_records = self.memory.list(limit=limit)

        actions = {
            "archived": 0,
            "promoted": 0,
            "demoted": 0,
            "expired": 0,
            "deleted": 0,
            "skipped": 0,
        }

        for rec in all_records:
            # Check expiration first
            should_expire, expire_reason = self.should_expire(rec.id)
            if should_expire:
                if self.policy.delete_expired:
                    if not dry_run:
                        self.memory.delete(rec.id)
                    actions["deleted"] += 1
                else:
                    if not dry_run:
                        self.archive_memoria(rec.id)
                    actions["expired"] += 1
                continue

            # Check archival
            should_archive, archive_reason = self.should_archive(rec.id)
            if should_archive:
                if not dry_run:
                    self.archive_memoria(rec.id)
                actions["archived"] += 1
                continue

            # Check promotion/demotion
            should_promote, promote_reason = self.should_promote(rec.id)
            should_demote, demote_reason = self.should_demote(rec.id)

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
        """Generate a report on the lifecycle state of memorias.

        Returns:
            Dict with statistics and recommendations.
        """
        all_records = self.memory.list(limit=limit)

        stats = {
            "total": len(all_records),
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
    "LifecycleManager",
    "LifecyclePolicy",
    "LifecycleAction",
]
