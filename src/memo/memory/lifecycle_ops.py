from __future__ import annotations

import contextlib
import logging
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import frontmatter

from memo.errors import NotFoundError, StorageError, ValidationError
from memo.memory._base import _MemoryBase
from memo.memory.record import MemoryRecord, _now_iso
from memo.project import slugify_project
from memo.redact import sanitize_persisted_text
from memo.tiers import DURABLE_TYPES, VerificationState

_log = logging.getLogger("memo.memory.record")


def review_interval_days(type_: str, tags: list[str]) -> int | None:
    """Pure initial review policy; mutable configuration takes precedence."""
    normalized = {str(tag).strip().lower() for tag in tags}
    if type_ == "preference" or (type_ in DURABLE_TYPES and "config" in normalized):
        return 90
    if type_ == "decision":
        return 180
    if type_ in DURABLE_TYPES and normalized & {"policy", "architecture"}:
        return 365
    return None


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed


def _next_review(record: MemoryRecord, *, base: datetime | None = None) -> str | None:
    interval = review_interval_days(record.type, record.tags)
    if interval is None:
        return None
    anchor = base or _parse_datetime(record.updated) or datetime.now(UTC)
    return (anchor + timedelta(days=interval)).isoformat()


def _validity_iso(value: str | None) -> str | None:
    parsed = _parse_datetime(value)
    return parsed.astimezone().isoformat(timespec="milliseconds") if parsed else value


class _LifecycleOpsMixin(_MemoryBase):
    """Explicit validity, review, and verification operations."""

    def ensure_review_schedule(self, record: MemoryRecord) -> MemoryRecord:
        if record.review_after is not None:
            return record
        review_after = _next_review(record)
        if review_after is None:
            return record
        try:
            self._set_review_metadata(
                record.id,
                review_after=review_after,
                verification_state=record.verification_state,
                verified_at=record.verified_at,
            )
        except Exception:
            # The canonical save is already committed. Reindex/doctor can
            # recover missing derived schedule metadata without falsifying it.
            return record
        return replace(record, review_after=review_after)

    def list_due_reviews(
        self,
        *,
        project: str | None = None,
        limit: int = 50,
        now: str | None = None,
    ) -> list[dict[str, Any]]:
        namespace = f"project:{slugify_project(project)}" if project else None
        return self.store.list_due_reviews(
            now=now or datetime.now(UTC).isoformat(),
            namespace=namespace,
            limit=limit,
        )

    def mark_reviewed(
        self,
        id_: str,
        *,
        evidence: str | None = None,
        actor: str | None = None,
        reviewed_at: str | None = None,
    ) -> MemoryRecord:
        record = self.get(id_)
        if record is None:
            raise NotFoundError(f"memory not found: {id_}")
        reviewed_dt = _parse_datetime(reviewed_at) or datetime.now(UTC)
        safe_evidence = sanitize_persisted_text(evidence or "").text or None
        # Treat a replay of the latest still-current review as a no-op. This
        # makes CLI/MCP retries idempotent without preventing a later review
        # once the record is due again.
        current_due = _parse_datetime(record.review_after)
        latest = self.store.review_evidence(record.id, limit=1)
        if (
            record.verification_state == VerificationState.VERIFIED
            and current_due is not None
            and current_due > reviewed_dt
            and latest
            and latest[0].get("evidence") == safe_evidence
            and latest[0].get("actor") == actor
        ):
            return record
        next_review = _next_review(record, base=reviewed_dt)
        self._set_review_metadata(
            record.id,
            review_after=next_review,
            verification_state=VerificationState.VERIFIED,
            verified_at=int(reviewed_dt.timestamp()),
            evidence=safe_evidence,
            actor=actor,
            reviewed_at=reviewed_dt.isoformat(),
        )
        updated = self.get(record.id)
        if updated is None:  # pragma: no cover - authoritative row vanished
            raise StorageError(f"reviewed memory disappeared: {record.id[:8]}")
        return updated

    def invalidate(
        self,
        id_: str,
        *,
        reason: str,
        at: str | None = None,
    ) -> MemoryRecord:
        record = self.get(id_)
        if record is None:
            raise NotFoundError(f"memory not found: {id_}")
        invalid_at = at or _now_iso()
        return self._set_validity_metadata(
            record.id,
            valid_at=record.valid_at or record.created,
            invalid_at=invalid_at,
            reason=reason,
        )

    def supersede(
        self,
        old_id: str,
        new_id: str,
        *,
        reason: str,
    ) -> MemoryRecord:
        old = self.get(old_id)
        new = self.get(new_id)
        if old is None:
            raise NotFoundError(f"memory not found: {old_id}")
        if new is None:
            raise NotFoundError(f"memory not found: {new_id}")
        if old.id == new.id:
            raise ValidationError("a memory cannot supersede itself")
        close_at = new.valid_at or new.created or datetime.now(UTC).isoformat()
        old_start = _parse_datetime(old.valid_at or old.created)
        close_dt = _parse_datetime(close_at)
        if old_start is not None and close_dt is not None and close_dt < old_start:
            close_at = old_start.isoformat()
        return self._set_validity_metadata(
            old.id,
            valid_at=old.valid_at or old.created,
            invalid_at=close_at,
            reason=f"superseded by {new.id[:8]}: {reason}",
        )

    def _drop_legacy_copy(self, source_path: Path, record: Any) -> None:
        """Remove the legacy vault copy after a metadata write landed in memory_dir.

        `_resolve_existing` can return `vault_path / rel` (legacy layout) while
        `_atomic_write_text` always writes under `memory_dir`, so the read and
        the write hit different files — leaving two `.md` carrying the same
        canonical id, which a later reindex reports as a duplicate-id conflict.
        Best-effort, mirroring `update_ops`' cleanup of the same hazard.
        """
        target_path = self.cfg.memory_dir / record.path
        if source_path == target_path:
            return
        try:
            source_path.unlink(missing_ok=True)
        except OSError as exc:
            _log.warning(
                "lifecycle(%s): stale legacy copy %s left in place — %s",
                record.id[:8],
                source_path,
                exc,
            )

    def _set_review_metadata(
        self,
        id_: str,
        *,
        review_after: str | None,
        verification_state: VerificationState | str,
        verified_at: int | None,
        evidence: str | None = None,
        actor: str | None = None,
        reviewed_at: str | None = None,
    ) -> None:
        record = self.get(id_)
        if record is None:
            raise NotFoundError(f"memory not found: {id_}")
        state = VerificationState(str(verification_state))
        path = self._resolve_existing(record.path)
        with self._data_dir_write_lock():
            original = path.read_text(encoding="utf-8")
            post = frontmatter.loads(original)
            if review_after is None:
                post.metadata.pop("review_after", None)
            else:
                post["review_after"] = review_after
            post["verification_state"] = state.value
            if verified_at is None:
                post.metadata.pop("verified_at", None)
            else:
                post["verified_at"] = verified_at
            self._atomic_write_text(record.path, frontmatter.dumps(post))
            try:
                updated = self.store.update_review_state(
                    id_=record.id,
                    review_after=review_after,
                    verification_state=state.value,
                    verified_at=verified_at,
                    evidence=evidence,
                    actor=actor,
                    reviewed_at=reviewed_at,
                )
                if not updated:
                    raise StorageError(f"review target missing from index: {record.id[:8]}")
            except Exception:
                # Roll back the file we actually READ. `_atomic_write_text`
                # always resolves under memory_dir, so on the legacy-vault
                # branch that is a DIFFERENT file from `path`.
                self._atomic_write_text(record.path, original)
                if path != self.cfg.memory_dir / record.path:
                    with contextlib.suppress(OSError):
                        path.write_text(original, encoding="utf-8")
                raise
            self._drop_legacy_copy(path, record)
        with contextlib.suppress(Exception):
            self.history.log_update(
                ts=datetime.now(UTC).isoformat(),
                record_id=record.id,
                title=record.title,
                type_=record.type,
                delta={
                    "review_after": (record.review_after, review_after),
                    "verification_state": (record.verification_state.value, state.value),
                    "verified_at": (record.verified_at, verified_at),
                },
            )
        self._write_gen += 1

    def _set_validity_metadata(
        self,
        id_: str,
        *,
        valid_at: str | None,
        invalid_at: str | None,
        reason: str,
    ) -> MemoryRecord:
        valid_at = _validity_iso(valid_at)
        invalid_at = _validity_iso(invalid_at)
        record = self.get(id_)
        if record is None:
            raise NotFoundError(f"memory not found: {id_}")
        path = self._resolve_existing(record.path)
        with self._data_dir_write_lock():
            original = path.read_text(encoding="utf-8")
            post = frontmatter.loads(original)
            if valid_at is None:
                post.metadata.pop("valid_at", None)
            else:
                post["valid_at"] = valid_at
            if invalid_at is None:
                post.metadata.pop("invalid_at", None)
            else:
                post["invalid_at"] = invalid_at
            self._atomic_write_text(record.path, frontmatter.dumps(post))
            try:
                updated = self.store.update_validity(
                    id_=record.id,
                    valid_at=valid_at,
                    invalid_at=invalid_at,
                )
                if not updated:
                    raise StorageError(f"validity target missing from index: {record.id[:8]}")
            except Exception:
                # Roll back the file we actually READ. `_atomic_write_text`
                # always resolves under memory_dir, so on the legacy-vault
                # branch that is a DIFFERENT file from `path`.
                self._atomic_write_text(record.path, original)
                if path != self.cfg.memory_dir / record.path:
                    with contextlib.suppress(OSError):
                        path.write_text(original, encoding="utf-8")
                raise
            self._drop_legacy_copy(path, record)
        with contextlib.suppress(Exception):
            self.history.log_update(
                ts=datetime.now(UTC).isoformat(),
                record_id=record.id,
                title=record.title,
                type_=record.type,
                delta={
                    "valid_at": (record.valid_at, valid_at),
                    "invalid_at": (record.invalid_at, invalid_at),
                    "validity_reason": (None, sanitize_persisted_text(reason).text),
                },
            )
        self._write_gen += 1
        return replace(record, valid_at=valid_at, invalid_at=invalid_at)
