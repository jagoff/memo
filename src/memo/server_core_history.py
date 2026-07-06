from __future__ import annotations

import contextlib
from typing import Any

from memo.memory import AmbiguousIdError, Memory
from memo.server_annotations import READ_ONLY, annotated_tool


def register(server: Any, memory: Memory) -> None:
    @annotated_tool(server, **READ_ONLY)
    def memo_provenance(id: str) -> dict[str, Any] | None:
        """Return provenance metadata for one memory.

        Read-only. Use with a full id or unique prefix when you need origin,
        lineage, file path, or related audit details for a record before
        trusting, editing, or citing it.
        """
        return memory.provenance(id)

    @annotated_tool(server, **READ_ONLY)
    def memo_record_diff(id: str, limit: int = 50) -> dict[str, Any]:
        """Return recent history events for one memory in chronological order.

        Read-only. Use to inspect how a memory changed over time before
        editing or rolling back. Accepts a full id or unique prefix; `limit`
        caps returned events and `has_more` reports truncation.
        """
        resolved_id = id
        if len(resolved_id) < 32:
            try:
                resolved_id = memory.resolve_id(resolved_id) or resolved_id
            except AmbiguousIdError as exc:
                return {"error": "ambiguous", "prefix": exc.prefix, "matches": exc.matches}
        r = memory.get(resolved_id)
        events = memory.history.list_recent(limit=limit, record_id=resolved_id)
        events = list(reversed(events))
        return {
            "id": resolved_id,
            "title": r.title if r else None,
            "type": r.type if r else None,
            "events": events,
            "returned_events": len(events),
            "has_more": len(events) >= limit,
        }

    @annotated_tool(server, **READ_ONLY)
    def memo_history(
        limit: int = 20,
        op: str | None = None,
        id: str | None = None,
    ) -> list[dict[str, Any]]:
        """List recent memory history events.

        Read-only. Use for audit trails across the corpus or filter by `op`
        and memory `id` when investigating a specific write, update, delete,
        forget, or reindex action. Short ids are resolved safely.
        """
        record_id = id
        if record_id and len(record_id) < 32:
            try:
                resolved = memory.resolve_id(record_id)
            except AmbiguousIdError as exc:
                return [{"error": "ambiguous", "prefix": exc.prefix, "matches": exc.matches}]
            if resolved is None:
                return []
            record_id = resolved
        return memory.history.list_recent(limit=limit, op=op, record_id=record_id)

    @annotated_tool(server, **READ_ONLY)
    def memo_session_list(limit: int = 10, project: str | None = None) -> list[dict[str, Any]]:
        """List tracked memo sessions.

        Read-only. Use to find recent session ids, transcript paths, and
        project context for capture or inspection. `project` narrows results
        and `limit` caps the number of sessions returned.
        """
        from memo.session import list_sessions

        return list_sessions(memory.cfg.state_dir, limit=limit, project=project)

    @annotated_tool(server, **READ_ONLY)
    def memo_session_get(session_id: str) -> dict[str, Any] | None:
        """Fetch metadata for one tracked memo session.

        Read-only. Use after memo_session_list or memo_start_session when you
        need the stored transcript path, project, checkpoints, or other session
        details. Returns None when the session id is unknown.
        """
        from memo.session import get_session

        return get_session(memory.cfg.state_dir, session_id)

    @annotated_tool(server, **READ_ONLY)
    def memo_stats() -> dict[str, Any]:
        """Return local memo corpus and runtime statistics.

        Read-only. Use for diagnostics, health checks, and environment
        inspection. Includes corpus count, storage paths, embedder model,
        history error count, and recall daemon health when available.
        """
        history_errors = 0
        with contextlib.suppress(Exception):
            history_errors = int(getattr(memory.history, "error_count", 0))
        stats: dict[str, Any] = {
            "total": memory.store.count(),
            "data_dir": str(memory.cfg.data_dir),
            "vault_path": (str(memory.cfg.vault_path) if memory.cfg.vault_path else None),
            "db_path": str(memory.cfg.db_path),
            "embedder_model": memory.cfg.embedder_model,
            "history_errors": history_errors,
        }
        with contextlib.suppress(Exception):
            from memo.dashboard import recall_health

            stats["recall_health"] = recall_health(memory.cfg.state_dir)
        return stats
