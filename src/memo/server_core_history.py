from __future__ import annotations

import contextlib
from typing import Any

from memo.memory import AmbiguousIdError, Memory
from memo.server_annotations import READ_ONLY, annotated_tool


def register(server: Any, memory: Memory) -> None:
    @annotated_tool(server, **READ_ONLY)
    def memo_provenance(id: str) -> dict[str, Any] | None:
        return memory.provenance(id)

    @annotated_tool(server, **READ_ONLY)
    def memo_record_diff(id: str, limit: int = 50) -> dict[str, Any]:
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
        from memo.session import list_sessions

        return list_sessions(memory.cfg.state_dir, limit=limit, project=project)

    @annotated_tool(server, **READ_ONLY)
    def memo_session_get(session_id: str) -> dict[str, Any] | None:
        from memo.session import get_session

        return get_session(memory.cfg.state_dir, session_id)

    @annotated_tool(server, **READ_ONLY)
    def memo_stats() -> dict[str, Any]:
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
