from __future__ import annotations

import contextlib
from typing import Any

from memo.memory import AmbiguousIdError, Memory


def register(server: Any, memory: Memory) -> None:
    @server.tool()
    def memo_save(
        content: str,
        title: str | None = None,
        type: str = "note",
        tags: list[str] | None = None,
        auto_derive: bool = False,
        extra: dict[str, Any] | None = None,
        respect_synapse_freeze: bool | None = None,
    ) -> dict[str, Any]:
        from memo.memory import WriteRefused

        try:
            rec = memory.save(
                content=content,
                title=title,
                type_=type,
                tags=tags,
                auto_derive=auto_derive,
                extra=extra,
                respect_synapse_freeze=respect_synapse_freeze,
            )
        except WriteRefused as exc:
            return {
                "status": "refused",
                "conflict": exc.conflict,
                "message": str(exc),
            }
        return rec.to_dict()

    @server.tool()
    def memo_list(limit: int = 20, type: str | None = None) -> list[dict[str, Any]]:
        return [r.to_dict() for r in memory.list(limit=limit, type_=type)]

    @server.tool()
    def memo_get(id: str) -> dict[str, Any] | None:
        try:
            rec = memory.get(id)
        except AmbiguousIdError as exc:
            return {"error": "ambiguous", "prefix": exc.prefix, "matches": exc.matches}
        if not rec:
            return None
        with contextlib.suppress(Exception):
            memory.contextual.record_click(rec.id)
        return rec.to_dict()

    @server.tool()
    def memo_update(
        id: str,
        title: str | None = None,
        type: str | None = None,
        tags: list[str] | None = None,
        content: str | None = None,
    ) -> dict[str, Any] | None:
        try:
            rec = memory.update(
                id,
                title=title,
                type_=type,
                tags=tags,
                content=content,
            )
        except AmbiguousIdError as exc:
            return {"error": "ambiguous", "prefix": exc.prefix, "matches": exc.matches}
        return rec.to_dict() if rec else None

    @server.tool()
    def memo_reindex(force: bool = False) -> dict[str, int]:
        return memory.reindex(force=force)

    @server.tool()
    def memo_delete(id: str) -> dict[str, Any]:
        try:
            return {"deleted": memory.delete(id)}
        except AmbiguousIdError as exc:
            return {"error": "ambiguous", "prefix": exc.prefix, "matches": exc.matches}

    @server.tool()
    def memo_forget(id: str, reason: str | None = None) -> dict[str, Any]:
        try:
            rec = memory.forget(id, reason=reason)
        except AmbiguousIdError as exc:
            return {"error": "ambiguous", "prefix": exc.prefix, "matches": exc.matches}
        if rec is None:
            return {"forgotten": False}
        return {"forgotten": True, "id": rec.id}

    @server.tool()
    def memo_unforget(id: str) -> dict[str, Any]:
        try:
            rec = memory.unforget(id)
        except AmbiguousIdError as exc:
            return {"error": "ambiguous", "prefix": exc.prefix, "matches": exc.matches}
        if rec is None:
            return {"unforgotten": False}
        return {"unforgotten": True, "id": rec.id}

    @server.tool()
    def memo_consolidate(
        threshold: float = 0.85,
        max_clusters: int = 20,
        type: str | None = None,
    ) -> list[dict[str, Any]]:
        return memory.consolidate(
            threshold=threshold,
            max_clusters=max_clusters,
            type_=type,
        )

    @server.tool()
    def memo_lint() -> dict[str, list[dict[str, Any]]]:
        return memory.lint()
