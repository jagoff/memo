from __future__ import annotations

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
        extract: bool | None = None,
        extra: dict[str, Any] | None = None,
        respect_synapse_freeze: bool | None = None,
        scope: str | None = None,
    ) -> dict[str, Any]:
        """Persist `content` to memo.

        When `extract` is true (defaults to the `MEMO_SAVE_EXTRACT` flag, off),
        the helper LLM decomposes `content` into atomic facts and saves each as
        its own memory (mem0 ADD-model) instead of one opaque blob; `tags`
        propagate to every fact. Returns an extraction summary
        (`status`, `saved` ids, `saved_titles`, counts) rather than a single
        record. If nothing extractable is found, the blob is saved verbatim.

        `scope` controls the auto `project:<repo>` tag for THIS call only:
        `"global"` skips it (the memory lands untagged → the global recall
        tier, +0.10 boost everywhere); `"project"` or None keep the default
        auto-detection. An explicit `project:` tag in `tags` always wins
        either way.
        """
        from memo.flags import flag_bool
        from memo.memory import WriteRefused

        if scope not in (None, "project", "global"):
            return {
                "error": "invalid_scope",
                "message": f"scope must be 'project' or 'global', got {scope!r}",
            }
        auto_project = scope != "global"

        if extract is None:
            extract = flag_bool("MEMO_SAVE_EXTRACT")
        if extract:
            from memo.capture import extract_and_save_text

            return extract_and_save_text(
                memory,
                memory.cfg,
                content,
                merge_tags=tags,
                title=title,
                type_=type,
                auto_project=auto_project,
            )

        try:
            rec = memory.save(
                content=content,
                title=title,
                type_=type,
                tags=tags,
                auto_derive=auto_derive,
                auto_project=auto_project,
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
    def memo_rename(title: str, id: str | None = None) -> dict[str, Any] | None:
        """Rename a memory's title. Without `id`, renames the memory most
        recently saved on this machine — e.g. right after a `memo_save`.
        """
        target = id or memory.last_saved_id()
        if target is None:
            return {
                "error": "no_recent_save",
                "message": "no recent save on this machine; pass `id` explicitly",
            }
        try:
            rec = memory.update(target, title=title)
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
    ) -> dict[str, Any]:
        """Detect near-duplicate clusters and propose merges.

        Read-only — returns proposals without modifying the corpus.
        Uses the AdvancedConsolidator under the hood (same as
        ``memo_consolidate_list_archived``).

        Args:
            threshold: Cosine similarity threshold (default 0.85).
            max_clusters: Maximum clusters to process (default 20).
            type: Optional filter by memory type.
        """
        return memory.consolidator.consolidate_all(
            threshold=threshold,
            max_clusters=max_clusters,
            type_=type,
            auto_apply=False,
            dry_run=True,
        )

    @server.tool()
    def memo_lint() -> dict[str, list[dict[str, Any]]]:
        return memory.lint()
