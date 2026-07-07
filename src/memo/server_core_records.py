from __future__ import annotations

from typing import Any

from memo.memory import AmbiguousIdError, Memory
from memo.server_annotations import (
    DESTRUCTIVE,
    READ_ONLY,
    WRITE,
    WRITE_IDEMPOTENT,
    annotated_tool,
)


def register(server: Any, memory: Memory) -> None:
    @annotated_tool(server, **WRITE)
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

    @annotated_tool(server, **READ_ONLY)
    def memo_list(limit: int = 20, type: str | None = None) -> list[dict[str, Any]]:
        """List recent memories, optionally filtered by memory type.

        Read-only. Use this to browse the corpus before choosing an id for
        memo_get, memo_update, memo_rename, memo_delete, or history tools.
        `limit` caps the number of returned records.
        """
        return [r.to_dict() for r in memory.list(limit=limit, type_=type)]

    @annotated_tool(server, **READ_ONLY)
    def memo_get(id: str) -> dict[str, Any] | None:
        """Fetch one memory by id or unique id prefix.

        Read-only. Returns the full memory record, `None` when it does not
        exist, or an ambiguity error when the prefix matches multiple records.
        Use memo_search or memo_list first when you do not know the id.
        """
        try:
            rec = memory.get(id)
        except AmbiguousIdError as exc:
            return {"error": "ambiguous", "prefix": exc.prefix, "matches": exc.matches}
        if not rec:
            return None
        return rec.to_dict()

    @annotated_tool(server, **DESTRUCTIVE)
    def memo_update(
        id: str,
        title: str | None = None,
        type: str | None = None,
        tags: list[str] | None = None,
        content: str | None = None,
        replace_old: str | None = None,
        replace_new: str | None = None,
        append: str | None = None,
    ) -> dict[str, Any] | None:
        """Patch fields on a memory. `content` replaces the whole body;
        `replace_old`+`replace_new` is a surgical exact-string edit (old must
        occur exactly once — unchanged text stays byte-identical); `append`
        adds a paragraph. All three are versioned (memo_version_rollback).
        """
        if (replace_old is None) != (replace_new is None):
            return {
                "error": "replace_incomplete",
                "message": "pass replace_old and replace_new together",
            }
        replace = (
            (replace_old, replace_new)
            if replace_old is not None and replace_new is not None
            else None
        )
        try:
            rec = memory.update(
                id,
                title=title,
                type_=type,
                tags=tags,
                content=content,
                replace=replace,
                append=append,
            )
        except AmbiguousIdError as exc:
            return {"error": "ambiguous", "prefix": exc.prefix, "matches": exc.matches}
        except ValueError as exc:
            return {"error": "edit_failed", "message": str(exc)}
        return rec.to_dict() if rec else None

    @annotated_tool(server, **DESTRUCTIVE)
    def memo_rename(title: str, id: str | None = None) -> dict[str, Any] | None:
        """Rename one memory title without changing its body or tags.

        Destructive metadata edit. Use after memo_save or memo_search when a
        record has the right content but the wrong title. Pass `id` for a
        specific memory; omit it only immediately after a save, when memo can
        target the most recent local save. Use memo_update instead when you
        need to edit content, type, or tags.
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

    @annotated_tool(server, **WRITE_IDEMPOTENT)
    def memo_reindex(force: bool = False) -> dict[str, int]:
        """Rebuild memo's searchable index from the markdown vault.

        Writes only derived index state; markdown remains the source of truth.
        Use after hand-editing vault files or changing indexing behavior.
        `force` reprocesses records even if memo thinks they are current.
        """
        return memory.reindex(force=force)

    @annotated_tool(server, **DESTRUCTIVE)
    def memo_delete(id: str) -> dict[str, Any]:
        """Permanently delete one memory by id or unique prefix.

        Destructive. Resolves ambiguous short ids safely and returns an error
        instead of guessing. When cross-reference indexing is enabled, the
        response warns about memories that linked to the deleted record.
        """
        from memo.flags import flag_bool

        referenced_by: list[str] = []
        if flag_bool("MEMO_CROSSREF_INDEX"):
            try:
                rec = memory.get(id)
                if rec is not None:
                    referenced_by = [
                        b.source_id for b in memory.crossref.referencing_sources(rec.id)
                    ]
            except AmbiguousIdError as exc:
                return {"error": "ambiguous", "prefix": exc.prefix, "matches": exc.matches}
            except Exception:
                referenced_by = []
        try:
            out: dict[str, Any] = {"deleted": memory.delete(id)}
        except AmbiguousIdError as exc:
            return {"error": "ambiguous", "prefix": exc.prefix, "matches": exc.matches}
        if referenced_by:
            out["cascade_warning"] = (
                f"{len(referenced_by)} memories link to the deleted id; "
                "their typed edges now dangle"
            )
            out["referenced_by"] = referenced_by
        return out

    @annotated_tool(server, **DESTRUCTIVE)
    def memo_forget(id: str, reason: str | None = None) -> dict[str, Any]:
        """Mark one memory as forgotten without deleting its history.

        Destructive in retrieval behavior: the memory is hidden from normal
        recall/search surfaces until memo_unforget restores it. Pass `reason`
        to record why the memory should no longer be surfaced.
        """
        try:
            rec = memory.forget(id, reason=reason)
        except AmbiguousIdError as exc:
            return {"error": "ambiguous", "prefix": exc.prefix, "matches": exc.matches}
        if rec is None:
            return {"forgotten": False}
        return {"forgotten": True, "id": rec.id}

    @annotated_tool(server, **WRITE_IDEMPOTENT)
    def memo_unforget(id: str) -> dict[str, Any]:
        """Restore a previously forgotten memory to normal retrieval.

        Idempotent write. Accepts a full id or unique prefix and returns
        whether a matching forgotten record was restored. Use after deciding a
        memory should participate in search and recall again.
        """
        try:
            rec = memory.unforget(id)
        except AmbiguousIdError as exc:
            return {"error": "ambiguous", "prefix": exc.prefix, "matches": exc.matches}
        if rec is None:
            return {"unforgotten": False}
        return {"unforgotten": True, "id": rec.id}

    @annotated_tool(server, **READ_ONLY)
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

    @annotated_tool(server, **READ_ONLY)
    def memo_lint() -> dict[str, list[dict[str, Any]]]:
        """Inspect the memory corpus for maintenance issues.

        Read-only. Returns grouped lint findings such as malformed metadata or
        other records that may need cleanup. Use memo_update, memo_delete, or
        vault edits separately to fix findings.
        """
        return memory.lint()
