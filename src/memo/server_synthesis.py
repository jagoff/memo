"""MCP tools — emergent synthesis domain.

Registered by ``build_server()`` via ``register(server, memory)``.
"""

from __future__ import annotations

import logging
from typing import Any

import frontmatter
from fastmcp import Context, FastMCP

from memo.memory import Memory
from memo.server_annotations import DESTRUCTIVE, READ_ONLY, WRITE, annotated_tool
from memo.server_common import run_synth

_log = logging.getLogger(__name__)


def register(server: FastMCP, memory: Memory) -> None:
    @annotated_tool(server, **WRITE)
    async def memo_synthesize_run(
        dry_run: bool = True,
        threshold: float | None = None,
        min_cluster_size: int | None = None,
        max_clusters: int | None = None,
        min_confidence: str | None = None,
        ctx: Context | None = None,
    ) -> list[dict[str, Any]]:
        """Generate emergent insights from semantically related memory clusters.

        Unlike consolidation (which merges duplicates), synthesis asks:
        "what do these memories collectively imply that none states alone?"

        Results are saved as type=synthesis memories with provenance links to
        the contributing source memories. Safe by default (dry_run=True).
        With client sampling enabled, synthesis runs on the calling model up
        to MEMO_SAMPLING_MAX_CALLS, then local MLX (list return — no
        synthesizer field).

        Args:
            dry_run: If True (default), propose without saving.
            threshold: Cosine similarity for clustering (default 0.78, looser than consolidation).
            min_cluster_size: Minimum memories per cluster to consider (default 3).
            max_clusters: Max clusters to process per pass (default 20).
            min_confidence: Minimum LLM confidence to save: low | medium | high (default medium).
        """
        kwargs: dict[str, Any] = {"dry_run": dry_run}
        if threshold is not None:
            kwargs["threshold"] = threshold
        if min_cluster_size is not None:
            kwargs["min_cluster_size"] = min_cluster_size
        if max_clusters is not None:
            kwargs["max_clusters"] = max_clusters
        if min_confidence is not None:
            kwargs["min_confidence"] = min_confidence
        res, _synthesizer = await run_synth(
            memory, ctx, lambda: memory.synthesize_cross_cluster(**kwargs)
        )
        return res

    @annotated_tool(server, **READ_ONLY)
    def memo_synthesize_list(
        confidence: str | None = None,
        scan_limit: int = 100,
    ) -> list[dict[str, Any]]:
        """List all existing synthesis memories with their provenance.

        Returns synthesis memories (type=synthesis) including the source
        memory IDs that contributed to each insight.

        Args:
            confidence: Optional filter by confidence level: "low", "medium",
                or "high". When None (default), all synthesis memories are
                returned regardless of confidence.
            scan_limit: Maximum number of synthesis memories to read from the
                index before applying the confidence filter (most recent first).
                Bounds rows scanned, not rows returned — the returned list may
                be smaller when confidence is set.
        """
        limit = scan_limit
        _valid_conf = {"low", "medium", "high"}
        if confidence is not None and confidence not in _valid_conf:
            raise ValueError(
                f"confidence must be one of {sorted(_valid_conf)} or None, got {confidence!r}"
            )

        store_conn = memory.store.connection
        rows = store_conn.execute(
            "SELECT meta.id, meta.title, meta.path, meta.created, meta.updated "
            "FROM meta WHERE meta.type = 'synthesis' ORDER BY meta.updated DESC LIMIT ?",
            (limit,),
        ).fetchall()

        results: list[dict[str, Any]] = []
        for r in rows:
            entry: dict[str, Any] = {
                "id": r["id"],
                "title": r["title"],
                "created": r["created"],
                "updated": r["updated"],
                "sources": [],
                "confidence": None,
                "rationale": None,
            }
            if r["path"]:
                p = memory._resolve_existing(r["path"])
                if p.is_file():
                    try:
                        post = frontmatter.loads(p.read_text(encoding="utf-8"))
                        ex: dict = post.get("extra") or {}  # type: ignore[assignment]
                        entry["sources"] = ex.get("synthesis_sources") or []
                        entry["confidence"] = ex.get("synthesis_confidence")
                        entry["rationale"] = ex.get("synthesis_rationale")
                        entry["body"] = post.content[:300] if post.content else ""
                    except Exception:
                        _log.debug("synthesis list: failed to read %s", r["path"], exc_info=True)
            if confidence is not None and entry["confidence"] != confidence:
                continue
            results.append(entry)
        return results

    @annotated_tool(server, **DESTRUCTIVE)
    def memo_synthesize_delete(id: str) -> dict[str, Any]:
        """Delete a synthesis memory by ID.

        Only deletes memories of type=synthesis. Raises an error if the
        given ID belongs to a non-synthesis memory, preventing accidental
        deletion of regular memories through this tool.

        Args:
            id: The ID (or unambiguous prefix) of the synthesis memory to delete.

        Returns:
            {"deleted": True, "id": "<full-id>"} on success, or
            {"deleted": False, "reason": "<reason>"} on failure.
        """
        from memo.errors import MemoError

        resolved = memory.resolve_id(id)
        if resolved is None:
            return {"deleted": False, "reason": f"No memory found with id {id!r}"}

        rec = memory.get(resolved)
        if rec is None:
            return {"deleted": False, "reason": f"No memory found with id {id!r}"}

        if rec.type != "synthesis":
            return {
                "deleted": False,
                "reason": (
                    f"Memory {resolved!r} has type={rec.type!r}, not 'synthesis'. "
                    "Use memo_delete to delete non-synthesis memories."
                ),
            }

        try:
            ok = memory.delete(resolved)
        except MemoError as exc:
            return {"deleted": False, "reason": str(exc)}

        if ok:
            return {"deleted": True, "id": resolved}
        return {"deleted": False, "reason": f"Delete returned False for {resolved!r}"}
