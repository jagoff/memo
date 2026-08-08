"""MCP tools — emergent synthesis domain.

Registered by ``build_server()`` via ``register(server, memory)``.
"""

from __future__ import annotations

import logging
from typing import Any

import frontmatter
from fastmcp import Context, FastMCP

from memo.mcp_budget import bounded_list
from memo.memory import Memory
from memo.server_annotations import DESTRUCTIVE, READ_ONLY, WRITE, annotated_tool
from memo.server_common import run_synth

_log = logging.getLogger(__name__)

# Per-result cap on the `sources` provenance list. `synthesize_cross_cluster`
# returns every id in the cluster it synthesised, and a cluster's size tracks
# the CORPUS, not `max_clusters`: measured on the conformance corpus, 9,043
# tokens at 2,000 memories (just under the 10k cap, which is why it slipped
# through) and 44,043 at 10,001. Bounded here at the MCP boundary rather than
# in `synthesize_cross_cluster` itself -- the save path writes the full list to
# the synthesis memory's `synthesis_sources` frontmatter, so the core return
# must stay complete. `total` reports the real cluster size and `sources_hash`
# still identifies the exact set.
_MAX_SYNTHESIS_SOURCES = 20


def _bounded_synthesis(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Trim each synthesis result's `sources` list, keeping the true total."""
    bounded: list[dict[str, Any]] = []
    for result in results:
        sources = result.get("sources") if isinstance(result, dict) else None
        if not isinstance(sources, list):
            bounded.append(result)
            continue
        kept, meta = bounded_list(sources, cap=_MAX_SYNTHESIS_SOURCES)
        bounded.append({**result, "sources": kept, **meta})
    return bounded


def _delete_synthesis(memory: Memory, id: str) -> dict[str, Any]:
    """Delete a resolved synthesis memory without exposing generic deletion."""
    from memo.errors import AmbiguousIdError, MemoError

    try:
        resolved = memory.resolve_id(id)
    except AmbiguousIdError as exc:
        return {"error": "ambiguous", "prefix": exc.prefix, "matches": exc.matches}
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

        Each result's `sources` list is a sample of the cluster that produced
        it; `total` carries the cluster's real size and `truncated` says
        whether any ids were dropped. A saved synthesis keeps the complete
        list in its `synthesis_sources` frontmatter.

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
        return _bounded_synthesis(res)

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
        limit = max(1, min(scan_limit, 500))  # a negative LIMIT binds as unbounded in sqlite
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
    async def memo_synthesize_delete(id: str, ctx: Context | None = None) -> dict[str, Any]:
        """Delete a synthesis memory by ID.

        Only deletes memories of type=synthesis. Raises an error if the
        given ID belongs to a non-synthesis memory, preventing accidental
        deletion of regular memories through this tool. Irreversible:
        elicitation-capable clients are asked to confirm first.

        Args:
            id: The ID (or unambiguous prefix) of the synthesis memory to delete.

        Returns:
            {"deleted": True, "id": "<full-id>"} on success,
            {"deleted": False, "reason": "<reason>"} on failure, or
            {"error": "ambiguous", "prefix", "matches"} when the prefix
            matches multiple records.
        """
        from memo.errors import AmbiguousIdError
        from memo.server_elicit import abort_result, confirm_destructive, sanitize_fragment

        try:
            resolved = memory.resolve_id(id)
        except AmbiguousIdError as exc:
            return {"error": "ambiguous", "prefix": exc.prefix, "matches": exc.matches}
        rec = memory.get(resolved) if resolved else None
        if rec is not None and rec.type == "synthesis":
            safe_title = sanitize_fragment(rec.title)
            gate = await confirm_destructive(
                ctx,
                action="delete",
                detail=(
                    f"Permanently delete synthesis '{safe_title}'? Same no-trash "
                    "delete path as memo_delete — recovery only via backup / "
                    "git-sync / versions."
                ),
            )
            if not gate.proceed:
                return abort_result(
                    gate,
                    memory,
                    tool="memo_synthesize_delete",
                    action="delete",
                    target=f"synthesis '{safe_title}' id={rec.id}",
                )
        return _delete_synthesis(memory, id)
