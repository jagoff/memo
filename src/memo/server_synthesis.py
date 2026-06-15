"""MCP tools — emergent synthesis domain.

Registered by ``build_server()`` via ``register(server, memory)``.
"""

from __future__ import annotations

from typing import Any

import frontmatter
from fastmcp import FastMCP

from memo.memory import Memory


def register(server: FastMCP, memory: Memory) -> None:
    @server.tool()
    def memory_synthesize_run(
        dry_run: bool = True,
        threshold: float | None = None,
        min_cluster_size: int | None = None,
        max_clusters: int | None = None,
        min_confidence: str | None = None,
    ) -> list[dict[str, Any]]:
        """Generate emergent insights from semantically related memory clusters.

        Unlike consolidation (which merges duplicates), synthesis asks:
        "what do these memories collectively imply that none states alone?"

        Results are saved as type=synthesis memorias with provenance links to
        the contributing source memories. Safe by default (dry_run=True).

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
        return memory.synthesize_cross_cluster(**kwargs)

    @server.tool()
    def memory_synthesize_list(
        confidence: str | None = None,
    ) -> list[dict[str, Any]]:
        """List all existing synthesis memorias with their provenance.

        Returns synthesis memories (type=synthesis) including the source
        memory IDs that contributed to each insight.

        Args:
            confidence: Optional filter by confidence level: "low", "medium",
                or "high". When None (default), all synthesis memorias are
                returned regardless of confidence.
        """
        _valid_conf = {"low", "medium", "high"}
        if confidence is not None and confidence not in _valid_conf:
            raise ValueError(f"confidence must be one of {sorted(_valid_conf)} or None, got {confidence!r}")

        store_conn = memory.store._conn
        rows = store_conn.execute(
            "SELECT meta.id, meta.title, meta.path, meta.created, meta.updated "
            "FROM meta WHERE meta.type = 'synthesis' ORDER BY meta.updated DESC",
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
                        pass
            if confidence is not None and entry["confidence"] != confidence:
                continue
            results.append(entry)
        return results

    @server.tool()
    def memory_synthesize_delete(id: str) -> dict[str, Any]:
        """Delete a synthesis memoria by ID.

        Only deletes memories of type=synthesis. Raises an error if the
        given ID belongs to a non-synthesis memoria, preventing accidental
        deletion of regular memories through this tool.

        Args:
            id: The ID (or unambiguous prefix) of the synthesis memoria to delete.

        Returns:
            {"deleted": True, "id": "<full-id>"} on success, or
            {"deleted": False, "reason": "<reason>"} on failure.
        """
        from memo.errors import MemoError

        resolved = memory.resolve_id(id)
        if resolved is None:
            return {"deleted": False, "reason": f"No memoria found with id {id!r}"}

        rec = memory.get(resolved)
        if rec is None:
            return {"deleted": False, "reason": f"No memoria found with id {id!r}"}

        if rec.type != "synthesis":
            return {
                "deleted": False,
                "reason": (
                    f"Memoria {resolved!r} has type={rec.type!r}, not 'synthesis'. "
                    "Use memory_delete to delete non-synthesis memories."
                ),
            }

        try:
            ok = memory.delete(resolved)
        except MemoError as exc:
            return {"deleted": False, "reason": str(exc)}

        if ok:
            return {"deleted": True, "id": resolved}
        return {"deleted": False, "reason": f"Delete returned False for {resolved!r}"}
