"""Nightly vector-index hygiene and derived-storage maintenance.

The Markdown vault and user feedback are primary data. Everything handled here
is rebuildable: stale repository embedding-cache identities and sqlite-vec
chunk allocation for source feedback. The pass is deliberately separate from
the recall hot path and is dry-run capable.
"""

from __future__ import annotations

from typing import Any


def run_vector_hygiene(
    cfg: Any,
    mem: Any,
    *,
    dry_run: bool = False,
    vacuum: bool = False,
) -> dict[str, Any]:
    """Prune stale embedding-cache identities and compact feedback vectors."""
    result: dict[str, Any] = {
        "status": "skipped",
        "cache_packed": 0,
        "cache_pruned": 0,
        "feedback": {"before": 0, "after": 0, "rebuilt": False},
        "vacuumed": False,
    }
    try:
        model = str(getattr(mem.store, "embedder_model", "") or "").strip()
        dims = int(getattr(cfg, "embedder_dims", 0) or 0)
        if model and dims > 0:
            result["cache_packed"] = mem.store.compact_repo_embedding_cache(dry_run=dry_run)
            result["cache_pruned"] = mem.store.prune_repo_embedding_cache(
                keep_models={(model, dims)},
                dry_run=dry_run,
            )
        else:
            result["cache_packed"] = 0
            result["cache_pruned"] = 0

        result["feedback"] = mem.store.compact_feedback_vectors(dry_run=dry_run)
        if not dry_run:
            # Checkpoint is best-effort; dream's receipt should not become an
            # error merely because a long-lived reader pins the WAL snapshot.
            mem.store._checkpoint()
            if vacuum:
                mem.store._conn.execute("VACUUM")
                result["vacuumed"] = True
        result["status"] = "dry_run" if dry_run else "done"
    except Exception as exc:  # surfaced in the dream receipt
        result["status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


__all__ = ["run_vector_hygiene"]
