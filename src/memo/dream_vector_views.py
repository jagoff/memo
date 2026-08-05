"""Build cheap deterministic vector views that complement whole-note vectors."""

from __future__ import annotations

from typing import Any

from .dream_hype import _active_variant, _embed_question
from .store.hype_store import HypeStore
from .tiers import DURABLE_TYPES


def run_title_view_pass(
    cfg: Any,
    mem: Any,
    *,
    night_cap: int = 1000,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Index title/tag views without an LLM call.

    These rows share the rebuildable HyPE sidecar but carry a distinct
    ``view_kind``. The existing whole-note vector remains the canonical body
    leg; search can now fold title views and hypothetical-question views into
    the same candidate pool.
    """
    result: dict[str, Any] = {
        "status": "skipped",
        "indexed": 0,
        "backlog": 0,
        "errors": 0,
        "view_kind": "title",
    }
    store: HypeStore | None = None
    try:
        identity = str(getattr(mem.store, "embedder_model", "") or "")
        store = HypeStore(cfg.db_path, cfg.embedder_dims, embedder_model=identity)
        pending: list[tuple[str, dict[str, Any], str]] = []
        for memory_id in mem.store.all_ids():
            row = mem.store.get(memory_id)
            if row is None or row.get("type") not in DURABLE_TYPES:
                continue
            body_hash = str(row.get("body_hash") or "")
            if not body_hash or store.view_body_hash_for(memory_id, "title") == body_hash:
                continue
            title = str(row.get("title") or "").strip()
            tags = row.get("tags") or []
            tag_text = " ".join(str(tag) for tag in tags)
            text = " ".join(part for part in (title, tag_text) if part).strip()
            if not text:
                continue
            pending.append((memory_id, row, text))
            if len(pending) >= max(1, int(night_cap)):
                break
        result["backlog"] = len(pending)
        if dry_run:
            result["status"] = "dry_run"
            return result
        for memory_id, row, text in pending:
            try:
                vector = _embed_question(mem, text)
                store.replace_for_memory(
                    memory_id,
                    str(row["body_hash"]),
                    identity or str(getattr(mem.cfg, "helper_model", "vector-view")),
                    [(text, vector)],
                    variant=_active_variant(),
                    view_kind="title",
                )
                result["indexed"] += 1
            except Exception:
                result["errors"] += 1
        result["status"] = "done"
    except Exception as exc:
        result["status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if store is not None:
            store.close()
    return result


__all__ = ["run_title_view_pass"]
