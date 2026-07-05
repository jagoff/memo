"""Dream pass: graduate quarantined ('_uncertain') auto-captures.

A low-confidence capture (MEMO_CAPTURE_MIN_CONFIDENCE) is saved tagged
'_uncertain' and excluded from auto-recall (MEMO_RECALL_EXCLUDE_UNCERTAIN).
This nightly pass promotes (untags) candidates that earned trust:

  * grounded     — a grounding.log row proves the memory was recalled and
                   actually USED in an answer;
  * corroborated — memory_health.support_count >= min_support (re-asserted
                   across independent captures; the column is added by the
                   corroboration workstream — an absent column counts as 0,
                   silently disabling this criterion).

Default off (MEMO_DREAM_GRADUATION_ENABLED). Reversible: promotion is a tag
edit through the versioned Memory.update()."""

from __future__ import annotations

import logging
from typing import Any

_log = logging.getLogger(__name__)


def _grounded_ids(state_dir: Any) -> set[str]:
    """8-hex prefixes of memories the grounding log proves were USED."""
    from memo.dashboard import grounding_used, read_grounding_log

    out: set[str] = set()
    for g in read_grounding_log(state_dir):
        rid = str(g.get("recall_id") or "")
        if len(rid) >= 8 and grounding_used(g):
            out.add(rid[:8])
    return out


def _support_count(mem: Any, id_: str) -> int:
    """support_count from memory_health; 0 when the column doesn't exist yet."""
    try:
        row = mem.store._conn.execute(
            "SELECT support_count FROM memory_health WHERE id = ?", (id_,)
        ).fetchone()
        return int(row["support_count"]) if row and row["support_count"] else 0
    except Exception:
        return 0


def run_graduation(
    cfg: Any, mem: Any, *, min_support: int = 2, dry_run: bool = False
) -> dict[str, Any]:
    candidates = mem.store.list_by_tag("_uncertain", limit=500)
    grounded8 = _grounded_ids(cfg.state_dir)
    promoted: list[dict[str, str]] = []
    for row in candidates:
        id_ = str(row["id"])
        if id_[:8] in grounded8:
            why = "grounded"
        elif _support_count(mem, id_) >= min_support:
            why = "corroborated"
        else:
            continue
        if not dry_run:
            tags = [t for t in (row.get("tags") or []) if t != "_uncertain"]
            mem.update(id_, tags=tags)
        promoted.append({"id": id_[:8], "title": str(row.get("title") or ""), "why": why})
    if promoted:
        _log.info("dream graduate: promoted %d of %d candidates", len(promoted), len(candidates))
    return {"candidates": len(candidates), "promoted": promoted, "dry_run": dry_run}
