"""`memo dream retag` — promote project memories proven general to global.

A memory saved under ``project:X`` but repeatedly *grounded* (used in answers,
``used_score >= GROUNDED_SCORE``) from sessions running in >= N OTHER projects
has proven general. Its project tag now hurts it everywhere else: the 3-tier
recall boost gives it +0 outside X, while an untagged (global) memory gets
+0.10 (``MEMO_RECALL_GLOBAL_BOOST``). This pass strips the ``project:`` tag
via the existing pure-retag update path — ``Memory.update()`` only re-embeds
when body/title change, so a tags-only patch costs zero embedder calls — and
every retag is reversible (``update()`` snapshots the prior record into
``versions.db``; ``memo version rollback``). OFF by default
(``MEMO_DREAM_RETAG_GLOBAL_ENABLED``).

The evidence source is ``grounding.log`` (``recall_id`` is an 8-char prefix,
``project`` is the ``project:<slug>`` tag of the grounded session's cwd).
Pure decision functions are fully unit-testable; the orchestrator wires the
real log, id resolution, and update, and is guarded so it never breaks the
dream pipeline.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from memo.project import is_project_tag
from memo.tiers import REFERENCE_TYPES

_log = logging.getLogger(__name__)

# --- pure core (testable) ----------------------------------------------------


def cross_project_counts(rows: list[dict[str, Any]], *, threshold: float) -> dict[str, set[str]]:
    """Map ``recall_id`` prefix → distinct ``project:`` tags whose sessions
    grounded it (``used_score >= threshold``). Rows with no ``project`` field
    carry no cross-project evidence and are ignored."""
    out: dict[str, set[str]] = {}
    for row in rows:
        rid = str(row.get("recall_id") or "")[:8]
        proj = row.get("project")
        score = row.get("used_score")
        if not rid or not isinstance(proj, str) or not proj:
            continue
        if not isinstance(score, (int, float)) or float(score) < threshold:
            continue
        out.setdefault(rid, set()).add(proj)
    return out


def retag_decisions(
    counts: dict[str, set[str]],
    *,
    get_record: Callable[[str], dict[str, Any] | None],
    min_other_projects: int = 2,
) -> list[dict[str, Any]]:
    """Decide which grounded memories to promote to global.

    ``get_record(prefix) -> {"id", "tags", "type"} | None`` (caller resolves
    prefixes; None = missing/ambiguous → skip). Promote iff the record carries
    a ``project:`` tag, is NOT reference tier, and was grounded from at least
    ``min_other_projects`` projects other than its own. Applying the retag is
    the caller's job — this returns decisions only."""
    decisions: list[dict[str, Any]] = []
    for rid, projects in sorted(counts.items()):
        rec = get_record(rid)
        if rec is None:
            continue
        tags = list(rec.get("tags") or [])
        own = [t for t in tags if is_project_tag(t)]
        if not own:
            continue  # already global
        if str(rec.get("type") or "") in REFERENCE_TYPES:
            continue  # bulk-ingested reference chunks are never promoted
        others = projects - set(own)
        if len(others) < min_other_projects:
            continue
        decisions.append(
            {
                "id": str(rec["id"]),
                "drop_tags": own,
                "new_tags": [t for t in tags if not is_project_tag(t)],
                "evidence_projects": sorted(others),
            }
        )
    return decisions


# --- orchestrator (guarded; wires real log + resolution + update) ------------


def run_retag_global(
    cfg: Any,
    mem: Any,
    *,
    min_other_projects: int = 2,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Mine grounding.log for cross-project use and retag proven-general
    memories to global. Tag-only updates (no re-embed); reversible via the
    version snapshot ``update()`` takes. Raises on unexpected failure — the
    cli_dream caller records it in ``receipt["errors"]``."""
    from memo.dashboard import GROUNDED_SCORE, read_grounding_log
    from memo.memory import AmbiguousIdError

    rows = read_grounding_log(cfg.state_dir)
    counts = cross_project_counts(rows, threshold=GROUNDED_SCORE)

    def _get(prefix: str) -> dict[str, Any] | None:
        try:
            rid = mem.resolve_id(prefix)
        except AmbiguousIdError:
            _log.debug("retag: ambiguous prefix %s — skipped", prefix)
            return None
        if rid is None:
            return None
        rec = mem.get(rid)
        if rec is None:
            return None
        return {"id": rec.id, "tags": list(rec.tags), "type": rec.type}

    decisions = retag_decisions(counts, get_record=_get, min_other_projects=min_other_projects)
    retagged: list[dict[str, Any]] = []
    for d in decisions:
        if not dry_run:
            mem.update(d["id"], tags=d["new_tags"])
        retagged.append(
            {
                "id": d["id"],
                "dropped": d["drop_tags"],
                "evidence_projects": d["evidence_projects"],
                "status": "would_retag" if dry_run else "retagged",
            }
        )
        _log.info(
            "retag %s: %s -> global (evidence: %s)%s",
            d["id"][:8],
            ",".join(d["drop_tags"]),
            ",".join(d["evidence_projects"]),
            " [dry-run]" if dry_run else "",
        )
    return {
        "status": "ok",
        "candidates": len(counts),
        "retagged": retagged,
        "dry_run": dry_run,
    }
