"""`memo dream profile` — distill identity/convention memories into profile.md.

Ecosystem-survey Tier-1 #1 (+ Tier-2 #24 directive graduation): ONE bounded,
rewritten-in-place "who is the user / what is this project" artifact, read in
one call at SessionStart — covers the "you didn't know what to search for"
failure mode that similarity recall structurally misses.

The pass distills preference/feedback/decision + synthesis memories into
char-budgeted markdown profiles (global + per-project) under
``memory_dir/_profile/`` with memory-id provenance in the frontmatter, plus a
deterministic **Standing rules** block graduated from grounding.log (memories
cited in >= K distinct sessions), retired when a resolved contradiction pair
supersedes them. OFF by default (``MEMO_DREAM_PROFILE_ENABLED``).

Profile files carry NO ``id:`` frontmatter key, so ``memo reindex`` skips
them (maintain_ops skips any .md without ``id:``) — derived artifacts, not
memories. Pure functions are fully testable with plain dicts/callables; the
orchestrator wires the real store/LLM and never raises (the cli_dream caller
records failures in ``receipt["errors"]``).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from memo.project import GLOBAL_BUCKET, project_bucket, slugify_project

PROFILE_BUCKET = "_profile"
PROFILE_TYPES = frozenset({"preference", "feedback", "decision", "synthesis"})

# --- pure core (testable) ----------------------------------------------------


def profile_dir(cfg: Any) -> Path:
    """Where profile documents live: ``memory_dir/_profile/``.

    Inside ``memory_dir`` so the artifact is vault markdown (rides git sync,
    human-editable in Obsidian). Reindex skips these files: no ``id:`` key.
    """
    return Path(cfg.memory_dir) / PROFILE_BUCKET


def profile_path(cfg: Any, project: str | None = None) -> Path:
    """``profile.md`` (global) or ``project-<slug>.md`` (per-project).

    Per-project files always carry the ``project-`` prefix so no project
    slug can ever collide with the global ``profile.md``.
    """
    if project is None:
        return profile_dir(cfg) / "profile.md"
    slug = slugify_project(project) or GLOBAL_BUCKET
    return profile_dir(cfg) / f"project-{slug}.md"


def _bucket_of(row: dict[str, Any]) -> str:
    tags = row.get("tags") or []
    if not isinstance(tags, list):
        tags = []
    return project_bucket([str(t) for t in tags])


def select_sources(
    rows: list[dict[str, Any]], *, project: str | None = None, limit: int = 40
) -> list[dict[str, Any]]:
    """Profile-relevant store rows for ONE scope, newest-first, capped.

    ``project=None`` → global scope (rows with no ``project:`` tag);
    otherwise rows whose on-disk bucket matches the project slug. Only the
    four identity/convention types feed the profile.
    """
    want = slugify_project(project) if project is not None else GLOBAL_BUCKET
    out: list[dict[str, Any]] = []
    for r in rows:
        if r.get("type") not in PROFILE_TYPES:
            continue
        if _bucket_of(r) != want:
            continue
        out.append(r)
        if len(out) >= limit:
            break
    return out


def project_buckets(rows: list[dict[str, Any]]) -> list[str]:
    """Distinct non-global buckets among profile-type rows, most-recent first."""
    seen: list[str] = []
    for r in rows:
        if r.get("type") not in PROFILE_TYPES:
            continue
        b = _bucket_of(r)
        if b != GLOBAL_BUCKET and b not in seen:
            seen.append(b)
    return seen


def render_profile(
    *,
    scope: str,
    narrative: str,
    rules: list[tuple[str, str]],
    source_ids: list[str],
    updated: str,
    char_budget: int,
) -> str:
    """Assemble the full profile document. Deterministic — no LLM here.

    The standing-rules block survives the char budget before narrative text
    does (rules are the graduated, load-bearing part). Frontmatter is
    metadata, exempt from the budget. No ``id:`` key → never reindexed.
    """
    head = [
        "---",
        "generated_by: memo dream profile",
        f"updated: {updated}",
        f"scope: {scope}",
        "sources: " + json.dumps([s[:8] for s in source_ids]),
        "---",
        "",
        f"# Profile — {scope}",
        "",
    ]
    rule_lines: list[str] = []
    if rules:
        rule_lines = ["## Standing rules", ""]
        rule_lines += [f"- {text} `[{rid[:8]}]`" for rid, text in rules]
    budget = max(0, char_budget)
    remaining = budget - len("\n".join(rule_lines))
    body = (narrative or "").strip()
    if len(body) > remaining:
        body = body[: max(0, remaining - 1)].rstrip() + ("…" if remaining > 0 else "")
    parts = list(head)
    if body:
        parts += [body, ""]
    parts += rule_lines
    return "\n".join(parts).rstrip() + "\n"


# --- directive graduation (Tier-2 #24) ----------------------------------------

_RETIRED_STATUSES = frozenset({"kept_newer", "kept_older", "evolved", "fused"})


def standing_rule_ids(
    grounding_rows: list[dict[str, Any]], *, k: int = 3, min_used: float = 0.5
) -> list[str]:
    """recall_id prefixes cited in >= k DISTINCT sessions (used_score >= min_used).

    grounding.log rows carry 8-char ``recall_id`` prefixes. Ordering is by
    distinct-session count desc, then prefix — fully deterministic, no LLM.
    """
    sessions: dict[str, set[str]] = {}
    for row in grounding_rows:
        rid = str(row.get("recall_id") or "")
        sid = str(row.get("session_id") or "")
        try:
            used = float(row.get("used_score") or 0.0)
        except (TypeError, ValueError):
            used = 0.0
        if not rid or not sid or used < min_used:
            continue
        sessions.setdefault(rid, set()).add(sid)
    ranked = sorted(
        ((len(s), rid) for rid, s in sessions.items() if len(s) >= k),
        key=lambda t: (-t[0], t[1]),
    )
    return [rid for _, rid in ranked]


def losing_ids(
    pairs: list[dict[str, Any]],
    updated_of: Callable[[str], str | None],
) -> set[str]:
    """Ids retired by resolved contradiction pairs (retire-on-supersede).

    Per contradict.VALID_STATUSES semantics: ``kept_newer`` / ``evolved``
    retire the OLDER side; ``kept_older`` (older side won — explicit user
    choice) retires the NEWER side; ``fused`` (both merged into a new
    memory) retires BOTH sides. Age is by ``updated_of(id)`` — ISO
    timestamps compare lexicographically, same rule as
    cli_maintain._older_id. A side whose record is gone (``updated_of`` →
    None, e.g. already archived by the supersede pass) is retired outright.
    Open/dismissed pairs retire nothing.
    """
    out: set[str] = set()
    for p in pairs:
        status = str(p.get("status") or "")
        if status not in _RETIRED_STATUSES:
            continue
        a = str(p.get("memory_id_a") or "")
        b = str(p.get("memory_id_b") or "")
        if status == "fused":
            out.update(x for x in (a, b) if x)
            continue
        ua, ub = updated_of(a), updated_of(b)
        if ua is None and a:
            out.add(a)
        if ub is None and b:
            out.add(b)
        if ua is not None and ub is not None:
            older, newer = (a, b) if ua <= ub else (b, a)
            out.add(newer if status == "kept_older" else older)
    return out
