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
