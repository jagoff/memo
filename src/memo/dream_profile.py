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
import os
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


def _clean_narrative(narrative: str, *, scope: str) -> str:
    """Strip what the model was told not to emit, and what it repeated.

    `_SYS` asks for "no top-level title" and for a distillation, but an
    instruction to a model is not a guarantee and nothing checked. Measured on
    the live `_profile/profile.md`, 2026-08-31: `# Profile — global` appeared
    on lines 8 AND 10 (the document's own heading plus the echoed one), and 13
    of its 62 bullets were literal duplicates — one line repeated 12 times — in
    a document that was 43% of the whole SessionStart briefing. The briefing
    injects this file verbatim on every session, so each repeat is billed
    again, forever.

    Deterministic and order-preserving: the first occurrence of a bullet stays
    where the model put it, later identical ones are dropped. Comparison is on
    the stripped text so indentation does not defeat it; non-bullet lines
    (headings, blanks, prose) are never deduped — repetition there can be
    meaningful.
    """
    seen: set[str] = set()
    out: list[str] = []
    for line in (narrative or "").splitlines():
        stripped = line.strip()
        if stripped.startswith("# ") and stripped.lstrip("# ").strip() == f"Profile — {scope}":
            continue
        if stripped.startswith(("- ", "* ")):
            if stripped in seen:
                continue
            seen.add(stripped)
        out.append(line)
    return "\n".join(out).strip()


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
    body = _clean_narrative(narrative, scope=scope)
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


# --- LLM distillation + orchestrator (guarded) --------------------------------

_MAX_RULES = 20
_RULE_TEXT_CHARS = 140

_SYS = (
    "You maintain ONE bounded profile document distilled from a user's saved "
    "memories. Rewrite the prior profile IN PLACE: keep still-true content, "
    "fold in the new memories, drop anything superseded. State ONLY what the "
    "memories state — never invent. Plain markdown bullet points under short "
    "headings (identity, preferences, conventions, key decisions). "
    # Without this the model fills `identity` with whatever the recent
    # sessions happened to be about. Verified 2026-08-31: a memory titled
    # "Qué es memo (identidad del proyecto)" sat at row 1 of the 40-row source
    # window and the rewrite still opened with last week's token work, so a
    # session still had to be told what the project was.
    "Under `identity`, lead with WHAT THE PROJECT OR SYSTEM IS whenever a "
    "memory states it — purpose, stack and scope — before anything about "
    "recent work; that line is why a new session does not have to ask. "
    "No frontmatter, no top-level title. Stay under {budget} characters."
)


def _llm_distill(
    mem: Any,
    docs: list[dict[str, str]],
    *,
    prior: str,
    scope: str,
    budget: int,
) -> str | None:
    """One bounded chat call → the rewritten narrative, or None (skip)."""
    from memo.memory.record import chat_with_timeout

    if not docs:
        return None
    mem_lines = "\n".join(f"- [{d['type']}] {d['title']}: {d['body'][:400]}" for d in docs[:40])
    prompt = (
        f"Scope: {scope}\n\nPRIOR PROFILE (may be empty):\n{prior[:budget]}\n\n"
        f"MEMORIES:\n{mem_lines}\n\nRewrite the profile."
    )
    out = chat_with_timeout(
        mem._ensure_chat(),
        timeout=60,
        model=mem.cfg.helper_model,
        messages=[
            {"role": "system", "content": _SYS.format(budget=budget)},
            {"role": "user", "content": prompt},
        ],
        options={"temperature": 0.0, "max_tokens": 1024, "thinking": False},
    )
    if out is None:
        return None
    text = ((out.get("message") or {}).get("content") or "").strip()
    return text or None


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".md.tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)


def _docs_for(mem: Any, rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    docs: list[dict[str, str]] = []
    for r in rows:
        rec = mem.get(str(r.get("id") or ""))
        if rec is None:
            continue
        docs.append(
            {
                "id": rec.id,
                "type": rec.type,
                "title": rec.title or "",
                "body": (rec.body or "").strip(),
            }
        )
    return docs


def _gather_rules(mem: Any, cfg: Any, *, k: int, min_used: float) -> list[tuple[str, str]]:
    """Graduated standing rules: cited >= k distinct sessions, not superseded."""
    from memo.dashboard import read_grounding_log

    grounding = read_grounding_log(Path(cfg.state_dir))
    prefixes = standing_rule_ids(grounding, k=k, min_used=min_used)
    pairs = [
        {"status": p.status, "memory_id_a": p.memory_id_a, "memory_id_b": p.memory_id_b}
        for p in mem.contradict_store.list_all()
    ]
    retired = losing_ids(pairs, lambda mid: getattr(mem.get(mid), "updated", None))
    rules: list[tuple[str, str]] = []
    for prefix in prefixes:
        rec = mem.get(prefix)  # resolves 8-char prefixes; archived → None → retired
        if rec is None or rec.id in retired or rec.type == "reference":
            continue
        text = (rec.title or "").strip()
        if not text:
            body_lines = (rec.body or "").strip().splitlines()
            text = body_lines[0].strip() if body_lines else ""
        if not text:
            continue
        rules.append((rec.id, text[:_RULE_TEXT_CHARS]))
        if len(rules) >= _MAX_RULES:
            break
    return rules


def run_profile_pass(
    cfg: Any,
    mem: Any,
    *,
    char_budget: int = 4000,
    max_projects: int = 5,
    directive_k: int = 3,
    directive_min_used: float = 0.5,
    scan_limit: int = 500,
    dry_run: bool = False,
) -> dict[str, Any]:
    """One nightly profile-distillation pass. Never raises — the cli_dream
    caller records a returned ``status="error"`` in ``receipt["errors"]``."""
    res: dict[str, Any] = {"status": "noop", "written": [], "standing_rules": 0}
    try:
        from datetime import UTC, datetime

        rows = mem.store.list_recent(
            limit=scan_limit,
            exclude_types={"reference", "secret"},
        )
        rules = _gather_rules(mem, cfg, k=directive_k, min_used=directive_min_used)
        res["standing_rules"] = len(rules)
        now = datetime.now(tz=UTC).isoformat(timespec="seconds")
        scopes: list[str | None] = [None, *project_buckets(rows)[:max_projects]]
        for scope in scopes:
            sources = select_sources(rows, project=scope)
            scope_rules = rules if scope is None else []  # rules render globally
            scope_name = scope or "global"
            if not sources and not scope_rules:
                continue
            path = profile_path(cfg, scope)
            prior = path.read_text(encoding="utf-8") if path.is_file() else ""
            docs = _docs_for(mem, sources)
            narrative = _llm_distill(mem, docs, prior=prior, scope=scope_name, budget=char_budget)
            if narrative is None and not scope_rules:
                res["written"].append({"scope": scope_name, "status": "skipped"})
                continue
            content = render_profile(
                scope=scope_name,
                narrative=narrative or "",
                rules=scope_rules,
                source_ids=[d["id"] for d in docs],
                updated=now,
                char_budget=char_budget,
            )
            if dry_run:
                res["written"].append(
                    {"scope": scope_name, "status": "would_write", "chars": len(content)}
                )
            else:
                _atomic_write(path, content)
                res["written"].append(
                    {
                        "scope": scope_name,
                        "status": "written",
                        "chars": len(content),
                        "path": str(path),
                    }
                )
        if any(w.get("status") in ("written", "would_write") for w in res["written"]):
            res["status"] = "done"
    except Exception as exc:  # surfaced via receipt["errors"], never silent
        res["status"] = "error"
        res["error"] = f"{type(exc).__name__}: {exc}"
    return res
