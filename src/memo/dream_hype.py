"""`memo dream hype` — nightly HyPE (Hypothetical Questions for Expansion) pass.

Each durable memory gets 2-3 LLM-generated hypothetical questions embedded and
indexed into `HypeStore` (`store/hype_store.py`). The backlog is watermarked
by `body_hash` (a memory whose body hasn't changed since it was last indexed
is skipped) and prioritized by ROI utility (`outcome.compute_utilities`) so a
capped nightly run spends its budget on memories that actually get used.

This is the generator half only — it builds the index "dark". The read-path
fold (`MEMO_HYPE_ENABLED`, `hype_fold.py`) is a separate, later-gated flag.
Building this index costs nothing for anyone who hasn't flipped the fold on.
"""

from __future__ import annotations

import json as _json
from typing import TYPE_CHECKING, Any

from .tiers import DURABLE_TYPES

if TYPE_CHECKING:
    from .store.hype_store import HypeStore

_SYS = (
    "You write hypothetical questions that a note answers. Given a note's "
    "title and body, reply with a JSON array of N short questions this note "
    "answers, written in the note's own language. Questions must be specific "
    "to THIS note — something only a reader of this exact note could answer. "
    "No generic questions. Reply with the JSON array only, nothing else."
)

_MIN_QUESTION_LEN = 12
_MAX_QUESTION_LEN = 200


def select_backlog(mem: Any, store: HypeStore, *, cap: int) -> list[dict[str, Any]]:
    """Durable memories (`tiers.DURABLE_TYPES`) whose `body_hash` differs from
    the one saved in `hype_questions` (or has no rows yet). Ordered by ROI
    utility desc (`outcome.compute_utilities`'s `by_prefix`; no data → neutral
    0.5). Capped at `cap`."""
    from . import outcome

    utilities = outcome.compute_utilities(mem.cfg.state_dir)
    by_prefix = utilities.get("by_prefix", {}) if isinstance(utilities, dict) else {}

    backlog: list[dict[str, Any]] = []
    for memory_id in mem.store.all_ids():
        row = mem.store.get(memory_id)
        if row is None:
            continue
        if row.get("type") not in DURABLE_TYPES:
            continue
        body_hash = row.get("body_hash")
        if store.body_hash_for(memory_id) == body_hash:
            continue
        utility = by_prefix.get(memory_id[:8], {}).get("utility", 0.5)
        backlog.append(
            {
                "id": memory_id,
                "title": row.get("title") or "",
                "body_hash": body_hash,
                "utility": utility,
            }
        )
    backlog.sort(key=lambda item: item["utility"], reverse=True)
    return backlog[:cap]


def _llm_questions(mem: Any, title: str, body: str, *, n: int) -> list[str] | None:
    """One `chat_with_timeout` call (timeout=30, `helper_model`, temp 0,
    thinking False). Prompt asks for a JSON array of `n` short questions this
    note answers. Filters out too-short (<12 chars) / too-long (>200 chars)
    entries, dedups, caps at `n`. Returns `None` on timeout or parse failure."""
    from .memory.record import chat_with_timeout

    prompt = f"N = {n}\nTitle: {title}\nBody:\n{body}"
    out = chat_with_timeout(
        mem._ensure_chat(),
        timeout=30,
        model=mem.cfg.helper_model,
        messages=[{"role": "system", "content": _SYS}, {"role": "user", "content": prompt}],
        options={"temperature": 0.0, "max_tokens": 256, "thinking": False},
    )
    if out is None:
        return None
    text = ((out.get("message") or {}).get("content") or "").strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        parsed = _json.loads(text)
    except (ValueError, TypeError):
        return None
    if not isinstance(parsed, list):
        return None

    questions: list[str] = []
    seen: set[str] = set()
    for item in parsed:
        if not isinstance(item, str):
            continue
        q = item.strip()
        if not (_MIN_QUESTION_LEN <= len(q) <= _MAX_QUESTION_LEN):
            continue
        if q in seen:
            continue
        seen.add(q)
        questions.append(q)
        if len(questions) >= n:
            break
    return questions


def run_hype_pass(
    cfg: Any,
    mem: Any,
    *,
    questions_per_memory: int = 3,
    night_cap: int = 400,
    dry_run: bool = False,
) -> dict[str, Any]:
    """One nightly HyPE pass. Never raises — the cli_dream caller records a
    returned ``status="error"`` in ``receipt["errors"]``.

    Per memory: `_llm_questions` → `embed_query` per question (str, one at a
    time) → `store.replace_for_memory`. A single memory's failure doesn't
    abort the pass (counted in `errors_items`). Prunes orphans against the
    live durable id set at the end. `dry_run` only computes the backlog.
    """
    res: dict[str, Any] = {
        "status": "skipped",
        "generated": 0,
        "memories": 0,
        "pruned": 0,
        "backlog_remaining": 0,
        "errors_items": 0,
    }
    store: HypeStore | None = None
    try:
        from .store.hype_store import HypeStore

        store = HypeStore(cfg.db_path, cfg.embedder_dims)

        backlog = select_backlog(mem, store, cap=night_cap)
        if not backlog:
            res["status"] = "skipped"
            return res

        if dry_run:
            res["status"] = "done"
            res["backlog_remaining"] = len(backlog)
            return res

        for item in backlog:
            body = mem.store.get_fts_body(item["id"])
            questions = _llm_questions(mem, item["title"], body, n=questions_per_memory)
            if not questions:
                res["errors_items"] += 1
                continue
            try:
                pairs = [(q, mem.embedder.embed_query(q)) for q in questions]
                inserted = store.replace_for_memory(
                    item["id"], item["body_hash"], mem.cfg.helper_model, pairs
                )
                res["generated"] += inserted
                res["memories"] += 1
            except Exception:  # one memory must never abort the pass
                res["errors_items"] += 1
                continue

        # Honest remaining count: failed items (counted in errors_items) were
        # never written to the store, so they're still pending — the cheap
        # proxy is len(backlog) minus the ones that succeeded (res["memories"]).
        res["backlog_remaining"] = max(0, len(backlog) - res["memories"])

        live_ids = {
            mid
            for mid in mem.store.all_ids()
            if (row := mem.store.get(mid)) is not None and row.get("type") in DURABLE_TYPES
        }
        res["pruned"] = store.prune_orphans(live_ids)
        res["status"] = (
            "all_items_failed" if res["errors_items"] == len(backlog) and backlog else "done"
        )
    except Exception as exc:  # surfaced via receipt["errors"], never silent
        res["status"] = "error"
        res["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if store is not None:
            store.close()
    return res


__all__ = ["run_hype_pass", "select_backlog"]
