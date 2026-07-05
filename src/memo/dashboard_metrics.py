from __future__ import annotations

import re
from typing import Any

from memo.dashboard_logs import (
    read_daily_trend,
    read_grounding_diag_log,
    read_grounding_log,
    read_recall_hook_log,
    read_recall_log,
    read_usage_log,
)

STRONG_SCORE = 0.85
# Reask/grounding-turn bar (kept moderate — used by reask_stats).
GROUNDED_SCORE = 0.6
# Utility bar: "the answer actually USED this memory", not just topical overlap.
# answer↔memory similarity is a continuum dominated by shared topic (same-topic
# cosine ~0.7 is baseline), so a low bar marks ~everything "used" (the 100%
# artifact). 0.8 requires a match well above topical baseline; a downstream
# action (the turn opened/ran what the memory named) always counts as used.
USED_SCORE_STRONG = 0.8
# Paraphrase recovery: the answer matches the memory at least this much MORE
# than the question already did (cos(answer,mem) - cos(question,mem)). Catches
# real use that paraphrases (modest absolute cosine, but clearly above the
# topical baseline the prompt set), without crediting same-topic overlap.
SPECIFIC_MARGIN = 0.06
# Verdict thresholds — drive the "DOES memo WORK?" panel. memo is judged USEFUL
# only when it is both read enough (volume) and actually helping (grounded).
VERDICT_MIN_CONSULTS = 20
VERDICT_MIN_GROUNDED = 0.10
# Flag weakness as soon as there's ANY measurement (deliberate: surface a low
# grounded_rate early rather than hide behind "unmeasured"). The honest-denominator
# logic in grounded_rate already excludes un-scored turns, so a measured turn here
# means a real recall→use observation.
VERDICT_MIN_MEASURED_TURNS = 1
# Keep "unmeasured" reserved for near-absence of grounding data. Once we have
# a modest sample, the verdict can already fall through to weak/ok.
VERDICT_MIN_MEASUREMENT_COVERAGE = 0.05
# Layers we EXPECT to read memo (flagged as "silent" if absent).
# Only always-on daemons/hooks belong here; on-demand tools (devin, opencode,
# devin-desktop) appear as readers if/when they actually query — no silent flag
# for tools the user invokes explicitly rather than tools that run continuously.
EXPECTED_CONSUMERS = (
    "claude-code",
    "synapse",
    "memflow",
    "codex",
)


def _bail_breakdown(bail_rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {"slash_command": 0, "short_prompt": 0, "min_sim": 0, "other": 0}
    for row in bail_rows:
        reason = row.get("reason", "")
        if "slash command" in reason:
            counts["slash_command"] += 1
        elif "too short" in reason:
            counts["short_prompt"] += 1
        elif "min_sim" in reason or "no hits above" in reason:
            counts["min_sim"] += 1
        else:
            counts["other"] += 1
    return counts


def _parse_ts(value: Any) -> Any:
    from datetime import datetime

    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _row_quality(row: dict[str, Any]) -> tuple[int, float]:
    hits = row.get("hits") or []
    top = hits[0].get("score") if hits else None
    return (len(hits), float(top) if isinstance(top, (int, float)) else 0.0)


def _median(xs: list[Any]) -> Any:
    """Median of a PRE-SORTED list; averages the two middle values for even n."""
    if not xs:
        return None
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2


def grounding_used(row: dict[str, Any]) -> bool:
    """Single production decision for whether a grounding row means "used"."""
    score = row.get("used_score")
    spec = row.get("specific_score")
    strong = isinstance(score, (int, float)) and float(score) >= USED_SCORE_STRONG
    specific = isinstance(spec, (int, float)) and float(spec) >= SPECIFIC_MARGIN
    return bool(strong or specific or row.get("downstream_action"))


def dedup_double_fire(
    rows: list[dict[str, Any]], *, window_s: float = 15.0
) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    anchor: dict[str, tuple[int, Any]] = {}
    for r in rows:
        prompt = (r.get("prompt") or "").strip()
        if not prompt:
            kept.append(r)
            continue
        ts = _parse_ts(r.get("ts"))
        prev = anchor.get(prompt)
        if (
            prev is not None
            and ts is not None
            and prev[1] is not None
            and abs((ts - prev[1]).total_seconds()) <= window_s
        ):
            idx = prev[0]
            if _row_quality(r) > _row_quality(kept[idx]):
                kept[idx] = r
                anchor[prompt] = (idx, ts)
            continue
        kept.append(r)
        anchor[prompt] = (len(kept) - 1, ts)
    return kept


def referenced_rate(state_dir, rows: list[dict[str, Any]]) -> dict[str, Any]:
    surfaced: dict[str, Any] = {}
    for r in rows:
        ts = _parse_ts(r.get("ts"))
        for h in r.get("hits") or []:
            hid = h.get("id")
            if not hid:
                continue
            cur = surfaced.get(hid)
            if cur is None or (ts is not None and (cur is None or ts < cur)):
                surfaced[hid] = ts
    if not surfaced:
        return {"referenced_rate": None, "surfaced": 0, "referenced": 0}
    uses: dict[str, list[Any]] = {}
    for u in read_usage_log(state_dir):
        uid = u.get("id")
        if uid:
            uses.setdefault(uid, []).append(_parse_ts(u.get("ts")))
    referenced = 0
    for hid, sts in surfaced.items():
        for uts in uses.get(hid, []):
            if sts is None or uts is None or uts >= sts:
                referenced += 1
                break
    return {
        "referenced_rate": round(referenced / len(surfaced), 3),
        "surfaced": len(surfaced),
        "referenced": referenced,
    }


_INTERROGATIVE = (
    "?",
    "¿",
    "qué",
    "que ",
    "cómo",
    "como ",
    "por qué",
    "porqué",
    "cuál",
    "cual",
    "cuándo",
    "cuando",
    "quién",
    "quien",
    "dónde",
    "donde",
    "what",
    "how",
    "why",
    "which",
    "when",
    "who",
    "where",
    "recordá",
    "recorda",
    "acordá",
    "sabés",
    "sabes",
    "explica",
    "explicá",
    "explain",
    "decidimos",
    "prefer",
)
# Leading verbs that mark a mechanical/coding turn unlikely to draw on durable
# memory (it draws on the codebase / current context instead).
_MECHANICAL_LEAD = (
    "arregla",
    "arreglá",
    "implementa",
    "implementá",
    "corré",
    "corre ",
    "ejecuta",
    "edita",
    "editá",
    "agrega",
    "agregá",
    "fix",
    "add ",
    "run ",
    "build",
    "refactor",
    "commit",
    "push",
    "test",
    "crea ",
    "creá",
    "borra",
    "borrá",
    "delete",
    "rename",
)


def _is_knowledge_prompt(prompt: str) -> bool:
    """Heuristic: would this turn plausibly draw on durable memory? Knowledge-
    seeking (questions, "what did we decide", recall) -> yes; slash commands and
    mechanical coding imperatives -> no. Used to segment the utility metric."""
    p = (prompt or "").strip().lower()
    if not p or p.startswith("/"):
        return False
    if any(p.startswith(v) for v in _MECHANICAL_LEAD):
        return False
    return any(tok in p for tok in _INTERROGATIVE) or len(p) >= 60


def grounded_rate(state_dir) -> dict[str, Any]:
    """Did surfaced memories actually get USED in the answer?

    Honest denominator: a surfaced memory only counts if its turn was actually
    grounding-scored (the Stop hook ran for it). Surfaced memories on un-scored
    turns — other consumers / bails / turns the Stop hook never reached — are
    "not measured", NOT "not used", so they are excluded rather than counted as
    misses (which would crush the rate). Coverage is reported alongside so the
    exclusion is transparent.

    Returns both the per-memory rate (`grounded_rate`) and the per-answer rate
    (`answer_rate` = turns that used ≥1 memory / turns measured), the latter
    matching the "answers WITH memo vs WITHOUT" framing.
    """
    # Merge recall_hook.log (session-aware, started after commit 67b8507) with
    # recall.log (rolling, has older session entries not yet in hook log).
    # Deduplication by (session_id, turn) keeps the richer entry.
    _hook_rows = read_recall_hook_log(state_dir)
    _recent_rows = read_recall_log(state_dir, limit=2000)
    _seen: set[tuple[str, int]] = set()
    rows: list[dict[str, Any]] = []
    for r in _hook_rows + _recent_rows:
        _turn_val = r.get("turn")
        _turn_int: int = _turn_val if isinstance(_turn_val, int) else -1
        key: tuple[str, int] = (r.get("session_id") or "", _turn_int)
        if key[0] and key[1] >= 0 and key not in _seen:
            _seen.add(key)
            rows.append(r)
    surfaced_by_turn: dict[tuple[str, int], set[str]] = {}
    prompt_by_turn: dict[tuple[str, int], str] = {}
    for r in rows:
        sid = r.get("session_id")
        turn = r.get("turn")
        if not sid or not isinstance(turn, int):
            continue
        if r.get("prompt"):
            prompt_by_turn.setdefault((sid, turn), str(r.get("prompt")))
        for h in r.get("hits") or []:
            hid = h.get("id")
            if hid:
                surfaced_by_turn.setdefault((sid, turn), set()).add(hid)

    # Turns the grounding detector actually scored, and the grounded keys.
    scored_turns: set[tuple[str, int]] = set()
    grounded_keys: set[tuple[str, int, str]] = set()
    grounding_last_seen: str | None = None
    for g in read_grounding_log(state_dir):
        sid = g.get("session_id")
        turn = g.get("turn")
        if not sid or not isinstance(turn, int):
            continue
        scored_turns.add((sid, turn))
        ts = g.get("ts")
        if isinstance(ts, str) and (grounding_last_seen is None or ts > grounding_last_seen):
            grounding_last_seen = ts
        rid = g.get("recall_id")
        if rid and grounding_used(g):
            grounded_keys.add((sid, turn, rid))

    total_surfaced = sum(len(ids) for ids in surfaced_by_turn.values())
    surfaced_turns = len(surfaced_by_turn)
    measured = {k: ids for k, ids in surfaced_by_turn.items() if k in scored_turns}
    measured_turns = len(measured)
    measured_surfaced = sum(len(ids) for ids in measured.values())
    measurement_coverage = round(measured_turns / surfaced_turns, 3) if surfaced_turns else None
    grounding_age_hours = None
    if grounding_last_seen:
        dt = _parse_ts(grounding_last_seen)
        if dt is not None:
            from datetime import UTC, datetime

            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)
            grounding_age_hours = round((datetime.now(UTC) - dt).total_seconds() / 3600, 2)
    if not measured_surfaced:
        return {
            "grounded_rate": None,
            "surfaced": 0,
            "grounded": 0,
            "answer_rate": None,
            "answers_total": 0,
            "answers_grounded": 0,
            "answer_rate_knowledge": None,
            "answers_knowledge_total": 0,
            "answers_knowledge_grounded": 0,
            "surfaced_turns": surfaced_turns,
            "measured_turns": 0,
            "measurement_coverage": measurement_coverage,
            "grounding_last_seen": grounding_last_seen,
            "grounding_age_hours": grounding_age_hours,
            "unmeasured_surfaced": total_surfaced,
        }

    grounded = sum(
        1
        for (sid, turn), ids in measured.items()
        for hid in ids
        if (sid, turn, hid) in grounded_keys
    )
    answers_grounded = sum(
        1
        for (sid, turn), ids in measured.items()
        if any((sid, turn, hid) in grounded_keys for hid in ids)
    )
    # Segment: of answers that COULD plausibly use memo (knowledge-seeking, not
    # mechanical edits/commands), how many did? Mechanical turns dilute the
    # overall rate, so this is the truer "is memo being used when it matters".
    knowledge = {
        k: ids for k, ids in measured.items() if _is_knowledge_prompt(prompt_by_turn.get(k, ""))
    }
    knowledge_grounded = sum(
        1
        for (sid, turn), ids in knowledge.items()
        if any((sid, turn, hid) in grounded_keys for hid in ids)
    )
    return {
        "grounded_rate": round(grounded / measured_surfaced, 3),
        "surfaced": measured_surfaced,
        "grounded": grounded,
        "answer_rate": round(answers_grounded / len(measured), 3),
        "answers_total": len(measured),
        "answers_grounded": answers_grounded,
        "answer_rate_knowledge": round(knowledge_grounded / len(knowledge), 3)
        if knowledge
        else None,
        "answers_knowledge_total": len(knowledge),
        "answers_knowledge_grounded": knowledge_grounded,
        "surfaced_turns": surfaced_turns,
        "measured_turns": measured_turns,
        "measurement_coverage": measurement_coverage,
        "grounding_last_seen": grounding_last_seen,
        "grounding_age_hours": grounding_age_hours,
        "unmeasured_surfaced": total_surfaced - measured_surfaced,
    }


def recall_health(state_dir, *, limit: int = 200) -> dict[str, Any]:
    # Scope the recall-hook health story to ambient recall: an agent's explicit
    # eval searches (cli:search/mcp:*) are real consults but they are NOT "memo
    # deciding to activate on your prompt", so counting them as the funnel
    # denominator makes the activation rate read 6% instead of the true ~86%.
    #
    # Source of truth = recall_hook.log: durable (cap 2000), ambient-by-
    # construction (only the session-aware recall-hook writes it), and never
    # flooded by an agent's cli:search traffic the way the shared recall.log is.
    # Bails aren't carried in the hook log, so fold in recent ambient bail rows
    # from recall.log. Fall back to recall.log (ambient-filtered) when the hook
    # log is empty — older runtimes / tests that log via= without a session_id.
    hook_rows = dedup_double_fire(
        [r for r in read_recall_hook_log(state_dir, limit=2000) if is_ambient_recall(r)]
    )
    shared = dedup_double_fire(read_recall_log(state_dir, limit=max(limit, 2000)))
    if hook_rows:
        bail_rows_shared = [r for r in shared if (r.get("via") or "").strip().lower() == "bail"]
        rows = hook_rows + bail_rows_shared
    else:
        rows = [r for r in shared if is_ambient_recall(r)]
    # Drop eval-probe sessions (single-turn throwaways an eval harness spawns);
    # the funnel/verdict should reflect YOUR real working sessions only.
    rows = filter_real_sessions(state_dir, rows)
    fired = [r for r in rows if r.get("via") in ("daemon", "subprocess")]
    bail_rows = [r for r in rows if r.get("via") == "bail"]
    bailed = len(bail_rows)
    with_hits = [r for r in fired if r.get("hits")]

    top_scores: list[float] = []
    strong = 0
    for r in with_hits:
        hits = r.get("hits") or []
        score = hits[0].get("score") if hits else None
        if isinstance(score, (int, float)):
            top_scores.append(float(score))
            if float(score) > STRONG_SCORE:
                strong += 1
    top_scores.sort()
    lats = sorted(
        int(r["latency_ms"]) for r in fired if isinstance(r.get("latency_ms"), (int, float))
    )

    ref = referenced_rate(state_dir, rows)
    grounded = grounded_rate(state_dir)
    return {
        "sampled": len(rows),
        "fired": len(fired),
        "bailed": bailed,
        "bail_breakdown": _bail_breakdown(bail_rows),
        "hit_rate": round(len(with_hits) / len(fired), 3) if fired else None,
        "strong_hit_rate": round(strong / len(fired), 3) if fired else None,
        "grounded_rate": grounded["grounded_rate"],
        "grounded": grounded["grounded"],
        "grounded_surfaced": grounded["surfaced"],
        "answer_rate": grounded["answer_rate"],
        "answers_total": grounded["answers_total"],
        "answers_grounded": grounded["answers_grounded"],
        "answer_rate_knowledge": grounded["answer_rate_knowledge"],
        "answers_knowledge_total": grounded["answers_knowledge_total"],
        "answers_knowledge_grounded": grounded["answers_knowledge_grounded"],
        "surfaced_turns": grounded["surfaced_turns"],
        "measured_turns": grounded["measured_turns"],
        "measurement_coverage": grounded["measurement_coverage"],
        "grounding_last_seen": grounded["grounding_last_seen"],
        "grounding_age_hours": grounded["grounding_age_hours"],
        "unmeasured_surfaced": grounded["unmeasured_surfaced"],
        "referenced_rate": ref["referenced_rate"],
        "referenced": ref["referenced"],
        "surfaced": ref["surfaced"],
        "median_top_score": round(_median(top_scores), 3) if top_scores else None,
        "p50_latency_ms": _median(lats),
    }


# Rows produced by the auto-firing recall-hook (the user's ambient recall),
# as opposed to an explicit tool/agent search (synapse `cli:search`, `mcp:*`).
# The "does memo work as YOUR memory?" story — funnel, gaps, verdict volume — scopes
# to these so an agent's eval traffic (e.g. synapse hammering memo with a generic
# eval corpus) can't distort the picture. The "who uses memo?" panel still
# counts every reader via consult_breakdown.
_AMBIENT_VIA = frozenset({"daemon", "subprocess", "bail", "daemon_error"})


def is_ambient_recall(row: dict[str, Any]) -> bool:
    return (row.get("via") or "").strip().lower() in _AMBIENT_VIA


def filter_real_sessions(state_dir, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep rows from real working sessions, dropping throwaway single-turn
    sessions an eval harness spawns. synapse's eval fires a generic corpus
    question (TCP vs UDP, git rebase…) into a fresh claude-code session that
    fires once (turn_count == 1) and vanishes; those arrive through the ambient
    recall-hook, so `is_ambient_recall` alone can't tell them from your own
    prompts.

    A session is "real" if it has ≥ 2 turns — judged by the durable session
    snapshot's turn_count, OR (when no snapshot exists) by contributing ≥ 2 rows
    to this window. The row-count fallback keeps the signal working without a
    session-store read and degrades gracefully on older runtimes. Rows with no
    session attribution (bails, pre-session-id runtimes) are kept — we can't
    prove they are probes and dropping them would hide real activity."""
    from memo.session import get_session

    row_counts: dict[str, int] = {}
    for r in rows:
        sid = r.get("session_id")
        if sid:
            row_counts[sid] = row_counts.get(sid, 0) + 1

    snap_turns: dict[str, int] = {}
    out: list[dict[str, Any]] = []
    for r in rows:
        sid = r.get("session_id")
        if not sid:
            out.append(r)
            continue
        if row_counts.get(sid, 0) >= 2:
            out.append(r)
            continue
        if sid not in snap_turns:
            snap = get_session(state_dir, sid) or {}
            snap_turns[sid] = int(snap.get("turn_count") or 0)
        if snap_turns[sid] >= 2:
            out.append(r)
    return out


def consumer_label(row: dict[str, Any]) -> str:
    client = (row.get("client") or "").strip().lower()
    if client:
        return client
    source = (row.get("source") or "").strip().lower()
    if source:
        return source
    via = (row.get("via") or "").strip().lower()
    if via in ("daemon", "subprocess", "daemon_error"):
        return "claude-code"
    if via.startswith("mcp:"):
        return "mcp:unknown"
    if via == "bail":
        return "claude-code"
    return "unknown"


def consult_breakdown(state_dir, *, limit: int = 500) -> dict[str, Any]:
    rows = dedup_double_fire(read_recall_log(state_dir, limit=limit))
    # scored_turns = turns the grounding detector actually measured; grounded_keys
    # = the subset where a memory was used. Per-consumer grounded_rate divides by
    # SCORED-surfaced only, mirroring the honest denominator in grounded_rate():
    # a memory surfaced on an un-scored turn is "not measured", not "not used".
    scored_turns: set[tuple[str, int]] = set()
    grounded_keys: set[tuple[str, int, str]] = set()
    for g in read_grounding_log(state_dir):
        sid = g.get("session_id")
        turn = g.get("turn")
        rid = g.get("recall_id")
        if sid and isinstance(turn, int):
            scored_turns.add((sid, turn))
            if rid and grounding_used(g):
                grounded_keys.add((sid, turn, rid))

    by: dict[str, dict[str, Any]] = {}
    for r in rows:
        name = consumer_label(r)
        agg = by.setdefault(
            name,
            {
                "consults": 0,
                "fired": 0,
                "with_hits": 0,
                "strong": 0,
                "top_scores": [],
                "last_seen": None,
                "surfaced": set(),
                "grounded": set(),
            },
        )
        agg["consults"] += 1
        is_bail = (r.get("via") or "") == "bail"
        if not is_bail:
            agg["fired"] += 1
            hits = r.get("hits") or []
            if hits:
                agg["with_hits"] += 1
                score = hits[0].get("score")
                if isinstance(score, (int, float)):
                    agg["top_scores"].append(float(score))
                    if float(score) > STRONG_SCORE:
                        agg["strong"] += 1
            sid = r.get("session_id")
            turn = r.get("turn")
            if sid and isinstance(turn, int) and (sid, turn) in scored_turns:
                for h in hits:
                    hid = h.get("id")
                    if not hid:
                        continue
                    key = (sid, turn, hid)
                    agg["surfaced"].add(key)
                    if key in grounded_keys:
                        agg["grounded"].add(key)
        ts = r.get("ts")
        if ts and (agg["last_seen"] is None or ts > agg["last_seen"]):
            agg["last_seen"] = ts

    consumers: list[dict[str, Any]] = []
    for name, agg in by.items():
        scores = sorted(agg["top_scores"])
        fired = agg["fired"]
        n_surfaced = len(agg["surfaced"])
        consumers.append(
            {
                "consumer": name,
                "consults": agg["consults"],
                "fired": fired,
                "bailed": agg["consults"] - fired,
                "hit_rate": round(agg["with_hits"] / fired, 3) if fired else None,
                "strong_hit_rate": round(agg["strong"] / fired, 3) if fired else None,
                "grounded_rate": round(len(agg["grounded"]) / n_surfaced, 3)
                if n_surfaced
                else None,
                "grounded_surfaced": n_surfaced,
                "median_top_score": round(_median(scores), 3) if scores else None,
                "last_seen": agg["last_seen"],
            }
        )
    consumers.sort(key=lambda c: c["consults"], reverse=True)

    # A consumer is silent if absent from both the rolling window AND the
    # persistent last-seen tracker (within 30 days).  This prevents false
    # positives when a low-frequency consumer (devin, opencode) scrolls out of
    # the rolling recall.log cap but queried memo within the last month.
    from datetime import UTC, datetime, timedelta

    from memo.dashboard_logs import read_consumer_last_seen

    last_seen = read_consumer_last_seen(state_dir)
    cutoff = (datetime.now(UTC) - timedelta(days=30)).isoformat(timespec="seconds")
    recently_seen = {c for c, ts in last_seen.items() if ts >= cutoff}
    active = set(by) | recently_seen
    silent = [c for c in EXPECTED_CONSUMERS if c not in active]
    return {"sampled": len(rows), "consumers": consumers, "silent": silent}


def verdict(state_dir, *, limit: int = 500) -> dict[str, Any]:
    """One-glance answer: does memo work, and who actually reads it?

    Combines read volume (consults) with outcome (grounded_rate) into a single
    status, plus a per-expected-consumer read/silent breakdown. Pure derivation
    over the recall + grounding logs — no side effects.
    """
    health = recall_health(state_dir, limit=limit)
    cb = consult_breakdown(state_dir, limit=limit)
    daily = read_daily_trend(state_dir)
    consults_total = sum(int(row.get("consultas") or 0) for row in daily.values())
    activations_total = sum(int(row.get("activado") or 0) for row in daily.values())
    # Volume that gates the verdict = ambient prompts (yours), not the all-tools
    # total. "Does memo work as YOUR memory?" needs enough of YOUR usage to judge;
    # an agent's eval flood shouldn't unlock or distort the verdict. The all-tools
    # per-consumer total stays in consult_breakdown for "who uses memo?".
    consults_sampled = int(health.get("sampled") or 0)
    grounded = health.get("grounded_rate")
    measured_turns = int(health.get("measured_turns") or 0)
    measurement_coverage = health.get("measurement_coverage")

    if consults_sampled < VERDICT_MIN_CONSULTS:
        status, label = "unused", "❌ NOT USED"
    elif (
        measured_turns < VERDICT_MIN_MEASURED_TURNS
        or (measurement_coverage or 0.0) < VERDICT_MIN_MEASUREMENT_COVERAGE
    ):
        status, label = "unmeasured", "⚠️ READ BUT NOT MEASURED"
    elif (grounded or 0.0) < VERDICT_MIN_GROUNDED:
        status, label = "weak", "⚠️ READ BUT NOT HELPING"
    else:
        status, label = "ok", "✅ USEFUL"

    readers = [str(c.get("consumer")) for c in cb.get("consumers", [])]
    silent = list(cb.get("silent") or [])
    per_consumer_names = list(EXPECTED_CONSUMERS)
    for name in readers:
        if name not in per_consumer_names:
            per_consumer_names.append(name)
    per_consumer = [{"name": name, "reads": name in readers} for name in per_consumer_names]
    return {
        "status": status,
        "label": label,
        "consults": consults_sampled,
        "consults_sampled": consults_sampled,
        "consults_total": consults_total,
        "activations_total": activations_total,
        "activations_sampled": int(health.get("fired") or 0),
        "grounded_rate": grounded,
        "hit_rate": health.get("hit_rate"),
        "measurement_coverage": measurement_coverage,
        "measured_turns": measured_turns,
        "surfaced_turns": health.get("surfaced_turns"),
        "grounding_last_seen": health.get("grounding_last_seen"),
        "grounding_age_hours": health.get("grounding_age_hours"),
        "grounding_diag": read_grounding_diag_log(state_dir, limit=5),
        "readers": readers,
        "silent": silent,
        "per_consumer": per_consumer,
    }


_REASK_TOKEN_RE = re.compile(r"[a-z0-9]{3,}")
_REASK_STOP = frozenset(
    [
        "the",
        "and",
        "for",
        "that",
        "with",
        "this",
        "from",
        "have",
        "are",
        "was",
        "were",
        "has",
        "not",
        "but",
        "you",
        "your",
        "una",
        "los",
        "las",
        "del",
        "por",
        "con",
        "para",
        "como",
        "que",
        "esta",
        "este",
        "más",
    ]
)


def _reask_tokens(text: str) -> set[str]:
    return {t for t in _REASK_TOKEN_RE.findall((text or "").lower()) if t not in _REASK_STOP}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def ablation_stats(
    state_dir, *, limit: int = 2000, window_turns: int = 4
) -> dict[str, Any]:
    """With-vs-without-recall cohort comparison over the live logs.

    Cohorts come from recall_hook.log ``via``: "disabled" (the instrumented
    MEMO_RECALL_DISABLE short-circuit) vs daemon/subprocess. Same cohort
    pattern as dream_tune_online.cohort_fraction, keyed on ``via`` instead of
    ``params_version``. Reports per-cohort turn counts, the grounded-per-turn
    rate for the ON cohort (OFF is 0 by construction), and a same-session
    re-ask rate per cohort — the honest live with-vs-without delta."""
    from memo.dashboard import read_grounding_log, read_recall_hook_log

    rows = read_recall_hook_log(state_dir, limit=limit)
    on_rows = [r for r in rows if r.get("via") in ("daemon", "subprocess")]
    off_rows = [r for r in rows if r.get("via") == "disabled"]

    grounded_turns: set[tuple[str, int]] = set()
    for g in read_grounding_log(state_dir, limit=4000):
        s = g.get("used_score")
        if (
            isinstance(s, (int, float))
            and float(s) >= GROUNDED_SCORE
            and g.get("session_id")
            and isinstance(g.get("turn"), int)
        ):
            grounded_turns.add((str(g["session_id"]), int(g["turn"])))
    on_turns = {
        (str(r["session_id"]), int(r["turn"]))
        for r in on_rows
        if r.get("session_id") and isinstance(r.get("turn"), int)
    }
    grounded_on = len(on_turns & grounded_turns)

    def _reask_rate(cohort_rows: list[dict[str, Any]]) -> float | None:
        """Fraction of prompts that re-ask (token-Jaccard >= 0.6) an earlier
        prompt of the same session within ``window_turns`` turns."""
        seen: dict[str, list[tuple[int, set[str]]]] = {}
        considered = reasked = 0
        for r in cohort_rows:
            sid, turn = r.get("session_id"), r.get("turn")
            prompt = str(r.get("prompt") or "")
            if not sid or not isinstance(turn, int) or len(prompt) < 8:
                continue
            tok = _reask_tokens(prompt)
            considered += 1
            if any(
                0 < turn - pt <= window_turns and _jaccard(tok, ptok) >= 0.6
                for pt, ptok in seen.get(str(sid), [])
            ):
                reasked += 1
            seen.setdefault(str(sid), []).append((turn, tok))
        return round(reasked / considered, 3) if considered else None

    return {
        "turns_on": len(on_rows),
        "turns_off": len(off_rows),
        "grounded_turns_on": grounded_on,
        "grounded_per_turn_on": round(grounded_on / len(on_turns), 3) if on_turns else None,
        "reask_rate_on": _reask_rate(on_rows),
        "reask_rate_off": _reask_rate(off_rows),
    }


def reask_stats(
    state_dir,
    *,
    window_turns: int = 4,
    sim_threshold: float = 0.6,
    limit: int = 500,
) -> dict[str, Any]:
    rows = dedup_double_fire(read_recall_log(state_dir, limit=limit))
    grounded_turns: set[tuple[str, int]] = set()
    for g in read_grounding_log(state_dir):
        sid = g.get("session_id")
        turn = g.get("turn")
        score = g.get("used_score")
        if (
            sid
            and isinstance(turn, int)
            and isinstance(score, (int, float))
            and float(score) >= GROUNDED_SCORE
        ):
            grounded_turns.add((sid, turn))

    by_session: dict[str, list[tuple[int, str]]] = {}
    for r in rows:
        sid = r.get("session_id")
        turn = r.get("turn")
        prompt = (r.get("prompt") or "").strip()
        if sid and isinstance(turn, int) and prompt:
            by_session.setdefault(sid, []).append((turn, prompt))
    for lst in by_session.values():
        lst.sort(key=lambda x: x[0])

    considered = 0
    reask = 0
    for sid, turn in grounded_turns:
        timeline = by_session.get(sid) or []
        this = next((p for t, p in timeline if t == turn), None)
        if not this:
            continue
        considered += 1
        this_tok = _reask_tokens(this)
        recurred = any(
            t > turn
            and (t - turn) <= window_turns
            and _jaccard(this_tok, _reask_tokens(p)) >= sim_threshold
            for t, p in timeline
        )
        if recurred:
            reask += 1
    reask_avoided = considered - reask
    return {
        "considered": considered,
        "reask": reask,
        "reask_avoided": reask_avoided,
        "reask_rate": round(reask / considered, 3) if considered else None,
    }
