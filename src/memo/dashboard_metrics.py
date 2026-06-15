from __future__ import annotations

import re
from typing import Any

from memo.dashboard_logs import read_grounding_log, read_recall_log, read_usage_log

STRONG_SCORE = 0.85
GROUNDED_SCORE = 0.5
# Verdict thresholds — drive the "¿funciona memo?" panel. memo is judged USEFUL
# only when it is both read enough (volume) and actually helping (grounded).
VERDICT_MIN_CONSULTS = 20
VERDICT_MIN_GROUNDED = 0.10
EXPECTED_CONSUMERS = (
    "claude-code",
    "synapse",
    "memflow",
    "codex",
    "devin",
    "opencode",
    "windsurf",
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


def dedup_double_fire(rows: list[dict[str, Any]], *, window_s: float = 15.0) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    anchor: dict[str, tuple[int, Any]] = {}
    for r in rows:
        prompt = (r.get("prompt") or "").strip()
        if not prompt:
            kept.append(r)
            continue
        ts = _parse_ts(r.get("ts"))
        prev = anchor.get(prompt)
        if prev is not None and ts is not None and prev[1] is not None and abs((ts - prev[1]).total_seconds()) <= window_s:
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


def grounded_rate(state_dir, rows: list[dict[str, Any]]) -> dict[str, Any]:
    surfaced: set[tuple[str, int, str]] = set()
    for r in rows:
        sid = r.get("session_id")
        turn = r.get("turn")
        if not sid or not isinstance(turn, int):
            continue
        for h in r.get("hits") or []:
            hid = h.get("id")
            if hid:
                surfaced.add((sid, turn, hid))
    if not surfaced:
        return {"grounded_rate": None, "surfaced": 0, "grounded": 0}
    grounded_keys: set[tuple[str, int, str]] = set()
    for g in read_grounding_log(state_dir):
        sid = g.get("session_id")
        turn = g.get("turn")
        rid = g.get("recall_id")
        score = g.get("used_score")
        if sid and isinstance(turn, int) and rid and isinstance(score, (int, float)) and float(score) >= GROUNDED_SCORE:
            grounded_keys.add((sid, turn, rid))
    grounded = sum(1 for k in surfaced if k in grounded_keys)
    return {
        "grounded_rate": round(grounded / len(surfaced), 3),
        "surfaced": len(surfaced),
        "grounded": grounded,
    }


def recall_health(state_dir, *, limit: int = 200) -> dict[str, Any]:
    rows = dedup_double_fire(read_recall_log(state_dir, limit=limit))
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
    lats = sorted(int(r["latency_ms"]) for r in fired if isinstance(r.get("latency_ms"), (int, float)))

    def _median(xs: list[Any]) -> Any:
        return xs[len(xs) // 2] if xs else None

    ref = referenced_rate(state_dir, rows)
    grounded = grounded_rate(state_dir, rows)
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
        "referenced_rate": ref["referenced_rate"],
        "referenced": ref["referenced"],
        "surfaced": ref["surfaced"],
        "median_top_score": round(_median(top_scores), 3) if top_scores else None,
        "p50_latency_ms": _median(lats),
    }


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
    grounded_keys: set[tuple[str, int, str]] = set()
    for g in read_grounding_log(state_dir):
        sid = g.get("session_id")
        turn = g.get("turn")
        rid = g.get("recall_id")
        score = g.get("used_score")
        if sid and isinstance(turn, int) and rid and isinstance(score, (int, float)) and float(score) >= GROUNDED_SCORE:
            grounded_keys.add((sid, turn, rid))

    by: dict[str, dict[str, Any]] = {}
    for r in rows:
        name = consumer_label(r)
        agg = by.setdefault(name, {"consults": 0, "fired": 0, "with_hits": 0, "strong": 0, "top_scores": [], "last_seen": None, "surfaced": set(), "grounded": set()})
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
            if sid and isinstance(turn, int):
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
                "grounded_rate": round(len(agg["grounded"]) / n_surfaced, 3) if n_surfaced else None,
                "grounded_surfaced": n_surfaced,
                "median_top_score": round(scores[len(scores) // 2], 3) if scores else None,
                "last_seen": agg["last_seen"],
            }
        )
    consumers.sort(key=lambda c: c["consults"], reverse=True)
    silent = [c for c in EXPECTED_CONSUMERS if c not in by]
    return {"sampled": len(rows), "consumers": consumers, "silent": silent}


def verdict(state_dir, *, limit: int = 500) -> dict[str, Any]:
    """One-glance answer: does memo work, and who actually reads it?

    Combines read volume (consults) with outcome (grounded_rate) into a single
    status, plus a per-expected-consumer read/silent breakdown. Pure derivation
    over the recall + grounding logs — no side effects.
    """
    health = recall_health(state_dir, limit=min(limit, 200))
    cb = consult_breakdown(state_dir, limit=limit)
    consults = int(cb.get("sampled") or 0)
    grounded = health.get("grounded_rate")

    if consults < VERDICT_MIN_CONSULTS:
        status, label = "unused", "❌ NO SE USA"
    elif (grounded or 0.0) < VERDICT_MIN_GROUNDED:
        status, label = "weak", "⚠️ SE LEE PERO NO AYUDA"
    else:
        status, label = "ok", "✅ ÚTIL"

    readers = [str(c.get("consumer")) for c in cb.get("consumers", [])]
    silent = list(cb.get("silent") or [])
    per_consumer = [
        {"name": name, "reads": name in readers} for name in EXPECTED_CONSUMERS
    ]
    return {
        "status": status,
        "label": label,
        "consults": consults,
        "grounded_rate": grounded,
        "hit_rate": health.get("hit_rate"),
        "readers": readers,
        "silent": silent,
        "per_consumer": per_consumer,
    }


_REASK_TOKEN_RE = re.compile(r"[a-z0-9]{3,}")
_REASK_STOP = frozenset(["the", "and", "for", "that", "with", "this", "from", "have", "are", "was", "were", "has", "not", "but", "you", "your", "una", "los", "las", "del", "por", "con", "para", "como", "que", "esta", "este", "más"])


def _reask_tokens(text: str) -> set[str]:
    return {t for t in _REASK_TOKEN_RE.findall((text or "").lower()) if t not in _REASK_STOP}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


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
        if sid and isinstance(turn, int) and isinstance(score, (int, float)) and float(score) >= GROUNDED_SCORE:
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
        recurred = any(t > turn and (t - turn) <= window_turns and _jaccard(this_tok, _reask_tokens(p)) >= sim_threshold for t, p in timeline)
        if recurred:
            reask += 1
    reask_avoided = considered - reask
    return {
        "considered": considered,
        "reask": reask,
        "reask_avoided": reask_avoided,
        "reask_rate": round(reask / considered, 3) if considered else None,
    }
