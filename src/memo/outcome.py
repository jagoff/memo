"""The Outcome Loop — memo learns from whether its recalls actually got USED.

Closes the recall → use → outcome loop. The grounding detector already records,
per turn, whether a surfaced memoria was actually used in the answer
(``grounding.log``). Today that signal dies in the dashboard while ranking's
``memory_health.roi_score`` is driven by mere ACCESS (every surfacing boosts it
— so noise that keeps showing up ranks higher). This module turns the outcome
signal into:

  - a per-memoria UTILITY score (Bayesian-smoothed grounded/surfaced) reconciled
    into ``roi_score`` so ranking promotes memorias that ground answers and
    demotes ones that surface but never help (:func:`reconcile_roi`);
  - a KNOWLEDGE-GAP report — prompts where recall bailed / returned nothing /
    was never grounded: what memo could not answer, i.e. what to capture next
    (:func:`detect_gaps`);
  - DEAD-WEIGHT detection — memorias surfaced often yet never grounded, for
    reversible archival by ``memo maintain`` (:func:`dead_weight`).

Pure reads over recall.log + grounding.log, plus a single roi write on reconcile.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

_DEFAULT_PRIOR_N = 3.0
_DEFAULT_ROI_FLOOR = 0.6
_DEFAULT_ROI_CAP = 1.5


def compute_utilities(
    state_dir: Path,
    *,
    prior_n: float = _DEFAULT_PRIOR_N,
    recall_limit: int = 2000,
    grounding_limit: int = 4000,
) -> dict[str, Any]:
    """Per-memoria utility from real outcomes, keyed by 8-char id prefix.

    ``surfaced`` = distinct (session, turn) the memoria was shown in;
    ``grounded`` = of those, how many the answer actually used (intersection,
    so utility never exceeds 1). ``utility`` is Bayesian-smoothed toward the
    global grounded rate so a memoria seen once isn't judged on one data point::

        utility = (grounded + prior_mean * prior_n) / (surfaced + prior_n)
    """
    from memo.dashboard import grounding_used, read_grounding_log, read_recall_log

    surfaced: dict[str, set[tuple[str, int]]] = defaultdict(set)
    for r in read_recall_log(state_dir, limit=recall_limit):
        sid, turn = r.get("session_id"), r.get("turn")
        if not (sid and isinstance(turn, int)):
            continue
        for h in r.get("hits") or []:
            pid = (h.get("id") or "")[:8]
            if pid:
                surfaced[pid].add((sid, turn))

    grounded: dict[str, set[tuple[str, int]]] = defaultdict(set)
    for g in read_grounding_log(state_dir, limit=grounding_limit):
        sid, turn = g.get("session_id"), g.get("turn")
        pid = (g.get("recall_id") or "")[:8]
        if not (sid and isinstance(turn, int) and pid):
            continue
        if grounding_used(g):
            grounded[pid].add((sid, turn))

    total_surf = sum(len(v) for v in surfaced.values())
    total_grnd = sum(len(grounded.get(p, set()) & s) for p, s in surfaced.items())
    prior_mean = (total_grnd / total_surf) if total_surf else 0.5

    by_prefix: dict[str, dict[str, Any]] = {}
    for pid, sset in surfaced.items():
        s = len(sset)
        g_count = len(grounded.get(pid, set()) & sset)
        utility = (g_count + prior_mean * prior_n) / (s + prior_n)
        by_prefix[pid] = {"surfaced": s, "grounded": g_count, "utility": round(utility, 4)}
    return {
        "prior_mean": round(prior_mean, 4),
        "surfaced_total": total_surf,
        "grounded_total": total_grnd,
        "by_prefix": by_prefix,
    }


def _prefix_to_id(memory: Any) -> dict[str, str]:
    out: dict[str, str] = {}
    for fid in memory.store.all_ids():
        out.setdefault(fid[:8], fid)
    return out


def reconcile_roi(
    memory: Any,
    *,
    prior_n: float | None = None,
    floor: float | None = None,
    cap: float | None = None,
) -> dict[str, Any]:
    """Recompute ``roi_score`` from grounding outcomes and write it.

    ``roi_score = floor + utility * (cap - floor)`` — utility 0 (surfaced, never
    grounded) → floor; utility 1 → cap; no-data memorias stay near neutral via
    the Bayesian prior. Authoritative absolute write (overwrites access drift).
    """
    from memo.flags import flag_float

    prior_n = prior_n if prior_n is not None else (flag_float("MEMO_OUTCOME_PRIOR_N") or _DEFAULT_PRIOR_N)
    floor = floor if floor is not None else (flag_float("MEMO_OUTCOME_ROI_FLOOR") or _DEFAULT_ROI_FLOOR)
    cap = cap if cap is not None else (flag_float("MEMO_OUTCOME_ROI_CAP") or _DEFAULT_ROI_CAP)
    span = max(0.0, cap - floor)

    u = compute_utilities(memory.cfg.state_dir, prior_n=prior_n)
    by_prefix = u["by_prefix"]
    if not by_prefix:
        return {"updated": 0, "scored": 0, "prior_mean": u["prior_mean"]}

    p2id = _prefix_to_id(memory)
    pairs: list[tuple[str, float]] = []
    for pid, st in by_prefix.items():
        fid = p2id.get(pid)
        if not fid:
            continue
        pairs.append((fid, floor + st["utility"] * span))
    written = memory.store.set_roi_batch(pairs, floor=floor, cap=cap)
    return {
        "updated": written,
        "scored": len(pairs),
        "prior_mean": u["prior_mean"],
        "floor": floor,
        "cap": cap,
    }


def dead_weight(
    memory: Any,
    *,
    min_surfaced: int,
    prior_n: float | None = None,
) -> list[dict[str, Any]]:
    """Memorias surfaced ``>= min_surfaced`` times yet NEVER grounded — recall
    noise that crowds out useful hits. Candidates for reversible archival.
    ``min_surfaced <= 0`` disables (returns [])."""
    from memo.flags import flag_float

    if min_surfaced <= 0:
        return []
    from memo.dashboard_metrics import grounded_rate

    coverage = grounded_rate(memory.cfg.state_dir).get("measurement_coverage")
    if (coverage or 0.0) <= 0.0:
        return []
    prior_n = prior_n if prior_n is not None else (flag_float("MEMO_OUTCOME_PRIOR_N") or _DEFAULT_PRIOR_N)
    u = compute_utilities(memory.cfg.state_dir, prior_n=prior_n)
    p2id = _prefix_to_id(memory)
    out: list[dict[str, Any]] = []
    for pid, st in u["by_prefix"].items():
        if st["surfaced"] >= min_surfaced and st["grounded"] == 0:
            fid = p2id.get(pid)
            if fid:
                out.append({"id": fid, "surfaced": st["surfaced"], "utility": st["utility"]})
    out.sort(key=lambda d: d["surfaced"], reverse=True)
    return out


_NOISE_MARKERS = (
    "<task-notification", "<tool-use-id", "task-id>", "tool-use-id>",
    "system-reminder", "<system", "hook additional context",
)


def _is_injected_noise(prompt: str) -> bool:
    """Drop prompts that are clearly injected machine context (hook blobs, tool
    notifications, system reminders) rather than real user questions — they are
    not knowledge gaps, just telemetry that landed in recall.log."""
    p = prompt.lstrip()
    if p.startswith("<"):
        return True
    low = p.lower()
    return any(m in low for m in _NOISE_MARKERS)


def detect_gaps(
    state_dir: Path,
    *,
    limit: int = 2000,
    sim_threshold: float = 0.6,
    min_count: int = 1,
) -> list[dict[str, Any]]:
    """Knowledge gaps — knowledge-seeking prompts memo could NOT answer.

    A gap is a knowledge prompt whose recall (a) bailed for no-match (NOT for a
    slash command / short prompt — those aren't gaps), (b) returned zero hits,
    or (c) surfaced something but the turn was never grounded (nothing used).
    Similar prompts are clustered by token Jaccard so a repeated question
    surfaces once with a count. Sorted by frequency, then recency.
    """
    from memo.dashboard import grounding_used, read_grounding_log, read_recall_log
    from memo.dashboard_metrics import _is_knowledge_prompt, _jaccard, _reask_tokens

    grounded_turns: set[tuple[str, int]] = set()
    for g in read_grounding_log(state_dir):
        sid, turn = g.get("session_id"), g.get("turn")
        if sid and isinstance(turn, int) and grounding_used(g):
            grounded_turns.add((sid, turn))

    raw: list[dict[str, Any]] = []
    for r in read_recall_log(state_dir, limit=limit):
        prompt = (r.get("prompt") or "").strip()
        if not prompt or not _is_knowledge_prompt(prompt) or _is_injected_noise(prompt):
            continue
        via = r.get("via")
        hits = r.get("hits") or []
        sid, turn = r.get("session_id"), r.get("turn")
        if via == "bail":
            reason_raw = r.get("reason") or ""
            if "min_sim" not in reason_raw and "no hits" not in reason_raw:
                continue  # slash command / short prompt → not a knowledge gap
            reason = "sin coincidencias"
        elif not hits:
            reason = "0 resultados"
        elif sid and isinstance(turn, int) and (sid, turn) not in grounded_turns:
            reason = "encontró algo pero no se usó"
        else:
            continue
        raw.append({"prompt": prompt, "reason": reason, "ts": r.get("ts")})

    clusters: list[dict[str, Any]] = []
    for g in raw:
        tok = _reask_tokens(g["prompt"])
        for c in clusters:
            if _jaccard(tok, c["tokens"]) >= sim_threshold:
                c["count"] += 1
                c["reasons"].add(g["reason"])
                if (g["ts"] or "") > (c["last_seen"] or ""):
                    c["last_seen"] = g["ts"]
                    c["prompt"] = g["prompt"]
                break
        else:
            clusters.append({
                "tokens": tok, "prompt": g["prompt"], "count": 1,
                "reasons": {g["reason"]}, "last_seen": g["ts"],
            })

    out = [
        {"prompt": c["prompt"], "count": c["count"],
         "reasons": sorted(c["reasons"]), "last_seen": c["last_seen"]}
        for c in clusters if c["count"] >= min_count
    ]
    out.sort(key=lambda c: (c["count"], c["last_seen"] or ""), reverse=True)
    return out
