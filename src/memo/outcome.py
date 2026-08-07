"""The Outcome Loop — memo learns from whether its recalls actually got USED.

Closes the recall → use → outcome loop. The grounding detector already records,
per turn, whether a surfaced memory was actually used in the answer
(``grounding.log``). Today that signal dies in the dashboard while ranking's
``memory_health.roi_score`` is driven by mere ACCESS (every surfacing boosts it
— so noise that keeps showing up ranks higher). This module turns the outcome
signal into:

  - a per-memory UTILITY score (Bayesian-smoothed grounded/surfaced) reconciled
    into ``roi_score`` so ranking promotes memories that ground answers and
    demotes ones that surface but never help (:func:`reconcile_roi`);
  - a KNOWLEDGE-GAP report — prompts where recall bailed / returned nothing /
    was never grounded: what memo could not answer, i.e. what to capture next
    (:func:`detect_gaps`);
  - DEAD-WEIGHT detection — memories surfaced often yet never grounded, for
    reversible archival by ``memo maintain`` (:func:`dead_weight`).
  - NEGATIVE-RECALL reinforcement — the ⛔ closed loop: a ``failure_pattern``
    anti-memory that was surfaced yet the mistake repeated (next-turn verdict
    negative/correction) is STRENGTHENED so it surfaces more forcefully; one
    that was surfaced and heeded gets a mild positive
    (:func:`reconcile_negative_recall`).

Pure reads over recall.log + grounding.log + verdict.log, plus roi/confidence
writes on reconcile.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)
_DEFAULT_PRIOR_N = 3.0
_DEFAULT_ROI_FLOOR = 0.6
_DEFAULT_ROI_CAP = 1.5
_DEFAULT_CITED_WEIGHT = 2.0


def compute_utilities(
    state_dir: Path,
    *,
    prior_n: float = _DEFAULT_PRIOR_N,
    cited_weight: float | None = None,
    recall_limit: int = 2000,
    grounding_limit: int = 4000,
) -> dict[str, Any]:
    """Per-memory utility from real outcomes, keyed by 8-char id prefix.

    ``surfaced`` = distinct (session, turn) the memory was shown in;
    ``grounded`` = of those, how many the answer actually used (intersection,
    so utility never exceeds 1). ``utility`` is Bayesian-smoothed toward the
    global grounded rate so a memory seen once isn't judged on one data point::

        utility = (grounded_w + prior_mean * prior_n) / (surfaced + prior_n)

    where ``grounded_w`` counts a turn the answer explicitly CITED the memory
    (grounding ``method == "cited"``, i.e. a literal ``[id]`` reference) as
    ``cited_weight`` grounded observations instead of 1 — a citation is
    stronger evidence of usefulness than lexical/embedding overlap. Utility is
    clamped at 1.0 so the weight sharpens ranking without overflowing the roi
    [floor, cap] range. ``cited_weight=None`` reads ``MEMO_OUTCOME_CITED_WEIGHT``
    (default 2.0); 1.0 restores exact unweighted parity. ``prior_mean`` and the
    reported ``surfaced``/``grounded``/``cited`` counts stay raw (unweighted).
    """
    from memo.dashboard import grounding_used, read_grounding_log, read_recall_log
    from memo.flags import flag_float

    if cited_weight is None:
        _cw = flag_float("MEMO_OUTCOME_CITED_WEIGHT")
        cited_weight = _cw if _cw is not None else _DEFAULT_CITED_WEIGHT

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
    cited: dict[str, set[tuple[str, int]]] = defaultdict(set)
    for g in read_grounding_log(state_dir, limit=grounding_limit):
        sid, turn = g.get("session_id"), g.get("turn")
        pid = (g.get("recall_id") or "")[:8]
        if not (sid and isinstance(turn, int) and pid):
            continue
        if grounding_used(g):
            grounded[pid].add((sid, turn))
            if g.get("method") == "cited":
                cited[pid].add((sid, turn))

    total_surf = sum(len(v) for v in surfaced.values())
    total_grnd = sum(len(grounded.get(p, set()) & s) for p, s in surfaced.items())
    prior_mean = (total_grnd / total_surf) if total_surf else 0.5

    by_prefix: dict[str, dict[str, Any]] = {}
    for pid, sset in surfaced.items():
        s = len(sset)
        g_count = len(grounded.get(pid, set()) & sset)
        c_count = len(cited.get(pid, set()) & sset)
        g_weighted = g_count + (cited_weight - 1.0) * c_count
        utility = min(1.0, (g_weighted + prior_mean * prior_n) / (s + prior_n))
        by_prefix[pid] = {
            "surfaced": s,
            "grounded": g_count,
            "cited": c_count,
            "utility": round(utility, 4),
        }
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
    cited_weight: float | None = None,
) -> dict[str, Any]:
    """Recompute ``roi_score`` from grounding outcomes and write it.

    ``roi_score = floor + utility * (cap - floor)`` — utility 0 (surfaced, never
    grounded) → floor; utility 1 → cap; no-data memories stay near neutral via
    the Bayesian prior. Authoritative absolute write (overwrites access drift).
    ``cited_weight`` passes through to :func:`compute_utilities` (``None`` →
    ``MEMO_OUTCOME_CITED_WEIGHT``).
    """
    from memo.flags import flag_float

    _pn = flag_float("MEMO_OUTCOME_PRIOR_N")
    prior_n = prior_n if prior_n is not None else (_pn if _pn is not None else _DEFAULT_PRIOR_N)
    _fl = flag_float("MEMO_OUTCOME_ROI_FLOOR")
    floor = floor if floor is not None else (_fl if _fl is not None else _DEFAULT_ROI_FLOOR)
    cap = cap if cap is not None else (flag_float("MEMO_OUTCOME_ROI_CAP") or _DEFAULT_ROI_CAP)
    span = max(0.0, cap - floor)

    u = compute_utilities(memory.cfg.state_dir, prior_n=prior_n, cited_weight=cited_weight)
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


# ── negative-recall reinforcement (the ⛔ closed loop) ────────────────────────
#
# How strongly a surfaced ``failure_pattern`` anti-memory's next-turn OUTCOME
# moves its roi. A REPEAT — the ⛔ warning was surfaced yet the mistake happened
# anyway (next-turn verdict ``negative``/``correction``) — is the strong signal:
# the anti-memory must surface MORE forcefully. A HEED — surfaced and the next
# turn was ``positive`` — is a mild positive. The mapping keeps every touched
# anti-memory at or above the 1.0 neutral, so a useful failure_pattern is never
# demoted below a never-surfaced one (unlike a normal memory, a warning that was
# heeded is a SUCCESS, not dead weight).
_NR_REPEAT_WEIGHT = 0.5
_NR_HEED_WEIGHT = 0.15
_NR_ROI_NEUTRAL = 1.0
_NR_ROI_CAP = 1.5
# Restore confidence on a recurring anti-memory that some other signal (a
# contradiction, an OCR-quality stamp) demoted, so it can surface again. Applied
# gradually and only while below cap, so repeated nightly runs converge on the
# cap with no unbounded drift.
_NR_CONF_LIFT = 0.2
_NR_CONF_CAP = 1.0

_NR_REPEAT_VERDICTS = frozenset({"negative", "correction"})


def reconcile_negative_recall(
    memory: Any,
    *,
    repeat_weight: float = _NR_REPEAT_WEIGHT,
    heed_weight: float = _NR_HEED_WEIGHT,
    roi_cap: float = _NR_ROI_CAP,
) -> dict[str, Any]:
    """Close the ⛔ negative-recall loop from next-turn verdicts.

    A ``failure_pattern`` anti-memory that was surfaced (its id appears in a
    next-turn verdict's ``recall_ids``) is reinforced by that turn's outcome:

      - REPEAT (verdict ``negative``/``correction`` — the mistake happened
        despite the ⛔ warning) STRENGTHENS it: roi climbs toward ``roi_cap`` so
        it surfaces more forcefully, and a demoted confidence is restored.
      - HEED (verdict ``positive`` — the warning was acted on) is a mild
        positive: roi lifts slightly above the 1.0 neutral.

    roi is written ABSOLUTELY from the full verdict.log (like
    :func:`reconcile_roi`), so the pass is idempotent and — run AFTER
    ``reconcile_roi`` — is the authoritative roi for anti-memories. This corrects
    ``reconcile_roi``'s grounding-only view, which under-credits a memory that
    was surfaced as a *warning* rather than cited in the answer (surfaced but
    "never grounded" would otherwise read as dead weight and demote it).

    Gated by ``MEMO_NEGATIVE_RECALL_REINFORCE_ENABLED`` (default off → no-op).
    Ids that don't resolve to a stored ``failure_pattern`` are skipped safely.
    """
    from memo.dashboard import read_verdict_log
    from memo.flags import flag_bool
    from memo.negative_recall import FAILURE_PATTERN_TYPE

    if not flag_bool("MEMO_NEGATIVE_RECALL_REINFORCE_ENABLED"):
        return {"enabled": False, "strengthened": 0, "heeded": 0, "scored": 0}

    p2id = _prefix_to_id(memory)
    fp_cache: dict[str, bool] = {}

    def _is_failure_pattern(fid: str) -> bool:
        cached = fp_cache.get(fid)
        if cached is None:
            row = memory.store.get(fid)
            cached = bool(row) and row.get("type") == FAILURE_PATTERN_TYPE
            fp_cache[fid] = cached
        return cached

    # Distinct (session, turn) outcomes per failure_pattern id — deduped so a
    # verdict row re-read on the next nightly run never inflates the count.
    repeats: dict[str, set[tuple[str, int]]] = defaultdict(set)
    heeds: dict[str, set[tuple[str, int]]] = defaultdict(set)
    for v in read_verdict_log(memory.cfg.state_dir):
        verdict = str(v.get("verdict") or "")
        is_repeat = verdict in _NR_REPEAT_VERDICTS
        if not (is_repeat or verdict == "positive"):
            continue
        sid, turn = v.get("session_id"), v.get("turn")
        if not (sid and isinstance(turn, int)):
            continue
        bucket = repeats if is_repeat else heeds
        for raw_id in v.get("recall_ids") or []:
            fid = p2id.get((raw_id or "")[:8])
            if fid is None or not _is_failure_pattern(fid):
                continue  # unknown / non-failure_pattern id → safe skip
            bucket[fid].add((str(sid), turn))

    fp_ids = set(repeats) | set(heeds)
    if not fp_ids:
        return {"enabled": True, "strengthened": 0, "heeded": 0, "scored": 0}

    span = max(0.0, roi_cap - _NR_ROI_NEUTRAL)
    pairs: list[tuple[str, float]] = []
    for fid in fp_ids:
        signal = min(
            1.0,
            len(repeats.get(fid, ())) * repeat_weight + len(heeds.get(fid, ())) * heed_weight,
        )
        pairs.append((fid, _NR_ROI_NEUTRAL + signal * span))
    memory.store.set_roi_batch(pairs, floor=_NR_ROI_NEUTRAL, cap=roi_cap)

    # Restore confidence on actively-recurring anti-memories that a prior signal
    # demoted — a gradual, only-raises lift toward the cap (never lowers a higher
    # existing value), so it converges and leaves untouched memories alone.
    repeat_ids = sorted(repeats)
    restored = 0
    if repeat_ids:
        health = memory.store.get_health_batch(repeat_ids)
        conf_pairs = [
            (fid, min(_NR_CONF_CAP, health[fid]["confidence"] + _NR_CONF_LIFT))
            for fid in repeat_ids
            if fid in health and health[fid]["confidence"] < _NR_CONF_CAP
        ]
        if conf_pairs:
            memory.store.set_confidence_batch(conf_pairs)
            restored = len(conf_pairs)

    return {
        "enabled": True,
        "strengthened": len(repeat_ids),
        "heeded": len([fid for fid in heeds if fid not in repeats]),
        "scored": len(pairs),
        "confidence_restored": restored,
    }


def reconcile_source_feedback(
    memory: Any,
    *,
    max_pairs: int = 500,
    include_negatives: bool = False,
    recall_limit: int = 2000,
    grounding_limit: int = 4000,
) -> dict[str, int]:
    """Mine per-QUERY feedback from grounding outcomes (auto hard-positive
    mining). ``roi_score`` is a single global scalar per memory, so a memory
    that grounds query-cluster A but is noise for B self-cancels. The
    source_feedback table already learns PER QUERY (it stores the query
    embedding and is consulted via kNN in search) but is fed only by manual
    👍/👎. This writes the signal already captured for free:

      grounded (memory, turn)            → implicit positive ("click", +0.08)
      surfaced-but-unused (memory, turn) → implicit negative ("ignore", ×0.7)
                                           — only when include_negatives.

    Keyed on the turn's query embedding, deduped by (memory, prompt), capped
    at ``max_pairs`` to bound embedding cost. ``only_if_absent`` guarantees a
    manual vote is never overwritten. Returns counts.
    """
    from memo.dashboard import grounding_used, read_grounding_log, read_recall_log
    from memo.grounding import _prompt_for_turn

    p2id = _prefix_to_id(memory)
    state_dir = memory.cfg.state_dir

    # grounded (prefix, session, turn) — the answer demonstrably used it.
    grounded: set[tuple[str, str, int]] = set()
    for g in read_grounding_log(state_dir, limit=grounding_limit):
        sid, turn = g.get("session_id"), g.get("turn")
        pid = (g.get("recall_id") or "")[:8]
        if sid and isinstance(turn, int) and pid and grounding_used(g):
            grounded.add((pid, str(sid), turn))

    # surfaced (prefix, session, turn) — shown, for the negative complement.
    surfaced: set[tuple[str, str, int]] = set()
    if include_negatives:
        for r in read_recall_log(state_dir, limit=recall_limit):
            sid, turn = r.get("session_id"), r.get("turn")
            if not (sid and isinstance(turn, int)):
                continue
            for h in r.get("hits") or []:
                pid = (h.get("id") or "")[:8]
                if pid:
                    surfaced.add((pid, str(sid), turn))

    prompt_cache: dict[tuple[str, int], str] = {}

    def _prompt(sid: str, turn: int) -> str:
        key = (sid, turn)
        if key not in prompt_cache:
            prompt_cache[key] = _prompt_for_turn(state_dir, sid, turn)
        return prompt_cache[key]

    written_pos = written_neg = 0
    seen: set[tuple[str, str]] = set()  # (full_id, prompt) — dedup embeds

    def _emit(triples: set[tuple[str, str, int]], rating: str) -> int:
        nonlocal seen
        n = 0
        for pid, sid, turn in triples:
            if written_pos + written_neg + n >= max_pairs:
                break
            fid = p2id.get(pid)
            if not fid:
                continue
            prompt = _prompt(sid, turn)
            if not prompt or len(prompt.strip()) < 8:
                continue
            key = (fid, prompt)
            if key in seen:
                continue
            seen.add(key)
            try:
                memory.feedback_record(
                    source_id=fid, query_text=prompt, rating=rating, only_if_absent=True
                )
                n += 1
            except Exception as exc:
                _log.debug("reconcile_source_feedback: skip %s: %s", fid[:8], exc)
                continue
        return n

    written_pos = _emit(grounded, "click")
    if include_negatives:
        # Only turns where the memory was surfaced but NOT grounded.
        written_neg = _emit(surfaced - grounded, "ignore")
    return {"positives": written_pos, "negatives": written_neg}


def dead_weight(
    memory: Any,
    *,
    min_surfaced: int,
    prior_n: float | None = None,
) -> list[dict[str, Any]]:
    """Memories surfaced ``>= min_surfaced`` times yet NEVER grounded — recall
    noise that crowds out useful hits. Candidates for reversible archival.
    ``min_surfaced <= 0`` disables (returns [])."""
    from memo.flags import flag_float

    if min_surfaced <= 0:
        return []
    from memo.dashboard_metrics import grounded_rate

    coverage = grounded_rate(memory.cfg.state_dir).get("measurement_coverage")
    if (coverage or 0.0) <= 0.0:
        return []
    _pn = flag_float("MEMO_OUTCOME_PRIOR_N")
    prior_n = prior_n if prior_n is not None else (_pn if _pn is not None else _DEFAULT_PRIOR_N)
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
    "<task-notification",
    "<tool-use-id",
    "task-id>",
    "tool-use-id>",
    "system-reminder",
    "<system",
    "hook additional context",
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
    from memo.dashboard_metrics import (
        _is_knowledge_prompt,
        _jaccard,
        _reask_tokens,
        filter_real_sessions,
        is_ambient_recall,
    )

    # scored_turns = turns the grounding detector actually measured; grounded_turns
    # = the subset where a memory was used. A turn that was never scored is
    # "unmeasured", NOT "found-but-unused" — counting it as a gap turns thin
    # measurement coverage into a flood of false gaps. Only a measured-and-unused
    # turn is a real "surfaced but didn't help" gap.
    scored_turns: set[tuple[str, int]] = set()
    grounded_turns: set[tuple[str, int]] = set()
    for g in read_grounding_log(state_dir):
        sid, turn = g.get("session_id"), g.get("turn")
        if sid and isinstance(turn, int):
            scored_turns.add((sid, turn))
            if grounding_used(g):
                grounded_turns.add((sid, turn))

    # Gaps answer "what couldn't memo answer for YOU": scope to ambient recall
    # (your prompts) AND to real working sessions. An automated eval harness
    # spawns throwaway single-turn claude-code sessions that fire a generic
    # corpus question (TCP vs UDP, git rebase…) through the same ambient hook, so
    # the via filter alone can't exclude them — filter_real_sessions drops the
    # turn_count == 1 probes so the panel shows questions that were actually yours.
    ambient = [r for r in read_recall_log(state_dir, limit=limit) if is_ambient_recall(r)]
    raw: list[dict[str, Any]] = []
    for r in filter_real_sessions(state_dir, ambient):
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
            reason = "no matches"
        elif not hits:
            reason = "0 results"
        elif (
            sid
            and isinstance(turn, int)
            and (sid, turn) in scored_turns
            and (sid, turn) not in grounded_turns
        ):
            reason = "found something but it wasn't used"
        else:
            continue
        raw.append({"prompt": prompt, "reason": reason, "ts": r.get("ts")})

    clusters: list[dict[str, Any]] = []
    for g in raw:
        tok = _reask_tokens(g["prompt"])
        for c in clusters:
            if _jaccard(tok, c["tokens"]) >= sim_threshold:
                c["count"] += 1
                # Single reason = the latest occurrence's state. The same prompt
                # can recall 0 results once and surface-but-unused another time;
                # unioning them renders the contradictory "0 results, found
                # something but it wasn't used". The most recent state is the truthful one.
                if (g["ts"] or "") > (c["last_seen"] or ""):
                    c["last_seen"] = g["ts"]
                    c["prompt"] = g["prompt"]
                    c["reason"] = g["reason"]
                break
        else:
            clusters.append(
                {
                    "tokens": tok,
                    "prompt": g["prompt"],
                    "count": 1,
                    "reason": g["reason"],
                    "last_seen": g["ts"],
                }
            )

    out = [
        {
            "prompt": c["prompt"],
            "count": c["count"],
            "reasons": [c["reason"]],
            "last_seen": c["last_seen"],
        }
        for c in clusters
        if c["count"] >= min_count
    ]
    out.sort(key=lambda c: (c["count"], c["last_seen"] or ""), reverse=True)
    return out


def self_heal_execution_failures(
    memory: Any,
    failed_memory_ids: list[str],
    *,
    reason: str = "execution_failure",
) -> dict[str, Any]:
    """Auto-heal pass: lower confidence on memories linked to a failed CLI/test execution."""
    from memo.flags import flag_bool

    if not flag_bool("MEMO_SELF_HEALING_ENABLED") or not failed_memory_ids:
        return {"healed": 0, "penalized": 0}

    penalized = 0
    for mid in failed_memory_ids:
        try:
            health = memory.store.get_health_batch([mid])
            curr_conf = float((health.get(mid) or {}).get("confidence", 1.0))
            new_conf = max(0.1, curr_conf - 0.35)
            memory.store.set_confidence_batch([(mid, new_conf)])
            penalized += 1
        except Exception as exc:
            _log.debug("self_heal_execution_failures error for %s: %s", mid, exc)

    return {"healed": len(failed_memory_ids), "penalized": penalized}
