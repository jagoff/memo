"""`memo dream` edge verification — graph edges EARN confidence from real use.

The recall hook labels every graph connection "via graph · unverified"
(``recall_assoc.render_associative_line``) because nothing ever validates an
edge against actual usage. This nightly pass closes that loop with the
evidence memo already logs: ``grounding.log`` rows record which recalled
memories were genuinely USED in a turn's answer (``grounding.score_turn`` →
``dashboard_metrics.grounding_used``). Two memories that were both used in the
same grounded turn are runtime evidence that the edge between them is real.

- **Promote:** for every NEW grounded turn where BOTH endpoints of an existing
  memory↔memory ``semantic_relations`` edge were used, raise the edge's
  ``confidence`` (capped). An edge is *verified* once confidence reaches
  ``VERIFIED_CONFIDENCE`` — the SAME shared constant ``recall_assoc`` reads to
  drop its "· unverified" framing, so the pass and the recall label agree by
  construction. The threshold sits above every deterministic-extractor prior
  (max 0.82 in ``semantic_relations._RELATION_PATTERNS``): only co-use
  evidence, never the extractor alone, can verify an edge.
- **Decay:** edges whose evidence stopped arriving lose a little confidence
  each night after a grace window (floored, reversible — a later evidenced
  night promotes them back). Nothing is ever deleted or invalidated:
  sovereignty-safe, non-destructive.
- **Reconcile:** confidence curated by this pass is re-applied when the DB row
  falls below it (see the ledger below).

**The ledger sidecar** (``state_dir/dream/edge_verify_ledger.json``) makes the
pass idempotent and its curation durable. Per canonical pair key (sorted
8-char id prefixes) it records:

- ``credited_turns`` — the ``(session_id, turn)`` keys already counted as
  evidence (bounded to the last ``LEDGER_MAX_TURNS``). A turn promotes an
  edge at most ONCE, ever — re-reading the same capped ``grounding.log`` on
  later nights is a no-op, and evidence for a pair with no edge yet is left
  uncredited until the edge exists.
- ``last_evidenced_at`` — refreshed whenever new turns are credited (seeded
  at first sight of the edge). Decay eligibility keys on THIS, not on the
  edge's never-refreshed ``created_at``, so a verified edge does not erode
  merely because its evidence rotated out of the capped log.
- ``curated_confidence`` — the confidence this pass last decided (per pair;
  the max across the pair's relation rows). ``memo graph relations rebuild``
  deletes by ``derived_from`` and re-creates rows at extractor priors — the
  nightly reconcile step re-applies ``curated_confidence`` to any row below
  it, so curation survives a full rebuild wipe with at most one night of lag.
  (Flip side: a manual downward edit of a memory↔memory confidence is
  indistinguishable from a wipe and is also restored.)

This is edge-confidence curation only — it never touches recall ranking (the
graph proximity re-rank was measured harmful and reverted; not re-litigated
here). OFF by default (``MEMO_DREAM_EDGE_VERIFY_ENABLED``).
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Promotion: each distinct NEW grounded co-use turn adds this much confidence.
PROMOTE_STEP = 0.1
CONFIDENCE_CAP = 0.95
# Verified = confidence earned past every deterministic-extractor prior
# (max 0.82). Shared with recall_assoc: the recall nudge drops its
# "· unverified" framing exactly when an edge clears this bar.
VERIFIED_CONFIDENCE = 0.85
# Decay: edges whose last credited evidence is older than the grace window
# shrink gently each night (same 0.98 cadence as the ROI decay pass), floored
# — never zeroed, never deleted.
DECAY_FACTOR = 0.98
CONFIDENCE_FLOOR = 0.05
DECAY_GRACE_DAYS = 14

# Ledger sidecar (see module docstring).
LEDGER_VERSION = 1
LEDGER_FILENAME = "edge_verify_ledger.json"
LEDGER_MAX_TURNS = 512  # per-pair cap on remembered credited turn keys

_EPS = 1e-9


def verified(confidence: float) -> bool:
    """Whether ``confidence`` clears ``VERIFIED_CONFIDENCE`` — the one shared
    threshold this pass promotes toward and ``recall_assoc`` renders by."""
    return confidence >= VERIFIED_CONFIDENCE - _EPS


def pair_key(a: str, b: str) -> str:
    """Canonical, direction-agnostic key for a memory pair: sorted 8-char id
    prefixes (grounding.log stores the 8-char prefix)."""
    a8, b8 = str(a)[:8], str(b)[:8]
    return f"{a8}|{b8}" if a8 <= b8 else f"{b8}|{a8}"


def turn_key(session_id: str, turn: int) -> str:
    return f"{session_id}|{turn}"


# --- ledger sidecar (idempotency + decay clock + curation memory) ------------


def ledger_path(state_dir: Path) -> Path:
    return Path(state_dir) / "dream" / LEDGER_FILENAME


def load_ledger(state_dir: Path) -> dict[str, Any]:
    """Load the ledger; a missing, corrupt, or wrong-version file starts
    fresh (the DB still holds current confidences — worst case is one lost
    curation memory, rebuilt by subsequent nights)."""
    try:
        data = json.loads(ledger_path(state_dir).read_text(encoding="utf-8"))
        if (
            isinstance(data, dict)
            and data.get("version") == LEDGER_VERSION
            and isinstance(data.get("edges"), dict)
        ):
            return data
    except (OSError, ValueError):
        pass
    return {"version": LEDGER_VERSION, "edges": {}}


def save_ledger(state_dir: Path, ledger: dict[str, Any]) -> None:
    """Atomic write (tmp + replace) so a crash never leaves a torn ledger."""
    path = ledger_path(state_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(ledger, indent=1, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def credit_turns(entry: dict[str, Any], fresh: list[str], now_iso: str) -> dict[str, Any]:
    """New ledger entry with *fresh* turn keys appended (bounded to the last
    ``LEDGER_MAX_TURNS``); ``last_evidenced_at`` refreshes only when there is
    something to credit. Returns a new dict — never mutates *entry*."""
    out = dict(entry)
    out["credited_turns"] = list(entry.get("credited_turns") or [])
    if fresh:
        out["credited_turns"] = (out["credited_turns"] + list(fresh))[-LEDGER_MAX_TURNS:]
        out["last_evidenced_at"] = now_iso
    return out


# --- pure core (testable) ----------------------------------------------------


def co_used_turns(rows: list[dict[str, Any]]) -> dict[str, set[str]]:
    """Co-use evidence from grounding.log rows.

    Returns ``{pair_key: {turn_key, ...}}`` — pairs of memories that were both
    recalled AND used (``grounding_used``) in the same ``(session_id, turn)``.
    Turn keys (not counts) so the caller can subtract already-credited turns.
    """
    from memo.dashboard_metrics import grounding_used

    by_turn: dict[tuple[str, int], set[str]] = {}
    for row in rows:
        sid = row.get("session_id")
        turn = row.get("turn")
        rid = row.get("recall_id")
        if not sid or not isinstance(turn, int) or not rid:
            continue
        if not grounding_used(row):
            continue
        by_turn.setdefault((str(sid), turn), set()).add(str(rid)[:8])

    pairs: dict[str, set[str]] = {}
    for (sid, turn), ids in by_turn.items():
        u = sorted(ids)
        for i in range(len(u)):
            for j in range(i + 1, len(u)):
                pairs.setdefault(pair_key(u[i], u[j]), set()).add(turn_key(sid, turn))
    return pairs


def _days_since(iso: Any, now: datetime) -> float | None:
    """Days since *iso*; None when missing/unparseable (treated as not yet
    eligible — conservative: never decay on absent or bad provenance)."""
    if not iso:
        return None
    try:
        dt = datetime.fromisoformat(str(iso))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return (now - dt).total_seconds() / 86400.0
    except (ValueError, TypeError):
        return None


def decide_edges(
    edges: list[dict[str, Any]],
    new_turns: dict[str, int],
    ledger_edges: dict[str, Any],
    *,
    now: datetime,
    min_pair_turns: int = 1,
) -> list[dict[str, Any]]:
    """One decision per edge: promote / decay / reconcile / hold. Pure — no I/O.

    *new_turns* counts only NOT-yet-credited evidence turns per pair key;
    *ledger_edges* supplies each pair's ``last_evidenced_at`` (decay clock)
    and ``curated_confidence`` (curation baseline). The working confidence is
    ``max(db, curated)`` so a rebuild-wiped row is restored (action
    ``reconcile``) rather than re-earned. Each decision carries the full edge
    row plus ``action``, the new ``confidence``, ``pair_key``,
    ``evidence_turns``, and ``verified`` (post-decision).
    """
    decisions: list[dict[str, Any]] = []
    for e in edges:
        key = pair_key(str(e.get("source_id") or ""), str(e.get("target_id") or ""))
        entry = ledger_edges.get(key) or {}
        db_conf = float(e.get("confidence") or 0.0)
        curated = entry.get("curated_confidence")
        base = max(db_conf, float(curated)) if isinstance(curated, (int, float)) else db_conf
        n = int(new_turns.get(key, 0))
        d: dict[str, Any] = {
            "edge": e,
            "pair_key": key,
            "evidence_turns": n,
            "old_confidence": db_conf,
        }

        if n >= min_pair_turns:
            new = min(CONFIDENCE_CAP, base + PROMOTE_STEP * n)
            if new > db_conf + _EPS:
                d.update(action="promote", confidence=new)
            else:
                d.update(action="hold", confidence=db_conf, reason="at_cap")
        else:
            since = _days_since(entry.get("last_evidenced_at"), now)
            if since is not None and since >= DECAY_GRACE_DAYS and base > CONFIDENCE_FLOOR:
                d.update(action="decay", confidence=max(CONFIDENCE_FLOOR, base * DECAY_FACTOR))
            elif base > db_conf + _EPS:
                # Curation survives a rebuild wipe: restore the ledger's
                # curated confidence (at most one night of lag).
                d.update(action="reconcile", confidence=base)
            else:
                reason = "first_seen" if not entry else "grace_or_floor"
                d.update(action="hold", confidence=db_conf, reason=reason)
        d["verified"] = verified(float(d["confidence"]))
        decisions.append(d)
    return decisions


# --- orchestrator (guarded; wires grounding log + graph store + ledger) ------


def run_edge_verify(
    cfg: Any,
    mem: Any,
    *,
    min_pair_turns: int = 1,
    dry_run: bool = False,
    now: datetime | None = None,
) -> dict[str, Any]:
    """One nightly edge-verification pass. Never raises — failures land in
    the returned fragment's ``error`` (the cli_dream caller surfaces it in
    ``receipt["errors"]``). Idempotent: re-running over the same
    grounding.log credits nothing new (see the ledger in the module doc)."""
    res: dict[str, Any] = {
        "status": "noop",
        "edges_total": 0,
        "pairs_evidenced": 0,
        "turns_credited": 0,
        "promoted": 0,
        "decayed": 0,
        "reconciled": 0,
        "held": 0,
        "verified": 0,
    }
    try:
        from memo.dashboard_logs import read_grounding_log

        graph = getattr(mem, "graph", None)
        if graph is None:
            res["status"] = "skipped"
            return res

        now_dt = now or datetime.now(UTC)
        ledger = load_ledger(cfg.state_dir)
        ledger_edges: dict[str, Any] = ledger["edges"]

        turns = co_used_turns(read_grounding_log(cfg.state_dir))
        res["pairs_evidenced"] = len(turns)

        edges = [
            dict(r)
            for r in graph._conn.execute(
                "SELECT * FROM semantic_relations "
                "WHERE source_kind = 'memory' AND target_kind = 'memory'"
            ).fetchall()
        ]
        res["edges_total"] = len(edges)
        if not edges:
            res["status"] = "skipped"
            return res

        # Only turns never credited before count as evidence (F1: idempotent).
        fresh_by_pair: dict[str, list[str]] = {}
        new_counts: dict[str, int] = {}
        for key, tks in turns.items():
            seen = set((ledger_edges.get(key) or {}).get("credited_turns") or [])
            fresh = sorted(tks - seen)
            if fresh:
                fresh_by_pair[key] = fresh
                new_counts[key] = len(fresh)

        decisions = decide_edges(
            edges, new_counts, ledger_edges, now=now_dt, min_pair_turns=min_pair_turns
        )
        for d in decisions:
            if d["action"] == "hold":
                continue
            if not dry_run:
                e = d["edge"]
                # In-place confidence update via the store's own idempotent
                # upsert (conflict on the full PK; created_at is preserved,
                # weight/evidence/validity pass through unchanged).
                graph.upsert_semantic_relation(
                    source_kind="memory",
                    source_id=str(e["source_id"]),
                    target_kind="memory",
                    target_id=str(e["target_id"]),
                    relation=str(e["relation"]),
                    weight=float(e.get("weight") or 1.0),
                    confidence=float(d["confidence"]),
                    evidence_id=e.get("evidence_id"),
                    derived_from=str(e["derived_from"]),
                    valid_at=e.get("valid_at"),
                    invalid_at=e.get("invalid_at"),
                )

        # Ledger update: credit ONLY pairs that matched an existing edge (a
        # pair with no edge keeps its evidence for the night the edge appears)
        # and remember the decided confidence per pair for reconciliation.
        now_iso = now_dt.isoformat()
        final_by_pair: dict[str, float] = {}
        for d in decisions:
            k = d["pair_key"]
            final_by_pair[k] = max(final_by_pair.get(k, 0.0), float(d["confidence"]))
        res["turns_credited"] = sum(len(fresh_by_pair.get(k, [])) for k in final_by_pair)
        if not dry_run:
            for k, conf in final_by_pair.items():
                entry = ledger_edges.get(k) or {
                    "credited_turns": [],
                    # First sight seeds the decay clock at NOW: a newly
                    # tracked edge gets the full grace window (F3), instead
                    # of eroding off its never-refreshed created_at.
                    "last_evidenced_at": now_iso,
                }
                entry = credit_turns(entry, fresh_by_pair.get(k, []), now_iso)
                entry["curated_confidence"] = round(conf, 6)
                ledger_edges[k] = entry
            save_ledger(cfg.state_dir, ledger)

        res["promoted"] = sum(1 for d in decisions if d["action"] == "promote")
        res["decayed"] = sum(1 for d in decisions if d["action"] == "decay")
        res["reconciled"] = sum(1 for d in decisions if d["action"] == "reconcile")
        res["held"] = sum(1 for d in decisions if d["action"] == "hold")
        res["verified"] = sum(1 for d in decisions if d["verified"])
        res["status"] = "done"
    except Exception as exc:  # surfaced into the receipt, never silent
        res["status"] = "error"
        res["error"] = f"{type(exc).__name__}: {exc}"
    return res


__all__ = [
    "CONFIDENCE_CAP",
    "CONFIDENCE_FLOOR",
    "DECAY_FACTOR",
    "DECAY_GRACE_DAYS",
    "LEDGER_MAX_TURNS",
    "PROMOTE_STEP",
    "VERIFIED_CONFIDENCE",
    "co_used_turns",
    "credit_turns",
    "decide_edges",
    "ledger_path",
    "load_ledger",
    "pair_key",
    "run_edge_verify",
    "save_ledger",
    "turn_key",
    "verified",
]
