"""Negative-recall CAPTURE — derive & persist ``failure_pattern`` anti-memories.

Two signals graduate into the durable ⛔ corpus:

  (a) **supersede / reversal** — when the nightly contradict pass archives a
      dominated approach in favour of a dominant one, the archived approach
      becomes the *Wrong* and the winner the *Right* (see
      :func:`memo.negative_recall.derive_failure_pattern_from_supersede`).
  (b) **avoid verdicts** — recalled memories the user's *next* turn corrected or
      rejected, mined from ``verdict.log`` (the same signal
      :func:`memo.eval_recall.harvest_negative_labels` turns into ``avoid_ids``).
      A recalled id that draws a negative/correction verdict graduates into a
      real ``failure_pattern`` capture
      (:func:`memo.negative_recall.derive_failure_pattern_from_avoid_verdict`).

The *pure* derivation lives in :mod:`memo.negative_recall` (imported, never
re-defined here). This module adds the store I/O the derivations deliberately
omit: a deterministic **provenance-hash dedup** (MLX-free — no embedding
similarity needed), a **markdown-first save**, and the flag gate
``MEMO_NEGATIVE_RECALL_CAPTURE_ENABLED`` (default OFF). Everything here runs on
the nightly ``memo dream`` passes — never on the 5s recall hook.
"""

from __future__ import annotations

import hashlib
import logging
from collections import defaultdict
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from memo.flags import flag_bool
from memo.negative_recall import (
    FAILURE_PATTERN_TYPE,
    FP_LINKS_KEY,
    FP_SOURCE_KEY,
    derive_failure_pattern_from_avoid_verdict,
    derive_failure_pattern_from_supersede,
)

if TYPE_CHECKING:
    from memo.config import Config
    from memo.memory.facade import Memory

_log = logging.getLogger(__name__)

# Flag gate for every capture path (default OFF — see flags_recall.py).
_CAPTURE_FLAG = "MEMO_NEGATIVE_RECALL_CAPTURE_ENABLED"

# ``extra`` key holding the stable provenance fingerprint used for dedup. Kept
# in the same dedicated namespace as the other FP_* provenance keys so it never
# clobbers a generic ``source`` key.
FP_PROVENANCE_HASH_KEY = "fp_provenance_hash"

# How many recent failure_patterns to scan when building the dedup index. The
# anti-memory corpus is small and this runs nightly (never the hot path).
_DEDUP_SCAN_LIMIT = 2000

# Avoid-verdict graduation defaults.
#   min_occurrences: a recalled id must draw a negative/correction verdict in at
#     least this many DISTINCT turns before it graduates. Defaults to 1 for
#     parity with ``harvest_negative_labels`` (which promotes a single verdict's
#     recall_ids to avoid_ids); the per-origin provenance dedup already bounds
#     each memory to at most one anti-memory, so 1 does not spam the corpus.
#     Raise it to demand recurrence.
_DEFAULT_MIN_OCCURRENCES = 1
#   max_captures: cap the anti-memories minted in a single pass.
_DEFAULT_MAX_CAPTURES = 20
# How many verdict rows to scan (the log self-caps at 500 rows on write).
_VERDICT_LOG_SCAN = 2000

# Verdicts that count as an "avoid" signal (mirrors harvest_negative_labels).
_AVOID_VERDICTS = ("negative", "correction")

# Origin types that never graduate: an anti-memory OF an anti-memory is
# nonsense, and bulk reference chunks / secrets are not lessons.
_SKIP_ORIGIN_TYPES = frozenset({FAILURE_PATTERN_TYPE, "reference", "secret"})


# ── provenance dedup (pure, MLX-free) ────────────────────────────────────────


def _provenance_hash(extra: Mapping[str, Any]) -> str:
    """Stable fingerprint of a derived anti-memory's origin.

    Hashes the ``source`` + the sorted provenance links so the *same*
    supersede pair (wrong_id + right_id) or the *same* avoid origin (origin_id)
    always yields the same hash — the basis for idempotent, embedding-free
    dedup across nightly re-runs.
    """
    source = str(extra.get(FP_SOURCE_KEY) or "")
    raw_links = extra.get(FP_LINKS_KEY)
    links = sorted(str(x) for x in raw_links) if isinstance(raw_links, (list, tuple)) else []
    payload = source + "|" + "|".join(links)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _existing_provenance_hashes(mem: Memory) -> set[str]:
    """Provenance hashes already present on stored failure_patterns.

    MLX-free: reads recent failure_pattern rows straight from the index (no
    body reads, no embedding) — the dedup decision is exact provenance, not
    semantic similarity. A scan failure degrades to "nothing stored" so a
    transient DB error never blocks a capture (a genuine duplicate is caught on
    the next run).
    """
    try:
        rows = mem.store.list_recent(limit=_DEDUP_SCAN_LIMIT, type_=FAILURE_PATTERN_TYPE)
    except Exception as exc:
        _log.debug("negative_capture: dedup scan failed (%s); treating as empty", exc)
        return set()
    hashes: set[str] = set()
    for row in rows:
        h = (row.get("extra") or {}).get(FP_PROVENANCE_HASH_KEY)
        if isinstance(h, str):
            hashes.add(h)
    return hashes


def _persist(mem: Memory, payload: dict[str, Any], prov_hash: str) -> str:
    """Markdown-first save of a derived anti-memory, stamping the dedup hash.

    ``auto_project=False`` mirrors ``git_miner`` — these anti-memories are
    derived from provenance, not the caller's cwd, so they must not pick up a
    stray ``project:`` tag from wherever dream happens to run.
    """
    extra = dict(payload["extra"])
    extra[FP_PROVENANCE_HASH_KEY] = prov_hash
    rec = mem.save(
        content=payload["body"],
        title=payload["title"],
        type_=FAILURE_PATTERN_TYPE,
        tags=payload.get("tags") or None,
        extra=extra,
        auto_project=False,
    )
    return rec.id


# ── (a) supersede / reversal ─────────────────────────────────────────────────


def capture_from_supersede(
    mem: Memory,
    *,
    superseded_id: str,
    superseding_id: str,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Derive + persist one ``failure_pattern`` from a supersede/reversal.

    Gated on ``MEMO_NEGATIVE_RECALL_CAPTURE_ENABLED`` (OFF → no-op). The
    dominated approach becomes the *Wrong*, the dominant the *Right*; the pair
    is deduped by provenance hash so re-running never re-mints. Best-effort:
    always returns a summary dict and never raises, so a capture failure cannot
    abort the contradict pass that calls it. Read the records BEFORE archiving
    (the caller's responsibility) so both bodies are still resolvable.

    Returns ``{"status", "captured_id"?, "error"?}`` where ``status`` is one of
    ``disabled`` / ``unresolved`` / ``skipped_dup`` / ``dry_run`` / ``captured``
    / ``error``.
    """
    if not flag_bool(_CAPTURE_FLAG):
        return {"status": "disabled", "captured_id": None}
    try:
        superseded = mem.get(superseded_id)
        superseding = mem.get(superseding_id)
        if superseded is None or superseding is None:
            return {"status": "unresolved", "captured_id": None}
        payload = derive_failure_pattern_from_supersede(superseded, superseding)
        prov_hash = _provenance_hash(payload["extra"])
        if prov_hash in _existing_provenance_hashes(mem):
            return {"status": "skipped_dup", "captured_id": None}
        if dry_run:
            return {"status": "dry_run", "captured_id": None}
        captured_id = _persist(mem, payload, prov_hash)
        return {"status": "captured", "captured_id": captured_id}
    except FileNotFoundError:
        # The index can retain a supersede pair after its source Markdown was
        # removed externally.  This is an unresolved provenance record, not a
        # capture failure; keep the nightly pass healthy and actionable.
        return {"status": "unresolved", "captured_id": None}
    except Exception as exc:
        # Surfaced (not swallowed): the caller records ``error`` in its receipt.
        _log.warning("negative_capture (supersede) failed: %s", exc)
        return {"status": "error", "captured_id": None, "error": f"{type(exc).__name__}: {exc}"}


# ── (b) avoid verdicts ───────────────────────────────────────────────────────


def _mine_avoid_candidates(
    entries: list[dict[str, Any]],
) -> tuple[dict[str, set[tuple[str, Any]]], dict[str, dict[str, Any]]]:
    """Group avoid verdicts by recalled id.

    Returns ``(occurrences, latest_verdict)`` where ``occurrences[id]`` is the
    set of distinct ``(session_id, turn)`` in which the id drew a negative /
    correction verdict, and ``latest_verdict[id]`` is the most recent such
    verdict record (its prompt + reaction give the anti-memory its context and
    corrected *Right*).
    """
    occurrences: dict[str, set[tuple[str, Any]]] = defaultdict(set)
    latest_verdict: dict[str, dict[str, Any]] = {}
    for row in entries:
        if row.get("verdict") not in _AVOID_VERDICTS:
            continue
        ids = [str(i) for i in (row.get("recall_ids") or []) if len(str(i)) >= 8]
        if not ids:
            continue
        turn_key = (str(row.get("session_id") or ""), row.get("turn"))
        ts = str(row.get("ts") or "")
        for rid in ids:
            occurrences[rid].add(turn_key)
            prev = latest_verdict.get(rid)
            if prev is None or ts >= str(prev.get("ts") or ""):
                latest_verdict[rid] = row
    return occurrences, latest_verdict


def graduate_avoid_verdicts(
    cfg: Config,
    mem: Memory,
    *,
    min_occurrences: int = _DEFAULT_MIN_OCCURRENCES,
    max_captures: int = _DEFAULT_MAX_CAPTURES,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Graduate recurrently-avoided recalled memories into ``failure_pattern``s.

    Mines ``verdict.log`` for recalled ids that drew a negative/correction
    verdict in at least ``min_occurrences`` distinct turns; the recalled memory
    becomes the *Wrong*, the correcting reaction the *Right*. Gated on
    ``MEMO_NEGATIVE_RECALL_CAPTURE_ENABLED`` (OFF → ``status="disabled"``).
    Deduped by provenance hash (per-origin) so a memory graduates at most once.
    Never raises — returns a summary suitable for the dream receipt.
    """
    result: dict[str, Any] = {
        "status": "ok",
        "captured": [],
        "skipped_dup": 0,
        "candidates": 0,
        "errors": [],
    }
    if not flag_bool(_CAPTURE_FLAG):
        result["status"] = "disabled"
        return result
    try:
        from memo.dashboard_logs import read_verdict_log

        entries = read_verdict_log(cfg.state_dir, limit=_VERDICT_LOG_SCAN)
        occurrences, latest_verdict = _mine_avoid_candidates(entries)
        candidates = sorted(
            (rid for rid, turns in occurrences.items() if len(turns) >= min_occurrences),
            key=lambda rid: len(occurrences[rid]),
            reverse=True,
        )
        result["candidates"] = len(candidates)

        existing_hashes = _existing_provenance_hashes(mem)
        for rid in candidates[:max_captures]:
            try:
                memrec = mem.get(rid)
            except Exception:
                # Ambiguous prefix or lookup error — skip this id only.
                memrec = None
            if memrec is None or memrec.type in _SKIP_ORIGIN_TYPES:
                continue
            payload = derive_failure_pattern_from_avoid_verdict(memrec, latest_verdict[rid])
            prov_hash = _provenance_hash(payload["extra"])
            if prov_hash in existing_hashes:
                result["skipped_dup"] += 1
                continue
            if dry_run:
                existing_hashes.add(prov_hash)
                result["captured"].append("<dry-run>")
                continue
            try:
                captured_id = _persist(mem, payload, prov_hash)
            except Exception as exc:
                result["errors"].append(f"{rid[:8]}: {type(exc).__name__}: {exc}")
                continue
            existing_hashes.add(prov_hash)
            result["captured"].append(captured_id)
    except Exception as exc:
        # Whole-pass failure — surfaced, never silent.
        result["status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"
        _log.warning("negative_capture (avoid verdicts) failed: %s", exc)
    return result


__all__ = [
    "FP_PROVENANCE_HASH_KEY",
    "capture_from_supersede",
    "graduate_avoid_verdicts",
]
