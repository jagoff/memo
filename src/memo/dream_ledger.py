"""Auditable learning ledger for the nightly dream pipeline.

Every ``memo dream run`` night mutates the corpus (supersede / merge / archive /
synthesize / tune / graduate / prune / evict / dead-archive / code-drift), but
the only durable trail today is the flat receipt (``state_dir/dream/last.json``,
overwritten every night) plus the tuner-only proof loop
(``dream_tune_online.py`` — ``tune_pending.json`` + ``tune_ledger.jsonl``, the
"act now, judge later" pattern this module generalizes).

This module is the general version of that pattern: an append-only JSONL ledger
that chains the full learning loop for *any* dream action ::

    source_signal -> candidate_memory -> proposed_action -> evidence
      -> confidence -> applied_change -> later_outcome -> rollback_or_reinforcement

Two event kinds live in one append-only file (``state_dir/dream/ledger.jsonl``):

* ``kind="action"``  — full provenance up to the applied change (written when a
  mutating pass acts). ``entry_id`` is the action id later events reference.
* ``kind="outcome"`` — the later outcome + rollback/reinforcement verdict,
  referencing a prior action's ``entry_id`` via ``action_id`` (written by a
  later night that judges the action).

MLX-free: pure JSONL I/O over ``state_dir/dream/`` (mirrors
``dream_tune_online.py``). Every public function is best-effort and NEVER raises
into the pipeline — a ledger failure must never abort a dream night. The whole
feature is gated by ``MEMO_DREAM_LEDGER_ENABLED`` at the call sites; this module
does not read the flag itself so it stays pure and unit-testable.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)

LEDGER_FILE = "ledger.jsonl"

_ACTION = "action"
_OUTCOME = "outcome"


def _dream_dir(state_dir: Path) -> Path:
    return Path(state_dir) / "dream"


def ledger_path(state_dir: Path) -> Path:
    """Path of the append-only learning ledger under ``state_dir/dream/``."""
    return _dream_dir(state_dir) / LEDGER_FILE


def _now_iso(now: datetime | None = None) -> str:
    return (now or datetime.now(UTC)).isoformat(timespec="seconds")


def _coerce_confidence(value: Any) -> float | None:
    # bool is an int subclass; a stray True/False is not a confidence.
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _append(state_dir: Path, entry: dict[str, Any]) -> bool:
    """Append one JSONL line. Best-effort: logs and returns False on failure."""
    try:
        path = ledger_path(state_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(entry, ensure_ascii=False) + "\n"
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line)
        return True
    except (OSError, TypeError, ValueError) as exc:
        _log.warning("dream_ledger: failed to append %s entry: %s", entry.get("kind"), exc)
        return False


def _read_all(state_dir: Path) -> list[dict[str, Any]]:
    """All ledger entries oldest->newest, skipping blank/corrupt lines.

    Missing file -> ``[]``. Non-dict JSON lines are dropped.
    """
    try:
        lines = ledger_path(state_dir).read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    out: list[dict[str, Any]] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            out.append(obj)
    return out


def read_ledger(state_dir: Path, *, limit: int = 50) -> list[dict[str, Any]]:
    """The last ``limit`` ledger entries (any kind), oldest->newest."""
    entries = _read_all(state_dir)
    if limit >= 0:
        return entries[-limit:]
    return entries


def record_action(
    state_dir: Path,
    *,
    action: str,
    pass_name: str | None = None,
    source_signal: Any = None,
    candidate_ids: Sequence[str] | None = None,
    evidence: dict[str, Any] | None = None,
    confidence: float | None = None,
    affected_ids: Sequence[str] | None = None,
    applied: bool = True,
    reversal: dict[str, Any] | None = None,
    params_version: str | None = None,
    dry_run: bool = False,
    now: datetime | None = None,
) -> str | None:
    """Record one dream action with its full provenance up to the applied change.

    Returns the new ``action_id`` (a uuid hex, referenced by a later
    :func:`resolve_action`) or ``None`` when nothing was written — i.e. on
    ``dry_run`` (matches every mutating pass), on invalid input, or on an I/O
    failure. Never raises.

    ``action`` is the action class (``supersede``/``merge``/``archive_stale``/
    ``synthesize``/``tune``/``graduate``/``prune``/``evict``/``dead_archive``/
    ``code_drift`` ...). ``applied`` distinguishes a mutation that actually
    landed from one merely recorded for review (held/competing/flagged).
    ``reversal`` carries the rollback handle, e.g.
    ``{"type": "inactive_md", "handle": "inactive/<id>.md"}``.
    """
    if not isinstance(action, str) or not action.strip():
        _log.warning("dream_ledger: record_action skipped — invalid action %r", action)
        return None
    if dry_run:
        return None

    entry_id = uuid.uuid4().hex
    entry: dict[str, Any] = {
        "entry_id": entry_id,
        "ts": _now_iso(now),
        "kind": _ACTION,
        "action": action.strip(),
        "pass_name": pass_name,
        "source_signal": source_signal,
        "candidate_ids": list(candidate_ids) if candidate_ids else [],
        "evidence": evidence or {},
        "confidence": _coerce_confidence(confidence),
        "affected_ids": list(affected_ids) if affected_ids else [],
        "applied": bool(applied),
        "dry_run": False,
        "reversal": reversal,
        "params_version": params_version,
    }
    if _append(state_dir, entry):
        return entry_id
    return None


def resolve_action(
    state_dir: Path,
    action_id: str,
    *,
    outcome: str,
    verdict: str | None = None,
    evidence: dict[str, Any] | None = None,
    delta: float | None = None,
    now: datetime | None = None,
) -> str | None:
    """Attach a later outcome + rollback/reinforcement verdict to a prior action.

    Appends a new ``kind="outcome"`` event referencing ``action_id`` (the ledger
    is append-only; the original action line is never rewritten). Returns the new
    outcome ``entry_id`` or ``None`` when nothing was written — on invalid input,
    on an unknown ``action_id`` (no matching action in the ledger), or on I/O
    failure. Never raises.

    ``outcome`` is the observed later_outcome (e.g. ``reinforced`` /
    ``rollback_candidate`` / ``confirmed`` / ``reverted`` / ``neutral``);
    ``verdict`` optionally records the rollback-or-reinforcement decision
    separately. ``delta`` carries the realized change that justified the verdict.
    """
    if not isinstance(action_id, str) or not action_id.strip():
        _log.warning("dream_ledger: resolve_action skipped — invalid action_id %r", action_id)
        return None
    if not isinstance(outcome, str) or not outcome.strip():
        _log.warning("dream_ledger: resolve_action skipped — invalid outcome %r", outcome)
        return None
    if not _action_exists(state_dir, action_id):
        _log.warning("dream_ledger: resolve_action skipped — unknown action_id %r", action_id)
        return None

    entry_id = uuid.uuid4().hex
    entry: dict[str, Any] = {
        "entry_id": entry_id,
        "ts": _now_iso(now),
        "kind": _OUTCOME,
        "action_id": action_id,
        "outcome": outcome.strip(),
        "verdict": verdict,
        "evidence": evidence or {},
        "delta": delta,
    }
    if _append(state_dir, entry):
        return entry_id
    return None


def _action_exists(state_dir: Path, action_id: str) -> bool:
    for entry in _read_all(state_dir):
        if entry.get("kind") == _ACTION and entry.get("entry_id") == action_id:
            return True
    return False


def recent_actions(
    state_dir: Path,
    *,
    limit: int = 50,
    action: str | None = None,
    phase: str | None = None,
) -> list[dict[str, Any]]:
    """Recent action events, newest-first, optionally filtered.

    ``action`` filters on the action class; ``phase`` filters on ``pass_name``
    (the dream pass that produced the action).
    """
    rows = [e for e in _read_all(state_dir) if e.get("kind") == _ACTION]
    if action is not None:
        rows = [e for e in rows if e.get("action") == action]
    if phase is not None:
        rows = [e for e in rows if e.get("pass_name") == phase]
    rows.reverse()
    if limit >= 0:
        return rows[:limit]
    return rows


def open_actions(state_dir: Path, *, limit: int = 500) -> list[dict[str, Any]]:
    """Action events with no matching outcome event yet, oldest-first.

    These are the actions a later night's resolution pass still has to judge.
    """
    entries = _read_all(state_dir)
    resolved = {e.get("action_id") for e in entries if e.get("kind") == _OUTCOME}
    unresolved = [
        e for e in entries if e.get("kind") == _ACTION and e.get("entry_id") not in resolved
    ]
    if limit >= 0:
        return unresolved[:limit]
    return unresolved


def get_action(state_dir: Path, action_id: str) -> dict[str, Any] | None:
    """The action entry for ``action_id`` folded with its latest outcome event.

    Returns the action dict plus an ``"outcome"`` key (the latest matching
    outcome event, or ``None`` if still open). ``None`` if the action is unknown.
    """
    action: dict[str, Any] | None = None
    outcome: dict[str, Any] | None = None
    for entry in _read_all(state_dir):
        kind = entry.get("kind")
        if kind == _ACTION and entry.get("entry_id") == action_id:
            action = entry
        elif kind == _OUTCOME and entry.get("action_id") == action_id:
            outcome = entry
    if action is None:
        return None
    return {**action, "outcome": outcome}


def _inactive_reversal(memory_id: str) -> dict[str, Any]:
    """Rollback handle for an archived memory (moved to ``inactive/<id>.md``)."""
    return {"type": "inactive_md", "handle": f"inactive/{memory_id}.md"}


def record_from_receipt(
    state_dir: Path, receipt: dict[str, Any], *, dry_run: bool = False
) -> dict[str, int]:
    """Record one ledger action per concrete, id-bearing mutation in ``receipt``.

    Maps the completed dream receipt's mutating fragments (contradict-supersede,
    consolidate-merge, archive-stale, evolve, synthesize) into the append-only
    learning ledger with each action's provenance, evidence, and rollback handle.
    Coarser than per-decision instrumentation but honest: every reversible corpus
    mutation of a night gets one auditable chain entry. Returns a per-action
    count. ``dry_run`` records nothing (matches every mutating pass). Never
    raises — a ledger failure must not abort the night.
    """
    counts: dict[str, int] = {}
    if dry_run:
        return counts

    def _bump(kind: str) -> None:
        counts[kind] = counts.get(kind, 0) + 1

    contradict = receipt.get("contradict") or {}
    for item in contradict.get("superseded", []) or []:
        older = str(item.get("older") or "")
        if not older:
            continue
        if record_action(
            state_dir,
            action="supersede",
            pass_name="contradict",  # noqa: S106 - dream pass name, not a secret
            source_signal={"pair_id": item.get("pair_id")},
            candidate_ids=[older],
            affected_ids=[older],
            evidence={"relationship": "contradiction"},
            confidence=0.9,
            reversal=_inactive_reversal(older),
        ):
            _bump("supersede")
    for pair_id in contradict.get("evolved", []) or []:
        if record_action(
            state_dir,
            action="evolve",
            pass_name="contradict",  # noqa: S106 - dream pass name, not a secret
            source_signal={"pair_id": pair_id},
            evidence={"relationship": "evolution"},
            reversal={"type": "confidence_restore"},
        ):
            _bump("evolve")

    consolidate = receipt.get("consolidate_dups") or {}
    for item in consolidate.get("merged", []) or []:
        archived = [str(a) for a in (item.get("archived_ids") or []) if str(a)]
        if not archived:
            continue
        if record_action(
            state_dir,
            action="merge",
            pass_name="consolidate_dups",  # noqa: S106 - dream pass name, not a secret
            candidate_ids=archived,
            affected_ids=archived,
            evidence={"merged_id": item.get("merged_id"), "archived": len(archived)},
            reversal={
                "type": "inactive_md_multi",
                "handles": [f"inactive/{a}.md" for a in archived],
            },
        ):
            _bump("merge")

    stale = receipt.get("stale") or {}
    for item in stale.get("archived", []) or []:
        mid = str(item.get("id") or "")
        if not mid:
            continue
        if record_action(
            state_dir,
            action="archive_stale",
            pass_name="stale",  # noqa: S106 - dream pass name, not a secret
            candidate_ids=[mid],
            affected_ids=[mid],
            evidence={"days_since_update": item.get("days")},
            reversal=_inactive_reversal(mid),
        ):
            _bump("archive_stale")

    return counts


def resolve_open_actions(
    state_dir: Path,
    is_live: Callable[[str], bool],
    *,
    min_age_hours: float = 20.0,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Close the loop on prior nights' reversible archives (later_outcome).

    For each still-open ``supersede`` / ``archive_stale`` / ``merge`` action older
    than ``min_age_hours`` (so tonight's own actions are never judged tonight),
    checks whether an affected (archived) memory came back to life
    (``is_live(id)`` — e.g. a human un-archived it). A resurrected memory means
    the archive was wrong → ``rollback_candidate``/``reopened``; a memory that
    stayed archived is ``reinforced``. Returns ``{reinforced, rollback_candidate,
    skipped}``. ``is_live`` is injected so this stays MLX-free and unit-testable.
    Never raises.
    """
    cutoff = now or datetime.now(UTC)
    out = {"reinforced": 0, "rollback_candidate": 0, "skipped": 0}
    reversible = {"supersede", "archive_stale", "merge"}
    for action in open_actions(state_dir):
        if action.get("action") not in reversible:
            continue
        ts = _parse_ts(action.get("ts"))
        if ts is None or (cutoff - ts).total_seconds() < min_age_hours * 3600.0:
            out["skipped"] += 1
            continue
        affected = [str(a) for a in (action.get("affected_ids") or []) if str(a)]
        try:
            resurrected = any(is_live(a) for a in affected)
        except Exception as exc:  # defensive: liveness probe must not abort
            _log.debug("dream_ledger: liveness probe failed: %s", exc)
            out["skipped"] += 1
            continue
        if resurrected:
            resolve_action(
                state_dir,
                str(action.get("entry_id")),
                outcome="rollback_candidate",
                verdict="reopened",
                evidence={"resurrected": [a for a in affected if _safe_live(is_live, a)]},
                now=now,
            )
            out["rollback_candidate"] += 1
        else:
            resolve_action(
                state_dir,
                str(action.get("entry_id")),
                outcome="reinforced",
                verdict="held",
                now=now,
            )
            out["reinforced"] += 1
    return out


def _safe_live(is_live: Callable[[str], bool], memory_id: str) -> bool:
    try:
        return bool(is_live(memory_id))
    except Exception:
        return False


def _parse_ts(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def summarize(state_dir: Path) -> dict[str, Any]:
    """A one-glance summary for ``memo dream status`` / ``memo dream ledger``.

    Counts total actions/outcomes, still-open actions, rollback candidates, and a
    per-action-class breakdown.
    """
    entries = _read_all(state_dir)
    actions = [e for e in entries if e.get("kind") == _ACTION]
    outcomes = [e for e in entries if e.get("kind") == _OUTCOME]
    resolved = {e.get("action_id") for e in outcomes}
    open_count = sum(1 for a in actions if a.get("entry_id") not in resolved)
    rollback = sum(
        1
        for o in outcomes
        if "rollback" in str(o.get("verdict") or "") or "rollback" in str(o.get("outcome") or "")
    )
    by_action: dict[str, int] = {}
    for a in actions:
        key = str(a.get("action", "?"))
        by_action[key] = by_action.get(key, 0) + 1
    return {
        "actions": len(actions),
        "outcomes": len(outcomes),
        "open": open_count,
        "rollback_candidates": rollback,
        "by_action": by_action,
    }
