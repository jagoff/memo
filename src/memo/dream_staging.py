"""Dream conflict-staging: park a blocked dream-minted memory instead of losing it.

Every dream pass that mints a durable memory calls ``mem.save(...)``, which runs
the native write policy and raises :class:`memo.errors.WriteRefused` when an
active ``freeze_write`` conflict matches the write's topic. Historically each
save site swallowed that in a broad ``except`` and lost the LLM-generated
candidate — no conflict id, no evidence, no way to re-apply once the human
resolved the conflict.

This module is the seam that fixes it, gated default-OFF behind
``MEMO_DREAM_STAGING_ENABLED``:

- :func:`staged_save` wraps ``mem.save``. Outside a dream pipeline run (or with
  the flag off) it is a pure pass-through — ``WriteRefused`` propagates exactly
  as before. Inside a dream run with the flag on, a ``WriteRefused`` (and only a
  ``WriteRefused``) parks the full proposal + conflict evidence in machine-local
  dream state and returns ``None``; every other error re-raises so callers'
  ``save_failed`` path is preserved.
- :func:`resume_staged_proposals` re-applies a parked proposal on a later run
  once the blocking conflict is resolved (human-only, via
  ``memo operational conflict resolve``) or auto-cleared (subject memory
  deleted). It never resolves a conflict itself.

The staging store is ``state_dir/dream/staging.json`` — derived, machine-local
dream state like ``last.json``. It is NOT a memory: no ``.md``, no ``id:``
frontmatter, never in the vault, never exported over git sync. Markdown remains
the source of truth.
"""

from __future__ import annotations

import contextvars
import hashlib
import json
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from memo.atomic_io import atomic_write_text
from memo.errors import MemoError, WriteRefused
from memo.flags import flag_bool, flag_int
from memo.util import utc_now_iso

if TYPE_CHECKING:
    from memo.config import Config
    from memo.memory import Memory
    from memo.memory.record import MemoryRecord

_SCHEMA = "memo.dream_staging.v1"
_STAGING_FLAG = "MEMO_DREAM_STAGING_ENABLED"
_STAGING_MAX_FLAG = "MEMO_DREAM_STAGING_MAX"
_DEFAULT_MAX = 200

# Extra-frontmatter keys a pass may carry a stable provenance hash under. Used
# to make the proposal id deterministic across nights so the same blocked
# candidate dedups instead of piling up.
_PROVENANCE_KEYS = ("synthesis_sources_hash", "sources_hash", "provenance_hash")


@dataclass(frozen=True)
class StagedProposal:
    """One dream-minted candidate parked because a write conflict blocked it.

    ``save_kwargs`` is replayed verbatim into ``mem.save`` on resume, so it holds
    exactly the keyword arguments the pass passed to :func:`staged_save`
    (content/title/type_/tags/extra/...).
    """

    proposal_id: str
    kind: str
    save_kwargs: dict[str, Any]
    source_ids: tuple[str, ...]
    conflict_ids: tuple[str, ...]
    conflict_summary: str
    evidence_uris: tuple[str, ...]
    staged_at: str
    state: str = "staged"  # "staged" | "applied" | "dropped"
    attempts: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "kind": self.kind,
            "save_kwargs": dict(self.save_kwargs),
            "source_ids": list(self.source_ids),
            "conflict_ids": list(self.conflict_ids),
            "conflict_summary": self.conflict_summary,
            "evidence_uris": list(self.evidence_uris),
            "staged_at": self.staged_at,
            "state": self.state,
            "attempts": self.attempts,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StagedProposal:
        return cls(
            proposal_id=str(data["proposal_id"]),
            kind=str(data.get("kind") or ""),
            save_kwargs=dict(data.get("save_kwargs") or {}),
            source_ids=tuple(str(s) for s in (data.get("source_ids") or ())),
            conflict_ids=tuple(str(c) for c in (data.get("conflict_ids") or ())),
            conflict_summary=str(data.get("conflict_summary") or ""),
            evidence_uris=tuple(str(u) for u in (data.get("evidence_uris") or ())),
            staged_at=str(data.get("staged_at") or ""),
            state=str(data.get("state") or "staged"),
            attempts=int(data.get("attempts") or 1),
            metadata=dict(data.get("metadata") or {}),
        )


# --- scope -----------------------------------------------------------------
# staged_save only diverts to staging inside a dream pipeline run. Interactive
# `memo synthesize` / presynthesis saves (which call the same public Memory
# methods) must never silently disappear into staging, so the seam is gated on
# this contextvar in addition to the flag.
_STAGING_SCOPE: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "memo_dream_staging_scope", default=False
)


@contextmanager
def dream_staging_scope() -> Iterator[None]:
    """Mark the current context as a dream pipeline run for :func:`staged_save`."""
    token = _STAGING_SCOPE.set(True)
    try:
        yield
    finally:
        _STAGING_SCOPE.reset(token)


def _staging_active() -> bool:
    return _STAGING_SCOPE.get() and flag_bool(_STAGING_FLAG)


# --- persistence -----------------------------------------------------------
def _staging_path(cfg: Config) -> Path:
    return cfg.state_dir / "dream" / "staging.json"


def _load(cfg: Config) -> list[StagedProposal]:
    path = _staging_path(cfg)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    proposals = raw.get("proposals") if isinstance(raw, dict) else None
    if not isinstance(proposals, list):
        return []
    out: list[StagedProposal] = []
    for item in proposals:
        if not isinstance(item, dict):
            continue
        try:
            out.append(StagedProposal.from_dict(item))
        except (KeyError, TypeError, ValueError):
            continue
    return out


def _save(cfg: Config, proposals: Sequence[StagedProposal]) -> None:
    payload = {"schema": _SCHEMA, "proposals": [p.to_dict() for p in proposals]}
    atomic_write_text(
        _staging_path(cfg),
        json.dumps(payload, ensure_ascii=False, indent=2),
    )


# --- helpers ---------------------------------------------------------------
def _provenance_hash(save_kwargs: dict[str, Any], source_ids: Sequence[str]) -> str:
    """A stable 16-hex fingerprint for a proposal.

    Prefers a pass-supplied provenance hash (so the id matches the pass's own
    dedup), else falls back to ``sha256(sorted(source_ids) + title)``.
    """
    extra = save_kwargs.get("extra")
    if isinstance(extra, dict):
        for key in _PROVENANCE_KEYS:
            val = str(extra.get(key) or "").strip()
            if val:
                return val[:16]
    title = str(save_kwargs.get("title") or "")
    basis = "\x00".join(sorted(str(s) for s in source_ids)) + "\x00" + title
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()[:16]


def _capture_conflict(mem: Memory, conflict: dict[str, Any]) -> tuple[list[str], str, list[str]]:
    """Extract (conflict_ids, summary, evidence_uris) for a WriteRefused.

    The WriteRefused only carries the first blocking conflict id + reason. We
    enrich it by matching that id against the live operational conflict rows to
    pull the row's summary, evidence uris, and subject memory ids. Best-effort:
    evidence enrichment must never fail the primary staging write, so a read
    failure degrades to the WriteRefused's own summary.
    """
    cid = str(conflict.get("conflict_id") or "").strip()
    summary = str(conflict.get("summary") or "").strip()
    conflict_ids = [cid] if cid else []
    evidence: list[str] = []
    try:
        rows = mem.operational.active_conflicts()
    except Exception:
        rows = []
    for row in rows:
        if str(row.get("id") or "") != cid:
            continue
        summary = str(row.get("summary") or summary).strip()
        evidence = [str(u) for u in (row.get("evidence_uris") or ()) if str(u).strip()]
        meta = row.get("metadata") or {}
        member_ids = [str(m) for m in (meta.get("memory_ids") or ()) if str(m).strip()]
        if not evidence and member_ids:
            evidence = [f"memo://memoria/{m}" for m in member_ids]
        break
    return conflict_ids, summary, evidence


def _enforce_cap(proposals: list[StagedProposal], max_staged: int) -> list[StagedProposal]:
    """Drop the oldest ``staged`` proposals beyond the cap (list order = age)."""
    if max_staged <= 0:
        return proposals
    staged_idx = [i for i, p in enumerate(proposals) if p.state == "staged"]
    excess = len(staged_idx) - max_staged
    if excess <= 0:
        return proposals
    drop = set(staged_idx[:excess])
    return [p for i, p in enumerate(proposals) if i not in drop]


# --- public API ------------------------------------------------------------
def stage_proposal(
    cfg: Config,
    mem: Memory,
    *,
    kind: str,
    save_kwargs: dict[str, Any],
    source_ids: Sequence[str],
    conflict: dict[str, Any],
) -> StagedProposal:
    """Park a blocked candidate (idempotent by deterministic proposal id).

    Staging the same proposal again bumps ``attempts`` and refreshes the
    captured conflict rather than creating a duplicate. Enforces
    ``MEMO_DREAM_STAGING_MAX`` by dropping the oldest staged proposals.
    """
    ids = [str(s) for s in source_ids]
    proposal_id = f"dream-{kind}-{_provenance_hash(save_kwargs, ids)}"
    conflict_ids, summary, evidence = _capture_conflict(mem, conflict)

    proposals = _load(cfg)
    existing = next((p for p in proposals if p.proposal_id == proposal_id), None)
    if existing is not None:
        updated = replace(
            existing,
            save_kwargs=dict(save_kwargs),
            source_ids=tuple(ids),
            conflict_ids=tuple(conflict_ids) or existing.conflict_ids,
            conflict_summary=summary or existing.conflict_summary,
            evidence_uris=tuple(evidence) or existing.evidence_uris,
            state="staged",
            attempts=existing.attempts + 1,
        )
        proposals = [updated if p.proposal_id == proposal_id else p for p in proposals]
        _save(cfg, proposals)
        return updated

    proposal = StagedProposal(
        proposal_id=proposal_id,
        kind=str(kind),
        save_kwargs=dict(save_kwargs),
        source_ids=tuple(ids),
        conflict_ids=tuple(conflict_ids),
        conflict_summary=summary,
        evidence_uris=tuple(evidence),
        staged_at=utc_now_iso(),
    )
    proposals.append(proposal)
    max_staged = flag_int(_STAGING_MAX_FLAG) or _DEFAULT_MAX
    proposals = _enforce_cap(proposals, max_staged)
    _save(cfg, proposals)
    return proposal


def staged_save(
    mem: Memory,
    cfg: Config,
    *,
    kind: str,
    source_ids: Sequence[str],
    **save_kwargs: Any,
) -> MemoryRecord | None:
    """``mem.save`` with dream conflict-staging.

    Outside a dream run or with the flag off, this is exactly ``mem.save`` and
    ``WriteRefused`` propagates as before. Inside a dream run with the flag on, a
    ``WriteRefused`` (and only that) parks the proposal and returns ``None``;
    every other exception re-raises so callers' ``save_failed`` path is intact.
    """
    if not _staging_active():
        return mem.save(**save_kwargs)
    try:
        return mem.save(**save_kwargs)
    except WriteRefused as exc:
        stage_proposal(
            cfg,
            mem,
            kind=kind,
            save_kwargs=save_kwargs,
            source_ids=source_ids,
            conflict=exc.conflict,
        )
        return None


def resume_staged_proposals(cfg: Config, mem: Memory) -> dict[str, Any]:
    """Re-apply parked proposals whose blocking conflicts are gone.

    A proposal is unblocked when none of its ``conflict_ids`` are still active.
    Resolution stays human-only (or auto-clear on subject delete) — this never
    resolves a conflict. Replay runs OUTSIDE the staging scope so a still-blocked
    proposal re-stages (bumping attempts) rather than being lost. Returns
    ``{applied, still_blocked, errors, total_open}``.
    """
    proposals = _load(cfg)
    try:
        active_ids = {str(r.get("id") or "") for r in mem.operational.active_conflicts()}
    except Exception:
        active_ids = set()

    applied: list[str] = []
    errors: list[str] = []
    still_blocked = 0
    remaining: list[StagedProposal] = []

    token = _STAGING_SCOPE.set(False)
    try:
        for proposal in proposals:
            if proposal.state != "staged":
                continue  # applied/dropped entries are pruned on rewrite
            if any(cid in active_ids for cid in proposal.conflict_ids):
                still_blocked += 1
                remaining.append(proposal)
                continue
            try:
                mem.save(**proposal.save_kwargs)
            except WriteRefused as exc:
                # A different / remaining conflict still blocks: re-stage with
                # the new blocker's evidence so the next resume targets it.
                still_blocked += 1
                cids, summ, evid = _capture_conflict(mem, exc.conflict)
                remaining.append(
                    replace(
                        proposal,
                        conflict_ids=tuple(cids) or proposal.conflict_ids,
                        conflict_summary=summ or proposal.conflict_summary,
                        evidence_uris=tuple(evid) or proposal.evidence_uris,
                        attempts=proposal.attempts + 1,
                    )
                )
                continue
            except MemoError as exc:
                errors.append(f"{proposal.proposal_id}: {type(exc).__name__}: {exc}")
                remaining.append(proposal)  # keep for a later retry
                continue
            applied.append(proposal.proposal_id)  # pruned (not carried forward)
    finally:
        _STAGING_SCOPE.reset(token)

    _save(cfg, remaining)
    return {
        "applied": applied,
        "still_blocked": still_blocked,
        "errors": errors,
        "total_open": sum(1 for p in remaining if p.state == "staged"),
    }


def list_staged(cfg: Config) -> list[StagedProposal]:
    """All currently-staged (awaiting-resolution) proposals."""
    return [p for p in _load(cfg) if p.state == "staged"]


def drop_staged(cfg: Config, proposal_id: str) -> bool:
    """Remove a staged proposal by id. Returns True if one was removed."""
    proposals = _load(cfg)
    kept = [p for p in proposals if p.proposal_id != proposal_id]
    if len(kept) == len(proposals):
        return False
    _save(cfg, kept)
    return True


def resolve_command(proposal: StagedProposal) -> str:
    """The exact human CLI command to resolve the proposal's blocking conflict.

    Mirrors the read-only MCP ``memo_conflict_resolve`` contract: conflict
    resolution is human-only via the CLI, never performed by staging.
    """
    cid = proposal.conflict_ids[0] if proposal.conflict_ids else "<conflict-id>"
    return f"memo operational conflict resolve {cid} '<resolution>' --actor <human>"


__all__ = [
    "StagedProposal",
    "dream_staging_scope",
    "drop_staged",
    "list_staged",
    "resolve_command",
    "resume_staged_proposals",
    "stage_proposal",
    "staged_save",
]
