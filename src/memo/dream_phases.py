"""Per-phase instrumentation + resumable checkpoints for `memo dream run`.

This is the observability keystone of the dream learning loop. Every dream
phase records a structured, timed receipt fragment so the pipeline stops being
an opaque batch job and becomes a measurable ``signal -> change -> result``
sequence:

    {
      "phase": "hype", "status": "done", "duration_ms": 1234,
      "input_count": 50, "changed_count": 12, "skipped_count": 38,
      "mutations": 36, "errors": [], "warnings": [],
      "quality_before": {}, "quality_after": {},
      "in_fingerprint": "...", "out_fingerprint": "..."
    }

Design constraints (kept deliberately small to stay surgical over the large,
concurrently-edited ``cli_dream.py``):

* ``PhaseRecorder`` uses an explicit ``begin()/end()`` pair — two lines per
  pass, no re-indentation of the existing bespoke ``try/except`` bodies.
* Counts are *inferred* from the pass result fragment (``receipt[key]``) via
  common maintenance keys, so passes need no hand-annotation; a pass may still
  override any field on the handle for precision.
* ``end()`` never raises: instrumentation must not be able to break a pass.
* ``DreamCheckpoint`` persists after every phase, so an interrupted run leaves a
  partial, readable receipt on disk and a ``--resume`` run can skip phases whose
  LLM calls / mutations already committed. The run fingerprint is the *previous*
  completed run's corpus fingerprint — stable across a crash+restart cycle
  (a crashed run never rewrites ``last.json``), and naturally invalidated once a
  full run completes and stamps a new ``corpus_fp``.

Markdown stays the source of truth; this module only reads the derived index
(``_corpus_fingerprint``) and writes a JSON sidecar under ``state_dir/dream/``.
"""

from __future__ import annotations

import json
import logging as _logging
import time
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from memo.dream_utils import _corpus_fingerprint

_log = _logging.getLogger(__name__)

CHECKPOINT_VERSION = 1
CHECKPOINT_NAME = "checkpoint.json"

_UNSET = object()

# Result-fragment keys that count as "inputs seen" and "mutations made". Passes
# return heterogeneous dicts; these cover the maintenance vocabulary so the
# recorder can infer counts without every call site annotating them by hand.
_INPUT_KEYS = ("scanned", "processed", "considered", "candidates", "examined", "seen")
_MUTATION_KEYS = (
    "saved",
    "updated",
    "written",
    "promoted",
    "merged",
    "archived",
    "superseded",
    "captured",
    "synthesized",
    "extracted",
    "reconciled",
    "decayed",
    "pruned",
    "evicted",
    "compressed",
    "retagged",
    "indexed",
    "applied",
    "remapped",
)


def _now_ms() -> float:
    """Monotonic millisecond clock for durations (never walks backwards)."""
    return time.perf_counter() * 1000.0


def _coerce_count(value: Any) -> int:
    """Best-effort count from a fragment value: len() of a list, int of a number."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, list):
        return len(value)
    return 0


def _infer_counts(fragment: Any) -> tuple[int, int]:
    """Infer (input_count, mutations) from a pass result fragment.

    Returns (0, 0) for anything that isn't a dict so unannotated / opaque passes
    degrade quietly rather than raising inside instrumentation.
    """
    if not isinstance(fragment, dict):
        return 0, 0
    inputs = max((_coerce_count(fragment.get(k)) for k in _INPUT_KEYS), default=0)
    mutations = sum(_coerce_count(fragment.get(k)) for k in _MUTATION_KEYS)
    return inputs, mutations


@dataclass
class PhaseHandle:
    """Mutable handle a pass body may populate for precise metrics.

    All fields are optional; anything left unset is inferred at ``end()`` from
    the pass result fragment. ``restored`` is set by ``PhaseRecorder.restore()``
    so the pass body can cheaply skip its work on a ``--resume`` run.
    """

    name: str
    fragment_key: str | None = None
    input_count: int | None = None
    changed_count: int | None = None
    skipped_count: int = 0
    mutations: int | None = None
    warnings: list[str] = field(default_factory=list)
    quality_before: dict[str, Any] = field(default_factory=dict)
    quality_after: dict[str, Any] = field(default_factory=dict)
    status: str | None = None
    restored: bool = False
    # internal bookkeeping
    _t0: float = 0.0
    _err_before: int = 0
    _in_fp: str | None = None
    _track_fp: bool = False
    _closed: bool = False
    _fragment: Any = _UNSET


class DreamCheckpoint:
    """Resumable, idempotent phase ledger under ``state_dir/dream/checkpoint.json``.

    Keyed on ``run_fingerprint`` so a *new* run (different previous-corpus fp)
    starts clean while an interrupted run with the same fp can resume from its
    last completed phase.
    """

    def __init__(self, path: Path, run_fingerprint: str) -> None:
        self.path = path
        self.run_fingerprint = run_fingerprint
        self._done: dict[str, dict[str, Any]] = {}
        self._loaded_fp: str | None = None
        self._load()

    def _load(self) -> None:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if not isinstance(data, dict):
            return
        self._loaded_fp = data.get("run_fingerprint")
        done = data.get("done")
        if isinstance(done, dict):
            self._done = done

    def resumable(self) -> bool:
        """True when an on-disk checkpoint matches this run and has content."""
        return self._loaded_fp == self.run_fingerprint and bool(self._done)

    def is_done(self, name: str) -> bool:
        return self.resumable() and name in self._done

    def phase_record(self, name: str) -> dict[str, Any] | None:
        entry = self._done.get(name) if self.is_done(name) else None
        return entry.get("phase") if entry else None

    def fragment(self, name: str) -> Any:
        entry = self._done.get(name) if self.is_done(name) else None
        return entry.get("fragment") if entry else None

    def record(self, name: str, phase_record: dict[str, Any], fragment: Any = None) -> None:
        """Persist one completed phase. Instrumentation must never crash a run, so
        a write failure is logged and swallowed (the in-memory receipt is intact)."""
        self._done[name] = {"phase": phase_record, "fragment": fragment}
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "version": CHECKPOINT_VERSION,
                "run_fingerprint": self.run_fingerprint,
                "ts": time.time(),
                "done": self._done,
            }
            self.path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except (OSError, TypeError, ValueError) as exc:
            _log.warning("dream checkpoint write failed for %s: %s", name, exc)

    def clear(self) -> None:
        """Drop the checkpoint after a fully successful run."""
        self._done = {}
        with suppress(OSError):
            self.path.unlink()


class PhaseRecorder:
    """Records a structured, timed record per dream phase into ``receipt['phases']``.

    Usage (no re-indentation of existing pass bodies)::

        _ph = rec.begin("hype", fragment_key="hype")
        if rec.restore(_ph):          # --resume hit: skip work, restore fragment
            ...                        # (optional) light restore-path branch
        else:
            ... run the pass, set receipt["hype"] = ... ...
        rec.end(_ph)

    ``end()`` is idempotent and never raises.
    """

    def __init__(
        self,
        receipt: dict[str, Any],
        *,
        mem: Any = None,
        checkpoint: DreamCheckpoint | None = None,
        resume: bool = False,
    ) -> None:
        self.receipt = receipt
        self.mem = mem
        self.checkpoint = checkpoint
        self.resume = resume
        receipt.setdefault("phases", [])

    # -- lifecycle --------------------------------------------------------

    def begin(
        self,
        name: str,
        *,
        fragment_key: str | None = None,
        input_count: int | None = None,
        track_fp: bool | None = None,
    ) -> PhaseHandle:
        """Open a phase. ``track_fp`` snapshots the corpus fingerprint before/after
        (cheap COUNT+MAX query) to detect real mutations; defaults on when a
        ``mem`` is available. Pass ``track_fp=False`` for pure-read passes."""
        h = PhaseHandle(name=name, fragment_key=fragment_key, input_count=input_count)
        h._t0 = _now_ms()
        h._err_before = len(self.receipt.get("errors", []))
        h._track_fp = self.mem is not None if track_fp is None else track_fp
        if h._track_fp:
            h._in_fp = _corpus_fingerprint(self.mem)
        return h

    def restore(self, handle: PhaseHandle) -> bool:
        """On a ``--resume`` run, if this phase already completed, append its cached
        record (flagged ``resumed``), restore its result fragment into the receipt,
        and return True so the caller skips the (LLM/mutation) work. Else False."""
        if not (self.resume and self.checkpoint and self.checkpoint.is_done(handle.name)):
            return False
        record = self.checkpoint.phase_record(handle.name)
        fragment = self.checkpoint.fragment(handle.name)
        if handle.fragment_key and fragment is not None:
            self.receipt[handle.fragment_key] = fragment
        if isinstance(record, dict):
            self.receipt["phases"].append({**record, "resumed": True})
        handle.restored = True
        handle._closed = True
        return True

    def end(self, handle: PhaseHandle, *, fragment: Any = _UNSET) -> dict[str, Any] | None:
        """Close a phase: compute duration/status/counts, append the structured
        record, and checkpoint it. No-op if already closed (e.g. after restore).

        ``fragment`` explicitly supplies the pass result for count inference —
        used by ``timed()`` where the result is not yet assigned to the receipt."""
        if handle._closed:
            return None
        handle._closed = True
        if fragment is not _UNSET:
            handle._fragment = fragment
        try:
            return self._finalize(handle)
        except Exception as exc:  # instrumentation must never break a run
            _log.warning("dream phase recorder failed for %s: %s", handle.name, exc)
            return None

    def timed(
        self,
        name: str,
        thunk: Any,
        *,
        fragment_key: str | None = None,
        resumable: bool = False,
    ) -> Any:
        """Run ``thunk()`` under phase instrumentation, returning its result.

        One-line call-site wrapper (no re-indentation of the surrounding
        ``try/except``)::

            receipt["hype"] = rec.timed("hype", lambda: run_hype_pass(...),
                                        fragment_key="hype", resumable=True)

        On a ``--resume`` run where this phase already committed, the cached
        result fragment is returned WITHOUT calling ``thunk`` (no repeated LLM
        calls / mutations). Counts are inferred from the returned fragment.
        """
        ph = self.begin(name, fragment_key=fragment_key)
        if resumable and self.resume and self.checkpoint and self.checkpoint.is_done(name):
            record = self.checkpoint.phase_record(name)
            fragment = self.checkpoint.fragment(name)
            if isinstance(record, dict):
                self.receipt["phases"].append({**record, "resumed": True})
            ph.restored = True
            ph._closed = True
            return fragment
        result: Any = None
        raised: BaseException | None = None
        try:
            result = thunk()
            return result
        except BaseException as exc:
            raised = exc
            raise
        finally:
            # A pass that raised appends to receipt["errors"] in ITS OWN except
            # block, which runs AFTER this finally — so mark the failure here or
            # the crashed phase would record status="done".
            if raised is not None:
                ph.status = "error"
                ph.warnings.append(f"{type(raised).__name__}: {raised}")
            # A helper-style pass returns None and mutates receipt[key] directly;
            # fall back to the receipt fragment so counts still infer. A pass that
            # returns its dict result uses that directly.
            self.end(ph, fragment=result if result is not None else _UNSET)

    # -- internals --------------------------------------------------------

    def _finalize(self, h: PhaseHandle) -> dict[str, Any]:
        duration_ms = round(_now_ms() - h._t0, 1)
        new_errors = list(self.receipt.get("errors", [])[h._err_before :])

        if h._fragment is not _UNSET:
            fragment = h._fragment
        elif h.fragment_key:
            fragment = self.receipt.get(h.fragment_key)
        else:
            fragment = None
        inferred_inputs, inferred_mutations = _infer_counts(fragment)

        out_fp = _corpus_fingerprint(self.mem) if h._track_fp and self.mem is not None else None
        fp_changed = h._in_fp is not None and out_fp is not None and h._in_fp != out_fp

        input_count = h.input_count if h.input_count is not None else inferred_inputs
        mutations = h.mutations if h.mutations is not None else inferred_mutations
        if h.changed_count is not None:
            changed_count = h.changed_count
        elif mutations:
            changed_count = mutations
        else:
            changed_count = 1 if fp_changed else 0

        # Status derivation: an explicit override wins; otherwise a pass whose
        # result fragment reports status=="error", or that appended to
        # receipt["errors"], is an error — matching the existing convention.
        status = h.status
        if status is None:
            frag_status = fragment.get("status") if isinstance(fragment, dict) else None
            status = "error" if frag_status == "error" or new_errors else "done"

        record: dict[str, Any] = {
            "phase": h.name,
            "status": status,
            "duration_ms": duration_ms,
            "input_count": input_count,
            "changed_count": changed_count,
            "skipped_count": h.skipped_count,
            "mutations": mutations,
            "errors": new_errors,
            "warnings": list(h.warnings),
            "quality_before": dict(h.quality_before),
            "quality_after": dict(h.quality_after),
            "in_fingerprint": h._in_fp,
            "out_fingerprint": out_fp,
        }
        if h.fragment_key:
            record["fragment_key"] = h.fragment_key
        self.receipt["phases"].append(record)
        if self.checkpoint is not None:
            self.checkpoint.record(h.name, record, fragment)
        return record


def summarize_phases(receipt: dict[str, Any]) -> dict[str, Any]:
    """Roll ``receipt['phases']`` into a compact summary for status/JSON output."""
    phases = receipt.get("phases") or []
    total_ms = round(sum(float(p.get("duration_ms", 0.0)) for p in phases), 1)
    slowest = max(
        phases,
        key=lambda p: float(p.get("duration_ms", 0.0)),
        default=None,
    )
    return {
        "count": len(phases),
        "total_duration_ms": total_ms,
        "mutations": sum(int(p.get("mutations", 0) or 0) for p in phases),
        "errors": sum(len(p.get("errors", []) or []) for p in phases),
        "resumed": sum(1 for p in phases if p.get("resumed")),
        "slowest": (
            {"phase": slowest["phase"], "duration_ms": slowest["duration_ms"]} if slowest else None
        ),
    }
