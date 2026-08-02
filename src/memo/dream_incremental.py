"""Fase 5 — per-pass incremental skip for the nightly dream pipeline.

The global convergence guard (``dream_utils.check_convergence``) skips the WHOLE
pipeline when the corpus fingerprint is unchanged. This module is the finer
grain: even when the corpus DID change somewhere, a content-derived pass whose
OWN dependency signal is unchanged since its last successful run can skip
re-deriving over the full corpus.

Each pass records a *dependency fingerprint* on success (``record_success``);
the next night skips it iff that fingerprint still matches (``should_skip``). The
dependency for the content-derived passes (entities / cross-cluster synthesis) is
the durable-memory content signal (``durable_content_fingerprint`` —
``(durable_count, max_updated)`` over durable types, mirroring
``dream_utils._corpus_fingerprint`` but scoped to durable content). Access-only
churn (ROI/decay) does not move it, so a night that only touched the access log
skips the content passes.

A skip is a pure optimization and always safe:

* **gated** default-off (``MEMO_DREAM_INCREMENTAL_ENABLED``),
* **reversible** (delete ``state_dir/dream/incremental.json`` or flip the flag),
* **self-healing** (any dependency change re-triggers the pass next night),
* **explainable** (the skip is recorded in the receipt with the fingerprint).

MLX-free: one cheap sqlite aggregate. Never raises into the pipeline.
"""

from __future__ import annotations

import contextlib
import json
import logging
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)

STATE_FILE = "incremental.json"


def _path(state_dir: Path) -> Path:
    return Path(state_dir) / "dream" / STATE_FILE


def durable_content_fingerprint(mem: Any) -> str | None:
    """``"<durable_count>:<max_updated>"`` over durable-tier rows — a cheap
    change-signal for content-derived passes. Any durable add/edit/delete moves
    it; access-only changes (ROI/decay) do not. ``None`` on any store error
    (caller then treats the pass as not-skippable — fail safe = run)."""
    try:
        from .tiers import DURABLE_TYPES

        # placeholders is only "?,?,..." (len == number of durable types); the
        # values are bound parameters, so this is not an injection vector.
        placeholders = ",".join("?" * len(DURABLE_TYPES))
        row = mem.store._conn.execute(
            f"SELECT COUNT(*), COALESCE(MAX(updated), '') FROM meta WHERE type IN ({placeholders})",  # noqa: S608
            tuple(DURABLE_TYPES),
        ).fetchone()
        return f"{row[0]}:{row[1]}"
    except Exception as exc:  # defensive: a fingerprint failure must never skip
        _log.debug("dream_incremental: durable fingerprint failed: %s", exc)
        return None


def _load(state_dir: Path) -> dict[str, Any]:
    try:
        doc = json.loads(_path(state_dir).read_text(encoding="utf-8"))
        return doc if isinstance(doc, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def should_skip(state_dir: Path, pass_name: str, fingerprint: str | None) -> bool:
    """True iff ``pass_name`` last succeeded on exactly this ``fingerprint``.

    A ``None`` fingerprint (store error) is never skippable — fail safe by
    running the pass. Unknown pass / mismatch → not skippable → run + re-stamp.
    """
    if fingerprint is None:
        return False
    return str(_load(state_dir).get(pass_name) or "") == fingerprint


def record_success(state_dir: Path, pass_name: str, fingerprint: str | None) -> None:
    """Stamp ``pass_name``'s dependency fingerprint after a successful run.

    A ``None`` fingerprint clears any stored value (so the pass re-runs next
    night rather than being wrongly skipped). Best-effort — never raises."""
    try:
        state = _load(state_dir)
        if fingerprint is None:
            state.pop(pass_name, None)
        else:
            state[pass_name] = fingerprint
        p = _path(state_dir)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(state, indent=2), encoding="utf-8")
    except OSError as exc:
        _log.debug("dream_incremental: record_success failed for %s: %s", pass_name, exc)


def clear(state_dir: Path) -> None:
    """Forget every per-pass fingerprint (next run re-derives everything)."""
    with contextlib.suppress(OSError):
        _path(state_dir).unlink()


def run_or_skip(
    state_dir: Path,
    pass_name: str,
    fingerprint: str | None,
    runner: Any,
) -> dict[str, Any]:
    """Run ``runner()`` unless ``pass_name``'s dependency is unchanged.

    On skip returns ``{"status": "skipped_incremental", "fingerprint": fp}``
    without invoking ``runner``. On run, invokes ``runner()``, and — only if the
    result did not error (no ``error`` key / ``status`` != ``"error"``) — stamps
    the fingerprint so the next unchanged night skips it. Returns the runner's
    own result dict verbatim on run. Never raises: a fingerprint failure just
    means the pass runs (fail safe)."""
    if should_skip(state_dir, pass_name, fingerprint):
        return {"status": "skipped_incremental", "fingerprint": fingerprint}
    result = runner()
    errored = isinstance(result, dict) and (
        "error" in result or result.get("status") == "error"
    )
    if not errored:
        record_success(state_dir, pass_name, fingerprint)
    return result if isinstance(result, dict) else {"result": result}
