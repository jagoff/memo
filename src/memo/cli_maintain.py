"""`memo maintain` — keep the corpus fresh so memo stays a source of truth.

Orchestrates the freshness machinery that previously only ran by hand:

  1. Contradictions — scan for near-neighbours the helper LLM flags as
     contradicting/evolving. An *evolution* is resolved `evolved` (both kept).
     A genuine *contradiction* supersedes: the OLDER side is archived (moved to
     `inactive/`, reversible) and the pair resolved `kept_newer`. `--hard-delete`
     opts into a real delete for the most confident contradictions instead.
  2. Duplicates — consolidate high-similarity clusters (merge → archive the
     sources with an `archived_for` pointer; reversible).
  3. Staleness — archive memories never accessed and older than `--stale-days`.

Every mutation is **reversible by default** (archive, not delete). `--dry-run`
previews without touching anything. A receipt of what changed is written to
`<state>/maintain/last.json` plus a timestamp the SessionStart guard reads so
the auto-run fires at most once per day.

The auto-run (MEMO_MAINTAIN_AUTO daily guard) always uses the safe archive
path; `--hard-delete` is manual-only.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import click

from memo.belief import ARCHIVE, COMPETING, HOLD_OPEN, supersede_decision
from memo.cli_common import console
from memo.cli_common import get_memory as _get_memory
from memo.config import Config
from memo.errors import ValidationError
from memo.flags import flag_bool, flag_int
from memo.util import safe_operation

_log = logging.getLogger(__name__)


def _state_path(cfg: Config):
    return cfg.state_dir / "maintain"


def _prepare_quality_compact_receipt_paths(cfg: Config) -> tuple[Path, Path, str]:
    """Fail closed if apply-mode receipt persistence cannot be set up."""

    d = _state_path(cfg)
    runs_dir = d / "runs"
    run_stamp = f"{int(time.time())}-{os.getpid()}-{time.time_ns()}"
    try:
        runs_dir.mkdir(parents=True, exist_ok=True)
        probe = d / ".quality_compact_receipt_probe"
        probe.write_text(run_stamp, encoding="utf-8")
        probe.unlink(missing_ok=True)
    except OSError as exc:
        raise ValidationError(f"quality compaction receipt setup failed: {exc}") from exc
    return d, runs_dir, run_stamp


def _persist_quality_compact_receipt(
    d: Path,
    runs_dir: Path,
    run_stamp: str,
    payload: str,
) -> None:
    """Publish run + last receipts without exposing a rolled-back last.json."""

    run_path = runs_dir / f"{run_stamp}.json"
    last_path = d / "last.json"
    staged: dict[Path, Path] = {}
    published_run = False
    try:
        for path in (run_path, last_path):
            tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
            tmp.write_text(payload, encoding="utf-8")
            staged[path] = tmp
        os.replace(staged[run_path], run_path)
        published_run = True
        os.replace(staged[last_path], last_path)
    except OSError:
        for tmp in staged.values():
            tmp.unlink(missing_ok=True)
        if published_run:
            run_path.unlink(missing_ok=True)
        raise


def _synthesis_state_path(cfg: Config) -> Path:
    return cfg.state_dir / "synthesis_state.json"


@safe_operation(fallback=None, log_level=logging.DEBUG, error_message="maintain: could not read synthesis_state.json")
def _read_synthesis_last_run(cfg: Config) -> str | None:
    """Return the ISO timestamp of the last synthesis run, or None."""
    p = _synthesis_state_path(cfg)
    data = json.loads(p.read_text(encoding="utf-8"))
    return data.get("last_run") or None


def _write_synthesis_last_run(cfg: Config, ts: str) -> None:
    """Persist the synthesis run timestamp to state_dir/synthesis_state.json."""
    p = _synthesis_state_path(cfg)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"last_run": ts}, ensure_ascii=False), encoding="utf-8")
    except Exception as exc:
        _log.warning("maintain: could not write synthesis_state.json: %s", exc)


def _older_id(mem: Any, id_a: str, id_b: str) -> tuple[str, str]:
    """Return (older_id, newer_id) by `updated` timestamp; falls back to the
    pair order (a, b) when a record or timestamp is missing."""
    ra, rb = mem.get(id_a), mem.get(id_b)
    ua = getattr(ra, "updated", "") or ""
    ub = getattr(rb, "updated", "") or ""
    if ua and ub:
        return (id_a, id_b) if ua <= ub else (id_b, id_a)
    return id_a, id_b


def _undo_targets(receipt: dict[str, Any]) -> tuple[list[str], list[str]]:
    """(archived_ids, soft_forgotten_ids) recorded in a maintain receipt."""
    archived: list[str] = []
    for s in receipt.get("superseded", []):
        if isinstance(s, dict) and s.get("action") == "archive" and s.get("older"):
            archived.append(s["older"])
    for m in receipt.get("merged", []):
        if isinstance(m, dict):
            archived.extend(m.get("archived_ids") or [])
    for a in receipt.get("archived_stale", []):
        if isinstance(a, dict) and a.get("id"):
            archived.append(a["id"])
    for q in receipt.get("quality_compacted", []):
        if isinstance(q, dict):
            archived.extend(q.get("archived_ids") or [])
    forgotten: list[str] = [x for x in receipt.get("dead_archived", []) if isinstance(x, str)]
    for f in receipt.get("forgotten", []):
        if isinstance(f, dict) and f.get("id"):
            forgotten.append(f["id"])
    return archived, forgotten


def _quality_compact_rollback_ids(receipt: dict[str, Any]) -> list[str]:
    """Conservatively include every attempted archive when rolling back apply."""

    rollback_ids: list[str] = []
    seen: set[str] = set()
    for item in receipt.get("quality_compacted", []):
        if not isinstance(item, dict):
            continue
        for value in item.get("attempted_ids") or item.get("archived_ids") or []:
            source_id = str(value or "")
            if not source_id or source_id in seen:
                continue
            seen.add(source_id)
            rollback_ids.append(source_id)
    return rollback_ids


def _rollback_quality_compaction(mem: Any, receipt: dict[str, Any]) -> tuple[list[str], list[str]]:
    rollback_ids = _quality_compact_rollback_ids(receipt)
    restored, missing = _restore_archived(mem, rollback_ids, dry_run=False)
    if rollback_ids:
        mem.reindex()
    return restored, missing


def _restore_archived(mem: Any, ids: list[str], *, dry_run: bool) -> tuple[list[str], list[str]]:
    """Move receipt-listed .md files back out of inactive/ (matched by
    frontmatter id — the receipt stores ids, not filenames). Returns
    (restored_ids, missing_ids). Caller runs `mem.reindex()` afterwards."""
    import shutil

    import frontmatter

    inactive_dir = mem.cfg.memory_dir / "inactive"
    wanted = set(ids)
    restored: list[str] = []
    if inactive_dir.is_dir():
        for p in sorted(inactive_dir.glob("*.md")):
            try:
                fid = str(frontmatter.load(p).get("id") or "")
            except Exception:  # noqa: S112 — a corrupt archived file must not sink the restore
                continue
            if fid not in wanted:
                continue
            dest = mem.cfg.memory_dir / p.name
            if dest.exists():
                _log.warning("maintain undo: %s exists; skipping restore of %s", dest, fid[:8])
                continue
            if not dry_run:
                shutil.move(str(p), str(dest))
            restored.append(fid)
    return restored, sorted(wanted - set(restored))


@click.group(name="maintain", invoke_without_command=True)
@click.option("--dry-run", is_flag=True, help="Preview actions; change nothing.")
@click.option(
    "--min-confidence",
    type=float,
    default=0.9,
    help="Confidence floor for auto-acting on a contradiction (default 0.9).",
)
@click.option(
    "--hard-delete",
    is_flag=True,
    help="Delete (not archive) the superseded side of a contradiction. "
    "Manual-only; the daily auto-run never does this.",
)
@click.option(
    "--stale-days",
    type=int,
    default=365,
    help="Archive never-accessed memories older than this (default 365).",
)
@click.option(
    "--dup-threshold",
    type=float,
    default=0.9,
    help="Cosine threshold for duplicate clustering (default 0.9).",
)
@click.option(
    "--max-pairs",
    type=int,
    default=200,
    help="Max contradiction candidate pairs to scan (default 200).",
)
@click.option(
    "--max-scan-seconds",
    type=float,
    default=None,
    help="Wall-clock timeout for the contradiction scan pass in seconds. "
    "Stops early if exceeded (default: no limit).",
)
@click.option("--skip-contradict", is_flag=True, help="Skip the contradiction pass.")
@click.option("--skip-consolidate", is_flag=True, help="Skip the duplicate-merge pass.")
@click.option("--skip-stale", is_flag=True, help="Skip the staleness pass.")
@click.option("--vacuum", is_flag=True, help="Permanently delete soft-deleted records older than --vacuum-days.")
@click.option("--vacuum-days", default=90, type=int, help="Age threshold for vacuum cleanup.")
@click.option(
    "--skip-synthesize",
    is_flag=True,
    help="Skip the emergent-synthesis pass (requires MEMO_SYNTHESIS_ENABLED=1).",
)
@click.option("--json", "as_json", is_flag=True, help="Emit the receipt as JSON.")
@click.option(
    "--if-due",
    is_flag=True,
    help="No-op unless >24h since the last run; then spawn maintain "
    "detached (safe archive-only) and return. For the daily "
    "SessionStart guard.",
)
@click.pass_context
def maintain_cmd(
    ctx: click.Context,
    dry_run: bool,
    min_confidence: float,
    hard_delete: bool,
    stale_days: int,
    dup_threshold: float,
    max_pairs: int,
    max_scan_seconds: float | None,
    skip_contradict: bool,
    skip_consolidate: bool,
    skip_stale: bool,
    skip_synthesize: bool,
    vacuum: bool,
    vacuum_days: int,
    as_json: bool,
    if_due: bool,
) -> None:
    """Supersede contradictions, merge duplicates, archive stale memories.

    Reversible by default (archives to inactive/). Example:
      memo maintain --dry-run
    """
    if ctx.invoked_subcommand is not None:
        return

    cfg = Config.from_env()

    # Daily guard for the SessionStart hook: cheap, builds no Memory.
    if if_due:
        import os as _os
        import subprocess as _sp

        if flag_bool("MEMO_MAINTAIN_DISABLE"):
            return
        ts_file = _state_path(cfg) / ".last_run_ts"
        try:
            last = float(ts_file.read_text().strip())
        except Exception:
            last = 0.0
        if (time.time() - last) < 24 * 3600:
            return  # ran recently — not due
        try:
            _state_path(cfg).mkdir(parents=True, exist_ok=True)
            # Optimistic stamp so repeated SessionStarts today don't pile up
            # spawns before the detached run finishes (it re-stamps on completion).
            ts_file.write_text(str(time.time()), encoding="utf-8")
            _sp.Popen(
                # safe defaults: archive-only, no --hard-delete
                # cap LLM work: 50 pairs × ~7s = ~350s max, plus 300s wall-clock guard
                ["memo", "maintain", "--max-pairs", "50", "--max-scan-seconds", "300"],
                stdin=_sp.DEVNULL,
                stdout=_sp.DEVNULL,
                stderr=_sp.DEVNULL,
                start_new_session=True,
                # Route the scan's embeds through the warm recall daemon: they
                # serialize in its queue instead of cold-loading a 2nd embedder
                # and grabbing the GPU cross-process flock independently, which
                # starves the live recall daemon (recall_lock_bail on
                # embed_query) for the whole maintain window. Falls back to
                # in-process automatically if the daemon is down; respects an
                # explicit override.
                env={
                    **_os.environ,
                    "MEMO_NONINTERACTIVE": "1",
                    "MEMO_EMBEDDER_VIA_DAEMON": _os.environ.get(
                        "MEMO_EMBEDDER_VIA_DAEMON", "1"
                    ),
                },
            )
        except Exception as exc:
            _log.warning("maintain --if-due: failed to spawn background maintain: %s", exc)
        return

    mem = _get_memory(cfg)
    receipt: dict[str, Any] = {
        "dry_run": dry_run,
        "hard_delete": hard_delete,
        "superseded": [],  # contradictions acted on
        "flagged_for_review": [],  # high-support losers held for manual triage (C2 gate)
        "competing": [],  # contradictions within trust margin: both kept (belief mode)
        "evolved": [],  # contradictions marked evolution (both kept)
        "merged": [],  # duplicate clusters consolidated
        "forgotten": [],  # forget_after TTL elapsed (soft, reversible)
        "archived_stale": [],
        "synthesized": [],  # emergent cross-memory insights generated
        "synthesis_count": 0,  # new clusters synthesized by proactive pass
        "outcome_reconciled": 0,  # memories whose roi_score was re-derived from outcomes
        "dead_archived": [],  # surfaced-never-grounded memories soft-forgotten
        "errors": [],
        "cascade_warnings": [],
    }

    # 0. Explicit forget TTLs ------------------------------------------------
    # Honour user-set `forget_after` dates: soft-forget (reversible) so the
    # memory drops out of recall/search without losing the file. Runs even
    # when other passes are skipped — it's explicit user intent, not heuristics.
    try:
        for item in mem.lifecycle.enforce_forget_ttl(dry_run=dry_run):
            receipt["forgotten"].append(item)
    except Exception as exc:
        receipt["errors"].append(f"forget: {type(exc).__name__}: {exc}")

    # 1. Contradictions ------------------------------------------------------
    if not skip_contradict:
        try:
            mem.contradict_scanner.scan_corpus(
                confidence_threshold=min_confidence,
                max_pairs=max_pairs,
                max_seconds=max_scan_seconds,
            )
            from memo.flags import flag_float as _flag_float

            _evo_conf = _flag_float("MEMO_EVOLUTION_CONFIDENCE")
            _evo_conf = 0.6 if _evo_conf is None else _evo_conf
            for pair in mem.contradict_store.list_open(min_confidence=min_confidence):
                rel = (pair.relationship or "").lower()
                if "evolu" in rel:
                    if not dry_run:
                        # Demote the superseded (older) side via its confidence
                        # so the evolution verdict steers ranking (health-score
                        # multiplier, default-on) instead of changing nothing.
                        older, _newer = _older_id(mem, pair.memory_id_a, pair.memory_id_b)
                        if _evo_conf < 1.0:
                            try:
                                mem.store.set_confidence_batch([(older, _evo_conf)])
                            except Exception as _exc:
                                receipt["errors"].append(
                                    f"evolution_confidence: {type(_exc).__name__}: {_exc}"
                                )
                        mem.contradict_store.resolve(
                            pair.pair_id,
                            "evolved",
                            note=f"auto: evolution, demoted older {older[:8]}",
                        )
                    receipt["evolved"].append(pair.pair_id)
                    continue
                if "contrad" not in rel:
                    continue  # consistent / unrelated — leave open
                older, _newer = _older_id(mem, pair.memory_id_a, pair.memory_id_b)
                if flag_bool("MEMO_CROSSREF_INDEX"):
                    try:
                        _refs = [
                            b.source_id for b in mem.crossref.referencing_sources(older)
                        ]
                    except Exception:
                        _refs = []
                    if _refs:
                        receipt["cascade_warnings"].append(
                            {"target": older, "action": "supersede", "referenced_by": _refs}
                        )
                decision = supersede_decision(mem, older_id=older, newer_id=_newer)
                if decision.action == COMPETING:
                    if not dry_run:
                        mem.contradict_store.resolve(
                            pair.pair_id, "competing", note=f"auto: competing — {decision.reason}"
                        )
                    receipt.setdefault("competing", []).append(pair.pair_id)
                    continue
                if decision.action == HOLD_OPEN:
                    receipt["flagged_for_review"].append(
                        {
                            "pair_id": pair.pair_id,
                            "older": decision.dominated_id,
                            "support_count": decision.support_dominated,
                        }
                    )
                    continue
                assert decision.action == ARCHIVE
                target = decision.dominated_id
                action = "delete" if hard_delete else "archive"
                if not dry_run:
                    ok = (
                        mem.delete(target)
                        if hard_delete
                        else mem.lifecycle.archive_memory(target, superseded_by=decision.dominant_id)
                    )
                    if ok:
                        mem.contradict_store.resolve(
                            pair.pair_id, "kept_newer", note=f"auto: {action}d {target} — {decision.reason}"
                        )
                receipt["superseded"].append(
                    {
                        "pair_id": pair.pair_id,
                        "older": target,
                        "action": action,
                        "confidence": pair.confidence,
                    }
                )
        except Exception as exc:
            receipt["errors"].append(f"contradict: {type(exc).__name__}: {exc}")

    # 2. Duplicates ----------------------------------------------------------
    if not skip_consolidate:
        try:
            res = mem.consolidator.consolidate_all(
                threshold=dup_threshold,
                auto_apply=True,
                dry_run=dry_run,
            )
            for r in res.get("results", []):
                receipt["merged"].append(
                    {"merged_id": r.get("merged_id"), "archived_ids": r.get("archived_ids", [])}
                )
            if not res.get("results") and res.get("proposals"):
                # dry_run path: proposals exist but nothing applied
                receipt["merged"] = [
                    {"would_merge": p.get("memory_ids")} for p in res.get("proposals", [])
                ]
        except Exception as exc:
            receipt["errors"].append(f"consolidate: {type(exc).__name__}: {exc}")

    # 3. Staleness -----------------------------------------------------------
    if not skip_stale:
        try:
            stale = mem.temporal.detect_stale_memories(
                days_threshold=stale_days, min_access_count=0
            )
            for item in stale:
                mid = item.get("id")
                if not mid:
                    continue
                if not dry_run:
                    mem.lifecycle.archive_memory(mid)
                receipt["archived_stale"].append({"id": mid, "days": item.get("days_since_update")})
        except Exception as exc:
            receipt["errors"].append(f"stale: {type(exc).__name__}: {exc}")

    # 4. Vacuum: permanently delete soft-deleted records older than --vacuum-days ---
    if vacuum:
        try:
            cutoff = (datetime.now(UTC) - timedelta(days=vacuum_days)).isoformat()
            ids = mem.store.list_soft_deleted(before=cutoff)
            if ids:
                receipt["vacuumed"] = len(ids)
                if not dry_run:
                    for vid in ids:
                        mem.store.hard_delete(vid)
                _log.info("vacuum: %d soft-deleted records purged", len(ids))
        except Exception as exc:
            receipt["errors"].append(f"vacuum: {type(exc).__name__}: {exc}")

    # 5. Emergent synthesis (opt-out: MEMO_SYNTHESIS_ENABLED=0 to disable) -----
    if not skip_synthesize and flag_bool("MEMO_SYNTHESIS_ENABLED"):
        try:
            results = mem.synthesize_cross_cluster(dry_run=dry_run)
            for r in results:
                receipt["synthesized"].append(
                    {
                        "title": r.get("title"),
                        "confidence": r.get("confidence"),
                        "sources": r.get("sources", []),
                        "saved": r.get("saved", False),
                        "id": r.get("id"),
                    }
                )
            # Keep the counter honest: the emergent pass (default-on) is what
            # actually saves syntheses, so derive the count from the array
            # instead of leaving it at 0 for everyone without MEMO_MAINT_SYNTHESIZE.
            receipt["synthesis_count"] = len([s for s in receipt["synthesized"] if s.get("saved")])
        except Exception as exc:
            receipt["errors"].append(f"synthesize: {type(exc).__name__}: {exc}")

    # 5. Proactive synthesis (opt-in: MEMO_MAINT_SYNTHESIZE=1) ----------------
    # Runs synthesis on clusters that have been updated since the last run.
    # `_last_run` records the prior timestamp for informational logging; the
    # `synthesize_cross_cluster` method already skips clusters whose
    # sources_hash hasn't changed (built-in dedup), so re-running is cheap.
    # Non-blocking: a failure logs a warning but does not abort the cycle.
    if not skip_synthesize and flag_bool("MEMO_MAINT_SYNTHESIZE"):
        _last_run = _read_synthesis_last_run(cfg)
        _synth_ts = datetime.now(UTC).isoformat()
        _log.debug("maintain: proactive synthesis since %s", _last_run or "never")
        try:
            results = mem.synthesize_cross_cluster(dry_run=dry_run)
            _n_saved = sum(1 for r in results if r.get("saved"))
            receipt["synthesis_count"] = _n_saved
            _log.info("maintain: proactive synthesis: %d new clusters synthesized", _n_saved)
            if not dry_run:
                _write_synthesis_last_run(cfg, _synth_ts)
        except Exception as exc:
            _log.warning("maintain: proactive synthesis failed (non-fatal): %s", exc)
            receipt["errors"].append(f"maint_synthesize: {type(exc).__name__}: {exc}")

    # 6. Outcome loop (default-on; disable with MEMO_OUTCOME_RANKING_ENABLED=0) -
    # Self-tuning recall: re-derive roi_score from real grounding outcomes (was
    # the surfaced memory USED in the answer?) so ranking promotes what helps,
    # and reversibly archive dead weight (surfaced often, never grounded). Pure
    # derivation over the recall/grounding logs + one roi write.
    if flag_bool("MEMO_OUTCOME_RANKING_ENABLED"):
        try:
            from memo.outcome import compute_utilities, dead_weight, reconcile_roi

            min_surfaced = flag_int("MEMO_OUTCOME_DEAD_MIN_SURFACED") or 0
            dead = dead_weight(mem, min_surfaced=min_surfaced)
            if dry_run:
                receipt["outcome_reconciled"] = len(
                    compute_utilities(cfg.state_dir)["by_prefix"]
                )
                receipt["dead_archived"] = [d["id"] for d in dead]
            else:
                receipt["outcome_reconciled"] = reconcile_roi(mem).get("updated", 0)
                for d in dead:
                    if mem.forget(
                        d["id"], reason=f"outcome: surfaced {d['surfaced']}x without grounding"
                    ) is not None:
                        receipt["dead_archived"].append(d["id"])
        except Exception as exc:
            receipt["errors"].append(f"outcome: {type(exc).__name__}: {exc}")

    # Persist receipt + timestamp (the daily guard reads the timestamp). Even
    # a dry-run stamps so a preview doesn't immediately re-trigger; the guard
    # cares only about "ran recently".
    if not dry_run:
        try:
            d = _state_path(cfg)
            runs_dir = d / "runs"
            runs_dir.mkdir(parents=True, exist_ok=True)
            run_stamp = str(int(time.time()))
            payload = json.dumps(
                {"ts": time.time(), "run": run_stamp, **receipt},
                ensure_ascii=False,
                indent=2,
            )
            (d / "last.json").write_text(payload, encoding="utf-8")
            (runs_dir / f"{run_stamp}.json").write_text(payload, encoding="utf-8")
            (d / ".last_run_ts").write_text(str(time.time()), encoding="utf-8")
        except Exception as exc:
            receipt["errors"].append(f"receipt: {type(exc).__name__}: {exc}")

    if as_json:
        click.echo(json.dumps(receipt, ensure_ascii=False, indent=2))
        return

    tag = "[dim](dry-run)[/dim] " if dry_run else ""
    console.print(f"{tag}[bold]memo maintain[/bold]")
    console.print(
        f"  contradictions superseded: {len(receipt['superseded'])} "
        f"({'delete' if hard_delete else 'archive'}), "
        f"evolutions marked: {len(receipt['evolved'])}"
    )
    console.print(f"  duplicate clusters merged: {len(receipt['merged'])}")
    console.print(f"  forget_after TTLs applied: {len(receipt['forgotten'])}")
    console.print(f"  stale memories archived: {len(receipt['archived_stale'])}")
    if receipt.get("vacuumed"):
        console.print(f"  soft-deleted records vacuumed: {receipt['vacuumed']}")
    if receipt["synthesized"]:
        saved = sum(1 for s in receipt["synthesized"] if s.get("saved"))
        console.print(
            f"  emergent syntheses: {saved} saved, {len(receipt['synthesized'])} proposed"
        )
    if receipt.get("synthesis_count"):
        console.print(f"  synthesis: {receipt['synthesis_count']} new clusters synthesized")
    if receipt.get("outcome_reconciled") or receipt.get("dead_archived"):
        console.print(
            f"  outcome loop: roi_score re-derived for {receipt['outcome_reconciled']} "
            f"memories, {len(receipt['dead_archived'])} dead-weight archived"
        )
    if receipt["errors"]:
        for e in receipt["errors"]:
            console.print(f"  [yellow]warn:[/yellow] {e}")


@maintain_cmd.command(name="undo")
@click.option(
    "--run",
    "run_stamp",
    default=None,
    help="Receipt stamp under <state>/maintain/runs/<STAMP>.json (default: last run).",
)
@click.option("--dry-run", is_flag=True, help="Preview the restore; move nothing.")
@click.option("--json", "as_json", is_flag=True, help="Emit the undo receipt as JSON.")
def maintain_undo_cmd(run_stamp: str | None, dry_run: bool, as_json: bool) -> None:
    """Batch-restore a maintain run from its receipt.

    Moves receipt-listed .md files back out of inactive/, unforgets
    soft-forgotten ids, then reindexes. The merged record a consolidation
    created is NOT deleted — sources are restored alongside it (restore-only).
    """
    cfg = Config.from_env()
    d = _state_path(cfg)
    receipt_path = (d / "runs" / f"{run_stamp}.json") if run_stamp else (d / "last.json")
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        console.print(f"[red]no readable receipt at {receipt_path}: {exc}[/red]")
        raise SystemExit(1) from exc
    mem = _get_memory(cfg)
    archived_ids, forgotten_ids = _undo_targets(receipt)
    restored, missing = _restore_archived(mem, archived_ids, dry_run=dry_run)
    unforgotten: list[str] = []
    for fid in forgotten_ids:
        if dry_run or mem.unforget(fid) is not None:
            unforgotten.append(fid)
    if restored and not dry_run:
        mem.reindex()
    undo_receipt = {
        "dry_run": dry_run,
        "source_receipt": str(receipt_path),
        "restored": restored,
        "unforgotten": unforgotten,
        "missing": missing,
    }
    if not dry_run:
        try:
            (d / "undo-last.json").write_text(
                json.dumps({"ts": time.time(), **undo_receipt}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError as exc:
            console.print(f"[yellow]warn: undo receipt not persisted: {exc}[/yellow]")
    if as_json:
        click.echo(json.dumps(undo_receipt, ensure_ascii=False, indent=2))
        return
    tag = "[dim](dry-run)[/dim] " if dry_run else ""
    console.print(
        f"{tag}[bold]memo maintain undo[/bold] — restored {len(restored)}, "
        f"unforgotten {len(unforgotten)}, missing {len(missing)}"
    )


@maintain_cmd.command(name="quality-compact")
@click.option("--preview", is_flag=True, help="Preview proposals without changing memories.")
@click.option("--apply", "apply_changes", is_flag=True, help="Apply proposals and write a receipt.")
@click.option("--limit", default=20, type=int, show_default=True)
@click.option("--json", "as_json", is_flag=True, help="Emit JSON.")
def quality_compact_cmd(preview: bool, apply_changes: bool, limit: int, as_json: bool) -> None:
    """Preview or apply quality compaction proposals."""

    from memo.quality_compact import apply_quality_compaction, preview_quality_compaction

    if not flag_bool("MEMO_QUALITY_COMPACT"):
        raise click.ClickException("MEMO_QUALITY_COMPACT=1 is required")
    if apply_changes and preview:
        raise click.UsageError("choose either --preview or --apply")
    if not preview:
        preview = True

    cfg = Config.from_env()
    receipt_targets: tuple[Path, Path, str] | None = None
    if apply_changes:
        try:
            receipt_targets = _prepare_quality_compact_receipt_paths(cfg)
        except ValidationError as exc:
            raise click.ClickException(str(exc)) from exc

    mem = _get_memory(cfg)
    try:
        preview_receipt = preview_quality_compaction(mem, limit=limit)
    except ValidationError as exc:
        raise click.UsageError(str(exc)) from exc

    if apply_changes:
        if preview_receipt.get("errors"):
            receipt = {
                "mode": "apply",
                "proposals": preview_receipt["proposals"],
                "applied": [],
                "quality_compacted": [],
                "errors": list(preview_receipt.get("errors", [])),
            }
        else:
            applied = apply_quality_compaction(mem, preview_receipt["proposals"], dry_run=False)
            receipt = {
                "mode": "apply",
                "proposals": preview_receipt["proposals"],
                "applied": applied["quality_compacted"],
                "quality_compacted": applied["quality_compacted"],
                "errors": [*preview_receipt.get("errors", []), *applied.get("errors", [])],
            }
            if receipt["errors"]:
                restored, missing = _rollback_quality_compaction(mem, receipt)
                detail = "; ".join(str(err) for err in receipt["errors"])
                if missing:
                    detail = (
                        f"{detail}; rolled back {len(restored)} archived ids, "
                        f"{len(missing)} could not be restored"
                    )
                elif _quality_compact_rollback_ids(receipt):
                    detail = f"{detail}; archived changes were rolled back"
                raise click.ClickException(f"quality compaction apply failed: {detail}")
            assert receipt_targets is not None
            d, runs_dir, run_stamp = receipt_targets
            payload = json.dumps(
                {"ts": time.time(), "run": run_stamp, **receipt},
                ensure_ascii=False,
                indent=2,
            )
            try:
                _persist_quality_compact_receipt(d, runs_dir, run_stamp, payload)
            except OSError as exc:
                restored, missing = _rollback_quality_compaction(mem, receipt)
                detail = f"{type(exc).__name__}: {exc}"
                if missing:
                    detail = (
                        f"{detail}; rolled back {len(restored)} archived ids, "
                        f"{len(missing)} could not be restored"
                    )
                elif _quality_compact_rollback_ids(receipt):
                    detail = f"{detail}; archived changes were rolled back"
                raise click.ClickException(
                    f"quality compaction receipt persistence failed: {detail}"
                ) from exc
    else:
        receipt = preview_receipt

    if as_json:
        click.echo(json.dumps(receipt, ensure_ascii=False, indent=2))
        return

    tag = "[dim](apply)[/dim] " if apply_changes else "[dim](preview)[/dim] "
    console.print(
        f"{tag}[bold]memo maintain quality-compact[/bold] — {len(receipt['proposals'])} proposals"
    )
