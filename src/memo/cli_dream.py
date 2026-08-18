"""`memo dream` — autonomous background maintenance pipeline.

Bundles every maintenance pass into a single nightly run:

  1. Contradictions — scan + auto-resolve evolutions, supersede older sides.
  2. Duplicates — consolidate high-similarity clusters.
  3. Staleness — archive memories never accessed after 365 days.
  4. Synthesis — emergent cross-cluster insights (runs regardless of
     MEMO_SYNTHESIS_ENABLED; Dream mode always synthesises).
  5. Entity backfill — extract entities for memories not yet indexed.
  6. ROI decay — multiply roi_score × 0.98 for memories idle > 30 days
     so unaccessed knowledge gradually yields to frequently-used facts.

`memo dream run` — run once (foreground).
`memo dream status` — when did it last run, what did it change.
`memo dream if-due` — no-op unless > 24h since last run (for launchd).
"""

from __future__ import annotations

import json
import logging as _logging
import time
from contextlib import ExitStack
from typing import Any, cast

import click

from memo.cli_common import console
from memo.cli_common import get_memory as _get_memory
from memo.cli_dream_passes import (
    _build_orientation,
    _render_run_summary,
    _run_capture_weights,
    _run_code_drift,
    _run_compress,
    _run_consolidate_dups,
    _run_contradict,
    _run_entities,
    _run_eval_recall,
    _run_eviction,
    _run_floor_calibration,
    _run_graph_projection,
    _run_harvest_labels,
    _run_negative_capture,
    _run_presynthesis,
    _run_prewarm_queries,
    _run_proactive_refresh,
    _run_prune_floor,
    _run_roi_decay,
    _run_roi_reconcile,
    _run_signal_gather,
    _run_stale,
    _run_synthesis,
    _run_validity_extract,
)
from memo.config import Config
from memo.dream_phases import (
    CHECKPOINT_NAME,
    DreamCheckpoint,
    PhaseRecorder,
    summarize_phases,
)
from memo.dream_utils import (
    _corpus_fingerprint,
    _iso_now,
    _make_progress,
    _state_path,
    acquire_dream_lock,
    check_convergence,
    read_previous_fingerprint,
    release_dream_lock,
)
from memo.memory.record import derived_save_scope

_log = _logging.getLogger(__name__)


def _run_verbatim_pass(
    cfg: Config,
    *,
    dry_run: bool,
    receipt: dict[str, Any],
    progress: Any,
    step: Any,
) -> None:
    """Run the optional private transcript index outside dream_run's budget."""
    from memo import verbatim_index
    from memo.flags import flag_bool

    if not flag_bool("MEMO_VERBATIM_INDEX"):
        return
    progress.update(step, description="[verbatim] indexing transcript turns...")
    receipt["verbatim"] = verbatim_index.run_verbatim_index_pass(cfg, dry_run=dry_run)
    verbatim = receipt["verbatim"]
    if verbatim.get("status") == "error":
        receipt["errors"].append(f"verbatim: {verbatim.get('error')}")
    progress.update(
        step,
        description=f"[verbatim] [green]✓[/green]  {verbatim.get('status')}",
    )


def _run_calibration_pass(
    cfg: Config, mem: Any, *, dry_run: bool, receipt: dict[str, Any]
) -> dict[str, Any]:
    """Nightly confidence calibration. Gated on MEMO_RECALL_CONFIDENCE_GATE +
    best-effort: never raises out. Builds the predicted-vs-grounded map the
    recall gate reads."""
    from memo.flags import flag_bool

    if not flag_bool("MEMO_RECALL_CONFIDENCE_GATE"):
        return receipt
    try:
        from memo.confidence_calibration import build_calibration, save_calibration

        doc = build_calibration(cfg.state_dir, mem)
        if not dry_run:
            save_calibration(cfg.state_dir, doc)
        receipt["calibration"] = doc
    except Exception as exc:  # best-effort pass, mirror the tuners
        receipt["errors"].append(f"calibration: {type(exc).__name__}: {exc}")
    return receipt


def _run_orientation_phase(mem: Any, receipt: dict[str, Any], progress: Any, step: Any) -> None:
    """Render the pre-mutation inventory and surface best-effort failures."""
    progress.update(step, description="[dim]orientation — inventorying corpus...[/dim]")
    try:
        orientation = _build_orientation(mem)
        receipt["orientation"] = orientation
        from rich.panel import Panel
        from rich.table import Table

        tbl = Table(show_header=False, box=None, padding=(0, 1))
        tbl.add_column("", style="dim")
        tbl.add_column("", justify="right")
        tbl.add_row("total memories", str(orientation["total"]))
        for memory_type, count in sorted(orientation["by_type"].items()):
            tbl.add_row(f"  {memory_type}", str(count))
        tbl.add_row("roi < 0.3", str(orientation["low_roi"]))
        tbl.add_row("stale candidates (>365d)", str(orientation["stale_candidates"]))
        tbl.add_row("open contradictions", str(orientation["open_contradictions"]))
        tbl.add_row("unindexed entities", str(orientation["unindexed_entities"]))
        console.print(Panel(tbl, title="[bold cyan]Pre-dream inventory[/bold cyan]", expand=False))
    except Exception as exc:
        receipt["errors"].append(f"orientation: {type(exc).__name__}: {exc}")


def _run_signal_gather_phase(
    cfg: Config,
    *,
    enabled: bool,
    receipt: dict[str, Any],
    progress: Any,
    overall: Any,
    step: Any,
) -> None:
    """Mine transcripts since the prior run and update progress/receipt."""
    if not enabled:
        progress.update(step, description="[0] signal gather [dim]skip[/dim]")
        return

    progress.update(step, description="[0] signal gather — mining transcripts...")
    try:
        ts_file = _state_path(cfg) / ".last_run_ts"
        try:
            last_ts = float(ts_file.read_text().strip())
            # Exact fractional-day lookback from the last run, plus a ~30-min
            # overlap so a transcript written around its finish is not missed.
            since_days = max(0.001, (time.time() - last_ts) / 86400 + 0.02)
        except Exception:
            since_days = 7
        gathered = _run_signal_gather(since_days=since_days, file_limit=20)
        receipt["signal_gathered"] = gathered
        if gathered.get("error"):
            receipt["errors"].append(f"signal_gather: {gathered['error']}")
        progress.update(
            step,
            description=(
                f"[0] signal gather [green]✓[/green]  "
                f"{gathered['files_processed']} files, {gathered['memories_saved']} saved"
            ),
        )
    except Exception as exc:
        receipt["errors"].append(f"signal_gather: {type(exc).__name__}: {exc}")
        progress.update(step, description="[0] signal gather [yellow]warn[/yellow]")
    progress.advance(overall)


def _apply_convergence_guard(
    mem: Any,
    *,
    force: bool,
    dry_run: bool,
    previous_fingerprint: str | None,
    skip_maintain: bool,
    skip_presynthesis: bool,
    presynthesis_n: int,
    receipt: dict[str, Any],
    progress: Any,
    overall: Any,
    active_steps: int,
) -> tuple[bool, int]:
    """Apply the unchanged-corpus guard and reconcile the progress total."""
    current_fingerprint = _corpus_fingerprint(mem)
    saved = int(receipt["signal_gathered"].get("memories_saved", 0) or 0)
    converged = check_convergence(force, dry_run, previous_fingerprint, current_fingerprint, saved)
    if not converged:
        return False, active_steps

    receipt["converged"] = True
    console.print(
        "[dim]converged — corpus unchanged since last run; "
        "skipping contradict / synthesize / consolidate.[/dim]"
    )
    convergence_skip = (4 if not skip_maintain else 0) + (
        1 if not skip_presynthesis and presynthesis_n > 0 else 0
    )
    active_steps -= convergence_skip
    progress.update(overall, total=active_steps)
    return True, active_steps


def _run_mandate_sync_phase(
    cfg: Config,
    mem: Any,
    receipt: dict[str, Any],
    progress: Any,
    step: Any,
) -> None:
    """Refresh opted-in mandate blocks outside the main pipeline entrypoint."""
    from memo.flags import flag_bool

    if not flag_bool("MEMO_DYNAMIC_MANDATE_SYNC_ENABLED"):
        return

    progress.update(step, description="[mandate-sync] refreshing rule blocks...")
    from memo.constitution import run_mandate_sync_pass

    mandate_sync = run_mandate_sync_pass(cfg, mem)
    receipt["mandate_sync"] = mandate_sync
    if mandate_sync.get("error"):
        receipt["errors"].append(f"mandate_sync: {mandate_sync['error']}")
    progress.update(
        step,
        description=(
            f"[mandate-sync] [green]✓[/green]  {len(mandate_sync.get('synced', []))} repo(s)"
        ),
    )


@click.group(name="dream")
def dream_cmd() -> None:
    """Autonomous nightly maintenance — synthesise, heal, decay."""


@dream_cmd.command(name="run")
@click.option("--dry-run", is_flag=True, help="Preview actions; change nothing.")
@click.option("--json", "as_json", is_flag=True, help="Emit receipt as JSON.")
@click.option("--skip-entities", is_flag=True, help="Skip entity backfill pass.")
@click.option("--skip-decay", is_flag=True, help="Skip ROI decay pass.")
@click.option(
    "--skip-maintain", is_flag=True, help="Skip the contradict/consolidate/stale/synthesize passes."
)
@click.option("--skip-orientation", is_flag=True, help="Skip the pre-mutation inventory panel.")
@click.option("--skip-signal-gather", is_flag=True, help="Skip transcript mining phase.")
@click.option("--skip-prune-floor", is_flag=True, help="Skip the quality-floor prune pass.")
@click.option("--skip-evict", is_flag=True, help="Skip the corpus eviction pass.")
@click.option("--skip-compress", is_flag=True, help="Skip the verbose-compression pass.")
@click.option("--skip-prewarm", is_flag=True, help="Skip the query cache pre-warm pass.")
@click.option(
    "--skip-presynthesis", is_flag=True, help="Skip the query-prediction pre-synthesis pass."
)
@click.option(
    "--force",
    is_flag=True,
    help="Run every pass even if the corpus is unchanged (disable the convergence guard).",
)
@click.option(
    "--resume",
    is_flag=True,
    help="Resume an interrupted run: skip phases already committed this cycle "
    "(no repeated LLM calls / mutations). Restored from the phase checkpoint.",
)
def dream_run(
    dry_run: bool,
    as_json: bool,
    skip_entities: bool,
    skip_decay: bool,
    skip_maintain: bool,
    skip_orientation: bool,
    skip_signal_gather: bool,
    skip_prune_floor: bool,
    skip_evict: bool,
    skip_compress: bool,
    skip_prewarm: bool,
    skip_presynthesis: bool,
    force: bool,
    resume: bool,
) -> None:
    """Run the full dream pipeline once.

    Example:
      memo dream run --dry-run
      memo dream run
    """
    cfg = Config.from_env()
    # Keep --json output pure: the human banner would otherwise land on stdout
    # ahead of the JSON receipt and break `memo dream run --json | jq`.
    if not as_json:
        tag = "[dim](dry-run)[/dim] " if dry_run else ""
        console.print(f"{tag}[bold cyan]memo dream[/bold cyan] — starting pipeline...")

    # Single-owner lock: a second `dream run` (manual, or the com.memo.dream
    # LaunchAgent firing while one is already in flight) would otherwise race on
    # the shared sidecar DBs and clobber last.json. Hold an flock for the run;
    # the OS releases it on process exit. (dry-run is read-only, no lock needed.)
    _lock_fh = None
    if not dry_run:
        try:
            _lock_fh = acquire_dream_lock(cfg)
        except OSError:
            console.print("[yellow]another dream run is already active — skipping.[/yellow]")
            return

    # Convergence guard: read the previous run's corpus fingerprint. If nothing
    # changed since then (and signal-gather adds nothing), the expensive LLM
    # passes (contradict/synthesize/consolidate) are skipped — a re-run becomes
    # near-instant instead of redoing the same work. `--force` overrides.
    _prev_fp = read_previous_fingerprint(cfg)

    from memo.dream_flags import CODE_DRIFT_FLAG  # import registers the flag spec
    from memo.flags import flag_bool, flag_float, flag_int

    _evict_max = flag_int("MEMO_DREAM_EVICT_MAX_COUNT") or 0
    _compress_threshold = flag_int("MEMO_DREAM_COMPRESS_THRESHOLD") or 0
    _prewarm_n = flag_int("MEMO_DREAM_PREWARM_QUERIES") or 0
    _presynthesis_n = flag_int("MEMO_DREAM_PRESYNTHESIS_QUERIES") or 0
    _outcome_on = flag_bool("MEMO_OUTCOME_RANKING_ENABLED")
    _projection_on = flag_bool("MEMO_GRAPH_PROJECTION_ENABLED")
    _code_drift_on = flag_bool(CODE_DRIFT_FLAG)

    receipt: dict[str, Any] = {
        "dry_run": dry_run,
        "orientation": {},
        "signal_gathered": {"files_processed": 0, "memories_saved": 0, "skipped_dup": 0},
        "superseded": [],
        "evolved": [],
        "competing": [],
        "flagged_for_review": [],
        "merged": [],
        "archived_stale": [],
        "synthesized": [],
        "entities_extracted": 0,
        "graph_projection": {"status": "disabled"},
        "code_drift": {"status": "disabled"},
        "roi_reconciled": 0,
        "dead_archived": [],
        "roi_decayed": 0,
        "confidence_penalized": 0,
        "pruned_floor": [],
        "evicted": [],
        "compressed": [],
        "prewarm": {},
        "presynthesis": [],
        "graduated": {},
        "floor_calibration": {},
        "errors": [],
    }

    total_steps = 15
    skipped = (
        (1 if skip_signal_gather or dry_run else 0)
        + (4 if skip_maintain else 0)
        + (1 if skip_entities or dry_run else 0)
        + (1 if not _projection_on else 0)
        + (1 if not _code_drift_on else 0)
        + (1 if not _outcome_on or dry_run else 0)
        + (1 if skip_decay or dry_run else 0)
        + (1 if skip_prune_floor or dry_run else 0)
        + (1 if skip_evict or _evict_max == 0 else 0)
        + (1 if skip_compress or _compress_threshold == 0 or dry_run else 0)
        + (1 if skip_prewarm or _prewarm_n == 0 else 0)
        + (1 if skip_presynthesis or _presynthesis_n == 0 else 0)
    )
    active_steps = total_steps - skipped

    # Mark every save inside the pipeline as derived/batch so the interactive
    # near-duplicate nag stays quiet — these near-dups are what the same run's
    # consolidate pass merges. (`skipped_dup` in the receipt is the real signal.)
    #
    # The whole pipeline body runs under one try/finally (via ExitStack, so the
    # body keeps its indentation): a hard crash — Memory construction, the
    # convergence guard, an un-guarded phase — must STILL fall through to the
    # receipt write below. Otherwise `state_dir/dream/last.json` keeps showing
    # the last good night and `memo dream status`/doctor report healthy while
    # the 03:00 pipeline is silently dead.
    mem: Any = None
    _checkpoint: DreamCheckpoint | None = None
    _pipeline_completed = False
    _pipeline_stack = ExitStack()
    try:
        _pipeline_stack.enter_context(derived_save_scope())
        progress = _pipeline_stack.enter_context(_make_progress())
        overall = progress.add_task("[bold cyan]pipeline[/bold cyan]", total=active_steps)
        step = progress.add_task("loading memory...", total=None)

        mem = _get_memory(cfg)
        progress.update(step, description="[green]memory loaded ✓[/green]")

        # Fase 1 — per-phase instrumentation + resumable checkpoint. The run
        # fingerprint is the PREVIOUS completed run's corpus fp: stable across a
        # crash+restart (a crashed run never rewrites last.json) yet naturally
        # invalidated once a full run stamps a new corpus_fp. dry-run stays
        # ephemeral — no checkpoint, no resume.
        _checkpoint = (
            None
            if dry_run
            else DreamCheckpoint(_state_path(cfg) / CHECKPOINT_NAME, _prev_fp or "cold")
        )
        rec = PhaseRecorder(receipt, mem=mem, checkpoint=_checkpoint, resume=resume and not dry_run)

        # Fase 7 — dream conflict-staging. Inside a run with
        # MEMO_DREAM_STAGING_ENABLED a write-conflict on a dream-minted memory
        # (e.g. synthesis) parks the candidate in staging instead of losing it;
        # resume re-applies any parked proposal whose blocking conflict a human
        # has since resolved (never resolves a conflict itself). Scope-gated so
        # interactive `memo synthesize` saves are unaffected.
        if flag_bool("MEMO_DREAM_STAGING_ENABLED") and not dry_run:
            from memo import dream_staging

            _pipeline_stack.enter_context(dream_staging.dream_staging_scope())
            try:
                receipt["staging_resume"] = dream_staging.resume_staged_proposals(cfg, mem)
            except Exception as exc:
                receipt["errors"].append(f"staging_resume: {type(exc).__name__}: {exc}")

        # Fase 5 — per-pass incremental skip. When on, a content-derived pass
        # whose durable-content dependency is unchanged since its last successful
        # run is skipped (finer than the whole-pipeline convergence guard). Gated
        # default-off; a skip is recorded in the receipt and is self-healing.
        from memo import dream_incremental

        _incremental = flag_bool("MEMO_DREAM_INCREMENTAL_ENABLED") and not dry_run

        # Orientation — read-only inventory before mutations -----------------
        if not skip_orientation:
            rec.timed(
                "orientation",
                lambda: _run_orientation_phase(mem, receipt, progress, step),
                fragment_key="orientation",
            )

        # Fase 6 — derived-index health check. Gated default-off: when on, scans
        # for divergence (NULL FTS bodies, orphan vectors/chunks, wrong dims,
        # MD↔index drift, duplicate HyPE attempts) and repairs ONLY derived
        # orphans — never a canonical .md. Best-effort, runs before mutations.
        if flag_bool("MEMO_DREAM_INDEX_REPAIR_ENABLED") and not dry_run:
            try:
                from memo.store.index_health import check_index_health

                receipt["index_health"] = check_index_health(cfg, mem, repair=True)
                _ih_errs = receipt["index_health"].get("errors") or []
                if _ih_errs:
                    receipt["errors"].append(f"index_health: {len(_ih_errs)} sub-check error(s)")
            except Exception as exc:
                receipt["errors"].append(f"index_health: {type(exc).__name__}: {exc}")

        # Phase 0 — Signal gather: mine new transcripts since last dream run --
        rec.timed(
            "signal_gather",
            lambda: _run_signal_gather_phase(
                cfg,
                enabled=not skip_signal_gather and not dry_run,
                receipt=receipt,
                progress=progress,
                overall=overall,
                step=step,
            ),
            fragment_key="signal_gathered",
            resumable=True,
        )

        # Convergence guard — if signal-gather added nothing and the corpus
        # fingerprint matches the last run, the heavy LLM passes (contradict /
        # synthesize / consolidate) would redo identical work. Skip them; a
        # re-run is then near-instant. `--force` overrides.
        _converged, active_steps = _apply_convergence_guard(
            mem,
            force=force,
            dry_run=dry_run,
            previous_fingerprint=_prev_fp,
            skip_maintain=skip_maintain,
            skip_presynthesis=skip_presynthesis,
            presynthesis_n=_presynthesis_n,
            receipt=receipt,
            progress=progress,
            overall=overall,
            active_steps=active_steps,
        )

        # Phase 0.5 — noise-quantile min_sim floor calibration, gated + reversible.
        # Runs BEFORE the min_sim tuner so a co-enabled tuner line-searches
        # upward from the measured floor instead of past it.
        if flag_bool("MEMO_FLOOR_CALIBRATION"):
            progress.update(step, description="[floor] noise-quantile calibration...")
            try:
                res = _run_floor_calibration(mem, dry_run=dry_run)
                if res.get("error"):
                    receipt["errors"].append(f"floor_calibration: {res['error']}")
                receipt["floor_calibration"] = res.get("floor_calibration", {})
                progress.update(
                    step,
                    description=(
                        f"[floor] noise-quantile calibration [green]✓[/green]  "
                        f"applied={receipt['floor_calibration'].get('applied')}"
                    ),
                )
            except Exception as exc:
                receipt["errors"].append(f"floor_calibration: {type(exc).__name__}: {exc}")
                progress.update(
                    step, description="[floor] noise-quantile calibration [yellow]warn[/yellow]"
                )

        # Phase 1 — recall self-tuner (min_sim), gated + reversible ----------
        if flag_bool("MEMO_DREAM_TUNE_ENABLED"):
            progress.update(step, description="[tune] recall self-tuner...")
            try:
                from memo import dream_tune
                from memo.flags import flag_float

                receipt["tuner"] = dream_tune.run_tuning_pass(
                    cfg,
                    mem,
                    k=5 if (_k := flag_int("MEMO_DREAM_TUNE_K")) is None else _k,
                    max_evals=20 if (_me := flag_int("MEMO_DREAM_TUNE_MAX_EVALS")) is None else _me,
                    min_used_score=0.5
                    if (_mus := flag_float("MEMO_DREAM_MINE_MIN_USED_SCORE")) is None
                    else _mus,
                    dry_run=dry_run,
                )
                if receipt["tuner"].get("status") == "error":
                    # failures land in receipt["errors"], never silently swallowed
                    receipt["errors"].append(f"tuner: {receipt['tuner'].get('error')}")
                progress.update(
                    step,
                    description=(
                        f"[tune] recall self-tuner [green]✓[/green]  "
                        f"{receipt['tuner'].get('status')}"
                    ),
                )
            except Exception as exc:
                receipt["errors"].append(f"tuner: {type(exc).__name__}: {exc}")
                progress.update(step, description="[tune] recall self-tuner [yellow]warn[/yellow]")

        # Fase 4 — learning-metrics snapshot for the audit trail: the eleven
        # measures (precision/noise before-after cohort, answerability, grounding
        # coverage, later-usefulness, correction rate, synthesis acceptance,
        # created→used, p50/p95 latency) assembled from existing logs. Gated by
        # the ledger flag (it IS audit data), never re-runs the eval, best-effort.
        if flag_bool("MEMO_DREAM_LEDGER_ENABLED") and not dry_run:
            try:
                from memo import dream_metrics
                from memo.tuned_overlay import params_version

                _tuner = receipt.get("tuner") or {}
                receipt["learning_metrics"] = dream_metrics.learning_metrics(
                    cfg,
                    mem,
                    params_version=params_version(cfg.state_dir),
                    k=5 if (_lmk := flag_int("MEMO_DREAM_TUNE_K")) is None else _lmk,
                    before=_tuner.get("before"),
                    after=_tuner.get("after"),
                )
            except Exception as exc:
                receipt["errors"].append(f"learning_metrics: {type(exc).__name__}: {exc}")

        # Fase 8 — shadow mode: measure-only evidence for opt-in phases without
        # mutating production. Reverts any overlay a shadow-reclassified flag was
        # auto-graduated into and snapshots the per-flag review rollup. Gated +
        # best-effort; producers (record_recall_shadow / maybe_shadow) attach as
        # flags are classified kind="shadow".
        if flag_bool("MEMO_DREAM_SHADOW_ENABLED") and not dry_run:
            try:
                from memo import dream_shadow
                from memo.dream_flags import GATES

                receipt["shadow_review"] = {
                    "reclassify_reverted": dream_shadow.migrate_reclassified_overlay(
                        cfg.state_dir, GATES
                    ),
                    "review": dream_shadow.review_rows(cfg.state_dir, GATES),
                }
            except Exception as exc:
                receipt["errors"].append(f"shadow: {type(exc).__name__}: {exc}")

        # Curated graph-signal tuner (graph-off plus bounded alpha candidates).
        # Runs after the general pass; both merge the overlay so they coexist.
        if flag_bool("MEMO_DREAM_TUNE_ENABLED"):
            progress.update(step, description="[tune] curated graph-signal tuner...")
            try:
                from memo import dream_tune
                from memo.flags import flag_float

                receipt["graph_tuner"] = dream_tune.run_graph_weight_pass(
                    cfg,
                    mem,
                    k=5 if (_k := flag_int("MEMO_DREAM_TUNE_K")) is None else _k,
                    max_evals=20 if (_me := flag_int("MEMO_DREAM_TUNE_MAX_EVALS")) is None else _me,
                    min_used_score=0.5
                    if (_mus := flag_float("MEMO_DREAM_MINE_MIN_USED_SCORE")) is None
                    else _mus,
                    dry_run=dry_run,
                )
                if receipt["graph_tuner"].get("status") == "error":
                    # failures land in receipt["errors"], never silently swallowed
                    receipt["errors"].append(f"graph_tuner: {receipt['graph_tuner'].get('error')}")
                progress.update(
                    step,
                    description=(
                        f"[tune] curated graph-signal tuner [green]✓[/green]  "
                        f"{receipt['graph_tuner'].get('status')}"
                    ),
                )
            except Exception as exc:
                receipt["errors"].append(f"graph_tuner: {type(exc).__name__}: {exc}")
                progress.update(
                    step, description="[tune] curated graph-signal tuner [yellow]warn[/yellow]"
                )
        # Phase 1 — confidence calibration: refresh the predicted-vs-grounded map.
        if flag_bool("MEMO_RECALL_CONFIDENCE_GATE"):
            progress.update(step, description="[calibration] confidence map...")
            _run_calibration_pass(cfg, mem, dry_run=dry_run, receipt=receipt)
            _cal = receipt.get("calibration", {})
            _remapped = sum(1 for b, m in (_cal.get("map") or {}).items() if b != m)
            progress.update(
                step,
                description=f"[calibration] confidence map [green]✓[/green]  {_remapped} remapped",
            )

        # Phase 2c — HyDE A/B (separate opt-in; +1 MLX chat call per prompt,
        # prompt count capped inside the pass).
        if flag_bool("MEMO_DREAM_HYDE_TUNE_ENABLED"):
            progress.update(step, description="[tune] hyde A/B...")
            try:
                from memo import dream_tune
                from memo.flags import flag_float

                receipt["hyde_tuner"] = dream_tune.run_hyde_pass(
                    cfg,
                    mem,
                    k=5 if (_k := flag_int("MEMO_DREAM_TUNE_K")) is None else _k,
                    min_used_score=0.5
                    if (_mus := flag_float("MEMO_DREAM_MINE_MIN_USED_SCORE")) is None
                    else _mus,
                    dry_run=dry_run,
                )
                progress.update(
                    step,
                    description=(
                        f"[tune] hyde A/B [green]✓[/green]  {receipt['hyde_tuner'].get('status')}"
                    ),
                )
            except Exception as exc:
                receipt["errors"].append(f"hyde_tuner: {type(exc).__name__}: {exc}")
                progress.update(step, description="[tune] hyde A/B [yellow]warn[/yellow]")

        # Dark-feature (flag) graduation: A/B-measure default-off *_ENABLED
        # flags with a recall gate, flip winners ON via the overlay, sweep
        # deadline cull candidates (separate opt-in).
        if flag_bool("MEMO_DREAM_FLAG_GRADUATION_ENABLED"):
            progress.update(step, description="[graduate] dark flags A/B...")
            try:
                from memo import dream_flags
                from memo.flags import flag_float

                receipt["flag_graduation"] = dream_flags.run_flag_graduation_pass(
                    cfg,
                    mem,
                    k=5 if (_k := flag_int("MEMO_DREAM_TUNE_K")) is None else _k,
                    min_used_score=0.5
                    if (_mus := flag_float("MEMO_DREAM_MINE_MIN_USED_SCORE")) is None
                    else _mus,
                    dry_run=dry_run,
                )
                _fg = receipt["flag_graduation"]
                if _fg.get("status") == "error":
                    receipt["errors"].append(f"flag_graduation: {_fg.get('error')}")
                progress.update(
                    step,
                    description=(f"[graduate] dark flags [green]✓[/green]  {_fg.get('status')}"),
                )
            except Exception as exc:
                receipt["errors"].append(f"flag_graduation: {type(exc).__name__}: {exc}")
                progress.update(step, description="[graduate] dark flags [yellow]warn[/yellow]")

        # Online-only project-boost explorer (separate opt-in; no offline gate).
        if flag_bool("MEMO_DREAM_TUNE_BOOST_ENABLED"):
            try:
                from memo import dream_tune
                from memo.flags import flag_float

                receipt["boost_tuner"] = dream_tune.run_boost_pass(
                    cfg,
                    mem,
                    step=0.05
                    if (_step := flag_float("MEMO_DREAM_TUNE_BOOST_STEP")) is None
                    else _step,
                    dry_run=dry_run,
                )
                if receipt["boost_tuner"].get("status") == "error":
                    # failures land in receipt["errors"], never silently swallowed
                    receipt["errors"].append(f"boost_tuner: {receipt['boost_tuner'].get('error')}")
            except Exception as exc:
                receipt["errors"].append(f"boost_tuner: {type(exc).__name__}: {exc}")

        # Phase 3 — anticipatory: surface unmet gaps + prewarm (no fabrication)
        if flag_bool("MEMO_DREAM_ANTICIPATE_ENABLED"):
            progress.update(step, description="[anticipate] surfacing gaps...")
            try:
                from memo import dream_anticipate

                receipt["anticipated"] = dream_anticipate.anticipate(
                    cfg,
                    mem,
                    top_gaps=5
                    if (_tg := flag_int("MEMO_DREAM_ANTICIPATE_TOP_GAPS")) is None
                    else _tg,
                )
                if receipt["anticipated"].get("error"):
                    # failures land in receipt["errors"], never silently swallowed
                    receipt["errors"].append(f"anticipate: {receipt['anticipated'].get('error')}")
                progress.update(
                    step,
                    description=(
                        f"[anticipate] [green]✓[/green]  "
                        f"{len(receipt['anticipated'].get('gaps', []))} gaps"
                    ),
                )
            except Exception as exc:
                receipt["errors"].append(f"anticipate: {type(exc).__name__}: {exc}")
                progress.update(step, description="[anticipate] [yellow]warn[/yellow]")

        # Refresh dynamic mandate rule blocks in opted-in repos (self-syncing
        # constitution): superseded rules retire, new ones appear, on their own.
        _run_mandate_sync_phase(cfg, mem, receipt, progress, step)
        # Phase 2 — episodic→semantic: cross-session consolidation -----------
        if flag_bool("MEMO_DREAM_CONSOLIDATE_EPISODES_ENABLED"):
            progress.update(step, description="[consolidate] cross-session...")
            try:
                from memo import dream_consolidate

                receipt["consolidated_episodes"] = dream_consolidate.run_consolidate_episodes(
                    cfg,
                    mem,
                    min_sessions=2
                    if (_ms := flag_int("MEMO_DREAM_CONSOLIDATE_MIN_SESSIONS")) is None
                    else _ms,
                    dry_run=dry_run,
                )
                _ce = receipt["consolidated_episodes"]
                if _ce.get("status") == "error":
                    # failures land in receipt["errors"], never silently swallowed
                    receipt["errors"].append(f"consolidate_episodes: {_ce.get('error')}")
                progress.update(
                    step,
                    description=(
                        f"[consolidate] [green]✓[/green]  "
                        f"{_ce.get('status')} ({len(_ce.get('consolidated', []))})"
                    ),
                )
            except Exception as exc:
                receipt["errors"].append(f"consolidate_episodes: {type(exc).__name__}: {exc}")
                progress.update(step, description="[consolidate] [yellow]warn[/yellow]")

        # Negative-recall CAPTURE (avoid verdicts): graduate recalled memories
        # the user corrected/rejected next-turn into failure_pattern anti-memories.
        # Gated on MEMO_NEGATIVE_RECALL_CAPTURE_ENABLED (the pass itself no-ops
        # when off); the supersede-derived capture rides the contradict pass.
        if flag_bool("MEMO_NEGATIVE_RECALL_CAPTURE_ENABLED"):
            progress.update(step, description="[avoid] graduating avoid verdicts...")
            try:
                receipt["negative_capture"] = _run_negative_capture(cfg, mem, dry_run=dry_run)
                _nc = receipt["negative_capture"]
                if _nc.get("status") == "error":
                    receipt["errors"].append(f"negative_capture: {_nc.get('error')}")
                for _nce in _nc.get("errors", []):
                    receipt["errors"].append(f"negative_capture: {_nce}")
                progress.update(
                    step,
                    description=(
                        f"[avoid] [green]✓[/green]  {len(_nc.get('captured', []))} captured"
                    ),
                )
            except Exception as exc:
                receipt["errors"].append(f"negative_capture: {type(exc).__name__}: {exc}")
                progress.update(step, description="[avoid] [yellow]warn[/yellow]")

        # Bi-temporal validity extraction: LLM off the recall hot path (default
        # OFF). For recent facts/decisions whose TEXT explicitly states a
        # validity window, set valid_at/invalid_at — never hallucinated.
        # `_run_validity_extract` is internally guarded: per-record + whole-pass
        # failures land in receipt["errors"], never raise. Real runs only.
        if flag_bool("MEMO_DREAM_VALIDITY_EXTRACT_ENABLED") and not dry_run:
            progress.update(step, description="[validity] extracting validity windows...")
            try:
                _run_validity_extract(
                    mem,
                    receipt,
                    limit=50
                    if (_vl := flag_int("MEMO_DREAM_VALIDITY_EXTRACT_LIMIT")) is None
                    else _vl,
                    dry_run=dry_run,
                )
                _ve = receipt.get("validity_extract", {})
                progress.update(
                    step,
                    description=(
                        f"[validity] [green]✓[/green]  "
                        f"{len(_ve.get('updated', []))} windows / {_ve.get('scanned', 0)} scanned"
                    ),
                )
            except Exception as exc:
                receipt["errors"].append(f"validity_extract: {type(exc).__name__}: {exc}")
                progress.update(step, description="[validity] [yellow]warn[/yellow]")

        # Tier-1 #1 — profile distillation: rewrite-in-place profile.md files -
        if flag_bool("MEMO_DREAM_PROFILE_ENABLED"):
            progress.update(step, description="[profile] distilling profile.md...")
            try:
                from memo import dream_profile
                from memo.flags import flag_float as _pf_float

                receipt["profile"] = dream_profile.run_profile_pass(
                    cfg,
                    mem,
                    char_budget=4000
                    if (_cb := flag_int("MEMO_DREAM_PROFILE_CHAR_BUDGET")) is None
                    else _cb,
                    max_projects=5
                    if (_mp := flag_int("MEMO_DREAM_PROFILE_MAX_PROJECTS")) is None
                    else _mp,
                    directive_k=3
                    if (_dk := flag_int("MEMO_DREAM_PROFILE_DIRECTIVE_K")) is None
                    else _dk,
                    directive_min_used=(_pf_float("MEMO_DREAM_PROFILE_DIRECTIVE_MIN_USED") or 0.5),
                    dry_run=dry_run,
                )
                if receipt["profile"].get("status") == "error":
                    # run_profile_pass never raises — surface its error here so
                    # failures land in receipt["errors"], never silently swallowed
                    receipt["errors"].append(f"profile: {receipt['profile'].get('error')}")
                _pr = receipt["profile"]
                progress.update(
                    step,
                    description=(
                        f"[profile] [green]✓[/green]  "
                        f"{_pr.get('status')} ({len(_pr.get('written', []))} files)"
                    ),
                )
            except Exception as exc:
                receipt["errors"].append(f"profile: {type(exc).__name__}: {exc}")
                progress.update(step, description="[profile] [yellow]warn[/yellow]")

        # Chronicle — nightly engineering diary ------------------------------
        if flag_bool("MEMO_DREAM_CHRONICLE_ENABLED"):
            progress.update(step, description="[chronicle] writing diary...")
            try:
                from memo import dream_chronicle

                receipt["chronicle"] = dream_chronicle.run_chronicle_pass(
                    cfg,
                    mem,
                    weekly=flag_bool("MEMO_CHRONICLE_WEEKLY"),
                    dry_run=dry_run,
                )
                if receipt["chronicle"].get("status") == "error":
                    receipt["errors"].append(f"chronicle: {receipt['chronicle'].get('error')}")
                _ch = receipt["chronicle"]
                progress.update(
                    step,
                    description=f"[chronicle] [green]✓[/green]  {_ch.get('status')}",
                )
            except Exception as exc:
                receipt["errors"].append(f"chronicle: {type(exc).__name__}: {exc}")
                progress.update(step, description="[chronicle] [yellow]warn[/yellow]")

        # HyPE — nightly hypothetical-question generation (builds the index dark;
        # the read-path fold is gated separately by MEMO_HYPE_ENABLED) ---------
        if flag_bool("MEMO_DREAM_VECTOR_HYGIENE_ENABLED"):
            progress.update(step, description="[vector-hygiene] compacting derived indexes...")
            try:
                from memo import dream_vector

                receipt["vector_hygiene"] = dream_vector.run_vector_hygiene(
                    cfg, mem, dry_run=dry_run
                )
                if receipt["vector_hygiene"].get("status") == "error":
                    receipt["errors"].append(
                        f"vector_hygiene: {receipt['vector_hygiene'].get('error')}"
                    )
                progress.update(
                    step,
                    description=(
                        "[vector-hygiene] [green]✓[/green]  "
                        f"{receipt['vector_hygiene'].get('cache_pruned', 0)} cache rows"
                    ),
                )
            except Exception as exc:
                receipt["errors"].append(f"vector_hygiene: {type(exc).__name__}: {exc}")
                progress.update(step, description="[vector-hygiene] [yellow]warn[/yellow]")

        if flag_bool("MEMO_DREAM_VECTOR_VIEWS_ENABLED"):
            progress.update(step, description="[vector-views] indexing title/tag views...")
            try:
                from memo import dream_vector_views

                receipt["vector_views"] = dream_vector_views.run_title_view_pass(
                    cfg,
                    mem,
                    night_cap=1000
                    if (_vc := flag_int("MEMO_DREAM_VECTOR_VIEWS_NIGHT_CAP")) is None
                    else _vc,
                    dry_run=dry_run,
                )
                if receipt["vector_views"].get("status") == "error":
                    receipt["errors"].append(
                        f"vector_views: {receipt['vector_views'].get('error')}"
                    )
                progress.update(
                    step,
                    description=(
                        "[vector-views] [green]✓[/green]  "
                        f"{receipt['vector_views'].get('indexed', 0)} indexed"
                    ),
                )
            except Exception as exc:
                receipt["errors"].append(f"vector_views: {type(exc).__name__}: {exc}")
                progress.update(step, description="[vector-views] [yellow]warn[/yellow]")

        if flag_bool("MEMO_DREAM_HYPE_ENABLED"):
            progress.update(step, description="[hype] generating questions...")
            try:
                from memo import dream_hype

                receipt["hype"] = rec.timed(
                    "hype",
                    lambda: dream_hype.run_hype_pass(
                        cfg,
                        mem,
                        questions_per_memory=3
                        if (_qpm := flag_int("MEMO_HYPE_QUESTIONS_PER_MEMORY")) is None
                        else _qpm,
                        night_cap=400 if (_nc := flag_int("MEMO_HYPE_NIGHT_CAP")) is None else _nc,
                        budget_s=flag_float("MEMO_DREAM_HYPE_BUDGET_S") or None,
                        dry_run=dry_run,
                    ),
                    fragment_key="hype",
                    resumable=True,
                )
                if receipt["hype"].get("status") == "error":
                    receipt["errors"].append(f"hype: {receipt['hype'].get('error')}")
                _hy = receipt["hype"]
                progress.update(
                    step,
                    description=f"[hype] [green]✓[/green]  {_hy.get('status')}",
                )
            except Exception as exc:
                receipt["errors"].append(f"hype: {type(exc).__name__}: {exc}")
                progress.update(step, description="[hype] [yellow]warn[/yellow]")

        # Proactive — nightly candidate refresh (Task 11): reliability +
        # continuity detectors repopulate proactive.db. Kind multipliers are
        # derived on read (no write needed here). `_run_proactive_refresh` is
        # internally guarded — failures land in receipt["errors"], never raise.
        if flag_bool("MEMO_PROACTIVE_ENABLED"):
            progress.update(step, description="[proactive] refreshing candidates...")
            _run_proactive_refresh(mem, cfg.state_dir / "proactive.db", receipt, now=_iso_now())
            _proactive_n = receipt.get("proactive", {}).get("candidates", 0)
            progress.update(
                step,
                description=f"[proactive] [green]✓[/green]  {_proactive_n} candidates",
            )

        _run_verbatim_pass(
            cfg,
            dry_run=dry_run,
            receipt=receipt,
            progress=progress,
            step=step,
        )

        if flag_bool("MEMO_DREAM_GRADUATION_ENABLED"):
            progress.update(step, description="[graduate] quarantined captures...")
            try:
                from memo import dream_graduate

                receipt["graduated"] = dream_graduate.run_graduation(
                    cfg,
                    mem,
                    min_support=2
                    if (_ms := flag_int("MEMO_DREAM_GRADUATION_MIN_SUPPORT")) is None
                    else _ms,
                    dry_run=dry_run,
                )
                _gr = receipt["graduated"]
                progress.update(
                    step,
                    description=f"[graduate] [green]✓[/green]  {len(_gr.get('promoted', []))} promoted",
                )
            except Exception as exc:
                receipt["errors"].append(f"graduate: {type(exc).__name__}: {exc}")
                progress.update(step, description="[graduate] [yellow]warn[/yellow]")

        # Scope — project→global promotion: retag memories proven general ----
        if flag_bool("MEMO_DREAM_RETAG_GLOBAL_ENABLED"):
            progress.update(step, description="[retag] project→global...")
            try:
                from memo import dream_retag

                receipt["retagged_global"] = dream_retag.run_retag_global(
                    cfg,
                    mem,
                    min_other_projects=2
                    if (_mop := flag_int("MEMO_DREAM_RETAG_MIN_PROJECTS")) is None
                    else _mop,
                    dry_run=dry_run,
                )
                _rg = receipt["retagged_global"]
                progress.update(
                    step,
                    description=(
                        f"[retag] [green]✓[/green]  "
                        f"{len(_rg.get('retagged', []))} promoted to global"
                    ),
                )
            except Exception as exc:
                receipt["errors"].append(f"retag_global: {type(exc).__name__}: {exc}")
                progress.update(step, description="[retag] [yellow]warn[/yellow]")

        # Phase 2 — graph→semantic: community synthesis (spec 3) -------------
        if flag_bool("MEMO_DREAM_COMMUNITIES_ENABLED"):
            progress.update(step, description="[communities] graph clusters...")
            try:
                from memo import dream_communities

                receipt["communities"] = dream_communities.run_synthesize_communities(
                    cfg,
                    mem,
                    min_size=4
                    if (_msz := flag_int("MEMO_DREAM_COMMUNITIES_MIN_SIZE")) is None
                    else _msz,
                    dry_run=dry_run,
                )
                _cm = receipt["communities"]
                progress.update(
                    step,
                    description=(
                        f"[communities] [green]✓[/green]  "
                        f"{_cm.get('status')} ({len(_cm.get('synthesized', []))})"
                    ),
                )
            except Exception as exc:
                receipt["errors"].append(f"communities: {type(exc).__name__}: {exc}")
                progress.update(step, description="[communities] [yellow]warn[/yellow]")

        # Phase 2 — distillation: upward re-abstraction of MATURE clusters -----
        if flag_bool("MEMO_DREAM_DISTILL_ENABLED"):
            progress.update(step, description="[distill] mature clusters...")
            try:
                from memo import dream_distill
                from memo.flags import flag_float as _distill_float

                receipt["distilled"] = dream_distill.run_distill(
                    cfg,
                    mem,
                    min_cluster=cast(int, flag_int("MEMO_DREAM_DISTILL_MIN_CLUSTER")),
                    min_support=cast(int, flag_int("MEMO_DREAM_DISTILL_MIN_SUPPORT")),
                    min_age_days=cast(int, flag_int("MEMO_DREAM_DISTILL_MIN_AGE_DAYS")),
                    max_clusters=cast(int, flag_int("MEMO_DREAM_DISTILL_MAX")),
                    threshold=cast(float, _distill_float("MEMO_DREAM_DISTILL_THRESHOLD")),
                    min_confidence=cast(float, _distill_float("MEMO_DREAM_DISTILL_MIN_CONFIDENCE")),
                    dry_run=dry_run,
                )
                _di = receipt["distilled"]
                _saved = sum(
                    1
                    for d in _di.get("distilled", [])
                    if d.get("status") in ("saved", "would_save")
                )
                progress.update(
                    step,
                    description=f"[distill] [green]✓[/green]  {_di.get('status')} ({_saved})",
                )
            except Exception as exc:
                receipt["errors"].append(f"distill: {type(exc).__name__}: {exc}")
                progress.update(step, description="[distill] [yellow]warn[/yellow]")

        # Phase 3 — graph→semantic: bridge / multi-hop link synthesis (spec 3) -
        if flag_bool("MEMO_DREAM_BRIDGES_ENABLED"):
            progress.update(step, description="[bridges] articulation links...")
            try:
                from memo import dream_bridges

                receipt["bridges"] = dream_bridges.run_synthesize_bridges(
                    cfg,
                    mem,
                    dry_run=dry_run,
                )
                _br = receipt["bridges"]
                progress.update(
                    step,
                    description=(
                        f"[bridges] [green]✓[/green]  "
                        f"{_br.get('status')} ({len(_br.get('synthesized', []))})"
                    ),
                )
            except Exception as exc:
                receipt["errors"].append(f"bridges: {type(exc).__name__}: {exc}")
                progress.update(step, description="[bridges] [yellow]warn[/yellow]")

        # Phase 2c — graph hygiene: MinHash-blocked LLM entity canonicalization
        if flag_bool("MEMO_DREAM_ENTITY_CANON_ENABLED"):
            progress.update(step, description="[entity-canon] blocking pairs...")
            try:
                from memo import dream_entity_canon

                receipt["entity_canon"] = dream_entity_canon.run_entity_canon(
                    cfg,
                    mem,
                    max_pairs=30
                    if (_mp := flag_int("MEMO_DREAM_ENTITY_CANON_MAX_PAIRS")) is None
                    else _mp,
                    dry_run=dry_run,
                )
                _ec = receipt["entity_canon"]
                progress.update(
                    step,
                    description=(
                        f"[entity-canon] [green]✓[/green]  "
                        f"{_ec.get('llm_calls')} LLM calls vs {_ec.get('pairs_naive')} naive"
                    ),
                )
            except Exception as exc:
                receipt["errors"].append(f"entity_canon: {type(exc).__name__}: {exc}")
                progress.update(step, description="[entity-canon] [yellow]warn[/yellow]")

        # Phase 2c — graph hygiene: grounded co-use edge verification ---------
        if flag_bool("MEMO_DREAM_EDGE_VERIFY_ENABLED"):
            progress.update(step, description="[edge-verify] grounded co-use...")
            try:
                from memo import dream_edge_verify

                receipt["edge_verify"] = dream_edge_verify.run_edge_verify(
                    cfg,
                    mem,
                    dry_run=dry_run,
                )
                _evf = receipt["edge_verify"]
                if _evf.get("status") == "error":
                    receipt["errors"].append(f"edge_verify: {_evf.get('error')}")
                progress.update(
                    step,
                    description=(
                        f"[edge-verify] [green]✓[/green]  "
                        f"{_evf.get('promoted', 0)} promoted, {_evf.get('decayed', 0)} decayed"
                    ),
                )
            except Exception as exc:
                receipt["errors"].append(f"edge_verify: {type(exc).__name__}: {exc}")
                progress.update(step, description="[edge-verify] [yellow]warn[/yellow]")

        # Phase 2d — reference tier: per-folder vault abstracts (K4)
        if flag_bool("MEMO_DREAM_FOLDER_ABSTRACTS_ENABLED"):
            progress.update(step, description="[folder-abstracts] grouping vault...")
            try:
                from memo import dream_folder_abstracts

                receipt["folder_abstracts"] = dream_folder_abstracts.run_folder_abstracts(
                    cfg,
                    mem,
                    min_members=5
                    if (_mm := flag_int("MEMO_DREAM_FOLDER_ABSTRACTS_MIN_MEMBERS")) is None
                    else _mm,
                    max_folders=5
                    if (_mf := flag_int("MEMO_DREAM_FOLDER_ABSTRACTS_MAX")) is None
                    else _mf,
                    dry_run=dry_run,
                )
                _fa = receipt["folder_abstracts"]
                progress.update(
                    step,
                    description=(
                        f"[folder-abstracts] [green]✓[/green]  "
                        f"{_fa.get('status')} ({len(_fa.get('abstracts', []))})"
                    ),
                )
            except Exception as exc:
                receipt["errors"].append(f"folder_abstracts: {type(exc).__name__}: {exc}")
                progress.update(step, description="[folder-abstracts] [yellow]warn[/yellow]")

        # 0. Forget TTLs (always — explicit user intent) ---------------------
        progress.update(step, description="[dim]TTLs — enforce forget...[/dim]")
        try:
            for _item in mem.lifecycle.enforce_forget_ttl(dry_run=dry_run):
                pass
        except Exception as exc:
            receipt["errors"].append(f"forget_ttl: {type(exc).__name__}: {exc}")

        if not skip_maintain and not _converged:
            # 1. Contradictions ----------------------------------------------
            progress.update(step, description="[1/6] contradictions — scanning corpus...")
            try:
                res = rec.timed(
                    "contradict",
                    lambda: _run_contradict(mem, dry_run=dry_run),
                    resumable=True,
                )
                if "error" in res:
                    receipt["errors"].append(f"contradict: {res['error']}")
                receipt["superseded"] = res.get("superseded", [])
                receipt["evolved"] = res.get("evolved", [])
                receipt["competing"] = res.get("competing", [])
                receipt["flagged_for_review"] = res.get("flagged_for_review", [])
                receipt["confidence_penalized"] = res.get("confidence_penalized", 0)
                # Negative-recall: anti-memories derived from supersede/reversal.
                if res.get("negative_captured"):
                    receipt["negative_captured"] = res["negative_captured"]
                for _nce in res.get("negative_capture_errors", []):
                    receipt["errors"].append(f"negative_capture: {_nce}")
                for _ae in res.get("errors", []):
                    receipt["errors"].append(f"contradict: {_ae}")
                progress.update(
                    step,
                    description=(
                        f"[1/6] contradictions [green]✓[/green]  "
                        f"{len(receipt['superseded'])} superseded, {len(receipt['evolved'])} evolved"
                    ),
                )
            except Exception as exc:
                progress.update(step, description="[1/6] contradictions [yellow]warn[/yellow]")
                receipt["errors"].append(f"contradict: {type(exc).__name__}: {exc}")
            progress.advance(overall)

            # 2. Duplicates --------------------------------------------------
            progress.update(
                step,
                description="[2/6] duplicates — consolidating clusters...",
                total=None,
                completed=0,
            )
            try:
                res = rec.timed(
                    "consolidate_dups",
                    lambda: _run_consolidate_dups(mem, dry_run=dry_run),
                    resumable=True,
                )
                if "error" in res:
                    receipt["errors"].append(f"consolidate: {res['error']}")
                receipt["merged"] = res.get("merged", [])
                progress.update(
                    step,
                    description=(
                        f"[2/6] duplicates [green]✓[/green]  {len(receipt['merged'])} merged"
                    ),
                )
            except Exception as exc:
                progress.update(step, description="[2/6] duplicates [yellow]warn[/yellow]")
                receipt["errors"].append(f"consolidate: {type(exc).__name__}: {exc}")
            progress.advance(overall)

            # 3. Staleness ---------------------------------------------------
            progress.update(
                step, description="[3/6] stale memories — detecting...", total=None, completed=0
            )
            try:
                res = rec.timed(
                    "stale",
                    lambda: _run_stale(mem, dry_run=dry_run),
                    resumable=True,
                )
                if "error" in res:
                    receipt["errors"].append(f"stale: {res['error']}")
                for _ae in res.get("errors", []):
                    receipt["errors"].append(f"stale: {_ae}")
                receipt["archived_stale"] = res.get("archived", [])
                progress.update(
                    step,
                    description=(
                        f"[3/6] stale [green]✓[/green]  {len(receipt['archived_stale'])} archived"
                    ),
                )
            except Exception as exc:
                progress.update(step, description="[3/6] stale [yellow]warn[/yellow]")
                receipt["errors"].append(f"stale: {type(exc).__name__}: {exc}")
            progress.advance(overall)

            # 4. Emergent synthesis ------------------------------------------
            progress.update(
                step,
                description="[4/6] emergent synthesis — generating insights...",
                total=None,
                completed=0,
            )
            try:
                res = rec.timed(
                    "synthesis",
                    lambda: (
                        dream_incremental.run_or_skip(
                            cfg.state_dir,
                            "synthesis",
                            dream_incremental.durable_content_fingerprint(mem),
                            lambda: _run_synthesis(mem, dry_run=dry_run),
                        )
                        if _incremental
                        else _run_synthesis(mem, dry_run=dry_run)
                    ),
                    resumable=True,
                )
                if "error" in res:
                    receipt["errors"].append(f"synthesize: {res['error']}")
                receipt["synthesized"] = res.get("synthesized", [])
                saved_n = sum(1 for s in receipt["synthesized"] if s.get("saved"))
                progress.update(
                    step,
                    description=(
                        f"[4/6] synthesis [green]✓[/green]  "
                        f"{saved_n} saved, {len(receipt['synthesized'])} proposed"
                    ),
                )
            except Exception as exc:
                progress.update(step, description="[4/6] synthesis [yellow]warn[/yellow]")
                receipt["errors"].append(f"synthesize: {type(exc).__name__}: {exc}")
            progress.advance(overall)

        # 5. Entity backfill -------------------------------------------------
        if not skip_entities and not dry_run:
            progress.update(
                step,
                description="[5/6] entities — extracting from unindexed memories...",
                total=None,
                completed=0,
            )
            try:
                res = rec.timed(
                    "entities",
                    lambda: (
                        dream_incremental.run_or_skip(
                            cfg.state_dir,
                            "entities",
                            dream_incremental.durable_content_fingerprint(mem),
                            lambda: _run_entities(mem, dry_run=dry_run),
                        )
                        if _incremental
                        else _run_entities(mem, dry_run=dry_run)
                    ),
                    resumable=True,
                )
                if "error" in res:
                    receipt["errors"].append(f"entities: {res['error']}")
                receipt["entities_extracted"] = res.get("extracted", 0)
                progress.update(
                    step,
                    description=(
                        f"[5/6] entities [green]✓[/green]  {receipt['entities_extracted']} extracted"
                    ),
                )
            except Exception as exc:
                progress.update(step, description="[5/6] entities [yellow]warn[/yellow]")
                receipt["errors"].append(f"entities: {type(exc).__name__}: {exc}")
            progress.advance(overall)
        else:
            progress.update(step, description="[5/6] entities [dim]skip[/dim]")

        # 5b. Curated graph projection — after typed entity upgrades ----------
        if _projection_on:
            progress.update(step, description="[graph] refreshing curated projection...")
            # _run_graph_projection's own guard catches only a narrow tuple; an
            # unexpected error class (RuntimeError/AttributeError/…) from
            # rebuild_graph()/graph_health() would otherwise escape to the outer
            # pipeline handler and abort every remaining pass (roi/prune/evict/
            # compress/prewarm). Isolate it here like every sibling pass does.
            try:
                graph_projection = rec.timed(
                    "graph_projection",
                    lambda: _run_graph_projection(mem, dry_run=dry_run),
                    fragment_key="graph_projection",
                )
                receipt["graph_projection"] = graph_projection
                if graph_projection.get("status") == "error":
                    receipt["errors"].append(f"graph_projection: {graph_projection.get('error')}")
                progress.update(
                    step,
                    description=(
                        f"[graph] curated projection [green]✓[/green]  "
                        f"{graph_projection.get('status')}"
                    ),
                )
            except Exception as exc:
                progress.update(step, description="[graph] projection [yellow]warn[/yellow]")
                receipt["errors"].append(f"graph_projection: {type(exc).__name__}: {exc}")
            progress.advance(overall)

        # 5c. Code drift — re-verify code_refs against the codegraph index ----
        if _code_drift_on:
            progress.update(step, description="[code-drift] verifying code_refs...")
            # Same isolation rationale as graph projection above: an unexpected
            # error class must not escape and abort every remaining pass.
            try:
                code_drift = rec.timed(
                    "code_drift",
                    lambda: _run_code_drift(mem, dry_run=dry_run),
                    fragment_key="code_drift",
                )
                receipt["code_drift"] = code_drift
                if "error" in code_drift:
                    receipt["errors"].append(f"code_drift: {code_drift['error']}")
                receipt["errors"].extend(code_drift.get("errors", []))
                progress.update(
                    step,
                    description=(f"[code-drift] [green]✓[/green]  {code_drift.get('status')}"),
                )
            except Exception as exc:
                progress.update(step, description="[code-drift] [yellow]warn[/yellow]")
                receipt["errors"].append(f"code_drift: {type(exc).__name__}: {exc}")
            progress.advance(overall)

        # 6a. ROI reconcile (outcome loop) — MUST run before decay so the
        # scores that decay are the outcome-derived ones, not a flat 1.0.
        # This is what actually closes the grounding→utility→roi→ranking loop;
        # without it nothing schedules reconcile_roi and every roi_score sits
        # at 1.0 monotonically decaying. Gated by MEMO_OUTCOME_RANKING_ENABLED.
        if _outcome_on and not dry_run:
            progress.update(
                step,
                description="[6/7] ROI reconcile — deriving scores from grounding...",
                total=None,
                completed=0,
            )
            try:
                res = rec.timed(
                    "roi_reconcile",
                    lambda: _run_roi_reconcile(mem, dry_run=dry_run),
                    resumable=True,
                )
                if "error" in res:
                    receipt["errors"].append(f"roi_reconcile: {res['error']}")
                receipt["roi_reconciled"] = res.get("reconciled", 0)
                receipt["dead_archived"] = res.get("dead_archived", [])
                receipt["source_feedback_mined"] = res.get("source_feedback_mined", 0)
                progress.update(
                    step,
                    description=(
                        f"[6/7] ROI reconcile [green]✓[/green]  "
                        f"{receipt['roi_reconciled']} rescored, "
                        f"{len(receipt['dead_archived'])} archived"
                    ),
                )
            except Exception as exc:
                progress.update(step, description="[6/7] ROI reconcile [yellow]warn[/yellow]")
                receipt["errors"].append(f"roi_reconcile: {type(exc).__name__}: {exc}")
            progress.advance(overall)
        else:
            progress.update(step, description="[6/7] ROI reconcile [dim]skip[/dim]")

        # 6b. ROI decay ------------------------------------------------------
        if not skip_decay and not dry_run:
            progress.update(
                step, description="[7/7] ROI decay — adjusting scores...", total=None, completed=0
            )
            try:
                res = rec.timed(
                    "roi_decay",
                    lambda: _run_roi_decay(mem, dry_run=dry_run),
                )
                if "error" in res:
                    receipt["errors"].append(f"roi_decay: {res['error']}")
                receipt["roi_decayed"] = res.get("decayed", 0)
                progress.update(
                    step,
                    description=(
                        f"[7/7] ROI decay [green]✓[/green]  {receipt['roi_decayed']} rows"
                    ),
                )
            except Exception as exc:
                progress.update(step, description="[7/7] ROI decay [yellow]warn[/yellow]")
                receipt["errors"].append(f"roi_decay: {type(exc).__name__}: {exc}")
            progress.advance(overall)
        else:
            progress.update(step, description="[7/7] ROI decay [dim]skip[/dim]")

        # 7. Quality-floor prune ---------------------------------------------
        if not skip_prune_floor and not dry_run:
            progress.update(
                step,
                description="[7] prune floor — finding memories below the floor...",
                total=None,
                completed=0,
            )
            try:
                from memo.flags import flag_float, flag_int

                roi_floor = flag_float("MEMO_DREAM_PRUNE_FLOOR") or 0.15
                min_age = 90 if (_ma := flag_int("MEMO_DREAM_PRUNE_MIN_AGE_DAYS")) is None else _ma
                pruned = rec.timed(
                    "prune_floor",
                    lambda: _run_prune_floor(
                        mem,
                        roi_floor=roi_floor,
                        min_age_days=min_age,
                        dry_run=False,
                        errors=receipt["errors"],
                    ),
                )
                receipt["pruned_floor"] = pruned
                progress.update(
                    step,
                    description=f"[7] prune floor [green]✓[/green]  {len(pruned)} archived",
                )
            except Exception as exc:
                progress.update(step, description="[7] prune floor [yellow]warn[/yellow]")
                receipt["errors"].append(f"prune_floor: {type(exc).__name__}: {exc}")
            progress.advance(overall)
        else:
            progress.update(step, description="[7] prune floor [dim]skip[/dim]")

        # 8. Eviction --------------------------------------------------------
        if not skip_evict and _evict_max > 0:
            progress.update(
                step,
                description=f"[8] eviction — cap={_evict_max}, finding LFU excess...",
                total=None,
                completed=0,
            )
            try:
                evicted = rec.timed(
                    "eviction",
                    lambda: _run_eviction(
                        mem, max_count=_evict_max, dry_run=dry_run, errors=receipt["errors"]
                    ),
                )
                receipt["evicted"] = evicted
                progress.update(
                    step,
                    description=f"[8] eviction [green]✓[/green]  {len(evicted)} archived",
                )
            except Exception as exc:
                progress.update(step, description="[8] eviction [yellow]warn[/yellow]")
                receipt["errors"].append(f"eviction: {type(exc).__name__}: {exc}")
            progress.advance(overall)
        else:
            progress.update(step, description="[8] eviction [dim]skip[/dim]")

        # 9. Verbose compression --------------------------------------------
        if not skip_compress and _compress_threshold > 0 and not dry_run:
            progress.update(
                step,
                description=f"[9] compress — threshold={_compress_threshold} chars...",
                total=None,
                completed=0,
            )
            try:
                compressed = rec.timed(
                    "compress",
                    lambda: _run_compress(mem, threshold=_compress_threshold, dry_run=False),
                )
                receipt["compressed"] = compressed
                progress.update(
                    step,
                    description=f"[9] compress [green]✓[/green]  {len(compressed)} compressed",
                )
            except Exception as exc:
                progress.update(step, description="[9] compress [yellow]warn[/yellow]")
                receipt["errors"].append(f"compress: {type(exc).__name__}: {exc}")
            progress.advance(overall)
        else:
            progress.update(step, description="[9] compress [dim]skip[/dim]")

        # 10. Query cache pre-warm -------------------------------------------
        if not skip_prewarm and _prewarm_n > 0:
            progress.update(
                step,
                description=f"[10] prewarm — pre-embedding top {_prewarm_n} queries...",
                total=None,
                completed=0,
            )
            try:
                pw = rec.timed(
                    "prewarm",
                    lambda: _run_prewarm_queries(cfg, mem, n=_prewarm_n),
                    fragment_key="prewarm",
                )
                receipt["prewarm"] = pw
                progress.update(
                    step,
                    description=f"[10] prewarm [green]✓[/green]  {pw.get('queries_warmed', 0)} queries",
                )
            except Exception as exc:
                progress.update(step, description="[10] prewarm [yellow]warn[/yellow]")
                receipt["errors"].append(f"prewarm: {type(exc).__name__}: {exc}")
            progress.advance(overall)
        else:
            progress.update(step, description="[10] prewarm [dim]skip[/dim]")

        # 11. Query-prediction pre-synthesis ---------------------------------
        if not skip_presynthesis and _presynthesis_n > 0 and not _converged:
            progress.update(
                step,
                description=f"[11] pre-synthesis — top {_presynthesis_n} queries...",
                total=None,
                completed=0,
            )
            try:
                ps = rec.timed(
                    "presynthesis",
                    lambda: _run_presynthesis(cfg, mem, top_n=_presynthesis_n, dry_run=dry_run),
                    resumable=True,
                )
                receipt["presynthesis"] = ps
                progress.update(
                    step,
                    description=f"[11] pre-synthesis [green]✓[/green]  {len(ps)} clusters",
                )
            except Exception as exc:
                progress.update(step, description="[11] pre-synthesis [yellow]warn[/yellow]")
                receipt["errors"].append(f"presynthesis: {type(exc).__name__}: {exc}")
            progress.advance(overall)
        else:
            progress.update(step, description="[11] pre-synthesis [dim]skip[/dim]")

        # 12. Observability — label harvest + nightly retrieval eval ----------
        if flag_bool("MEMO_DREAM_EVAL_ENABLED") and not dry_run:
            progress.update(step, description="[12] harvest labels — mining grounding.log...")
            try:
                receipt["harvest_labels"] = rec.timed(
                    "harvest_labels",
                    lambda: _run_harvest_labels(cfg),
                    fragment_key="harvest_labels",
                )
                _hl = receipt["harvest_labels"]
                progress.update(
                    step,
                    description=(
                        f"[12] harvest labels [green]✓[/green]  "
                        f"+{_hl['new']} (total {_hl['total']})"
                    ),
                )
            except Exception as exc:
                receipt["errors"].append(f"harvest_labels: {type(exc).__name__}: {exc}")
                progress.update(step, description="[12] harvest labels [yellow]warn[/yellow]")

            progress.update(step, description="[13] eval recall — retrieval-only eval...")
            try:
                receipt["eval_recall"] = rec.timed(
                    "eval_recall",
                    lambda: _run_eval_recall(
                        cfg,
                        mem,
                        max_labels=200
                        if (_ml := flag_int("MEMO_DREAM_EVAL_MAX_LABELS")) is None
                        else _ml,
                    ),
                    fragment_key="eval_recall",
                )
                _ev = receipt["eval_recall"]
                # Headline the CURATED figure when it exists: that is the set
                # the regression discipline gates on. The blended number (which
                # mixes in ~150 auto-harvested labels) follows it.
                _headline = _ev.get("prec_at_k_curated", _ev["prec_at_k"])
                _headline_noise = _ev.get("noise_at_k_curated", _ev["noise_at_k"])
                _blend = (
                    f" · blended {_ev['prec_at_k']} over {_ev['labels_total']}"
                    if "prec_at_k_curated" in _ev
                    else f" ({_ev['labels_total']} labels)"
                )
                progress.update(
                    step,
                    description=(
                        f"[13] eval recall [green]✓[/green]  "
                        f"prec@{_ev['k']} {_headline} · noise@{_ev['k']} "
                        f"{_headline_noise} ({_ev.get('curated', 0)} curated){_blend}"
                    ),
                )
            except Exception as exc:
                receipt["errors"].append(f"eval_recall: {type(exc).__name__}: {exc}")
                progress.update(step, description="[13] eval recall [yellow]warn[/yellow]")

            progress.update(step, description="[14] capture weights — citation-type feedback...")
            try:
                receipt["capture_weights"] = rec.timed(
                    "capture_weights",
                    lambda: _run_capture_weights(cfg, mem),
                    fragment_key="capture_weights",
                )
                _cw = receipt["capture_weights"]
                _cw_top = f" · top {_cw['top']}" if _cw.get("top") else ""
                progress.update(
                    step,
                    description=(
                        f"[14] capture weights [green]✓[/green]  {_cw['types']} type(s){_cw_top}"
                    ),
                )
            except Exception as exc:
                receipt["errors"].append(f"capture_weights: {type(exc).__name__}: {exc}")
                progress.update(step, description="[14] capture weights [yellow]warn[/yellow]")
        else:
            progress.update(step, description="[12] harvest+eval [dim]skip[/dim]")

        # Mark step task complete so spinner stops
        progress.update(step, total=1, completed=1)
        _pipeline_completed = True
    except Exception as exc:
        # A crash before/around the per-pass guards must not vanish silently:
        # record it and fall through to the receipt write so the failed night
        # is visible to `memo dream status`/doctor. The phase checkpoint is
        # deliberately NOT cleared here so a `--resume` run can skip the phases
        # that already committed before the crash.
        receipt["errors"].append(f"pipeline: {type(exc).__name__}: {exc}")
        _log.exception("dream pipeline crashed before completion")
    finally:
        _pipeline_stack.close()

    # Fase 3 — auditable learning ledger: judge prior nights' reversible archives
    # (close-the-loop) then record tonight's mutations with provenance + rollback
    # handles. Gated default-off; best-effort so a ledger failure never affects
    # the receipt or the night. Runs after the finally so it captures whatever
    # mutations landed even if the pipeline crashed mid-way.
    if flag_bool("MEMO_DREAM_LEDGER_ENABLED"):
        try:
            from memo import dream_ledger

            _resolved = (
                dream_ledger.resolve_open_actions(
                    cfg.state_dir, lambda mid: mem.store.get(mid) is not None
                )
                if not dry_run
                else {}
            )
            _recorded = dream_ledger.record_from_receipt(cfg.state_dir, receipt, dry_run=dry_run)
            receipt["ledger"] = {
                "recorded": _recorded,
                "resolved": _resolved,
                "summary": dream_ledger.summarize(cfg.state_dir),
            }
        except Exception as exc:
            receipt["errors"].append(f"ledger: {type(exc).__name__}: {exc}")

    # Fase 1 — roll per-phase records into a compact summary for status/JSON.
    receipt["phases_summary"] = summarize_phases(receipt)

    # Persist receipt + timestamp --------------------------------------------
    if not dry_run:
        # A fully-completed run has nothing to resume: drop the checkpoint so the
        # next night starts clean. An interrupted run keeps it for `--resume`.
        if _pipeline_completed and _checkpoint is not None:
            _checkpoint.clear()
        try:
            d = _state_path(cfg)
            d.mkdir(parents=True, exist_ok=True)
            # Stamp the post-mutation corpus fingerprint so the next run can
            # detect "nothing changed" and converge (skip the heavy passes).
            # Guarded so a fingerprint failure (or a crashed `mem`) can't block
            # the receipt write itself.
            try:
                receipt["corpus_fp"] = _corpus_fingerprint(mem) if mem is not None else None
            except Exception as exc:
                receipt["errors"].append(f"corpus_fp: {type(exc).__name__}: {exc}")
            (d / "last.json").write_text(
                json.dumps({"ts": time.time(), **receipt}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            (d / ".last_run_ts").write_text(str(time.time()), encoding="utf-8")
        except Exception as exc:
            receipt["errors"].append(f"receipt: {type(exc).__name__}: {exc}")

    if as_json:
        click.echo(json.dumps(receipt, ensure_ascii=False, indent=2))
    else:
        _render_run_summary(receipt, dry_run)

    # Clean up lock (will auto-release on process exit, but explicit cleanup is cleaner)
    release_dream_lock(_lock_fh)


# Read-only dream reporting commands (status/ledger/timeline/…) live in
# cli_dream_report.py; register them on the dream group at import time so the
# CLI surface stays identical to when they were defined inline here.
def _register_dream_report_commands() -> None:
    try:
        from memo.cli_dream_report import register_dream_report_commands
    except Exception:  # pragma: no cover — import-time only, degrades to no reporting
        return
    register_dream_report_commands(dream_cmd)


_register_dream_report_commands()
