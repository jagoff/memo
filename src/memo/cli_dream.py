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
from typing import Any

import click
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from memo.cli_common import console
from memo.cli_common import get_memory as _get_memory
from memo.cli_dream_passes import (
    _build_orientation,
    _run_compress,
    _run_eviction,
    _run_presynthesis,
    _run_prewarm_queries,
    _run_prune_floor,
    _run_signal_gather,
)
from memo.config import Config

_log = _logging.getLogger(__name__)


def _state_path(cfg: Config):
    return cfg.state_dir / "dream"


def _older_id(mem: Any, id_a: str, id_b: str) -> tuple[str, str]:
    ra, rb = mem.get(id_a), mem.get(id_b)
    ua = getattr(ra, "updated", "") or ""
    ub = getattr(rb, "updated", "") or ""
    if ua and ub:
        return (id_a, id_b) if ua <= ub else (id_b, id_a)
    return id_a, id_b


@click.group(name="dream")
def dream_cmd() -> None:
    """Autonomous nightly maintenance — synthesise, heal, decay."""


def _make_progress() -> Progress:
    import sys

    from memo.flags import flag_bool

    # Non-interactive runs (launchd dream, piped output) still get the live-render
    # ANSI control stream from Rich — ~2MB of escapes per run. Disable the bar
    # there; the per-pass `console.print` summary at the end still emits.
    disable = flag_bool("MEMO_NONINTERACTIVE") or not sys.stderr.isatty()
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=24),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=False,
        disable=disable,
    )


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
) -> None:
    """Run the full dream pipeline once.

    Example:
      memo dream run --dry-run
      memo dream run
    """
    cfg = Config.from_env()
    tag = "[dim](dry-run)[/dim] " if dry_run else ""
    console.print(f"{tag}[bold cyan]memo dream[/bold cyan] — starting pipeline...")

    # Single-owner lock: a second `dream run` (manual, or the com.memo.dream
    # LaunchAgent firing while one is already in flight) would otherwise race on
    # the shared sidecar DBs and clobber last.json. Hold an flock for the run;
    # the OS releases it on process exit. (dry-run is read-only, no lock needed.)
    _lock_fh = None
    if not dry_run:
        import fcntl as _fcntl

        try:
            cfg.state_dir.mkdir(parents=True, exist_ok=True)
            _lock_fh = (cfg.state_dir / ".dream.lock").open("w")
            _flags = _fcntl.fcntl(_lock_fh.fileno(), _fcntl.F_GETFD)
            _fcntl.fcntl(_lock_fh.fileno(), _fcntl.F_SETFD, _flags | _fcntl.FD_CLOEXEC)
            _fcntl.flock(_lock_fh.fileno(), _fcntl.LOCK_EX | _fcntl.LOCK_NB)
        except OSError:
            if _lock_fh is not None:
                _lock_fh.close()
            console.print("[yellow]another dream run is already active — skipping.[/yellow]")
            return
    # Convergence guard: read the previous run's corpus fingerprint. If nothing
    # changed since then (and signal-gather adds nothing), the expensive LLM
    # passes (contradict/synthesize/consolidate) are skipped — a re-run becomes
    # near-instant instead of redoing the same work. `--force` overrides.
    _prev_fp: str | None = None
    try:
        import json as _json

        _prev_fp = _json.loads(
            (cfg.state_dir / "dream" / "last.json").read_text(encoding="utf-8")
        ).get("corpus_fp")
    except Exception:
        _prev_fp = None

    def _corpus_fingerprint() -> str | None:
        """A cheap change-signal: (row count, latest update timestamp) of the
        canonical `meta` table. Any save/edit/delete moves at least one."""
        try:
            row = mem.store._conn.execute(
                "SELECT COUNT(*), COALESCE(MAX(updated), '') FROM meta"
            ).fetchone()
            return f"{row[0]}:{row[1]}"
        except Exception:
            return None

    from memo.flags import flag_bool, flag_int

    _evict_max = flag_int("MEMO_DREAM_EVICT_MAX_COUNT") or 0
    _compress_threshold = flag_int("MEMO_DREAM_COMPRESS_THRESHOLD") or 0
    _prewarm_n = flag_int("MEMO_DREAM_PREWARM_QUERIES") or 0
    _presynthesis_n = flag_int("MEMO_DREAM_PRESYNTHESIS_QUERIES") or 0
    _outcome_on = flag_bool("MEMO_OUTCOME_RANKING_ENABLED")

    receipt: dict[str, Any] = {
        "dry_run": dry_run,
        "orientation": {},
        "signal_gathered": {"files_processed": 0, "memories_saved": 0, "skipped_dup": 0},
        "superseded": [],
        "evolved": [],
        "merged": [],
        "archived_stale": [],
        "synthesized": [],
        "entities_extracted": 0,
        "roi_reconciled": 0,
        "dead_archived": [],
        "roi_decayed": 0,
        "confidence_penalized": 0,
        "pruned_floor": [],
        "evicted": [],
        "compressed": [],
        "prewarm": {},
        "presynthesis": [],
        "errors": [],
    }

    total_steps = 13
    skipped = (
        (1 if skip_signal_gather or dry_run else 0)
        + (4 if skip_maintain else 0)
        + (1 if skip_entities or dry_run else 0)
        + (1 if not _outcome_on or dry_run else 0)
        + (1 if skip_decay or dry_run else 0)
        + (1 if skip_prune_floor or dry_run else 0)
        + (1 if skip_evict or _evict_max == 0 else 0)
        + (1 if skip_compress or _compress_threshold == 0 or dry_run else 0)
        + (1 if skip_prewarm or _prewarm_n == 0 else 0)
        + (1 if skip_presynthesis or _presynthesis_n == 0 else 0)
    )
    active_steps = total_steps - skipped

    with _make_progress() as progress:
        overall = progress.add_task("[bold cyan]pipeline[/bold cyan]", total=active_steps)
        step = progress.add_task("loading memory...", total=None)

        mem = _get_memory(cfg)
        progress.update(step, description="[green]memory loaded ✓[/green]")

        # Orientation — read-only inventory before mutations -----------------
        if not skip_orientation:
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
                for t, n in sorted(orientation["by_type"].items()):
                    tbl.add_row(f"  {t}", str(n))
                tbl.add_row("roi < 0.3", str(orientation["low_roi"]))
                tbl.add_row("stale candidates (>365d)", str(orientation["stale_candidates"]))
                tbl.add_row("open contradictions", str(orientation["open_contradictions"]))
                tbl.add_row("unindexed entities", str(orientation["unindexed_entities"]))
                console.print(
                    Panel(tbl, title="[bold cyan]Pre-dream inventory[/bold cyan]", expand=False)
                )
            except Exception as exc:
                receipt["errors"].append(f"orientation: {type(exc).__name__}: {exc}")

        # Phase 0 — Signal gather: mine new transcripts since last dream run --
        if not skip_signal_gather and not dry_run:
            progress.update(step, description="[0] signal gather — mining transcripts...")
            try:
                ts_file = _state_path(cfg) / ".last_run_ts"
                try:
                    last_ts = float(ts_file.read_text().strip())
                    # Exact fractional-day lookback from the last run (the miner
                    # multiplies by 86400), plus a ~30-min overlap so a transcript
                    # written around the last run's finish isn't missed. No more
                    # day-rounding / +1 inflation that re-mined ~1-2 days each run.
                    since_days = max(0.001, (time.time() - last_ts) / 86400 + 0.02)
                except Exception:
                    since_days = 7
                sg = _run_signal_gather(since_days=since_days, file_limit=20)
                receipt["signal_gathered"] = sg
                progress.update(
                    step,
                    description=(
                        f"[0] signal gather [green]✓[/green]  "
                        f"{sg['files_processed']} files, {sg['memories_saved']} saved"
                    ),
                )
            except Exception as exc:
                receipt["errors"].append(f"signal_gather: {type(exc).__name__}: {exc}")
                progress.update(step, description="[0] signal gather [yellow]warn[/yellow]")
            progress.advance(overall)
        else:
            progress.update(step, description="[0] signal gather [dim]skip[/dim]")

        # Convergence guard — if signal-gather added nothing and the corpus
        # fingerprint matches the last run, the heavy LLM passes (contradict /
        # synthesize / consolidate) would redo identical work. Skip them; a
        # re-run is then near-instant. `--force` overrides.
        _cur_fp = _corpus_fingerprint()
        _sg_saved = int(receipt["signal_gathered"].get("memories_saved", 0) or 0)
        _converged = (
            not force
            and not dry_run
            and _prev_fp is not None
            and _cur_fp is not None
            and _cur_fp == _prev_fp
            and _sg_saved == 0
        )
        if _converged:
            receipt["converged"] = True
            console.print(
                "[dim]converged — corpus unchanged since last run; "
                "skipping contradict / synthesize / consolidate.[/dim]"
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
                    k=flag_int("MEMO_DREAM_TUNE_K") or 5,
                    max_evals=flag_int("MEMO_DREAM_TUNE_MAX_EVALS") or 20,
                    min_used_score=flag_float("MEMO_DREAM_MINE_MIN_USED_SCORE") or 0.5,
                    dry_run=dry_run,
                )
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

        # Phase 3 — anticipatory: surface unmet gaps + prewarm (no fabrication)
        if flag_bool("MEMO_DREAM_ANTICIPATE_ENABLED"):
            progress.update(step, description="[anticipate] surfacing gaps...")
            try:
                from memo import dream_anticipate

                receipt["anticipated"] = dream_anticipate.anticipate(
                    cfg, mem, top_gaps=flag_int("MEMO_DREAM_ANTICIPATE_TOP_GAPS") or 5
                )
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

        # Phase 2 — episodic→semantic: cross-session consolidation -----------
        if flag_bool("MEMO_DREAM_CONSOLIDATE_EPISODES_ENABLED"):
            progress.update(step, description="[consolidate] cross-session...")
            try:
                from memo import dream_consolidate

                receipt["consolidated_episodes"] = dream_consolidate.run_consolidate_episodes(
                    cfg,
                    mem,
                    min_sessions=flag_int("MEMO_DREAM_CONSOLIDATE_MIN_SESSIONS") or 2,
                    dry_run=dry_run,
                )
                _ce = receipt["consolidated_episodes"]
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

        # Phase 2 — graph→semantic: community synthesis (spec 3) -------------
        if flag_bool("MEMO_DREAM_COMMUNITIES_ENABLED"):
            progress.update(step, description="[communities] graph clusters...")
            try:
                from memo import dream_communities

                receipt["communities"] = dream_communities.run_synthesize_communities(
                    cfg,
                    mem,
                    min_size=flag_int("MEMO_DREAM_COMMUNITIES_MIN_SIZE") or 4,
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

                def _contradict_progress(current: int, total: int, _title: str) -> None:
                    progress.update(
                        step,
                        description=f"[1/6] contradictions — {current}/{total}...",
                        total=total,
                        completed=current,
                    )

                mem.contradict_scanner.scan_corpus(
                    confidence_threshold=0.9,
                    max_pairs=50,
                    progress=_contradict_progress,
                    persist=not dry_run,
                )
                from memo.flags import flag_float as _flag_float

                _evo_conf = _flag_float("MEMO_EVOLUTION_CONFIDENCE")
                _evo_conf = 0.6 if _evo_conf is None else _evo_conf
                contradicted_ids: list[str] = []
                for pair in mem.contradict_store.list_open(min_confidence=0.9):
                    rel = (pair.relationship or "").lower()
                    if "evolu" in rel:
                        if not dry_run:
                            # Demote the superseded (older) side so the temporal
                            # verdict actually steers ranking: lower its
                            # confidence (health-score multiplier, default-on)
                            # instead of marking "both kept" and changing nothing.
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
                                note=f"dream: evolution, demoted older {older[:8]}",
                            )
                        receipt["evolved"].append(pair.pair_id)
                        continue
                    if "contrad" not in rel:
                        continue
                    older, _newer = _older_id(mem, pair.memory_id_a, pair.memory_id_b)
                    contradicted_ids.extend([pair.memory_id_a, pair.memory_id_b])
                    if not dry_run:
                        ok = mem.lifecycle.archive_memory(older)
                        if ok:
                            mem.contradict_store.resolve(
                                pair.pair_id, "kept_newer", note=f"dream: archived older {older}"
                            )
                    receipt["superseded"].append({"pair_id": pair.pair_id, "older": older})
                if contradicted_ids and not dry_run:
                    mem.store.penalize_confidence_batch(contradicted_ids)
                    receipt["confidence_penalized"] = len(set(contradicted_ids))
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
                res = mem.consolidator.consolidate_all(
                    threshold=0.9,
                    auto_apply=True,
                    dry_run=dry_run,
                )
                for r in res.get("results", []):
                    receipt["merged"].append(
                        {"merged_id": r.get("merged_id"), "archived_ids": r.get("archived_ids", [])}
                    )
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
                stale = mem.temporal.detect_stale_memories(days_threshold=365, min_access_count=0)
                for item in stale:
                    mid = item.get("id")
                    if not mid:
                        continue
                    if not dry_run:
                        mem.lifecycle.archive_memory(mid)
                    receipt["archived_stale"].append(
                        {"id": mid, "days": item.get("days_since_update")}
                    )
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
                results = mem.synthesize_cross_cluster(
                    dry_run=dry_run, min_cluster_size=5, max_clusters=8
                )
                for r in results:
                    receipt["synthesized"].append(
                        {
                            "title": r.get("title"),
                            "confidence": r.get("confidence"),
                            "saved": r.get("saved", False),
                        }
                    )
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
                counts = mem.extract_entities(all_=True, skip_already_indexed=True, max_batch=50)
                receipt["entities_extracted"] = counts.get("entities_extracted", 0)
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
                from memo.outcome import (
                    dead_weight,
                    reconcile_roi,
                    reconcile_source_feedback,
                )

                receipt["roi_reconciled"] = reconcile_roi(mem).get("updated", 0)
                # Per-query learning: mine implicit feedback from grounding so
                # ranking sharpens for the queries actually asked (opt-in).
                if flag_bool("MEMO_OUTCOME_SOURCE_FEEDBACK"):
                    try:
                        fb = reconcile_source_feedback(
                            mem,
                            include_negatives=flag_bool("MEMO_OUTCOME_SOURCE_FEEDBACK_NEG"),
                        )
                        receipt["source_feedback_mined"] = fb
                    except Exception as _exc:
                        receipt["errors"].append(f"source_feedback: {type(_exc).__name__}: {_exc}")
                min_surfaced = flag_int("MEMO_OUTCOME_DEAD_MIN_SURFACED") or 0
                for d in dead_weight(mem, min_surfaced=min_surfaced):
                    if (
                        mem.forget(
                            d["id"], reason=f"outcome: surfaced {d['surfaced']}x without grounding"
                        )
                        is not None
                    ):
                        receipt["dead_archived"].append(d["id"])
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
                n = mem.store.decay_roi(factor=0.98, older_than_days=30)
                receipt["roi_decayed"] = n
                progress.update(step, description=(f"[7/7] ROI decay [green]✓[/green]  {n} rows"))
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
                min_age = flag_int("MEMO_DREAM_PRUNE_MIN_AGE_DAYS") or 90
                pruned = _run_prune_floor(
                    mem, roi_floor=roi_floor, min_age_days=min_age, dry_run=False
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
                evicted = _run_eviction(mem, max_count=_evict_max, dry_run=dry_run)
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
                compressed = _run_compress(mem, threshold=_compress_threshold, dry_run=False)
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
                pw = _run_prewarm_queries(cfg, mem, n=_prewarm_n)
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
                ps = _run_presynthesis(cfg, mem, top_n=_presynthesis_n, dry_run=dry_run)
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

        # Mark step task complete so spinner stops
        progress.update(step, total=1, completed=1)

    # Persist receipt + timestamp --------------------------------------------
    if not dry_run:
        try:
            d = _state_path(cfg)
            d.mkdir(parents=True, exist_ok=True)
            # Stamp the post-mutation corpus fingerprint so the next run can
            # detect "nothing changed" and converge (skip the heavy passes).
            receipt["corpus_fp"] = _corpus_fingerprint()
            (d / "last.json").write_text(
                json.dumps({"ts": time.time(), **receipt}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            (d / ".last_run_ts").write_text(str(time.time()), encoding="utf-8")
        except Exception as exc:
            receipt["errors"].append(f"receipt: {type(exc).__name__}: {exc}")

    if as_json:
        click.echo(json.dumps(receipt, ensure_ascii=False, indent=2))
        return

    tag = "[dim](dry-run)[/dim] " if dry_run else ""
    console.print(f"{tag}[bold]memo dream[/bold]")
    console.print(
        f"  contradictions superseded: {len(receipt['superseded'])}, "
        f"evolutions: {len(receipt['evolved'])}, "
        f"confidence penalized: {receipt['confidence_penalized']}"
    )
    console.print(f"  duplicate clusters merged: {len(receipt['merged'])}")
    console.print(f"  stale memories archived:   {len(receipt['archived_stale'])}")
    if receipt["synthesized"]:
        saved = sum(1 for s in receipt["synthesized"] if s.get("saved"))
        console.print(
            f"  emergent syntheses:        {saved} saved, {len(receipt['synthesized'])} proposed"
        )
    console.print(f"  entities extracted:        {receipt['entities_extracted']}")
    console.print(
        f"  roi reconciled (grounding):{receipt['roi_reconciled']} rescored, "
        f"{len(receipt['dead_archived'])} dead-archived"
    )
    console.print(f"  roi rows decayed:          {receipt['roi_decayed']}")
    console.print(f"  quality-floor pruned:      {len(receipt['pruned_floor'])}")
    if receipt.get("evicted"):
        console.print(f"  evicted (LFU):             {len(receipt['evicted'])}")
    if receipt.get("compressed"):
        console.print(f"  compressed:                {len(receipt['compressed'])}")
    pw = receipt.get("prewarm", {})
    if pw.get("queries_warmed"):
        console.print(f"  cache pre-warmed:          {pw['queries_warmed']} queries")
    if receipt.get("presynthesis"):
        console.print(f"  pre-syntheses:             {len(receipt['presynthesis'])} clusters")
    sg = receipt.get("signal_gathered", {})
    if sg.get("files_processed") or sg.get("memories_saved"):
        console.print(
            f"  signal gather:             {sg['files_processed']} files, "
            f"{sg['memories_saved']} saved, {sg.get('skipped_dup', 0)} dup skipped"
        )
    if receipt["errors"]:
        for e in receipt["errors"]:
            console.print(f"  [yellow]warn:[/yellow] {e}")


@dream_cmd.command(name="status")
def dream_status() -> None:
    """Show when dream last ran and what it changed."""
    cfg = Config.from_env()
    last_json = _state_path(cfg) / "last.json"
    if not last_json.exists():
        console.print("[dim]dream has never run[/dim]")
        return
    try:
        data = json.loads(last_json.read_text(encoding="utf-8"))
    except Exception as exc:
        console.print(f"[yellow]could not read last receipt: {exc}[/yellow]")
        return
    ts = data.get("ts")
    import datetime

    when = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M") if ts else "?"
    console.print(f"[bold]last dream run:[/bold] {when}")
    console.print(f"  superseded: {len(data.get('superseded', []))}")
    console.print(f"  merged:     {len(data.get('merged', []))}")
    console.print(f"  stale:      {len(data.get('archived_stale', []))}")
    console.print(f"  syntheses:  {len(data.get('synthesized', []))}")
    console.print(f"  entities:   {data.get('entities_extracted', 0)}")
    console.print(f"  roi decay:  {data.get('roi_decayed', 0)} rows")
    if data.get("tuner"):
        t = data["tuner"]
        extra = (
            f" (min_sim {t.get('floor_before')}→{t.get('floor_after')})"
            if t.get("status") == "applied"
            else ""
        )
        console.print(f"  tuner:      {t.get('status')}{extra}")
    if data.get("anticipated"):
        from memo.dream_anticipate import briefing_line

        console.print(f"  {briefing_line(data['anticipated'])}")
    if data.get("consolidated_episodes"):
        ce = data["consolidated_episodes"]
        saved = sum(1 for d in ce.get("consolidated", []) if d.get("status") == "saved")
        console.print(f"  consolidate: {ce.get('status')} — {saved} cross-session memo(s)")
    if data.get("errors"):
        for e in data["errors"]:
            console.print(f"  [yellow]warn:[/yellow] {e}")


@dream_cmd.command(name="anticipate")
@click.option("--json", "as_json", is_flag=True, help="Emit the anticipated-needs fragment as JSON.")
def dream_anticipate_cmd(as_json: bool) -> None:
    """Anticipatory pass — surface recurring unmet gaps + hot queries (no fabrication)."""
    from memo import dream_anticipate
    from memo.flags import flag_int

    cfg = Config.from_env()
    mem = _get_memory(cfg)
    frag = dream_anticipate.anticipate(
        cfg, mem, top_gaps=flag_int("MEMO_DREAM_ANTICIPATE_TOP_GAPS") or 5
    )
    if as_json:
        click.echo(json.dumps(frag, indent=2, ensure_ascii=False))
        return
    console.print(f"[bold]{dream_anticipate.briefing_line(frag)}[/bold]")
    for g in frag.get("gaps", []):
        console.print(f"  gap (x{g['count']}): {g['prompt']}")
    if frag.get("prewarmed"):
        console.print(f"  prewarmed {frag['prewarmed']} queries")


@dream_cmd.command(name="consolidate-episodes")
@click.option("--dry-run", is_flag=True, help="Cluster + synthesize, save nothing.")
@click.option("--json", "as_json", is_flag=True, help="Emit the consolidation fragment as JSON.")
def dream_consolidate_cmd(dry_run: bool, as_json: bool) -> None:
    """Episodic→semantic — abstract recurring cross-session work into durable memos."""
    from memo import dream_consolidate
    from memo.flags import flag_int

    cfg = Config.from_env()
    mem = _get_memory(cfg)
    res = dream_consolidate.run_consolidate_episodes(
        cfg, mem, min_sessions=flag_int("MEMO_DREAM_CONSOLIDATE_MIN_SESSIONS") or 2, dry_run=dry_run
    )
    if as_json:
        click.echo(json.dumps(res, indent=2, ensure_ascii=False))
        return
    console.print(f"[bold]consolidate-episodes:[/bold] {res.get('status')}")
    for d in res.get("consolidated", []):
        console.print(f"  [{d['status']}] {d.get('project')}: {d.get('title', '')}")


@dream_cmd.command(name="communities")
@click.option("--dry-run", is_flag=True, help="Detect + preview communities, save nothing.")
@click.option("--json", "as_json", is_flag=True, help="Emit the synthesis fragment as JSON.")
def dream_communities_cmd(dry_run: bool, as_json: bool) -> None:
    """Graph→semantic — abstract each entity-graph community into a synthesis memo."""
    from memo import dream_communities
    from memo.flags import flag_int

    cfg = Config.from_env()
    mem = _get_memory(cfg)
    res = dream_communities.run_synthesize_communities(
        cfg,
        mem,
        min_size=flag_int("MEMO_DREAM_COMMUNITIES_MIN_SIZE") or 4,
        dry_run=dry_run,
    )
    if as_json:
        click.echo(json.dumps(res, indent=2, ensure_ascii=False))
        return
    console.print(f"[bold]communities:[/bold] {res.get('status')}")
    for d in res.get("synthesized", []):
        rep = d.get("representative") or ""
        console.print(f"  [{d['status']}] {rep}: {d.get('title', '')}")


@dream_cmd.command(name="tune")
@click.option("--dry-run", is_flag=True, help="Measure + search, write nothing.")
@click.option("--rollback", "do_rollback", is_flag=True, help="Restore the previous tuned params.")
@click.option("--status", "show_status", is_flag=True, help="Show the overlay + baseline.")
def dream_tune_cmd(dry_run: bool, do_rollback: bool, show_status: bool) -> None:
    """Self-improving recall tuner (MEMO_RECALL_MIN_SIM) — gated, reversible."""
    from memo import dream_tune, tuned_overlay
    from memo.flags import flag_float, flag_int

    cfg = Config.from_env()
    if show_status:
        click.echo(
            json.dumps(
                {
                    "overlay": tuned_overlay.read_overlay(cfg.state_dir),
                    "baseline": dream_tune.load_baseline(cfg.state_dir),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return
    if do_rollback:
        restored = tuned_overlay.rollback_overlay(cfg.state_dir)
        click.echo(json.dumps({"rolled_back": restored}, ensure_ascii=False))
        return
    mem = _get_memory(cfg)
    res = dream_tune.run_tuning_pass(
        cfg,
        mem,
        k=flag_int("MEMO_DREAM_TUNE_K") or 5,
        max_evals=flag_int("MEMO_DREAM_TUNE_MAX_EVALS") or 20,
        min_used_score=flag_float("MEMO_DREAM_MINE_MIN_USED_SCORE") or 0.5,
        dry_run=dry_run,
    )
    click.echo(json.dumps(res, indent=2, ensure_ascii=False))


@dream_cmd.command(name="if-due")
def dream_if_due() -> None:
    """Spawn a background dream run if > 24h since last run (for launchd)."""
    import os as _os
    import subprocess as _sp

    cfg = Config.from_env()
    # Own debounce file — do NOT reuse .last_run_ts: dream run reads that to
    # size its signal-gather lookback, and clobbering it here collapses the
    # lookback to ~0. This file only guards against double-spawn within 24h.
    ts_file = _state_path(cfg) / ".last_if_due_ts"
    try:
        last = float(ts_file.read_text().strip())
    except Exception:
        last = 0.0
    if (time.time() - last) < 24 * 3600:
        return
    try:
        _state_path(cfg).mkdir(parents=True, exist_ok=True)
        ts_file.write_text(str(time.time()), encoding="utf-8")
        _sp.Popen(
            ["memo", "dream", "run"],
            stdin=_sp.DEVNULL,
            stdout=_sp.DEVNULL,
            stderr=_sp.DEVNULL,
            start_new_session=True,
            env={**_os.environ, "MEMO_NONINTERACTIVE": "1"},
        )
    except Exception as exc:
        _log.warning("dream --if-due: failed to spawn background dream: %s", exc)
