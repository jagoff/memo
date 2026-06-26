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
from typing import TYPE_CHECKING, Any

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
from memo.config import Config
from memo.transcript_miner import mine_transcripts

if TYPE_CHECKING:
    from memo.memory.facade import Memory

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


def _build_orientation(mem: Memory) -> dict:
    """Read-only corpus inventory — runs before any mutation."""
    conn = mem.store._conn
    result: dict = {
        "total": 0,
        "by_type": {},
        "low_roi": 0,
        "stale_candidates": 0,
        "open_contradictions": 0,
        "unindexed_entities": 0,
    }
    try:
        row = conn.execute("SELECT COUNT(*) AS n FROM meta WHERE type != 'reference'").fetchone()
        result["total"] = int(row["n"]) if row else 0
    except Exception:  # noqa: S110
        pass

    try:
        rows = conn.execute(
            "SELECT type, COUNT(*) AS n FROM meta WHERE type != 'reference' GROUP BY type"
        ).fetchall()
        result["by_type"] = {r["type"]: int(r["n"]) for r in rows}
    except Exception:  # noqa: S110
        pass

    try:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM meta m "
            "LEFT JOIN memory_health h ON h.id = m.id "
            "WHERE COALESCE(h.roi_score, 1.0) < 0.3 AND m.type != 'reference'"
        ).fetchone()
        result["low_roi"] = int(row["n"]) if row else 0
    except Exception:  # noqa: S110
        pass

    try:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM meta m "
            "LEFT JOIN access a ON a.id = m.id "
            "WHERE m.updated < datetime('now', '-365 days') "
            "AND COALESCE(a.access_count, 0) = 0 "
            "AND m.type != 'reference'"
        ).fetchone()
        result["stale_candidates"] = int(row["n"]) if row else 0
    except Exception:  # noqa: S110
        pass

    try:
        pairs = mem.contradict_store.list_open()
        result["open_contradictions"] = len(pairs)
    except Exception:  # noqa: S110
        pass

    try:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM meta m "
            "WHERE m.type != 'reference' "
            "AND NOT EXISTS ("
            "  SELECT 1 FROM entity_memoria em "
            "  JOIN entities e ON e.id = em.entity_id "
            "  WHERE em.memoria_id = m.id"
            ")"
        ).fetchone()
        result["unindexed_entities"] = int(row["n"]) if row else 0
    except Exception:  # noqa: S110
        pass

    return result


def _run_signal_gather(since_days: int, file_limit: int = 20) -> dict:
    """Run transcript mining and return a compact summary.

    Never raises — exceptions are captured in the returned dict.
    """
    try:
        res = mine_transcripts(since_days=since_days, file_limit=file_limit)
        return {
            "files_processed": res.get("files_processed", 0),
            "memorias_saved": len(res.get("saved") or []),
            "skipped_dup": res.get("skipped_dup", 0),
        }
    except Exception as exc:
        return {"files_processed": 0, "memorias_saved": 0, "skipped_dup": 0, "error": str(exc)}


def _run_prune_floor(
    mem: Memory,
    roi_floor: float,
    min_age_days: int,
    dry_run: bool,
) -> list[dict]:
    """Archive memories below roi_floor with zero access and age >= min_age_days.

    Returns list of {id, roi_score, days_old} candidates (even in dry-run).
    """
    candidates = mem.store.prune_floor_candidates(roi_floor=roi_floor, min_age_days=min_age_days)
    if not dry_run:
        for c in candidates:
            try:
                mem.lifecycle.archive_memoria(c["id"])
            except Exception as exc:
                _log.warning("prune_floor: archive failed for %s: %s", c["id"], exc)
    return candidates


def _run_eviction(mem: Memory, max_count: int, dry_run: bool) -> list[dict]:
    """Archive LFU candidates until corpus size <= max_count.

    Returns list of {id, access_count} archived (or would-archive in dry-run).
    """
    conn = mem.store._conn
    try:
        total_row = conn.execute(
            "SELECT COUNT(*) AS n FROM meta WHERE type != 'reference'"
        ).fetchone()
        total = int(total_row["n"]) if total_row else 0
    except Exception:
        return []

    excess = total - max_count
    if excess <= 0:
        return []

    candidates = mem.store.eviction_candidates(
        policy="lfu",
        limit=excess,
        exclude_types={"reference", "synthesis"},
    )
    if not dry_run:
        for c in candidates:
            try:
                mem.lifecycle.archive_memoria(c["id"])
            except Exception as exc:
                _log.warning("eviction: archive failed for %s: %s", c["id"], exc)
    return [{"id": c["id"], "access_count": c.get("access_count", 0)} for c in candidates]


def _run_compress(mem: Memory, threshold: int, dry_run: bool) -> list[dict]:
    """Compress verbose memories (body > threshold chars) to 2-3 sentences.

    Returns list of {id, original_len, compressed_len}.
    """
    conn = mem.store._conn
    try:
        rows = conn.execute(
            "SELECT m.id, m.path FROM meta m "
            "JOIN fts ON fts.id = m.id "
            "WHERE m.type NOT IN ('reference','synthesis') "
            "AND length(fts.body) > ?",
            (threshold,),
        ).fetchall()
    except Exception:
        return []

    if not rows:
        return []

    from memo.memory.record import chat_with_timeout

    chat = mem._ensure_chat()
    results = []
    for row in rows:
        mid = row["id"]
        try:
            rec = mem.get(mid)
            if not rec or not rec.body:
                continue
            body_len = len(rec.body)
            if body_len <= threshold:
                continue
            user_prompt = (
                "Compress the following memory note to 2-3 concise sentences "
                "preserving all key facts, decisions, and context. "
                "Output ONLY the compressed text, no preamble.\n\n" + rec.body[:4000]
            )
            chat_out = chat_with_timeout(
                chat,
                timeout=30,
                model=mem.cfg.helper_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a concise technical writer. Compress memory notes.",
                    },
                    {"role": "user", "content": user_prompt},
                ],
                options={"temperature": 0.0, "max_tokens": 256, "thinking": False},
            )
            if chat_out is None:
                continue
            compressed = ((chat_out.get("message") or {}).get("content") or "").strip()
            if not compressed or len(compressed) >= body_len:
                continue
            if not dry_run:
                mem.update(mid, content=compressed)
            results.append({"id": mid, "original_len": body_len, "compressed_len": len(compressed)})
        except Exception as exc:
            _log.warning("compress: failed for %s: %s", mid, exc)
    return results


def _run_prewarm_queries(cfg: Any, mem: Memory, n: int) -> dict:
    """Pre-embed the n most recent unique queries from recall.log.

    Warms the LRU embed cache so the next recall-hook invocation hits cached
    embeddings instead of recomputing them. Never raises.
    """
    try:
        from memo.dashboard_logs import read_recall_log

        entries = read_recall_log(cfg.state_dir, limit=n * 3)
        seen: list[str] = []
        for e in entries:
            q = (e.get("prompt") or "").strip()
            if q and q not in seen:
                seen.append(q)
            if len(seen) >= n:
                break

        warmed = 0
        for q in seen:
            try:
                mem.embedder.embed_query(q)
                warmed += 1
            except Exception:  # noqa: S110
                pass
        return {"queries_warmed": warmed, "queries_available": len(seen)}
    except Exception as exc:
        return {"queries_warmed": 0, "queries_available": 0, "error": str(exc)}


def _run_presynthesis(cfg: Any, mem: Memory, top_n: int, dry_run: bool) -> list[dict]:
    """Pre-synthesize clusters for the top recurring queries.

    Reads recall.log, picks the top_n most frequent queries, runs a focused
    synthesis pass on the memories each query surfaces. Returns a list of
    synthesis results per query.
    """
    try:
        from collections import Counter

        from memo.dashboard_logs import read_recall_log

        entries = read_recall_log(cfg.state_dir, limit=200)
        counts: Counter = Counter()
        for e in entries:
            q = (e.get("prompt") or "").strip()
            if q:
                counts[q] += 1

        top_queries = [q for q, _ in counts.most_common(top_n)]
        if not top_queries:
            return []

        all_results = []
        for query in top_queries:
            try:
                hits = mem.search(query, limit=20, disable_reranker=True)
                if len(hits) < 3:
                    continue
                # Synthesize across the hit cluster
                source_ids = [h.id for h in hits]
                result = mem.synthesize_cross_cluster(
                    dry_run=dry_run, min_cluster_size=3, max_clusters=1
                )
                if result:
                    all_results.append(
                        {
                            "query": query[:80],
                            "hits": len(source_ids),
                            "synthesized": len(result),
                        }
                    )
            except Exception as exc:
                _log.warning("presynthesis: failed for query %r: %s", query[:50], exc)
        return all_results
    except Exception as exc:
        return [{"error": str(exc)}]


@click.group(name="dream")
def dream_cmd() -> None:
    """Autonomous nightly maintenance — synthesise, heal, decay."""


def _make_progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=24),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=False,
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
) -> None:
    """Run the full dream pipeline once.

    Example:
      memo dream run --dry-run
      memo dream run
    """
    cfg = Config.from_env()
    tag = "[dim](dry-run)[/dim] " if dry_run else ""
    console.print(f"{tag}[bold cyan]memo dream[/bold cyan] — starting pipeline...")

    from memo.flags import flag_int

    _evict_max = flag_int("MEMO_DREAM_EVICT_MAX_COUNT") or 0
    _compress_threshold = flag_int("MEMO_DREAM_COMPRESS_THRESHOLD") or 0
    _prewarm_n = flag_int("MEMO_DREAM_PREWARM_QUERIES") or 0
    _presynthesis_n = flag_int("MEMO_DREAM_PRESYNTHESIS_QUERIES") or 0

    receipt: dict[str, Any] = {
        "dry_run": dry_run,
        "orientation": {},
        "signal_gathered": {"files_processed": 0, "memorias_saved": 0, "skipped_dup": 0},
        "superseded": [],
        "evolved": [],
        "merged": [],
        "archived_stale": [],
        "synthesized": [],
        "entities_extracted": 0,
        "roi_decayed": 0,
        "confidence_penalized": 0,
        "pruned_floor": [],
        "evicted": [],
        "compressed": [],
        "prewarm": {},
        "presynthesis": [],
        "errors": [],
    }

    total_steps = 12
    skipped = (
        (1 if skip_signal_gather or dry_run else 0)
        + (4 if skip_maintain else 0)
        + (1 if skip_entities or dry_run else 0)
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
                    since_days = max(1, int((time.time() - last_ts) / 86400) + 1)
                except Exception:
                    since_days = 7
                sg = _run_signal_gather(since_days=since_days, file_limit=20)
                receipt["signal_gathered"] = sg
                progress.update(
                    step,
                    description=(
                        f"[0] signal gather [green]✓[/green]  "
                        f"{sg['files_processed']} files, {sg['memorias_saved']} saved"
                    ),
                )
            except Exception as exc:
                receipt["errors"].append(f"signal_gather: {type(exc).__name__}: {exc}")
                progress.update(step, description="[0] signal gather [yellow]warn[/yellow]")
            progress.advance(overall)
        else:
            progress.update(step, description="[0] signal gather [dim]skip[/dim]")

        # 0. Forget TTLs (always — explicit user intent) ---------------------
        progress.update(step, description="[dim]TTLs — enforce forget...[/dim]")
        try:
            for _item in mem.lifecycle.enforce_forget_ttl(dry_run=dry_run):
                pass
        except Exception as exc:
            receipt["errors"].append(f"forget_ttl: {type(exc).__name__}: {exc}")

        if not skip_maintain:
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
                contradicted_ids: list[str] = []
                for pair in mem.contradict_store.list_open(min_confidence=0.9):
                    rel = (pair.relationship or "").lower()
                    if "evolu" in rel:
                        if not dry_run:
                            mem.contradict_store.resolve(
                                pair.pair_id, "evolved", note="dream: evolution, both kept"
                            )
                        receipt["evolved"].append(pair.pair_id)
                        continue
                    if "contrad" not in rel:
                        continue
                    older, _newer = _older_id(mem, pair.memoria_id_a, pair.memoria_id_b)
                    contradicted_ids.extend([pair.memoria_id_a, pair.memoria_id_b])
                    if not dry_run:
                        ok = mem.lifecycle.archive_memoria(older)
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
                stale = mem.temporal.detect_stale_memorias(days_threshold=365, min_access_count=0)
                for item in stale:
                    mid = item.get("id")
                    if not mid:
                        continue
                    if not dry_run:
                        mem.lifecycle.archive_memoria(mid)
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

        # 6. ROI decay -------------------------------------------------------
        if not skip_decay and not dry_run:
            progress.update(
                step, description="[6/6] ROI decay — adjusting scores...", total=None, completed=0
            )
            try:
                n = mem.store.decay_roi(factor=0.98, older_than_days=30)
                receipt["roi_decayed"] = n
                progress.update(step, description=(f"[6/6] ROI decay [green]✓[/green]  {n} rows"))
            except Exception as exc:
                progress.update(step, description="[6/6] ROI decay [yellow]warn[/yellow]")
                receipt["errors"].append(f"roi_decay: {type(exc).__name__}: {exc}")
            progress.advance(overall)
        else:
            progress.update(step, description="[6/6] ROI decay [dim]skip[/dim]")

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
        if not skip_presynthesis and _presynthesis_n > 0:
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
    if sg.get("files_processed") or sg.get("memorias_saved"):
        console.print(
            f"  signal gather:             {sg['files_processed']} files, "
            f"{sg['memorias_saved']} saved, {sg.get('skipped_dup', 0)} dup skipped"
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
    if data.get("errors"):
        for e in data["errors"]:
            console.print(f"  [yellow]warn:[/yellow] {e}")


@dream_cmd.command(name="if-due")
def dream_if_due() -> None:
    """Spawn a background dream run if > 24h since last run (for launchd)."""
    import os as _os
    import subprocess as _sp

    cfg = Config.from_env()
    ts_file = _state_path(cfg) / ".last_run_ts"
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
    except Exception:  # noqa: S110
        pass
