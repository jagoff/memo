"""`memo dream` — autonomous background maintenance pipeline.

Bundles every maintenance pass into a single nightly run:

  1. Contradictions — scan + auto-resolve evolutions, supersede older sides.
  2. Duplicates — consolidate high-similarity clusters.
  3. Staleness — archive memorias never accessed after 365 days.
  4. Synthesis — emergent cross-cluster insights (runs regardless of
     MEMO_SYNTHESIS_ENABLED; Dream mode always synthesises).
  5. Entity backfill — extract entities for memorias not yet indexed.
  6. ROI decay — multiply roi_score × 0.98 for memorias idle > 30 days
     so unaccessed knowledge gradually yields to frequently-used facts.

`memo dream run` — run once (foreground).
`memo dream status` — when did it last run, what did it change.
`memo dream --if-due` — no-op unless > 24h since last run (for launchd).
"""

from __future__ import annotations

import json
import time
from typing import Any

import click

from memo.cli_common import console
from memo.cli_common import get_memory as _get_memory
from memo.config import Config


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


@dream_cmd.command(name="run")
@click.option("--dry-run", is_flag=True, help="Preview actions; change nothing.")
@click.option("--json", "as_json", is_flag=True, help="Emit receipt as JSON.")
@click.option("--skip-entities", is_flag=True, help="Skip entity backfill pass.")
@click.option("--skip-decay", is_flag=True, help="Skip ROI decay pass.")
@click.option("--skip-maintain", is_flag=True, help="Skip the contradict/consolidate/stale/synthesize passes.")
def dream_run(
    dry_run: bool, as_json: bool,
    skip_entities: bool, skip_decay: bool, skip_maintain: bool,
) -> None:
    """Run the full dream pipeline once.

    Example:
      memo dream run --dry-run
      memo dream run
    """
    cfg = Config.from_env()
    tag = "[dim](dry-run)[/dim] " if dry_run else ""
    console.print(f"{tag}[bold cyan]memo dream[/bold cyan] — iniciando pipeline...")
    console.print("  cargando memoria...", end=" ")
    mem = _get_memory(cfg)
    console.print("[green]✓[/green]")
    receipt: dict[str, Any] = {
        "dry_run": dry_run,
        "superseded": [],
        "evolved": [],
        "merged": [],
        "archived_stale": [],
        "synthesized": [],
        "entities_extracted": 0,
        "roi_decayed": 0,
        "confidence_penalized": 0,
        "errors": [],
    }

    # 0. Forget TTLs (always — explicit user intent) -------------------------
    try:
        for item in mem.lifecycle.enforce_forget_ttl(dry_run=dry_run):
            pass
    except Exception as exc:
        receipt["errors"].append(f"forget_ttl: {type(exc).__name__}: {exc}")

    if not skip_maintain:
        # 1. Contradictions --------------------------------------------------
        console.print("  [1/6] contradicciones — escaneando corpus...", end=" ")
        try:
            mem.contradict_scanner.scan_corpus(confidence_threshold=0.9, max_pairs=300)
            contradicted_ids: list[str] = []
            for pair in mem.contradict_store.list_open(min_confidence=0.9):
                rel = (pair.relationship or "").lower()
                if "evolu" in rel:
                    if not dry_run:
                        mem.contradict_store.resolve(pair.pair_id, "evolved",
                                                     note="dream: evolution, both kept")
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
                            pair.pair_id, "kept_newer",
                            note=f"dream: archived older {older}")
                receipt["superseded"].append({"pair_id": pair.pair_id, "older": older})
            if contradicted_ids and not dry_run:
                mem.store.penalize_confidence_batch(contradicted_ids)
                receipt["confidence_penalized"] = len(set(contradicted_ids))
            console.print(
                f"[green]✓[/green] {len(receipt['superseded'])} superseded, "
                f"{len(receipt['evolved'])} evolved")
        except Exception as exc:
            console.print(f"[yellow]warn[/yellow]")
            receipt["errors"].append(f"contradict: {type(exc).__name__}: {exc}")

        # 2. Duplicates ------------------------------------------------------
        console.print("  [2/6] duplicados — consolidando clusters...", end=" ")
        try:
            res = mem.consolidator.consolidate_all(
                threshold=0.9, auto_apply=True, dry_run=dry_run,
            )
            for r in res.get("results", []):
                receipt["merged"].append(
                    {"merged_id": r.get("merged_id"),
                     "archived_ids": r.get("archived_ids", [])})
            console.print(f"[green]✓[/green] {len(receipt['merged'])} merged")
        except Exception as exc:
            console.print(f"[yellow]warn[/yellow]")
            receipt["errors"].append(f"consolidate: {type(exc).__name__}: {exc}")

        # 3. Staleness -------------------------------------------------------
        console.print("  [3/6] memorias stale — detectando...", end=" ")
        try:
            stale = mem.temporal.detect_stale_memorias(days_threshold=365, min_access_count=0)
            for item in stale:
                mid = item.get("id")
                if not mid:
                    continue
                if not dry_run:
                    mem.lifecycle.archive_memoria(mid)
                receipt["archived_stale"].append(
                    {"id": mid, "days": item.get("days_since_update")})
            console.print(f"[green]✓[/green] {len(receipt['archived_stale'])} archivadas")
        except Exception as exc:
            console.print(f"[yellow]warn[/yellow]")
            receipt["errors"].append(f"stale: {type(exc).__name__}: {exc}")

        # 4. Emergent synthesis (always on in Dream mode) --------------------
        console.print("  [4/6] síntesis emergente — generando insights...", end=" ")
        try:
            results = mem.synthesize_cross_cluster(dry_run=dry_run, min_cluster_size=5)
            for r in results:
                receipt["synthesized"].append({
                    "title": r.get("title"),
                    "confidence": r.get("confidence"),
                    "saved": r.get("saved", False),
                })
            saved_n = sum(1 for s in receipt["synthesized"] if s.get("saved"))
            console.print(f"[green]✓[/green] {saved_n} guardadas, {len(receipt['synthesized'])} propuestas")
        except Exception as exc:
            console.print(f"[yellow]warn[/yellow]")
            receipt["errors"].append(f"synthesize: {type(exc).__name__}: {exc}")

    # 5. Entity backfill (memorias not yet in graph) -------------------------
    if not skip_entities and not dry_run:
        console.print("  [5/6] entidades — extrayendo de memorias sin indexar...", end=" ")
        try:
            counts = mem.extract_entities(all_=True, skip_already_indexed=True)
            receipt["entities_extracted"] = counts.get("entities_extracted", 0)
            console.print(f"[green]✓[/green] {receipt['entities_extracted']} extraídas")
        except Exception as exc:
            console.print(f"[yellow]warn[/yellow]")
            receipt["errors"].append(f"entities: {type(exc).__name__}: {exc}")
    elif skip_entities or dry_run:
        console.print("  [5/6] entidades — [dim]skip[/dim]")

    # 6. ROI decay (idle memorias lose score over time) ----------------------
    if not skip_decay and not dry_run:
        console.print("  [6/6] ROI decay — ajustando scores...", end=" ")
        try:
            n = mem.store.decay_roi(factor=0.98, older_than_days=30)
            receipt["roi_decayed"] = n
            console.print(f"[green]✓[/green] {n} filas")
        except Exception as exc:
            console.print(f"[yellow]warn[/yellow]")
            receipt["errors"].append(f"roi_decay: {type(exc).__name__}: {exc}")
    elif skip_decay or dry_run:
        console.print("  [6/6] ROI decay — [dim]skip[/dim]")

    # Persist receipt + timestamp --------------------------------------------
    if not dry_run:
        try:
            d = _state_path(cfg)
            d.mkdir(parents=True, exist_ok=True)
            (d / "last.json").write_text(
                json.dumps({"ts": time.time(), **receipt}, ensure_ascii=False, indent=2),
                encoding="utf-8")
            (d / ".last_run_ts").write_text(str(time.time()), encoding="utf-8")
        except Exception as exc:
            receipt["errors"].append(f"receipt: {type(exc).__name__}: {exc}")

    if as_json:
        click.echo(json.dumps(receipt, ensure_ascii=False, indent=2))
        return

    tag = "[dim](dry-run)[/dim] " if dry_run else ""
    console.print(f"{tag}[bold]memo dream[/bold]")
    console.print(f"  contradictions superseded: {len(receipt['superseded'])}, "
                  f"evolutions: {len(receipt['evolved'])}, "
                  f"confidence penalized: {receipt['confidence_penalized']}")
    console.print(f"  duplicate clusters merged: {len(receipt['merged'])}")
    console.print(f"  stale memorias archived:   {len(receipt['archived_stale'])}")
    if receipt["synthesized"]:
        saved = sum(1 for s in receipt["synthesized"] if s.get("saved"))
        console.print(f"  emergent syntheses:        {saved} saved, "
                      f"{len(receipt['synthesized'])} proposed")
    console.print(f"  entities extracted:        {receipt['entities_extracted']}")
    console.print(f"  roi rows decayed:          {receipt['roi_decayed']}")
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
            stdin=_sp.DEVNULL, stdout=_sp.DEVNULL, stderr=_sp.DEVNULL,
            start_new_session=True,
            env={**_os.environ, "MEMO_NONINTERACTIVE": "1"},
        )
    except Exception:
        pass
