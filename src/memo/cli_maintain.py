"""`memo maintain` — keep the corpus fresh so memo stays a source of truth.

Orchestrates the freshness machinery that previously only ran by hand:

  1. Contradictions — scan for near-neighbours the helper LLM flags as
     contradicting/evolving. An *evolution* is resolved `evolved` (both kept).
     A genuine *contradiction* supersedes: the OLDER side is archived (moved to
     `inactive/`, reversible) and the pair resolved `kept_newer`. `--hard-delete`
     opts into a real delete for the most confident contradictions instead.
  2. Duplicates — consolidate high-similarity clusters (merge → archive the
     sources with an `archived_for` pointer; reversible).
  3. Staleness — archive memorias never accessed and older than `--stale-days`.

Every mutation is **reversible by default** (archive, not delete). `--dry-run`
previews without touching anything. A receipt of what changed is written to
`<state>/maintain/last.json` plus a timestamp the SessionStart guard reads so
the auto-run fires at most once per day.

The auto-run (MEMO_MAINTAIN_AUTO daily guard) always uses the safe archive
path; `--hard-delete` is manual-only.
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
    return cfg.state_dir / "maintain"


def _older_id(mem: Any, id_a: str, id_b: str) -> tuple[str, str]:
    """Return (older_id, newer_id) by `updated` timestamp; falls back to the
    pair order (a, b) when a record or timestamp is missing."""
    ra, rb = mem.get(id_a), mem.get(id_b)
    ua = getattr(ra, "updated", "") or ""
    ub = getattr(rb, "updated", "") or ""
    if ua and ub:
        return (id_a, id_b) if ua <= ub else (id_b, id_a)
    return id_a, id_b


@click.command(name="maintain")
@click.option("--dry-run", is_flag=True, help="Preview actions; change nothing.")
@click.option("--min-confidence", type=float, default=0.9,
              help="Confidence floor for auto-acting on a contradiction (default 0.9).")
@click.option("--hard-delete", is_flag=True,
              help="Delete (not archive) the superseded side of a contradiction. "
                   "Manual-only; the daily auto-run never does this.")
@click.option("--stale-days", type=int, default=365,
              help="Archive never-accessed memorias older than this (default 365).")
@click.option("--dup-threshold", type=float, default=0.9,
              help="Cosine threshold for duplicate clustering (default 0.9).")
@click.option("--max-pairs", type=int, default=200,
              help="Max contradiction candidate pairs to scan (default 200).")
@click.option("--skip-contradict", is_flag=True, help="Skip the contradiction pass.")
@click.option("--skip-consolidate", is_flag=True, help="Skip the duplicate-merge pass.")
@click.option("--skip-stale", is_flag=True, help="Skip the staleness pass.")
@click.option("--json", "as_json", is_flag=True, help="Emit the receipt as JSON.")
@click.option("--if-due", is_flag=True,
              help="No-op unless >24h since the last run; then spawn maintain "
                   "detached (safe archive-only) and return. For the daily "
                   "SessionStart guard.")
def maintain_cmd(dry_run: bool, min_confidence: float, hard_delete: bool,
                 stale_days: int, dup_threshold: float, max_pairs: int,
                 skip_contradict: bool, skip_consolidate: bool, skip_stale: bool,
                 as_json: bool, if_due: bool) -> None:
    """Supersede contradictions, merge duplicates, archive stale memorias.

    Reversible by default (archives to inactive/). Example:
      memo maintain --dry-run
    """
    cfg = Config.from_env()

    # Daily guard for the SessionStart hook: cheap, builds no Memory.
    if if_due:
        import os as _os
        import subprocess as _sp
        if _os.environ.get("MEMO_MAINTAIN_DISABLE") == "1":
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
                ["memo", "maintain"],  # safe defaults: archive-only, no --hard-delete
                stdin=_sp.DEVNULL, stdout=_sp.DEVNULL, stderr=_sp.DEVNULL,
                start_new_session=True,
                env={**_os.environ, "MEMO_NONINTERACTIVE": "1"},
            )
        except Exception:
            pass
        return

    mem = _get_memory(cfg)
    receipt: dict[str, Any] = {
        "dry_run": dry_run,
        "hard_delete": hard_delete,
        "superseded": [],   # contradictions acted on
        "evolved": [],      # contradictions marked evolution (both kept)
        "merged": [],       # duplicate clusters consolidated
        "archived_stale": [],
        "errors": [],
    }

    # 1. Contradictions ------------------------------------------------------
    if not skip_contradict:
        try:
            mem.contradict_scanner.scan_corpus(
                confidence_threshold=min_confidence, max_pairs=max_pairs,
            )
            for pair in mem.contradict_store.list_open(min_confidence=min_confidence):
                rel = (pair.relationship or "").lower()
                if "evolu" in rel:
                    if not dry_run:
                        mem.contradict_store.resolve(pair.pair_id, "evolved",
                                                     note="auto: evolution, both kept")
                    receipt["evolved"].append(pair.pair_id)
                    continue
                if "contrad" not in rel:
                    continue  # consistent / unrelated — leave open
                older, _newer = _older_id(mem, pair.memoria_id_a, pair.memoria_id_b)
                action = "delete" if hard_delete else "archive"
                if not dry_run:
                    ok = (mem.delete(older) if hard_delete
                          else mem.lifecycle.archive_memoria(older))
                    if ok:
                        mem.contradict_store.resolve(
                            pair.pair_id, "kept_newer",
                            note=f"auto: {action}d older {older}")
                receipt["superseded"].append(
                    {"pair_id": pair.pair_id, "older": older, "action": action,
                     "confidence": pair.confidence})
        except Exception as exc:  # noqa: BLE001 — never let one pass abort the rest
            receipt["errors"].append(f"contradict: {type(exc).__name__}: {exc}")

    # 2. Duplicates ----------------------------------------------------------
    if not skip_consolidate:
        try:
            res = mem.consolidator.consolidate_all(
                threshold=dup_threshold, auto_apply=True, dry_run=dry_run,
            )
            for r in res.get("results", []):
                receipt["merged"].append(
                    {"merged_id": r.get("merged_id"),
                     "archived_ids": r.get("archived_ids", [])})
            if not res.get("results") and res.get("proposals"):
                # dry_run path: proposals exist but nothing applied
                receipt["merged"] = [{"would_merge": p.get("memoria_ids")}
                                     for p in res.get("proposals", [])]
        except Exception as exc:  # noqa: BLE001
            receipt["errors"].append(f"consolidate: {type(exc).__name__}: {exc}")

    # 3. Staleness -----------------------------------------------------------
    if not skip_stale:
        try:
            stale = mem.temporal.detect_stale_memorias(
                days_threshold=stale_days, min_access_count=0)
            for item in stale:
                mid = item.get("id")
                if not mid:
                    continue
                if not dry_run:
                    mem.lifecycle.archive_memoria(mid)
                receipt["archived_stale"].append(
                    {"id": mid, "days": item.get("days_since_update")})
        except Exception as exc:  # noqa: BLE001
            receipt["errors"].append(f"stale: {type(exc).__name__}: {exc}")

    # Persist receipt + timestamp (the daily guard reads the timestamp). Even
    # a dry-run stamps so a preview doesn't immediately re-trigger; the guard
    # cares only about "ran recently".
    if not dry_run:
        try:
            d = _state_path(cfg)
            d.mkdir(parents=True, exist_ok=True)
            (d / "last.json").write_text(
                json.dumps({"ts": time.time(), **receipt}, ensure_ascii=False, indent=2),
                encoding="utf-8")
            (d / ".last_run_ts").write_text(str(time.time()), encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            receipt["errors"].append(f"receipt: {type(exc).__name__}: {exc}")

    if as_json:
        click.echo(json.dumps(receipt, ensure_ascii=False, indent=2))
        return

    tag = "[dim](dry-run)[/dim] " if dry_run else ""
    console.print(f"{tag}[bold]memo maintain[/bold]")
    console.print(f"  contradictions superseded: {len(receipt['superseded'])} "
                  f"({'delete' if hard_delete else 'archive'}), "
                  f"evolutions marked: {len(receipt['evolved'])}")
    console.print(f"  duplicate clusters merged: {len(receipt['merged'])}")
    console.print(f"  stale memorias archived: {len(receipt['archived_stale'])}")
    if receipt["errors"]:
        for e in receipt["errors"]:
            console.print(f"  [yellow]warn:[/yellow] {e}")
