"""`memo onboard` — Day-0 wizard: recall hook + transcript backfill + first briefing.

Orchestrates already-shipped pieces (wire_recall_hook, install_shims_cmd,
mine_transcripts); owns no heavy logic of its own.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import click

from memo.cli_common import console
from memo.config import Config
from memo.flags import flag_bool, flag_int
from memo.project import GLOBAL_BUCKET
from memo.runtime.daemon import _warm_embedder
from memo.runtime.shims import install_shims_cmd

_FM_TITLE_RE = re.compile(r"^title:\s*(.+)$", re.MULTILINE)
_FM_ID_RE = re.compile(r"^id:\s*(\S+)", re.MULTILINE)

DEFAULT_BACKFILL_DAYS = 90


def _recent_memories(memory_dir: Path, n: int = 3) -> list[dict[str, str]]:
    """Newest saved memories by mtime — the '3 cosas que ya sé de vos'.

    Disk-only on purpose: markdown is the source of truth and this must not
    cold-load MLX inside a first-run wizard.

    Title extraction priority: (a) YAML frontmatter title: field,
    (b) first H1 heading (# ), (c) filename stem."""
    if not memory_dir.exists():
        return []
    files = [
        p
        for p in memory_dir.rglob("*.md")
        if not any(
            part.startswith("_") and part != GLOBAL_BUCKET
            for part in p.relative_to(memory_dir).parts
        )
    ]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    out: list[dict[str, str]] = []
    for p in files:
        if len(out) >= n:
            break
        head = p.read_text(encoding="utf-8", errors="ignore")[:1000]
        if _FM_ID_RE.search(head) is None:
            continue  # no id: -> not a memory record

        # Priority 1: YAML frontmatter title:
        fm_match = _FM_TITLE_RE.search(head)
        if fm_match:
            title = fm_match.group(1).strip().strip("'\"")
        else:
            # Priority 2: First H1 heading; Priority 3: filename stem
            h1_line = next(
                (ln for ln in head.splitlines() if ln.startswith("# ")),
                None,
            )
            title = h1_line.removeprefix("# ").strip() if h1_line else p.stem

        out.append({"title": title, "file": p.name})
    return out


def _step_hook() -> dict[str, Any]:
    # Deferred so tests stub it at its home module (memo.cli_hooks).
    from memo.cli_hooks import _claude_dir, wire_recall_hook

    return wire_recall_hook(_claude_dir())


def _step_backfill(days: int, *, dry_run: bool) -> dict[str, Any]:
    # Deferred: transcript_miner drags capture deps; stubs also target its home.
    from memo.transcript_miner import mine_transcripts

    return mine_transcripts(since_days=days, dry_run=dry_run)


@click.command(name="onboard")
@click.option("--yes", is_flag=True, help="Correr todos los pasos sin preguntar.")
@click.option(
    "--days",
    type=int,
    default=None,
    help="Ventana de backfill en días (default: MEMO_ONBOARD_BACKFILL_DAYS=90).",
)
@click.option("--dry-run", is_flag=True, help="Estimar el backfill sin guardar nada.")
@click.option("--json", "as_json", is_flag=True, help="Emitir resumen JSON al final.")
@click.pass_context
def onboard(ctx: click.Context, yes: bool, days: int | None, dry_run: bool, as_json: bool) -> None:
    """Wizard Day-0: hook de recall + backfill de historial + primer briefing."""
    cfg = Config.from_env()
    interactive = not (flag_bool("MEMO_NONINTERACTIVE") or not sys.stdin.isatty())
    if not yes and not interactive:
        # Never prompt from hooks / pipes — mirror of `memo sync setup`.
        console.print(
            "Corré [cyan]memo onboard[/cyan] en una terminal interactiva, "
            "o [cyan]memo onboard --yes[/cyan] para el modo automático."
        )
        return

    summary: dict[str, Any] = {}

    # 1/4 — recall hook + shims (both idempotent)
    if dry_run:
        summary["hook"] = {"action": "skipped_dry_run"}
        console.print("1/4 · hook: (dry-run, salteado)")
    elif yes or click.confirm("1/4 · ¿Instalar el recall hook de Claude Code?", default=True):
        summary["hook"] = _step_hook()
        console.print(f"[green]✓[/green] hook: {summary['hook'].get('action')}")
        ctx.invoke(install_shims_cmd)

    # Warm the embedder + stamp .prewarm_ts so the FIRST recall runs vec, not the
    # cold-start bm25 fallback. On a fresh install a bm25 hit scores under the
    # vec-calibrated min_sim floor, so without this the first save is invisible to
    # the first recall. Lifecycle-only (no threshold change); best-effort.
    if not dry_run:
        # Embedder only — the reranker isn't needed for first recall and its model
        # may be uncached on a fresh install (which would stall the wizard).
        _warm_embedder(cfg, warm_reranker=False)
        summary["prewarm"] = {"action": "warmed"}
        console.print("[green]✓[/green] prewarm: embedder warmed (first recall runs vec)")

    # 2/4 — transcript backfill (Day-0 corpus from history already on disk)
    window = (
        days
        if days is not None
        else (flag_int("MEMO_ONBOARD_BACKFILL_DAYS") or DEFAULT_BACKFILL_DAYS)
    )
    if dry_run:
        summary["backfill"] = _step_backfill(window, dry_run=True)
        console.print(
            f"Backfill (dry-run): {summary['backfill'].get('files_total', 0)} transcripts, "
            f"~{summary['backfill'].get('candidates', 0)} candidatos ({window} días)."
        )
    elif yes:
        summary["backfill"] = _step_backfill(window, dry_run=False)
        console.print(
            f"[green]✓[/green] backfill: {summary['backfill'].get('saved', 0)} memorias nuevas "
            f"({summary['backfill'].get('skipped_dup', 0)} duplicados salteados)"
        )
    else:
        estimate = _step_backfill(window, dry_run=True)
        console.print(
            f"Backfill: {estimate.get('files_total', 0)} transcripts, "
            f"~{estimate.get('candidates', 0)} candidatos ({window} días)."
        )
        if click.confirm(f"2/4 · ¿Minar {window} días de historial ahora?", default=True):
            summary["backfill"] = _step_backfill(window, dry_run=False)
            console.print(
                f"[green]✓[/green] backfill: {summary['backfill'].get('saved', 0)} memorias nuevas"
            )
        else:
            summary["backfill"] = {"status": "skipped"}

    # 3/4 — other sources (pointers only; each has its own command)
    console.print(
        "3/4 · Otras fuentes: [cyan]memo import whatsapp[/cyan] · "
        "[cyan]memo import json[/cyan] · [cyan]memo import csv[/cyan]"
    )

    # 4/4 — first briefing: newest memories, straight from disk
    known = _recent_memories(Path(cfg.memory_dir))
    summary["memories"] = known
    if known:
        console.print("\n4/4 · [bold]3 cosas que ya sé de vos:[/bold]")
        for m in known:
            console.print(f"  · {m['title']}")
    else:
        console.print("4/4 · Todavía no hay memorias — van a aparecer solas mientras trabajás.")
    console.print("\nListo. Reiniciá la sesión de Claude Code para que el hook arranque.")

    if as_json:
        click.echo(json.dumps(summary, indent=2, ensure_ascii=False))
