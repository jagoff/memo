"""`memo hype status` — inspect the HyPE index coverage."""

from __future__ import annotations

import json

import click

from memo.cli_common import get_memory as _get_memory
from memo.config import Config
from memo.flags import flag_int
from memo.store.hype_store import HypeStore
from memo.tiers import DURABLE_TYPES


@click.group(name="hype")
def hype_group() -> None:
    """HyPE — Hypothetical Questions for Expansion index status."""


@hype_group.command(name="status")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON.")
def status_cmd(as_json: bool) -> None:
    """Show HyPE index coverage: memories indexed, questions, backlog."""
    from memo.dream_hype import _active_variant, select_backlog

    cfg = Config.from_env()
    mem = _get_memory(cfg)
    identity = str(getattr(mem.store, "embedder_model", "") or "")
    store = HypeStore(
        cfg.db_path,
        cfg.embedder_dims,
        **({"embedder_model": identity} if identity else {}),
    )

    try:
        stats = store.stats()

        # Total durable memories
        durable_total = sum(
            1
            for memory_id in mem.store.all_ids()
            if (row := mem.store.get(memory_id)) and row.get("type") in DURABLE_TYPES
        )

        # Coverage percentage
        coverage_pct = (stats["memories"] / durable_total * 100.0) if durable_total > 0 else 0.0

        # Backlog (read-only, respects cap)
        night_cap = flag_int("MEMO_HYPE_NIGHT_CAP") or 400
        backlog = select_backlog(mem, store, cap=night_cap)

        result = {
            "indexed_memories": stats["memories"],
            "questions": stats["questions"],
            "variants": stats.get("by_variant", {}),
            "active_variant": _active_variant(),
            "durable_total": durable_total,
            "coverage_pct": coverage_pct,
            "backlog": len(backlog),
        }

        if as_json:
            click.echo(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            click.echo(
                f"Indexed:  {stats['memories']}/{durable_total} memories ({coverage_pct:.1f}%)"
            )
            click.echo(f"Questions: {stats['questions']}")
            variants = ", ".join(
                f"{variant}={count}"
                for variant, count in sorted(stats.get("by_variant", {}).items())
            )
            click.echo(f"Variants:  {variants or 'none'} (active={_active_variant()})")
            click.echo(f"Backlog:  {len(backlog)} pending")
    finally:
        store.close()
