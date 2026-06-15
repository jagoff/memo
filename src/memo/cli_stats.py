from __future__ import annotations

import contextlib
from typing import Any

import click

from memo.cli_common import console
from memo.config import Config


@click.command()
def stats() -> None:
    """Summary stats — total records, vault path, embedder model."""
    from memo.memory import Memory

    mem = Memory(Config.from_env())
    history_errors = 0
    with contextlib.suppress(Exception):
        history_errors = int(getattr(mem.history, "error_count", 0))
    info: dict[str, Any] = {
        "total": mem.store.count(),
        "data_dir": str(mem.cfg.data_dir),
        "vault_path": str(mem.cfg.vault_path) if mem.cfg.vault_path else "(unset)",
        "db_path": str(mem.cfg.db_path),
        "model_profile": mem.cfg.model_profile,
        "embedder_model": mem.cfg.embedder_model,
        "llm_model": mem.cfg.llm_model,
        "history_errors": history_errors,
    }
    for k, v in info.items():
        console.print(f"[dim]{k:14s}[/dim] {v}")
    with contextlib.suppress(Exception):
        from memo.dashboard import recall_health

        h = recall_health(mem.cfg.state_dir)
        if h.get("sampled"):
            console.print(
                f"[dim]recall_health [/dim] fired={h['fired']} bailed={h['bailed']} "
                f"hit_rate={h['hit_rate']} top_score={h['median_top_score']} "
                f"p50={h['p50_latency_ms']}ms [dim](last {h['sampled']})[/dim]"
            )
