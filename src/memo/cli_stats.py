from __future__ import annotations

import contextlib
from datetime import datetime
from typing import Any

import click

from memo.cli_common import console
from memo.config import Config
from memo.memory import Memory
from memo.dashboard import (
    consult_breakdown,
    grounding_log_path,
    read_context_cost_log,
    read_grounding_log,
    usage_log_path,
)
from memo.dashboard_metrics import recall_health


@click.command(name="stats")
def stats() -> None:
    """Comprehensive stats: utility, recall quality, consumers, tokens saved."""
    mem = Memory(Config.from_env())
    state_dir = mem.cfg.state_dir

    console.print(f"\n[bold cyan]memo stats — {datetime.now().strftime('%H:%M:%S')}[/bold cyan]")

    console.print(f"\n[bold]📚 Corpus[/bold]")
    console.print(f"  total       {mem.store.count():,}")
    console.print(f"  data_dir   {mem.cfg.data_dir}")

    console.print(f"\n[bold]🧠 Models[/bold]")
    console.print(f"  profile    {mem.cfg.model_profile}")
    console.print(f"  embedder  {mem.cfg.embedder_model.split('/')[-1]}")
    console.print(f"  llm      {mem.cfg.llm_model.split('/')[-1]}")

    console.print(f"\n[bold]⚡ Utility (últimos 7 días)[/bold]")
    total_tokens = 0
    try:
        for e in read_context_cost_log(state_dir, limit=2000):
            total_tokens += e.get("tokens_est", 0)
    except Exception:
        pass

    usage_ids: set[str] = set()
    try:
        from memo.dashboard import read_usage_log
        for e in read_usage_log(state_dir, limit=2000):
            if e.get("id"):
                usage_ids.add(e["id"])
    except Exception:
        pass

    grounding_yes = grounding_no = 0
    try:
        for e in read_grounding_log(state_dir, limit=2000):
            g = e.get("grounded")
            if g is True:
                grounding_yes += 1
            elif g is False:
                grounding_no += 1
    except Exception:
        pass

    tokens_saved = total_tokens
    cost_usd = tokens_saved * 0.00001
    grounding_rate = (grounding_yes / (grounding_yes + grounding_no) * 100) if (grounding_yes + grounding_no) else 0

    console.print(f"  tokens evitados   {tokens_saved:,}")
    console.print(f"  costo $         ${cost_usd:.2f}")
    console.print(f"  memorias usadas {len(usage_ids)} únicas")

    console.print(f"\n[bold]🎯 Recall Quality[/bold]")
    try:
        h = recall_health(state_dir)
        if h.get("fired"):
            hit_pct = (h.get("hit_rate", 0) * 100)
            strong_pct = (h.get("strong_hit_rate", 0) * 100)
            console.print(f"  recall hooks   {h['fired']} fired")
            console.print(f"  hit rate      {hit_pct:.0f}%")
            console.print(f"  strong hits   {strong_pct:.0f}% (score >0.7)")
            console.print(f"  latency p50  {h.get('p50_latency_ms', '—')}ms")
        else:
            console.print("  (sin datos)")
    except Exception as e:
        console.print(f"  [dim](error: {e})[/dim]")

    console.print(f"\n[bold]👥 Consumers[/bold]")
    try:
        cb = consult_breakdown(state_dir)
        consumers = cb.get("consumers", [])[:5]
        if consumers:
            for c in consumers:
                name = c.get("consumer", "?")[:15]
                consults = c.get("consults", 0)
                hit = c.get("hit_rate")
                hit_s = f"{hit*100:.0f}%" if hit else "—"
                console.print(f"  {name:<15} {consults:>4} consults {hit_s:>5} hit")
        else:
            console.print("  [dim](sin datos)[/dim]")
    except Exception:
        console.print("  [dim](sin datos)[/dim]")
