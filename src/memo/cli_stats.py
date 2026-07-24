from __future__ import annotations

import json
import logging
from datetime import UTC, datetime

import click

from memo.cli_common import console
from memo.config import Config
from memo.dashboard import (
    consult_breakdown,
    read_context_cost_log,
    read_grounding_log,
)
from memo.dashboard_metrics import recall_health
from memo.memory import Memory

_log = logging.getLogger(__name__)


@click.command(name="stats")
@click.option("--json", "as_json", is_flag=True, help="Emit a stable JSON stats report.")
def stats(as_json: bool) -> None:
    """Comprehensive stats: utility, recall quality, consumers, tokens saved."""
    mem = Memory(Config.from_env())
    try:
        cfg = mem.cfg
        state_dir = cfg.state_dir
        embedder_identity = mem.store.embedder_model

        if as_json:
            report: dict[str, object] = {
                "schema": "memo.stats.v1",
                "corpus": {"total": mem.store.count(), "data_dir": str(cfg.data_dir)},
                "models": {
                    "profile": cfg.model_profile,
                    "embedder": embedder_identity,
                    "llm": cfg.llm_model,
                },
            }
            try:
                from memo.dashboard import read_usage_log

                report["utility"] = {
                    "tokens_saved": sum(
                        int(e.get("tokens_est", 0))
                        for e in read_context_cost_log(state_dir, limit=2000)
                    ),
                    "memories_used": len(
                        {e.get("id") for e in read_usage_log(state_dir, limit=2000) if e.get("id")}
                    ),
                }
            except Exception as exc:
                report["utility_error"] = str(exc)
            try:
                report["recall"] = recall_health(state_dir)
            except Exception as exc:
                report["recall_error"] = str(exc)
            try:
                report["consumers"] = consult_breakdown(state_dir)
            except Exception as exc:
                report["consumers_error"] = str(exc)
            click.echo(json.dumps(report, ensure_ascii=False, indent=2, default=str))
            return

        console.print(
            f"\n[bold cyan]memo stats — {datetime.now(UTC).strftime('%H:%M:%S')}[/bold cyan]"
        )

        console.print("\n[bold]📚 Corpus[/bold]")
        console.print(f"  total       {mem.store.count():,}")
        try:
            row = mem.store._conn.execute(
                "SELECT COUNT(*) FROM meta WHERE tags LIKE ? ESCAPE '\\'",
                ('%"\\_uncertain"%',),
            ).fetchone()
            n_uncertain = int(row[0]) if row else 0
            if n_uncertain:
                console.print(
                    f"  uncertain   {n_uncertain:,} (low-confidence capture, tag _uncertain)"
                )
        except Exception as exc:
            _log.debug("uncertain-tag count failed: %s", exc)
        console.print(f"  data_dir   {cfg.data_dir}")
    finally:
        mem.close()

    console.print("\n[bold]🧠 Models[/bold]")
    console.print(f"  profile    {cfg.model_profile}")
    console.print(f"  embedder  {embedder_identity.split('/')[-1]}")
    console.print(f"  llm      {cfg.llm_model.split('/')[-1]}")

    console.print("\n[bold]⚡ Utility (last 7 days)[/bold]")
    total_tokens = 0
    try:
        for e in read_context_cost_log(state_dir, limit=2000):
            total_tokens += e.get("tokens_est", 0)
    except Exception as exc:
        _log.debug("context cost log read failed: %s", exc)

    usage_ids: set[str] = set()
    try:
        from memo.dashboard import read_usage_log

        for e in read_usage_log(state_dir, limit=2000):
            if e.get("id"):
                usage_ids.add(e["id"])
    except Exception as exc:
        _log.debug("usage log read failed: %s", exc)

    grounding_yes = grounding_no = 0
    try:
        for e in read_grounding_log(state_dir, limit=2000):
            g = e.get("grounded")
            if g is True:
                grounding_yes += 1
            elif g is False:
                grounding_no += 1
    except Exception as exc:
        _log.debug("grounding log read failed: %s", exc)

    tokens_saved = total_tokens
    cost_usd = tokens_saved * 0.00001

    console.print(f"  tokens saved      {tokens_saved:,}")
    console.print(f"  cost $          ${cost_usd:.2f}")
    console.print(f"  memories used   {len(usage_ids)} unique")

    console.print("\n[bold]🎯 Recall Quality[/bold]")
    try:
        h = recall_health(state_dir)
        if h.get("fired"):
            hit_pct = h.get("hit_rate", 0) * 100
            strong_pct = h.get("strong_hit_rate", 0) * 100
            console.print(f"  recall hooks   {h['fired']} fired")
            console.print(f"  hit rate      {hit_pct:.0f}%")
            console.print(f"  strong hits   {strong_pct:.0f}% (score >0.7)")
            console.print(f"  latency p50  {h.get('p50_latency_ms', '—')}ms")
        else:
            console.print("  (no data)")
    except Exception as e:
        console.print(f"  [dim](error: {e})[/dim]")

    try:
        from memo.recall_metrics import summarize as _latency_summary

        latency = _latency_summary(state_dir, days=7)
    except Exception as exc:
        _log.debug("recall metrics summary failed: %s", exc)
        latency = {}
    if latency:
        console.print("\n[bold]⏱ Recall Latency (last 7 days)[/bold]")
        for path_name in ("daemon", "subprocess"):
            s = latency.get(path_name)
            if not s:
                continue
            console.print(
                f"  {path_name:<11} p50 {s['p50']:.0f}ms  p95 {s['p95']:.0f}ms  "
                f"p99 {s['p99']:.0f}ms  n={s['count']}"
            )

    console.print("\n[bold]👥 Consumers[/bold]")
    try:
        cb = consult_breakdown(state_dir)
        consumers = cb.get("consumers", [])[:5]
        if consumers:
            for c in consumers:
                name = c.get("consumer", "?")[:15]
                consults = c.get("consults", 0)
                hit = c.get("hit_rate")
                hit_s = f"{hit * 100:.0f}%" if hit else "—"
                console.print(f"  {name:<15} {consults:>4} consults {hit_s:>5} hit")
        else:
            console.print("  [dim](no data)[/dim]")
    except Exception as exc:
        _log.debug("consult breakdown failed: %s", exc)
        console.print("  [dim](no data)[/dim]")
