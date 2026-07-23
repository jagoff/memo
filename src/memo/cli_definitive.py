"""CLI readiness gate for the definitive Memo runtime."""

from __future__ import annotations

import json
from typing import Any

import click

from memo.config import Config


def _json(value: Any) -> None:
    click.echo(json.dumps(value, ensure_ascii=False, indent=2, default=str))


@click.group(name="definitive")
def definitive_group() -> None:
    """Verify and benchmark Memo's independent-memory guarantees."""


@definitive_group.command(name="check")
def definitive_check_cmd() -> None:
    from memo.definitive import definitive_check
    from memo.memory import Memory

    memory = Memory(Config.from_env())
    try:
        report = definitive_check(memory)
    finally:
        memory.close()
    _json(report)
    if not report["ok"]:
        raise click.ClickException("definitive readiness gate failed")


@definitive_group.command(name="benchmark")
@click.option("--events", default=250, type=click.IntRange(10, 10_000))
@click.option("--minimum-eps", default=25.0, type=click.FloatRange(min=0.01))
def definitive_benchmark_cmd(events: int, minimum_eps: float) -> None:
    from memo.definitive import run_journal_benchmark

    report = run_journal_benchmark(
        events=events,
        min_events_per_second=minimum_eps,
    )
    _json(report)
    if not report["ok"]:
        raise click.ClickException("definitive benchmark gate failed")


__all__ = ["definitive_group"]
