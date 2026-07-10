import json

import click

from memo.config import Config


@click.group(name="guard")
def guard_group() -> None:
    """Inspect guard (prior-decision flag) activity."""


@guard_group.command(name="stats")
def guard_stats() -> None:
    """Print guard fire count and the most-flagged memory ids."""
    from collections import Counter

    cfg = Config.from_env()
    path = cfg.state_dir / "guard.log"
    if not path.exists():
        click.echo("no guard fires logged")
        return
    counts: Counter = Counter()
    n = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        n += 1
        for i in rec.get("ids", []):
            counts[i] += 1
    click.echo(f"{n} guard fires")
    for i, c in counts.most_common(10):
        click.echo(f"  {c:>4}  {i}")
