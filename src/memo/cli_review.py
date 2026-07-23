from __future__ import annotations

import json

import click

from memo.cli_common import get_memory
from memo.config import Config


@click.group(name="review")
def review_group() -> None:
    """Inspect and close explicit memory review obligations."""


@review_group.command(name="due")
@click.option("--project", default=None)
@click.option("--limit", type=click.IntRange(1, 1000), default=50)
@click.option("--json", "as_json", is_flag=True)
def review_due(project: str | None, limit: int, as_json: bool) -> None:
    rows = get_memory(Config.from_env()).list_due_reviews(project=project, limit=limit)
    if as_json:
        click.echo(json.dumps(rows, ensure_ascii=False, default=str))
        return
    if not rows:
        click.echo("No reviews due.")
        return
    for row in rows:
        conflict = " conflict" if row.get("open_conflict") else ""
        click.echo(
            f"{str(row['id'])[:8]}  {row['type']:<12} {row['title']}"
            f"  due={row.get('review_after') or 'now'}{conflict}"
        )


@review_group.command(name="mark")
@click.argument("id_")
@click.option("--evidence", default=None)
@click.option("--actor", default=None)
@click.option("--json", "as_json", is_flag=True)
def review_mark(id_: str, evidence: str | None, actor: str | None, as_json: bool) -> None:
    record = get_memory(Config.from_env()).mark_reviewed(id_, evidence=evidence, actor=actor)
    if as_json:
        click.echo(json.dumps(record.to_dict(), ensure_ascii=False, default=str))
    else:
        click.echo(f"Reviewed {record.id[:8]}; next={record.review_after or 'unscheduled'}")
