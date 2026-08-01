from __future__ import annotations

import json

import click

from memo.config import Config
from memo.event_surface import ingest_event, list_events


@click.group("events")
def events_group() -> None:
    """Local diagnostic event log; not replicated. Use `memo mesh` for peers."""


@events_group.command("ingest")
@click.argument("payload", required=False)
@click.option("--expected-epoch", type=int)
def ingest_cmd(payload: str | None, expected_epoch: int | None) -> None:
    raw = payload if payload is not None else click.get_text_stream("stdin").read()
    click.echo(
        json.dumps(
            ingest_event(
                json.loads(raw),
                state_dir=Config.from_env().state_dir,
                expected_epoch=expected_epoch,
            ),
            sort_keys=True,
        )
    )


@events_group.command("list")
@click.option("--kind")
@click.option("--limit", default=100, show_default=True)
def list_cmd(kind: str | None, limit: int) -> None:
    click.echo(
        json.dumps(
            list_events(
                state_dir=Config.from_env().state_dir,
                kind=kind,
                limit=limit,
            ),
            sort_keys=True,
        )
    )
