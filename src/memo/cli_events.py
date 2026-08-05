from __future__ import annotations

import json

import click

from .config import Config
from .event_surface import ingest_event, list_event_page, list_events


@click.group("events")
def events_group() -> None:
    """Read and write append-only runtime events in Memo state."""


@events_group.command("ingest")
@click.argument("payload", required=False)
@click.option("--expected-epoch", type=int)
def ingest_cmd(payload: str | None, expected_epoch: int | None) -> None:
    raw = payload if payload is not None else click.get_text_stream("stdin").read()
    try:
        result = ingest_event(
            json.loads(raw),
            state_dir=Config.from_env().state_dir,
            expected_epoch=expected_epoch,
        )
    except ValueError as exc:
        raise click.UsageError(str(exc)) from exc
    click.echo(json.dumps(result, sort_keys=True))


@events_group.command("list")
@click.option("--kind")
@click.option("--limit", default=100, show_default=True)
@click.option(
    "--cursor",
    default=None,
    help="Opaque cursor; enables bounded paginated output (empty means the beginning).",
)
@click.option("--since", help="ISO-8601 lower bound for initial cursor migration.")
def list_cmd(kind: str | None, limit: int, cursor: str | None, since: str | None) -> None:
    if cursor is None:
        if since is not None:
            raise click.UsageError("--since requires --cursor")
        payload: object = list_events(
            state_dir=Config.from_env().state_dir,
            kind=kind,
            limit=limit,
        )
    else:
        try:
            payload = list_event_page(
                state_dir=Config.from_env().state_dir,
                kind=kind,
                limit=limit,
                cursor=cursor,
                since=since,
            )
        except ValueError as exc:
            raise click.UsageError(str(exc)) from exc
    click.echo(
        json.dumps(
            payload,
            sort_keys=True,
        )
    )
