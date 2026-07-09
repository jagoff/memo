"""`memo chat` command group — chat-shaped RAG over the memory archive.

Extracted as a new CLI surface (3a decomposition). Registered onto the
root group in cli.py via `cli.add_command(chat_group)`.

The flat `memo chat-ask` command (from cli_search.py) remains registered
for backwards compatibility. This group exposes the same functionality
as `memo chat ask` — a more natural command hierarchy.
"""

from __future__ import annotations

import json
import logging
import sys

import click
from rich.panel import Panel

from memo.cli_common import console
from memo.cli_common import get_memory as _get_memory
from memo.config import Config

_log = logging.getLogger(__name__)


def _format_source_score(score: object) -> str:
    return f"{score:.3f}" if isinstance(score, (int, float)) else "—"


@click.group(name="chat")
def chat_group() -> None:
    """Chat-shaped RAG commands."""
    pass


@chat_group.command(name="ask")
@click.argument("question")
@click.option(
    "--k", default=7, type=int, show_default=True,
    help="Top-K memories to feed the LLM as context.",
)
@click.option("--type", "type_", default=None, help="Restrict retrieval to one record type.")
@click.option(
    "--snippet-chars",
    default=800,
    type=int,
    show_default=True,
    help="Preview length for retrieved memory snippets.",
)
@click.option(
    "--history-json",
    type=click.File("r"),
    default=None,
    help="Conversation history JSON: list of {role,text}. '-' reads stdin.",
)
@click.option(
    "--context-json",
    type=click.File("r"),
    default=None,
    help="Caller-supplied federation context (e.g. Synapse packet) for richer synthesis.",
)
@click.option("--json", "as_json", is_flag=True)
@click.option(
    "--stream",
    "as_stream",
    is_flag=True,
    help=(
        "Emit one NDJSON event per line (context/token/done) flushed "
        "immediately. Forces JSON output and disables the panel."
    ),
)
def chat_ask(
    question: str,
    k: int,
    type_: str | None,
    snippet_chars: int,
    history_json,
    context_json,
    as_json: bool,
    as_stream: bool,
) -> None:
    """Chat-shaped RAG over memo — synthesise a conversational answer.

    Examples:

      memo chat ask "what do I know about MLX?"
      memo chat ask --stream "explain my setup"
      memo chat ask --history-json history.json "follow-up question"
    """
    history: list[dict] = []
    if history_json is not None:
        try:
            raw = json.load(history_json)
            if isinstance(raw, list):
                history = [h for h in raw if isinstance(h, dict)]
        except Exception as exc:
            _log.warning("chat ask: failed to load history JSON: %s", exc)
            history = []

    context: dict = {}
    if context_json is not None:
        try:
            ctx = json.load(context_json)
            if isinstance(ctx, dict):
                context = ctx
        except Exception as exc:
            _log.warning("chat ask: failed to load context JSON: %s", exc)
            context = {}

    mem = _get_memory(Config.from_env())

    if as_stream:
        for event in mem.chat_ask_stream(
            question,
            k=k,
            type_=type_,
            history=history,
            context=context,
            snippet_chars=snippet_chars,
        ):
            sys.stdout.write(json.dumps(event, ensure_ascii=False) + "\n")
            sys.stdout.flush()
        return

    envelope = mem.chat_ask(
        question,
        k=k,
        type_=type_,
        history=history,
        context=context,
        snippet_chars=snippet_chars,
    )
    if as_json:
        click.echo(json.dumps(envelope, ensure_ascii=False, indent=2))
        return
    console.print(
        Panel.fit(
            envelope["answer"] or "[dim](no answer)[/dim]",
            title=f"? {question[:60]}",
            border_style="magenta",
        )
    )
    if envelope["sources"]:
        console.print("[dim]sources:[/dim]")
        for s in envelope["sources"]:
            console.print(
                f"  [dim][{s.get('id_short', '?')}][/dim] {(s.get('title', '') or '')[:60]}  "
                f"[dim](score {_format_source_score(s.get('score'))})[/dim]"
            )
