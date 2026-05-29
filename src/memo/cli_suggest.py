"""`memo suggest` command group — proactive suggestions.

Extracted from cli.py (3a god-module decomposition). Registered onto the
root group in cli.py via `cli.add_command(suggest_group)`.
"""

from __future__ import annotations

import json

import click

from memo.cli_common import console
from memo.cli_common import get_memory as _get_memory
from memo.config import Config

# -- proactive suggestions commands ---------------------------------------------


@click.group(name="suggest")
def suggest_group() -> None:
    """Proactive memory suggestions from conversation analysis."""
    pass


@suggest_group.command(name="analyze")
@click.argument("transcript_path", type=click.Path(exists=True))
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON")
def suggest_analyze(transcript_path: str, as_json: bool) -> None:
    """Analyze a transcript and suggest memories to save.

    Example: memo suggest analyze /path/to/transcript.jsonl
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    from pathlib import Path

    from memo.capture import _read_last_exchange

    # For now, just use the last exchange as a sample
    # In a full implementation, would analyze the full transcript
    pair = _read_last_exchange(Path(transcript_path))
    if pair is None:
        console.print("[yellow]Could not read transcript[/yellow]")
        return

    user_text, assistant_text = pair
    turns = [{"user": user_text, "assistant": assistant_text}]

    suggestions = mem.proactive.analyze_conversation(turns, limit=3)

    if as_json:
        click.echo(json.dumps([s.__dict__ for s in suggestions], indent=2))
        return

    if not suggestions:
        console.print("[dim]No suggestions found[/dim]")
        return

    console.print(f"[bold]Found {len(suggestions)} suggestions[/bold]")
    console.print()

    for i, s in enumerate(suggestions, 1):
        console.print(f"[cyan]{i}. {s.title}[/cyan]")
        console.print(f"   Type: {s.type}")
        console.print(f"   Confidence: {s.confidence:.2f}")
        console.print(f"   Tags: {', '.join(s.tags)}")
        console.print(f"   Rationale: {s.rationale}")
        console.print(f"   Snippet: {s.body_snippet[:100]}")
        console.print()


@suggest_group.command(name="feedback-stats")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON")
def suggest_feedback_stats(as_json: bool) -> None:
    """Show statistics on suggestion feedback (acceptance rate).

    Example: memo suggest feedback-stats
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    stats = mem.proactive.get_feedback_stats()

    if as_json:
        click.echo(json.dumps(stats, indent=2))
        return

    console.print("[bold]Suggestion Feedback Stats[/bold]")
    console.print()
    console.print(f"Total suggestions: {stats['total']}")
    console.print(f"Accepted: {stats['accepted']}")
    console.print(f"Rejected: {stats['rejected']}")
    console.print(f"Acceptance rate: {stats['acceptance_rate']:.2%}")


@suggest_group.command(name="patterns")
@click.argument("transcript_path", type=click.Path(exists=True))
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON")
def suggest_patterns(transcript_path: str, as_json: bool) -> None:
    """Detect patterns in a transcript (recurring themes, decisions, etc.).

    Example: memo suggest patterns /path/to/transcript.jsonl
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    from pathlib import Path

    patterns = mem.proactive.detect_patterns(Path(transcript_path))

    if as_json:
        click.echo(json.dumps(patterns, indent=2))
        return

    console.print("[bold]Conversation Patterns[/bold]")
    console.print()
    console.print(f"Total turns: {patterns['total_turns']}")
    console.print(f"Decision points: {patterns['decision_points']}")
    console.print(f"Technical discoveries: {patterns['technical_discoveries']}")
    console.print(f"Recurring themes: {', '.join(patterns['recurring_themes'])}")
