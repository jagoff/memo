"""`memo token-savings` — show recall injection stats and token-saving techniques.

Read-only over context_cost.log; prints a human-readable summary of how
much context space recall injections consumed and what techniques reduce it.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import click

from .config import Config
from .dashboard import read_context_cost_log


@click.command(name="token-savings")
def token_savings_cmd() -> None:
    """Show recall injection stats and token economy tips (last 7 days)."""
    cfg = Config.from_env()
    entries = read_context_cost_log(cfg.state_dir)

    cutoff = datetime.now(UTC) - timedelta(days=7)
    recall_entries = [
        e
        for e in entries
        if e.get("kind") == "recall"
        and _parse_ts(e.get("ts", "")) >= cutoff
    ]

    if not recall_entries:
        click.echo(
            "No recall injections logged yet."
            " Run memo doctor to verify the recall hook is active."
        )
        return

    n_injections = len(recall_entries)
    total_chars = sum(e.get("chars", 0) for e in recall_entries)
    avg_chars = total_chars // n_injections if n_injections else 0

    compact_savings_pct = 65
    compact_chars_saved = int(total_chars * compact_savings_pct / 100)
    compact_tokens_saved = compact_chars_saved // 4

    click.echo("memo token savings (last 7 days)")
    click.echo("")
    click.echo(f"  Recall injections:    {n_injections:,} prompts")
    click.echo(f"  Avg context chars:    {avg_chars:,} per injection")
    click.echo("")
    click.echo("  Compact format savings (MEMO_RECALL_FORMAT=compact):")
    click.echo(f"    ~{compact_savings_pct}% fewer chars  →  ~{compact_tokens_saved:,} tokens saved")
    click.echo("")
    click.echo("  Trivial bail gate (on by default):")
    click.echo("    Skips ~25% of confirmation prompts automatically")
    click.echo("")
    click.echo("  compress-context (one-time):")
    click.echo("    Run: memo compress-context CLAUDE.md")
    click.echo("    Typical saving: 30-40% of context file size")
    click.echo("")
    click.echo("  Enable compact: export MEMO_RECALL_FORMAT=compact")


def _parse_ts(ts: str) -> datetime:
    """Parse an ISO timestamp string; return epoch on failure."""
    try:
        return datetime.fromisoformat(ts).replace(tzinfo=UTC) if ts else datetime(1970, 1, 1, tzinfo=UTC)
    except ValueError:
        return datetime(1970, 1, 1, tzinfo=UTC)
