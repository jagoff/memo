"""`memo token-savings` — show recall injection stats and token-saving techniques.

Read-only over context_cost.log and recall_hook.log; prints a brief table of
how much context space recall injections consumed, how many trivial prompts were
skipped, and an estimated total tokens saved.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import click

from .config import Config
from .dashboard import read_context_cost_log, read_recall_log
from .flags import flag_str


@click.command(name="token-savings")
def token_savings_cmd() -> None:
    """Show recall injection stats and token economy tips (last 7 days)."""
    cfg = Config.from_env()
    cost_log = read_context_cost_log(cfg.state_dir)
    hook_log = read_recall_log(cfg.state_dir, limit=4000)

    cutoff = datetime.now(UTC) - timedelta(days=7)
    recall_entries = [
        e
        for e in cost_log
        if e.get("kind") == "recall"
        and _parse_ts(e.get("ts", "")) >= cutoff
    ]
    trivial_bails = sum(
        1
        for e in hook_log
        if e.get("via") == "bail"
        and e.get("reason") == "trivial prompt"
        and _parse_ts(e.get("ts", "")) >= cutoff
    )

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

    avg_tokens_per_injection = avg_chars // 4 if avg_chars else 40
    recall_format = flag_str("MEMO_RECALL_FORMAT")
    is_compact = recall_format == "compact"
    if is_compact:
        total_tokens_saved = compact_tokens_saved + trivial_bails * avg_tokens_per_injection
    else:
        compact_tokens_saved = 0
        total_tokens_saved = trivial_bails * avg_tokens_per_injection

    click.echo("memo token savings (last 7 days)")
    click.echo("")
    click.echo(f"  Recall injections:  {n_injections:,} prompts")
    click.echo(f"  Context chars:      {avg_chars:,} avg per injection")
    if is_compact:
        click.echo(f"  Compact savings:    ~{compact_savings_pct}%  (active)")
    else:
        click.echo(
            f"  Compact savings:    ~{compact_savings_pct}% potential"
            "  (enable: export MEMO_RECALL_FORMAT=compact)"
        )
    click.echo(f"  Trivial bails:      {trivial_bails}  (prompts skipped)")
    click.echo("")
    click.echo(f"  Estimated total:    ~{total_tokens_saved:,} tokens saved vs. model rederiving context")
    click.echo("")
    if not is_compact:
        click.echo("  Enable compact: export MEMO_RECALL_FORMAT=compact")
    click.echo("  Run:            memo compress-context CLAUDE.md  (one-time context file shrink)")


def _parse_ts(ts: str) -> datetime:
    """Parse an ISO timestamp string; return epoch on failure."""
    try:
        return datetime.fromisoformat(ts).replace(tzinfo=UTC) if ts else datetime(1970, 1, 1, tzinfo=UTC)
    except ValueError:
        return datetime(1970, 1, 1, tzinfo=UTC)
