"""`memo tokens` — TUI showing tokens saved by memo (today / month / historic).

Rolls up the durable token-savings ledger (see :mod:`memo.token_ledger`) then
renders the headline numbers plus a daily and a monthly bar chart. The signal is
*grounded* recalls — surfaced memories the answer actually USED, i.e.
re-derivations memo prevented — so the total rises as memo accumulates more
useful memories. Read-only; `--json` for machine output.
"""

from __future__ import annotations

import json as _json

import click
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from memo import token_ledger
from memo.cli_common import console
from memo.config import Config

_BLOCKS = "▏▎▍▌▋▊▉█"

# Turns needed in BOTH cohorts before the measured net reads as evidence rather
# than as noise. Neither cohort is assigned — a user does not run sessions with
# recall off on purpose — so a handful of ungrounded turns can swing the sign.
_MIN_COHORT_TURNS = 30


def _fmt_tokens(tokens: float) -> str:
    tokens = int(tokens)
    if tokens < 1000:
        return f"{tokens}"
    if tokens < 1_000_000:
        return f"{tokens / 1000:.1f}k"
    return f"{tokens / 1_000_000:.2f}M"


def _bar(value: float, vmax: float, width: int = 22) -> str:
    """Eighth-block horizontal bar scaled to ``vmax``."""
    if vmax <= 0 or value <= 0:
        return ""
    units = value / vmax * width
    full = int(units)
    bar = "█" * min(full, width)
    rem = units - full
    if rem > 0 and full < width:
        bar += _BLOCKS[min(7, int(rem * 8))]
    return bar


def _header_panel(s: dict) -> Panel:
    grid = Table.grid(expand=True)
    grid.add_column(justify="center", ratio=1)
    grid.add_column(justify="center", ratio=1)
    grid.add_column(justify="center", ratio=1)

    def cell(label: str, bucket: dict, color: str) -> Text:
        t = Text(justify="center")
        t.append(f"{label}\n", style="dim")
        t.append(f"{_fmt_tokens(bucket['tokens'])}\n", style=f"bold {color}")
        t.append("tokens saved (est.)\n", style="dim")
        used = f"{bucket['grounded']} used"
        consults = int(bucket.get("consults", 0))
        if consults:
            used += f" · {consults} consults"
        t.append(used, style="dim")
        return t

    grid.add_row(
        cell("TODAY", s["today"], "cyan"),
        cell(f"MONTH · {s['month']['month']}", s["month"], "magenta"),
        cell("ALL-TIME", s["historic"], "green"),
    )
    return Panel(
        grid,
        title="[bold]memo · tokens saved (estimated)[/bold]",
        subtitle=(
            "[dim]per-grounded-recall estimate — the measured net is in the panel above[/dim]"
        ),
        border_style="bright_blue",
        padding=(1, 2),
    )


def _chart(rows: list[tuple[str, int, int]], color: str, title: str) -> Panel:
    """rows = [(label, grounded, tokens)]; bar scaled to the max tokens."""
    vmax = max((tok for _, _, tok in rows), default=0)
    tbl = Table.grid(padding=(0, 1))
    tbl.add_column(justify="right", style="dim")  # label
    tbl.add_column()  # bar
    tbl.add_column(justify="right")  # number
    for label, _g, tok in rows:
        bar = Text(_bar(tok, vmax), style=color)
        num = Text(_fmt_tokens(tok) if tok else "·", style="" if tok else "dim")
        tbl.add_row(label, bar, num)
    return Panel(tbl, title=f"[bold {color}]{title}[/bold {color}]", border_style=color)


def _by_client_panel(s: dict) -> Panel | None:
    """All-time savings attributed to each agent that reads memo — Claude Code
    (grounded) plus every other LLM (codex/opencode/devin/other agents)
    measured by its productive consults."""
    bc = s.get("by_client", {}).get("historic", {})
    if not bc:
        return None
    tbl = Table.grid(padding=(0, 2))
    tbl.add_column(style="bold")  # agent
    tbl.add_column(justify="right")  # tokens
    tbl.add_column(justify="left", style="dim")  # signal
    for client, rec in bc.items():
        signal = []
        if rec.get("grounded"):
            signal.append(f"{rec['grounded']} grounded")
        if rec.get("consults"):
            signal.append(f"{rec['consults']} consults")
        tbl.add_row(
            client,
            Text(_fmt_tokens(rec["tokens"]) + " tok", style="green"),
            " · ".join(signal),
        )
    return Panel(
        tbl,
        title="[bold]by agent · all-time[/bold]",
        subtitle="[dim]every LLM that reads memo — grounded (Claude Code) + consults (others)[/dim]",
        border_style="blue",
        padding=(0, 2),
    )


def _transcript_side_line(measured: dict) -> str:
    """The measured prompt-side surface (ccusage accounting): input footprint,
    cache-read/created volumes, and per-model output spend. Empty when the
    transcripts never carried the richer `usage` fields."""
    parts = []
    if int(measured.get("input_tok", 0)) > 0:
        parts.append(
            f"[bold]{_fmt_tokens(measured['input_tok'])}[/bold] tok input footprint "
            f"([dim]max prompt[/dim])"
        )
        if int(measured.get("cache_read_tok", 0)) > 0:
            parts.append(f"[bold]{_fmt_tokens(measured['cache_read_tok'])}[/bold] tok cache-read")
        if int(measured.get("cache_creation_tok", 0)) > 0:
            parts.append(
                f"[bold]{_fmt_tokens(measured['cache_creation_tok'])}[/bold] tok cache-written"
            )
    models = measured.get("models") or {}
    if models:
        parts.append(
            "by model: "
            + " · ".join(
                f"[bold]{name}[/bold] {_fmt_tokens(out)}" for name, out in models.items()
            )
        )
    return "\n" + " · ".join(parts) if parts else ""


def _growth_text(s: dict) -> Text:
    g = s["growth"]
    t = Text()
    if g["up"] is None:
        t.append("growth: no prior month to compare yet", style="dim")
    else:
        arrow = "▲" if g["up"] else "▼"
        style = "green" if g["up"] else "yellow"
        pct = g["pct"]
        t.append(f"{arrow} {abs(pct):.0f}% ", style=f"bold {style}")
        t.append("vs prior month  ", style="dim")
        t.append(
            f"({_fmt_tokens(g['prev_month_tokens'])} → {_fmt_tokens(g['this_month_tokens'])} tok)",
            style="dim",
        )
    return t


@click.command(name="tokens")
@click.option("--days", default=14, show_default=True, help="Days in the daily chart.")
@click.option("--months", default=6, show_default=True, help="Months in the monthly chart.")
@click.option("--json", "as_json", is_flag=True, help="Machine-readable output.")
def tokens_cmd(*, days: int = 14, months: int = 6, as_json: bool = False) -> None:
    """Show how many tokens memo saved: today, this month, and all-time total."""
    from memo import token_meter

    cfg = Config.from_env()
    token_ledger.roll_up(cfg.state_dir)
    s = token_ledger.summarize(cfg.state_dir, days_back=days, months_back=months)

    # Measured surface (additive; the token_ledger schema above stays frozen).
    measured = token_meter.summarize(cfg.state_dir)

    if as_json:
        s["measured"] = measured  # additive key; existing keys untouched
        click.echo(_json.dumps(s, ensure_ascii=False, indent=2))
        return

    # Print measured panel BEFORE the early-return guard so it shows even when
    # token_ledger is empty (no grounding yet).
    if measured["sessions"]:
        p = measured["proxy"]
        delta = p["delta"]
        net = p["net_tok_per_turn"]
        have_both = (
            p["grounded_tool_tok_per_turn"] is not None
            and p["ungrounded_tool_tok_per_turn"] is not None
        )
        proxy_line = (
            f"tool-spend grounded {p['grounded_tool_tok_per_turn']} vs "
            f"ungrounded {p['ungrounded_tool_tok_per_turn']} tok/turn"
            + (f"  (Δ {delta:+.0f})" if delta is not None else "")
            + f"  -{p['injected_tok_per_turn']:g} injected"
            if have_both
            else "proxy: no grounded+ungrounded sessions to compare yet"
        )
        # The net line leads because it is the only measured answer to "does
        # memo save tokens" — tool-spend delta alone ignores the context memo
        # injects to earn it. Negative means memo cost more than it saved; that
        # is a real result and it is reported as one.
        if net is None:
            net_line = "[dim]net: needs grounded and ungrounded sessions to compare[/dim]"
        else:
            colour = "green" if net > 0 else "red"
            verb = "saved" if net > 0 else "cost"
            # Both cohorts are observational, not assigned: a thin one means the
            # sign is not yet evidence. Say so rather than let the number stand
            # unqualified.
            thin = min(p["grounded_turns"], p["ungrounded_turns"]) < _MIN_COHORT_TURNS
            net_line = (
                f"[bold {colour}]{abs(net):,.0f} tok/turn {verb}[/bold {colour}] net of injection "
                f"[dim](n={p['grounded_turns']} grounded / {p['ungrounded_turns']} ungrounded "
                f"turns{' · provisional, thin cohort' if thin else ''})[/dim]"
            )
        console.print(
            Panel(
                Text.from_markup(
                    f"{net_line}\n"
                    f"[bold]{_fmt_tokens(measured['answer_tok'])}[/bold] tok answer · "
                    f"[bold]{_fmt_tokens(measured['tool_tok'])}[/bold] tok tool-loops · "
                    f"[bold]{_fmt_tokens(measured['injected_tokens'])}[/bold] tok injected "
                    f"([dim]{measured['sessions']} measured sessions[/dim])"
                    + _transcript_side_line(measured)
                    + f"\n[dim]{proxy_line}[/dim]"
                ),
                title="[bold]memo · measured (real transcript)[/bold]",
                border_style="green",
                padding=(0, 2),
            )
        )

    if s["historic"]["tokens"] == 0:
        console.print(
            "[dim]No savings recorded yet (no grounded Claude answers, no\n"
            "productive consults from other agents). Run a few sessions —\n"
            "each memory an answer uses, or each memo search another agent\n"
            "makes, adds to the total. Check the hook with [/dim][bold]memo doctor[/bold][dim].[/dim]"
        )
        return

    console.print(_header_panel(s))

    agent_panel = _by_client_panel(s)
    if agent_panel is not None:
        console.print(agent_panel)

    daily_rows = [
        (d["date"][5:], d["grounded"], d["tokens"])
        for d in s["daily"]  # MM-DD
    ]
    console.print(_chart(daily_rows, "cyan", f"last {days} days · tokens/day"))

    if s["monthly"]:
        monthly_rows = [(m["month"], m["grounded"], m["tokens"]) for m in s["monthly"]]
        console.print(_chart(monthly_rows, "magenta", "by month · tokens/month"))

    console.print(_growth_text(s))
    abl = s.get("ablation") or {}
    if abl.get("turns_off"):
        console.print(
            f"[dim]ablation: {abl['turns_on']} turns with memo · "
            f"{abl['turns_off']} without memo (MEMO_RECALL_DISABLE) — see `memo roi`[/dim]"
        )
    console.print(
        f"[dim]est: {s['tpg']} tok/memory-used (MEMO_ROI_TOKENS_PER_GROUNDED) · "
        f"{s.get('tpc', 200)} tok/consult by other agents (MEMO_ROI_TOKENS_PER_CONSULT)[/dim]"
    )
