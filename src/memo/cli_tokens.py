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

    def cell(label: str, tokens: int, grounded: int, color: str) -> Text:
        t = Text(justify="center")
        t.append(f"{label}\n", style="dim")
        t.append(f"{_fmt_tokens(tokens)}\n", style=f"bold {color}")
        t.append("tokens saved\n", style="dim")
        t.append(f"{grounded} memories used", style="dim")
        return t

    grid.add_row(
        cell("TODAY", s["today"]["tokens"], s["today"]["grounded"], "cyan"),
        cell(
            f"MONTH · {s['month']['month']}",
            s["month"]["tokens"],
            s["month"]["grounded"],
            "magenta",
        ),
        cell("ALL-TIME", s["historic"]["tokens"], s["historic"]["grounded"], "green"),
    )
    return Panel(
        grid,
        title="[bold]memo · tokens saved[/bold]",
        subtitle="[dim]memo-only savings (memories the answer actually used)[/dim]",
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
        s["measured"] = measured          # additive key; existing keys untouched
        click.echo(_json.dumps(s, ensure_ascii=False, indent=2))
        return

    # Print measured panel BEFORE the early-return guard so it shows even when
    # token_ledger is empty (no grounding yet).
    if measured["sessions"]:
        p = measured["proxy"]
        delta = p["delta"]
        proxy_line = (
            f"tool-spend grounded {p['grounded_tool_tok_per_turn']} vs "
            f"ungrounded {p['ungrounded_tool_tok_per_turn']} tok/turn"
            + (f"  (Δ {delta:+.0f})" if delta is not None else "")
            if p["grounded_tool_tok_per_turn"] is not None and p["ungrounded_tool_tok_per_turn"] is not None
            else "proxy: no grounded+ungrounded sessions to compare yet"
        )
        console.print(Panel(
            Text.from_markup(
                f"[bold]{_fmt_tokens(measured['answer_tok'])}[/bold] tok answer · "
                f"[bold]{_fmt_tokens(measured['tool_tok'])}[/bold] tok tool-loops · "
                f"[bold]{_fmt_tokens(measured['injected_tokens'])}[/bold] tok injected "
                f"([dim]{measured['sessions']} measured sessions[/dim])\n[dim]{proxy_line}[/dim]"
            ),
            title="[bold]memo · measured (real transcript)[/bold]",
            border_style="green", padding=(0, 2),
        ))

    if s["historic"]["grounded"] == 0:
        console.print(
            "[dim]No memories used in answers yet (grounding.log empty).\n"
            "Run a few Claude Code sessions: each memory the answer\n"
            "uses adds to the savings. Check the hook with [/dim][bold]memo doctor[/bold][dim].[/dim]"
        )
        return

    console.print(_header_panel(s))

    daily_rows = [
        (d["date"][5:], d["grounded"], d["tokens"]) for d in s["daily"]  # MM-DD
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
        f"more useful memories ⇒ more memories used ⇒ more savings[/dim]"
    )
