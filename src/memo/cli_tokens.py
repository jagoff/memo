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
        t.append("tokens ahorrados\n", style="dim")
        t.append(f"{grounded} recuerdos usados", style="dim")
        return t

    grid.add_row(
        cell("HOY", s["today"]["tokens"], s["today"]["grounded"], "cyan"),
        cell(
            f"MES · {s['month']['month']}",
            s["month"]["tokens"],
            s["month"]["grounded"],
            "magenta",
        ),
        cell("HISTÓRICO", s["historic"]["tokens"], s["historic"]["grounded"], "green"),
    )
    return Panel(
        grid,
        title="[bold]memo · tokens ahorrados[/bold]",
        subtitle="[dim]ahorro exclusivo de memo (recuerdos que la respuesta usó)[/dim]",
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
        t.append("crecimiento: sin mes previo para comparar todavía", style="dim")
    else:
        arrow = "▲" if g["up"] else "▼"
        style = "green" if g["up"] else "yellow"
        pct = g["pct"]
        t.append(f"{arrow} {abs(pct):.0f}% ", style=f"bold {style}")
        t.append("vs mes anterior  ", style="dim")
        t.append(
            f"({_fmt_tokens(g['prev_month_tokens'])} → {_fmt_tokens(g['this_month_tokens'])} tok)",
            style="dim",
        )
    return t


@click.command(name="tokens")
@click.option("--days", default=14, show_default=True, help="Días en el gráfico diario.")
@click.option("--months", default=6, show_default=True, help="Meses en el gráfico mensual.")
@click.option("--json", "as_json", is_flag=True, help="Salida machine-readable.")
def tokens_cmd(*, days: int = 14, months: int = 6, as_json: bool = False) -> None:
    """Mostrar cuántos tokens ahorró memo: hoy, este mes y total histórico."""
    cfg = Config.from_env()
    token_ledger.roll_up(cfg.state_dir)
    s = token_ledger.summarize(cfg.state_dir, days_back=days, months_back=months)

    if as_json:
        click.echo(_json.dumps(s, ensure_ascii=False, indent=2))
        return

    if s["historic"]["grounded"] == 0:
        console.print(
            "[dim]Todavía no hay recuerdos usados en respuestas (grounding.log vacío).\n"
            "Corré algunas sesiones de Claude Code: cada recuerdo que la respuesta\n"
            "use suma al ahorro. Verificá el hook con [/dim][bold]memo doctor[/bold][dim].[/dim]"
        )
        return

    console.print(_header_panel(s))

    daily_rows = [
        (d["date"][5:], d["grounded"], d["tokens"]) for d in s["daily"]  # MM-DD
    ]
    console.print(_chart(daily_rows, "cyan", f"últimos {days} días · tokens/día"))

    if s["monthly"]:
        monthly_rows = [(m["month"], m["grounded"], m["tokens"]) for m in s["monthly"]]
        console.print(_chart(monthly_rows, "magenta", "por mes · tokens/mes"))

    console.print(_growth_text(s))
    abl = s.get("ablation") or {}
    if abl.get("turns_off"):
        console.print(
            f"[dim]ablación: {abl['turns_on']} turnos con memo · "
            f"{abl['turns_off']} sin memo (MEMO_RECALL_DISABLE) — ver `memo roi`[/dim]"
        )
    console.print(
        f"[dim]est: {s['tpg']} tok/recuerdo-usado (MEMO_ROI_TOKENS_PER_GROUNDED) · "
        f"más memorias útiles ⇒ más recuerdos usados ⇒ más ahorro[/dim]"
    )
