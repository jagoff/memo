"""`memo tokens` — memo's real, measured token cost/savings. No estimates.

Two independently measured surfaces, both sourced from real provider `usage`
(never a hardcoded per-event constant):

  * **transcript** (:mod:`memo.token_meter`) — per-session accounting read
    from the Claude Code transcript: answer/tool-loop token spend, grounded
    vs ungrounded tool-loop delta, and the context memo injects to earn it.
  * **proxy holdout** (:mod:`memo.proxy.meter`) — the local context-compression
    proxy assigns each request to a treated or an uncompressed holdout arm and
    records the real `input_tokens` the provider billed, so the saving
    fraction is a measured A/B result, not an estimate.

An earlier version of this command additionally printed a "tokens saved
(estimated)" panel — `grounded × MEMO_ROI_TOKENS_PER_GROUNDED (350) +
consults × MEMO_ROI_TOKENS_PER_CONSULT (200)`, hardcoded constants with no
control arm. That panel stood beside the measured cost and read as a savings
claim memo could not support, so it was removed (see CHANGELOG). "No
measured data yet" and "measured zero savings" are different claims — this
command never prints a zero in place of missing data.
"""

from __future__ import annotations

import json as _json

import click
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from memo.cli_common import console
from memo.config import Config

# Turns needed in BOTH cohorts before the measured net reads as evidence rather
# than as noise. Neither cohort is assigned — a user does not run sessions with
# recall off on purpose — so a handful of ungrounded turns can swing the sign.
_MIN_COHORT_TURNS = 30

# Requests needed in BOTH the treated and holdout arms before the proxy's
# saving fraction reads as evidence rather than noise — same floor as the
# transcript panel's cohort, for the same reason: a handful of requests can
# swing a ratio wildly (n=1 vs n=1 can print a two-digit swing in bold colour
# with nothing marking it as unproven).
_MIN_PROXY_SAMPLE = 30


def _fmt_tokens(tokens: float) -> str:
    tokens = int(tokens)
    if tokens < 1000:
        return f"{tokens}"
    if tokens < 1_000_000:
        return f"{tokens / 1000:.1f}k"
    return f"{tokens / 1_000_000:.2f}M"


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
            + " · ".join(f"[bold]{name}[/bold] {_fmt_tokens(out)}" for name, out in models.items())
        )
    return "\n" + " · ".join(parts) if parts else ""


def _transcript_panel(measured: dict) -> Panel:
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
    return Panel(
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


def _proxy_panel(proxy: dict) -> Panel:
    frac = proxy["measured_saving_frac"]
    n_t, n_h = proxy["n_treated"], proxy["n_holdout"]
    if frac is None:
        if n_t or n_h:
            body = (
                f"[dim]not enough data to compare arms yet[/dim] "
                f"([dim]{n_t} treated / {n_h} holdout requests[/dim])"
            )
        else:
            body = "[dim]no measured data yet — the proxy has not logged any requests[/dim]"
    else:
        # frac >= 0 reads as "saved": an exact-zero delta is a null result
        # (nothing lost, nothing gained), not a regression — only a real
        # negative frac (treated arm costs MORE than holdout) is "cost".
        colour = "green" if frac >= 0 else "red"
        verb = "saved" if frac >= 0 else "cost"
        mean_t = proxy["mean_input_treated"]
        mean_h = proxy["mean_input_holdout"]
        thin = min(n_t, n_h) < _MIN_PROXY_SAMPLE
        body = (
            f"[bold {colour}]{abs(frac) * 100:.1f}% {verb}[/bold {colour}] on input tokens "
            f"(treated {mean_t:.0f} vs holdout {mean_h:.0f} tok/request) "
            f"[dim](n={n_t} treated / {n_h} holdout requests"
            f"{' · provisional, thin sample' if thin else ''})[/dim]"
        )
    if proxy.get("retrieved"):
        body += f"\n[dim]{proxy['retrieved']} recovered originals (cost their tokens twice)[/dim]"
    if proxy.get("n_passthrough"):
        body += (
            f"\n[dim]{proxy['n_passthrough']} passthrough request(s) excluded from both arms "
            "(proxy was disabled for them, so nothing was rewritten)[/dim]"
        )
    return Panel(
        Text.from_markup(body),
        title="[bold]memo · proxy (real holdout A/B)[/bold]",
        border_style="blue",
        padding=(0, 2),
    )


def _by_transform_table(proxy: dict) -> Table:
    tbl = Table(title="by transform", show_lines=False)
    tbl.add_column("transform")
    tbl.add_column("requests", justify="right")
    tbl.add_column("share of savings", justify="right")
    for name, stats in sorted(
        proxy["by_transform"].items(), key=lambda kv: kv[1]["n"], reverse=True
    ):
        share = stats.get("share")
        share_s = f"{share * 100:.0f}%" if share is not None else "—"
        tbl.add_row(name, str(stats["n"]), share_s)
    return tbl


@click.command(name="tokens")
@click.option(
    "--by-transform",
    "by_transform",
    is_flag=True,
    help="Break the measured proxy saving down per transform.",
)
@click.option("--json", "as_json", is_flag=True, help="Machine-readable output.")
def tokens_cmd(*, by_transform: bool = False, as_json: bool = False) -> None:
    """Show memo's measured token cost/savings — real transcript + real proxy holdout."""
    from memo import token_meter
    from memo.proxy import meter as proxy_meter

    cfg = Config.from_env()
    measured = token_meter.summarize(cfg.state_dir)
    proxy = proxy_meter.summarize(cfg.state_dir)

    if as_json:
        click.echo(
            _json.dumps({"measured": measured, "proxy": proxy}, ensure_ascii=False, indent=2)
        )
        return

    has_data = bool(measured["sessions"]) or proxy["n_treated"] or proxy["n_holdout"]
    if not has_data:
        console.print(
            "[dim]No measured data yet (no Claude Code sessions rolled up, no proxy "
            "requests logged). Run a few sessions through the proxy — each request it "
            "sees, treated or held out, adds to the comparison. Check the hook with "
            "[/dim][bold]memo doctor[/bold][dim].[/dim]"
        )
        return

    if measured["sessions"]:
        console.print(_transcript_panel(measured))

    console.print(_proxy_panel(proxy))

    if by_transform:
        if proxy["by_transform"]:
            console.print(_by_transform_table(proxy))
        else:
            console.print("[dim]no per-transform data yet[/dim]")
