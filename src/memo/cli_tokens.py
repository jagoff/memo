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
# Distinct sessions required per arm. Arms are assigned per session, so this is
# the count of independent draws; the request floor above only bounds how much
# evidence each draw carries. Three is the smallest number that cannot be one
# atypical session plus noise.
_MIN_PROXY_SESSIONS = 3


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
    transcripts never carried the richer `usage` fields.

    The three `*_tok` fields are independent measurements and must be gated
    independently: `input_tok` is `token_meter._transcript_input_side`'s
    per-session PEAK of the Messages API's `input_tokens` field, summed
    across sessions by `summarize()` -- and `input_tokens` itself is the
    UNCACHED remainder of a call's prompt, not its full size (real, billed
    volume either way, but not "the prompt"; see the identical finding
    already fixed on the proxy A/B side, `proxy/meter.py`'s prompt-cost
    weighting). `cache_read_tok`/`cache_creation_tok` are the two counters
    that actually carry a cache-heavy session's real prompt volume, and used
    to be nested inside `input_tok > 0` -- so a session with a legitimately
    (or a `_transcript_input_side`-zeroed, degenerate-build) small
    `input_tok` hid real, non-zero cache volumes entirely.
    """
    parts = []
    if int(measured.get("input_tok", 0)) > 0:
        parts.append(
            f"[bold]{_fmt_tokens(measured['input_tok'])}[/bold] tok input footprint "
            f"([dim]uncached only, summed peaks -- not full prompt[/dim])"
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
    # Both cohorts are observational, not assigned, and they are confounded by
    # session length: a session that never grounds is usually a session that
    # barely started. Measured 2026-08-31 on the live ledger — 100 grounded
    # sessions (median 15 turns) against an "ungrounded" control of 3 sessions
    # totalling 9 turns, while the 9 sessions in NEITHER cohort, equally short,
    # spent MORE per turn than the grounded ones. So the sign was not evidence
    # in either direction, and the panel printed it in bold anyway.
    #
    # `_proxy_panel` withholds below its floor for exactly this reason, three
    # screens-worth of code below, with the note that a sample too thin to
    # compare is not a measurement with a caveat — it is not a measurement.
    # Applying that standard to one panel and a "provisional" adjective to the
    # other, on the same screen, is the inconsistency this removes.
    thin = min(p["grounded_turns"], p["ungrounded_turns"]) < _MIN_COHORT_TURNS
    if net is None:
        net_line = "[dim]net: needs grounded and ungrounded sessions to compare[/dim]"
    elif thin:
        net_line = (
            "[dim]net: not enough data to compare cohorts yet "
            f"({p['grounded_turns']} grounded / {p['ungrounded_turns']} ungrounded turns "
            f"— need {_MIN_COHORT_TURNS} per cohort)[/dim]"
        )
    else:
        colour = "green" if net > 0 else "red"
        verb = "saved" if net > 0 else "cost"
        net_line = (
            f"[bold {colour}]{abs(net):,.0f} tok/turn {verb}[/bold {colour}] net of injection "
            f"[dim](n={p['grounded_turns']} grounded / {p['ungrounded_turns']} ungrounded "
            f"turns)[/dim]"
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
    # Defect 2: holdout is assigned per SESSION, not per request, so n_t/n_h
    # (request counts) overstate the effective sample size — every request in
    # one holdout session is correlated, not an independent draw. Report the
    # distinct-session count beside the request count rather than hiding that
    # the unit changed.
    n_ts = proxy.get("n_treated_sessions") or 0
    n_hs = proxy.get("n_holdout_sessions") or 0
    sessions_note = f", {n_ts}/{n_hs} sessions" if (n_ts or n_hs) else ""
    # A sample too thin to compare is not a measurement with a caveat — it is
    # not a measurement. Because the control arm is drawn per session at
    # MEMO_PROXY_HOLDOUT_FRAC, it fills ~1/frac times slower than the treated
    # one, and its first entries can be a single atypical session: live
    # traffic once put 2 holdout requests worth 5 tok-equiv against 4984
    # treated worth 19320 and rendered "386295.8% cost" in bold. Withhold the
    # ratio below the floor instead of shipping it with a "provisional" word.
    # …and the request floor alone cannot catch that shape: one holdout session
    # of 400 requests clears `n_h >= 30` while still being a single draw. The
    # session counts computed just above are what the arms are actually drawn
    # in, so they gate too — otherwise the floor withholds the small version of
    # the bad number and ships the large one.
    thin = min(n_t, n_h) < _MIN_PROXY_SAMPLE or min(n_ts, n_hs) < _MIN_PROXY_SESSIONS
    if frac is None or thin:
        if n_t or n_h:
            body = (
                f"[dim]not enough data to compare arms yet[/dim] "
                f"([dim]{n_t} treated / {n_h} holdout requests{sessions_note}"
                f" — need {_MIN_PROXY_SAMPLE} requests and"
                f" {_MIN_PROXY_SESSIONS} sessions per arm[/dim])"
            )
        else:
            body = "[dim]no measured data yet — the proxy has not logged any requests[/dim]"
    else:
        # frac >= 0 reads as "saved": an exact-zero delta is a null result
        # (nothing lost, nothing gained), not a regression — only a real
        # negative frac (treated arm costs MORE than holdout) is "cost".
        colour = "green" if frac >= 0 else "red"
        verb = "saved" if frac >= 0 else "cost"
        # The headline is now a weighted PROMPT COST ratio (input_tokens +
        # cache-creation weighted BY TIER, 1.25x for a 5m write and 2x for a
        # 1h one, + 0.1x cache-read — see meter.py's weight constants), not a
        # bare input_tokens ratio: real traffic bills
        # almost the whole prompt through the two cache counters, so
        # input_tokens alone is a ratio of noise. The raw input_tokens
        # means stay visible alongside it.
        cost_t = proxy["mean_prompt_cost_treated"]
        cost_h = proxy["mean_prompt_cost_holdout"]
        mean_in_t = proxy["mean_input_treated"]
        mean_in_h = proxy["mean_input_holdout"]
        body = (
            f"[bold {colour}]{abs(frac) * 100:.1f}% {verb}[/bold {colour}] on prompt cost "
            f"(treated {cost_t:.0f} vs holdout {cost_h:.0f} tok-equiv/request; "
            f"raw input_tokens {mean_in_t:.0f} vs {mean_in_h:.0f}) "
            f"[dim](n={n_t} treated / {n_h} holdout requests{sessions_note})[/dim]"
        )
    # An A/B that is still filling its control arm must not read as a measured
    # zero. The counterfactual needs no control arm (see
    # `meter._prefix_counterfactual`) and is labelled so it can never be
    # mistaken for the arm comparison above it.
    cf = proxy.get("prefix_counterfactual")
    if cf and cf.get("saving_frac") is not None:
        body += (
            f"\n[dim]prefix counterfactual (not an A/B): "
            f"[/dim][bold]{cf['saving_frac'] * 100:.1f}%[/bold][dim] — "
            f"{_fmt_tokens(cf['removed_tok'])} tok of prefix removed over {cf['n']} requests, "
            f"priced at the {cf['effective_prefix_weight']:.3f}x weight the surviving "
            f"prefix actually billed at[/dim]"
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
    # `saved_by` is an unweighted chars/4 diff of the text a transform removed.
    # The proxy panel's headline saving is a cache-weighted BILLED ratio, and
    # the two are different currencies: prefix content a transform removes is
    # credited here at full value every turn while the counterfactual bills it
    # at the 0.1x cache-read weight. Naming the unit keeps the reader from
    # reading this table as a decomposition of the billed saving.
    tbl = Table(title="by transform — raw text removed, not billed saving", show_lines=False)
    tbl.add_column("transform")
    tbl.add_column("requests", justify="right")
    tbl.add_column("share of text removed", justify="right")
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
    help="Break down which transforms removed the most text (raw, not billed cost).",
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
