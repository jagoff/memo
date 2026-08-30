"""`memo token-savings` — show measured per-lever token savings.

Reads `state_dir/eval/token_baseline.json` (written by
`memo eval tokens --update-baseline`) and reports the measured, gate-passed
savings per lever. A lever that never passed the gate reports nothing —
honest zero, not an estimate.

A stored `passed: true` is a claim about the run that wrote it, not about
today, and the fraction beside it means nothing without the sample it was
folded from. Live example (2026-08-30): this command printed
`crusher_L1 +44.4% (measured, gate-passed)` from the 3 synthetic cases in
`eval/token_corpus.json` — while `memo eval tokens --gate` exited 1 on that
very lever with `Δquality -0.48`. So the sample travels with the number and
anything below `_MIN_LEVER_SAMPLES` is named rather than published.
"""

from __future__ import annotations

import click

from .config import Config

# Independent cases a lever must be folded from before its saving fraction
# reads as evidence. Chosen to separate the two planes this gate measures: the
# recall plane runs the full curated label set (49 prompts as of 2026-08),
# while the capture plane's committed corpus is 3 hand-written cases whose own
# `_doc` says they exist to make the quality guard fail. A corpus that small
# cannot distinguish a real effect from the shape of the corpus.
_MIN_LEVER_SAMPLES = 10


def _measured_savings(cfg: Config) -> list[tuple[str, float, int]]:
    """PASSED levers, their measured saving fraction, and the sample size.

    `n_samples` is 0 for a baseline written before it was recorded — unknown
    basis, which is not the same as a large one, so it is treated as thin.
    """
    import json

    path = cfg.state_dir / "eval" / "token_baseline.json"
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    out: list[tuple[str, float, int]] = []
    for lever, m in data.items():
        if isinstance(m, dict) and m.get("passed"):
            out.append((str(lever), float(m.get("saved_frac", 0.0)), int(m.get("n_samples", 0))))
    return sorted(out, key=lambda t: t[1], reverse=True)


@click.command(name="token-savings")
def token_savings_cmd() -> None:
    """Show measured per-lever token savings (from `memo eval tokens`)."""
    cfg = Config.from_env()
    savings = _measured_savings(cfg)
    publishable = [(lever, frac, n) for lever, frac, n in savings if n >= _MIN_LEVER_SAMPLES]
    thin = [(lever, n) for lever, _frac, n in savings if n < _MIN_LEVER_SAMPLES]

    click.echo("memo token savings (measured)")
    click.echo("")
    for lever, frac, n in publishable:
        click.echo(f"  {lever:<28} {frac * 100:+.1f}%  (measured, gate-passed, {n} samples)")
    for lever, n in thin:
        basis = f"{n} samples" if n else "an unrecorded sample"
        click.echo(f"  {lever:<28} measured on {basis} — too thin to publish")
    if not publishable:
        click.echo("")
        click.echo("  No measured savings yet.")
        click.echo("  Seed the gate:  memo eval tokens --update-baseline")
        click.echo("  Then re-run:    memo token-savings")
        # This command only knows the lever gate. The proxy is a separate,
        # independently measured surface and it is where memo's savings
        # actually come from (ToolSchemas alone accounts for the large
        # majority of text the proxy removes), so an empty gate must not read
        # as "memo saves nothing".
        click.echo("")
        click.echo("  Levers are not the only surface — the context proxy is measured")
        click.echo("  separately:     memo tokens --by-transform")
        return
    click.echo("")
    click.echo("  Re-measure after any change:  memo eval tokens --gate")
