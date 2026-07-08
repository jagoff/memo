"""`memo token-savings` — show measured per-lever token savings.

Reads `state_dir/eval/token_baseline.json` (written by
`memo eval tokens --update-baseline`) and reports the measured, gate-passed
savings per lever. A lever that never passed the gate reports nothing —
honest zero, not an estimate.
"""

from __future__ import annotations

import click

from .config import Config


def _measured_savings(cfg: Config) -> list[tuple[str, float]]:
    """PASSED levers and their measured token-saving fraction, from the last
    `memo eval tokens --update-baseline`. Empty when never measured."""
    import json

    path = cfg.state_dir / "eval" / "token_baseline.json"
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    out: list[tuple[str, float]] = []
    for lever, m in data.items():
        if isinstance(m, dict) and m.get("passed"):
            out.append((str(lever), float(m.get("saved_frac", 0.0))))
    return sorted(out, key=lambda t: t[1], reverse=True)


@click.command(name="token-savings")
def token_savings_cmd() -> None:
    """Show measured per-lever token savings (from `memo eval tokens`)."""
    cfg = Config.from_env()
    savings = _measured_savings(cfg)

    click.echo("memo token savings (measured)")
    click.echo("")
    if not savings:
        click.echo("  No measured savings yet.")
        click.echo("  Seed the gate:  memo eval tokens --update-baseline")
        click.echo("  Then re-run:    memo token-savings")
        return
    for lever, frac in savings:
        click.echo(f"  {lever:<28} {frac * 100:+.1f}%  (measured, gate-passed)")
    click.echo("")
    click.echo("  Re-measure after any change:  memo eval tokens --gate")
