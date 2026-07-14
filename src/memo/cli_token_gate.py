"""`memo token-gate` — footprint regression gate over the measured meter.

Metric = cost_per_grounded (injected_tokens / grounded) and grounded_rate
(grounded / injected_sessions). A footprint change must not raise the cost per
grounded recall nor drop the grounded rate vs the saved baseline. Machine-local
(baseline under state_dir/eval/), analogous to `memo eval recall --gate`.
"""

from __future__ import annotations

import json
from pathlib import Path

import click

from .config import Config


def _baseline_path(state_dir: Path) -> Path:
    return state_dir / "eval" / "token_gate_baseline.json"


def gate_metrics(state_dir: Path) -> dict:
    from memo import token_meter

    s = token_meter.summarize(state_dir)
    grounded = max(0, int(s["grounded"]))
    injected = int(s["injected_tokens"])
    ledger = token_meter._read_ledger(state_dir)
    inj_sessions = sum(
        1 for r in ledger.get("sessions", {}).values() if int(r.get("injected_chars", 0)) > 0
    )
    return {
        "cost_per_grounded": float("inf") if grounded == 0 else round(injected / grounded, 2),
        "grounded_rate": round(grounded / inj_sessions, 4) if inj_sessions else 0.0,
        "injected_tokens": injected,
        "grounded": grounded,
    }


def check_gate(state_dir: Path, *, update_baseline: bool) -> tuple[bool, dict]:
    cur = gate_metrics(state_dir)
    bp = _baseline_path(state_dir)
    if update_baseline or not bp.is_file():
        bp.parent.mkdir(parents=True, exist_ok=True)
        bp.write_text(json.dumps(cur, indent=2), encoding="utf-8")
        return True, {**cur, "baseline": cur, "seeded": True}
    base = json.loads(bp.read_text(encoding="utf-8"))
    # tolerance: cost may not rise >2%, grounded_rate may not fall >2%.
    cost_ok = cur["cost_per_grounded"] <= base["cost_per_grounded"] * 1.02
    rate_ok = cur["grounded_rate"] >= base["grounded_rate"] * 0.98
    return (cost_ok and rate_ok), {**cur, "baseline": base}


@click.command(name="token-gate")
@click.option(
    "--update-baseline", is_flag=True, help="Seed/refresh the baseline instead of gating."
)
@click.option("--json", "as_json", is_flag=True, help="Machine-readable output.")
def token_gate_cmd(*, update_baseline: bool = False, as_json: bool = False) -> None:
    """Gate footprint changes against the measured cost-per-grounded baseline."""
    cfg = Config.from_env()
    ok, info = check_gate(cfg.state_dir, update_baseline=update_baseline)
    if as_json:
        click.echo(json.dumps({"ok": ok, **info}, ensure_ascii=False, indent=2))
    else:
        click.echo(
            f"cost/grounded {info['cost_per_grounded']}  "
            f"grounded_rate {info['grounded_rate']}  → {'PASS' if ok else 'FAIL'}"
        )
    raise SystemExit(0 if ok else 1)
