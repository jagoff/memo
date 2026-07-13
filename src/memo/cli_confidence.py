"""`memo confidence` — inspect the recall confidence calibration map."""
from __future__ import annotations

import click

from memo.confidence_calibration import build_calibration, load_calibration
from memo.config import Config


@click.group(name="confidence")
def confidence_group() -> None:
    """Confidence calibration (see MEMO_RECALL_CONFIDENCE_GATE)."""


def _print_doc(doc: dict) -> None:
    bins = doc.get("bins") or {}
    mapping = doc.get("map") or {}
    if not bins and not mapping:
        click.echo("no calibration yet — run `memo dream run` with the gate on.")
        return
    for band in ("high", "med", "low"):
        b = bins.get(band) or {}
        remap = mapping.get(band, band)
        arrow = f"  → {remap}" if remap != band else ""
        click.echo(
            f"{band:<5} predicted {b.get('predicted', 0.0):.2f}  "
            f"observed {b.get('observed', 0.0):.2f}  n {b.get('n', 0)}{arrow}"
        )


@confidence_group.command(name="status")
def status_cmd() -> None:
    _print_doc(load_calibration(Config.from_env().state_dir))


@confidence_group.command(name="explain")
def explain_cmd() -> None:
    from memo.memory import Memory

    cfg = Config.from_env()
    _print_doc(build_calibration(cfg.state_dir, Memory(cfg)))
