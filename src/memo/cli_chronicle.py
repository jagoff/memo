"""`memo chronicle` — read the nightly engineering diary (see dream_chronicle)."""
from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console
from rich.markdown import Markdown

from memo.config import Config
from memo.dream_chronicle import chronicle_dir, chronicle_path

console = Console()


@click.command(name="chronicle")
@click.option("--date", "date", default=None, help="Day to show (YYYY-MM-DD, default: latest).")
@click.option("--week", "week", is_flag=True, help="Show the latest weekly rollup instead.")
def chronicle_cmd(date: str | None, week: bool) -> None:
    """Show the engineering diary memo wrote for a day (or week)."""
    cfg = Config.from_env()
    root = chronicle_dir(cfg)
    target: Path | None
    if date is not None:
        target = chronicle_path(cfg, date)
    else:
        pattern = "week-*.md" if week else "[0-9]*.md"
        candidates = sorted(root.glob(pattern)) if root.exists() else []
        target = candidates[-1] if candidates else None
    if target is None or not target.exists():
        console.print(
            "No hay crónica todavía — encendé el pase nocturno con "
            "[cyan]MEMO_DREAM_CHRONICLE_ENABLED[/cyan] o corré [cyan]memo dream chronicle[/cyan]."
        )
        return
    console.print(Markdown(target.read_text(encoding="utf-8")))
