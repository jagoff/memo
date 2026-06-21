from __future__ import annotations

import contextlib
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.align import Align
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.text import Text

from memo.dashboard_panels import (
    _panel_consumers,
    _panel_corpus,
    _panel_utility,
    _panel_verdict,
)


def render(memory: Any, state_dir: Path) -> Layout:
    """Production TUI: utility focus, minimal panels, terminal fit."""
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=10),
        Layout(name="mid", size=18),
        Layout(name="footer", size=1),
    )
    layout["header"].split_row(Layout(name="utility"), Layout(name="verdict"))
    layout["mid"].split_row(Layout(name="stats"), Layout(name="consumers"))

    layout["utility"].update(_panel_utility(state_dir))
    layout["verdict"].update(_panel_verdict(state_dir))
    layout["stats"].update(_panel_corpus(memory))
    layout["consumers"].update(_panel_consumers(state_dir))
    now = datetime.now().strftime("%H:%M:%S")
    footer = Text.from_markup(
        f"[dim]memo TUI  ·  {memory.cfg.memory_dir}  ·  [/dim][cyan]{now}[/cyan]"
        f"  [dim]·  [/dim][bold]q[/bold][dim] / [/dim][bold]ESC[/bold][dim] / Ctrl+C to quit[/dim]"
    )
    layout["footer"].update(Align.center(footer))
    return layout


def _spawn_key_reader(stop_event: threading.Event) -> None:
    import select
    import sys
    import termios
    import tty

    try:
        fd = sys.stdin.fileno()
        if not sys.stdin.isatty():
            return
        old = termios.tcgetattr(fd)
    except (OSError, termios.error):
        return

    try:
        tty.setcbreak(fd)
        while not stop_event.is_set():
            r, _, _ = select.select([fd], [], [], 0.25)
            if not r:
                continue
            ch = sys.stdin.read(1)
            if ch in ("q", "Q", "\x1b"):
                stop_event.set()
                return
    except Exception:
        pass
    finally:
        with contextlib.suppress(Exception):
            termios.tcsetattr(fd, termios.TCSADRAIN, old)


def run_tui(*, refresh: float = 1.0, no_clear: bool = False) -> None:
    os.environ.setdefault("MEMO_SUPPRESS_LEGACY_WARN", "1")

    from memo.config import Config
    from memo.memory import Memory

    cfg = Config.from_env()
    mem = Memory(cfg)
    console = Console()

    stop = threading.Event()
    reader = threading.Thread(target=_spawn_key_reader, args=(stop,), daemon=True)
    reader.start()

    with Live(
        render(mem, cfg.state_dir),
        console=console,
        refresh_per_second=max(1.0 / refresh, 1.0),
        screen=not no_clear,
        transient=False,
    ) as live:
        try:
            while not stop.is_set():
                time.sleep(refresh)
                live.update(render(mem, cfg.state_dir))
        except KeyboardInterrupt:
            stop.set()
    reader.join(timeout=1.0)


__all__ = ["Group", "render", "run_tui"]
