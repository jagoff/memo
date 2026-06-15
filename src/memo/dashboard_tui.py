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
    _panel_activity,
    _panel_consumers,
    _panel_corpus,
    _panel_recall_quality,
    _panel_recent_recalls,
    _panel_recent_saves,
    _panel_runtime,
    _panel_top_tags,
)


def render(memory: Any, state_dir: Path) -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="top", size=3),
        Layout(name="mid", size=8),
        Layout(name="bot", size=6),
        Layout(name="ext", size=9),
        Layout(name="footer", size=1),
    )
    layout["top"].split_row(Layout(name="corpus"), Layout(name="runtime"))
    layout["mid"].split_row(Layout(name="saves"), Layout(name="recalls"))
    layout["bot"].split_row(Layout(name="tags"), Layout(name="activity"))
    layout["ext"].split_row(Layout(name="recall_quality"), Layout(name="consumers"))

    layout["corpus"].update(_panel_corpus(memory))
    layout["runtime"].update(_panel_runtime(memory))
    layout["saves"].update(_panel_recent_saves(memory, limit=5))
    layout["recalls"].update(_panel_recent_recalls(state_dir, limit=4))
    layout["tags"].update(_panel_top_tags(memory, limit=4))
    layout["activity"].update(_panel_activity(memory, state_dir))
    layout["recall_quality"].update(_panel_recall_quality(state_dir))
    layout["consumers"].update(_panel_consumers(state_dir))
    now = datetime.now().strftime("%H:%M:%S")
    footer = Text.from_markup(
        f"[dim]memo · live  ·  {memory.cfg.memory_dir}  ·  [/dim][cyan]{now}[/cyan]"
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
