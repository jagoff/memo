"""Terminal dashboard — `memo tui`.

Live, read-only view over the memo corpus and runtime state. Uses Rich
Live (already a dep) so there's no Textual / curses overhead. Refresh
every `--refresh` seconds (default 1.0). Press Ctrl+C to exit.

Panels
------
- **corpus** — total memorias, breakdown by type, distinct project tags
- **runtime** — MLX model warm/cold flags, vault path + size, watcher
  status (launchd job)
- **recent saves** — last N entries from `history.db` (op=save)
- **recent recalls** — last N entries from the recall log JSONL
  (`~/.local/share/memo/recall.log`), written by `memo recall-hook`
- **top tags** — most frequent tags across the corpus
- **activity** — sparklines of saves/day and recalls/day

The dashboard is intentionally allocation-cheap on the refresh path —
the heavy work (corpus scan, launchctl probe) is sampled at a slower
cadence than the UI tick so a 1 s refresh doesn't thrash the disk.
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import threading
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rich.align import Align
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# Unicode block characters for sparklines (8 levels, low→high).
_SPARK = "▁▂▃▄▅▆▇█"
_WATCH_LABEL = "com.memo.watch"


# -------------------- recall log (write side) --------------------

def recall_log_path(state_dir: Path) -> Path:
    return state_dir / "recall.log"


def append_recall_log(
    state_dir: Path,
    *,
    prompt: str,
    hits: list[dict[str, Any]],
    cap: int = 200,
    mode: str | None = None,
    latency_ms: int | None = None,
    via: str | None = None,
) -> None:
    """Append a recall event to the JSONL ring buffer. Rotates by
    keeping only the most recent `cap` lines after writing. Errors are
    swallowed — the recall hook must never fail because of telemetry."""
    try:
        state_dir.mkdir(parents=True, exist_ok=True)
        entry: dict[str, Any] = {
            "ts": datetime.now(UTC).isoformat(timespec="seconds"),
            "prompt": prompt[:200],
            "hits": [
                {"id": h.get("id", "")[:8], "score": h.get("score"), "title": h.get("title", "")[:80]}
                for h in hits[:5]
            ],
        }
        if mode is not None:
            entry["mode"] = mode
        if latency_ms is not None:
            entry["latency_ms"] = latency_ms
        if via is not None:
            entry["via"] = via
        path = recall_log_path(state_dir)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        # Rotate when the file gets too big (cheap line count). Skip
        # rotation 99% of the time — only fire when over cap*1.5.
        if path.stat().st_size > 1024 * 200:  # ~200 KB
            lines = path.read_text(encoding="utf-8").splitlines()[-cap:]
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except Exception:
        pass


def read_recall_log(state_dir: Path, *, limit: int = 10) -> list[dict[str, Any]]:
    path = recall_log_path(state_dir)
    if not path.is_file():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return []
    out: list[dict[str, Any]] = []
    for line in lines[-limit:]:
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    out.reverse()  # newest first
    return out


# -------------------- helpers --------------------

def sparkline(values: list[int], width: int = 12) -> str:
    """Render `values` as a unicode-block sparkline of `width` chars.

    Buckets the series into `width` slots (averaging when len > width,
    padding with the lowest-level char when len < width). All-zero
    series renders as ▁▁▁…, not blank, so the slot stays visually present.
    """
    if not values:
        return _SPARK[0] * width
    if len(values) > width:
        # Bucket into `width` chunks.
        step = len(values) / width
        buckets = []
        for i in range(width):
            lo = int(i * step)
            hi = max(lo + 1, int((i + 1) * step))
            chunk = values[lo:hi]
            buckets.append(sum(chunk) / len(chunk) if chunk else 0)
        series: list[float] = buckets
    else:
        series = [float(v) for v in values]
        # Pad left with zeros so the latest sample sits at the right.
        series = [0.0] * (width - len(series)) + series

    hi = max(series) or 1.0
    out = []
    for v in series:
        idx = int((v / hi) * (len(_SPARK) - 1))
        out.append(_SPARK[max(0, min(len(_SPARK) - 1, idx))])
    return "".join(out)


def _human_age(ts: str | None) -> str:
    """Render an ISO ts as 'Nm ago' / 'Nh ago' / 'Nd ago'."""
    if not ts:
        return "—"
    try:
        # Accept both Z and +00:00 forms; strip any trailing Z.
        t = ts.rstrip("Z")
        dt = datetime.fromisoformat(t)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        delta = datetime.now(UTC) - dt
        secs = int(delta.total_seconds())
    except Exception:
        return "—"
    if secs < 0:
        return "now"
    if secs < 60:
        return f"{secs}s ago"
    if secs < 3600:
        return f"{secs // 60}m ago"
    if secs < 86400:
        return f"{secs // 3600}h ago"
    return f"{secs // 86400}d ago"


def _human_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 ** 2:
        return f"{n / 1024:.1f} KB"
    if n < 1024 ** 3:
        return f"{n / (1024 ** 2):.1f} MB"
    return f"{n / (1024 ** 3):.2f} GB"


def _dir_size(p: Path) -> int:
    if not p.is_dir():
        return 0
    total = 0
    for f in p.rglob("*"):
        try:
            if f.is_file():
                total += f.stat().st_size
        except OSError:
            continue
    return total


def _watcher_status() -> tuple[bool, str]:
    """Return (running, status_text). `running` is True iff the launchd
    job is loaded; status_text is a one-liner suitable for display."""
    uid = os.getuid()
    target = f"gui/{uid}/{_WATCH_LABEL}"
    res = subprocess.run(
        ["launchctl", "print", target],
        capture_output=True, text=True,
    )
    if res.returncode != 0:
        return False, "not installed (memo install-watcher)"
    out = res.stdout
    state = "running"
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("state ="):
            state = line.split("=", 1)[1].strip()
            break
    return True, state


# -------------------- panel builders --------------------

def _panel_corpus(memory: Any) -> Panel:
    types_counter: Counter[str] = Counter()
    projects: set[str] = set()
    # Avoid loading every body — list_recent yields meta-only rows from
    # the store layer (path/title/type/tags/created/updated).
    rows = memory.store.list_recent(limit=10_000)
    for r in rows:
        types_counter[r["type"]] += 1
        for t in r["tags"] or []:
            if t.startswith("project:"):
                projects.add(t)

    total = sum(types_counter.values())
    top3 = types_counter.most_common(3)
    types_line = "  ".join(f"[bold]{n}[/bold] {t}" for t, n in top3) or "—"
    body = Text.from_markup(
        f"[bold cyan]{total}[/bold cyan] memorias  ·  "
        f"[bold cyan]{len(projects)}[/bold cyan] proj  ·  {types_line}",
    )
    return Panel(body, title="[bold magenta]corpus[/bold magenta]",
                 border_style="magenta", padding=(0, 1))


def _panel_runtime(memory: Any) -> Panel:
    cfg = memory.cfg
    embedder_warm = memory.embedder._model is not None  # type: ignore[attr-defined]
    chat_warm = bool(getattr(getattr(memory, "_chat", None), "_loaded", None))
    rerank_warm = False
    try:
        rr = getattr(memory, "_reranker", None)
        if rr is not None:
            rerank_warm = bool(getattr(rr, "_model", None))
    except Exception:
        pass

    vault_size = _dir_size(cfg.memory_dir)
    watcher_loaded, watcher_state = _watcher_status()

    def _dot(ok: bool, label: str) -> str:
        return (
            f"[bold green]●[/bold green] {label}"
            if ok else f"[dim]○ {label}[/dim]"
        )

    mlx_line = "  ".join([
        _dot(embedder_warm, "emb"),
        _dot(rerank_warm, "rrk"),
        _dot(chat_warm, "chat"),
    ])
    watcher_line = (
        f"[green]✓ {watcher_state}[/green]"
        if watcher_loaded
        else f"[yellow]{watcher_state}[/yellow]"
    )
    body = Text.from_markup(
        f"{mlx_line}  ·  [cyan]{_human_bytes(vault_size)}[/cyan]  ·  "
        f"{watcher_line}",
    )
    return Panel(body, title="[bold blue]runtime[/bold blue]",
                 border_style="blue", padding=(0, 1))


def _panel_recent_saves(memory: Any, limit: int = 10) -> Panel:
    events = memory.history.list_recent(limit=limit, op="save")
    tbl = Table.grid(padding=(0, 1))
    tbl.add_column(style="dim", width=8)
    tbl.add_column(style="bold yellow", width=10)
    tbl.add_column(style="bold")
    tbl.add_column(style="cyan", width=10)
    if not events:
        tbl.add_row("—", "—", Text("(no saves yet)", style="dim italic"), "—")
    for ev in events:
        tbl.add_row(
            _human_age(ev.get("ts")),
            "[" + (ev.get("record_id") or "")[:8] + "]",
            (ev.get("title") or "")[:60],
            ev.get("type") or "",
        )
    return Panel(tbl, title="[bold yellow]recent saves[/bold yellow]",
                 border_style="yellow")


def _daemon_status(state_dir: Path) -> str:
    """Return a one-line daemon status string for the TUI panels."""
    try:
        from memo.recall_server import _is_pid_alive, _read_pid
        pid = _read_pid(state_dir)
        running = pid is not None and _is_pid_alive(pid)
    except Exception:
        running = False

    try:
        warm_signal = state_dir / ".prewarm_ts"
        warm = (
            warm_signal.exists()
            and (time.time() - float(warm_signal.read_text().strip())) < 3600
        )
    except Exception:
        warm = False

    daemon_label = "[green]running[/green]" if running else "[dim]off[/dim]"
    warm_label = "[green]warm[/green]" if warm else "[yellow]cold[/yellow]"
    return f"daemon: {daemon_label} | {warm_label}"


def _panel_recent_recalls(state_dir: Path, limit: int = 8) -> Panel:
    entries = read_recall_log(state_dir, limit=limit)
    tbl = Table.grid(padding=(0, 1))
    tbl.add_column(style="dim", width=8)
    tbl.add_column(style="cyan", width=6)   # mode column
    tbl.add_column()
    if not entries:
        tbl.add_row("—", "—", Text("(no recalls logged yet)", style="dim italic"))
    for e in entries:
        prompt = (e.get("prompt") or "").replace("\n", " ")[:60]
        hits = e.get("hits") or []
        mode_val = e.get("mode") or "—"
        scores = ", ".join(
            f"{h.get('score', 0):.2f}" for h in hits if h.get("score") is not None
        )
        if scores:
            line = Text.assemble(
                ("\"" + prompt + "\"", "white"),
                ("  → ", "dim"),
                (f"{len(hits)} hits", "bold cyan"),
                ("  @ ", "dim"),
                (scores, "magenta"),
            )
        else:
            line = Text.assemble(
                ("\"" + prompt + "\"", "white"),
                ("  → no hits", "dim"),
            )
        tbl.add_row(_human_age(e.get("ts")), mode_val, line)

    status_line = _daemon_status(state_dir)
    title = f"[bold green]recent recalls[/bold green]  [{status_line}]"
    return Panel(tbl, title=title, border_style="green")


def _panel_top_tags(memory: Any, limit: int = 8) -> Panel:
    rows = memory.store.list_recent(limit=10_000)
    counter: Counter[str] = Counter()
    for r in rows:
        for t in r["tags"] or []:
            counter[t] += 1
    tbl = Table.grid(padding=(0, 1))
    tbl.add_column(style="cyan")
    tbl.add_column(justify="right", style="bold")
    if not counter:
        tbl.add_row(Text("(no tags)", style="dim italic"), "")
    for tag, n in counter.most_common(limit):
        style = "bold magenta" if tag.startswith("project:") else "cyan"
        tbl.add_row(Text(tag, style=style), str(n))
    return Panel(tbl, title="[bold cyan]top tags[/bold cyan]",
                 border_style="cyan")


def _panel_activity(memory: Any, state_dir: Path) -> Panel:
    # saves: aggregate `events.ts` per day for the last 14 days.
    events = memory.history.list_recent(limit=2000, op="save")
    buckets: dict[str, int] = {}
    now = datetime.now(UTC)
    days = 14
    for i in range(days):
        d = (now - _td(i)).date().isoformat()
        buckets[d] = 0
    for ev in events:
        ts = ev.get("ts") or ""
        try:
            day = datetime.fromisoformat(ts.rstrip("Z")).date().isoformat()
        except Exception:
            continue
        if day in buckets:
            buckets[day] += 1
    saves_series = [buckets[k] for k in sorted(buckets.keys())]

    # recalls: same bucketing over recall.log
    recall_buckets: dict[str, int] = {k: 0 for k in buckets}
    for e in read_recall_log(state_dir, limit=1000):
        ts = e.get("ts") or ""
        try:
            day = datetime.fromisoformat(ts.rstrip("Z")).date().isoformat()
        except Exception:
            continue
        if day in recall_buckets:
            recall_buckets[day] += 1
    recalls_series = [recall_buckets[k] for k in sorted(recall_buckets.keys())]

    body = Table.grid(padding=(0, 1))
    body.add_column(style="dim", width=14)
    body.add_column(style="bold", width=18)
    body.add_column(justify="right", style="cyan", width=10)
    body.add_row(
        f"saves/day ({days}d)",
        Text(sparkline(saves_series, width=14), style="yellow"),
        f"Σ {sum(saves_series)}",
    )
    body.add_row(
        f"recalls/day ({days}d)",
        Text(sparkline(recalls_series, width=14), style="green"),
        f"Σ {sum(recalls_series)}",
    )
    return Panel(body, title="[bold]activity[/bold]", border_style="bright_black")


def _td(days: int):
    from datetime import timedelta
    return timedelta(days=days)


# -------------------- main loop --------------------

def render(memory: Any, state_dir: Path) -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="top", size=3),
        Layout(name="mid", size=8),
        Layout(name="bot", size=6),
        Layout(name="footer", size=1),
    )
    layout["top"].split_row(Layout(name="corpus"), Layout(name="runtime"))
    layout["mid"].split_row(Layout(name="saves"), Layout(name="recalls"))
    layout["bot"].split_row(Layout(name="tags"), Layout(name="activity"))

    layout["corpus"].update(_panel_corpus(memory))
    layout["runtime"].update(_panel_runtime(memory))
    layout["saves"].update(_panel_recent_saves(memory, limit=5))
    layout["recalls"].update(_panel_recent_recalls(state_dir, limit=4))
    layout["tags"].update(_panel_top_tags(memory, limit=4))
    layout["activity"].update(_panel_activity(memory, state_dir))
    now = datetime.now().strftime("%H:%M:%S")
    footer = Text.from_markup(
        f"[dim]memo · live  ·  {memory.cfg.memory_dir}  ·  [/dim][cyan]{now}[/cyan]"
        f"  [dim]·  [/dim][bold]q[/bold][dim] / [/dim][bold]ESC[/bold][dim] / Ctrl+C to quit[/dim]",
    )
    layout["footer"].update(Align.center(footer))
    return layout


def _spawn_key_reader(stop_event: threading.Event) -> None:
    """Background thread that reads single keystrokes from stdin and
    sets `stop_event` when the user presses `q`, `Q`, or ESC. Falls
    back gracefully when stdin isn't a TTY (CI, pipes) — the thread
    just exits and the user can still Ctrl+C.
    """
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
        # Stdin reads can fail during shutdown — never propagate.
        pass
    finally:
        with contextlib.suppress(Exception):
            termios.tcsetattr(fd, termios.TCSADRAIN, old)


def run_tui(*, refresh: float = 1.0, no_clear: bool = False) -> None:
    """Block on a Live dashboard until the user presses q / ESC / Ctrl+C."""
    import threading

    # The legacy-path warning from Memory() startup pollutes the alt
    # screen (and the user can't act on it from inside the TUI anyway).
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
    # Reader thread will see stop and clean up termios on its own.
    reader.join(timeout=1.0)


# Keep `Group` importable for tests / external users of the layout
# (avoids an "unused import" lint when downstream users compose panels).
__all__ = [
    "Group",
    "append_recall_log",
    "read_recall_log",
    "recall_log_path",
    "render",
    "run_tui",
    "sparkline",
]
