"""`memo tui` / `hook-log` / `logs` — recall telemetry + log viewers.

Extracted from cli.py (2b god-module decomposition). Registered onto the root
group in cli.py via `cli.add_command(...)`.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import click

from memo.cli_common import console
from memo.config import Config


@click.command(name="tui")
@click.option(
    "--refresh", type=float, default=1.0, show_default=True, help="Refresh interval in seconds."
)
@click.option(
    "--no-clear",
    is_flag=True,
    help="Don't take over the terminal screen — render inline (handy for tmux/screen).",
)
@click.option(
    "--once",
    "--no-loop",
    "once",
    is_flag=True,
    default=False,
    help="Render a single frame and exit (no live loop) — for scripts/headless.",
)
def tui(refresh: float, no_clear: bool, once: bool) -> None:
    """Live terminal dashboard — context activity, recall outcomes, consumers.

    Shows: tokens injected, recall hooks, composite-score rate, surfaced
    memories, and grounding rate. Ctrl+C to exit.
    """
    from memo.dashboard import run_tui

    run_tui(refresh=refresh, no_clear=no_clear, once=once)


@click.command(name="hook-log")
@click.option(
    "--limit", default=20, type=int, show_default=True, help="Number of recent entries to show."
)
@click.option("--follow", is_flag=True, help="Tail the log file (like tail -f). Ctrl+C to stop.")
def hook_log(limit: int, follow: bool) -> None:
    """Show recent recall-hook activity.

    Reads the recall log written by `memo recall-hook` and prints the
    last N entries with timestamp, mode (vec/bm25/daemon), hit count,
    and latency. Use --follow to stream new entries as they arrive.

    \b
    Fields printed per entry:
      ts       — ISO timestamp of the recall
      mode     — vec / bm25 / daemon (how the search ran)
      hits     — number of memories injected
      latency  — round-trip latency if logged
      via      — subprocess or daemon
    """
    from memo.dashboard import read_recall_log

    cfg = Config.from_env()
    state_dir = cfg.state_dir

    def _fmt_entry(e: dict) -> str:
        ts = (e.get("ts") or "")[:19].replace("T", " ")
        mode_val = e.get("mode") or "—"
        via_val = e.get("via") or "—"
        hits_list = e.get("hits") or []
        n_hits = len(hits_list)
        latency = e.get("latency_ms")
        latency_str = f"{latency} ms" if latency is not None else "—"
        prompt = (e.get("prompt") or "").replace("\n", " ")[:60]
        return (
            f"[dim]{ts}[/dim]  "
            f"mode=[cyan]{mode_val}[/cyan]  "
            f"via=[yellow]{via_val}[/yellow]  "
            f"hits=[bold]{n_hits}[/bold]  "
            f"latency=[magenta]{latency_str}[/magenta]  "
            f'[dim]"{prompt}"[/dim]'
        )

    if not follow:
        entries = read_recall_log(state_dir, limit=limit)
        if not entries:
            console.print("[dim](no recall log entries yet)[/dim]")
            return
        # entries is newest-first; print oldest-first for readability
        for e in reversed(entries):
            console.print(_fmt_entry(e))
        return

    # --follow: tail the log file
    from memo.dashboard import recall_log_path

    log_path = recall_log_path(state_dir)
    console.print(f"[dim]tailing {log_path} … Ctrl+C to stop[/dim]")

    # Seek to end, then loop
    last_pos = log_path.stat().st_size if log_path.exists() else 0
    try:
        while True:
            if log_path.exists():
                new_size = log_path.stat().st_size
                if new_size < last_pos:
                    # File was rotated (truncated and rewritten) — reset position
                    last_pos = 0
                if new_size > last_pos:
                    with log_path.open("r", encoding="utf-8") as f:
                        f.seek(last_pos)
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                e = json.loads(line)
                                console.print(_fmt_entry(e))
                            except json.JSONDecodeError:
                                pass
                        last_pos = f.tell()
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass


@click.command(name="logs")
@click.option(
    "--source",
    type=click.Choice(["recall", "daemon", "watcher", "all"]),
    default="all",
    show_default=True,
    help="Which log to read. 'all' interleaves by timestamp where possible.",
)
@click.option(
    "--tail", default=40, type=int, show_default=True, help="Number of recent lines per source."
)
@click.option("--paths", is_flag=True, help="Just print the log file paths (for tail -f / less).")
def logs(source: str, tail: int, paths: bool) -> None:
    """Show recent memo log activity in one place.

    \b
    Aggregates three log surfaces:
      recall   — JSONL of every recall-hook invocation
                 (bails, daemon hits, in-process fallbacks)
      daemon   — recall-daemon stdout/stderr (MLX warm-state, errors)
      watcher  — filesystem watcher stdout/stderr (launchd plist)

    Use --paths if you'd rather pipe to your own `tail -f` / `less +F`.
    """
    from memo.dashboard import read_recall_log, recall_log_path

    cfg = Config.from_env()
    state_dir = cfg.state_dir

    recall_p = recall_log_path(state_dir)
    daemon_p = Path.home() / "Library" / "Logs" / "memo" / "recall-daemon.log"
    watch_out_p = Path.home() / "Library" / "Logs" / "memo" / "watch.out.log"
    watch_err_p = Path.home() / "Library" / "Logs" / "memo" / "watch.err.log"

    if paths:
        console.print(f"recall:  {recall_p}")
        console.print(f"daemon:  {daemon_p}")
        console.print(f"watcher: {watch_out_p}  +  {watch_err_p}")
        return

    def _tail_file(p: Path, n: int) -> list[str]:
        if not p.is_file():
            return []
        try:
            return p.read_text(encoding="utf-8", errors="replace").splitlines()[-n:]
        except Exception as exc:  # pragma: no cover - permission edge
            return [f"# (read error: {exc})"]

    if source in ("recall", "all"):
        entries = read_recall_log(state_dir, limit=tail)
        console.print(f"[bold]recall[/bold]  [dim]{recall_p}[/dim]")
        if not entries:
            console.print("  [dim](empty)[/dim]")
        else:
            for e in reversed(entries):
                ts = (e.get("ts") or "")[:19].replace("T", " ")
                via = e.get("via") or "—"
                hits = len(e.get("hits") or [])
                latency = e.get("latency_ms")
                latency_str = f" {latency}ms" if latency is not None else ""
                reason = e.get("reason")
                error = e.get("error")
                tail_str = ""
                if reason:
                    tail_str = f"  [yellow]reason=[/yellow]{reason}"
                elif error:
                    tail_str = f"  [red]error=[/red]{error}"
                else:
                    prompt = (e.get("prompt") or "").replace("\n", " ")[:50]
                    tail_str = f'  [dim]"{prompt}"[/dim]' if prompt else ""
                console.print(
                    f"  [dim]{ts}[/dim] via=[cyan]{via}[/cyan] "
                    f"hits=[bold]{hits}[/bold]{latency_str}{tail_str}",
                    highlight=False,
                )
        if source == "all":
            console.print()

    if source in ("daemon", "all"):
        console.print(f"[bold]daemon[/bold]  [dim]{daemon_p}[/dim]")
        lines = _tail_file(daemon_p, tail)
        if not lines:
            console.print("  [dim](no log — daemon never started, or logs rotated)[/dim]")
        else:
            for line in lines:
                console.print(f"  {line}")
        if source == "all":
            console.print()

    if source in ("watcher", "all"):
        console.print(f"[bold]watcher.out[/bold]  [dim]{watch_out_p}[/dim]")
        out_lines = _tail_file(watch_out_p, tail)
        if not out_lines:
            console.print("  [dim](no log — watcher inactive or never wrote)[/dim]")
        else:
            for line in out_lines:
                console.print(f"  {line}")
        err_lines = _tail_file(watch_err_p, tail)
        if err_lines:
            console.print(f"[bold]watcher.err[/bold]  [dim]{watch_err_p}[/dim]")
            for line in err_lines:
                console.print(f"  [red]{line}[/red]")
