"""`memo health` — operational health snapshot + continuous watch mode.

Thin CLI over `memo.health_report.build_health_report`: corpus size,
index dims, embedder profile, health-score coverage, and warnings.

`memo health --watch [--interval N] [--json]` streams live health signals
in a loop (Ctrl+C to stop). The watch mode is intentionally MLX-free — it
checks: recall daemon, vault-sync freshness, open file-descriptor count,
and DB connectivity.
"""

from __future__ import annotations

import json as _json
import os
import sqlite3
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import click

from memo.cli_common import console
from memo.cli_common import get_memory as _get_memory
from memo.config import Config
from memo.health_report import build_health_report

# Stale-sync threshold: if the history.db hasn't been modified in this many
# seconds, report sync as "stale". Default: 24 hours.
_SYNC_STALE_SECONDS = 24 * 3600


def _fmt_bytes(n: int | None) -> str:
    if not n:
        return "—"
    units = ["B", "KB", "MB", "GB"]
    size = float(n)
    for u in units:
        if size < 1024 or u == units[-1]:
            return f"{size:.0f}{u}" if u == "B" else f"{size:.1f}{u}"
        size /= 1024
    return f"{n}B"


# ---------------------------------------------------------------------------
# Watch-mode signal helpers (no MLX, no heavy imports)
# ---------------------------------------------------------------------------

def _check_daemon(cfg: Config) -> str:
    """Return 'running' | 'stopped' | 'unknown'."""
    try:
        from memo.recall_server import _is_pid_alive, _read_pid, _socket_path

        sock = _socket_path(cfg.state_dir)
        pid = _read_pid(cfg.state_dir)
        if pid is None and not sock.exists():
            return "stopped"
        alive = pid is not None and _is_pid_alive(pid)
        return "running" if alive and sock.exists() else "stopped"
    except Exception:
        return "unknown"


def _check_sync(cfg: Config) -> str:
    """Return 'ok' | 'stale' | 'unknown'.

    Checks mtime of history.db (updated on every ingest/sync write). If
    the file is missing or older than _SYNC_STALE_SECONDS, return 'stale'.
    """
    try:
        history_db = cfg.history_db
        if not history_db.exists():
            return "unknown"
        mtime = history_db.stat().st_mtime
        age = time.time() - mtime
        return "ok" if age < _SYNC_STALE_SECONDS else "stale"
    except Exception:
        return "unknown"


def _check_fds() -> int:
    """Return approximate open file-descriptor count for the current process.

    Uses /proc/self/fd on Linux; probes up to a reasonable cap on macOS/BSD.
    Returns -1 on failure.
    """
    try:
        proc_fd = Path("/proc/self/fd")
        if proc_fd.is_dir():
            return sum(1 for _ in proc_fd.iterdir())
        # macOS: probe by trying to dup small-numbered FDs.  We count from 0
        # up to a cap of 4096 (well above any healthy process) and stop early
        # when we see a long run of closed descriptors.
        import fcntl

        cap = 4096
        count = 0
        consecutive_closed = 0
        for fd in range(cap):
            try:
                fcntl.fcntl(fd, fcntl.F_GETFD)
                count += 1
                consecutive_closed = 0
            except OSError:
                consecutive_closed += 1
                # Once we've seen 64 consecutive closed FDs past fd 64,
                # it's safe to stop — any open FDs above will be sparse.
                if fd > 64 and consecutive_closed >= 64:
                    break
        return count
    except Exception:
        return -1


def _check_db(cfg: Config) -> str:
    """Return 'ok' | 'error' | 'missing'."""
    try:
        db_path = cfg.db_path
        if not db_path.exists():
            return "missing"
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            conn.execute("SELECT 1").fetchone()
        finally:
            conn.close()
        return "ok"
    except Exception:
        return "error"


def _collect_watch_signals(cfg: Config) -> dict[str, Any]:
    """Collect all watch-mode health signals.  Never raises."""
    daemon = _check_daemon(cfg)
    sync = _check_sync(cfg)
    fds = _check_fds()
    db = _check_db(cfg)

    # Derive overall status
    if daemon == "stopped" or db in ("error", "missing") or sync == "stale":
        status = "degraded"
    elif daemon == "unknown" or fds < 0 or sync == "unknown":
        status = "degraded"
    else:
        status = "healthy"

    # Escalate to error on DB problems
    if db == "error":
        status = "error"

    return {
        "timestamp": datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S"),
        "status": status,
        "daemon": daemon,
        "sync": sync,
        "db": db,
        "fds": fds,
    }


def _format_watch_line(sig: dict[str, Any]) -> str:
    status = sig["status"]
    icon = {"healthy": "\U0001f7e2", "degraded": "\U0001f7e1", "error": "\U0001f534"}.get(
        status, "?"
    )
    fds_str = str(sig["fds"]) if sig["fds"] >= 0 else "?"
    return (
        f"[{sig['timestamp']}] {icon} {status} | "
        f"daemon={sig['daemon']} sync={sig['sync']} db={sig['db']} fds={fds_str}"
    )


# ---------------------------------------------------------------------------
# CLI command
# ---------------------------------------------------------------------------


@click.command(name="health")
@click.option("--probe", is_flag=True, help="Time one embed_query call (loads the embedder).")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON.")
@click.option(
    "--watch",
    is_flag=True,
    help="Stream live health signals in a loop (Ctrl+C to stop).",
)
@click.option(
    "--interval",
    default=30,
    type=int,
    show_default=True,
    help="Seconds between checks when --watch is active.",
)
def health(probe: bool, as_json: bool, watch: bool, interval: int) -> None:
    """Report corpus/index/embedder health and warnings.

    Read-only. Example: memo health   |   memo health --probe --json

    Continuous mode: memo health --watch [--interval 10] [--json]
    """
    cfg = Config.from_env()

    # ------------------------------------------------------------------
    # Watch mode — lightweight, no MLX
    # ------------------------------------------------------------------
    if watch:
        try:
            while True:
                sig = _collect_watch_signals(cfg)
                if as_json:
                    click.echo(_json.dumps(sig))
                else:
                    click.echo(_format_watch_line(sig))
                time.sleep(interval)
        except KeyboardInterrupt:
            pass
        return

    # ------------------------------------------------------------------
    # One-shot mode — full health report
    # ------------------------------------------------------------------
    mem = _get_memory(cfg)
    report = build_health_report(mem, probe_embedder=probe)

    if as_json:
        click.echo(_json.dumps(report, indent=2))
        return

    corpus = report["corpus"]
    index = report["index"]
    embedder = report["embedder"]
    health_tbl = report["health_table"]

    console.print("[bold]memo health[/bold]")
    console.print(
        f"  corpus     : {corpus['memorias']} memorias "
        f"({corpus.get('archived') or 0} archived, {_fmt_bytes(corpus.get('db_size_bytes'))})"
    )
    dims_flag = "[green]ok[/green]" if index["dims_ok"] else "[red]MISMATCH[/red]"
    console.print(
        f"  index      : vec dims {index['vec_dims']}/{index['expected_dims']} {dims_flag}, "
        f"fts {'ready' if index['fts_ready'] else 'missing'} ({index['fts_backend']})"
    )
    lat = embedder["latency_ms"]
    console.print(
        f"  embedder   : {embedder['model']} ({embedder['dims']}d)"
        + (f", {lat}ms/query" if lat is not None else "")
    )
    console.print(
        f"  health tbl : {health_tbl['tracked']} tracked "
        f"({health_tbl['low_confidence']} low-confidence, {health_tbl['high_roi']} high-ROI)"
    )
    console.print(f"  feedback   : {report['feedback']['records'] or 0} signals")

    warnings = report.get("warnings") or []
    if warnings:
        console.print()
        for w in warnings:
            console.print(f"  [yellow]⚠ {w}[/yellow]")
    else:
        console.print("\n  [green]✓ no warnings[/green]")
