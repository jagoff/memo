from __future__ import annotations

import logging
import os
import subprocess
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from memo.dashboard_logs import read_recall_log
from memo.dashboard_metrics import consult_breakdown, reask_stats, recall_health, verdict

_log = logging.getLogger(__name__)
_SPARK = "▁▂▃▄▅▆▇█"
_WATCH_LABEL = "com.fer.memo.watch"


@dataclass
class _TTLCache:
    ttl_s: float
    _key: Any = field(default=None, init=False, repr=False)
    _value: Any = field(default=None, init=False, repr=False)
    _ts: float = field(default=0.0, init=False, repr=False)

    def get(self, key: Any = None) -> Any:
        if self._key == key and (time.monotonic() - self._ts) < self.ttl_s:
            return self._value
        return None

    def set(self, value: Any, key: Any = None) -> None:
        self._key = key
        self._value = value
        self._ts = time.monotonic()


_corpus_cache: _TTLCache = _TTLCache(ttl_s=10.0)
_recall_quality_cache: _TTLCache = _TTLCache(ttl_s=30.0)
_consumers_cache: _TTLCache = _TTLCache(ttl_s=30.0)
_verdict_cache: _TTLCache = _TTLCache(ttl_s=30.0)
_memflow_cache: _TTLCache = _TTLCache(ttl_s=60.0)


def _get_corpus_rows(memory: Any) -> list[dict[str, Any]]:
    key = id(memory)
    cached = _corpus_cache.get(key)
    if cached is not None:
        return cached
    rows = memory.store.list_recent(limit=10_000)
    _corpus_cache.set(rows, key)
    return rows


def sparkline(values: list[int], width: int = 12) -> str:
    if not values:
        return _SPARK[0] * width
    if len(values) > width:
        step = len(values) / width
        buckets: list[float] = []
        for i in range(width):
            lo = int(i * step)
            bucket_hi = max(lo + 1, int((i + 1) * step))
            chunk = values[lo:bucket_hi]
            buckets.append(sum(chunk) / len(chunk) if chunk else 0)
        series: list[float] = buckets
    else:
        series = [float(v) for v in values]
        series = [0.0] * (width - len(series)) + series

    hi = max(series) or 1.0
    out = []
    for v in series:
        idx = int((v / hi) * (len(_SPARK) - 1))
        out.append(_SPARK[max(0, min(len(_SPARK) - 1, idx))])
    return "".join(out)


def _human_age(ts: str | None) -> str:
    if not ts:
        return "—"
    try:
        t = ts.rstrip("Z")
        dt = datetime.fromisoformat(t)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        delta = datetime.now(UTC) - dt
        secs = int(delta.total_seconds())
    except (ValueError, TypeError):
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
    if n < 1024**2:
        return f"{n / 1024:.1f} KB"
    if n < 1024**3:
        return f"{n / (1024**2):.1f} MB"
    return f"{n / (1024**3):.2f} GB"


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
    uid = os.getuid()
    target = f"gui/{uid}/{_WATCH_LABEL}"
    res = subprocess.run(["launchctl", "print", target], capture_output=True, text=True)
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


def _panel_corpus(memory: Any) -> Panel:
    types_counter: Counter[str] = Counter()
    projects: set[str] = set()
    rows = _get_corpus_rows(memory)
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
        f"[bold cyan]{len(projects)}[/bold cyan] proj  ·  {types_line}"
    )
    return Panel(body, title="[bold magenta]corpus[/bold magenta]", border_style="magenta", padding=(0, 1))


def _panel_runtime(memory: Any) -> Panel:
    cfg = memory.cfg
    embedder_warm = bool(getattr(memory.embedder, "is_warm", False))
    chat_warm = bool(getattr(getattr(memory, "_chat", None), "_loaded", None))
    rerank_warm = False
    try:
        rr = getattr(memory, "_reranker", None)
        if rr is not None:
            rerank_warm = bool(getattr(rr, "_model", None))
    except Exception as exc:
        _log.debug("dashboard: reranker warm probe failed: %s", exc)

    vault_size = _dir_size(cfg.memory_dir)
    watcher_loaded, watcher_state = _watcher_status()

    def _dot(ok: bool, label: str) -> str:
        return f"[bold green]●[/bold green] {label}" if ok else f"[dim]○ {label}[/dim]"

    mlx_line = "  ".join([_dot(embedder_warm, "emb"), _dot(rerank_warm, "rrk"), _dot(chat_warm, "chat")])
    watcher_line = f"[green]✓ {watcher_state}[/green]" if watcher_loaded else f"[yellow]{watcher_state}[/yellow]"
    body = Text.from_markup(f"{mlx_line}  ·  [cyan]{_human_bytes(vault_size)}[/cyan]  ·  {watcher_line}")
    return Panel(body, title="[bold blue]runtime[/bold blue]", border_style="blue", padding=(0, 1))


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
        tbl.add_row(_human_age(ev.get("ts")), "[" + (ev.get("record_id") or "")[:8] + "]", (ev.get("title") or "")[:60], ev.get("type") or "")
    return Panel(tbl, title="[bold yellow]recent saves[/bold yellow]", border_style="yellow")


def _daemon_status(state_dir: Path) -> str:
    try:
        from memo.recall_server import _is_pid_alive, _read_pid

        pid = _read_pid(state_dir)
        running = pid is not None and _is_pid_alive(pid)
    except (OSError, ValueError):
        running = False

    try:
        warm_signal = state_dir / ".prewarm_ts"
        warm = warm_signal.exists() and (time.time() - float(warm_signal.read_text().strip())) < 3600
    except (OSError, ValueError):
        warm = False

    daemon_label = "[green]running[/green]" if running else "[dim]off[/dim]"
    warm_label = "[green]warm[/green]" if warm else "[yellow]cold[/yellow]"
    return f"daemon: {daemon_label} | {warm_label}"


def _panel_recent_recalls(state_dir: Path, limit: int = 8) -> Panel:
    entries = read_recall_log(state_dir, limit=limit)
    tbl = Table.grid(padding=(0, 1))
    tbl.add_column(style="dim", width=8)
    tbl.add_column(style="cyan", width=6)
    tbl.add_column()
    if not entries:
        tbl.add_row("—", "—", Text("(no recalls logged yet)", style="dim italic"))
    for e in entries:
        prompt = (e.get("prompt") or "").replace("\n", " ")[:60]
        hits = e.get("hits") or []
        mode_val = e.get("mode") or "—"
        scores = ", ".join(f"{h.get('score', 0):.2f}" for h in hits if h.get("score") is not None)
        if scores:
            line = Text.assemble((f'"{prompt}"', "white"), ("  → ", "dim"), (f"{len(hits)} hits", "bold cyan"), ("  @ ", "dim"), (scores, "magenta"))
        else:
            line = Text.assemble((f'"{prompt}"', "white"), ("  → no hits", "dim"))
        tbl.add_row(_human_age(e.get("ts")), mode_val, line)
    status_line = _daemon_status(state_dir)
    title = f"[bold green]recent recalls[/bold green]  [{status_line}]"
    return Panel(tbl, title=title, border_style="green")


def _panel_top_tags(memory: Any, limit: int = 8) -> Panel:
    rows = _get_corpus_rows(memory)
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
    return Panel(tbl, title="[bold cyan]top tags[/bold cyan]", border_style="cyan")


def _panel_activity(memory: Any, state_dir: Path) -> Panel:
    events = memory.history.list_recent(limit=2000, op="save")
    buckets: dict[str, int] = {}
    now = datetime.now(UTC)
    days = 14
    for i in range(days):
        d = (now - timedelta(days=i)).date().isoformat()
        buckets[d] = 0
    for ev in events:
        ts = ev.get("ts") or ""
        try:
            day = datetime.fromisoformat(ts.rstrip("Z")).date().isoformat()
        except ValueError:
            continue
        if day in buckets:
            buckets[day] += 1
    saves_series = [buckets[k] for k in sorted(buckets.keys())]

    recall_buckets: dict[str, int] = {k: 0 for k in buckets}
    for e in read_recall_log(state_dir, limit=1000):
        ts = e.get("ts") or ""
        try:
            day = datetime.fromisoformat(ts.rstrip("Z")).date().isoformat()
        except ValueError:
            continue
        if day in recall_buckets:
            recall_buckets[day] += 1
    recalls_series = [recall_buckets[k] for k in sorted(recall_buckets.keys())]

    body = Table.grid(padding=(0, 1))
    body.add_column(style="dim", width=14)
    body.add_column(style="bold", width=18)
    body.add_column(justify="right", style="cyan", width=10)
    body.add_row(f"saves/day ({days}d)", Text(sparkline(saves_series, width=14), style="yellow"), f"Σ {sum(saves_series)}")
    body.add_row(f"recalls/day ({days}d)", Text(sparkline(recalls_series, width=14), style="green"), f"Σ {sum(recalls_series)}")
    return Panel(body, title="[bold]activity[/bold]", border_style="bright_black")


def _panel_recall_quality(state_dir: Path) -> Panel:
    key = str(state_dir)
    cached = _recall_quality_cache.get(key)
    if cached is None:
        try:
            health = recall_health(state_dir, limit=200)
            reask = reask_stats(state_dir, limit=500)
        except Exception as exc:
            _log.debug("dashboard: recall_quality fetch failed: %s", exc)
            health = {}
            reask = {}
        cached = {"health": health, "reask": reask}
        _recall_quality_cache.set(cached, key)

    health = cached.get("health") or {}
    reask = cached.get("reask") or {}

    def _pct(v: Any) -> str:
        return "—" if v is None else f"{v * 100:.0f}%"

    def _val(v: Any, fmt: str = "{}") -> str:
        return "—" if v is None else fmt.format(v)

    tbl = Table.grid(padding=(0, 2))
    tbl.add_column(style="dim", width=18)
    tbl.add_column(style="bold")
    tbl.add_row("grounded", f"[bold green]{_pct(health.get('grounded_rate'))}[/bold green]  [dim](primary)[/dim]")
    tbl.add_row("hit / strong", f"{_pct(health.get('hit_rate'))}  /  {_pct(health.get('strong_hit_rate'))}")
    tbl.add_row("median score", f"{_val(health.get('median_top_score'), '{:.2f}')}  [dim]p50 lat[/dim] {_val(health.get('p50_latency_ms'), '{}ms')}")
    tbl.add_row("sampled/fired", f"{_val(health.get('sampled'))} / {_val(health.get('fired'))}")
    tbl.add_row("reask avoided", f"{_val(reask.get('reask_avoided'))} [dim]of[/dim] {_val(reask.get('considered'))}  [dim]reask%[/dim] {_pct(reask.get('reask_rate'))}")
    return Panel(tbl, title="[bold green]recall quality[/bold green]", border_style="green")


_VERDICT_STYLE = {
    "ok": ("green", "bold green"),
    "weak": ("yellow", "bold yellow"),
    "unused": ("red", "bold red"),
}


def _panel_verdict(state_dir: Path) -> Panel:
    """Top-of-dashboard answer to '¿funciona memo y quién lo lee?'."""
    key = str(state_dir)
    data = _verdict_cache.get(key)
    if data is None:
        try:
            data = verdict(state_dir, limit=500)
        except Exception as exc:
            _log.debug("dashboard: verdict fetch failed: %s", exc)
            data = {}
        _verdict_cache.set(data, key)

    status = str(data.get("status") or "unused")
    border, label_style = _VERDICT_STYLE.get(status, ("red", "bold red"))
    label = str(data.get("label") or "❌ NO SE USA")
    grounded = data.get("grounded_rate")
    gr = "—" if grounded is None else f"{grounded * 100:.0f}%"
    consults = data.get("consults") or 0

    body = Text()
    body.append(f"memo: {label}\n", style=label_style)
    body.append(f"grounded {gr} · {consults} consultas\n", style="dim")

    reads = [p["name"] for p in data.get("per_consumer", []) if p.get("reads")]
    silent = [p["name"] for p in data.get("per_consumer", []) if not p.get("reads")]
    if reads:
        body.append("lee:  ", style="dim")
        body.append("  ".join(f"{n} ✅" for n in reads) + "\n", style="green")
    if silent:
        body.append("NO lee:  ", style="dim")
        body.append("  ".join(f"{n} ❌" for n in silent), style="bold red")
    return Panel(body, title="[bold]¿FUNCIONA memo?[/bold]", border_style=border, padding=(0, 1))


def _memflow_bin() -> str | None:
    for cand in (
        os.path.expanduser("~/.memflow/bin/memflow"),
        os.path.expanduser("~/repos/memflow/.venv/bin/memflow"),
    ):
        if os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    from shutil import which

    return which("memflow")


def _fetch_memflow_utility() -> dict[str, Any]:
    cached = _memflow_cache.get("u")
    if cached is not None:
        return cached
    data: dict[str, Any] = {}
    bin_path = _memflow_bin()
    if bin_path:
        try:
            out = subprocess.run(
                [bin_path, "utility", "--since-days", "7", "--json"],
                capture_output=True,
                text=True,
                timeout=8,
            )
            if out.returncode == 0 and out.stdout.strip():
                import json

                data = json.loads(out.stdout)
        except Exception as exc:
            _log.debug("dashboard: memflow utility fetch failed: %s", exc)
            data = {}
    _memflow_cache.set(data, "u")
    return data


def _panel_memflow(_state_dir: Path) -> Panel:
    """memflow's own utility report (B) — does the layer above memo work?"""
    data = _fetch_memflow_utility()
    if not data:
        body = Text("memflow no disponible", style="dim italic")
        return Panel(body, title="[bold]memflow[/bold]", border_style="bright_black", padding=(0, 1))

    cons = data.get("consumption") or {}
    out = data.get("outcome") or {}
    reads = int(cons.get("total_read_calls") or 0)
    re_explain = int(out.get("re_explain") or 0)
    used_rate = out.get("memory_used_rate")

    # Verdict: useful only when read and not drowning in re-explains.
    if reads < 5:
        border, head = "red", "❌ casi no se lee"
    elif used_rate is None:
        border, head = "yellow", "⚠️ sin outcomes aún"
    elif used_rate >= 0.10 and re_explain < 10:
        border, head = "green", "✅ útil"
    else:
        border, head = "yellow", "⚠️ poco útil"

    ur = "—" if used_rate is None else f"{used_rate * 100:.0f}%"
    tbl = Table.grid(padding=(0, 2))
    tbl.add_column(style="dim", width=14)
    tbl.add_column(style="bold")
    tbl.add_row("estado", Text(head, style=border if border != "bright_black" else "dim"))
    tbl.add_row("reads 7d", str(reads))
    tbl.add_row("memory_used", ur)
    tbl.add_row("re_explain", f"[red]{re_explain}[/red]" if re_explain >= 10 else str(re_explain))
    return Panel(tbl, title="[bold]memflow utility[/bold]", border_style=border, padding=(0, 1))


def _panel_consumers(state_dir: Path) -> Panel:
    key = str(state_dir)
    cached = _consumers_cache.get(key)
    if cached is None:
        try:
            data = consult_breakdown(state_dir, limit=500)
        except Exception as exc:
            _log.debug("dashboard: consult_breakdown fetch failed: %s", exc)
            data = {}
        cached = data
        _consumers_cache.set(cached, key)

    consumers = cached.get("consumers") or []
    silent = cached.get("silent") or []
    tbl = Table.grid(padding=(0, 1))
    tbl.add_column(style="cyan", width=14)
    tbl.add_column(justify="right", width=7)
    tbl.add_column(justify="right", width=7)
    tbl.add_column(justify="right", width=7)
    tbl.add_column(style="dim", width=10)
    if not consumers:
        tbl.add_row(Text("(no recall log data)", style="dim italic"), "", "", "", "")
    for c in consumers[:6]:
        name = (c.get("consumer") or "?")[:13]
        consults = str(c.get("consults") or "—")
        hit = c.get("hit_rate")
        hit_s = f"{hit * 100:.0f}%" if hit is not None else "—"
        grounded = c.get("grounded_rate")
        gr_s = f"{grounded * 100:.0f}%" if grounded is not None else "—"
        last = _human_age(c.get("last_seen"))
        tbl.add_row(name, consults, hit_s, gr_s, last)
    body = (
        Group(tbl, Text.assemble(("NO leen memo: ", "dim"), (", ".join(silent), "bold red")))
        if silent
        else tbl
    )
    return Panel(body, title="[bold cyan]consumers[/bold cyan]", border_style="cyan")
