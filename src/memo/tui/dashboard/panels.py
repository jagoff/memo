from __future__ import annotations

import json
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

from memo.dashboard_logs import (
    read_context_cost_log,
    read_grounding_log,
    read_recall_log,
    read_usage_log,
)
from memo.dashboard_metrics import (
    consult_breakdown,
    grounding_used,
    reask_stats,
    recall_health,
    verdict,
)
from memo.watcher import _PLIST_LABEL as _WATCH_LABEL

_log = logging.getLogger(__name__)
_SPARK = "▁▂▃▄▅▆▇█"

# Console-style rendering: one accent colour, flat sections with a thin rule
# header instead of boxed panels (see render() in app.py). Data functions are
# shared with the legacy boxed panels; only the presentation differs.
_ACCENT = "cyan"
_HDR_WIDTH = 46


def _hdr(title: str) -> Text:
    """A flat section header: accented title + a dim rule filling the line."""
    t = Text()
    t.append(title, style=f"bold {_ACCENT}")
    t.append("  ")
    t.append("─" * max(2, _HDR_WIDTH - len(title)), style="bright_black")
    return t


def _kv(width: int = 14) -> Table:
    """A two-column key/value grid (dim label, plain value) — no box."""
    tbl = Table.grid(padding=(0, 2))
    tbl.add_column(style="dim", width=width)
    tbl.add_column(overflow="fold")
    return tbl


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
    try:
        res = subprocess.run(
            ["launchctl", "print", target], capture_output=True, text=True, timeout=5
        )
    except subprocess.TimeoutExpired:
        return False, "not installed (memo install-watcher)"
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


def _panel_corpus(memory: Any) -> Group:
    types_counter: Counter[str] = Counter()
    projects: set[str] = set()
    rows = _get_corpus_rows(memory)
    for r in rows:
        types_counter[r.get("type", "unknown")] += 1
        for t in r.get("tags") or []:
            if t.startswith("project:"):
                projects.add(t)
    total = sum(types_counter.values())
    types_line = " · ".join(f"{n} {t}" for t, n in types_counter.most_common(4)) or "—"
    body = Text.assemble(
        (str(total), "bold"),
        (" memories · ", "dim"),
        (str(len(projects)), "bold"),
        (" projects", "dim"),
        ("    ", ""),
        (types_line, "default"),
    )
    return Group(_hdr("corpus"), body, Text())


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

    mlx_line = "  ".join(
        [_dot(embedder_warm, "emb"), _dot(rerank_warm, "rrk"), _dot(chat_warm, "chat")]
    )
    watcher_line = (
        f"[green]✓ {watcher_state}[/green]"
        if watcher_loaded
        else f"[yellow]{watcher_state}[/yellow]"
    )
    body = Text.from_markup(
        f"{mlx_line}  ·  [cyan]{_human_bytes(vault_size)}[/cyan]  ·  {watcher_line}"
    )
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
        tbl.add_row(
            _human_age(ev.get("ts")),
            "[" + (ev.get("record_id") or "")[:8] + "]",
            (ev.get("title") or "")[:60],
            ev.get("type") or "",
        )
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
        warm = (
            warm_signal.exists() and (time.time() - float(warm_signal.read_text().strip())) < 3600
        )
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
            line = Text.assemble(
                (f'"{prompt}"', "white"),
                ("  → ", "dim"),
                (f"{len(hits)} hits", "bold cyan"),
                ("  @ ", "dim"),
                (scores, "magenta"),
            )
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
        for t in r.get("tags") or []:
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
    tbl.add_row(
        "grounded",
        f"[bold green]{_pct(health.get('grounded_rate'))}[/bold green]  [dim](primary)[/dim]",
    )
    tbl.add_row(
        f"hit / comp>{health.get('composite_score_threshold', 0.85):.2f}",
        f"{_pct(health.get('hit_rate'))}  /  "
        f"{_pct(health.get('top_composite_score_rate', health.get('strong_hit_rate')))}",
    )
    tbl.add_row(
        "composite p50",
        f"{_val(health.get('median_top_composite_score', health.get('median_top_score')), '{:.2f}')}  [dim]p50 lat[/dim] {_val(health.get('p50_latency_ms'), '{}ms')}",
    )
    tbl.add_row("sampled/fired", f"{_val(health.get('sampled'))} / {_val(health.get('fired'))}")
    tbl.add_row(
        "reask avoided",
        f"{_val(reask.get('reask_avoided'))} [dim]of[/dim] {_val(reask.get('considered'))}  [dim]reask%[/dim] {_pct(reask.get('reask_rate'))}",
    )
    return Panel(tbl, title="[bold green]recall quality[/bold green]", border_style="green")


_VERDICT_STYLE = {
    "ok": ("green", "bold green"),
    "weak": ("yellow", "bold yellow"),
    "unmeasured": ("yellow", "bold yellow"),
    "unused": ("red", "bold red"),
}


_VERDICT_LABEL = {
    "ok": "USEFUL",
    "weak": "WEAK",
    "unmeasured": "UNMEASURED",
    "unused": "NOT USED",
}


def _panel_verdict(state_dir: Path) -> Group:
    """Top-of-dashboard answer to 'does memo work and who reads it?'."""
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
    _, label_style = _VERDICT_STYLE.get(status, ("red", "bold red"))
    label = _VERDICT_LABEL.get(status, "NOT USED")
    grounded = data.get("grounded_rate")
    gr = "—" if grounded is None else f"{grounded * 100:.0f}%"
    consults = data.get("consults") or 0

    reads = [p["name"] for p in data.get("per_consumer", []) if p.get("reads")]
    silent = [p["name"] for p in data.get("per_consumer", []) if not p.get("reads")]

    kv = _kv()
    kv.add_row("verdict", Text(label, style=label_style))
    kv.add_row("grounded", f"{gr} · {consults} consults")
    kv.add_row("reading", Text(", ".join(reads) or "—", style="green"))
    if silent:
        kv.add_row("NOT reading", Text(", ".join(silent), style="bold red"))
    return Group(_hdr("does it work?"), kv, Text())


_recall_trend_cache: _TTLCache = _TTLCache(ttl_s=30.0)


def _read_eval_history(state_dir: Path, limit: int = 7) -> list[dict[str, Any]]:
    """Last `limit` valid entries of state_dir/eval/history.jsonl (oldest→newest).

    Lines look like {"ts": ..., "prec_at_k": float, "noise_at_k": float, "k": int,
    "labels": int, "source": "dream"}. Corrupt lines are skipped; a missing or
    unreadable file degrades to [].
    """
    path = state_dir / "eval" / "history.jsonl"
    if not path.is_file():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        _log.debug("dashboard: eval history read failed: %s", exc)
        return []
    out: list[dict[str, Any]] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            out.append(row)
    return out[-limit:]


def _grounding_citation_stats(state_dir: Path, limit: int = 2000) -> dict[str, Any]:
    """Citation ground truth from grounding.log: top-cited ids + never-cited count."""
    cited: Counter[str] = Counter()
    seen: set[str] = set()
    for r in read_grounding_log(state_dir, limit=limit):
        rid = str(r.get("recall_id") or "").strip()[:8]
        if not rid:
            continue
        seen.add(rid)
        if r.get("method") == "cited":
            cited[rid] += 1
    return {
        "top_cited": cited.most_common(5),
        "never_cited": len(seen - set(cited)),
        "seen": len(seen),
    }


def _panel_recall_trend(state_dir: Path) -> Group:
    """'Recall Quality' — prec@5 eval trend + citation ground truth from grounding.log."""
    key = str(state_dir)
    data = _recall_trend_cache.get(key)
    if data is None:
        prec: list[float] = []
        try:
            for row in _read_eval_history(state_dir, limit=7):
                v = row.get("prec_at_k")
                if isinstance(v, (int, float)):
                    prec.append(float(v))
        except Exception as exc:
            _log.debug("dashboard: eval history parse failed: %s", exc)
            prec = []
        try:
            cites = _grounding_citation_stats(state_dir)
        except Exception as exc:
            _log.debug("dashboard: grounding citation stats failed: %s", exc)
            cites = {"top_cited": [], "never_cited": 0, "seen": 0}
        data = {"prec": prec, "cites": cites}
        _recall_trend_cache.set(data, key)

    prec = data.get("prec") or []
    cites = data.get("cites") or {}
    top_cited = cites.get("top_cited") or []
    seen = int(cites.get("seen") or 0)
    never_cited = int(cites.get("never_cited") or 0)

    header = _hdr("recall quality")
    if not prec and not seen:
        return Group(header, Text("sin datos aún", style="dim italic"), Text())

    tbl = _kv()

    if prec:
        # Absolute 0..1 scale (precision metric) — no max-normalization,
        # so a flat 0.2 trend doesn't render as full-height blocks.
        spark = "".join(_SPARK[round(max(0.0, min(1.0, v)) * (len(_SPARK) - 1))] for v in prec)
        tbl.add_row(
            "prec@5",
            Text.assemble(
                (f"{prec[-1]:.2f}", "bold"),
                ("  ", ""),
                (spark, _ACCENT),
                (f"  ({len(prec)} runs)", "dim"),
            ),
        )
    else:
        tbl.add_row("prec@5", Text("sin datos aún", style="dim italic"))

    if top_cited:
        label = "top cited"
        for rid, n in top_cited:
            tbl.add_row(
                label,
                Text.assemble((f"[{rid}]", "bold yellow"), (f" ×{n}", "cyan")),  # noqa: RUF001
            )
            label = ""
    else:
        tbl.add_row("top cited", Text("sin datos aún", style="dim italic"))

    if seen:
        tbl.add_row(
            "never cited",
            Text.assemble(
                (str(never_cited), "bold"),
                (f" of {seen} recalled", "dim"),
            ),
        )
    else:
        tbl.add_row("never cited", Text("sin datos aún", style="dim italic"))
    return Group(header, tbl, Text())


def _panel_consumers(state_dir: Path) -> Group:
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
    tbl = Table.grid(padding=(0, 2))
    tbl.add_column(style=_ACCENT, width=14)
    tbl.add_column(justify="right", width=8)
    tbl.add_column(justify="right", width=6)
    tbl.add_column(justify="right", width=6)
    tbl.add_column(style="dim", width=8)
    tbl.add_row(
        Text("consumer", style="dim"),
        Text("consults", style="dim"),
        Text("hit", style="dim"),
        Text("grnd", style="dim"),
        Text("last", style="dim"),
    )
    if not consumers:
        tbl.add_row(Text("(no recall log data)", style="dim italic"), "", "", "", "")
    for c in consumers[:8]:
        name = (c.get("consumer") or "?")[:13]
        consults = str(c.get("consults") or "—")
        hit = c.get("hit_rate")
        hit_s = f"{hit * 100:.0f}%" if hit is not None else "—"
        grounded = c.get("grounded_rate")
        gr_s = f"{grounded * 100:.0f}%" if grounded is not None else "—"
        last = _human_age(c.get("last_seen"))
        tbl.add_row(name, consults, hit_s, gr_s, last)
    parts: list[Any] = [_hdr("consumers"), tbl]
    if silent:
        parts.append(Text.assemble(("not reading memo: ", "dim"), (", ".join(silent), "bold red")))
    parts.append(Text())
    return Group(*parts)


_utility_cache: _TTLCache = _TTLCache(ttl_s=30.0)


def _panel_utility(state_dir: Path) -> Group:
    """Panel summarizing context activity, recall, and grounding."""
    key = str(state_dir)
    cached = _utility_cache.get(key)
    if cached is None:
        total_tokens = 0
        usage_ids: set[str] = set()
        grounding_yes = 0
        grounding_no = 0

        try:
            for entry in read_context_cost_log(state_dir, limit=2000):
                total_tokens += entry.get("tokens_est", 0)
        except Exception as exc:
            _log.debug("dashboard: context_cost fetch failed: %s", exc)

        try:
            for entry in read_usage_log(state_dir, limit=2000):
                eid = entry.get("id", "")
                if eid:
                    usage_ids.add(eid)
        except Exception as exc:
            _log.debug("dashboard: usage_log fetch failed: %s", exc)

        try:
            for entry in read_grounding_log(state_dir, limit=2000):
                if grounding_used(entry):
                    grounding_yes += 1
                else:
                    grounding_no += 1
        except Exception as exc:
            _log.debug("dashboard: grounding fetch failed: %s", exc)

        fired = with_hits = above_composite_threshold = 0
        composite_threshold = 0.85
        try:
            health = recall_health(state_dir, limit=500)
            fired = health.get("fired", 0)
            with_hits = int(health.get("hit_rate", 0) * fired) if fired else 0
            composite_threshold = health.get("composite_score_threshold", composite_threshold)
            composite_rate = health.get(
                "top_composite_score_rate", health.get("strong_hit_rate", 0)
            )
            above_composite_threshold = int(composite_rate * fired) if fired else 0
        except Exception as exc:
            _log.debug("dashboard: recall_health fetch failed: %s", exc)

        composite_rate_pct = (above_composite_threshold / fired * 100) if fired else 0
        grounding_rate_pct = (
            (grounding_yes / (grounding_yes + grounding_no) * 100)
            if (grounding_yes + grounding_no)
            else 0
        )
        unique_mems = len(usage_ids)

        def _n(v: Any) -> str:
            return "—" if v is None else str(v)

        def _pct(v: float) -> str:
            return "—" if v <= 0 else f"{v:.0f}%"

        rows = [
            ("tokens injected", f"[yellow]{total_tokens:,}[/yellow]"),
            ("recall hooks", f"{_n(fired)} fired / {_n(with_hits)} hits"),
            (
                "top composite",
                f"{_pct(composite_rate_pct)} (final score >{composite_threshold:.2f})",
            ),
            ("memories surfaced", f"[cyan]{unique_mems}[/cyan] unique"),
            ("grounding", f"{_pct(grounding_rate_pct)} answered"),
        ]

        tbl = _kv(18)
        for label, value in rows:
            # `value` carries Rich console markup (e.g. "[yellow]…[/yellow]");
            # Text(value) would render the tags literally, so parse the markup
            # and keep bold as the base style.
            tbl.add_row(Text(label, style="dim"), Text.from_markup(value, style="bold"))
        cached = tbl
        _utility_cache.set(cached, key)

    return Group(_hdr("context activity"), cached, Text())
