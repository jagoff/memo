#!/usr/bin/env python3
"""Maintainer-facing adoption report for memo (Tier 1 — passive proxies).

Pulls signals that already exist publicly — no code ships to users, no phone-home:

  - GitHub:  stars / forks / watchers, and the 14-day traffic window
             (clones + unique cloners, views + unique viewers, referrers).
  - PyPI:    recent + last-month download counts for ``mlx-memo``.

Two hard caveats this report makes loud rather than hiding:

  1. **Clone counts are polluted.** memo installs and auto-updates via
     ``git+https://github.com/jagoff/memo.git``, and CI clones on every run,
     and you dev across several Macs. So ``clones`` is NOT users. ``unique
     cloners`` is closer but still includes CI runners and your own machines.
     Treat ``unique viewers`` and ``PyPI unique-ish downloads`` as the better
     interest proxies, and Tier-2 deduped heartbeats as the real active count.

  2. **GitHub traffic is a rolling 14-day window.** Anything older is gone from
     the API. This script snapshots each run to a JSONL log so a longer trend
     accumulates locally — run it on a schedule (launchd/cron) to build history.

Usage:
    uv run --no-sync python scripts/telemetry_report.py            # render + snapshot
    uv run --no-sync python scripts/telemetry_report.py --json     # machine output
    uv run --no-sync python scripts/telemetry_report.py --no-snapshot
    uv run --no-sync python scripts/telemetry_report.py --log ~/somewhere.jsonl

Requires: ``gh`` CLI authenticated with push access to the repo (needed for the
traffic endpoints), and network access to pypistats.org.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from typing import Any

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

REPO = "jagoff/memo"
PYPI_PACKAGE = "mlx-memo"
DEFAULT_LOG = pathlib.Path.home() / ".memo" / "telemetry-snapshots.jsonl"
PYPI_RETRIES = 4
PYPI_BACKOFF_S = 3.0


def _gh_json(path: str, jq: str | None = None) -> Any:
    """Call ``gh api <path>`` and return parsed JSON, or None on any failure."""
    cmd = ["gh", "api", path]
    if jq:
        cmd += ["--jq", jq]
    # gh colorizes even piped stdout on some setups → strip color + disable it,
    # else json.loads chokes on ANSI escapes.
    env = {**os.environ, "NO_COLOR": "1", "CLICOLOR": "0", "GH_NO_UPDATE_NOTIFIER": "1"}
    try:
        cp = subprocess.run(cmd, capture_output=True, text=True, timeout=30, env=env)
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        return {"_error": str(exc)}
    if cp.returncode != 0:
        return {"_error": cp.stderr.strip() or f"gh api {path} rc={cp.returncode}"}
    out = _ANSI_RE.sub("", cp.stdout).strip()
    if not out:
        return None
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        return {"_error": f"non-JSON from gh api {path}"}


def fetch_github() -> dict[str, Any]:
    repo = _gh_json(f"repos/{REPO}") or {}
    if "_error" in repo:
        return {"_error": repo["_error"]}
    clones = _gh_json(f"repos/{REPO}/traffic/clones") or {}
    views = _gh_json(f"repos/{REPO}/traffic/views") or {}
    referrers = _gh_json(f"repos/{REPO}/traffic/popular/referrers") or []
    paths = _gh_json(f"repos/{REPO}/traffic/popular/paths") or []
    return {
        "stars": repo.get("stargazers_count"),
        "forks": repo.get("forks_count"),
        "watchers": repo.get("subscribers_count"),
        "open_issues": repo.get("open_issues_count"),
        "clones_total": clones.get("count"),
        "clones_unique": clones.get("uniques"),
        "views_total": views.get("count"),
        "views_unique": views.get("uniques"),
        "referrers": [
            {"source": r.get("referrer"), "unique": r.get("uniques")}
            for r in referrers
            if isinstance(r, dict)
        ][:6],
        "top_paths": [
            {"path": p.get("path"), "unique": p.get("uniques")}
            for p in paths
            if isinstance(p, dict)
        ][:6],
    }


def _http_json(url: str) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": "memo-telemetry-report"})  # noqa: S310
    with urllib.request.urlopen(req, timeout=20) as resp:  # noqa: S310 (trusted host)
        return json.loads(resp.read().decode())


def fetch_pypi() -> dict[str, Any]:
    """pypistats recent (last day/week/month). Retries on 429 rate-limit."""
    url = f"https://pypistats.org/api/packages/{PYPI_PACKAGE}/recent"
    last_err = ""
    for attempt in range(PYPI_RETRIES):
        try:
            data = _http_json(url)
            d = data.get("data", {})
            return {
                "last_day": d.get("last_day"),
                "last_week": d.get("last_week"),
                "last_month": d.get("last_month"),
            }
        except urllib.error.HTTPError as exc:
            last_err = f"HTTP {exc.code}"
            if exc.code == 429 and attempt < PYPI_RETRIES - 1:
                time.sleep(PYPI_BACKOFF_S * (attempt + 1))
                continue
            break
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_err = str(exc)
            break
    return {"_error": f"pypistats unavailable ({last_err})"}


def build_snapshot() -> dict[str, Any]:
    return {
        "ts": datetime.now(UTC).isoformat(timespec="seconds"),
        "github": fetch_github(),
        "pypi": fetch_pypi(),
    }


def append_snapshot(log: pathlib.Path, snap: dict[str, Any]) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(snap, ensure_ascii=False) + "\n")


def read_prev_snapshot(log: pathlib.Path) -> dict[str, Any] | None:
    if not log.exists():
        return None
    prev: dict[str, Any] | None = None
    with log.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                prev = json.loads(line)
            except json.JSONDecodeError:
                continue
    return prev


def _delta(cur: Any, prev: Any) -> str:
    if not isinstance(cur, (int, float)) or not isinstance(prev, (int, float)):
        return ""
    d = cur - prev
    if d == 0:
        return " (—)"
    return f" ([green]+{d}[/green])" if d > 0 else f" ([red]{d}[/red])"


def render(snap: dict[str, Any], prev: dict[str, Any] | None, log: pathlib.Path) -> None:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    console = Console(force_terminal=True)
    gh = snap["github"]
    pypi = snap["pypi"]
    pgh = (prev or {}).get("github", {})
    ppypi = (prev or {}).get("pypi", {})

    if "_error" in gh:
        console.print(f"[red]GitHub:[/red] {gh['_error']}")
        gh = {}

    t = Table(title="memo — adoption proxies (Tier 1)", title_style="bold", expand=False)
    t.add_column("signal", style="cyan")
    t.add_column("value", justify="right")
    t.add_column("read as", style="dim")

    def row(label: str, key: str, src: dict[str, Any], psrc: dict[str, Any], note: str) -> None:
        val = src.get(key)
        shown = "—" if val is None else f"{val}{_delta(val, psrc.get(key))}"
        t.add_row(label, shown, note)

    row("★ stars", "stars", gh, pgh, "vanity / discovery")
    row("forks", "forks", gh, pgh, "dev interest")
    row("watchers", "watchers", gh, pgh, "committed followers")
    t.add_section()
    row("views (14d)", "views_total", gh, pgh, "README hits")
    row("unique viewers (14d)", "views_unique", gh, pgh, "[green]best interest proxy[/green]")
    t.add_section()
    row(
        "clones (14d)", "clones_total", gh, pgh, "[yellow]polluted: install+CI+auto-update[/yellow]"
    )
    row(
        "unique cloners (14d)",
        "clones_unique",
        gh,
        pgh,
        "[yellow]closer, still CI+your Macs[/yellow]",
    )
    t.add_section()
    if "_error" in pypi:
        t.add_row("PyPI downloads", f"[dim]{pypi['_error']}[/dim]", "retry later (429)")
    else:
        row("PyPI last day", "last_day", pypi, ppypi, "installs, not users")
        row("PyPI last week", "last_week", pypi, ppypi, "installs, incl. mirrors/CI")
        row("PyPI last month", "last_month", pypi, ppypi, "installs, incl. mirrors/CI")

    console.print(t)

    refs = gh.get("referrers") or []
    if refs:
        rt = Table(title="top referrers (14d)", title_style="dim", box=None)
        rt.add_column("source", style="cyan")
        rt.add_column("unique", justify="right")
        for r in refs:
            rt.add_row(str(r.get("source")), str(r.get("unique")))
        console.print(rt)

    prev_ts = (prev or {}).get("ts", "—")
    console.print(
        Panel(
            f"[bold]No metric here is 'active users.'[/bold] Clones are polluted by memo's "
            f"own git install/auto-update + CI. Unique viewers + PyPI are the honest interest "
            f"floor. For a deduped [italic]active[/italic] count, ship Tier 2 (heartbeat).\n"
            f"Snapshot log: [cyan]{log}[/cyan]  •  prev snapshot: [dim]{prev_ts}[/dim]  •  "
            f"traffic API = rolling 14d, so run this on a schedule to build history.",
            title="how to read this",
            border_style="yellow",
        )
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="memo adoption report (Tier 1 passive proxies)")
    ap.add_argument("--json", action="store_true", help="emit raw snapshot JSON, no render")
    ap.add_argument("--no-snapshot", action="store_true", help="do not append to the log")
    ap.add_argument("--log", type=pathlib.Path, default=DEFAULT_LOG, help="snapshot JSONL path")
    args = ap.parse_args(argv)

    snap = build_snapshot()
    prev = read_prev_snapshot(args.log)
    if not args.no_snapshot:
        append_snapshot(args.log, snap)

    if args.json:
        json.dump(snap, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
        return 0

    render(snap, prev, args.log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
