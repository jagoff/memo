"""`memo cross-dedup` — detect overlapping knowledge between memo and memflow.

Scans memo memories against memflow's event store and reports candidates
where the same fact/decision appears in both systems. Read-only: reports
only, never modifies either system.

Algorithm:
  1. Load memo memories (excluding reference tier, up to --limit).
  2. For each, query memflow with "<title>: <body_snippet>" via `memflow query`.
  3. Report hits where memflow score ≥ --threshold and kind not in
     conversation/channel (durable knowledge kinds only).

Examples:
  memo cross-dedup
  memo cross-dedup --threshold 0.55 --limit 100
  memo cross-dedup --json
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from typing import Any

import click

from memo.cli_common import console
from memo.cli_common import get_memory as _get_memory
from memo.config import Config

_SKIP_KINDS = {"conversation", "channel"}


def _memflow_project_root() -> str | None:
    """Resolve memflow project root: env var, then common default."""
    if "MEMFLOW_PROJECT_ROOT" in os.environ:
        return os.environ["MEMFLOW_PROJECT_ROOT"]
    # Common default on this machine
    default = os.path.expanduser("~/repos/memflow")
    if os.path.isdir(default):
        return default
    return None


def _memflow_query(
    query: str,
    *,
    k: int = 3,
    timeout: float = 5.0,
) -> list[dict[str, Any]]:
    """Run `memflow query` and parse the output.

    Returns list of {score, kind, path, snippet}.
    """
    env = {**os.environ}
    mf_root = _memflow_project_root()
    if mf_root:
        env["MEMFLOW_PROJECT_ROOT"] = mf_root
    try:
        proc = subprocess.run(
            ["memflow", "query", query, "-k", str(k)],
            capture_output=True, text=True, timeout=timeout, env=env,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []

    lines = proc.stdout.splitlines()
    results: list[dict[str, Any]] = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        # Pattern: "<score> <kind> <path>"
        m = re.match(r"^([0-9.]+)\s+(\S+)\s+(.+)$", line)
        if m:
            score = float(m.group(1))
            kind = m.group(2)
            path = m.group(3)
            snippet = lines[i + 1].strip() if i + 1 < len(lines) else ""
            results.append({"score": score, "kind": kind, "path": path, "snippet": snippet[:120]})
            i += 2
        else:
            i += 1
    return results


@click.command(name="cross-dedup")
@click.option("--threshold", type=float, default=0.50,
              help="Memflow score floor to flag an overlap (default 0.50).")
@click.option("--limit", type=int, default=50,
              help="Max memo memories to scan (default 50).")
@click.option("--type", "type_", default=None,
              help="Restrict memo side to a single type (e.g. decision, fact).")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON.")
@click.option("--verbose", is_flag=True, help="Show snippet text in table.")
def cross_dedup_cmd(
    threshold: float,
    limit: int,
    type_: str | None,
    as_json: bool,
    verbose: bool,
) -> None:
    """Detect knowledge duplicated between memo and memflow.

    Scans memo memories and searches for matching content in memflow.
    Read-only — reports only, never modifies either system.

    Examples:
      memo cross-dedup
      memo cross-dedup --threshold 0.55 --limit 100 --json
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    # Fetch more than limit to account for reference-tier entries that get filtered.
    fetch = max(limit * 4, 200) if not type_ else limit
    recs = mem.list(limit=fetch, type_=type_)
    if not type_:
        recs = [r for r in recs if r.type != "reference"]
    hits = recs[:limit]

    overlaps: list[dict[str, Any]] = []

    if not as_json:
        console.print(f"[bold]memo cross-dedup[/bold] — scanning {len(hits)} memories against memflow")

    for rec in hits:
        title = getattr(rec, "title", "") or ""
        body = getattr(rec, "body", "") or ""
        rid = getattr(rec, "id", "") or ""
        rtype = getattr(rec, "type", "") or ""

        query = f"{title}: {body[:100]}" if body else title
        if not query.strip():
            continue

        mf_hits = _memflow_query(query, k=3, timeout=5.0)

        for mfh in mf_hits:
            if mfh["score"] < threshold:
                continue
            if mfh["kind"] in _SKIP_KINDS:
                continue
            overlaps.append({
                "memo_id": rid[:8],
                "memo_type": rtype,
                "memo_title": title,
                "memflow_kind": mfh["kind"],
                "memflow_path": mfh["path"].split("/")[-1][:40],
                "score": round(mfh["score"], 3),
                "snippet": mfh["snippet"],
            })

    if as_json:
        click.echo(json.dumps(overlaps, ensure_ascii=False, indent=2))
        return

    if not overlaps:
        console.print("[dim]no overlaps found above threshold[/dim]")
        return

    console.print(f"\n[bold]{len(overlaps)} potential overlap(s) found[/bold] (threshold={threshold})\n")

    for o in overlaps:
        console.print(
            f"  [cyan]{o['memo_id']}[/cyan] [{o['memo_type']}] [yellow]{o['memo_title'][:50]}[/yellow]"
        )
        console.print(
            f"    → memflow [{o['memflow_kind']}] score=[green]{o['score']}[/green]  {o['memflow_path']}"
        )
        if verbose and o["snippet"]:
            console.print(f"    [dim]{o['snippet'][:100]}[/dim]")
