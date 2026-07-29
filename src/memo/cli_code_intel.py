"""`memo code-nudge` + `memo code-health` — graph↔memory awareness commands.

Thin consumers of the :mod:`memo.code_intel` engine (read-only, fail-open,
repo_id-gated — the invariants live there, not here):

- ``code-nudge`` surfaces the memories whose ``code_refs`` cite the files a
  commit touched. Silent (exit 0, no output) whenever there is nothing to
  say — it runs from the post-commit hook and must never dirty a clean
  commit. Pure SQL against the store; budget <300 ms, no MLX.
- ``code-health`` joins the codegraph index with the memory corpus: the last
  nightly drift receipt, dead knowledge (memories citing symbols nobody
  calls), and the top call hubs no memory documents. A missing index
  degrades to a notice, exit 0.

Registered onto the root group in cli.py via ``cli.add_command``.
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
from typing import Any

import click
from rich.panel import Panel

from memo.cli_common import console
from memo.cli_common import get_memory as _get_memory
from memo.config import Config

# Nudge output cap: enough to be useful after a commit, short enough to stay
# out of the way in a terminal that just printed the commit summary.
_NUDGE_CAP = 3
_GIT_TIMEOUT_S = 2.0
# Hubs reported by code-health (same top-N spirit as cli_code_facts hubs).
_HUB_LIMIT = 10
# Receipt list keys summarized as counts ("repaired" lands with auto-repair).
_DRIFT_LIST_KEYS = ("outdated", "partial", "repaired")


def _changed_files(rev: str) -> list[str]:
    """Files touched by ``rev`` per ``git diff-tree``; [] on any git failure."""
    try:
        proc = subprocess.run(
            ["git", "diff-tree", "--no-commit-id", "--name-only", "-r", rev],
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if proc.returncode != 0:
        return []
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


@click.command(name="code-nudge")
@click.option(
    "--commit",
    "rev",
    default="HEAD",
    show_default=True,
    help="Commit whose touched files are matched against memories.",
)
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON.")
def code_nudge_cmd(rev: str, as_json: bool) -> None:
    """Surface memories citing the files a commit touched.

    Reads the commit's file list from git and prints up to 3 memories whose
    code_refs cite them — one `🧠 [id] title` line each. Silent (exit 0, no
    output) when nothing matches or git fails, so the post-commit hook never
    dirties a clean commit. Pure SQL — no MLX, no graph queries.

    Example: memo code-nudge --commit HEAD~1
    """
    from memo import code_intel

    files = _changed_files(rev)
    if not files:
        return
    mem = _get_memory(Config.from_env())
    hits = code_intel.memories_citing(mem.store._conn, paths=files, limit=_NUDGE_CAP)
    if not hits:
        return
    if as_json:
        payload = [{"id": hit["id"], "title": hit["title"]} for hit in hits]
        click.echo(json.dumps(payload, ensure_ascii=False))
        return
    for hit in hits:
        click.echo(f"🧠 [{hit['id'][:8]}] {hit['title']}")


def _drift_summary(cfg: Config) -> dict[str, Any]:
    """code_drift section of the last dream receipt, as counts; fail-open."""
    path = cfg.state_dir / "dream" / "last.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        section = data.get("code_drift") if isinstance(data, dict) else None
        if not isinstance(section, dict):
            return {"status": "no-receipt"}
        summary: dict[str, Any] = {
            "status": str(section.get("status") or "unknown"),
            "scanned": int(section.get("scanned") or 0),
        }
        for key in _DRIFT_LIST_KEYS:
            value = section.get(key)
            summary[key] = len(value) if isinstance(value, list) else 0
        return summary
    except (OSError, ValueError, TypeError):
        return {"status": "no-receipt"}


def _cited_memories(store_conn: Any) -> list[dict[str, Any]]:
    """Non-reference memories carrying code_refs, with their raw ref dicts."""
    try:
        rows = store_conn.execute(
            "SELECT id, title, extra_json FROM meta WHERE type != 'reference' "
            "AND json_extract(extra_json, '$.code_refs') IS NOT NULL"
        ).fetchall()
    except sqlite3.Error:
        return []
    out: list[dict[str, Any]] = []
    for row in rows:
        try:
            extra = json.loads(row["extra_json"] or "{}")
        except json.JSONDecodeError:
            continue
        refs = extra.get("code_refs")
        if not isinstance(refs, list):
            continue
        out.append(
            {
                "id": str(row["id"]),
                "title": str(row["title"] or ""),
                "refs": [ref for ref in refs if isinstance(ref, dict)],
            }
        )
    return out


def _ref_symbols(refs: list[dict[str, Any]], db_repo_id: str) -> set[str]:
    """Symbols those refs cite, minus file refs and other repos' refs.

    The repo claim uses the engine's :func:`code_intel.ref_repo_claim` (field,
    else ``codegraph://`` uri host) — the same gate ``ref_status`` applies, so
    refs minted by ``memo code-facts --project`` (uri claim only, no field)
    are never judged against the wrong DB.
    """
    from memo import code_intel

    symbols: set[str] = set()
    for ref in refs:
        if str(ref.get("kind") or "").strip().lower() == "file":
            continue
        repo = code_intel.ref_repo_claim(ref)
        if repo and repo != db_repo_id:
            continue
        symbol = str(ref.get("label") or "").strip()
        if symbol:
            symbols.add(symbol)
    return symbols


def _dead_knowledge(
    graph: sqlite3.Connection, db_repo_id: str, cited: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Memories citing symbols that exist in the graph with zero callers.

    A symbol absent from the graph is drift (the nightly pass's business),
    not dead knowledge — only symbols present with 0 incoming ``calls``
    edges count.
    """
    all_symbols = sorted({s for memory in cited for s in _ref_symbols(memory["refs"], db_repo_id)})
    if not all_symbols:
        return []
    marks = ", ".join("?" * len(all_symbols))
    try:
        rows = graph.execute(
            "SELECT n.name, COUNT(e.source) FROM nodes n "  # noqa: S608 — placeholders only
            "LEFT JOIN edges e ON e.target = n.id AND e.kind = 'calls' "
            f"WHERE n.kind != 'file' AND n.name IN ({marks}) GROUP BY n.name",
            all_symbols,
        ).fetchall()
    except sqlite3.Error:
        return []
    callers = {str(row[0]): int(row[1]) for row in rows}
    out: list[dict[str, Any]] = []
    for memory in cited:
        dead = sorted(s for s in _ref_symbols(memory["refs"], db_repo_id) if callers.get(s) == 0)
        if dead:
            out.append({"id": memory["id"], "title": memory["title"], "symbols": dead})
    return out


def _hubs_without_memory(graph: sqlite3.Connection, store_conn: Any) -> list[dict[str, Any]]:
    """Top src/ call hubs (by incoming ``calls`` edges) that no memory cites."""
    from memo import code_intel

    try:
        rows = graph.execute(
            "SELECT t.name, t.qualified_name, t.file_path, COUNT(*) AS n "
            "FROM edges e JOIN nodes t ON e.target = t.id "
            "WHERE e.kind = 'calls' AND t.file_path LIKE 'src/%' "
            "GROUP BY t.id ORDER BY n DESC, t.qualified_name LIMIT ?",
            (_HUB_LIMIT,),
        ).fetchall()
    except sqlite3.Error:
        return []
    out: list[dict[str, Any]] = []
    for name, qualified, file_path, n in rows:
        symbol = str(name or "")
        if not symbol or code_intel.memories_citing(store_conn, symbols={symbol}, limit=1):
            continue
        out.append(
            {
                "name": symbol,
                "qualified_name": str(qualified or symbol),
                "file_path": str(file_path or ""),
                "callers": int(n),
            }
        )
    return out


def _live_counts(
    graph: sqlite3.Connection, db_repo_id: str, cited: list[dict[str, Any]]
) -> dict[str, int]:
    """Live re-verification of every cited ref via ``code_intel.ref_status``."""
    from memo import code_intel

    counts = {"vigente": 0, "desaparecido": 0, "no_verificable": 0}
    for memory in cited:
        for ref in memory["refs"]:
            status = code_intel.ref_status(graph, ref, db_repo_id)
            counts["no_verificable" if status is None else status] += 1
    return counts


def _render_health(report: dict[str, Any]) -> None:
    drift = report["drift"]
    if drift.get("status") == "no-receipt":
        console.print("[dim]drift: sin receipt nocturno — corré `memo dream run`[/dim]")
    else:
        console.print(
            f"drift: {drift['status']} — {drift['scanned']} scanned, "
            f"{drift['outdated']} outdated, {drift['partial']} partial, "
            f"{drift['repaired']} repaired"
        )
    live = drift.get("live")
    if live:
        console.print(
            f"live: {live['vigente']} vigentes, {live['desaparecido']} desaparecidos, "
            f"{live['no_verificable']} no verificables"
        )
    dead = report["dead_knowledge"]
    if dead:
        console.print(f"[yellow]dead knowledge ({len(dead)}):[/yellow]")
        for entry in dead:
            symbols = ", ".join(entry["symbols"])
            console.print(f"  [{entry['id'][:8]}] {entry['title']} — 0 callers: {symbols}")
    else:
        console.print("[dim]dead knowledge: none[/dim]")
    hubs = report["hubs_sin_memoria"]
    if hubs:
        console.print(f"[yellow]hubs sin memoria ({len(hubs)}):[/yellow]")
        for hub in hubs:
            console.print(f"  {hub['name']} ({hub['callers']} callers) — {hub['file_path']}")
    else:
        console.print("[dim]hubs sin memoria: none[/dim]")


@click.command(name="code-health")
@click.option("--live", is_flag=True, help="Re-verify every code ref against the live index.")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON.")
def code_health_cmd(live: bool, as_json: bool) -> None:
    """Health report joining the codegraph index with the memory corpus.

    Three sections: the last nightly drift receipt (memo dream run), dead
    knowledge (memories citing symbols nobody calls), and the top call hubs
    no memory documents. --live re-verifies every code ref against the live
    index. Read-only over codegraph.db; a missing index degrades to a
    notice, exit 0.

    Example: memo code-health --live --json
    """
    from memo import code_intel

    cfg = Config.from_env()
    report: dict[str, Any] = {
        "drift": _drift_summary(cfg),
        "dead_knowledge": [],
        "hubs_sin_memoria": [],
    }
    opened = code_intel.open_graph()
    if opened is None:
        if as_json:
            click.echo(json.dumps(report, ensure_ascii=False, indent=2))
        else:
            console.print(Panel("codegraph index no disponible", border_style="yellow"))
        return
    graph, db_repo_id = opened
    mem = _get_memory(cfg)
    try:
        cited = _cited_memories(mem.store._conn)
        if live:
            report["drift"]["live"] = _live_counts(graph, db_repo_id, cited)
        report["dead_knowledge"] = _dead_knowledge(graph, db_repo_id, cited)
        report["hubs_sin_memoria"] = _hubs_without_memory(graph, mem.store._conn)
    finally:
        graph.close()
    if as_json:
        click.echo(json.dumps(report, ensure_ascii=False, indent=2))
        return
    _render_health(report)
