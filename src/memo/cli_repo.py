"""`memo repo` command group — code-repo indexing + semantic search.

Extracted from cli.py (3a god-module decomposition). Registered onto the
root group in cli.py via `cli.add_command(repo_group)`.
"""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import click
from rich.panel import Panel
from rich.table import Table

from memo.cli_common import console
from memo.config import Config
from memo.trace import ambient_trace as _ambient_trace
from memo.util import stable_hash, utc_now_iso


def _repo_index_operational_receipt(out: dict[str, Any]) -> dict[str, Any]:
    name = str(out.get("name") or "repo")
    commit = str(out.get("commit_sha") or "")
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip("-") or "repo"
    receipt_id = f"{safe_name}/{commit[:12] or 'unknown'}"
    status = "partial" if int(out.get("errors") or 0) else "ok"
    receipt = {
        "schema": "memo.operational_receipt.v1",
        "source": "memo",
        "operation": "repo_index",
        "status": status,
        "uri": f"memo://repo-index/{receipt_id}",
        "generated_at": utc_now_iso(),
        "trace_id": _ambient_trace(),
        "repo": {
            "id": out.get("repo_id") or "",
            "name": name,
            "url": out.get("url") or "",
            "ref": out.get("ref") or "",
            "commit_sha": commit,
            "semantic_status": out.get("semantic_status") or "",
        },
        "counts": {
            "checked_files": int(out.get("checked_files") or 0),
            "indexed_files": int(out.get("indexed_files") or 0),
            "unchanged_files": int(out.get("unchanged_files") or 0),
            "deleted_files": int(out.get("deleted_files") or 0),
            "indexed_chunks": int(out.get("indexed_chunks") or 0),
            "indexed_lines": int(out.get("indexed_lines") or 0),
            "embedded_chunks": int(out.get("embedded_chunks") or 0),
            "pending_chunks": int(out.get("pending_chunks") or 0),
            "errors": int(out.get("errors") or 0),
        },
        "provenance": {
            "memo_uri": f"memo://repo/{out.get('repo_id') or safe_name}",
            "clone_path": out.get("clone_path") or "",
        },
    }
    receipt["content_hash"] = stable_hash(receipt)
    return receipt


@click.group(name="repo")
def repo_group() -> None:
    """Index and search Git repositories."""


def _run_with_repo_progress(
    thunk: Callable[[Callable[[str, dict[str, Any]], None]], dict[str, Any]],
) -> dict[str, Any]:
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
    )

    progress_state: dict[str, Any] = {
        "files_task": None,
        "semantic_task": None,
    }

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:

        def _on_progress(event: str, data: dict[str, Any]) -> None:
            if event == "clone_start":
                console.print(f"[dim]clone/fetch[/dim] {data['url']}")
            elif event == "clone_done":
                console.print(f"[dim]commit[/dim] {str(data['commit_sha'])[:8]}")
            elif event == "scan_start":
                progress_state["files_task"] = progress.add_task(
                    "scan/index files",
                    total=int(data.get("total") or 0),
                )
            elif event in {"file_skipped", "file_indexed", "file_error"}:
                task_id = progress_state.get("files_task")
                if task_id is not None:
                    progress.advance(task_id)
            elif event == "write_start":
                console.print(f"[dim]write sqlite[/dim] flush_batch={data.get('flush_batch')}")
            elif event == "write_done":
                console.print(
                    "[dim]write sqlite done[/dim] "
                    f"files={data.get('files')} chunks={data.get('chunks')} "
                    f"lines={data.get('lines')}"
                )
            elif event == "semantic_prepare":
                console.print(
                    f"[dim]embedder[/dim] preparing pending chunks for {data.get('repo') or 'repo'}"
                )
            elif event == "semantic_start":
                total = int(data.get("chunks") or 0)
                if total:
                    repo = data.get("repo") or "repo"
                    console.print(
                        f"[dim]embedder[/dim] {total} pending chunks; use --no-embeddings to skip"
                    )
                    progress_state["semantic_task"] = progress.add_task(
                        f"embedder {repo}",
                        total=total,
                    )
                else:
                    console.print("[dim]embedder[/dim] no pending chunks")
            elif event == "semantic_batch":
                task_id = progress_state.get("semantic_task")
                if task_id is not None:
                    progress.update(task_id, completed=int(data.get("completed") or 0))
            elif event == "semantic_done":
                task_id = progress_state.get("semantic_task")
                if task_id is not None:
                    total = int(data.get("total") or data.get("embedded") or 0)
                    progress.update(task_id, completed=total)

        return thunk(_on_progress)


@repo_group.command(name="index")
@click.argument("url")
@click.option("--name", default=None, help="Stable repo label. Defaults to repo basename.")
@click.option(
    "--ref", "ref_", default=None, help="Branch, tag, or commit. Default: current/default HEAD."
)
@click.option("--force", is_flag=True, help="Re-index even when commit/file hashes are unchanged.")
@click.option(
    "--refresh",
    is_flag=True,
    help="Scan file hashes even when HEAD is unchanged; only changed files are rewritten.",
)
@click.option(
    "--no-embeddings",
    is_flag=True,
    help="Write exact line/BM25 index only; run `memo repo embed` later.",
)
@click.option(
    "--include", multiple=True, help="Glob to include. Repeatable. Default: all text files."
)
@click.option("--exclude", multiple=True, help="Glob to exclude. Repeatable.")
@click.option("--max-file-bytes", default=None, type=int, help="Skip files above this byte size.")
@click.option("--json", "as_json", is_flag=True)
def repo_index_cmd(
    url: str,
    name: str | None,
    ref_: str | None,
    force: bool,
    refresh: bool,
    no_embeddings: bool,
    include: tuple[str, ...],
    exclude: tuple[str, ...],
    max_file_bytes: int | None,
    as_json: bool,
) -> None:
    """Clone/fetch URL and index included text files line-by-line."""
    from memo.memory import Memory

    mem = Memory(Config.from_env())
    try:
        if as_json:
            out = mem.repo_index(
                url,
                name=name,
                ref=ref_,
                force=force,
                refresh=refresh,
                with_embeddings=not no_embeddings,
                include=list(include),
                exclude=list(exclude),
                max_file_bytes=max_file_bytes,
            )
        else:

            def _run(progress: Callable[[str, dict[str, Any]], None]) -> dict[str, Any]:
                out = mem.repo_index(
                    url,
                    name=name,
                    ref=ref_,
                    force=force,
                    refresh=refresh,
                    with_embeddings=not no_embeddings,
                    include=list(include),
                    exclude=list(exclude),
                    max_file_bytes=max_file_bytes,
                    progress=progress,
                )
                return out

            out = _run_with_repo_progress(_run)
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc
    finally:
        mem.close()

    receipt = _repo_index_operational_receipt(out)
    out = {**out, "operational_receipt": receipt}
    if as_json:
        click.echo(json.dumps(out, ensure_ascii=False, indent=2))
        return
    console.print(
        f"[green]done[/green] repo={out['name']} commit={out['commit_sha'][:8]} "
        f"files={out['indexed_files']} unchanged={out['unchanged_files']} "
        f"deleted={out['deleted_files']} chunks={out['indexed_chunks']} "
        f"lines={out['indexed_lines']} embedded={out['embedded_chunks']} "
        f"model={out.get('model_chunks', 0)} cached={out.get('cached_chunks', 0)} "
        f"pending={out['pending_chunks']} status={out['semantic_status']} "
        f"errors={out['errors']}",
        highlight=False,
    )
    console.print(f"[dim]receipt[/dim] {receipt['uri']}")


@repo_group.command(name="embed")
@click.argument("repo")
@click.option("--force", is_flag=True, help="Re-embed all chunks, not only pending chunks.")
@click.option("--json", "as_json", is_flag=True)
def repo_embed_cmd(repo: str, force: bool, as_json: bool) -> None:
    """Embed pending chunks for an already indexed repo."""
    from memo.memory import Memory

    mem = Memory(Config.from_env())
    try:
        if as_json:
            out = mem.repo_embed(repo, force=force)
        else:
            out = _run_with_repo_progress(
                lambda progress: mem.repo_embed(repo, force=force, progress=progress),
            )
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc
    finally:
        mem.close()

    if as_json:
        click.echo(json.dumps(out, ensure_ascii=False, indent=2))
        return
    console.print(
        f"[green]done[/green] repo={out['name']} embedded={out['embedded_chunks']} "
        f"model={out.get('model_chunks', 0)} cached={out.get('cached_chunks', 0)} "
        f"total={out['total_chunks']} pending={out['pending_chunks']} "
        f"status={out['semantic_status']}",
        highlight=False,
    )


@repo_group.command(name="status")
@click.argument("repo")
@click.option("--json", "as_json", is_flag=True)
def repo_status_cmd(repo: str, as_json: bool) -> None:
    """Show exact and semantic index status for one repo."""
    from memo.memory import Memory

    mem = Memory(Config.from_env())
    try:
        out = mem.repo_status(repo)
    finally:
        mem.close()
    if out is None:
        console.print(f"[red]not found:[/red] {repo}")
        sys.exit(1)
    if as_json:
        click.echo(json.dumps(out, ensure_ascii=False, indent=2))
        return
    console.print(
        f"[bold]{out['name']}[/bold] {out['commit_sha'][:8]} "
        f"status={out['semantic_status']} files={out['files']} "
        f"lines={out['lines']} chunks={out['chunks']} "
        f"embedded={out['embedded_chunks']} pending={out['pending_chunks']}"
    )


@repo_group.command(name="list")
@click.option("--limit", default=100, type=int, show_default=True)
@click.option("--json", "as_json", is_flag=True)
def repo_list_cmd(limit: int, as_json: bool) -> None:
    """List indexed repositories."""
    from memo.memory import Memory

    mem = Memory(Config.from_env())
    try:
        rows = mem.repo_list(limit=limit)
    finally:
        mem.close()
    if as_json:
        click.echo(json.dumps(rows, ensure_ascii=False, indent=2))
        return
    if not rows:
        console.print("[dim]no repos indexed[/dim]")
        return
    tbl = Table(show_lines=False, expand=True)
    tbl.add_column("indexed", width=20)
    tbl.add_column("name", width=18)
    tbl.add_column("ref", width=12)
    tbl.add_column("commit", width=10)
    tbl.add_column("url", overflow="fold")
    for r in rows:
        tbl.add_row(
            str(r.get("indexed_at", ""))[:19],
            r["name"],
            r["ref"],
            r["commit_sha"][:8],
            r["url"],
        )
    console.print(tbl)


@repo_group.command(name="search")
@click.argument("query")
@click.option("--limit", default=10, type=int, show_default=True)
@click.option("--repo", default=None, help="Repo id/name/url filter.")
@click.option("--path", "path_glob", default=None, help="Path glob filter, e.g. 'src/**/*.py'.")
@click.option(
    "--mode",
    default="hybrid",
    type=click.Choice(["hybrid", "lexical", "unified", "vec", "bm25", "line"]),
    show_default=True,
)
@click.option(
    "--scope",
    default="all",
    type=click.Choice(["all", "production", "tests", "vendor"]),
    show_default=True,
)
@click.option("--explain", is_flag=True, help="Show contributing rank channels.")
@click.option("--json", "as_json", is_flag=True)
def repo_search_cmd(
    query: str,
    limit: int,
    repo: str | None,
    path_glob: str | None,
    mode: str,
    scope: str,
    explain: bool,
    as_json: bool,
) -> None:
    """Search indexed repos by semantic chunks, keyword chunks, or exact lines."""
    from memo.memory import Memory

    mem = Memory(Config.from_env())
    try:
        hits = mem.repo_search(
            query,
            limit=limit,
            repo=repo,
            path=path_glob,
            mode=mode,
            scope=scope,
        )
    finally:
        mem.close()
    if as_json:
        click.echo(json.dumps([h.to_dict() for h in hits], ensure_ascii=False, indent=2))
        return
    if not hits:
        console.print("[dim]no results[/dim]")
        return
    tbl = Table(show_lines=False, expand=True)
    tbl.add_column("score", justify="right", width=7)
    tbl.add_column("repo", width=16)
    tbl.add_column("loc", overflow="fold")
    tbl.add_column("text", overflow="fold")
    for h in hits:
        preview = " ".join((h.text or "").split())[:180]
        tbl.add_row(
            f"{h.score:.3f}" if h.score is not None else "—",
            h.repo_name,
            f"{h.path}:{h.line_start}-{h.line_end}",
            (
                f"{preview}\n[dim]channels={','.join(h.channel_scores) or h.match_type}[/dim]"
                if explain
                else preview
            ),
        )
    console.print(tbl)


@repo_group.command(name="artifact")
@click.argument("repo")
@click.argument("kind", type=click.Choice(["generation", "change_signals"]))
@click.argument(
    "destination",
    type=click.Path(file_okay=False, path_type=Path),
)
@click.option("--json", "as_json", is_flag=True)
def repo_artifact_cmd(repo: str, kind: str, destination: Path, as_json: bool) -> None:
    """Export a verified, content-addressed repo artifact."""
    from memo.memory import Memory

    mem = Memory(Config.from_env())
    try:
        out = mem.repo_export_artifact(repo, kind, destination)
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc
    finally:
        mem.close()
    if as_json:
        click.echo(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        console.print(
            f"[green]exported[/green] {kind} digest={str(out['digest'])[:12]} "
            f"artifact={out['artifact']}"
        )


@repo_group.command(name="watch")
@click.argument("repo")
@click.option("--delay", type=click.FloatRange(min=0.05), default=1.0, show_default=True)
@click.option("--debug", is_flag=True)
def repo_watch_cmd(repo: str, delay: float, debug: bool) -> None:
    """Watch a managed clone and incrementally refresh changed files."""
    from memo.repo_watcher import run_repo_watcher

    run_repo_watcher(repo, delay=delay, debug=debug)


@repo_group.command(name="get")
@click.argument("repo")
@click.argument("path")
@click.option("--start", default=None, type=int, help="First line to return.")
@click.option("--end", default=None, type=int, help="Last line to return.")
@click.option("--json", "as_json", is_flag=True)
def repo_get_cmd(
    repo: str,
    path: str,
    start: int | None,
    end: int | None,
    as_json: bool,
) -> None:
    """Fetch one indexed repo file or line range."""
    from memo.memory import Memory

    mem = Memory(Config.from_env())
    try:
        out = mem.repo_get_file(repo, path, start=start, end=end)
    finally:
        mem.close()
    if out is None:
        console.print(f"[red]not found:[/red] {repo} {path}")
        sys.exit(1)
    if as_json:
        click.echo(json.dumps(out, ensure_ascii=False, indent=2))
        return
    console.print(
        Panel.fit(
            out["text"],
            title=f"{out['repo_name']}:{out['path']}:{out['start']}-{out['end']}",
            border_style="cyan",
        )
    )


@repo_group.command(name="delete")
@click.argument("repo")
@click.option("--yes", is_flag=True, help="Skip confirmation.")
@click.option(
    "--keep-clone", is_flag=True, help="Only delete index rows; keep memo's managed clone."
)
def repo_delete_cmd(repo: str, yes: bool, keep_clone: bool) -> None:
    """Delete one indexed repo."""
    from memo.memory import Memory

    mem = Memory(Config.from_env())
    try:
        if not yes:
            click.confirm(f"Delete indexed repo {repo!r}?", abort=True)
        ok = mem.repo_delete(repo, remove_clone=not keep_clone)
    finally:
        mem.close()
    console.print(f"[{'green' if ok else 'red'}]{'deleted' if ok else 'not found'}[/]: {repo}")
    if not ok:
        sys.exit(1)
