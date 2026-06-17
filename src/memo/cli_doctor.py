from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from memo.cli_common import console
from memo.cli_diag import _db_health_report, _doctor_report, _recall_daemon_health
from memo.cli_runtime import _print_runtime_install_report, _runtime_install_report
from memo.config import Config


@click.command()
@click.option("--gc", "do_gc", is_flag=True, help="Detect orphans between store and disk.")
@click.option(
    "--fix",
    is_flag=True,
    help="With --gc: drop orphan store rows. .md files are never deleted automatically.",
)
@click.option(
    "--db", "check_db", is_flag=True, help="Run read-only integrity checks on managed sqlite DBs."
)
@click.option(
    "--strict-runtime",
    is_flag=True,
    help="Exit non-zero if memo/memo-mcp are not running from an isolated tool install.",
)
@click.option("--json", "as_json", is_flag=True, help="Emit a stable JSON health report.")
def doctor(do_gc: bool, fix: bool, check_db: bool, strict_runtime: bool, as_json: bool) -> None:
    """Self-check: vault present, sqlite-vec loadable, MLX importable, models in cache.

    `--gc` reports orphans (store rows whose `.md` is gone, `.md` files
    whose `id` isn't in the store). `--gc --fix` removes orphan store
    rows; orphan `.md` files are listed but never deleted automatically.
    """
    cfg = Config.from_env()
    if as_json:
        report = _doctor_report(
            cfg,
            check_db=check_db,
            strict_runtime=strict_runtime,
            do_gc=do_gc,
            fix=fix,
        )
        click.echo(json.dumps(report, ensure_ascii=False, indent=2, default=str))
        sys.exit(0 if report["ok"] else 1)

    ok = True

    runtime_report = _runtime_install_report()
    _print_runtime_install_report(runtime_report)
    if strict_runtime and runtime_report["warnings"]:
        ok = False

    if cfg.data_dir.is_dir():
        console.print(f"[green]✓[/green] data_dir: {cfg.data_dir}")
    else:
        console.print(f"[red]✗[/red] data_dir missing: {cfg.data_dir}")
        ok = False
    if cfg.vault_path is not None:
        if cfg.vault_path.is_dir():
            console.print(f"[green]✓[/green] vault_path: {cfg.vault_path}")
        else:
            console.print(f"[yellow]![/yellow] vault_path set but missing: {cfg.vault_path}")

    try:
        import sqlite3

        import sqlite_vec  # type: ignore[import-untyped]

        conn = sqlite3.connect(":memory:")
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.close()
        console.print("[green]✓[/green] sqlite-vec loadable")
    except Exception as exc:
        console.print(f"[red]✗[/red] sqlite-vec: {exc}")
        ok = False

    try:
        import sqlite3 as _sqlite3_fts

        _c = _sqlite3_fts.connect(":memory:")
        try:
            _c.execute("CREATE VIRTUAL TABLE _fts_probe USING fts5(x)")
            _c.execute("DROP TABLE _fts_probe")
        finally:
            _c.close()
        console.print("[green]✓[/green] sqlite FTS5 available")
    except Exception as exc:
        console.print(
            f"[red]✗[/red] sqlite FTS5 unavailable: {exc}  "
            "[dim](BM25/hybrid search will degrade to vec-only)[/dim]"
        )
        ok = False

    try:
        import mlx.core  # noqa: F401
        import mlx_lm  # noqa: F401

        console.print("[green]✓[/green] mlx + mlx_lm importable")
    except Exception as exc:
        console.print(f"[red]✗[/red] mlx: {exc}")
        ok = False

    hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
    for model in (cfg.embedder_model, cfg.llm_model, cfg.helper_model):
        cache_dir = hf_cache / f"models--{model.replace('/', '--')}"
        if cache_dir.is_dir():
            console.print(f"[green]✓[/green] cached: {model}")
        else:
            console.print(
                f"[yellow]![/yellow] not cached: {model}  [dim](run `hf download {model}`)[/dim]",
            )

    daemon = _recall_daemon_health(cfg)
    if daemon.get("running"):
        console.print(f"[green]✓[/green] recall-daemon: running (pid={daemon.get('pid')}, ping ok)")
    elif daemon.get("pid_alive") and not daemon.get("ping_ok"):
        err = daemon.get("error") or "no ping response"
        console.print(
            f"[yellow]![/yellow] recall-daemon: pid {daemon.get('pid')} alive "
            f"but socket unresponsive ({err})"
        )
    elif daemon.get("socket_exists") and not daemon.get("pid_alive"):
        console.print(
            "[yellow]![/yellow] recall-daemon: stale socket without process — "
            "run `memo recall-daemon stop` to clean up"
        )
    else:
        console.print(
            "[dim]•[/dim] recall-daemon: not running "
            "[dim](`memo recall-daemon start` to enable warm recall)[/dim]"
        )

    # GitHub sync health — surfaces the silent no-op (data_dir not a git clone)
    # and stranded commits (committed locally but never pushed).
    from memo.sync_git import sync_status as _sync_status

    sync = _sync_status(cfg)
    if not sync.get("is_git_clone"):
        console.print(
            "[dim]•[/dim] github sync: OFF "
            "[dim](data_dir not a git clone — `memo sync bootstrap <url>` to enable)[/dim]"
        )
    elif sync.get("pending"):
        console.print(
            f"[red]✗[/red] github sync: STRANDED — local commit(s) not pushed "
            f"(ahead {sync['ahead']}); offline/auth? next `memo sync push` retries"
        )
    elif sync.get("ahead") or sync.get("dirty_files"):
        console.print(
            f"[yellow]![/yellow] github sync: {sync['ahead']} unpushed, "
            f"{sync['dirty_files']} uncommitted (auto-syncs in-session / on Stop)"
        )
    else:
        console.print(
            f"[green]✓[/green] github sync: up to date ({sync.get('remote') or 'no remote'})"
        )

    if check_db:
        for db in _db_health_report(cfg):
            marker = "[green]✓[/green]" if db["ok"] else "[red]✗[/red]"
            if db["exists"]:
                console.print(
                    f"{marker} db:{db['label']} {db['status']} "
                    f"integrity={db.get('integrity_check', '-')} "
                    f"tables={db.get('table_count', 0)} "
                    f"size={db.get('size_bytes', 0)}"
                )
                if db.get("vec_dims") is not None:
                    console.print(
                        f"  dims vec={db.get('vec_dims')} "
                        f"repo_vec={db.get('repo_vec_dims')} "
                        f"expected={db.get('expected_dims')}"
                    )
                if db.get("latest_memory_update"):
                    console.print(f"  latest_memory_update={db['latest_memory_update']}")
                if db.get("latest_repo_index"):
                    console.print(f"  latest_repo_index={db['latest_repo_index']}")
            else:
                console.print(f"[yellow]![/yellow] db:{db['label']} missing: {db['path']}")
            if not db["ok"]:
                ok = False

    if do_gc:
        from memo.memory import Memory

        mem = Memory(cfg)
        report = mem.gc(fix=fix)
        n_store = len(report["orphan_store"])
        n_disk = len(report["orphan_disk"])
        n_stale_synth = len(report.get("stale_synthesis", []))
        if n_store == 0 and n_disk == 0 and n_stale_synth == 0:
            console.print("[green]✓[/green] no orphans")
        else:
            if n_store:
                verb = "dropped" if fix else "found"
                console.print(
                    f"[yellow]{verb} {n_store} orphan store row(s)[/yellow] "
                    f"(in store, .md missing)",
                )
                for oid in report["orphan_store"][:20]:
                    console.print(f"  · {oid}")
                if n_store > 20:
                    console.print(f"  · …and {n_store - 20} more")
            if n_disk:
                console.print(
                    f"[yellow]found {n_disk} orphan .md file(s)[/yellow] "
                    f"(on disk, not in store — try `memo reindex`)",
                )
                for p in report["orphan_disk"][:20]:
                    console.print(f"  · {p}")
                if n_disk > 20:
                    console.print(f"  · …and {n_disk - 20} more")
            if n_stale_synth:
                verb = "archived" if fix else "found"
                console.print(
                    f"[yellow]{verb} {n_stale_synth} stale synthesis memori{'a' if n_stale_synth == 1 else 'as'}[/yellow] "
                    f"(synthesis sources deleted — use `memo gc --fix` to archive)",
                )
                for sid in report.get("stale_synthesis", [])[:20]:
                    console.print(f"  · {sid[:8]}")

    # Memory-first adoption: the "always consult memo first" guarantee is only as
    # good as adoption, so surface any expected consumer that reads memo zero
    # times (wired but silent). A warning, not a hard failure.
    try:
        from memo.dashboard_metrics import consult_breakdown

        gap = consult_breakdown(cfg.state_dir).get("silent") or []
        if gap:
            console.print(
                f"[yellow]![/yellow] memo-first gap: {', '.join(gap)} not consulting "
                "memo — wire with `memo install-mcp` / `memo mandate`"
            )
        else:
            console.print("[green]✓[/green] memo-first: all expected consumers active")
    except Exception as exc:
        console.print(f"[dim]•[/dim] memo-first adoption check skipped: {exc}")

    sys.exit(0 if ok else 1)
