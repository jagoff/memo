from __future__ import annotations

import json
import sys

import click

from memo.cli_common import console
from memo.cli_diag import (
    _db_health_report,
    _doctor_report,
    _gc_report,
    _recall_daemon_health,
    freshness_report,
)
from memo.cli_runtime import _print_runtime_install_report, _runtime_install_report
from memo.config import Config
from memo.runtime.mcp_config import repair_mcp_configs, scan_mcp_configs

# Codegraph doctor check — WARN-only (never fails doctor). Consumers of the
# code graph degrade silently when the index is absent, but
# MEMO_GRAPH_USE_CODEGRAPH is default-on, so absence IS signal worth surfacing.
_CODEGRAPH_MIN_CLI_VERSION = (1, 5, 0)
_CODEGRAPH_FRESH_MAX_AGE_S = 24 * 3600.0
_CODEGRAPH_INSTALL_HINT = "npm i -g @colbymchenry/codegraph && codegraph init"


def _codegraph_cli_version(timeout_s: float = 2.0) -> tuple[int, int, int] | None:
    """Best-effort `codegraph --version` probe.

    Returns the parsed (major, minor, patch) or None when the binary is
    missing, the probe times out, or the output carries no semver. Never
    raises — the doctor codegraph check is WARN-only.
    """
    import re
    import subprocess

    try:
        proc = subprocess.run(
            ["codegraph", "--version"],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except Exception:
        return None
    match = re.search(r"(\d+)\.(\d+)\.(\d+)", f"{proc.stdout}\n{proc.stderr}")
    if not match:
        return None
    return (int(match.group(1)), int(match.group(2)), int(match.group(3)))


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
@click.option(
    "--check",
    "check_only",
    is_flag=True,
    help="Report only — skip the default auto-repair of MCP config paths.",
)
@click.option(
    "--agent",
    "agents",
    multiple=True,
    type=click.Choice(["codex", "claude-code"]),
    help="Verify one first-class agent setup (repeatable).",
)
@click.option("--json", "as_json", is_flag=True, help="Emit a stable JSON health report.")
def doctor(
    do_gc: bool,
    fix: bool,
    check_db: bool,
    strict_runtime: bool,
    check_only: bool,
    agents: tuple[str, ...],
    as_json: bool,
) -> None:
    """Self-check: vault present, sqlite-vec loadable, MLX importable, models in cache.

    Safe, mechanical drift is repaired by default: a stale MCP config path
    (a dead pipx/uv-internal binary) is repointed to the stable `~/.local/bin`
    shim, with a `.bak` backup. Pass `--check` to report without writing; the
    `--json` health report never mutates.

    `--gc` reports orphans (store rows whose `.md` is gone, `.md` files
    whose `id` isn't in the store). `--gc --fix` removes orphan store
    rows; orphan `.md` files are listed but never deleted automatically.
    """
    if fix and not do_gc:
        raise click.UsageError("--fix requires --gc")
    cfg = Config.from_env()
    if as_json:
        report = _doctor_report(
            cfg,
            check_db=check_db,
            strict_runtime=strict_runtime,
            do_gc=do_gc,
            fix=fix,
        )
        if agents:
            from memo.runtime.agent_registry import verify_agent

            agent_reports = [verify_agent(agent) for agent in agents]
            report["agent_setup"] = agent_reports
            report["ok"] = bool(report["ok"] and all(item["ok"] for item in agent_reports))
        click.echo(json.dumps(report, ensure_ascii=False, indent=2, default=str))
        sys.exit(0 if report["ok"] else 1)

    ok = True

    if agents:
        from memo.runtime.agent_registry import verify_agent

        for agent in agents:
            agent_report = verify_agent(agent)
            marker = "[green]✓[/green]" if agent_report["ok"] else "[red]✗[/red]"
            checks = agent_report["checks"]
            console.print(
                f"{marker} agent:{agent} detected={checks['detected']} "
                f"mcp={checks['mcp_configured']} mcp-runtime={checks['mcp_runtime_current']} "
                f"runtime={checks['runtime_isolated']} pair={checks['runtime_pair']} "
                f"version={checks['runtime_version']} "
                f"version-match={checks['runtime_version_match']} smoke={checks['runtime_smoke']} "
                f"storage={checks['storage_writable']} profile={checks['profile']} "
                f"profile-current={checks['profile_current']} protocol={checks['protocol_mode']} "
                f"protocol-current={checks['protocol_current']} "
                f"instructions={checks['instruction_marker']} writable={checks['instruction_writable']}"
            )
            if not agent_report["ok"]:
                console.print(f"  [dim]repair with `memo setup {agent}`[/dim]")
                ok = False

    runtime_report = _runtime_install_report()
    _print_runtime_install_report(runtime_report)
    if strict_runtime and runtime_report["warnings"]:
        ok = False

    _fresh = freshness_report()
    if _fresh["status"] == "stale":
        console.print(f"[yellow]![/yellow] install freshness: {_fresh['message']}")
    elif _fresh["status"] in ("fresh", "repo-ahead"):
        console.print(f"[green]✓[/green] install freshness: {_fresh['message']}")

    if not scan_mcp_configs():
        console.print("[green]✓[/green] mcp config paths: stable")
    else:
        for _r in repair_mcp_configs(apply=not check_only):
            if _r["status"] == "repaired":
                console.print(
                    f"[green]✓[/green] mcp config repaired: {_r['config']} → "
                    f"{_r['suggestion']} [dim](was {_r['issue']}; backup .bak)[/dim]"
                )
            elif _r["status"] == "would-repair":
                console.print(
                    f"[yellow]![/yellow] mcp config: {_r['config']} → {_r['command']} "
                    f"({_r['issue']}); would repoint to {_r['suggestion']} "
                    "[dim](drop --check to apply)[/dim]"
                )
            else:  # skipped-no-target: shim binary not installed
                console.print(
                    f"[yellow]![/yellow] mcp config: {_r['config']} → {_r['command']} "
                    f"({_r['issue']}); shim {_r['suggestion']} missing — install runtime first"
                )

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

        from memo.sqlite_compat import import_sqlite_vec

        conn = sqlite3.connect(":memory:")
        try:
            conn.enable_load_extension(True)
            import_sqlite_vec().load(conn)
        finally:
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

    from memo.embedder_select import resolve_backend

    backend = resolve_backend(cfg)
    if backend == "mlx":
        try:
            from memo.mlx_gpu import suppress_swig_deprecation_warnings

            suppress_swig_deprecation_warnings()
            import mlx.core  # noqa: F401
            import mlx_lm  # noqa: F401

            console.print("[green]✓[/green] mlx + mlx_lm importable")
        except Exception as exc:
            console.print(f"[red]✗[/red] mlx: {exc}")
            ok = False
        models: tuple[str, ...] = (cfg.embedder_model, cfg.llm_model, cfg.helper_model)
    else:
        # CPU backend (Linux/Ubuntu): MLX is expected to be absent.
        # The load-bearing dependency is sentence-transformers instead.
        try:
            import sentence_transformers  # noqa: F401

            console.print("[green]✓[/green] sentence-transformers (CPU backend) importable")
        except Exception as exc:
            console.print(
                f"[red]✗[/red] sentence-transformers: {exc}  [dim](pip install mlx-memo)[/dim]"
            )
            ok = False
        models = (cfg.st_embedder_model,)

    from memo.model_pins import hf_hub_cache_dir

    hf_cache = hf_hub_cache_dir()
    for model in models:
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

    # Recall hook wiring — memo-owned settings.json entry, self-healed on
    # memo-mcp start. If it's missing, recall silently stops firing (the whole
    # recall pipeline goes dark with no error). Surfaces that here.
    from memo.cli_hooks import recall_hook_wired

    if recall_hook_wired():
        _last = None
        try:
            from memo.dashboard import read_recall_hook_log

            _rows = read_recall_hook_log(cfg.state_dir, limit=1)
            if _rows:
                _last = _rows[-1].get("ts")
        except Exception:
            _last = None
        _fired = f", last fired {_last}" if _last else ""
        console.print(f"[green]✓[/green] recall hook: wired in settings.json{_fired}")
    else:
        console.print(
            "[yellow]![/yellow] recall hook: NOT wired in settings.json — "
            "run `memo install-recall-hook` (auto-heals on next memo-mcp start "
            "unless MEMO_HOOK_SELFHEAL=0)"
        )

    # GitHub sync health — surfaces the silent no-op (data_dir not a git clone)
    # and stranded commits (committed locally but never pushed).
    from memo.sync_git import sync_status as _sync_status

    sync = _sync_status(cfg, check_remote=True)
    if not sync.get("is_git_clone"):
        console.print(
            "[dim]•[/dim] github sync: OFF "
            "[dim](data_dir not a git clone — `memo sync setup` to share across machines)[/dim]"
        )
    elif sync.get("pending"):
        _reason = sync.get("pending_reason")
        _hint = f" — {_reason}" if _reason else "; offline/auth? next `memo sync push` retries"
        console.print(
            f"[red]✗[/red] github sync: STRANDED — local commit(s) not pushed "
            f"(ahead {sync['ahead']}){_hint}"
        )
    elif sync.get("ahead") or sync.get("dirty_files"):
        console.print(
            f"[yellow]![/yellow] github sync: {sync['ahead']} unpushed, "
            f"{sync['dirty_files']} uncommitted (auto-syncs in-session / on Stop)"
        )
    elif sync.get("behind"):
        # A machine stuck behind (e.g. a persistent `.md` rebase conflict that
        # `sync_pull` aborts) must not read as healthy — without this branch
        # doctor printed "up to date" while N remote commits stayed unpulled.
        console.print(
            f"[yellow]![/yellow] github sync: {sync['behind']} commit(s) behind remote "
            "— run `memo sync pull` (a persistent rebase conflict needs manual resolution)"
        )
    else:
        console.print(
            f"[green]✓[/green] github sync: up to date ({sync.get('remote') or 'no remote'})"
        )

    # Codegraph index health — WARN-only, never flips `ok`. The graph consumers
    # (navigation pathfinding, impact) degrade silently when the index is
    # missing, but MEMO_GRAPH_USE_CODEGRAPH is default-on, so a missing index
    # NEXT TO an installed CLI is signal. When neither the index nor the CLI
    # exists (the common pipx/uv-tool install of memo), codegraph simply isn't
    # part of this setup — total absence is not signal, stay informative.
    from memo import codegraph_loader

    _cg_version = _codegraph_cli_version()
    _cg_db = codegraph_loader._resolve_db()
    _cg_db_present = _cg_db.is_file()
    if not _cg_db_present:
        if _cg_version is None:
            console.print(
                f"[dim]•[/dim] codegraph: not installed "
                f"[dim](optional — `{_CODEGRAPH_INSTALL_HINT}`)[/dim]"
            )
        else:
            console.print(
                f"[yellow]![/yellow] codegraph: index missing at {_cg_db} "
                f"[dim](`{_CODEGRAPH_INSTALL_HINT}`)[/dim]"
            )
    else:
        try:
            import sqlite3 as _cg_sqlite3
            import time as _cg_time

            _cg_conn = _cg_sqlite3.connect(f"file:{_cg_db}?mode=ro", uri=True)
            try:
                _cg_journal = str(_cg_conn.execute("PRAGMA journal_mode").fetchone()[0]).lower()
                _cg_nodes = int(_cg_conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0])
                _cg_edges = int(_cg_conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0])
            finally:
                _cg_conn.close()
            _cg_age_s = _cg_time.time() - _cg_db.stat().st_mtime
            _cg_warnings: list[str] = []
            if _cg_journal != "wal":
                _cg_warnings.append(f"journal_mode={_cg_journal} (expected wal)")
            if _cg_nodes == 0 or _cg_edges == 0:
                _cg_warnings.append(f"index empty (nodes={_cg_nodes} edges={_cg_edges})")
            if _cg_age_s > _CODEGRAPH_FRESH_MAX_AGE_S:
                _cg_warnings.append("index older than 24h — run `codegraph sync`")
            if _cg_warnings:
                console.print(f"[yellow]![/yellow] codegraph: {'; '.join(_cg_warnings)}")
            else:
                console.print(
                    f"[green]✓[/green] codegraph: index ok "
                    f"(nodes={_cg_nodes} edges={_cg_edges}, wal, fresh <24h)"
                )
        except Exception as exc:
            console.print(f"[yellow]![/yellow] codegraph: index unreadable: {exc}")

    if _cg_version is None:
        if _cg_db_present:
            console.print(
                f"[dim]•[/dim] codegraph: CLI not found [dim](`{_CODEGRAPH_INSTALL_HINT}`)[/dim]"
            )
    elif _cg_version < _CODEGRAPH_MIN_CLI_VERSION:
        _cg_min = ".".join(str(p) for p in _CODEGRAPH_MIN_CLI_VERSION)
        _cg_ver = ".".join(str(p) for p in _cg_version)
        console.print(
            f"[yellow]![/yellow] codegraph: CLI v{_cg_ver} < v{_cg_min} — impact/edges "
            f"degraded [dim](npm i -g @colbymchenry/codegraph@latest)[/dim]"
        )
    else:
        console.print(f"[green]✓[/green] codegraph: CLI v{'.'.join(str(p) for p in _cg_version)}")

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
        from memo.trust_preflight import trust_preflight

        trust = trust_preflight(cfg)
        marker = "[green]✓[/green]" if trust["ok"] else "[yellow]![/yellow]"
        console.print(
            f"{marker} trust: identity={trust['identity_constraint']} "
            f"topic_collisions={trust['topic_collision_groups']} "
            f"exact_duplicates={trust['exact_duplicate_groups']} "
            f"ambiguous_projects={trust['multiple_project_tag_rows']} "
            f"legacy_rows={trust['legacy_identity_rows']} "
            f"secret_files={trust['secret_pattern_files']} "
            f"private_files={trust['private_marker_files']}"
        )
        if not trust["ok"]:
            console.print(
                "  [dim]review the vault, then run `memo reindex --rebuild`; "
                "doctor never rewrites or merges trust findings[/dim]"
            )
            ok = False

    if do_gc:
        report = _gc_report(cfg, fix=fix)
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
                    f"[yellow]{verb} {n_stale_synth} stale synthesis memor{'y' if n_stale_synth == 1 else 'ies'}[/yellow] "
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

    # Token efficiency summary — quick snapshot of profile cost and ROI.
    try:
        from memo.surface import mcp_profile, mcp_profile_token_cost

        _profile = mcp_profile()
        _tool_count, _tok_cost, _is_reduced = mcp_profile_token_cost(_profile)
        _profile_label = f"MEMO_MCP_PROFILE={_profile}"
        if _is_reduced:
            console.print(
                f"[green]✓[/green] token cost: {_profile_label}  {_tool_count} tools "
                f"({_tok_cost} tokens/connection)"
            )
        else:
            console.print(
                f"[yellow]![/yellow] token cost: {_profile_label}  {_tool_count} tools "
                f"({_tok_cost} tokens/connection)  "
                "[dim](set MEMO_MCP_PROFILE=agent for 30 tools / ~3k tokens, or "
                "`memo install-mcp --profile core` for 50 tools / ~4.6k tokens — "
                "for constrained clients)[/dim]"
            )
        try:
            from memo.cli_roi import compute_roi

            _roi = compute_roi(cfg.state_dir, limit=200)
            _saved = _roi.get("tokens_saved_human") or "0"
            _grounded = _roi.get("grounded") or 0
            if _grounded > 0:
                console.print(
                    f"[green]✓[/green] tokens saved: ~{_saved} "
                    f"(from {_grounded} grounded recalls — run `memo roi` for details)"
                )
        except Exception:  # noqa: S110
            pass
    except Exception:  # noqa: S110
        pass

    sys.exit(0 if ok else 1)
