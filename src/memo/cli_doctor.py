from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path
from typing import Any

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
from memo.runtime.mcp_config import repair_mcp_configs, scan_mcp_configs, scan_mcp_store_env

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


def _newest_source_mtime(project_root: Path) -> float | None:
    """Newest mtime among git-tracked files, or None when git cannot answer.

    Tracked files only: an index is not stale because a build artifact or a
    scratch file changed, and walking the whole tree would read node_modules.
    """
    import subprocess

    try:
        proc = subprocess.run(
            ["git", "-C", str(project_root), "ls-files", "-z"],
            capture_output=True,
            timeout=10.0,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    newest: float | None = None
    for raw in proc.stdout.split(b"\0"):
        if not raw:
            continue
        try:
            mtime = (project_root / raw.decode()).stat().st_mtime
        except OSError:
            continue
        if newest is None or mtime > newest:
            newest = mtime
    return newest


def codegraph_staleness(db_path: Path, *, newest_source_mtime: float | None) -> str | None:
    """Return a staleness warning for the codegraph index, or None if current.

    Staleness is "behind the code", not "old". Comparing the index's age
    against a flat 24-hour window warned on every repo nobody had touched for a
    day, and `codegraph sync` answered "Already up to date" — a warning the
    documented remedy could not clear, which trains the operator to ignore the
    line and takes the real signal with it.

    ``newest_source_mtime`` is None when git cannot enumerate the tree; there
    is nothing to compare against, so age remains the best guess available.
    """
    import time

    index_mtime = db_path.stat().st_mtime
    if newest_source_mtime is not None:
        if index_mtime >= newest_source_mtime:
            return None
        return "index is behind the working tree — run `codegraph sync`"
    if time.time() - index_mtime > _CODEGRAPH_FRESH_MAX_AGE_S:
        return "index older than 24h and freshness unverifiable — run `codegraph sync`"
    return None


def _check_codegraph() -> None:
    """Codegraph index health — WARN-only, never flips doctor's `ok`.

    The graph consumers (navigation pathfinding, impact) degrade silently when
    the index is missing, but MEMO_GRAPH_USE_CODEGRAPH is default-on, so a
    missing index NEXT TO an installed CLI is signal. When neither the index
    nor the CLI exists (the common pipx/uv-tool install of memo), codegraph
    simply isn't part of this setup — total absence is not signal, stay
    informative.
    """
    from memo import codegraph_loader

    cg_version = _codegraph_cli_version()
    cg_db = codegraph_loader._resolve_db()
    cg_db_present = cg_db.is_file()
    if not cg_db_present:
        if cg_version is None:
            console.print(
                f"[dim]•[/dim] codegraph: not installed "
                f"[dim](optional — `{_CODEGRAPH_INSTALL_HINT}`)[/dim]"
            )
        else:
            console.print(
                f"[yellow]![/yellow] codegraph: index missing at {cg_db} "
                f"[dim](`{_CODEGRAPH_INSTALL_HINT}`)[/dim]"
            )
    else:
        try:
            import sqlite3 as cg_sqlite3

            cg_conn = cg_sqlite3.connect(f"file:{cg_db}?mode=ro", uri=True)
            try:
                cg_journal = str(cg_conn.execute("PRAGMA journal_mode").fetchone()[0]).lower()
                cg_nodes = int(cg_conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0])
                cg_edges = int(cg_conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0])
            finally:
                cg_conn.close()
            cg_warnings: list[str] = []
            if cg_journal != "wal":
                cg_warnings.append(f"journal_mode={cg_journal} (expected wal)")
            if cg_nodes == 0 or cg_edges == 0:
                cg_warnings.append(f"index empty (nodes={cg_nodes} edges={cg_edges})")
            cg_stale = codegraph_staleness(
                cg_db, newest_source_mtime=_newest_source_mtime(cg_db.parent.parent)
            )
            if cg_stale:
                cg_warnings.append(cg_stale)
            if cg_warnings:
                console.print(f"[yellow]![/yellow] codegraph: {'; '.join(cg_warnings)}")
            else:
                console.print(
                    f"[green]✓[/green] codegraph: index ok "
                    f"(nodes={cg_nodes} edges={cg_edges}, wal, current with the tree)"
                )
        except Exception as exc:
            console.print(f"[yellow]![/yellow] codegraph: index unreadable: {exc}")

    if cg_version is None:
        if cg_db_present:
            console.print(
                f"[dim]•[/dim] codegraph: CLI not found [dim](`{_CODEGRAPH_INSTALL_HINT}`)[/dim]"
            )
    elif cg_version < _CODEGRAPH_MIN_CLI_VERSION:
        cg_min = ".".join(str(p) for p in _CODEGRAPH_MIN_CLI_VERSION)
        cg_ver = ".".join(str(p) for p in cg_version)
        console.print(
            f"[yellow]![/yellow] codegraph: CLI v{cg_ver} < v{cg_min} — impact/edges "
            f"degraded [dim](npm i -g @colbymchenry/codegraph@latest)[/dim]"
        )
    else:
        console.print(f"[green]✓[/green] codegraph: CLI v{'.'.join(str(p) for p in cg_version)}")


def _check_proxy() -> None:
    """Proxy health — WARN-only, never flips doctor's `ok`.

    The failure that matters here is silent: ANTHROPIC_BASE_URL pointed at
    this machine's loopback while no proxy listens there. Claude Code then
    fails exactly like a dead network, because from the CLI's side there is
    no proxy between here and there — just connection refused. Agent load
    state and the listening probe are context for that diagnosis.
    """
    import os as _os
    import socket as _sock
    import subprocess as _sp
    from urllib.parse import urlparse

    from memo.flags import flag_int
    from memo.ops_launchd import PROXY_LABEL, parse_launchctl_list

    port = flag_int("MEMO_PROXY_PORT") or 8768

    def _listening(p: int) -> bool:
        try:
            with _sock.socket(_sock.AF_INET, _sock.SOCK_STREAM) as sock:
                sock.settimeout(0.25)
                return sock.connect_ex(("127.0.0.1", p)) == 0
        except OSError:
            return False

    listening = _listening(port)
    loaded = False
    try:
        out = _sp.run(
            ["launchctl", "list"], capture_output=True, text=True, timeout=5, check=False
        ).stdout
        loaded = any(row["label"] == PROXY_LABEL for row in parse_launchctl_list(out))
    except Exception:  # noqa: S110  # launchctl probing must never fail doctor
        pass

    pointed_url = ""
    pointed_port: int | None = None
    base_url = _os.environ.get("ANTHROPIC_BASE_URL", "").strip().rstrip("/")
    if base_url:
        try:
            parsed = urlparse(base_url)
            host = (parsed.hostname or "").lower()
            bport = parsed.port or (443 if parsed.scheme == "https" else 80)
            if host in ("127.0.0.1", "localhost", "::1"):
                pointed_url = base_url
                pointed_port = bport
        except ValueError:
            pass

    if listening:
        state = "loaded (launchd)" if loaded else "not loaded in launchd"
        console.print(
            f"[green]✓[/green] proxy: {PROXY_LABEL} {state} — listening on 127.0.0.1:{port}"
        )
        return
    if not (pointed_url and pointed_port is not None):
        console.print(
            "[dim]•[/dim] proxy: not running "
            "[dim](`memo proxy serve`, or `memo ops install proxy` for a launchd agent)[/dim]"
        )
        return
    console.print(
        f"[yellow]![/yellow] proxy: ANTHROPIC_BASE_URL={pointed_url} is set, but "
        f"nothing is listening on 127.0.0.1:{pointed_port} — Claude Code fails "
        "as if the network were down. Run `memo proxy serve` (foreground) or "
        "`memo ops install proxy` (launchd agent)."
    )


def _report_mcp_store_env() -> bool:
    """Print MCP client store-path findings; return False when any is broken.

    A store path in a client's `env` is the failure the CLI cannot feel: memo
    creates whatever directory it is handed, so the MCP client answers from an
    empty corpus while every other check here stays green.
    """
    findings = scan_mcp_store_env()
    if not findings:
        console.print("[green]✓[/green] mcp store env: no overrides pointing elsewhere")
        return True
    for finding in findings:
        why = (
            "resolves against the client's cwd"
            if finding["issue"] == "relative"
            else "does not exist — the server would create a new empty store"
        )
        console.print(
            f"[red]✗[/red] mcp store env: {finding['config']} → "
            f"{finding['var']}={finding['value']} "
            f"[dim]({why}; set an absolute path or drop the override)[/dim]"
        )
    return False


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

    ok = _report_mcp_store_env() and ok

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

    _check_codegraph()

    _check_proxy()

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
                "[dim](set MEMO_MCP_PROFILE=agent for 41 tools / ~9.4k tokens, or "
                "`memo install-mcp --profile core` for 58 tools / ~12.9k tokens — "
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
        with contextlib.suppress(Exception):
            _report_derived_storage(cfg)
        with contextlib.suppress(Exception):
            _report_dark_flags(cfg)
    except Exception:  # noqa: S110
        pass

    sys.exit(0 if ok else 1)


def _report_dark_flags(cfg: Any) -> None:
    """Count the shipped-but-never-enabled feature flags.

    Every one is code that ships, is tested, and never runs — and some cost
    real resources while off (HyPE held 110 MB of vectors for a pass that had
    not run in months). `dream_flags` already models a cull deadline, but the
    sweep that applies it lives inside a pass that is itself default-off, so
    nothing surfaced the backlog. Deleting a flag stays a human decision; being
    unable to see how many there are should not be one.
    """
    from memo.dream_flags import status_rows

    rows = status_rows(cfg)
    dark = [r for r in rows if r["status"] not in ("graduated", "human_graduated")]
    if not dark:
        console.print("[green]✓[/green] dark flags: none")
        return
    overdue = [r for r in dark if r["days_left"] == 0]
    suffix = f", {len(overdue)} past the cull deadline" if overdue else ""
    console.print(
        f"[dim]•[/dim] dark flags: {len(dark)} shipped but never enabled{suffix} "
        "[dim](`memo dream graduate-flags --status`)[/dim]"
    )


# Bytes of state per stored memory above which derived storage is out of
# proportion to the corpus. A 6.4k-memory install measured 2.3 GB — ~370 KB per
# memory against ~3.5 KB of actual vector — because the compaction passes that
# reclaim it had never run. Roughly 10x a healthy install, so it flags a real
# leak without firing on a small corpus whose fixed overhead dominates.
_STATE_BYTES_PER_MEMORY_WARN = 120_000
_STATE_BYTES_FLOOR = 200 * 1024 * 1024


def _report_derived_storage(cfg: Any) -> None:
    """Warn when derived storage has outgrown the corpus it derives from.

    Markdown is the source of truth and everything under state_dir is
    rebuildable, so bloat here is silent by construction: nothing fails, search
    keeps working, and the only symptom is disk. This surfaces it.
    """
    state_dir = Path(cfg.state_dir)
    total = 0
    for p in state_dir.rglob("*"):
        # A nightly job or the recall daemon may unlink a file mid-walk; one
        # vanished temp file must not cost the whole report.
        with contextlib.suppress(OSError):
            if p.is_file():
                total += p.stat().st_size
    if total < _STATE_BYTES_FLOOR:
        return
    from memo.cli_common import get_memory

    n_memories = int(get_memory(cfg).store.count() or 0)
    if n_memories <= 0:
        return
    per_memory = total / n_memories
    if per_memory < _STATE_BYTES_PER_MEMORY_WARN:
        console.print(
            f"[green]✓[/green] derived storage: {_human_bytes(total)} for {n_memories} memories"
        )
        return
    console.print(
        f"[yellow]![/yellow] derived storage: {_human_bytes(total)} for {n_memories} memories "
        f"({_human_bytes(per_memory)}/memory) "
        "[dim]— MEMO_DREAM_VECTOR_HYGIENE_ENABLED=1 packs the rebuildable caches and "
        "vacuums whichever DBs have outgrown their data[/dim]"
    )


def _human_bytes(n: float) -> str:
    """Human-readable size. Divides in float: truncating at each step compounds,
    so an integer division chain reports 1.9GB as 1.0GB."""
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f}{unit}" if unit == "B" else f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}GB"
