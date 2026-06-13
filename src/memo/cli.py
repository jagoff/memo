"""CLI — `memo` entry point.

A handful of operational commands so the user can interact with the
memory store from the shell without spinning up the MCP server:

- `memo save 'content here' --title 'X' --tag x --tag y`
- `memo search 'query' --limit 5`
- `memo list --limit 20 --type decision`
- `memo get <id>`
- `memo delete <id>`
- `memo stats`
- `memo doctor` — verify vault path, embedder loadable, sqlite-vec
  available, MLX present.

Output style:
- Default: rich table for list/search, panel for `get`, plain stats.
- `--json` flag (where applicable): emit raw JSON for piping.

Most command groups have been extracted into `cli_*.py` modules
(`cli_memory`, `cli_repo`, `cli_session`, `cli_graph`, `cli_diag`, …);
this file wires them together and hosts the remaining inline commands.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import click

from memo.cli_analytics import analytics_group
from memo.cli_as_of import as_of_group
from memo.cli_backend_native import backend_native_group
from memo.cli_backup import backup_group
from memo.cli_briefing import briefing
from memo.cli_chat import chat_group
from memo.cli_capture import capture_stop, resume
from memo.cli_collaborative import collaborative_group
from memo.cli_common import _short, console
from memo.cli_common import get_memory as _get_memory
from memo.cli_config import config_group
from memo.cli_consolidate import consolidate_group
from memo.cli_contextual import contextual_group
from memo.cli_contradict import contradict_group
from memo.cli_crossdedup import cross_dedup_cmd
from memo.cli_diag import _db_health_report, _doctor_report, _recall_daemon_health
from memo.cli_dream import dream_cmd
from memo.cli_embed_daemon import embed_daemon_group
from memo.cli_encrypt import encrypt_group
from memo.cli_entities import entities, entity, extract_entities
from memo.cli_eval import eval_group
from memo.cli_export import export_group
from memo.cli_feedback import feedback_group
from memo.cli_graph import graph_group
from memo.cli_health import health as health_cmd
from memo.cli_history import diff_cmd, historia_cmd
from memo.cli_import import import_group
from memo.cli_ingest import ingest
from memo.cli_ingest_daemon import ingest_daemon_group
from memo.cli_links import links_group
from memo.cli_maint_daemon import maint_daemon_group
from memo.cli_maintain import maintain_cmd
from memo.cli_mandate import mandate as mandate_cmd
from memo.cli_memory import (
    delete,
    get,
    history,
    lint,
    list_cmd,
    ocr_image,
    provenance,
    reindex,
    restore,
    save,
    update,
)
from memo.cli_multimodal import multimodal_group
from memo.cli_profile import profile_group
from memo.cli_query import query_group
from memo.cli_recall_daemon import recall_daemon_group
from memo.cli_repo import repo_group
from memo.cli_retier import retier_cmd
from memo.cli_roi import roi as roi_cmd
from memo.cli_runtime import (
    _print_runtime_install_report,
    _runtime_install_report,
    init_cmd,
    install_shell_wrapper,
    install_slash,
    install_watcher,
    mcp_command,
    migrate_vault,
    prewarm,
    self_update,
    sleep_cycle,
    uninstall_watcher_cmd,
    watch,
)
from memo.cli_search import ask, chat_ask, embed_cmd, rerank_cmd, search
from memo.cli_session import session_group
from memo.cli_share import share_group
from memo.cli_sync import sync_group
from memo.cli_synthesize import synthesize_cmd
from memo.cli_temporal import temporal_group
from memo.cli_transcripts import mine_history, reflect
from memo.cli_tui import hook_log, logs, tui
from memo.cli_usefulness import usefulness as usefulness_cmd
from memo.cli_version import version_group
from memo.cli_viz import mapa_cmd
from memo.config import Config
from memo.flags import flag_bool, flag_float, flag_int, flag_str

# Imported at module scope (not lazily) so tests can `patch("memo.cli.run_picker", ...)`.
# `run_picker` itself defers the heavy `questionary` import until called.
from memo.setup import run_picker, write_config_file

_log = logging.getLogger("memo.cli")

# Adaptive-context intent patterns for the recall hook. Compiled once at import
# (not rebuilt per prompt) — they don't depend on runtime state. Each entry is
# (name, pattern, memory-types-to-boost). First match wins.
_RECALL_CONTEXTS: tuple[tuple[str, re.Pattern[str], set[str]], ...] = (
    (
        "code",
        re.compile(r"\b(implement|fix|debug|test|refactor|deploy|build|install)\b", re.I),
        {"decision", "bug", "preference"},
    ),
    (
        "decision",
        re.compile(r"\b(should i|which|choose|decide|recommend|tradeoff|vs\.?|versus)\b", re.I),
        {"decision", "fact"},
    ),
    (
        "write",
        re.compile(r"\b(write|document|explain|describe|summarize|draft)\b", re.I),
        {"note", "fact", "reference"},
    ),
)


@click.group()
@click.version_option(package_name="mlx-memo")
@click.pass_context
def cli(ctx: click.Context) -> None:
    """memo — local MLX memory.

    Stable core: save/search/ask CRUD, briefing/recall-hook, reindex/doctor,
    and history/as-of flows. Many other commands are advanced or experimental.
    """
    _first_run_gate(ctx)


# Command groups extracted from this module live in cli_*.py and register here.
cli.add_command(graph_group)
cli.add_command(eval_group)
cli.add_command(dream_cmd)
cli.add_command(maintain_cmd)
cli.add_command(synthesize_cmd)
cli.add_command(retier_cmd)
cli.add_command(usefulness_cmd)
cli.add_command(roi_cmd)
cli.add_command(mandate_cmd)
cli.add_command(mapa_cmd)
cli.add_command(tui)
cli.add_command(hook_log)
cli.add_command(logs)
cli.add_command(mine_history)
cli.add_command(ingest)
cli.add_command(capture_stop)
cli.add_command(reflect)
cli.add_command(resume)
cli.add_command(diff_cmd)
cli.add_command(historia_cmd)
cli.add_command(briefing)
cli.add_command(init_cmd)
cli.add_command(migrate_vault)
cli.add_command(migrate_vault, name="migrate")  # alias
cli.add_command(mcp_command)
cli.add_command(install_slash)
cli.add_command(self_update)
cli.add_command(watch)
cli.add_command(install_watcher)
cli.add_command(uninstall_watcher_cmd)
cli.add_command(sleep_cycle)
cli.add_command(prewarm)
cli.add_command(install_shell_wrapper)
cli.add_command(config_group)
cli.add_command(save)
cli.add_command(search)
cli.add_command(ask)
cli.add_command(embed_cmd)
cli.add_command(chat_ask)
cli.add_command(chat_group)
cli.add_command(rerank_cmd)
cli.add_command(list_cmd)
cli.add_command(get)
cli.add_command(update)
cli.add_command(reindex)
cli.add_command(delete)
cli.add_command(history)
cli.add_command(ocr_image)
cli.add_command(provenance)
cli.add_command(extract_entities)
cli.add_command(entities)
cli.add_command(entity)
cli.add_command(lint)
cli.add_command(restore)
cli.add_command(profile_group)
cli.add_command(backend_native_group)
cli.add_command(feedback_group)
cli.add_command(repo_group)
cli.add_command(recall_daemon_group)
cli.add_command(ingest_daemon_group)
cli.add_command(maint_daemon_group)
cli.add_command(embed_daemon_group)
cli.add_command(as_of_group)
cli.add_command(session_group)
cli.add_command(temporal_group)
cli.add_command(consolidate_group)
cli.add_command(health_cmd)
cli.add_command(cross_dedup_cmd)
cli.add_command(contextual_group)
cli.add_command(links_group)
cli.add_command(version_group)
cli.add_command(query_group)
cli.add_command(backup_group)
cli.add_command(sync_group)
cli.add_command(encrypt_group)
cli.add_command(share_group)
cli.add_command(analytics_group)
cli.add_command(import_group)
cli.add_command(export_group)
cli.add_command(multimodal_group)
cli.add_command(collaborative_group)
cli.add_command(contradict_group)


# Subcommands that must NEVER trigger the first-run picker — either
# because they're part of setup/diagnostics, they don't need storage,
# or they run from non-interactive hooks (the TTY check + the
# `MEMO_NONINTERACTIVE=1` env var in `hooks.json` handle the latter,
# but listing the names here is a belt-and-suspenders defence in case
# something invokes them from an interactive shell while debugging).
_FIRST_RUN_GATE_SKIP_COMMANDS = {
    "init",
    "doctor",
    "migrate-vault",
    "mcp-command",
    "install-slash",
    "prewarm",
    "recall-hook",
    "recall-daemon",
    "capture-stop",
    "session",
    "ingest",
    "historia",
    "briefing",
    "mapa",
    "backend-native",
    "profile",
}


def _first_run_gate(ctx: click.Context) -> None:
    """If the user hasn't configured `memo` yet, run the picker first.

    Resolution: skip when invoked from hooks (MEMO_NONINTERACTIVE=1 or
    non-TTY), when an env var already configures `data_dir`, when a
    config file already exists, or when the legacy `MEMO_VAULT_PATH`
    pair is set (back-compat path). Also skip for setup/diagnostic
    subcommands so the user can always recover via `memo doctor`.
    """
    import sys as _sys

    if ctx.invoked_subcommand in (None, *_FIRST_RUN_GATE_SKIP_COMMANDS):
        return
    if os.environ.get("MEMO_NONINTERACTIVE") == "1":
        return
    # Both stdin and stdout must be a TTY for the picker to make sense.
    if not (_sys.stdin.isatty() and _sys.stdout.isatty()):
        return
    if "MEMO_DATA_DIR" in os.environ:
        return
    if "MEMO_VAULT_PATH" in os.environ and "MEMO_MEMORY_SUBDIR" in os.environ:
        return
    # Re-resolve the config file at gate-firing time (env may have
    # changed between import and invocation, e.g. in tests).
    from memo.setup.config_io import _resolve_config_path

    if _resolve_config_path().is_file():
        return
    _run_picker_and_save()


def _run_picker_and_save() -> None:
    """Drive the interactive picker → persist to TOML → return.

    Caller is expected to be the first-run gate (or `memo init`). Picker
    aborts (Ctrl-C / ESC) raise `click.exceptions.Exit(130)` so the
    surrounding CLI invocation halts cleanly.
    """
    console.print(
        "[bold]memo first-run setup[/bold] — pick where memorias should live.\n",
    )
    try:
        result = run_picker()
    except KeyboardInterrupt:
        console.print(
            "[yellow]aborted.[/yellow] Re-run any memo command to retry, "
            "or run `memo init` to configure.",
        )
        raise click.exceptions.Exit(130) from None
    cfg_path = write_config_file(
        data_dir=result.data_dir,
        vault_path=result.vault_path,
    )
    result.data_dir.mkdir(parents=True, exist_ok=True)
    console.print(
        f"[green]✓[/green] data_dir = {result.data_dir}",
    )
    if result.vault_path is not None:
        console.print(
            f"[green]✓[/green] vault_path = {result.vault_path}  "
            "[dim](used by `memo ingest`)[/dim]",
        )
    console.print(f"[dim]config saved: {cfg_path}[/dim]")


@cli.command()
def stats() -> None:
    """Summary stats — total records, vault path, embedder model."""
    from memo.memory import Memory

    mem = Memory(Config.from_env())
    history_errors = 0
    with contextlib.suppress(Exception):
        history_errors = int(getattr(mem.history, "error_count", 0))
    info: dict[str, Any] = {
        "total": mem.store.count(),
        "data_dir": str(mem.cfg.data_dir),
        "vault_path": str(mem.cfg.vault_path) if mem.cfg.vault_path else "(unset)",
        "db_path": str(mem.cfg.db_path),
        "model_profile": mem.cfg.model_profile,
        "embedder_model": mem.cfg.embedder_model,
        "llm_model": mem.cfg.llm_model,
        "history_errors": history_errors,
    }
    for k, v in info.items():
        console.print(f"[dim]{k:14s}[/dim] {v}")
    # Recall health — is memo actually consulted + returning confident hits,
    # or a write-only store? Best-effort summary of the recall ring buffer.
    with contextlib.suppress(Exception):
        from memo.dashboard import recall_health

        h = recall_health(mem.cfg.state_dir)
        if h.get("sampled"):
            console.print(
                f"[dim]recall_health [/dim] fired={h['fired']} bailed={h['bailed']} "
                f"hit_rate={h['hit_rate']} top_score={h['median_top_score']} "
                f"p50={h['p50_latency_ms']}ms [dim](last {h['sampled']})[/dim]"
            )


@cli.command()
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

    # 1. Data dir (memorias)
    if cfg.data_dir.is_dir():
        console.print(f"[green]✓[/green] data_dir: {cfg.data_dir}")
    else:
        # Data dir is auto-created by `ensure_dirs()`; missing here means
        # something went wrong with permissions.
        console.print(f"[red]✗[/red] data_dir missing: {cfg.data_dir}")
        ok = False
    # Optional vault_path (only relevant for `memo ingest`).
    if cfg.vault_path is not None:
        if cfg.vault_path.is_dir():
            console.print(f"[green]✓[/green] vault_path: {cfg.vault_path}")
        else:
            console.print(f"[yellow]![/yellow] vault_path set but missing: {cfg.vault_path}")

    # 2. sqlite-vec
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

    # 2b. FTS5 (BM25 backbone — hybrid search silently downgrades without it)
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

    # 3. MLX importable
    try:
        import mlx.core  # noqa: F401
        import mlx_lm  # noqa: F401

        console.print("[green]✓[/green] mlx + mlx_lm importable")
    except Exception as exc:
        console.print(f"[red]✗[/red] mlx: {exc}")
        ok = False

    # 4. Models in HF cache

    hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
    for model in (cfg.embedder_model, cfg.llm_model, cfg.helper_model):
        cache_dir = hf_cache / f"models--{model.replace('/', '--')}"
        if cache_dir.is_dir():
            console.print(f"[green]✓[/green] cached: {model}")
        else:
            console.print(
                f"[yellow]![/yellow] not cached: {model}  [dim](run `hf download {model}`)[/dim]",
            )

    # 5. Recall daemon health (best-effort, never blocks)
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

    sys.exit(0 if ok else 1)


# ── Ambient memory hooks (v0.3.0) ──────────────────────────────────────────
#
# `recall-hook` and `prewarm` are designed to be wired into Claude Code's
# `UserPromptSubmit` and `SessionStart` hooks respectively (see
# `hooks/hooks.json` in this plugin / repo). They turn memo from a manual
# memory store into an **ambient** context layer: the agent automatically
# sees relevant memories before answering, with zero `/memo` invocations
# from the user.
#
# Both commands fail SILENTLY by design — a hook crash must never block
# Claude Code's prompt submission. On any error (DB locked, model load
# failure, malformed stdin) we exit 0 with empty stdout, and Claude Code
# proceeds without injection.


@cli.command(name="recall-hook")
def recall_hook() -> None:
    """UserPromptSubmit hook — inject relevant memorias as additionalContext.

    Reads a JSON object from stdin (Claude Code hook format), embeds the
    `prompt` field via the MLX embedder, runs search, and outputs
    the top-k results as `additionalContext` in `hookSpecificOutput`.

    Configure via env vars (all optional, sensible defaults for v0.3.x):

      MEMO_RECALL_DISABLE          — set to "1" to make this a no-op.
      MEMO_RECALL_TOP_K            — default 3
      MEMO_RECALL_MIN_SIM          — default 0.5. Floor over the
        recency-decayed `score` (decay compresses raw cosine by ~0.15,
        so 0.6 on the decayed score ≈ 0.75 raw — too aggressive, a major
        bail cause). 0.5 recovers borderline hits; reference-tier
        exclusion (on by default) is the bigger noise lever. For pure
        mode=vec without recency, 0.6 is a tighter choice.
      MEMO_RECALL_MIN_PROMPT_CHARS — default 12 (skip very short prompts)
      MEMO_RECALL_BODY_CHARS       — default 400 (snippet length per result)
      MEMO_RECALL_SKIP_SLASH       — default "1" (skip if prompt starts with /)
      MEMO_RECALL_MODE             — default "vec" (pure cosine, fast:
        ~120ms p50). "hybrid" (RRF + cross-encoder rerank) is markedly
        more precise but `memo eval recall` measured its p50 at ~10s on
        this corpus — over the 5s hook budget — so it's reserved for
        `memo ask`/chat, not the hook. "bm25" is keyword-only.
      MEMO_RECALL_RERANK_INPUT_K   — default 10 (only used when MODE=hybrid).
        How many fused candidates to feed the reranker; lower for tighter
        latency, higher for better recall on diffuse queries.

    Output (stdout, JSON):
      `{}` — no injection (no results / disabled / error / silent fail)
      `{"hookSpecificOutput": {"hookEventName": "UserPromptSubmit",
                               "additionalContext": "<markdown>"}}`
    """
    import json as _json
    import sys as _sys

    # Build Config once (was re-parsed 8x below). Guard it so a broken config
    # still honors the "always exit 0" hook contract instead of crashing.
    try:
        cfg = Config.from_env()
    except Exception:
        print("{}")
        _sys.exit(0)

    # Always exit 0 — hooks must not block Claude Code on memo failures.
    def _bail(reason: str = "") -> None:
        if reason and flag_bool("MEMO_RECALL_DEBUG"):
            print(f"# memo recall-hook: {reason}", file=_sys.stderr)
        # Always append to recall.log so `memo logs` surfaces bails even when
        # MEMO_RECALL_DEBUG is unset. Best-effort; swallows internal errors.
        if reason:
            try:
                from memo.dashboard import append_recall_log

                append_recall_log(
                    cfg.state_dir,
                    prompt="",
                    hits=[],
                    via="bail",
                    reason=reason,
                )
            except Exception as exc:
                _log.debug("bail recall-log write failed: %s", exc)
        print("{}")
        _sys.exit(0)

    if flag_bool("MEMO_RECALL_DISABLE"):
        _bail("disabled via MEMO_RECALL_DISABLE")
        return

    # Read stdin (Claude Code passes hook input as JSON).
    try:
        raw = _sys.stdin.read()
        if not raw.strip():
            _bail("empty stdin")
            return
        payload = _json.loads(raw)
    except _json.JSONDecodeError as exc:
        _bail(f"stdin parse fail: {exc}")
        return

    prompt = (payload.get("prompt") or "").strip()
    _sid = (payload.get("session_id") or "").strip() or None
    min_chars = flag_int("MEMO_RECALL_MIN_PROMPT_CHARS") or 12

    # INVARIANT: all prompt gating + rewriting happens here, BEFORE the daemon
    # dispatch (connect_and_recall below) and the subprocess search. `prompt` is
    # reassigned in place, so both paths recall on the rewritten query and the
    # rewrite stays single-source. Do NOT move the daemon dispatch above this.

    # Slash commands: by default the user is invoking another command, so raw
    # recall is noise — but a slash command WITH substantive args carries real
    # intent ("/plan how does memo work" → recall on "how does memo work").
    # Strip the leading /command token and recall on the args; still bail on
    # bare commands and a denylist of pure-UI/noise verbs. Disable the whole
    # gate with MEMO_RECALL_SKIP_SLASH=0. Bail reasons keep the substring
    # "slash command" so dashboard._bail_breakdown classifies them.
    if flag_bool("MEMO_RECALL_SKIP_SLASH") and prompt.startswith("/"):
        head, _, rest = prompt[1:].partition(" ")
        rest = rest.strip()
        slash_min = flag_int("MEMO_RECALL_SLASH_MIN_ARG_CHARS") or 8
        denylist = {
            c.strip().lower()
            for c in (flag_str("MEMO_RECALL_SLASH_DENYLIST") or "").split(",")
            if c.strip()
        }
        if len(rest) < slash_min:
            _bail("slash command (no args)")
            return
        if head.lower() in denylist:
            _bail(f"slash command (noise: {head.lower()})")
            return
        prompt = rest

    # Short prompts skip recall — but a short follow-up inside an active session
    # ("y eso?", "seguimos") still has intent. When a session is active and
    # MEMO_RECALL_EXPAND_CONTEXT is on, prepend the last few session prompts
    # (prompt_trail) to re-anchor before bailing. Bail reasons keep "too short".
    if len(prompt) < min_chars:
        expanded = ""
        n_turns = flag_int("MEMO_RECALL_SHORT_EXPAND_TURNS") or 0
        if _sid and n_turns > 0 and flag_bool("MEMO_RECALL_EXPAND_CONTEXT"):
            try:
                from memo import session as _session_mod

                prior = _session_mod.recent_prompts(cfg.state_dir, _sid, n_turns)
                prior = [p.strip() for p in prior if p.strip() and p.strip() != prompt]
                if prior:
                    expanded = "\n".join([*prior, prompt]).strip()
            except Exception:
                expanded = ""
        if len(expanded) >= min_chars:
            prompt = expanded
        else:
            _bail(f"prompt too short ({len(prompt)} < {min_chars})")
            return

    # Correlation keys (P0): tag this recall with the session + turn so the
    # Stop-hook grounding detector can match the answer back to it. Stamp the
    # SAME turn label into the session snapshot (last_recall_turn) so Stop reads
    # it back race-free. client names the front-end for per-client value.
    _client = flag_str("MEMO_RECALL_CLIENT")
    _turn: int | None = None
    if _sid:
        try:
            from memo import session as _session_mod

            _turn = _session_mod.next_turn(cfg.state_dir, _sid)
            _session_mod.stamp_recall_turn(cfg.state_dir, _sid, _turn)
        except Exception:
            _turn = None

    # Fast path: try the recall daemon socket first (<200 ms when running).
    # If the socket is not there or the connection fails, fall through to
    # the regular in-process search below. The daemon returns the same JSON
    # format as the subprocess path, so we can print-and-exit immediately.
    _t0 = time.time()
    try:
        from memo.recall_server import connect_and_recall

        # Daemon socket timeout: float flag takes precedence; fall back to
        # the int-ms flag; default 2.0 s. 2s gives warm daemon ample time
        # (typical p95 < 500 ms) while leaving headroom for subprocess
        # fallback (~1-2s) within the 5s recall-hook budget.
        _raw_float = flag_float("MEMO_RECALL_DAEMON_TIMEOUT")
        if _raw_float is not None and _raw_float >= 0.1:
            _daemon_timeout = _raw_float
        else:
            _daemon_timeout = max(0.2, (flag_int("MEMO_RECALL_DAEMON_TIMEOUT_MS") or 2000) / 1000.0)
        _daemon_result = connect_and_recall(
            cfg.state_dir,
            prompt=prompt,
            cwd=payload.get("cwd"),
            timeout=_daemon_timeout,
            session_id=_sid,
            turn=_turn,
            client=_client,
        )
        if _daemon_result is not None:
            _latency_ms = int((time.time() - _t0) * 1000)
            if flag_bool("MEMO_RECALL_DEBUG"):
                print(f"# memo recall-hook: daemon hit ({_latency_ms} ms)", file=_sys.stderr)
            # Log to recall.log (daemon path already logs internally, but
            # update the 'via' field for hook-log observability)
            print(_daemon_result)
            _sys.exit(0)
    except Exception as _daemon_exc:
        # Don't bail — fall back to in-process search below. Telemetry only:
        # capture daemon failure so `memo logs` can surface why we paid the
        # cold-start cost.
        try:
            from memo.dashboard import append_recall_log

            append_recall_log(
                cfg.state_dir,
                prompt=prompt,
                hits=[],
                via="daemon_error",
                error=f"{type(_daemon_exc).__name__}: {_daemon_exc}",
            )
        except Exception as exc:
            _log.debug("daemon-error recall-log write failed: %s", exc)

    # All recall params come from the flags registry — no hardcoded fallbacks
    # that can silently diverge from it. Flags where 0 is a meaningful user
    # value (min_sim floor off, project boost off) use an explicit None check
    # so the registered default only applies when the flag is unset.
    top_k = flag_int("MEMO_RECALL_TOP_K") or 3
    _ms = flag_float("MEMO_RECALL_MIN_SIM")
    min_sim = 0.5 if _ms is None else _ms
    body_chars = flag_int("MEMO_RECALL_BODY_CHARS") or 400
    token_budget = flag_int("MEMO_RECALL_TOKEN_BUDGET") or 0
    _pb = flag_float("MEMO_RECALL_PROJECT_BOOST")
    project_boost = 0.15 if _pb is None else _pb

    # T11 — session mode adjusts recall aggressiveness via MEMFLOW_SESSION_MODE.
    # focus: tight (fewer, higher-confidence hits only)
    # explore: broad (more hits, lower bar)
    # maintenance: minimal (1 hit max)
    # review: default unchanged
    _session_mode = os.environ.get("MEMFLOW_SESSION_MODE", "").strip().lower()
    if _session_mode == "focus":
        top_k = min(top_k, 2)
        min_sim = max(min_sim, 0.65)
    elif _session_mode == "explore":
        top_k = max(top_k, 5)
        min_sim = min(min_sim, 0.4)
    elif _session_mode == "maintenance":
        top_k = 1
        min_sim = max(min_sim, 0.70)

    # Read cwd from the hook payload (Claude Code passes it) so we can
    # derive the project tag the user is currently working under.
    payload_cwd = payload.get("cwd")

    # Suppress HF download progress bars on stderr — they'd contaminate
    # the hook's debug output and confuse users tailing logs. The model
    # is already downloaded for any working memo install; this only
    # silences first-run cache-check noise.
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

    # Defer the heavy import — only paid if we get past the early-exits.
    #
    # Mode selection:
    # - `vec` (default): pure cosine similarity, fast (~120ms p50). The
    #   MIN_SIM floor (0.5, on the recency-decayed score) cuts noise.
    # - `hybrid`: RRF fusion + cross-encoder rerank. Far more precise
    #   (eval: prec@3 0.87 vs 0.67, noise 0) BUT `memo eval recall`
    #   measured p50 ~10s here — over the hook budget — so it's reserved
    #   for `memo ask`/chat, not auto-recall.
    # - `bm25`: keyword-only, useful for queries with literal tag
    #   names or filenames where the embedder under-recalls.
    mode = flag_str("MEMO_RECALL_MODE") or "vec"
    if mode == "hybrid":
        # Hook latency budget is tight: cap the rerank pool unless the
        # user explicitly set MEMO_RERANK_INPUT_K elsewhere. Setdefault
        # respects an upstream override (CI bench, custom shell rc).
        os.environ.setdefault(
            "MEMO_RERANK_INPUT_K",
            str(flag_int("MEMO_RECALL_RERANK_INPUT_K") or 10),
        )

    # Warm-signal check: if prewarm hasn't run recently and the requested
    # mode needs the embedder, downgrade to bm25 to avoid cold-load timeout.
    # The signal file is written by `memo prewarm` after a successful warm.
    # A missing or >60 min old file means the Mac just woke / first boot.
    if mode in ("vec", "hybrid") and not flag_bool("MEMO_RECALL_FORCE_MODE"):
        try:
            import time as _time_mod

            _signal = cfg.state_dir / ".prewarm_ts"
            _warm = (
                _signal.exists() and (_time_mod.time() - float(_signal.read_text().strip())) < 3600
            )
            if not _warm:
                if flag_bool("MEMO_RECALL_DEBUG"):
                    print("# memo recall-hook: cold start — downgrading to bm25", file=_sys.stderr)
                mode = "bm25"
        except Exception as exc:
            _log.debug("warm-signal read failed, staying in %s mode: %s", mode, exc)

    # Widen the pool when a project boost is active — we need enough
    # candidates so that off-project hits can be re-ranked below
    # on-project ones without starving the final top_k.
    project_tag = None
    if project_boost > 0:
        try:
            from memo.project import current_project_tag

            project_tag = current_project_tag(payload_cwd)
        except Exception:
            project_tag = None
    search_k = top_k * 3 if project_tag else top_k
    # Tier gate: keep the bulk `reference` tier out of the prompt (mirror of
    # the daemon path in recall_server.py). See `memo.tiers`.
    from memo.tiers import REFERENCE_TYPES

    exclude_types = set(REFERENCE_TYPES) if flag_bool("MEMO_RECALL_EXCLUDE_REFERENCE") else None
    try:
        from memo.memory import Memory

        # Reuse the Config built at the top of the hook — re-running
        # Config.from_env() here would reopen sqlite and re-init the schema
        # (lock acquisition + DDL check) a second time on the hot path.
        mem = Memory(cfg)
    except Exception as exc:
        _bail(f"search failed: {exc}")
        return

    _mbc = flag_int("MEMO_RECALL_MIN_BODY_CHARS")
    min_body_chars = 40 if _mbc is None else _mbc
    staleness_days = flag_int("MEMO_RECALL_STALENESS_DAYS") or 0

    def _search_filter(query_text: str) -> list:
        """Search + project-boost + floor + stub/staleness filters + dedup.

        Factored out of the inline pipeline so the query-expansion fallback
        can reuse the exact same chain on the expanded query (mirror of the
        daemon's `_rank` in recall_server.py)."""
        try:
            hits = mem.search(
                query_text, limit=search_k, mode=mode, recency=True, exclude_types=exclude_types
            )
        except Exception as exc:
            if flag_bool("MEMO_RECALL_DEBUG"):
                print(f"# memo recall-hook: search failed: {exc}", file=_sys.stderr)
            return []
        # Apply project boost — additive on the raw score, then re-sort.
        if project_tag:
            from memo.recall_server import _apply_project_boost

            hits = _apply_project_boost(hits, project_tag, project_boost)
        # Trim back to top_k after boost-aware re-sort.
        hits = hits[:top_k]
        # Filter by similarity floor. The hook searches with recency=True,
        # which compresses raw cosine by ~0.15, so the floor applies to the
        # decayed score: 0.5 ≈ 0.65 raw cosine. The old 0.6 (≈0.75 raw)
        # over-filtered and was a major bail cause; reference-tier exclusion
        # (on by default) is the bigger noise lever. Tune via MEMO_RECALL_MIN_SIM.
        rel = [h for h in hits if h.score is None or h.score >= min_sim]
        # Stub filter: skip tiny-body fragments (auto-saved without content).
        if min_body_chars > 0:
            rel = [h for h in rel if len((h.body or "").strip()) >= min_body_chars]
        # Staleness suppression: memorias older than MEMO_RECALL_STALENESS_DAYS
        # only pass if they score well above min_sim (1.5x), so generic
        # queries aren't dominated by old entries.
        if staleness_days > 0:
            from datetime import UTC as _UTC
            from datetime import datetime as _dt

            _now = _dt.now(_UTC)
            stale_threshold = min_sim * 1.5
            filtered: list = []
            for h in rel:
                try:
                    updated = _dt.fromisoformat(h.updated)
                    if updated.tzinfo is None:
                        updated = updated.replace(tzinfo=_UTC)
                    days = (_now - updated).total_seconds() / 86400
                    if days > staleness_days and (h.score or 0.0) < stale_threshold:
                        continue
                except Exception:
                    pass
                filtered.append(h)
            rel = filtered
        # Collapse near-duplicates so the same fact isn't injected twice.
        from memo.recall_server import dedup_hits

        return dedup_hits(rel)

    relevant = _search_filter(prompt)

    # Query-expansion fallback: bare continuity prompts ("que queda pendiente",
    # "seguimos") embed far from any single memoria and bail. Prepending recent
    # open-loop titles re-anchors them in the user's active work. Fires ONLY on
    # a zero-hit result, so queries that already recall are untouched and the
    # extra search is paid only on a miss (experiment: 4 bare prompts went
    # 0 → 5 hits, top ~0.62). Mirror of the daemon path.
    if not relevant and flag_bool("MEMO_RECALL_EXPAND_CONTEXT"):
        from memo.recall_server import _session_context

        _ctx = _session_context(mem, exclude_types)
        if _ctx:
            relevant = _search_filter(f"{_ctx}\n{prompt}")
            if relevant and flag_bool("MEMO_RECALL_DEBUG"):
                print(
                    f"# memo recall-hook: query expansion recovered {len(relevant)} hits",
                    file=_sys.stderr,
                )

    # Adaptive context: re-weight results by detected prompt intent.
    # Zero extra search cost — pure score boost on returned hits so
    # decision queries surface decision/fact types, code queries surface
    # bug/preference types, etc. Gated by MEMO_RECALL_ADAPTIVE_CONTEXT.
    if relevant and flag_bool("MEMO_RECALL_ADAPTIVE_CONTEXT"):
        _boost_types: set[str] = set()
        for _ctx_name, _ctx_pat, _ctx_types in _RECALL_CONTEXTS:
            if _ctx_pat.search(prompt):
                _boost_types |= _ctx_types
                break  # first match wins
        if _boost_types:
            from dataclasses import replace as _dc_replace

            _boosted = [
                _dc_replace(h, score=round((h.score or 0.0) * 1.25, 6))
                if h.type in _boost_types
                else h
                for h in relevant
            ]
            _boosted.sort(key=lambda h: h.score or 0.0, reverse=True)
            relevant = _boosted

    # Telemetry: append every recall (with or without hits) to the
    # JSONL ring buffer consumed by `memo tui`. Best-effort; failures
    # are swallowed inside the helper.
    _latency_ms_subprocess = int((time.time() - _t0) * 1000)
    try:
        from memo.dashboard import append_recall_log

        append_recall_log(
            cfg.state_dir,
            prompt=prompt,
            hits=[
                {"id": h.id, "score": h.score, "title": h.title, "snippet": (h.body or "")[:240]}
                for h in relevant
            ],
            mode=mode,
            latency_ms=_latency_ms_subprocess,
            via="subprocess",
            session_id=_sid,
            turn=_turn,
            client=_client,
        )
    except Exception as exc:
        _log.debug("subprocess recall-log write failed: %s", exc)

    if not relevant:
        _bail(f"no hits above min_sim={min_sim}")
        return

    # Format as markdown additionalContext. Be terse — context budget is
    # capped at 10k chars by Claude Code; we want each prompt to inject
    # ~500-1500 chars at most so the user's actual prompt isn't drowned.
    #
    # If MEMO_RECALL_TOKEN_BUDGET is set, pack memorias greedily by
    # score until the budget is met. Token estimate is 1 token ≈ 4 chars
    # (English/Spanish prose); good-enough rule-of-thumb that avoids a
    # tiktoken dep. Last memoria gets head-truncated to fit instead of
    # being dropped wholesale.
    from memo.recall_server import RECALL_DIRECTIVE, RECALL_FOOTER, RECALL_HEADER

    footer = RECALL_FOOTER
    lines = [RECALL_HEADER, RECALL_DIRECTIVE, ""]
    used_chars = 0  # chars of formatted block body, excluding header/footer

    def _est_tokens(s: str) -> int:
        return max(1, len(s) // 4)

    budget_chars = token_budget * 4 if token_budget > 0 else None

    for h in relevant:
        score_tag = f" (score {h.score:.2f})" if h.score is not None else ""
        body = (h.body or "").strip().replace("\n", " ")
        if len(body) > body_chars:
            body = body[:body_chars].rstrip() + "…"
        block_lines = [f"**[{h.id[:8]}] {h.title}**{score_tag}"]
        if h.tags:
            block_lines.append(f"_tags_: {', '.join(h.tags)}")
        if body:
            block_lines.append(f"> {body}")
        block_lines.append("")
        block = "\n".join(block_lines)

        if budget_chars is None:
            lines.extend(block_lines)
            continue

        remaining = budget_chars - used_chars
        if remaining <= 0:
            break
        if len(block) <= remaining:
            lines.extend(block_lines)
            used_chars += len(block)
        else:
            # Truncate the body in this final block to fit the budget.
            if body:
                # Reserve space for header line + tags + closing "…"
                head_len = len(block_lines[0]) + 1
                tags_len = (len(block_lines[1]) + 1) if h.tags else 0
                avail = max(0, remaining - head_len - tags_len - 3)
                if avail > 20:
                    trunc_body = body[:avail].rstrip() + "…"
                    block_lines[-2 if h.tags else -1] = f"> {trunc_body}"
                    lines.extend(block_lines)
            break

    lines.append(footer)
    if token_budget > 0 and flag_bool("MEMO_RECALL_DEBUG"):
        approx = _est_tokens("\n".join(lines))
        print(f"# memo recall-hook: ~{approx} tokens (budget {token_budget})", file=_sys.stderr)

    output = {
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": "\n".join(lines),
        }
    }
    print(_json.dumps(output, ensure_ascii=False))
    _sys.exit(0)


# ── Recall daemon — persistent socket server for low-latency recall ──────────


@cli.command(name="dedupe")
@click.option(
    "--threshold",
    type=float,
    default=0.92,
    help="Cosine threshold for near-duplicate clustering (default: 0.92)",
)
@click.option("--max-clusters", type=int, default=50, help="Max clusters to surface (default: 50)")
@click.option("--type", "type_", help="Filter by memoria type")
@click.option(
    "--apply",
    "do_apply",
    is_flag=True,
    help="Interactively merge each cluster (default: list-only)",
)
@click.option("--dry-run", is_flag=True, help="With --apply: show merges without writing")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON")
def dedupe_cmd(
    threshold: float,
    max_clusters: int,
    type_: str | None,
    do_apply: bool,
    dry_run: bool,
    as_json: bool,
) -> None:
    """Find and (optionally) merge near-duplicate memorias.

    Thin wrapper over `memo consolidate` with a higher default threshold,
    aimed at obvious dups (paste-restate, double-save, etc.) — not at
    semantic clustering. Use the lower-threshold `consolidate` group
    when you want LLM synthesis across loosely-related notes.
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    clusters = mem.consolidate(
        threshold=threshold,
        max_clusters=max_clusters,
        type_=type_,
    )
    dup_clusters = [c for c in clusters if c.get("relationship") in ("duplicate", "evolution")]

    if as_json:
        click.echo(json.dumps(dup_clusters, indent=2))
        return

    if not dup_clusters:
        console.print("[green]No near-duplicate clusters found at this threshold.[/green]")
        return

    console.print(f"[bold]Found {len(dup_clusters)} duplicate-like cluster(s).[/bold]")

    if not do_apply:
        for c in dup_clusters[:20]:
            console.print()
            console.print(
                f"[cyan]cluster {c.get('cluster_id', '?')}[/cyan] · "
                f"rel={c.get('relationship')} · n={len(c.get('members', []))}"
            )
            console.print(f"  [dim]{_short(c.get('summary', ''), 200)}[/dim]")
            for m in c.get("members", []):
                console.print(f"    - {m['id'][:8]} · {_short(m.get('title', ''), 70)}")
        if len(dup_clusters) > 20:
            console.print(f"[dim]…and {len(dup_clusters) - 20} more[/dim]")
        console.print()
        console.print("[dim]Re-run with --apply to merge interactively.[/dim]")
        return

    for c in dup_clusters:
        console.print()
        console.print(
            f"[cyan]cluster {c.get('cluster_id', '?')}[/cyan] · "
            f"rel={c.get('relationship')} · n={len(c.get('members', []))}"
        )
        for m in c.get("members", []):
            console.print(f"    - {m['id'][:8]} · {_short(m.get('title', ''), 70)}")

        if not click.confirm("Propose merge for this cluster?", default=True):
            continue

        proposal = mem.consolidator.propose_merge(c)
        if proposal is None:
            console.print("[red]No merge proposal generated.[/red]")
            continue
        console.print(f"[bold]merged title:[/bold] {proposal.merged_title}")
        console.print(f"[dim]strategy={proposal.merge_strategy}[/dim]")
        console.print(f"[dim]rationale={proposal.rationale}[/dim]")

        if not click.confirm("Apply merge?", default=False):
            continue

        result = mem.consolidator.apply_merge(proposal, dry_run=dry_run)
        console.print(
            f"[green]merged →[/green] "
            f"{result.merged_id[:8] if result.merged_id else 'n/a'}  "
            f"archived={len(result.archived_ids)}"
        )


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
