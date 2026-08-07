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

Command modules are loaded LAZILY: importing this module costs ~40 ms
instead of ~220 ms because the heavy `cli_*.py` imports (which pull in
numpy, pydantic, the full `Memory` facade) are deferred until a specific
command is actually resolved. `SurfaceGroup.get_command` imports the
one module that registers the requested command; `list_commands` /
`format_commands` materialize everything so `memo --help` still shows
the full surface.
"""

from __future__ import annotations

import os
from typing import Any

import click

from memo.cli_common import console

# Legacy re-export: some diagnostics/tests import this symbol from memo.cli.
from memo.cli_diag import _recall_daemon_health  # noqa: F401 imported for tests

# Imported at module scope (not lazily) so tests can `patch("memo.cli.run_picker", ...)`.
# `run_picker` itself defers the heavy `questionary` import until called.
from memo.setup import run_picker

# Subcommands that must NEVER trigger the first-run picker — either
# because they're part of setup/diagnostics, they don't need storage,
# or they run from non-interactive hooks (the TTY check + the
# `MEMO_NONINTERACTIVE=1` env var in `hooks.json` handle the latter).
# Listed here as a belt-and-suspenders defence for interactive shells.
_FIRST_RUN_GATE_SKIP_COMMANDS = {
    "init",
    "config",
    "doctor",
    "migrate-vault",
    "migrate",  # alias for migrate-vault
    "mcp-command",
    "install-slash",
    "install-mcp",
    "install-statusline",
    "install-recall-hook",
    "continuity",
    "prewarm",
    "recall-hook",
    "recall-daemon",
    "capture-stop",
    "capture-tick",
    "session",
    "ingest",
    "record-history",
    "briefing",
    "map",
    "backend-native",
    "profile",
    "http-api",
    "startup-banner",
    "codex-badge",
    "install-shims",
    "terminal",
}


# ---------------------------------------------------------------------------
# Lazy command registry
# ---------------------------------------------------------------------------
# Each command name -> (module, symbol, registered_name). Modules are only
# imported when the command is first resolved.
_LAZY: dict[str, tuple[str, str, str]] = {}

def _reg(name: str, module: str, symbol: str, alias: str | None = None) -> None:
    _LAZY[name] = (module, symbol, alias or name)


def _register_lazy_commands() -> None:
    """Populate the lazy map once (idempotent)."""
    if _LAZY:
        return
    # ─ core memories ──────────────────────────────────────────────────────
    for sym in ("delete", "fix", "get", "history", "lint",
                "provenance", "reindex", "rename", "restore", "save",
                "undo", "update"):
        _reg(sym, "memo.cli_memory", sym)
    _reg("ocr-image", "memo.cli_memory", "ocr_image")
    _reg("edit", "memo.cli_memory", "update")
    _reg("list", "memo.cli_memory", "list_cmd")
    # ─ runtime / install ──────────────────────────────────────────────────
    for sym, name in (
        ("init_cmd", "init"),
        ("install_shell_wrapper", "install-shell-wrapper"),
        ("install_slash", "install-slash"),
        ("install_watcher", "install-watcher"),
        ("mcp_command", "mcp-command"),
        ("migrate_vault", "migrate-vault"),
        ("prewarm", "prewarm"),
        ("self_update", "update"),
        ("sleep_cycle", "sleep-cycle"),
        ("uninstall_watcher_cmd", "uninstall-watcher"),
        ("watch", "watch"),
    ):
        _reg(name, "memo.cli_runtime", sym)
    _reg("migrate", "memo.cli_runtime", "migrate_vault", alias="migrate")
    _reg("upgrade", "memo.cli_runtime", "self_update", alias="upgrade")
    _reg("self-update", "memo.cli_runtime", "self_update", alias="self-update")
    # ─ search / ask ───────────────────────────────────────────────────────
    _reg("ask", "memo.cli_search", "ask")
    _reg("chat-ask", "memo.cli_search", "chat_ask", alias="chat-ask")
    _reg("context", "memo.cli_search", "context_cmd", alias="context")
    _reg("context-pack", "memo.cli_search", "context_pack_cmd", alias="context-pack")
    _reg("embed", "memo.cli_search", "embed_cmd", alias="embed")
    _reg("recall", "memo.cli_search", "recall")
    _reg("rerank", "memo.cli_search", "rerank_cmd", alias="rerank")
    _reg("search", "memo.cli_search", "search")
    # ─ groups as their exported symbols ───────────────────────────────────
    for name, module, symbol in [
        ("graduation", "memo.cli_graduation", "graduation_group"),
        ("confidence", "memo.cli_confidence", "confidence_group"),
        ("graph", "memo.cli_graph", "graph_group"),
        ("guard", "memo.cli_guard", "guard_group"),
        ("hype", "memo.cli_hype", "hype_group"),
        ("interject", "memo.cli_interject", "interject_group"),
        ("ask-gaps", "memo.cli_interject", "ask_group"),
        ("related", "memo.cli_related", "related"),
        ("eval", "memo.cli_eval", "eval_group"),
        ("journey-check", "memo.cli_journey", "journey_check"),
        ("debug-recall", "memo.cli_debug_recall", "debug_recall_cmd"),
        ("dream", "memo.cli_dream", "dream_cmd"),
        ("chronicle", "memo.cli_chronicle", "chronicle_cmd"),
        ("code-facts", "memo.cli_code_facts", "code_facts_cmd"),
        ("code-nudge", "memo.cli_code_intel", "code_nudge_cmd"),
        ("code-health", "memo.cli_code_intel", "code_health_cmd"),
        ("maintain", "memo.cli_maintain", "maintain_cmd"),
        ("invalidate", "memo.cli_invalidate", "invalidate_cmd"),
        ("synthesize", "memo.cli_synthesize", "synthesize_cmd"),
        ("retier", "memo.cli_retier", "retier_cmd"),
        ("usefulness", "memo.cli_usefulness", "usefulness"),
        ("verbatim", "memo.cli_verbatim", "verbatim_group"),
        ("roi", "memo.cli_roi", "roi"),
        ("gaps", "memo.cli_outcome", "gaps"),
        ("outcome", "memo.cli_outcome", "outcome"),
        ("digest", "memo.cli_proactive", "digest"),
        ("mandate", "memo.cli_mandate", "mandate"),
        ("drift", "memo.cli_drift", "drift"),
        ("map", "memo.cli_viz", "map_cmd"),
        ("tui", "memo.cli_tui", "tui"),
        ("hook-log", "memo.cli_tui", "hook_log"),
        ("logs", "memo.cli_tui", "logs"),
        ("mine-history", "memo.cli_transcripts", "mine_history"),
        ("mine-git", "memo.cli_transcripts", "mine_git"),
        ("ingest", "memo.cli_ingest", "ingest"),
        ("capture-stop", "memo.cli_capture", "capture_stop"),
        ("capture-tick", "memo.cli_capture", "capture_tick"),
        ("reflect", "memo.cli_transcripts", "reflect"),
        ("resume", "memo.cli_capture", "resume"),
        ("episodes", "memo.cli_capture", "episodes_group"),
        ("events", "memo.cli_events", "events_group"),
        ("diff", "memo.cli_history", "diff_cmd"),
        ("record-history", "memo.cli_history", "history_cmd"),
        ("briefing", "memo.cli_briefing", "briefing"),
        ("stats", "memo.cli_stats", "stats"),
        ("token-gate", "memo.cli_token_gate", "token_gate_cmd"),
        ("token-savings", "memo.cli_token_savings", "token_savings_cmd"),
        ("tokens", "memo.cli_tokens", "tokens_cmd"),
        ("doctor", "memo.cli_doctor", "doctor"),
        ("install-recall-hook", "memo.cli_hooks", "install_recall_hook"),
        ("install-statusline", "memo.cli_statusline", "install_statusline"),
        ("setup", "memo.cli_setup", "setup_cmd"),
        ("install-mcp", "memo.cli_install_mcp", "install_mcp"),
        ("recall-hook", "memo.cli_recall_hook", "recall_hook"),
        ("config", "memo.cli_config", "config_group"),
        ("chat", "memo.cli_chat", "chat_group"),
        ("chat-session", "memo.cli_chat_session", "chat_session_group"),
        ("retrieve", "memo.cli_retrieve", "retrieve_cmd"),
        ("extract-entities", "memo.cli_entities", "extract_entities"),
        ("entities", "memo.cli_entities", "entities"),
        ("entity", "memo.cli_entities", "entity"),
        ("profile", "memo.cli_profile", "profile_group"),
        ("backend-native", "memo.cli_backend_native", "backend_native_group"),
        ("evidence", "memo.cli_operational", "evidence_cmd"),
        ("definitive", "memo.cli_definitive", "definitive_group"),
        ("federation", "memo.cli_federation", "federation_group"),
        ("migrate-independence", "memo.cli_operational", "migrate_independence_cmd"),
        ("operational", "memo.cli_operational", "operational_group"),
        ("feedback", "memo.cli_feedback", "feedback_group"),
        ("repo", "memo.cli_repo", "repo_group"),
        ("recall-daemon", "memo.cli_recall_daemon", "recall_daemon_group"),
        ("idle-daemon", "memo.cli_idle_daemon", "idle_daemon_group"),
        ("ingest-daemon", "memo.cli_ingest_daemon", "ingest_daemon_group"),
        ("maint-daemon", "memo.cli_maint_daemon", "maint_daemon_group"),
        ("embed-daemon", "memo.cli_embed_daemon", "embed_daemon_group"),
        ("as-of", "memo.cli_as_of", "as_of_group"),
        ("session", "memo.cli_session", "session_group"),
        ("secret", "memo.cli_secret", "secret"),
        ("continuity", "memo.cli_session", "continuity_cmd"),
        ("temporal", "memo.cli_temporal", "temporal_group"),
        ("consolidate", "memo.cli_consolidate", "consolidate_group"),
        ("health", "memo.cli_health", "health"),
        ("dashboard", "memo.cli_dashboard", "dashboard_cmd"),
        ("compress-context", "memo.cli_compress_context", "compress_context_cmd"),
        ("daemons", "memo.cli_daemons", "daemons_group"),
        ("dedupe", "memo.cli_dedupe", "dedupe_cmd"),
        ("cross-dedup", "memo.cli_dedupe", "dedupe_cmd"),
        ("ops", "memo.cli_ops", "ops_group"),
        ("contextual", "memo.cli_contextual", "contextual_group"),
        ("links", "memo.cli_links", "links_group"),
        ("version", "memo.cli_version", "version_group"),
        ("release", "memo.cli_release", "release_group"),
        ("review", "memo.cli_review", "review_group"),
        ("query", "memo.cli_query", "query_group"),
        ("backup", "memo.cli_backup", "backup_group"),
        ("sync", "memo.cli_sync", "sync_group"),
        ("terminal", "memo.cli_terminal", "terminal_group"),
        ("http-api", "memo.cli_http", "http_api"),
        ("analytics", "memo.cli_analytics", "analytics_group"),
        ("import", "memo.cli_import", "import_group"),
        ("export", "memo.cli_export", "export_group"),
        ("collaborative", "memo.cli_collaborative", "collaborative_group"),
        ("contradict", "memo.cli_contradict", "contradict_group"),
        ("coordinate", "memo.cli_coordinate", "coordinate_group"),
        ("startup-banner", "memo.cli_banner", "startup_banner_cmd"),
        ("codex-badge", "memo.cli_banner", "codex_badge_cmd"),
        ("install-shims", "memo.runtime.shims", "install_shims_cmd"),
        ("onboard", "memo.cli_onboard", "onboard"),
    ]:
        name, module, symbol = name, module, symbol
        _reg(name, module, symbol)


_register_lazy_commands()


_LOADED_MODULES: set[str] = set()


def _import_symbol(module: str, symbol: str) -> Any:
    """Import `module` (exactly once across the lazy map) and return its
    `symbol` — the Click command/group object registered for a command name."""
    import importlib

    if module not in _LOADED_MODULES:
        _LOADED_MODULES.add(module)
        importlib.import_module(module)
    return getattr(importlib.import_module(module), symbol, None)


_COMMAND_SECTIONS: list[tuple[str, list[str]]] = [
    (
        "Core",
        [
            "save",
            "search",
            "ask",
            "context",
            "context-pack",
            "get",
            "edit",
            "rename",
            "delete",
            "list",
        ],
    ),
    (
        "Recall & Hooks",
        ["recall", "recall-hook", "briefing", "continuity", "prewarm", "capture-tick", "capture-stop"],
    ),
    (
        "Session & History",
        ["history", "as-of", "diff", "record-history", "session", "resume", "reflect", "mine-history", "episodes"],
    ),
    (
        "Maintenance",
        ["reindex", "maintain", "dream", "consolidate", "synthesize", "dedupe", "cross-dedup", "retier", "contradict", "coordinate", "terminal", "invalidate", "temporal", "compress-context"],
    ),
    (
        "Analysis & Quality",
        ["health", "stats", "doctor", "lint", "analytics", "eval", "roi", "tokens", "token-savings", "usefulness", "gaps", "outcome", "profile"],
    ),
    (
        "Knowledge Graph",
        ["graph", "entities", "entity", "extract-entities", "links", "version", "related"],
    ),
    (
        "Advanced Search",
        ["embed", "rerank", "contextual", "chat", "chat-ask", "repo"],
    ),
    (
        "Import / Export / Sync",
        ["import", "export", "backup", "restore", "sync", "ingest"],
    ),
    (
        "Visualization",
        ["tui", "dashboard", "map", "logs", "hook-log"],
    ),
    (
        "Setup & Config",
        ["init", "config", "install-mcp", "install-watcher", "uninstall-watcher", "install-slash", "install-statusline", "install-recall-hook", "install-shell-wrapper", "install-shims", "startup-banner", "migrate", "migrate-vault", "update", "watch", "release"],
    ),
    (
        "Daemons",
        ["recall-daemon", "ingest-daemon", "maint-daemon", "embed-daemon", "idle-daemon"],
    ),
]
_SECTIONED: set[str] = {cmd for _, cmds in _COMMAND_SECTIONS for cmd in cmds}


class SurfaceGroup(click.Group):
    """Root Click group: lazy command resolution + surface-profile filtering."""

    def list_commands(self, ctx: click.Context) -> list[str]:
        from memo.surface import cli_command_visible

        names = set(super().list_commands(ctx)) | set(_LAZY.keys())
        return [name for name in sorted(names) if cli_command_visible(name)]

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        from memo.surface import cli_command_visible

        if not cli_command_visible(cmd_name):
            return None
        if cmd_name not in self.commands and cmd_name in _LAZY:
            module, symbol, alias = _LAZY[cmd_name]
            cmd = _import_symbol(module, symbol)
            if cmd is not None:
                self.add_command(cmd, name=alias or cmd_name)
        return super().get_command(ctx, cmd_name)

    def resolve_command(
        self, ctx: click.Context, args: list[str]
    ) -> tuple[str | None, click.Command | None, list[str]]:
        # A lazy command's name is not yet in `self.commands`, so Click's
        # parse_args cannot discover it. Pre-resolve: if the first non-option
        # token names a lazy command, materialize it before Click resolves.
        for arg in args:
            if arg.startswith("-"):
                continue
            if arg in _LAZY and arg not in self.commands:
                self.get_command(ctx, arg)
            break
        return super().resolve_command(ctx, args)

    def load_all(self, ctx: click.Context | None = None) -> None:
        """Import + register every lazy command (for help walkers / tests)."""
        for name in _LAZY:
            module, symbol, alias = _LAZY[name]
            cmd = _import_symbol(module, symbol)
            if cmd is not None and name not in self.commands:
                self.add_command(cmd, name=alias or name)

    def format_commands(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        from memo.surface import cli_command_visible

        visible = {
            name
            for name in self.list_commands(ctx)
            if not getattr(self.get_command(ctx, name), "hidden", False)
        }
        overflow: list[str] = [
            cmd for cmd in sorted(visible) if cmd not in _SECTIONED and cli_command_visible(cmd)
        ]
        sections = list(_COMMAND_SECTIONS)
        if overflow:
            other_idx = next((i for i, (title, _) in enumerate(sections) if title == "Other"), None)
            if other_idx is not None:
                title, existing = sections[other_idx]
                sections[other_idx] = (title, existing + overflow)
            else:
                sections.append(("Other", overflow))

        for section_title, cmd_names in sections:
            cmds_in_section = [
                (name, self.get_command(ctx, name)) for name in cmd_names if name in visible
            ]
            cmds_in_section = [(n, c) for n, c in cmds_in_section if c is not None]
            if not cmds_in_section:
                continue
            rows: list[tuple[str, str]] = []
            for name, cmd in cmds_in_section:
                if cmd is None:
                    continue
                rows.append((name, cmd.get_short_help_str(limit=formatter.width)))
            with formatter.section(section_title):
                formatter.write_dl(rows)


@click.group(cls=SurfaceGroup)
@click.version_option(package_name="mlx-memo")
@click.pass_context
def cli(ctx: click.Context) -> None:
    """memo — local MLX memory.

    Stable core: save/search/ask CRUD, EvidencePack, operational continuity,
    outcome learning, signed federation, briefing/recall-hook, reindex/doctor,
    and history/as-of flows. Many other commands are advanced or experimental.
    """
    _first_run_gate(ctx)


def _load_all_commands() -> None:
    """Import + register every lazy command onto the root group.

    Used by help/boot paths and tests that walk `cli.commands` directly.
    """
    cli.load_all()


def _first_run_gate(ctx: click.Context) -> None:
    import sys as _sys

    from memo.flags import flag_bool

    if ctx.invoked_subcommand in (None, *_FIRST_RUN_GATE_SKIP_COMMANDS):
        return
    if flag_bool("MEMO_NONINTERACTIVE"):
        return
    if not (_sys.stdin.isatty() and _sys.stdout.isatty()):
        return
    if "MEMO_DATA_DIR" in os.environ:
        return
    if "MEMO_VAULT_PATH" in os.environ and "MEMO_MEMORY_SUBDIR" in os.environ:
        return
    from memo.config_md import config_dir as _markdown_config_dir
    from memo.config_md import index_path as _markdown_index_path
    from memo.setup.config_io import _resolve_config_path

    if _markdown_index_path().is_file() or _markdown_config_dir().is_dir():
        return
    if _resolve_config_path().is_file():
        return
    _run_picker_and_save()


def _run_picker_and_save() -> None:
    from memo.config_md import write_default_config

    console.print(
        "[bold]memo first-run setup[/bold] — pick where memories should live.\n",
    )
    try:
        result = run_picker()
    except KeyboardInterrupt:
        console.print(
            "[yellow]aborted.[/yellow] Re-run any memo command to retry, "
            "or run `memo init` to configure.",
        )
        raise click.exceptions.Exit(130) from None
    written = write_default_config(
        data_dir=result.data_dir,
        vault_path=result.vault_path,
        force=True,
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
    console.print(f"[dim]config saved: {written[0].parent}[/dim]")
    console.print("💡 Tip: sincronizá memorias entre Macs → `memo sync setup` cuando quieras")


def main() -> None:
    from memo.errors import MemoError

    try:
        cli()
    except MemoError as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()