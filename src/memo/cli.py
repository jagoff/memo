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

import os

import click

from memo.cli_analytics import analytics_group
from memo.cli_as_of import as_of_group
from memo.cli_backend_native import backend_native_group
from memo.cli_backup import backup_group
from memo.cli_banner import codex_badge_cmd, startup_banner_cmd
from memo.cli_briefing import briefing
from memo.cli_capture import capture_stop, capture_tick, episodes_group, resume
from memo.cli_chat import chat_group
from memo.cli_chronicle import chronicle_cmd
from memo.cli_collaborative import collaborative_group
from memo.cli_common import console
from memo.cli_compress_context import compress_context_cmd
from memo.cli_confidence import confidence_group
from memo.cli_config import config_group
from memo.cli_consolidate import consolidate_group
from memo.cli_contextual import contextual_group
from memo.cli_contradict import contradict_group
from memo.cli_dashboard import dashboard_cmd
from memo.cli_debug_recall import debug_recall_cmd
from memo.cli_dedupe import dedupe_cmd
from memo.cli_definitive import definitive_group
from memo.cli_diag import (
    _recall_daemon_health,  # noqa: F401 — re-exported for test: tests/test_logs_and_doctor.py imports from memo.cli
)
from memo.cli_doctor import doctor
from memo.cli_dream import dream_cmd
from memo.cli_drift import drift as drift_cmd
from memo.cli_embed_daemon import embed_daemon_group
from memo.cli_entities import entities, entity, extract_entities
from memo.cli_eval import eval_group
from memo.cli_export import export_group
from memo.cli_federation import federation_group
from memo.cli_feedback import feedback_group
from memo.cli_graduation import graduation_group
from memo.cli_graph import graph_group
from memo.cli_guard import guard_group
from memo.cli_health import health as health_cmd
from memo.cli_history import diff_cmd, history_cmd
from memo.cli_hooks import install_recall_hook
from memo.cli_http import http_api
from memo.cli_hype import hype_group
from memo.cli_idle_daemon import idle_daemon_group
from memo.cli_import import import_group
from memo.cli_ingest import ingest
from memo.cli_ingest_daemon import ingest_daemon_group
from memo.cli_install_mcp import install_mcp
from memo.cli_interject import ask_group, interject_group
from memo.cli_invalidate import invalidate_cmd
from memo.cli_journey import journey_check
from memo.cli_links import links_group
from memo.cli_maint_daemon import maint_daemon_group
from memo.cli_maintain import maintain_cmd
from memo.cli_mandate import mandate as mandate_cmd
from memo.cli_memory import (
    delete,
    fix,
    get,
    history,
    lint,
    list_cmd,
    ocr_image,
    provenance,
    reindex,
    rename,
    restore,
    save,
    undo,
    update,
)
from memo.cli_onboard import onboard
from memo.cli_operational import evidence_cmd, migrate_independence_cmd, operational_group
from memo.cli_outcome import gaps as gaps_cmd
from memo.cli_outcome import outcome as outcome_cmd
from memo.cli_proactive import digest
from memo.cli_profile import profile_group
from memo.cli_query import query_group
from memo.cli_recall_daemon import recall_daemon_group
from memo.cli_recall_hook import recall_hook
from memo.cli_related import related
from memo.cli_release import release_group
from memo.cli_repo import repo_group
from memo.cli_retier import retier_cmd
from memo.cli_retrieve import retrieve_cmd
from memo.cli_review import review_group
from memo.cli_roi import roi as roi_cmd
from memo.cli_runtime import (
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
from memo.cli_search import (
    ask,
    chat_ask,
    context_cmd,
    context_pack_cmd,
    embed_cmd,
    recall,
    rerank_cmd,
    search,
)
from memo.cli_secret import secret as secret_group
from memo.cli_session import continuity_cmd, session_group
from memo.cli_setup import setup_cmd
from memo.cli_stats import stats
from memo.cli_statusline import install_statusline
from memo.cli_sync import sync_group
from memo.cli_synthesize import synthesize_cmd
from memo.cli_temporal import temporal_group
from memo.cli_token_gate import token_gate_cmd
from memo.cli_token_savings import token_savings_cmd
from memo.cli_tokens import tokens_cmd
from memo.cli_transcripts import mine_git, mine_history, reflect
from memo.cli_tui import hook_log, logs, tui
from memo.cli_usefulness import usefulness as usefulness_cmd
from memo.cli_verbatim import verbatim_group
from memo.cli_version import version_group
from memo.cli_viz import map_cmd
from memo.runtime.shims import install_shims_cmd

# Imported at module scope (not lazily) so tests can `patch("memo.cli.run_picker", ...)`.
# `run_picker` itself defers the heavy `questionary` import until called.
from memo.setup import run_picker

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
        [
            "recall",
            "recall-hook",
            "briefing",
            "continuity",
            "prewarm",
            "capture-tick",
            "capture-stop",
        ],
    ),
    (
        "Session & History",
        [
            "history",
            "as-of",
            "diff",
            "record-history",
            "session",
            "resume",
            "reflect",
            "mine-history",
            "episodes",
        ],
    ),
    (
        "Maintenance",
        [
            "reindex",
            "maintain",
            "dream",
            "consolidate",
            "synthesize",
            "dedupe",
            "cross-dedup",
            "retier",
            "contradict",
            "invalidate",
            "temporal",
            "compress-context",
        ],
    ),
    (
        "Analysis & Quality",
        [
            "health",
            "stats",
            "doctor",
            "lint",
            "analytics",
            "eval",
            "roi",
            "tokens",
            "token-savings",
            "usefulness",
            "gaps",
            "outcome",
            "profile",
        ],
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
        [
            "init",
            "config",
            "install-mcp",
            "install-watcher",
            "uninstall-watcher",
            "install-slash",
            "install-statusline",
            "install-recall-hook",
            "install-shell-wrapper",
            "install-shims",
            "startup-banner",
            "migrate",
            "migrate-vault",
            "update",
            "watch",
            "release",
        ],
    ),
    (
        "Daemons",
        ["recall-daemon", "ingest-daemon", "maint-daemon", "embed-daemon", "idle-daemon"],
    ),
    (
        "Other",
        [
            "backend-native",
            "collaborative",
            "feedback",
            "query",
            "mandate",
            "sleep-cycle",
            "ocr-image",
            "provenance",
            "mcp-command",
            "secret",
        ],
    ),
]

# Flat set of all commands with an assigned section (for "Other" overflow).
_SECTIONED: set[str] = {cmd for _, cmds in _COMMAND_SECTIONS for cmd in cmds}


class SurfaceGroup(click.Group):
    """Root Click group that filters commands by the configured surface profile."""

    def list_commands(self, ctx: click.Context) -> list[str]:
        from memo.surface import cli_command_visible

        return [name for name in super().list_commands(ctx) if cli_command_visible(name)]

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        from memo.surface import cli_command_visible

        if not cli_command_visible(cmd_name):
            return None
        return super().get_command(ctx, cmd_name)

    def format_commands(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        from memo.surface import cli_command_visible

        visible = {
            name
            for name in self.list_commands(ctx)
            if not getattr(self.get_command(ctx, name), "hidden", False)
        }

        # Collect commands per section, then an overflow "Other" bucket.
        overflow: list[str] = [
            cmd for cmd in sorted(visible) if cmd not in _SECTIONED and cli_command_visible(cmd)
        ]

        sections = list(_COMMAND_SECTIONS)
        if overflow:
            # Merge overflow into the last "Other" section or append it.
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
                short_help = cmd.get_short_help_str(limit=formatter.width)
                rows.append((name, short_help))

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


# Command groups extracted from this module live in cli_*.py and register here.
cli.add_command(graduation_group)
cli.add_command(confidence_group)
cli.add_command(graph_group)
cli.add_command(guard_group)
cli.add_command(hype_group)
cli.add_command(interject_group)
cli.add_command(ask_group)
cli.add_command(related)
cli.add_command(eval_group)
cli.add_command(journey_check)
cli.add_command(debug_recall_cmd)
cli.add_command(dream_cmd)
cli.add_command(chronicle_cmd)
cli.add_command(maintain_cmd)
cli.add_command(invalidate_cmd)
cli.add_command(synthesize_cmd)
cli.add_command(retier_cmd)
cli.add_command(usefulness_cmd)
cli.add_command(verbatim_group)
cli.add_command(roi_cmd)
cli.add_command(gaps_cmd)
cli.add_command(outcome_cmd)
cli.add_command(digest)
cli.add_command(mandate_cmd)
cli.add_command(drift_cmd)
cli.add_command(map_cmd)
cli.add_command(tui)
cli.add_command(hook_log)
cli.add_command(logs)
cli.add_command(mine_history)
cli.add_command(mine_git)
cli.add_command(ingest)
cli.add_command(capture_stop)
cli.add_command(capture_tick)
cli.add_command(reflect)
cli.add_command(resume)
cli.add_command(episodes_group)
cli.add_command(diff_cmd)
cli.add_command(history_cmd)
cli.add_command(briefing)
cli.add_command(init_cmd)
cli.add_command(stats)
cli.add_command(token_gate_cmd)
cli.add_command(token_savings_cmd)
cli.add_command(tokens_cmd)
cli.add_command(doctor)
cli.add_command(migrate_vault)
cli.add_command(migrate_vault, name="migrate")  # alias
cli.add_command(mcp_command)
cli.add_command(install_slash)
cli.add_command(install_mcp)
cli.add_command(setup_cmd)
cli.add_command(install_statusline)
cli.add_command(install_recall_hook)
cli.add_command(self_update)  # primary name: "update"
# Back-compat: keep the old `memo upgrade` / `memo self-update` names working
# (now hidden) so any auto-update path or muscle memory still resolves. Same
# callback as the primary `update` command.
_upgrade_alias = click.Command(
    "upgrade",
    callback=self_update.callback,
    params=self_update.params,
    help=self_update.help,
    hidden=True,
)
cli.add_command(_upgrade_alias)
_self_update_alias = click.Command(
    "self-update",
    callback=self_update.callback,
    params=self_update.params,
    help=self_update.help,
    hidden=True,
)
cli.add_command(_self_update_alias)
cli.add_command(watch)
cli.add_command(install_watcher)
cli.add_command(uninstall_watcher_cmd)
cli.add_command(sleep_cycle)
cli.add_command(prewarm)
cli.add_command(recall_hook)
cli.add_command(install_shell_wrapper)
cli.add_command(config_group)
cli.add_command(save)
cli.add_command(search)
cli.add_command(recall)
cli.add_command(ask)
cli.add_command(context_cmd)
cli.add_command(context_pack_cmd)
cli.add_command(embed_cmd)
cli.add_command(chat_ask)
cli.add_command(chat_group)
cli.add_command(rerank_cmd)
cli.add_command(list_cmd)
cli.add_command(get)
cli.add_command(update)
cli.add_command(rename)
cli.add_command(reindex)
cli.add_command(delete)
cli.add_command(undo)
cli.add_command(fix)
cli.add_command(history)
cli.add_command(retrieve_cmd)
cli.add_command(ocr_image)
cli.add_command(provenance)
cli.add_command(extract_entities)
cli.add_command(entities)
cli.add_command(entity)
cli.add_command(lint)
cli.add_command(restore)
cli.add_command(profile_group)
cli.add_command(backend_native_group)
cli.add_command(evidence_cmd)
cli.add_command(definitive_group)
cli.add_command(federation_group)
cli.add_command(migrate_independence_cmd)
cli.add_command(operational_group)
cli.add_command(feedback_group)
cli.add_command(repo_group)
cli.add_command(recall_daemon_group)
cli.add_command(idle_daemon_group)
cli.add_command(ingest_daemon_group)
cli.add_command(maint_daemon_group)
cli.add_command(embed_daemon_group)
cli.add_command(as_of_group)
cli.add_command(session_group)
cli.add_command(secret_group)
cli.add_command(continuity_cmd)
cli.add_command(temporal_group)
cli.add_command(consolidate_group)
cli.add_command(health_cmd)
cli.add_command(dashboard_cmd)
cli.add_command(compress_context_cmd)
cli.add_command(dedupe_cmd)
cli.add_command(contextual_group)
cli.add_command(links_group)
cli.add_command(version_group)
cli.add_command(release_group)
cli.add_command(review_group)
cli.add_command(query_group)
cli.add_command(backup_group)
cli.add_command(sync_group)
cli.add_command(analytics_group)
cli.add_command(import_group)
cli.add_command(export_group)
cli.add_command(collaborative_group)
cli.add_command(contradict_group)
cli.add_command(http_api)
cli.add_command(startup_banner_cmd)
cli.add_command(codex_badge_cmd)
cli.add_command(install_shims_cmd)
cli.add_command(onboard)


# Subcommands that must NEVER trigger the first-run picker — either
# because they're part of setup/diagnostics, they don't need storage,
# or they run from non-interactive hooks (the TTY check + the
# `MEMO_NONINTERACTIVE=1` env var in `hooks.json` handle the latter,
# but listing the names here is a belt-and-suspenders defence in case
# something invokes them from an interactive shell while debugging).
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
}


def _first_run_gate(ctx: click.Context) -> None:
    """If the user hasn't configured `memo` yet, run the picker first.

    Resolution: skip when invoked from hooks (MEMO_NONINTERACTIVE=1 or
    non-TTY), when an env var already configures `data_dir`, when a
    Markdown or legacy config already exists, or when the legacy `MEMO_VAULT_PATH`
    pair is set (back-compat path). Also skip for setup/diagnostic
    subcommands so the user can always recover via `memo doctor`.
    """
    import sys as _sys

    from memo.flags import flag_bool

    if ctx.invoked_subcommand in (None, *_FIRST_RUN_GATE_SKIP_COMMANDS):
        return
    if flag_bool("MEMO_NONINTERACTIVE"):
        return
    # Both stdin and stdout must be a TTY for the picker to make sense.
    if not (_sys.stdin.isatty() and _sys.stdout.isatty()):
        return
    if "MEMO_DATA_DIR" in os.environ:
        return
    if "MEMO_VAULT_PATH" in os.environ and "MEMO_MEMORY_SUBDIR" in os.environ:
        return
    # Re-resolve config locations at gate-firing time (env may have
    # changed between import and invocation, e.g. in tests).
    from memo.config_md import config_dir as _markdown_config_dir
    from memo.config_md import index_path as _markdown_index_path
    from memo.setup.config_io import _resolve_config_path

    if _markdown_index_path().is_file() or _markdown_config_dir().is_dir():
        return
    if _resolve_config_path().is_file():
        return
    _run_picker_and_save()


def _run_picker_and_save() -> None:
    """Drive the interactive picker → persist Markdown config → return.

    Caller is expected to be the first-run gate (or `memo init`). Picker
    aborts (Ctrl-C / ESC) raise `click.exceptions.Exit(130)` so the
    surrounding CLI invocation halts cleanly.
    """
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
    from memo.config_md import write_default_config

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
    # Present memo's own domain errors (e.g. "LLM features require MLX" on the
    # CPU backend) as a clean message instead of an uncaught traceback.
    from memo.errors import MemoError

    try:
        cli()
    except MemoError as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
