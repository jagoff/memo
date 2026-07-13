"""`memo briefing` — El Briefing (open loops + memory of the day) for SessionStart.

Extracted from cli.py (2b god-module decomposition). Registered via
`cli.add_command(briefing)`.
"""

from __future__ import annotations

import contextlib
from typing import Any

import click

from memo.config import Config


@click.command(name="briefing")
@click.option(
    "--compact",
    is_flag=True,
    help="Emit a startup capsule capped at 480 characters.",
)
def briefing(*, compact: bool) -> None:
    """SessionStart hook — rich context panel.

    Emits a `hookSpecificOutput` JSON with `additionalContext` markdown
    containing:
      - Last session for the current project (crash recovery)
      - Open loops: recently updated memories (in-flight decisions)
      - Memory of the day: one memory picked deterministically by date
      - Quick interaction guide

    All errors swallowed — a failed briefing is worse than no briefing
    only if it blocks the session. Exit 0 + `{}` on any failure.

    Env vars:
      MEMO_BRIEFING_DISABLE        — set to "1" to skip entirely
      MEMO_BRIEFING_LOOPS_N        — open-loop count to show (default 5)
      MEMO_BRIEFING_LOOPS_DAYS     — how recent counts as "open" (default 7)
      MEMO_BRIEFING_DEBUG          — print errors to stderr
    """
    import json as _json
    import os as _os
    import sys as _sys

    from memo.flags import flag_bool, flag_int

    debug = flag_bool("MEMO_BRIEFING_DEBUG")
    mem: Any | None = None

    def _close_memory() -> None:
        if mem is not None:
            with contextlib.suppress(Exception):
                mem.close()

    def _bail(reason: str = "") -> None:
        if reason and debug:
            print(f"# memo briefing: {reason}", file=_sys.stderr)
        _close_memory()
        print("{}")
        _sys.exit(0)

    def _log_context_cost(context: str, session_id: str | None = None) -> None:
        try:
            from memo.dashboard import append_context_cost_log

            append_context_cost_log(
                cfg.state_dir,
                kind="briefing",
                chars=len(context),
                client="claude-code",
                session_id=session_id,
            )
        except Exception as exc:
            if debug:
                print(f"# memo briefing: context-cost log failed: {exc}", file=_sys.stderr)

    if flag_bool("MEMO_BRIEFING_DISABLE"):
        _bail("disabled")
        return

    try:
        cfg = Config.from_env()
        from memo.memory import Memory

        mem = Memory(cfg)
    except Exception as exc:
        _bail(f"Memory init failed: {exc}")
        return

    if compact:
        from pathlib import Path as _Path

        from memo.briefing import compact_text
        from memo.session import format_relative, list_sessions

        cur_cwd = str(_Path(_os.getcwd()).resolve())
        # Targeted lookup first so the project session is never buried by
        # sessions from other repos (avoids false "No recent session" when
        # the project session is past the global limit).
        same_proj = list_sessions(cfg.state_dir, cwd=cur_cwd, limit=1)
        sid = ""
        compact_lines = ["## Memo"]
        if same_proj:
            top = same_proj[0]
            summary = (top.get("summary") or top.get("last_user_msg") or "—").replace(
                "\n", " "
            )
            state = (top.get("running_summary") or "").replace("\n", " ")
            compact_lines.append(
                f"Last session ({format_relative(top.get('updated'))}): "
                f"{compact_text(summary, max_chars=160)}"
            )
            sid = str(top.get("session_id") or "")
            if sid:
                compact_lines.append(f"Resume: `claude --resume {sid}`")
            if state:
                compact_lines.append(f"State: {compact_text(state, max_chars=200)}")
        else:
            compact_lines.append("No recent session in this project.")
        # Sync-onboarding nudge — honest (only when no remote is configured),
        # dismissable, and included in the compact output budget.
        from memo.sync_git import sync_nudge_dismissed, sync_tier

        nudge = ""
        if sync_tier(cfg) == "local" and not sync_nudge_dismissed(cfg):
            nudge = (
                "\n💡 memo sync off — compartí entre Macs: `memo sync setup` "
                "(ocultar: `memo sync setup --never`)"
            )
        context = compact_text(
            "\n".join(compact_lines),
            max_chars=max(1, 480 - len(nudge)),
        )
        context += nudge
        output = {
            "hookSpecificOutput": {
                "hookEventName": "SessionStart",
                "additionalContext": context,
            }
        }
        _log_context_cost(context, sid if same_proj else None)
        print(_json.dumps(output, ensure_ascii=False))
        _close_memory()
        return

    loops_n_flag = flag_int("MEMO_BRIEFING_LOOPS_N")
    loops_days_flag = flag_int("MEMO_BRIEFING_LOOPS_DAYS")
    loops_n = 5 if loops_n_flag is None else max(0, loops_n_flag)
    loops_days = 7 if loops_days_flag is None else max(0, loops_days_flag)

    lines: list[str] = []

    # ── 1. Last session for this project ──────────────────────────────────
    try:
        from pathlib import Path as _Path

        from memo.session import format_relative, list_sessions, render_active_memory

        cur_cwd = str(_Path(_os.getcwd()).resolve())
        all_sessions = list_sessions(cfg.state_dir, limit=20)
        same_proj = [r for r in all_sessions if (r.get("cwd") or "") == cur_cwd]
        if same_proj:
            top = same_proj[0]
            sid = top.get("session_id") or ""
            when = format_relative(top.get("updated"))
            summary = (top.get("summary") or top.get("last_user_msg") or "—").replace("\n", " ")[
                :120
            ]
            lines.append("## Briefing")
            lines.append("")
            lines.append(f"**Last session in this project** ({when}): {summary}")
            lines.append(f"`claude --resume {sid}`")
            lines.append("")
            active_memory = render_active_memory(top)
            if active_memory:
                lines.extend(active_memory)
                lines.append("")
    except Exception as exc:
        if debug:
            print(f"# memo briefing: session lookup failed: {exc}", file=_sys.stderr)
        if not lines:
            lines.append("## Briefing")
            lines.append("")

    # ── 1a. Onboarding proof (install seed — shown once) ──────────────────
    try:
        from memo.briefing import install_seed_lines

        lines.extend(install_seed_lines(cfg.state_dir))
    except Exception as exc:
        if debug:
            print(f"# memo briefing: install seed failed: {exc}", file=_sys.stderr)

    # ── 1b. Unified consciousness (Synapse) ───────────────────────────────
    # Pulls present_state (memflow handoffs/focus) + reality_conflicts from
    # `synapse packet`. No-op when synapse is not installed or unreachable —
    # the rest of the briefing is unaffected (graceful, opt-in boundary).
    if not flag_bool("MEMO_BRIEFING_SYNAPSE_DISABLE"):
        try:
            from memo.briefing import synapse_briefing_lines

            try:
                cur_cwd_str = _os.getcwd()
            except Exception:
                cur_cwd_str = ""
            syn_lines = synapse_briefing_lines(cur_cwd_str)
            if syn_lines:
                lines.extend(syn_lines)
        except Exception as exc:
            if debug:
                print(f"# memo briefing: synapse lookup failed: {exc}", file=_sys.stderr)

    # ── 1c. Dream digest (last nightly run — shown once per receipt) ──────
    if flag_bool("MEMO_BRIEFING_DREAM_DIGEST"):
        try:
            from memo.briefing import dream_digest_lines

            lines.extend(dream_digest_lines(cfg.state_dir))
        except Exception as exc:
            if debug:
                print(f"# memo briefing: dream digest failed: {exc}", file=_sys.stderr)

    # ── 1d. Ask one recurring gap (Phase 3, default OFF) ──────────────────
    try:
        from memo.ask_gaps import briefing_lines as _ask_lines
        from memo.identity import current as _identity

        _sid = _identity(cfg).session_id or "_no_session"
        lines.extend(_ask_lines(cfg, session_id=_sid))
    except Exception as exc:
        if debug:
            print(f"# memo briefing: ask-gaps failed: {exc}", file=_sys.stderr)

    # ── 2+3. Open loops + memory of the day (memo's own corpus) ───────────
    # Shared with the memo_unified_briefing MCP tool so both surfaces compose
    # identical memo-native sections.
    try:
        from memo.briefing import memo_native_briefing_lines

        lines.extend(
            memo_native_briefing_lines(mem, loops_n=loops_n, loops_days=loops_days)
        )
    except Exception as exc:
        if debug:
            print(f"# memo briefing: memo-native sections failed: {exc}", file=_sys.stderr)

    # ── 4. Interaction guide ──────────────────────────────────────────────
    lines.append(
        "**MEMORY-FIRST MANDATE:** Your first action should be querying the memory store "
        "to ensure your context is up-to-date. Do not rely on internal training data for "
        "project-specific details.\n\n"
        "_To continue: `give me loop N` (resume by number) · "
        "`/memo get <id>` · `/memo ask <question>`_"
    )

    if not any(ln for ln in lines if ln and not ln.startswith("#") and not ln.startswith("_")):
        _bail("nothing to show")
        return

    context = "\n".join(lines)
    output = {
        "hookSpecificOutput": {
            "hookEventName": "SessionStart",
            "additionalContext": context,
        }
    }
    _log_context_cost(context)
    print(_json.dumps(output, ensure_ascii=False))
    _close_memory()
