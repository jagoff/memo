"""`memo session` command group — session checkpoints + autosave.

Extracted from cli.py (3a god-module decomposition). Registered onto the
root group in cli.py via `cli.add_command(session_group)`.
"""

from __future__ import annotations

import contextlib
import sys
from typing import Any

import click

from memo.cli_common import console
from memo.config import Config
from memo.flags import flag_bool, flag_int


@click.command(name="continuity")
@click.option("--limit", default=12, show_default=True, help="Sessions to scan for this cwd.")
def continuity_cmd(limit: int) -> None:
    """What was I working on? — resume the latest session for this directory.

    Native-to-memo parity with memflow's flow_continuity, rendered from memo's
    own session snapshots (cwd / branch / running summary / open loops). Reads
    only; the data is captured by the Stop-hook checkpoint.
    """
    import os
    from pathlib import Path

    from memo.session import list_sessions, render_continuity

    cfg = Config.from_env()
    try:
        rows = list_sessions(cfg.state_dir, limit=limit)
    except Exception as exc:
        console.print(f"[red]continuity failed:[/red] {exc}")
        sys.exit(1)
    cwd = str(Path(os.getcwd()).resolve())
    console.print(render_continuity(rows, cwd))


@click.group(name="session")
def session_group() -> None:
    """Internal session-snapshot ops — hook targets, not user-facing.

    User-facing entry is `memo resume` (list / inspect). This group
    holds the wiring the hooks call: `checkpoint` (Stop hook), `recent`
    (SessionStart additionalContext), and `prune` (LRU cleanup). Stays
    namespaced so `memo --help` doesn't surface plumbing as if it were
    everyday CLI.
    """


@session_group.command(name="checkpoint")
@click.option(
    "--session-id", default=None, help="Override session_id (default: read from stdin payload)."
)
@click.option(
    "--cwd",
    default=None,
    help="Override cwd (default: read from stdin payload, fallback os.getcwd).",
)
@click.option("--transcript-path", default=None, help="Override transcript path.")
@click.option("--lru-cap", default=50, type=int, show_default=True)
@click.option("--json", "as_json", is_flag=True, help="Print the persisted snapshot as JSON.")
def session_checkpoint(
    session_id: str | None,
    cwd: str | None,
    transcript_path: str | None,
    lru_cap: int,
    as_json: bool,
) -> None:
    """Stop hook entrypoint — upsert a session snapshot from stdin JSON.

    Reads the Stop hook payload from stdin (Claude Code passes
    `{"session_id", "transcript_path", "cwd", ...}`). Falls back to
    flags/cwd if stdin is empty (lets you run it manually for testing).

    Always exits 0 — like the other hooks, a checkpoint failure must
    not block Claude Code. On any exception we swallow + print `{}`.
    """
    import json as _json
    import os as _os
    import sys as _sys

    if flag_bool("MEMO_SESSION_DISABLE"):
        if as_json:
            click.echo("{}")
        sys.exit(0)

    payload: dict[str, Any] = {}
    # Stdin is a TTY when run interactively → don't block on read.
    if not _sys.stdin.isatty():
        try:
            raw = _sys.stdin.read()
            if raw.strip():
                payload = _json.loads(raw)
        except _json.JSONDecodeError:
            payload = {}

    sid = session_id or payload.get("session_id")
    cwd_resolved = cwd or payload.get("cwd") or _os.getcwd()
    transcript = transcript_path or payload.get("transcript_path")
    prompt_text = payload.get("prompt")  # present in UserPromptSubmit events

    # Async hooks don't receive piped stdin — fall back to the payload file
    # written by the preceding sync `autosave` hook.
    if not sid:
        try:
            import json as _json2

            _cfg0 = Config.from_env()
            _pfile = _cfg0.state_dir / "last_hook_payload.json"
            if _pfile.exists():
                _fb = _json2.loads(_pfile.read_text(encoding="utf-8"))
                sid = _fb.get("session_id")
                cwd_resolved = cwd_resolved or _fb.get("cwd") or _os.getcwd()
                transcript = transcript or _fb.get("transcript_path")
        except Exception:
            pass

    if not sid:
        # Without a session_id we can't key the snapshot. Fail silently
        # so the hook still exits 0.
        if as_json:
            click.echo("{}")
        sys.exit(0)

    try:
        from memo.session import checkpoint as _checkpoint

        cfg = Config.from_env()
        cfg.ensure_dirs()
        snap = _checkpoint(
            cfg.state_dir,
            session_id=sid,
            cwd=cwd_resolved,
            transcript_path=transcript,
            prompt=prompt_text,
            lru_cap=lru_cap,
        )
    except Exception as exc:
        if flag_bool("MEMO_SESSION_DEBUG"):
            print(f"# memo session checkpoint failed: {exc}", file=_sys.stderr)
        if as_json:
            click.echo("{}")
        sys.exit(0)

    if as_json:
        click.echo(_json.dumps(snap, ensure_ascii=False, indent=2))


@session_group.command(name="autosave")
@click.option(
    "--threshold-kb",
    default=1024,
    type=int,
    show_default=True,
    help="Transcript size (KB) that triggers an autosave.",
)
@click.option(
    "--cooldown",
    default=300,
    type=int,
    show_default=True,
    help="Minimum seconds between autosaves for the same session.",
)
def session_autosave(threshold_kb: int, cooldown: int) -> None:
    """UserPromptSubmit hook — proactive save when context approaches limits.

    Stats the transcript file (fast, O(1)). If it exceeds threshold_kb
    and the per-session cooldown has elapsed, spawns ``memo capture-stop``
    in a detached background process and emits an additionalContext warning.

    Always exits 0 — never blocks the prompt.
    """
    import json as _json
    import os as _os
    import subprocess as _sp
    import sys as _sys

    if flag_bool("MEMO_SESSION_DISABLE"):
        print("{}")
        _sys.exit(0)

    payload: dict[str, Any] = {}
    if not _sys.stdin.isatty():
        try:
            raw = _sys.stdin.read()
            if raw.strip():
                payload = _json.loads(raw)
        except _json.JSONDecodeError:
            payload = {}

    sid = payload.get("session_id")
    transcript = payload.get("transcript_path")
    cwd = payload.get("cwd") or _os.getcwd()

    if not sid or not transcript:
        print("{}")
        _sys.exit(0)

    try:
        from memo.session import check_autosave, mark_autosaved

        cfg = Config.from_env()
        # Persist the payload so async hooks (idle-maintenance, checkpoint)
        # can read session_id/transcript_path without stdin being piped.
        try:
            cfg.ensure_dirs()
            _payload_data = _json.dumps(
                {"session_id": sid, "transcript_path": transcript, "cwd": cwd},
                ensure_ascii=False,
            )
            (cfg.state_dir / "last_hook_payload.json").write_text(_payload_data, encoding="utf-8")
        except Exception:
            pass

        should_save, size_kb = check_autosave(
            cfg.state_dir,
            session_id=sid,
            transcript_path=transcript,
            threshold_kb=threshold_kb,
            cooldown_secs=cooldown,
        )
    except Exception as exc:
        if flag_bool("MEMO_SESSION_DEBUG"):
            print(f"# session autosave check failed: {exc}", file=_sys.stderr)
        print("{}")
        _sys.exit(0)

    if not should_save:
        print("{}")
        _sys.exit(0)

    # Spawn capture-stop detached, passing the hook payload via stdin.
    try:
        env = {**_os.environ, "MEMO_NONINTERACTIVE": "1"}
        capture_payload = _json.dumps(
            {
                "session_id": sid,
                "transcript_path": transcript,
                "cwd": cwd,
            }
        ).encode()
        proc = _sp.Popen(
            ["memo", "capture-stop"],
            stdin=_sp.PIPE,
            stdout=_sp.DEVNULL,
            stderr=_sp.DEVNULL,
            start_new_session=True,
            env=env,
            cwd=cwd,
        )
        if proc.stdin is not None:
            proc.stdin.write(capture_payload)
            proc.stdin.close()
        mark_autosaved(cfg.state_dir, sid)
    except Exception as exc:
        if flag_bool("MEMO_SESSION_DEBUG"):
            print(f"# session autosave spawn failed: {exc}", file=_sys.stderr)
        print("{}")
        _sys.exit(0)

    output = {
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": (
                "## ⚠️ Automatic snapshot\n\n"
                f"This session's transcript exceeds {size_kb} KB "
                f"(threshold: {threshold_kb} KB) — context near the limit. "
                "I launched `memo capture-stop` in the background to preserve "
                "the most important insights. You can continue as normal; "
                "if auto-compaction occurs, the essentials are already in memo.\n"
            ),
        }
    }
    print(_json.dumps(output, ensure_ascii=False))


@session_group.command(name="idle-maintenance")
@click.option(
    "--mode",
    type=click.Choice(["capture", "reflect"], case_sensitive=False),
    default="capture",
    show_default=True,
    help="Delayed idle action to run after the session goes quiet.",
)
@click.option("--delay-secs", default=None, type=int, show_default=True)
@click.option(
    "--detached-worker",
    is_flag=True,
    hidden=True,
    help="Internal: marks the re-spawned detached worker so it does the work "
    "instead of detaching again.",
)
def session_idle_maintenance(mode: str, delay_secs: int | None, detached_worker: bool) -> None:
    """Async idle worker — run capture or reflect after a quiet period.

    The prompt-submit hook invokes this; it immediately re-spawns itself DETACHED
    (`start_new_session`, marked with `_MEMO_IDLE_DETACHED`) and returns, so the
    worker outlives the turn — Claude Code reaps the inline async hook at turn
    end, which used to kill the sleep before it elapsed. The detached worker then
    sleeps for the delay and proceeds only if the user's last prompt (read from
    the transcript) is unchanged — a genuinely new prompt means the user kept
    going, so it self-cancels and a fresh worker handles the next quiet window.

    `capture` mines the current session chunk into durable memories.
    `reflect` synthesizes the active session into a durable arc note.
    Both paths are best-effort and exit 0. Heartbeats land in `idle_capture.log`.
    """
    import json as _json
    import sys as _sys
    import time as _time
    from pathlib import Path as _Path

    from memo.flags import flag_bool, flag_int

    if flag_bool("MEMO_SESSION_DISABLE"):
        print("{}")
        _sys.exit(0)
    if flag_bool("MEMO_CAPTURE_DISABLE"):
        print("{}")
        _sys.exit(0)

    payload: dict[str, Any] = {}
    if not _sys.stdin.isatty():
        try:
            raw = _sys.stdin.read()
            if raw.strip():
                payload = _json.loads(raw)
        except _json.JSONDecodeError:
            payload = {}

    sid = payload.get("session_id")
    transcript = payload.get("transcript_path")

    # Async hooks don't receive piped stdin — fall back to the payload file
    # written by the preceding sync `autosave` hook.
    if not sid:
        try:
            from memo.config import Config as _Config

            _cfg = _Config.from_env()
            _pfile = _cfg.state_dir / "last_hook_payload.json"
            if _pfile.exists():
                _fallback = _json.loads(_pfile.read_text(encoding="utf-8"))
                sid = sid or _fallback.get("session_id")
                transcript = transcript or _fallback.get("transcript_path")
        except Exception:
            pass

    if not sid:
        print("{}")
        _sys.exit(0)

    delay = delay_secs
    if delay is None:
        if mode == "reflect":
            delay = flag_int("MEMO_SESSION_IDLE_REFLECT_SECS") or 300
        else:
            delay = flag_int("MEMO_SESSION_IDLE_CAPTURE_SECS") or 10
    delay = max(0, int(delay))

    # Survive turn-end reaping. Claude Code reaps this inline async hook when the
    # turn ends — before the quiet window elapses — so the sleep was killed and
    # the capture never ran (the idle watermark sat frozen for hours). Re-spawn
    # ourselves in a NEW session (detached from the hook's process group, like
    # the autosave hook does for capture-stop) and return immediately; the
    # detached worker (--detached-worker) owns the sleep + capture and outlives
    # the turn.
    if not detached_worker:
        import os as _os
        import subprocess as _sp

        try:
            _child = _sp.Popen(
                [
                    "memo",
                    "session",
                    "idle-maintenance",
                    "--mode",
                    mode,
                    "--delay-secs",
                    str(delay),
                    "--detached-worker",
                ],
                stdin=_sp.PIPE,
                stdout=_sp.DEVNULL,
                stderr=None,  # inherit parent stderr → appears in terminal
                start_new_session=True,
                env={**_os.environ, "MEMO_NONINTERACTIVE": "1"},
            )
            if _child.stdin is not None:
                _child.stdin.write(
                    _json.dumps({"session_id": sid, "transcript_path": transcript}).encode()
                )
                _child.stdin.close()
        except Exception as _exc:
            if flag_bool("MEMO_SESSION_DEBUG"):
                print(f"# idle-maintenance detach failed: {_exc}", file=_sys.stderr)
        print("{}")
        _sys.exit(0)

    # --- detached worker (own process group, survives the turn) ---
    def _hb(stage: str, **extra: Any) -> None:
        """Heartbeat to idle_capture.log so the inactivity path is observable —
        proves the worker fired, survived the sleep, and what it captured."""
        try:
            from datetime import UTC, datetime

            _sd = Config.from_env().state_dir
            _sd.mkdir(parents=True, exist_ok=True)
            rec = {
                "ts": datetime.now(UTC).isoformat(timespec="seconds"),
                "stage": stage,
                "sid": str(sid)[:8],
                "mode": mode,
                **extra,
            }
            _log = _sd / "idle_capture.log"
            with _log.open("a", encoding="utf-8") as f:
                f.write(_json.dumps(rec, ensure_ascii=False) + "\n")
            # Cap growth: a few lines fire per prompt, so trim to the last 500
            # once it crosses ~200KB — never grows unbounded.
            if _log.stat().st_size > 1024 * 200:
                _lines = _log.read_text(encoding="utf-8").splitlines()[-500:]
                _log.write_text("\n".join(_lines) + "\n", encoding="utf-8")
        except Exception:
            pass

    _hb("start", delay=delay)

    try:
        from memo.session import get_session, read_last_user_msg

        cfg = Config.from_env()
        snap = get_session(cfg.state_dir, str(sid))
        if not snap:
            print("{}")
            _sys.exit(0)
        # Inactivity gate keyed on the user's last prompt, NOT the session
        # `updated` stamp. `updated` is bumped by the same turn's Stop checkpoint
        # (and the per-prompt checkpoint), so keying on it made the worker
        # self-cancel on every turn — the inactivity capture never fired. The
        # transcript's last user message changes ONLY when a genuinely new prompt
        # arrives, so it's the real "did the user keep going?" signal. Read from
        # the transcript (source of truth) to dodge the async checkpoint race.
        _tx = _Path(str(transcript)).expanduser() if transcript else None
        expected_prompt = read_last_user_msg(_tx) if _tx else str(snap.get("last_user_msg") or "")
        if delay > 0:
            _time.sleep(delay)

        current = get_session(cfg.state_dir, str(sid))
        if not current:
            print("{}")
            _sys.exit(0)
        current_prompt = read_last_user_msg(_tx) if _tx else str(current.get("last_user_msg") or "")
        if current_prompt != expected_prompt:
            # A newer prompt arrived during the window → still active; a fresh
            # worker spawned for that turn will handle the quiet period.
            _hb("self-cancel-new-prompt")
            print("{}")
            _sys.exit(0)
        _hb("survived-quiet-window")

        if mode.lower() == "capture":
            if not transcript:
                print("{}")
                _sys.exit(0)
            from memo.capture import run_capture_incremental

            result = run_capture_incremental(
                _Path(str(transcript)).expanduser(), str(sid), debug=flag_bool("MEMO_SESSION_DEBUG")
            )
            _hb("captured", status=str(result.get("status")), saved=len(result.get("saved", [])))
            _titles: list[str] = []
            if result.get("status") == "ok":
                _titles = result.get("saved_titles") or []
            n = len(_titles)
            if _titles:
                _shown = "; ".join(t for t in _titles[:3])
                if n > 3:
                    _shown += f"; +{n - 3} more"
                _notif = f"※ auto save (idle): {_shown}"
            else:
                _notif = "※ auto save (idle): scanned (0 new insights)"
            # Write to the terminal TTY if captured by the memo shim; fall back
            # to stderr (which Claude Code captures to its log, not the terminal).
            import os as _os
            _tty = _os.environ.get("MEMO_AGENT_TTY") or _os.environ.get("MEMFLOW_AGENT_TTY")
            if _tty:
                try:
                    with open(_tty, "w") as _tty_f:
                        _tty_f.write(_notif + "\n")
                except OSError:
                    print(_notif, file=_sys.stderr)
            else:
                print(_notif, file=_sys.stderr)
            if _titles:
                from memo.cli_capture import _write_capture_notification
                _write_capture_notification(cfg.state_dir, _titles, idle=True)
            else:
                with contextlib.suppress(OSError):
                    (cfg.state_dir / "pending_idle_notification.txt").write_text(
                        _notif + "\n", encoding="utf-8"
                    )
            _hb("captured-notified", saved=n)
            print("{}")
        else:
            from memo.cli_transcripts import _reflect_session
            from memo.memory import Memory

            mem = Memory(cfg)
            _reflect_session(str(sid), mem, cfg, debug=flag_bool("MEMO_SESSION_DEBUG"))
    except Exception as exc:
        if flag_bool("MEMO_SESSION_DEBUG"):
            print(f"# memo session idle-maintenance failed: {exc}", file=_sys.stderr)

    print("{}")
    _sys.exit(0)


@session_group.command(name="refresh-summary")
def session_refresh_summary() -> None:
    """Stop hook entrypoint — generate/update running_summary for the active session.

    Reads the Stop hook payload from stdin, runs the 3B LLM helper to
    summarize the session arc, and stores it in the snapshot. Throttled:
    skips if fewer than 3 new turns since the last summary. Always exits 0.
    """
    import json as _json
    import sys as _sys

    if flag_bool("MEMO_SESSION_DISABLE"):
        _sys.exit(0)

    payload: dict[str, Any] = {}
    if not _sys.stdin.isatty():
        try:
            raw = _sys.stdin.read()
            if raw.strip():
                payload = _json.loads(raw)
        except _json.JSONDecodeError:
            payload = {}

    sid = payload.get("session_id")
    if not sid:
        _sys.exit(0)

    try:
        from memo.session import refresh_summary as _refresh_summary

        cfg = Config.from_env()
        cfg.ensure_dirs()
        _refresh_summary(cfg.state_dir, sid)
    except Exception as exc:
        if flag_bool("MEMO_SESSION_DEBUG"):
            print(f"# memo session refresh-summary failed: {exc}", file=_sys.stderr)
    _sys.exit(0)


@session_group.command(name="recent")
@click.option("--limit", default=None, type=int, show_default=True)
def session_recent(limit: int | None) -> None:
    """SessionStart hook entrypoint — emit `additionalContext` markdown
    listing recent sessions. Same exit-0-silent contract as recall-hook."""
    import json as _json
    import os as _os
    import sys as _sys

    if flag_bool("MEMO_SESSION_DISABLE"):
        print("{}")
        _sys.exit(0)

    if limit is None:
        limit = flag_int("MEMO_SESSION_RECENT_LIMIT") or 12

    try:
        from memo.session import (
            _strip_command_wrappers,
            format_relative,
            is_command_noise,
            list_sessions,
            render_active_memory,
        )

        cfg = Config.from_env()
        rows = list_sessions(cfg.state_dir, limit=limit)
    except Exception as exc:
        if flag_bool("MEMO_SESSION_DEBUG"):
            print(f"# memo session recent failed: {exc}", file=_sys.stderr)
        print("{}")
        _sys.exit(0)

    if not rows:
        print("{}")
        _sys.exit(0)

    from pathlib import Path as _Path

    cur_cwd = str(_Path(_os.getcwd()).resolve())
    same_cwd = [r for r in rows if (r.get("cwd") or "") == cur_cwd]
    top = same_cwd[0] if same_cwd else None

    def _clean_summary(r: dict, width: int) -> str:
        """First non-noise candidate among stored summary / last prompt /
        running-summary — heals junk persisted by pre-fix checkpoints."""
        for cand in (r.get("summary"), r.get("last_user_msg")):
            if not is_command_noise(cand):
                return _strip_command_wrappers(cand or "").replace("\n", " ")[:width]
        rs = r.get("running_summary")
        if rs and not is_command_noise(rs):
            return rs.strip().replace("\n", " ")[:width]
        return "—"

    lines: list[str] = []

    if top:
        sid = top.get("session_id") or ""
        when = format_relative(top.get("updated"))
        branch = top.get("branch") or "—"
        turns = top.get("turn_count") or 0
        summary = _clean_summary(top, 120)
        lines.extend(
            [
                "## Previous session detected — continue?",
                "",
                f"There was an active session in this directory ({when}):",
                f"- **Summary**: {summary}",
                f"- **Branch**: `{branch}`  |  **Turns**: {turns}",
                f"- **To resume** (in a new terminal): `claude --resume {sid}`",
                "",
            ]
        )
        active_memory = render_active_memory(top)
        if active_memory:
            lines.extend(active_memory)
            lines.append("")
        lines.extend(
            [
                "> **Action**: When starting this conversation, ask the user whether they want to "
                "resume the previous session (by running the command above in the terminal) "
                "or continue with this new session.",
                "",
            ]
        )

    def _render_table(title: str, items: list[dict]) -> None:
        if not items:
            return
        lines.extend(
            [
                f"### {title}",
                "",
                "| when | project | branch | summary | id |",
                "|--------|----------|--------|---------|----|",
            ]
        )
        for r in items:
            s = _clean_summary(r, 55).replace("|", "·")
            lines.append(
                f"| {format_relative(r.get('updated'))} | "
                f"{(r.get('project') or '—')[:18]} | "
                f"{(r.get('branch') or '—')[:14]} | "
                f"{s} | "
                f"`{(r.get('session_id') or '')[:8]}` |"
            )
        lines.append("")

    remaining = [r for r in rows if r is not top]
    if remaining:
        cur_project = _Path(cur_cwd).name
        this_project = [
            r
            for r in remaining
            if (r.get("cwd") or "") == cur_cwd or (r.get("project") or "") == cur_project
        ]
        others = [r for r in remaining if r not in this_project][:8]
        _render_table("Recent sessions in this project", this_project)
        _render_table("Other projects", others)
        lines.append("_`memo resume <id>` to see details. `claude --resume <id>` to resume._")

    output = {
        "hookSpecificOutput": {
            "hookEventName": "SessionStart",
            "additionalContext": "\n".join(lines),
        }
    }
    print(_json.dumps(output, ensure_ascii=False))


@session_group.command(name="prune")
@click.option("--cap", default=50, type=int, show_default=True)
def session_prune(cap: int) -> None:
    """Delete oldest sessions beyond `cap`. Idempotent."""
    from memo.session import prune_lru

    cfg = Config.from_env()
    n = prune_lru(cfg.state_dir, cap=cap)
    console.print(f"[green]✓[/green] pruned {n} session(s); cap={cap}")
