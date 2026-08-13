"""`memo capture-stop` / `resume` — Stop-hook capture + session resume.

Extracted from cli.py (god-module decomposition); ingest/mine-history/reflect
moved out to cli_ingest.py / cli_transcripts.py. Registered onto the root group
in cli.py via `cli.add_command(...)`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import click
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from memo.cli_common import console
from memo.config import Config

if TYPE_CHECKING:
    from memo.resume import ResumeCandidate


def _write_capture_notification(state_dir: Path, saved: list[dict], *, idle: bool = False) -> None:
    """Write a pending notification the next recall-hook surfaces, so passive
    auto-capture is visible to the user and correctable.

    With ``MEMO_CAPTURE_RECEIPT`` on, writes a multi-line receipt — title/type/id
    per saved memory plus a `memo fix`/`memo undo` hint. With the flag off
    (default), writes the legacy muted one-liner (``※ MEMO auto-saved``).
    Best-effort; never raises."""
    if not saved:
        return
    from memo.flags import flag_bool

    try:
        state_dir.mkdir(parents=True, exist_ok=True)
        if flag_bool("MEMO_CAPTURE_RECEIPT"):
            lines = [f"☾ memo saved {len(saved)} this session:"]
            for s in saved:
                lines.append(
                    f'  • "{s.get("title", "?")}" ({s.get("type", "note")}) [{s.get("id", "?")}]'
                )
            lines.append("  fix: memo fix <id> · undo: memo undo <id>")
            body = "\n".join(lines) + "\n"
        else:
            body = "※ MEMO auto-saved\n"
        (state_dir / "pending_idle_notification.txt").write_text(body, encoding="utf-8")
    except OSError:
        pass


@click.command(name="capture-stop")
def capture_stop() -> None:
    """Stop hook — passive auto-extract of insights from the last turn.

    Reads the Stop hook payload from stdin (Claude Code format), pulls
    the last (user, assistant) exchange from the transcript, asks the
    configured helper LLM to extract any actionable insights, dedups
    against the existing corpus, and saves survivors via Memory.save().

    Hook input (stdin, JSON):
      {"transcript_path": "/path/to/...jsonl", ...}

    Hook output (stdout):
      `{}`  — always. Capture is silent; the user discovers new
      memories via `memo list` or the next ambient recall.

    Env vars:
      MEMO_CAPTURE_DISABLE  — set to "1" to make this a no-op.
      MEMO_CAPTURE_DEBUG    — set to "1" to print extraction progress
                              to stderr (helpful while tuning the
                              extraction prompt or trigger keywords).

    Failure modes are absorbed. The hook never blocks the user — at
    worst you get no auto-save for that turn.
    """
    import json as _json
    import sys as _sys

    from memo.flags import flag_bool

    if flag_bool("MEMO_CAPTURE_DISABLE"):
        print("{}")
        _sys.exit(0)

    debug = flag_bool("MEMO_CAPTURE_DEBUG")

    try:
        raw = _sys.stdin.read()
        payload = _json.loads(raw) if raw.strip() else {}
    except _json.JSONDecodeError:
        print("{}")
        _sys.exit(0)

    transcript_path = payload.get("transcript_path")
    if not transcript_path:
        # Some hook events omit transcript_path (seen 2026-06-27 onward) —
        # recover it from session_id before giving up, so capture/grounding/
        # the token ledger don't go dark for want of one field.
        from memo.session import find_transcript_path

        transcript_path = find_transcript_path(str(payload.get("session_id") or ""))
        if transcript_path:
            payload = {**payload, "transcript_path": transcript_path}
    if not transcript_path:
        print("{}")
        _sys.exit(0)

    try:
        from memo.capture import run_capture
        from memo.config import Config

        result = run_capture(Path(transcript_path), debug=debug)
        # Surface what was saved so auto-capture is VISIBLE: write a pending
        # notification the next recall-hook prepends to its context (the same
        # channel the user already sees). Without this, capture-stop — the path
        # that does most of the saving — is silent and the user can't tell it
        # ran. Only fires when memories were actually saved (not on dedup/cooldown).
        # `saved_records` (id+title+type per memory) is the enriched receipt
        # source; fall back to wrapping `saved_titles` when it's absent so an
        # older/mocked result dict still notifies.
        saved = result.get("saved_records") or [
            {"id": "?", "title": t, "type": "note"} for t in (result.get("saved_titles") or [])
        ]
        if saved:
            # Default-OFF parity: pre-branch, capture-stop only printed to the
            # console and never wrote the notification file. Gate the new file
            # write behind MEMO_CAPTURE_RECEIPT so the flag-off Stop path stays
            # byte-identical to before (console line only).
            if flag_bool("MEMO_CAPTURE_RECEIPT"):
                _write_capture_notification(Config.from_env().state_dir, saved)
            console.print("[dim]※ MEMO auto-saved[/dim]")
    except Exception as exc:
        if debug:
            print(f"# memo capture-stop failed: {exc}", file=_sys.stderr)

    # Grounding (P0): score how much the answer used this turn's recalled
    # memories → grounding.log (the outcome-based utility signal). Best-effort,
    # budget-guarded inside score_turn, never fails the turn.
    try:
        from memo import grounding
        from memo.config import Config

        summary = grounding.score_turn(Config.from_env().state_dir, payload)
        if debug and summary:
            print(f"# memo grounding: {summary}", file=_sys.stderr)
    except Exception as exc:
        if debug:
            print(f"# memo grounding failed: {exc}", file=_sys.stderr)

    # Next-turn verdict (opt-in): classify THIS user turn as a reaction to the
    # PRIOR turn's recalled memories → implicit source_feedback + verdict.log.
    # Stop hook only — never the 5s recall hook. Best-effort.
    try:
        from memo.flags import flag_bool as _flag_bool

        if _flag_bool("MEMO_VERDICT_ENABLED"):
            from memo import verdict
            from memo.config import Config

            v = verdict.record_verdicts(Config.from_env(), payload)
            if debug and v:
                print(f"# memo verdict: {v['verdict']} → {v['recall_ids']}", file=_sys.stderr)
    except Exception as exc:
        if debug:
            print(f"# memo verdict failed: {exc}", file=_sys.stderr)

    # Token-savings ledger: fold the (just-updated) grounded events into the
    # durable per-day file before grounding.log rotates them out, so `memo
    # tokens` keeps a monotonic all-time total. Trivial JSON I/O, never fails
    # the turn.
    try:
        from memo import token_ledger
        from memo.config import Config

        token_ledger.roll_up(Config.from_env().state_dir)
    except Exception as exc:
        if debug:
            print(f"# memo token-ledger rollup failed: {exc}", file=_sys.stderr)

    # Measured token meter: real per-turn usage from the transcript, joined with
    # injection cost + grounding. Stop-hook only (never the 5s recall hook).
    try:
        from memo.flags import flag_bool

        if flag_bool("MEMO_TOKEN_METER_ENABLED"):
            from memo import token_meter
            from memo.config import Config

            token_meter.roll(
                Config.from_env().state_dir,
                str(payload.get("session_id") or ""),
                payload.get("transcript_path"),
            )
    except Exception as exc:
        if debug:
            print(f"# memo token-meter rollup failed: {exc}", file=_sys.stderr)

    try:
        from memo import presence
        from memo import token_ledger as _tl
        from memo.config import Config

        _state = Config.from_env().state_dir
        _summary = _tl.summarize(_state, days_back=1, months_back=1)
        presence.set_tokens(_state, int(_summary["today"]["tokens"]))
    except Exception as exc:
        if debug:
            print(f"# memo presence set_tokens failed: {exc}", file=_sys.stderr)

    # Episodic memory: index this session's prompt-arc so `memo resume` can find
    # it by meaning. Content-hash-skip makes most Stops a no-op; never fails the hook.
    try:
        from memo.config import Config
        from memo.resume._index import index_memo_session

        sid = payload.get("session_id") or ""
        if sid:
            index_memo_session(Config.from_env(), sid, payload.get("transcript_path"))
    except Exception as exc:
        if debug:
            print(f"# memo episode index failed: {exc}", file=_sys.stderr)

    print("{}")
    _sys.exit(0)


@click.command(name="capture-tick")
@click.option(
    "--session-id",
    default=None,
    help="Override session_id (default: read from the hook stdin payload).",
)
@click.option(
    "--transcript-path",
    default=None,
    help="Override transcript path (default: read from the hook stdin payload).",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help=(
        "Bypass the per-session throttle (PreCompact force-flush at the "
        "compaction boundary). Also clears this session's emission ledger, "
        "since compaction invalidates every claim about what is in the window."
    ),
)
def capture_tick(session_id: str | None, transcript_path: str | None, force: bool) -> None:
    """UserPromptSubmit hook — incremental mid-session ambient capture.

    The Stop hook (`memo capture-stop`) only fires at session end, so a long
    or crashed session's durable insight never reaches `.md` (and thus the
    local index + git sync) until Stop. This mines the NEW turns since a
    per-session watermark into memories — reusing capture-stop's
    extract/dedup/save pipeline — so durable knowledge lands on disk
    mid-session, not only at the end.

    Hook input (stdin, JSON): Claude Code passes
    `{"session_id", "transcript_path", ...}` on UserPromptSubmit. Falls back
    to the flags for manual runs.

    Hook output (stdout): `{}` — always. Capture is silent.

    Self-throttled per session via ``MEMO_CAPTURE_INTERVAL_S`` (default 600s),
    measured off the watermark's ``updated`` stamp — so firing on every prompt
    is a cheap no-op when not due (a small JSON read, no transcript parse and
    no MLX). Bounded: each due pass processes only the turns added since the
    previous pass; the watermark guarantees old turns are never reprocessed.

    Intended hook wiring (UserPromptSubmit, async, soft-fail):

        MEMO_NONINTERACTIVE=1 memo capture-tick

    Env vars:
      MEMO_CAPTURE_DISABLE     — "1" → no-op (shared with capture-stop).
      MEMO_CAPTURE_INTERVAL_S  — min seconds between ticks per session
                                 (default 600; 0 = every prompt, no throttle).
      MEMO_CAPTURE_DEBUG       — "1" → print progress to stderr.
      PreCompact wiring passes --force so the last throttle window is flushed
      before context is destroyed. --force also resets this session's
      emission ledger (MEMO_EMITTED_LEDGER) — see the --force option's help.

    Failure modes are absorbed: the hook never blocks or breaks the session.
    """
    import json as _json
    import sys as _sys

    from memo.flags import flag_bool, flag_int

    debug = flag_bool("MEMO_CAPTURE_DEBUG")

    # Resolved at most once per invocation: the ledger-reset block below sets
    # it when it runs far enough to need `state_dir`, and the mining
    # try-block further down reuses it instead of re-parsing the on-disk
    # TOML/Markdown config a second time on the same --force pass.
    cfg: Config | None = None

    # PreCompact boundary. Claude Code's PreCompact hook runs exactly this —
    # `MEMO_NONINTERACTIVE=1 memo capture-tick --force` (hooks/hooks.json,
    # cli_hooks.wire_precompact_hook) — so `--force` doubles as the
    # compaction-boundary signal: once the window is rewritten, memo can no
    # longer claim any previously-emitted body is still in it, so the
    # emission ledger must not outlive compaction. Gated on `force` alone —
    # an ordinary throttled tick fires on every prompt and must never touch
    # the ledger, or it would silently erase the whole feature's savings on
    # every turn.
    #
    # Deliberately UNCONDITIONAL on MEMO_EMITTED_LEDGER — do not add that
    # gate back as an "optimisation". The flag is not static across a
    # session: dream_flags' flag-graduation/auto-revert machinery flips
    # default-off flags through the tuned overlay between nights. A flag
    # gate here would make this sequence reachable: entries accumulate while
    # the flag is ON -> compaction happens while the flag is OFF, so a gated
    # reset would skip and the stale file would survive -> the flag flips
    # back ON. Every reader (apply_ledger, recall_logic._log, the subprocess
    # hook) gates on the flag's value at READ time and none of them compares
    # an entry's age against the last compaction, so those stale entries
    # would resurface and `partition`'s monotonic rule would digest hits for
    # content the model can no longer see — the exact bug this task exists
    # to prevent. With the feature off there is no ledger file, so this is a
    # free `stat()` on a --force-only path either way.
    #
    # Runs first, ahead of MEMO_CAPTURE_DISABLE and the stdin/session
    # resolution below: the ledger reset is not a capture-mining concern, and
    # PreCompact must be able to reset it even with mining disabled or with
    # no session/transcript to mine this tick (both of those bail out below,
    # before ever reaching the mining try-block).
    #
    # Session id: `identity._session_id()` reads MEMO_SESSION_ID /
    # CLAUDE_SESSION_ID / CLAUDE_CODE_SESSION_ID from THIS process's own env
    # — the same resolution the subprocess recall hook (cli_recall_hook.py)
    # and the MCP tools (server_common.py's `_effective_session_id()`) use to
    # key their own ledger writes, per those modules' own comments, and
    # capture-tick is spawned the same way (a per-invocation subprocess of
    # the Claude Code session), so it inherits the same env and resolves the
    # same id. Deliberately NOT the payload's `session_id` field read below —
    # that only keys the unrelated capture watermark. If no env var is set,
    # reset nothing rather than guess: wiping the wrong session's ledger
    # would cost that session its savings for no reason.
    #
    # `emitted_ledger.reset` is itself idempotent (unlink-if-present) and
    # fail-open (never raises), but this whole block still gets its own
    # try/except: a reset failure must never skip the mining pass below it.
    if force:
        try:
            from memo.identity import _session_id as _identity_session_id

            ledger_sid = _identity_session_id()
            if ledger_sid:
                from memo import emitted_ledger as _el

                cfg = Config.from_env()
                _el.reset(cfg.state_dir, ledger_sid)
        except Exception as exc:
            if debug:
                print(f"# memo capture-tick: ledger reset failed: {exc}", file=_sys.stderr)

    if flag_bool("MEMO_CAPTURE_DISABLE"):
        print("{}")
        _sys.exit(0)

    payload: dict = {}
    # Stdin is a TTY when run interactively → don't block on a read.
    if not _sys.stdin.isatty():
        try:
            raw = _sys.stdin.read()
            if raw.strip():
                payload = _json.loads(raw)
        except _json.JSONDecodeError:
            payload = {}

    sid = session_id or payload.get("session_id")
    transcript = transcript_path or payload.get("transcript_path")
    if not sid or not transcript:
        # Without both we can't key the watermark or read the turns — no-op.
        print("{}")
        _sys.exit(0)

    try:
        from memo.capture import incremental_tick_due, run_capture_incremental

        if cfg is None:
            cfg = Config.from_env()
        interval_s = flag_int("MEMO_CAPTURE_INTERVAL_S")
        if interval_s is None:
            interval_s = 600

        if not force and not incremental_tick_due(cfg.state_dir, sid, interval_s):
            if debug:
                print("# memo capture-tick: throttled (not due)", file=_sys.stderr)
        else:
            out = run_capture_incremental(Path(transcript), sid, debug=debug)
            if debug:
                print(f"# memo capture-tick: {out}", file=_sys.stderr)
    except Exception as exc:
        if debug:
            print(f"# memo capture-tick failed: {exc}", file=_sys.stderr)

    print("{}")
    _sys.exit(0)


def _print_candidate_detail(candidate: ResumeCandidate) -> None:
    """Render one federated candidate without executing it (scripted / piped)."""
    from memo.resume._utils import _format_relative_time

    rel = _format_relative_time(candidate.updated_at) or "—"
    resume_cmd = " ".join(candidate.resume_command) or "(context resume)"
    console.print(
        Panel.fit(
            f"[bold]{candidate.summary or candidate.title or candidate.session_id}[/bold]\n"
            f"[dim]agent:[/dim]   {candidate.agent}\n"
            f"[dim]session:[/dim] {candidate.session_id}\n"
            f"[dim]cwd:[/dim]     {candidate.cwd or '—'}\n"
            f"[dim]updated:[/dim] {candidate.updated_at}  ({rel})\n"
            f"[dim]resume:[/dim]  [cyan]{resume_cmd}[/cyan]",
            title="session",
            border_style="cyan",
        )
    )


def _candidate_summary_text(candidate: ResumeCandidate, *, max_chars: int = 360) -> str:
    text = (candidate.summary or candidate.title or "").strip()
    if not text:
        text = f"{candidate.agent} session {candidate.session_id[:12]}"
    text = " ".join(text.split())
    if len(text) > max_chars:
        return text[: max_chars - 1].rstrip() + "…"
    return text


def _candidate_recent_prompts(candidate: ResumeCandidate) -> list[str]:
    raw = candidate.metadata.get("prompt_trail")
    if not isinstance(raw, list):
        return []
    prompts = [str(item).strip() for item in raw if isinstance(item, str) and item.strip()]
    return prompts[-2:]


def _print_candidate_resume_list(candidates: list[ResumeCandidate]) -> None:
    """Render resumable sessions as continuity notes, not a cramped id table."""
    from pathlib import Path as _Path

    from memo.resume._utils import _format_relative_time

    for index, c in enumerate(candidates, start=1):
        when = _format_relative_time(c.updated_at) or "—"
        status = f" · {c.status}" if c.status else ""
        resume_cmd = " ".join(c.resume_command) if c.resume_command else "(context resume)"
        cwd_label = str(_Path(c.cwd).name) if c.cwd else "—"
        console.print(
            f"[bold]{index}. {escape(_candidate_summary_text(c))}[/bold]\n"
            f"   [dim]{escape(when)} · {escape(c.agent)}{escape(status)} · "
            f"{escape(c.session_id[:12])} · {escape(cwd_label)}[/dim]\n"
            f"   [dim]cwd:[/dim] {escape(c.cwd or '—')}\n"
            f"   [dim]resume:[/dim] [cyan]{escape(resume_cmd)}[/cyan]"
        )
        prompts = _candidate_recent_prompts(c)
        if prompts:
            console.print("   [dim]recent prompts:[/dim]")
            for prompt in prompts:
                short = " ".join(prompt.split())
                if len(short) > 160:
                    short = short[:159].rstrip() + "…"
                console.print(f"   - {escape(short)}")
        if index != len(candidates):
            console.print()


def _resume_federated(
    *,
    session_id: str | None,
    agent: str,
    limit: int,
    cwd_filter: str | None,
    project: str | None,
    all_cwd: bool,
    as_json: bool,
) -> None:
    """Cross-agent federated resume owned by Memo.

    Discovers recent/active sessions across codex/claude/devin/gemini/opencode
    native stores (plus memo's own snapshots), merges by (agent, session_id).
    On a TTY it opens the interactive arrow-key picker (↑/↓ browse, type to
    search, Tab toggles cwd/all + sort, Enter resumes — execs the native
    `claude --resume`/`codex resume`/… and replaces this process). Piped or
    `--json`, it prints a static table / report / candidate instead.
    """
    import os
    from pathlib import Path as _Path

    from memo.resume import (
        discover_resume_candidates,
        execute_resume_candidate,
        pick_resume_candidate_interactive,
        resolve_resume_candidate,
    )
    from memo.resume._utils import _same_cwd

    cwd = cwd_filter or os.getcwd()
    interactive = not session_id and not as_json and sys.stdin.isatty() and sys.stdout.isatty()
    if interactive:
        # Load everything so the picker can page/filter across all sessions; the
        # cwd filter is applied live inside the TUI (Tab → Filter: Cwd/All).
        report = discover_resume_candidates(
            agent=agent, cwd=cwd, include_all_cwd=True, limit=max(limit, 100000)
        )
    else:
        report = discover_resume_candidates(
            agent=agent, cwd=cwd, include_all_cwd=all_cwd, limit=limit
        )

    candidates = report.candidates
    if project:
        candidates = [c for c in candidates if _Path(c.cwd or "").name == project]

    if as_json:
        if session_id:
            cand = resolve_resume_candidate(candidates, session_id)
            click.echo(json.dumps(cand.to_dict() if cand else None, ensure_ascii=False, indent=2))
            return
        payload = report.to_dict()
        payload["candidates"] = [c.to_dict() for c in candidates]
        click.echo(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    # Resume one candidate by id/prefix.
    if session_id:
        candidate = resolve_resume_candidate(candidates, session_id)
        if candidate is None:
            console.print(f"[red]resume candidate not found:[/red] {session_id}")
            sys.exit(1)
        if not sys.stdout.isatty():
            # Scripted / piped: show detail, never exec into another process.
            _print_candidate_detail(candidate)
            return
        console.print(
            f"[green]Resuming[/green] [bold]{candidate.agent}[/bold] "
            f"{candidate.session_id[:12]} · [dim]{candidate.cwd or '?'}[/dim]"
        )
        # Execs the native resume command (replaces this process) or prints
        # context-resume guidance when no native resume exists.
        sys.exit(execute_resume_candidate(candidate))

    notice = ""
    if report.provider_errors:
        joined = " · ".join(f"{e.provider}: {e.detail}" for e in report.provider_errors)
        notice = f"⚠ {joined}"[:200]

    if not candidates:
        console.print("[dim]no sessions found[/dim]")
        if notice:
            console.print(f"[dim]{notice}[/dim]")
        return

    # Interactive TTY: the selectable arrow-key session picker.
    if interactive:
        # Default to the cwd filter, but if nothing matches the current dir fall
        # back to All so the picker never opens empty while sessions exist.
        start_filter = "all" if all_cwd else "cwd"
        if start_filter == "cwd" and not any(_same_cwd(c.cwd, cwd) for c in candidates):
            start_filter = "all"
        # Episodic memory: type-to-search re-ranks by meaning over the full
        # history (self-degrades to substring when the embedder is cold / disabled).
        from memo.config import Config
        from memo.resume._index import semantic_search
        from memo.resume._preview import session_preview

        cfg = Config.from_env()

        def _semantic(q: str) -> list:
            return semantic_search(cfg, q)

        def _preview(c: ResumeCandidate) -> list[str]:
            return session_preview(cfg, c)

        candidate = pick_resume_candidate_interactive(
            candidates,
            current_cwd=cwd,
            start_filter=start_filter,
            notice=notice,
            semantic_fn=_semantic,
            preview_fn=_preview,
        )
        if candidate is None:
            return
        sys.exit(execute_resume_candidate(candidate))

    # Non-TTY (piped / captured): prioritize continuity over a cramped table.
    _print_candidate_resume_list(candidates)
    console.print(
        "[dim]Resume: `memo resume <session> --agent all` (or copy a `resume` command above).[/dim]"
    )
    if notice:
        console.print(f"[dim]{notice}[/dim]")


@click.command(name="resume")
@click.argument("session_id", required=False)
@click.option(
    "--limit",
    default=10,
    type=int,
    show_default=True,
    help="Max sessions to show (only used when SESSION_ID is omitted).",
)
@click.option("--project", default=None, help="Filter to one project basename.")
@click.option(
    "--cwd",
    "cwd_filter",
    default=None,
    help="Filter to sessions for this exact cwd (resolved). "
    "Used by the shell wrapper to ask 'what was open here?' "
    "without manual path comparison.",
)
@click.option(
    "--agent",
    type=click.Choice(["memo", "all", "claude", "codex", "devin", "gemini", "opencode"]),
    default=None,
    help="`all` (or a specific agent) runs the cross-agent federated picker — "
    "scanning codex/claude/devin/gemini/opencode "
    "native stores. `memo` forces memo's own snapshots only. Unset: the human "
    "picker federates (`all`); `--json` stays memo-only for the shell wrapper.",
)
@click.option(
    "--all-cwd",
    "all_cwd",
    is_flag=True,
    help="Federated mode only: do not filter candidates to the current cwd.",
)
@click.option("--json", "as_json", is_flag=True)
def resume(
    session_id: str | None,
    limit: int,
    project: str | None,
    cwd_filter: str | None,
    agent: str | None,
    all_cwd: bool,
    as_json: bool,
) -> None:
    """Recent sessions to resume — picker for the SessionStart flow.

    Bare `memo resume` federates across agents:
    it discovers and natively resumes codex/claude/devin/gemini/opencode
    sessions for the current cwd, with memo's own snapshots merged in as a
    first-class provider. `--all-cwd` widens beyond the current directory.

    `--agent memo` forces memo's own sidecar snapshots under
    `~/.local/share/memo/sessions/` (auto-written by the Stop hook, LRU-capped),
    shown as the rich project/branch/turns table — pass SESSION_ID (full or
    unique prefix ≥4 chars) to inspect one. The machine `--json` path defaults
    to this memo-only list so shell wrappers keep their stable contract.
    """
    # `--agent` unset: the human picker federates (what you want when you type
    # `memo resume`); the `--json` path stays memo-only so the SessionStart shell
    # wrapper and other programmatic clients keep their list contract.
    if agent is None:
        agent = "memo" if as_json else "all"
    if agent != "memo":
        _resume_federated(
            session_id=session_id,
            agent=agent,
            limit=limit,
            cwd_filter=cwd_filter,
            project=project,
            all_cwd=all_cwd,
            as_json=as_json,
        )
        return

    from memo.session import format_relative, get_session, list_sessions

    cfg = Config.from_env()

    # Detail view — one session.
    if session_id:
        snap = get_session(cfg.state_dir, session_id)
        if snap is None:
            console.print(f"[red]not found:[/red] {session_id}")
            sys.exit(1)
        if as_json:
            click.echo(json.dumps(snap, ensure_ascii=False, indent=2))
            return
        mods = snap.get("modified_files") or []
        mods_line = ", ".join(mods[:5])
        if len(mods) > 5:
            mods_line += f", …(+{len(mods) - 5})"
        sid = snap.get("session_id") or ""
        console.print(
            Panel.fit(
                f"[bold]{snap.get('summary') or snap.get('last_user_msg') or 'session'}[/bold]\n"
                f"[dim]session_id:[/dim] {sid}\n"
                f"[dim]project:[/dim]    {snap.get('project') or '—'}\n"
                f"[dim]cwd:[/dim]        {snap.get('cwd') or '—'}\n"
                f"[dim]branch:[/dim]     {snap.get('branch') or '—'}\n"
                f"[dim]head:[/dim]       {snap.get('head_commit') or '—'}\n"
                f"[dim]modified:[/dim]   {mods_line or '—'}\n"
                f"[dim]transcript:[/dim] {snap.get('transcript_path') or '—'}\n"
                f"[dim]created:[/dim]    {snap.get('created')}  ({format_relative(snap.get('created'))})\n"
                f"[dim]updated:[/dim]    {snap.get('updated')}  ({format_relative(snap.get('updated'))})\n"
                f"[dim]turns:[/dim]      {snap.get('turn_count')}\n\n"
                f"{snap.get('last_user_msg') or ''}",
                title="session",
                border_style="cyan",
            )
        )
        if sid:
            console.print(
                f"\n[bold green]To resume:[/bold green]  "
                f"[cyan]claude --resume {sid}[/cyan]\n"
                f"[dim](copy-paste; run the command from "
                f"`{snap.get('cwd') or '?'}`)[/dim]",
            )
        return

    # List view — picker.
    rows = list_sessions(
        cfg.state_dir,
        limit=limit,
        project=project,
        cwd=cwd_filter,
    )
    if as_json:
        click.echo(json.dumps(rows, ensure_ascii=False, indent=2))
        return
    if not rows:
        console.print("[dim]no sessions yet — run a checkpoint first[/dim]")
        return

    # When the caller passed an explicit --cwd, the list is already
    # filtered to that cwd — printing a "Latest in this project"
    # banner on top of a homogeneous list would be redundant.
    if cwd_filter:
        same_cwd = []
    else:
        # Bias: if there's a session for the current cwd, surface it on top
        # with the exact resume command. The whole point of the picker is
        # crash recovery — if you crashed and reopened terminal in the same
        # project, the very first thing you want to see is "click here to
        # resume", not a generic chronological list.
        import os as _os
        from pathlib import Path as _Path

        cur_cwd = str(_Path(_os.getcwd()).resolve())
        same_cwd = [r for r in rows if (r.get("cwd") or "") == cur_cwd]
    if same_cwd:
        top = same_cwd[0]
        sid = top.get("session_id") or ""
        console.print(
            f"[bold green]Latest in this project[/bold green]  "
            f"[dim]({format_relative(top.get('updated'))})[/dim]: "
            f"{(top.get('summary') or top.get('last_user_msg') or '—')[:80]}",
        )
        console.print(
            f"[bold green]To resume:[/bold green]  [cyan]claude --resume {sid}[/cyan]\n",
        )

    tbl = Table(show_lines=False, expand=True)
    tbl.add_column("when", width=10)
    tbl.add_column("project", width=14, overflow="fold")
    tbl.add_column("branch", width=14, overflow="fold")
    tbl.add_column("turns", justify="right", width=5)
    tbl.add_column("summary", overflow="fold")
    tbl.add_column("session_id", overflow="fold")
    for r in rows:
        tbl.add_row(
            format_relative(r.get("updated")),
            r.get("project") or "—",
            r.get("branch") or "—",
            str(r.get("turn_count") or 0),
            (r.get("summary") or r.get("last_user_msg") or "—")[:80],
            r.get("session_id") or "—",
        )
    console.print(tbl)
    console.print(
        "[dim]Detail: `memo resume <id|prefix>`  ·  "
        "Resume: `claude --resume <session_id>` (copy from the table).[/dim]",
    )


@click.group(name="episodes")
def episodes_group() -> None:
    """Episodic memory — the semantic index behind `memo resume` search.

    One embedding per work session (its prompt-arc) so the picker finds a
    session by *meaning* across the full history, not just recency. The index is
    derived from transcripts and rebuildable; it never enters the recall hook.
    """


@episodes_group.command(name="index")
@click.option(
    "--agent",
    type=click.Choice(["all", "claude", "codex", "devin", "gemini", "opencode"]),
    default="all",
    show_default=True,
    help="Limit the backfill to one agent's sessions.",
)
@click.option("--rebuild", is_flag=True, help="Drop and re-embed the whole index.")
@click.option("--json", "as_json", is_flag=True)
def episodes_index(agent: str, rebuild: bool, as_json: bool) -> None:
    """Backfill the episode index (newest-first, bounded per run by
    MEMO_RESUME_INDEX_BATCH). Run by `memo-nightly`; safe to run manually."""
    from memo.config import Config
    from memo.resume._index import backfill

    result = backfill(Config.from_env(), agent=agent, rebuild=rebuild)
    if as_json:
        click.echo(json.dumps(result))
        return
    if not result.get("enabled"):
        console.print("[dim]episodic memory disabled (MEMO_EPISODIC_ENABLED=0)[/dim]")
        return
    console.print(
        f"[green]episodes indexed[/green] {result['indexed']} · "
        f"skipped {result['skipped']} · total in index {result['total']}"
    )


@episodes_group.command(name="search")
@click.argument("query", nargs=-1, required=True)
@click.option("--limit", default=10, type=int, show_default=True, help="Max results.")
@click.option("--json", "as_json", is_flag=True)
def episodes_search(query: tuple[str, ...], limit: int, as_json: bool) -> None:
    """Find past sessions by MEANING — `memo episodes search vec0 timeout bug`.

    Queries the episodic index (cold-loads MLX if the recall daemon is down).
    Each hit is a resumable session: copy its `resume` command to continue it.
    """
    from memo.config import Config
    from memo.resume._index import semantic_search

    q = " ".join(query).strip()
    hits = semantic_search(Config.from_env(), q, k=limit, allow_cold=True)
    if as_json:
        click.echo(json.dumps([h.to_dict() for h in hits], ensure_ascii=False, indent=2))
        return
    if not hits:
        console.print(
            "[dim]no episode matches — run `memo episodes index` to populate the index, "
            "or check the recall daemon[/dim]"
        )
        return
    tbl = Table(show_lines=False, expand=True)
    tbl.add_column("score", width=6, justify="right")
    tbl.add_column("agent", width=9)
    tbl.add_column("summary", overflow="fold")
    tbl.add_column("resume", overflow="fold")
    for h in hits:
        score = h.metadata.get("score")
        resume_cmd = " ".join(h.resume_command) if h.resume_command else "(context)"
        tbl.add_row(
            f"{float(score):.3f}" if isinstance(score, (int, float)) else "—",
            h.agent,
            (h.summary or h.title or "—")[:120],
            resume_cmd,
        )
    console.print(tbl)
