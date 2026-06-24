"""`memo mine-history` / `reflect` — transcript mining + session reflection.

Extracted from cli_capture.py (god-module decomposition). Registered onto the
root group in cli.py. Carries the reflect helpers (`_read_full_transcript`,
`_build_reflect_prompt`, `_reflect_session`) used only by these commands.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import click
from rich.panel import Panel

from memo.cli_common import console
from memo.config import Config


@click.command(name="mine-history")
@click.option(
    "--path", "root_path", default=None, help="Transcripts root (default: ~/.claude/projects)."
)
@click.option(
    "--since",
    "since_days",
    type=int,
    default=None,
    help="Only process transcripts modified in the last N days.",
)
@click.option(
    "--limit",
    "file_limit",
    type=int,
    default=None,
    help="Cap on number of transcripts to process (newest first).",
)
@click.option(
    "--dry-run", is_flag=True, help="Walk + extract, don't save. Useful for cost estimation."
)
@click.option("--debug", is_flag=True, help="Print per-file/per-candidate info to stderr.")
@click.option("--json", "as_json", is_flag=True, help="Emit JSON summary instead of a panel.")
def mine_history(
    root_path: str | None,
    since_days: int | None,
    file_limit: int | None,
    dry_run: bool,
    debug: bool,
    as_json: bool,
) -> None:
    """Mine past Claude Code conversations for actionable insights.

    Walks `~/.claude/projects/<hash>/*.jsonl`, runs the same prefilter +
    helper-LLM extraction + embedding-based dedup as the live capture
    hook, and saves what's new. Resumable: per-file processed-line
    counts are tracked under `~/.local/share/memo/mine-history.json`.

    Tips:
        - First run on a long history is slow (helper LLM is the bottleneck).
          Use `--limit 10 --since 30` to start with the freshest sessions.
        - `--dry-run` reports candidate counts without writing.
    """
    from pathlib import Path as _Path

    from memo.transcript_miner import mine_transcripts

    root = _Path(root_path).expanduser() if root_path else None

    console_progress = None
    if not as_json:
        from rich.progress import (
            BarColumn,
            Progress,
            SpinnerColumn,
            TextColumn,
            TimeElapsedColumn,
        )

        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        )
        progress.start()
        task = progress.add_task("mining transcripts", total=None)

        def cb(idx: int, total: int, p: _Path) -> None:
            progress.update(
                task,
                total=total,
                completed=idx,
                description=f"[{idx + 1}/{total}] {p.name}",
            )

        console_progress = (progress, task, cb)

    try:
        summary = mine_transcripts(
            root=root,
            since_days=since_days,
            file_limit=file_limit,
            dry_run=dry_run,
            debug=debug,
            progress_cb=console_progress[2] if console_progress else None,
        )
    finally:
        if console_progress:
            console_progress[0].stop()

    if as_json:
        click.echo(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    status = summary.get("status")
    if status == "no_files":
        console.print(f"[yellow]No transcripts found under {summary['root']}.[/yellow]")
        return

    saved = summary.get("saved", [])
    body = (
        f"[dim]root:[/dim] {summary['root']}\n"
        f"[dim]files:[/dim] {summary['files_processed']}/{summary['files_total']} processed"
        f" ([dim]{summary['files_skipped']} skipped — already mined[/dim])\n"
        f"[dim]candidates:[/dim] {summary['candidates']}\n"
        f"[bold green]saved:[/bold green] {len(saved)}"
        f"{' [yellow](dry-run)[/yellow]' if summary['dry_run'] else ''}\n"
        f"[dim]skipped duplicates:[/dim] {summary['skipped_dup']}"
    )
    console.print(Panel.fit(body, title="✓ mine-history", border_style="green"))


_REFLECT_TRANSCRIPT_WORD_BUDGET = 8000


def _read_full_transcript(transcript_path: Path) -> list[tuple[str, str]]:
    """Return all (role, text) pairs from a JSONL transcript. role ∈ {"user", "assistant"}.
    Skips tool-use blocks, system lines, and empty turns."""
    if not transcript_path.is_file():
        return []
    try:
        raw = transcript_path.read_text(encoding="utf-8")
    except OSError:
        return []

    exchanges: list[tuple[str, str]] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        role = obj.get("type") or obj.get("role")
        if role not in ("user", "assistant"):
            continue
        msg = obj.get("message", obj)
        content = msg.get("content") if isinstance(msg, dict) else None
        if content is None:
            continue
        if isinstance(content, str):
            text = content.strip()
        elif isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text":
                    t = (block.get("text") or "").strip()
                    if t:
                        parts.append(t)
            text = "\n\n".join(parts).strip()
        else:
            text = ""
        if text:
            exchanges.append((role, text))
    return exchanges


def _build_reflect_prompt(
    exchanges: list[tuple[str, str]],
    *,
    cwd: str | None = None,
    branch: str | None = None,
    running_summary: str | None = None,
) -> str:
    """Build the transcript block for the reflect LLM call.
    Word-budgeted to ~8k words: keep last N exchanges that fit.
    """
    # Build context header.
    ctx_parts: list[str] = []
    if cwd:
        ctx_parts.append(f"cwd: {cwd}")
    if branch:
        ctx_parts.append(f"branch: {branch}")
    if running_summary:
        ctx_parts.append(f"session summary: {running_summary}")
    header = " | ".join(ctx_parts)

    # Word-budget the transcript (most recent exchanges preferred).
    budget = _REFLECT_TRANSCRIPT_WORD_BUDGET
    blocks: list[str] = []
    for role, text in reversed(exchanges):
        label = "User" if role == "user" else "Assistant"
        # Truncate per-turn to avoid a single monster turn eating the budget.
        snippet = text[:2000]
        block = f"[{label}] {snippet}"
        words = len(block.split())
        if words > budget:
            break
        blocks.append(block)
        budget -= words
    blocks.reverse()

    transcript = "\n\n".join(blocks)
    if header:
        return f"Context: {header}\n\nTranscript:\n{transcript}"
    return f"Transcript:\n{transcript}"


def _reflect_session(
    session_id: str,
    mem: Any,
    cfg: Any,
    *,
    dry_run: bool = False,
    debug: bool = False,
) -> dict[str, Any]:
    """Core reflect logic. Returns a result dict.

    Loads the session snapshot, reads the full transcript, calls the LLM,
    saves memories, stamps `reflected_at`. All heavy imports are deferred.
    """
    from memo.session import get_session, mark_reflected

    snap = get_session(cfg.state_dir, session_id)
    if snap is None:
        return {"status": "not_found", "session_id": session_id}

    transcript_path_str = snap.get("transcript_path")
    if not transcript_path_str:
        return {"status": "no_transcript", "session_id": session_id}

    transcript_path = Path(transcript_path_str).expanduser()
    exchanges = _read_full_transcript(transcript_path)

    user_turns = [e for e in exchanges if e[0] == "user"]
    if len(user_turns) < 3:
        return {"status": "too_short", "session_id": session_id, "user_turns": len(user_turns)}

    prompt = _build_reflect_prompt(
        exchanges,
        cwd=snap.get("cwd"),
        branch=snap.get("branch"),
        running_summary=snap.get("running_summary"),
    )

    # LLM call — use the configured llm_model (7B default).
    try:
        from memo.memory.record import _REFLECT_SYSTEM_PROMPT, strip_llm_output

        result = mem._ensure_chat().chat(
            cfg.llm_model,
            [
                {"role": "system", "content": _REFLECT_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            options={"temperature": 0.0, "num_predict": 1024},
        )
        raw_json = (result.get("message") or {}).get("content") or ""
    except Exception as exc:
        if debug:
            print(f"# memo reflect: LLM failed: {exc}", file=sys.stderr)
        return {"status": "llm_error", "session_id": session_id, "error": str(exc)}

    # Strip <think> traces + markdown fences (shared helper).
    raw_json = strip_llm_output(raw_json)
    try:
        parsed = json.loads(raw_json)
    except json.JSONDecodeError:
        if debug:
            print(f"# memo reflect: JSON parse failed: {raw_json[:200]}", file=sys.stderr)
        parsed = {}

    session_title = (parsed.get("session_title") or "").strip()
    arc_summary = (parsed.get("summary") or "").strip()

    # Gather all items across categories.
    from memo.capture import is_near_duplicate

    saved_ids: list[str] = []
    skipped_dup = 0

    category_type_map = {
        "decisions": "decision",
        "facts": "fact",
        "bugs": "bug",
        "followups": "note",
    }

    for cat_key, mem_type in category_type_map.items():
        items = parsed.get(cat_key) or []
        for item in items:
            if not isinstance(item, dict):
                continue
            title = (item.get("title") or "").strip()[:80]
            body = (item.get("body") or "").strip()[:300]
            tags = [str(t).lower().strip() for t in (item.get("tags") or []) if t]
            if not title or not body:
                continue

            cand = {"title": title, "body": body}
            if not dry_run and is_near_duplicate(mem, cand):
                skipped_dup += 1
                if debug:
                    print(f"# memo reflect: skip dup '{title}'", file=sys.stderr)
                continue

            if not dry_run:
                try:
                    rec = mem.save(content=body, title=title, type_=mem_type, tags=tags)
                    saved_ids.append(rec.id)
                    if debug:
                        print(f"# memo reflect: saved [{rec.id[:8]}] {rec.title}", file=sys.stderr)
                except Exception as exc:
                    if debug:
                        print(f"# memo reflect: save failed: {exc}", file=sys.stderr)
            else:
                saved_ids.append(f"(dry-run) {title}")

    # Arc note — a single note linking the session narrative.
    arc_id: str | None = None
    if arc_summary and not dry_run:
        project = snap.get("project") or "unknown"
        branch_str = snap.get("branch") or ""
        id_refs = " ".join(f"[{i[:8]}]" for i in saved_ids if not i.startswith("("))
        arc_body = f"{arc_summary}"
        if id_refs:
            arc_body += f"\n\nInsights: {id_refs}"
        arc_title = session_title or f"{project} session"
        try:
            arc_tags = ["session-arc", f"project:{project}"]
            if branch_str:
                arc_tags.append(f"branch:{branch_str}")
            arc_rec = mem.save(content=arc_body, title=arc_title, type_="note", tags=arc_tags)
            arc_id = arc_rec.id
            if debug:
                print(f"# memo reflect: arc note [{arc_id[:8]}] {arc_title}", file=sys.stderr)
        except Exception as exc:
            if debug:
                print(f"# memo reflect: arc save failed: {exc}", file=sys.stderr)

    if not dry_run:
        mark_reflected(cfg.state_dir, session_id)

    return {
        "status": "ok",
        "session_id": session_id,
        "session_title": session_title,
        "saved": saved_ids,
        "skipped_dup": skipped_dup,
        "arc_id": arc_id,
        "dry_run": dry_run,
    }


@click.command(name="reflect")
@click.argument("session_id", required=False)
@click.option(
    "--last",
    is_flag=True,
    default=False,
    help="Reflect on the most recent completed session (default if no SESSION_ID).",
)
@click.option(
    "--if-due",
    is_flag=True,
    default=False,
    help="Skip if the session was already reflected (idempotent).",
)
@click.option("--quiet", is_flag=True, default=False, help="Output JSON only (for hook use).")
@click.option(
    "--dry-run", is_flag=True, default=False, help="Show what would be saved without saving."
)
@click.option("--debug", is_flag=True, default=False, help="Print extraction progress to stderr.")
def reflect(
    session_id: str | None,
    last: bool,
    if_due: bool,
    quiet: bool,
    dry_run: bool,
    debug: bool,
) -> None:
    """Synthesize a session transcript into durable memories.

    Reads the full session transcript (not just the last 3 turns),
    extracts decisions/facts/bugs/follow-ups, deduplicates against the
    existing corpus, and saves survivors plus a session arc note.

    Idempotent: a `reflected_at` stamp prevents re-processing the same
    session. Pass `--if-due` to skip cleanly when already reflected.

    Examples:

      memo reflect --last            # reflect on the most recent session
      memo reflect --last --if-due   # no-op if already reflected (hook use)
      memo reflect <id>              # reflect on a specific session
      memo reflect --last --dry-run  # preview without saving
    """
    from memo.flags import flag_bool
    from memo.session import get_session, list_sessions

    if flag_bool("MEMO_CAPTURE_DISABLE"):
        if quiet:
            click.echo(json.dumps({"status": "disabled"}))
        return

    cfg = Config.from_env()

    # Resolve which session to reflect on.
    target_id: str | None = session_id
    if not target_id:
        sessions = list_sessions(cfg.state_dir, limit=2)
        if not sessions:
            result = {"status": "no_sessions"}
            if quiet:
                click.echo(json.dumps(result))
            else:
                console.print("[yellow]No sessions found.[/yellow]")
            return
        # `--last` or no arg: use the most recent session.
        # If session is still "active" (no reflected_at, recent), use it anyway.
        target_id = sessions[0].get("session_id") or ""

    if not target_id:
        result = {"status": "no_session_id"}
        click.echo(json.dumps(result) if quiet else "")
        return

    # Idempotence guard.
    if if_due:
        snap = get_session(cfg.state_dir, target_id)
        if snap and snap.get("reflected_at"):
            result = {
                "status": "already_reflected",
                "session_id": target_id,
                "reflected_at": snap["reflected_at"],
            }
            if quiet:
                click.echo(json.dumps(result))
            else:
                console.print(f"[dim]already reflected: {target_id[:8]}[/dim]")
            return

    # Load Memory with LLM warmed.
    from memo.memory import Memory

    # `_reflect_session` lazily warms the chat model via `mem._ensure_chat()`.
    mem = Memory(cfg)
    result = _reflect_session(target_id, mem, cfg, dry_run=dry_run, debug=debug)

    if quiet:
        click.echo(json.dumps(result, ensure_ascii=False))
        return

    status = result.get("status")
    if status == "not_found":
        console.print(f"[red]session not found:[/red] {target_id}")
        sys.exit(1)
    if status == "no_transcript":
        console.print(f"[yellow]no transcript for session:[/yellow] {target_id[:8]}")
        return
    if status == "too_short":
        console.print(
            f"[dim]session too short ({result.get('user_turns')} user turns) — skipping[/dim]",
        )
        return
    if status == "llm_error":
        console.print(f"[red]LLM error:[/red] {result.get('error')}")
        sys.exit(1)
    if status == "already_reflected":
        console.print(f"[dim]already reflected: {target_id[:8]}[/dim]")
        return

    saved = list(result.get("saved") or [])
    skipped = result.get("skipped_dup") or 0
    arc_id = result.get("arc_id")
    dry_label = " [yellow](dry-run)[/yellow]" if dry_run else ""
    title = result.get("session_title") or target_id[:8]

    body = (
        f"[dim]session:[/dim] {target_id[:8]}\n"
        f"[dim]title:[/dim]   {title}\n"
        f"[bold green]saved:[/bold green]   {len(saved)}{dry_label}\n"
        f"[dim]dup skip:[/dim] {skipped}\n"
        f"[dim]arc:[/dim]     {arc_id[:8] if arc_id else '—'}"
    )
    console.print(Panel.fit(body, title="✓ reflect", border_style="green"))
