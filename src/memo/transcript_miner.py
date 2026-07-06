"""Historical transcript miner — backfill memories from past Claude Code
conversations.

`capture.py` only fires on the current Stop hook so it sees the active
turn. Months of historical transcripts live in `~/.claude/projects/<hash>/*.jsonl`
and never become memories. This module walks them, runs the same
prefilter → extract → dedup pipeline as `capture.run_capture`, and saves
survivors.

State file `~/.local/share/memo/mine-history.json` records how many
lines of each transcript have already been processed so re-running is
incremental.
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from memo.capture import (
    _extract_text,
    _hash_assistant,
    _passes_prefilter,
    extract_insights,
    is_near_duplicate,
)


def _state_file(state_dir: Path, name: str = "mine-history.json") -> Path:
    return state_dir / name


def _load_state(state_dir: Path, name: str = "mine-history.json") -> dict[str, Any]:
    f = _state_file(state_dir, name)
    if not f.is_file():
        return {}
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_state(state_dir: Path, state: dict[str, Any], name: str = "mine-history.json") -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    _state_file(state_dir, name).write_text(json.dumps(state), encoding="utf-8")


def find_transcripts(
    root: Path,
    *,
    since_days: float | None = None,
) -> list[Path]:
    """Return all `.jsonl` files under `root`, newest first.

    `since_days` filters to files modified in the last N days.
    """
    if not root.exists():
        return []
    files = list(root.rglob("*.jsonl"))
    if since_days is not None and since_days > 0:
        cutoff = time.time() - (since_days * 86400)
        files = [f for f in files if f.stat().st_mtime >= cutoff]
    files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    return files


def iter_exchanges(transcript_path: Path, text: str | None = None) -> Iterator[tuple[str, str]]:
    """Yield (user_text, assistant_text) pairs from a transcript.

    Walks forward. When a user msg is followed by one or more assistant
    msgs (possibly interleaved with tool_use/tool_result blocks), all
    assistant text is concatenated into a single "response" for that
    user turn.

    Pass ``text`` to avoid re-opening the file when the caller already
    has its content in memory.
    """
    if text is None:
        if not transcript_path.is_file():
            return
        try:
            text = transcript_path.read_text(encoding="utf-8")
        except Exception:
            return
    lines = text.splitlines()

    pending_user: str | None = None
    pending_assist: list[str] = []
    for line in lines:
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
        text = _extract_text(content)
        if not text:
            continue
        if role == "user":
            # New user turn — flush any pending assistant response tied
            # to the previous user msg.
            if pending_user is not None and pending_assist:
                yield (pending_user, "\n\n".join(pending_assist))
            pending_user = text
            pending_assist = []
        else:  # assistant
            if pending_user is None:
                # Assistant message with no prior user msg in this file —
                # rare (system-initiated session?). Skip.
                continue
            pending_assist.append(text)
    # Tail flush
    if pending_user is not None and pending_assist:
        yield (pending_user, "\n\n".join(pending_assist))


def mine_exchange_stream(
    mem: Any,
    chat: Any,
    cfg: Any,
    exchanges: Iterator[tuple[str, str]],
    *,
    turn_hashes: set[str],
    dry_run: bool = False,
    debug: bool = False,
    source_name: str = "",
    extra_fn: Callable[[str, str, str], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run the capture pipeline (prefilter → extract → dedup → save) over an
    iterator of (user_text, assistant_text) exchange pairs.

    Shared by `mine_transcripts` (Claude Code JSONL) and the cold-start
    importers in `history_importers.py` (Codex, opencode, ChatGPT/Claude.ai
    exports). `turn_hashes` is caller-owned so identical assistant turns
    dedup across files within one run. `extra_fn(user, assistant, turn_hash)`
    optionally builds the per-save `extra` provenance bag (Claude Code mining
    stamps session/transcript/turn_hash; importers pass none).
    """
    candidates = 0
    saved: list[str] = []
    dup = 0
    for user_text, assist_text in exchanges:
        if not _passes_prefilter(assist_text):
            continue
        h = _hash_assistant(assist_text)
        if h in turn_hashes:
            continue
        turn_hashes.add(h)

        insights = extract_insights(chat, cfg.helper_model, user_text, assist_text)
        candidates += len(insights)
        if debug and insights:
            print(
                f"# mine: {source_name or 'exchange'} → {len(insights)} candidate(s)",
                file=sys.stderr,
            )
        if extra_fn is not None:
            extra = extra_fn(user_text, assist_text, h)
        elif source_name:
            extra = {"source": f"imported:{source_name}"}
        else:
            extra = None
        for cand in insights:
            if is_near_duplicate(mem, cand):
                dup += 1
                continue
            if dry_run:
                saved.append("<dry-run>")
                continue
            try:
                rec = mem.save(
                    content=cand["body"],
                    title=cand["title"],
                    type_=cand["type"],
                    tags=cand["tags"],
                    auto_project=False,  # historical: project context unreliable
                    extra=extra,
                )
                saved.append(rec.id)
            except Exception as exc:
                if debug:
                    print(f"# mine: save failed: {exc}", file=sys.stderr)
    return {"candidates": candidates, "saved": saved, "skipped_dup": dup}


def _transcript_extra_fn(f: Path) -> Callable[[str, str, str], dict[str, Any]]:
    """Provenance-stamp builder for Claude Code transcript mining."""

    def _build(_user: str, _assist: str, turn_hash: str) -> dict[str, Any]:
        return {
            "session_id": f.stem,
            "transcript_path": str(f),
            "turn_hash": turn_hash,
        }

    return _build


def mine_transcripts(
    root: Path | None = None,
    *,
    since_days: float | None = None,
    file_limit: int | None = None,
    dry_run: bool = False,
    debug: bool = False,
    progress_cb: Any = None,
) -> dict[str, Any]:
    """Walk transcripts, extract insights, dedup, save.

    `root` defaults to `~/.claude/projects/`. `progress_cb` is an
    optional callable invoked as `(file_idx, total, path)` per file —
    used by the CLI to drive a Rich progress bar.

    Returns a result summary: counts of candidates, saves, dedups,
    skipped files (already at last-processed line).
    """
    from memo.config import Config
    from memo.memory import Memory

    cfg = Config.from_env()
    state = _load_state(cfg.state_dir)
    root = root or Path.home() / ".claude" / "projects"

    files = find_transcripts(root, since_days=since_days)
    if file_limit is not None and file_limit > 0:
        files = files[:file_limit]

    if not files:
        return {"status": "no_files", "root": str(root), "files": 0}

    mem = Memory(cfg)
    chat = mem._ensure_chat()

    total_candidates = 0
    total_saved: list[str] = []
    total_dup = 0
    files_processed = 0
    files_skipped = 0
    turn_hashes: set[str] = set()  # in-run dedup of identical assistant turns

    import contextlib

    for idx, f in enumerate(files):
        if progress_cb is not None:
            with contextlib.suppress(Exception):
                progress_cb(idx, len(files), f)

        key = str(f)
        prev_count = state.get(key, {}).get("lines_processed", 0)
        try:
            text = f.read_text(encoding="utf-8")
            line_count = text.count("\n") + 1 if text else 0
        except (OSError, UnicodeDecodeError):
            text = ""
            line_count = 0
        if line_count <= prev_count:
            files_skipped += 1
            continue

        result = mine_exchange_stream(
            mem,
            chat,
            cfg,
            iter_exchanges(f, text=text),
            turn_hashes=turn_hashes,
            dry_run=dry_run,
            debug=debug,
            source_name=f.name,
            extra_fn=_transcript_extra_fn(f),
        )
        total_candidates += result["candidates"]
        total_saved.extend(result["saved"])
        total_dup += result["skipped_dup"]

        if not dry_run:
            state[key] = {
                "lines_processed": line_count,
                "mtime": f.stat().st_mtime,
            }
            _save_state(cfg.state_dir, state)
        files_processed += 1

    return {
        "status": "ok",
        "root": str(root),
        "files_total": len(files),
        "files_processed": files_processed,
        "files_skipped": files_skipped,
        "candidates": total_candidates,
        "saved": total_saved,
        "skipped_dup": total_dup,
        "dry_run": dry_run,
    }
