"""Cold-start importers — parse other harnesses' chat logs into memo's
exchange format.

Every `iter_*_exchanges` yields the same `(user_text, assistant_text)` pairs
as `transcript_miner.iter_exchanges` (Claude Code), so all sources replay
through ONE pipeline: `transcript_miner.mine_exchange_stream` (prefilter →
helper-LLM extract → embedding dedup → save). Nothing here touches the
recall-hook path.

Formats (verified against real files, 2026-07):
- Codex CLI: `~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl`, lines
  `{"type": "response_item", "payload": {"type": "message", "role": ...,
  "content": [{"type": "input_text"|"output_text", "text": ...}]}}`.
  Injected AGENTS.md / <environment_context> blocks arrive as role=user.
- opencode: SQLite (`~/.local/share/opencode/opencode.db`), tables
  `message(id, session_id, time_created, data JSON{role})` and
  `part(id, message_id, session_id, time_created, data JSON{type, text})`.
- ChatGPT export: `conversations.json`, mapping node tree
  ({author.role, content.parts}, ordered by message.create_time).
- Claude.ai export: `conversations.json`, `chat_messages`
  ({sender: human|assistant, text | content blocks}).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

__all__ = [
    "iter_chatgpt_exchanges",
    "iter_claude_export_exchanges",
    "iter_codex_exchanges",
    "iter_opencode_exchanges",
]

_log = logging.getLogger(__name__)


def _pair_turns(turns: Iterable[tuple[str, str]]) -> Iterator[tuple[str, str]]:
    """Fold an ordered (role, text) stream into (user, assistant) exchange
    pairs — same state machine as `transcript_miner.iter_exchanges`:
    consecutive assistant texts concatenate onto the previous user turn."""
    pending_user: str | None = None
    pending_assist: list[str] = []
    for role, text in turns:
        text = (text or "").strip()
        if not text:
            continue
        if role == "user":
            if pending_user is not None and pending_assist:
                yield (pending_user, "\n\n".join(pending_assist))
            pending_user = text
            pending_assist = []
        elif role == "assistant":
            if pending_user is None:
                continue
            pending_assist.append(text)
    if pending_user is not None and pending_assist:
        yield (pending_user, "\n\n".join(pending_assist))


# -- Codex CLI rollouts -----------------------------------------------------

_CODEX_SKIP_PREFIXES = ("<", "# AGENTS.md")


def iter_codex_exchanges(rollout_path: Path) -> Iterator[tuple[str, str]]:
    """Yield (user, assistant) pairs from one Codex rollout JSONL file.

    Skips harness-injected user blocks (AGENTS.md instructions,
    <environment_context>, <user_instructions>)."""

    def _turns() -> Iterator[tuple[str, str]]:
        try:
            lines = rollout_path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            return
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                item = obj.get("payload", obj) if isinstance(obj, dict) else None
                if not isinstance(item, dict) or item.get("type") != "message":
                    continue
                role = item.get("role")
                if role not in ("user", "assistant"):
                    continue
                parts = item.get("content") or []
                text = "\n".join(
                    str(c.get("text", "")) for c in parts if isinstance(c, dict) and c.get("text")
                ).strip()
                if not text:
                    continue
                if role == "user" and text.startswith(_CODEX_SKIP_PREFIXES):
                    continue
                yield (role, text)
            except Exception:  # noqa: S112 — untrusted external data; one bad line must not abort
                continue

    yield from _pair_turns(_turns())


# -- opencode SQLite ---------------------------------------------------------


def iter_opencode_exchanges(db_path: Path) -> Iterator[tuple[str, str]]:
    """Yield (user, assistant) pairs from an opencode SQLite store.

    Read-only connection; exchanges never pair across session boundaries.
    Only `part.data.type == "text"` parts are used (reasoning/tool parts
    are skipped)."""
    import sqlite3

    if not Path(db_path).is_file():
        return
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    except sqlite3.Error as exc:
        _log.warning("opencode import: cannot open %s: %s", db_path, exc)
        return
    try:
        rows = conn.execute(
            """
            SELECT m.session_id, m.id, m.data, p.data
            FROM message m JOIN part p ON p.message_id = m.id
            ORDER BY m.session_id, m.time_created, m.id, p.time_created, p.id
            """
        ).fetchall()
    except sqlite3.Error as exc:
        _log.warning("opencode import: schema mismatch in %s: %s", db_path, exc)
        return
    finally:
        conn.close()

    # Collapse parts into per-message texts, keyed by session.
    messages: list[tuple[str, str, str]] = []  # (session_id, role, text)
    cur_key: str | None = None
    cur_role = ""
    cur_session = ""
    buf: list[str] = []
    for session_id, msg_id, msg_data, part_data in rows:
        try:
            role = str(json.loads(msg_data).get("role") or "")
            part = json.loads(part_data)
            if not isinstance(part, dict) or part.get("type") != "text":
                continue
            text = str(part.get("text") or "").strip()
            if not text:
                continue
            if msg_id != cur_key:
                if buf:
                    messages.append((cur_session, cur_role, "\n".join(buf)))
                cur_key, cur_role, cur_session, buf = msg_id, role, str(session_id), []
            buf.append(text)
        except Exception:  # noqa: S112 — untrusted external data; one bad row must not abort
            continue
    if buf:
        messages.append((cur_session, cur_role, "\n".join(buf)))

    last_session: str | None = None
    turns: list[tuple[str, str]] = []
    for session_id, role, text in messages:
        if session_id != last_session and turns:
            yield from _pair_turns(turns)
            turns = []
        last_session = session_id
        turns.append((role, text))
    if turns:
        yield from _pair_turns(turns)


# -- ChatGPT export ----------------------------------------------------------


def iter_chatgpt_exchanges(export_path: Path) -> Iterator[tuple[str, str]]:
    """Yield (user, assistant) pairs from a ChatGPT `conversations.json` export.

    Mapping nodes are ordered by `message.create_time` (None-safe), which
    reconstructs the main thread well enough for insight mining."""
    try:
        data = json.loads(export_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    if isinstance(data, dict):
        data = data.get("conversations", [])
    if not isinstance(data, list):
        return
    for convo in data:
        if not isinstance(convo, dict):
            continue
        mapping = convo.get("mapping")
        if not isinstance(mapping, dict):
            continue
        msgs: list[tuple[float, str, str]] = []
        for node in mapping.values():
            try:
                msg = node.get("message") if isinstance(node, dict) else None
                if not isinstance(msg, dict):
                    continue
                role = str((msg.get("author") or {}).get("role") or "")
                if role not in ("user", "assistant"):
                    continue
                content = msg.get("content") or {}
                if not isinstance(content, dict) or content.get("content_type") != "text":
                    continue
                text = "\n".join(
                    str(p) for p in (content.get("parts") or []) if isinstance(p, str)
                ).strip()
                if not text:
                    continue
                msgs.append((float(msg.get("create_time") or 0.0), role, text))
            except Exception:  # noqa: S112 — untrusted external data; one bad node must not abort
                continue
        msgs.sort(key=lambda t: t[0])
        yield from _pair_turns((role, text) for _, role, text in msgs)


# -- Claude.ai export ---------------------------------------------------------


def iter_claude_export_exchanges(export_path: Path) -> Iterator[tuple[str, str]]:
    """Yield (user, assistant) pairs from a Claude.ai `conversations.json`
    export. Handles both the flat `text` field and the newer `content`
    block-list format; `sender: human` maps to user."""
    try:
        data = json.loads(export_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    if not isinstance(data, list):
        return
    for convo in data:
        if not isinstance(convo, dict):
            continue

        def _turns(convo: dict = convo) -> Iterator[tuple[str, str]]:
            for msg in convo.get("chat_messages") or []:
                try:
                    if not isinstance(msg, dict):
                        continue
                    sender = msg.get("sender")
                    if sender == "human":
                        role = "user"
                    elif sender == "assistant":
                        role = "assistant"
                    else:
                        continue
                    text = str(msg.get("text") or "").strip()
                    if not text:
                        blocks = msg.get("content") or []
                        text = "\n".join(
                            str(b.get("text", ""))
                            for b in blocks
                            if isinstance(b, dict) and b.get("type") == "text"
                        ).strip()
                    if text:
                        yield (role, text)
                except Exception:  # noqa: S112 — untrusted external data; one bad msg must not abort
                    continue

        yield from _pair_turns(_turns())


# -- runners (cursored + one-shot) -------------------------------------------

_IMPORT_STATE = "import-history.json"


def run_codex_import(
    root: Path | None = None,
    *,
    since_days: float | None = None,
    file_limit: int | None = None,
    dry_run: bool = False,
    debug: bool = False,
) -> dict[str, Any]:
    """Walk Codex rollout JSONLs, mine exchanges, save. Resumable via
    per-file line cursors in `state_dir/import-history.json` (same
    semantics as mine-history)."""
    from memo.config import Config
    from memo.memory import Memory
    from memo.transcript_miner import (
        _load_state,
        _save_state,
        find_transcripts,
        mine_exchange_stream,
    )

    cfg = Config.from_env()
    root = root or Path.home() / ".codex" / "sessions"
    files = find_transcripts(root, since_days=since_days)
    if file_limit is not None and file_limit > 0:
        files = files[:file_limit]
    if not files:
        return {"status": "no_files", "root": str(root), "files": 0}

    state = _load_state(cfg.state_dir, name=_IMPORT_STATE)
    mem = Memory(cfg)
    chat = mem._ensure_chat()
    turn_hashes: set[str] = set()
    candidates = 0
    saved: list[str] = []
    skipped_dup = 0
    files_processed = 0
    files_skipped = 0
    for f in files:
        key = str(f)
        prev = state.get(key, {}).get("lines_processed", 0)
        try:
            text = f.read_text(encoding="utf-8")
            line_count = text.count("\n") + 1 if text else 0
        except (OSError, UnicodeDecodeError):
            line_count = 0
        if line_count <= prev:
            files_skipped += 1
            continue
        result = mine_exchange_stream(
            mem,
            chat,
            cfg,
            iter_codex_exchanges(f),
            turn_hashes=turn_hashes,
            dry_run=dry_run,
            debug=debug,
            source_name=f.name,
        )
        candidates += result["candidates"]
        saved.extend(result["saved"])
        skipped_dup += result["skipped_dup"]
        if not dry_run:
            state[key] = {"lines_processed": line_count, "mtime": f.stat().st_mtime}
            _save_state(cfg.state_dir, state, name=_IMPORT_STATE)
        files_processed += 1
    return {
        "status": "ok",
        "root": str(root),
        "files_total": len(files),
        "files_processed": files_processed,
        "files_skipped": files_skipped,
        "candidates": candidates,
        "saved": saved,
        "skipped_dup": skipped_dup,
        "dry_run": dry_run,
    }


def run_file_import(
    exchanges: Iterator[tuple[str, str]],
    *,
    dry_run: bool = False,
    debug: bool = False,
    source_name: str = "",
) -> dict[str, Any]:
    """One-shot mine over a single exchange iterator (opencode db, ChatGPT /
    Claude.ai export files). No cursor state — these are point-in-time
    dumps; the embedding near-dup check keeps re-runs from double-saving."""
    from memo.config import Config
    from memo.memory import Memory
    from memo.transcript_miner import mine_exchange_stream

    cfg = Config.from_env()
    mem = Memory(cfg)
    chat = mem._ensure_chat()
    result = mine_exchange_stream(
        mem,
        chat,
        cfg,
        exchanges,
        turn_hashes=set(),
        dry_run=dry_run,
        debug=debug,
        source_name=source_name,
    )
    return {"status": "ok", "dry_run": dry_run, **result}
