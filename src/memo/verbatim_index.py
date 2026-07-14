"""Incremental turn-level indexer for the verbatim (lexical) index.

Parses `~/.claude/projects/**/*.jsonl` transcripts into timestamped turns and
feeds them into `TurnStore` (FTS5, no embeddings). Watermarked so a nightly
pass only re-ingests sessions that grew since the last run — the watermark
format mirrors `transcript_miner.py`'s `mine-history.json`
(`{path: {lines_processed, mtime}}`).

Never enters the recall hook (CLAUDE.md); this is the on-demand `memo
verbatim` surface's ingestion side. `run_verbatim_index_pass` never raises —
callers (the nightly `memo dream run`) always get a status dict back.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from memo.capture_core import _extract_text
from memo.flags import flag_int
from memo.redact import redact_secrets
from memo.store.turn_store import TurnStore
from memo.transcript_miner import _load_state, _save_state, find_transcripts

if TYPE_CHECKING:
    from memo.config import Config

_log = logging.getLogger(__name__)

_WATERMARK_FILE = "verbatim-index.json"


def parse_turns(transcript_path: Path) -> list[dict[str, Any]]:
    """Parse a JSONL transcript into timestamped, redacted turns.

    Variant of `capture_core._parse_transcript` that PRESERVES the JSONL
    `timestamp` field `_parse_transcript` discards. Text is the same
    tool-evidence-capped projection (`_extract_text`), then redacted
    (`redact_secrets`). Turns shorter than `MEMO_VERBATIM_MIN_CHARS` (default
    20) are skipped. Returns `[{idx, role, ts, text}]`, `idx` a running index
    over the surviving turns.
    """
    if not transcript_path.is_file():
        return []
    try:
        lines = transcript_path.read_text(encoding="utf-8").splitlines()
    except Exception as exc:
        _log.debug("verbatim: transcript read failed (%s): %s", transcript_path, exc)
        return []

    min_chars = flag_int("MEMO_VERBATIM_MIN_CHARS")
    if min_chars is None:
        min_chars = 20

    turns: list[dict[str, Any]] = []
    idx = 0
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
        text = redact_secrets(text).text
        if len(text) < min_chars:
            continue
        turns.append({"idx": idx, "role": role, "ts": obj.get("timestamp"), "text": text})
        idx += 1
    return turns


def run_verbatim_index_pass(
    cfg: Config,
    *,
    max_days: float | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Incrementally index transcript turns into `TurnStore`. Never raises.

    Walks `~/.claude/projects/**/*.jsonl` (via `find_transcripts`), skipping
    any file whose line count/mtime match the watermark from the last run.
    Changed files are re-parsed and their session fully replaced
    (`TurnStore.replace_session`, idempotent). Ends with a
    `prune_older_than(max_days or MEMO_VERBATIM_MAX_DAYS)` sweep.
    """
    result: dict[str, Any] = {
        "status": "ok",
        "sessions_indexed": 0,
        "turns_indexed": 0,
        "pruned": 0,
        "skipped_unchanged": 0,
    }
    try:
        effective_max_days = (
            max_days if max_days is not None else flag_int("MEMO_VERBATIM_MAX_DAYS")
        )
        if effective_max_days is None:
            effective_max_days = 90

        root = Path.home() / ".claude" / "projects"
        files = find_transcripts(root, since_days=effective_max_days)

        state = _load_state(cfg.state_dir, name=_WATERMARK_FILE)
        store = TurnStore(cfg.verbatim_db)
        try:
            for f in files:
                key = str(f)
                prev = state.get(key, {})
                prev_count = prev.get("lines_processed", 0)
                prev_mtime = prev.get("mtime")
                try:
                    text = f.read_text(encoding="utf-8")
                    line_count = text.count("\n") + 1 if text else 0
                    mtime = f.stat().st_mtime
                except (OSError, UnicodeDecodeError):
                    continue

                if line_count <= prev_count and mtime == prev_mtime:
                    result["skipped_unchanged"] += 1
                    continue

                turns = parse_turns(f)
                if not dry_run:
                    n = store.replace_session(session_id=f.stem, agent="claude-code", turns=turns)
                    result["sessions_indexed"] += 1
                    result["turns_indexed"] += n
                    state[key] = {"lines_processed": line_count, "mtime": mtime}
                    _save_state(cfg.state_dir, state, name=_WATERMARK_FILE)
                else:
                    result["sessions_indexed"] += 1
                    result["turns_indexed"] += len(turns)

            if not dry_run:
                result["pruned"] = store.prune_older_than(int(effective_max_days))
        finally:
            store.close()
    except Exception as exc:
        _log.warning("verbatim index pass failed: %s", exc)
        result["status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


__all__ = ["parse_turns", "run_verbatim_index_pass"]
