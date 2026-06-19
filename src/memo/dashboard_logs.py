from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)


def recall_log_path(state_dir: Path) -> Path:
    return state_dir / "recall.log"


def recall_hook_log_path(state_dir: Path) -> Path:
    return state_dir / "recall_hook.log"


def _write_jsonl_entry(
    path: Path,
    entry: dict[str, Any],
    *,
    cap: int,
    size_limit: int,
) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        if path.stat().st_size > size_limit:
            lines = path.read_text(encoding="utf-8").splitlines()[-cap:]
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except OSError as exc:
        _log.debug("dashboard: jsonl write/trim failed for %s: %s", path, exc)


def _read_jsonl(
    path: Path,
    *,
    limit: int,
    newest_first: bool = False,
) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        _log.debug("dashboard: log read failed for %s: %s", path, exc)
        return []
    out: list[dict[str, Any]] = []
    for line in lines[-limit:]:
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    if newest_first:
        out.reverse()
    return out


def append_recall_log(
    state_dir: Path,
    *,
    prompt: str,
    hits: list[dict[str, Any]],
    cap: int = 200,
    mode: str | None = None,
    latency_ms: int | None = None,
    via: str | None = None,
    source: str | None = None,
    reason: str | None = None,
    error: str | None = None,
    session_id: str | None = None,
    turn: int | None = None,
    client: str | None = None,
) -> None:
    entry: dict[str, Any] = {
        "ts": datetime.now(UTC).isoformat(timespec="seconds"),
        "prompt": prompt[:200],
        "hits": [
            {
                "id": h.get("id", "")[:8],
                "score": h.get("score"),
                "title": h.get("title", "")[:80],
                **({"snippet": h["snippet"][:240]} if h.get("snippet") else {}),
            }
            for h in hits[:5]
        ],
    }
    if mode is not None:
        entry["mode"] = mode
    if latency_ms is not None:
        entry["latency_ms"] = latency_ms
    if via is not None:
        entry["via"] = via
    if source is not None:
        entry["source"] = source
    if reason is not None:
        entry["reason"] = reason[:200]
    if error is not None:
        entry["error"] = error[:200]
    if session_id is not None:
        entry["session_id"] = session_id
    if turn is not None:
        entry["turn"] = turn
    if client is not None:
        entry["client"] = client
    _write_jsonl_entry(recall_log_path(state_dir), entry, cap=cap, size_limit=1024 * 200)
    if session_id is not None:
        _write_jsonl_entry(recall_hook_log_path(state_dir), entry, cap=2000, size_limit=2_000_000)


def read_recall_log(state_dir: Path, *, limit: int = 10) -> list[dict[str, Any]]:
    return _read_jsonl(recall_log_path(state_dir), limit=limit, newest_first=True)


def read_recall_hook_log(state_dir: Path, *, limit: int = 2000) -> list[dict[str, Any]]:
    return _read_jsonl(recall_hook_log_path(state_dir), limit=limit)


def usage_log_path(state_dir: Path) -> Path:
    return state_dir / "usage.log"


def append_usage_log(state_dir: Path, memoria_id: str, *, cap: int = 500) -> None:
    entry = {
        "ts": datetime.now(UTC).isoformat(timespec="seconds"),
        "id": (memoria_id or "")[:8],
    }
    _write_jsonl_entry(usage_log_path(state_dir), entry, cap=cap, size_limit=1024 * 100)


def read_usage_log(state_dir: Path, *, limit: int = 2000) -> list[dict[str, Any]]:
    return _read_jsonl(usage_log_path(state_dir), limit=limit)


def grounding_log_path(state_dir: Path) -> Path:
    return state_dir / "grounding.log"


def append_grounding_log(
    state_dir: Path,
    *,
    session_id: str,
    turn: int,
    recall_id: str,
    used_score: float,
    method: str,
    client: str | None = None,
    answer_len: int | None = None,
    recall_top_score: float | None = None,
    downstream_action: str | None = None,
    action_evidence: str | None = None,
    specific_score: float | None = None,
    cap: int = 1000,
) -> None:
    entry: dict[str, Any] = {
        "ts": datetime.now(UTC).isoformat(timespec="seconds"),
        "session_id": session_id,
        "turn": int(turn),
        "recall_id": (recall_id or "")[:8],
        "used_score": round(float(used_score), 4),
        "method": method,
    }
    if client is not None:
        entry["client"] = client
    if answer_len is not None:
        entry["answer_len"] = int(answer_len)
    if recall_top_score is not None:
        entry["recall_top_score"] = round(float(recall_top_score), 4)
    if specific_score is not None:
        entry["specific_score"] = round(float(specific_score), 4)
    if downstream_action is not None:
        entry["downstream_action"] = downstream_action
    if action_evidence is not None:
        entry["action_evidence"] = action_evidence[:200]
    _write_jsonl_entry(grounding_log_path(state_dir), entry, cap=cap, size_limit=1024 * 200)


def read_grounding_log(state_dir: Path, *, limit: int = 4000) -> list[dict[str, Any]]:
    return _read_jsonl(grounding_log_path(state_dir), limit=limit)


def grounding_diag_log_path(state_dir: Path) -> Path:
    return state_dir / "grounding_diag.log"


def append_grounding_diag_log(
    state_dir: Path,
    *,
    reason: str,
    session_id: str | None = None,
    turn: int | None = None,
    cap: int = 500,
) -> None:
    entry: dict[str, Any] = {
        "ts": datetime.now(UTC).isoformat(timespec="seconds"),
        "reason": reason,
    }
    if session_id is not None:
        entry["session_id"] = session_id
    if turn is not None:
        entry["turn"] = int(turn)
    _write_jsonl_entry(
        grounding_diag_log_path(state_dir),
        entry,
        cap=cap,
        size_limit=1024 * 100,
    )


def read_grounding_diag_log(state_dir: Path, *, limit: int = 500) -> list[dict[str, Any]]:
    return _read_jsonl(grounding_diag_log_path(state_dir), limit=limit, newest_first=True)
