"""Per-candidate graduation ledger: one JSONL file per flag under
``state_dir/graduation/``. Entry shape matches what ``graduation_streak``
consumes: winning nights are ``{"verdict": "confirmed", "realized_delta": >=0}``."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from memo.dream_tune_online import graduation_streak


def _flag_path(state_dir: Path, flag: str) -> Path:
    return Path(state_dir) / "graduation" / f"{flag}.jsonl"


def record(state_dir: Path, flag: str, entry: dict[str, Any]) -> None:
    p = _flag_path(state_dir, flag)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


def history(state_dir: Path, flag: str, *, limit: int = 50) -> list[dict[str, Any]]:
    try:
        lines = _flag_path(state_dir, flag).read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    out: list[dict[str, Any]] = []
    for line in lines[-limit:]:
        stripped = line.strip()
        if not stripped:
            continue
        try:
            out.append(json.loads(stripped))
        except json.JSONDecodeError:
            continue
    return out


def streak(state_dir: Path, flag: str, *, limit: int = 50) -> int:
    return graduation_streak(history(state_dir, flag, limit=limit))
