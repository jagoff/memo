"""Nightly chronicle — a human engineering diary distilled from memo's own logs.

Structural clone of dream_profile.py: dream pass -> markdown under a `_` bucket
in ``memory_dir`` (vault when memories_in_vault is on, data_dir otherwise).
Files carry no ``id:`` frontmatter key, so reindex never ingests them.
Gated by ``MEMO_DREAM_CHRONICLE_ENABLED`` (default off).
"""
from __future__ import annotations

import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

CHRONICLE_BUCKET = "_chronicle"

_BULLET_RE = re.compile(r"^\s*[-*]\s+")
_ID_RE = re.compile(r"\[([0-9a-f]{8})\]")


def chronicle_dir(cfg: Any) -> Path:
    """Where chronicle documents live: ``memory_dir/_chronicle/``."""
    return Path(cfg.memory_dir) / CHRONICLE_BUCKET


def chronicle_path(cfg: Any, day: str) -> Path:
    return chronicle_dir(cfg) / f"{day}.md"


def default_day(now: datetime | None = None) -> str:
    """The day being chronicled. A 03:00 nightly run chronicles *yesterday*."""
    return ((now or datetime.now()) - timedelta(hours=6)).date().isoformat()


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".md.tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)


def filter_cited(text: str, allowed: set[str]) -> tuple[str, float]:
    """Drop bullet lines whose citations are missing or not in ``allowed``.

    A bullet survives only if it cites at least one id AND every id it cites
    is allowed — one fabricated id kills the whole bullet. Non-bullet lines
    always pass. Returns (filtered_text, kept_bullets / total_bullets).
    """
    kept = total = 0
    out: list[str] = []
    for line in text.splitlines():
        if not _BULLET_RE.match(line):
            out.append(line)
            continue
        total += 1
        ids = set(_ID_RE.findall(line))
        if ids and ids <= allowed:
            kept += 1
            out.append(line)
    ratio = (kept / total) if total else 1.0
    return "\n".join(out), ratio
