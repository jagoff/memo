"""`memo onboard` — Day-0 wizard: recall hook + transcript backfill + first briefing.

Orchestrates already-shipped pieces (wire_recall_hook, install_shims_cmd,
mine_transcripts); owns no heavy logic of its own.
"""
from __future__ import annotations

from pathlib import Path


def _recent_memories(memory_dir: Path, n: int = 3) -> list[dict[str, str]]:
    """Newest saved memories by mtime — the '3 cosas que ya sé de vos'.

    Disk-only on purpose: markdown is the source of truth and this must not
    cold-load MLX inside a first-run wizard."""
    if not memory_dir.exists():
        return []
    files = [
        p
        for p in memory_dir.rglob("*.md")
        if not any(part.startswith("_") for part in p.relative_to(memory_dir).parts)
    ]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    out: list[dict[str, str]] = []
    for p in files[:n]:
        head = p.read_text(encoding="utf-8", errors="ignore")[:1000]
        title = next(
            (ln.lstrip("# ").strip() for ln in head.splitlines() if ln.startswith("# ")),
            p.stem,
        )
        out.append({"title": title, "file": p.name})
    return out
