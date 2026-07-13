"""`memo onboard` — Day-0 wizard: recall hook + transcript backfill + first briefing.

Orchestrates already-shipped pieces (wire_recall_hook, install_shims_cmd,
mine_transcripts); owns no heavy logic of its own.
"""
from __future__ import annotations

import re
from pathlib import Path

_FM_TITLE_RE = re.compile(r"^title:\s*(.+)$", re.MULTILINE)


def _recent_memories(memory_dir: Path, n: int = 3) -> list[dict[str, str]]:
    """Newest saved memories by mtime — the '3 cosas que ya sé de vos'.

    Disk-only on purpose: markdown is the source of truth and this must not
    cold-load MLX inside a first-run wizard.

    Title extraction priority: (a) YAML frontmatter title: field,
    (b) first H1 heading (# ), (c) filename stem."""
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

        # Priority 1: YAML frontmatter title:
        fm_match = _FM_TITLE_RE.search(head)
        if fm_match:
            title = fm_match.group(1).strip().strip('\'"')
        else:
            # Priority 2: First H1 heading; Priority 3: filename stem
            h1_line = next(
                (ln for ln in head.splitlines() if ln.startswith("# ")),
                None,
            )
            title = (
                h1_line.removeprefix("# ").strip()
                if h1_line
                else p.stem
            )

        out.append({"title": title, "file": p.name})
    return out
