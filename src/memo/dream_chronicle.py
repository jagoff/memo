"""Nightly chronicle — a human engineering diary distilled from memo's own logs.

Structural clone of dream_profile.py: dream pass -> markdown under a `_` bucket
in ``memory_dir`` (vault when memories_in_vault is on, data_dir otherwise).
Files carry no ``id:`` frontmatter key, so reindex never ingests them.
Gated by ``MEMO_DREAM_CHRONICLE_ENABLED`` (default off).
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

CHRONICLE_BUCKET = "_chronicle"

_BULLET_RE = re.compile(r"^\s*[-*]\s+")
_ID_RE = re.compile(r"\[([0-9a-f]{8})\]")
_FM_ID_RE = re.compile(r"^id:\s*([0-9a-f]{8,})", re.MULTILINE)
_FM_TYPE_RE = re.compile(r"^type:\s*(\S+)", re.MULTILINE)
_FM_TITLE_RE = re.compile(r"^title:\s*(.+)$", re.MULTILINE)
_DATE_KEYS = ("created:", "created_at:", "updated:", "date:")
_RECEIPT_KEYS = ("superseded", "merged", "archived_stale", "synthesized")


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


def _memories_created_on(cfg: Any, day: str, cap: int = 50) -> list[dict[str, str]]:
    """Memories whose frontmatter date starts with ``day``. Disk scan — markdown
    is the source of truth; `_`-prefixed buckets (_profile/_chronicle) are not
    memories."""
    root = Path(cfg.memory_dir)
    if not root.exists():
        return []
    out: list[dict[str, str]] = []
    for p in sorted(root.rglob("*.md")):
        if any(part.startswith("_") for part in p.relative_to(root).parts):
            continue
        head = p.read_text(encoding="utf-8", errors="ignore")[:2000]
        m_id = _FM_ID_RE.search(head)
        if m_id is None:
            continue  # no id: -> not a memory record

        # Extract the full frontmatter block (between --- delimiters)
        lines = head.splitlines()
        fm_end = None
        for i, line in enumerate(lines[1:], start=1):
            if line.startswith("---"):
                fm_end = i + 1
                break
        fm_lines = lines[:60] if fm_end is None else lines[:fm_end]

        fm_text = "\n".join(fm_lines)
        date_lines = [
            ln.strip() for ln in fm_text.splitlines() if ln.strip().startswith(_DATE_KEYS)
        ]
        if not any(day in ln for ln in date_lines):
            continue

        m_type = _FM_TYPE_RE.search(head)

        # Extract title: (a) frontmatter title:, (b) first # heading, (c) file stem
        title = p.stem
        m_title = _FM_TITLE_RE.search(head)
        if m_title:
            title = m_title.group(1).strip('\'"')
        else:
            # Fallback to first # heading
            title = next(
                (ln.lstrip("# ").strip() for ln in head.splitlines() if ln.startswith("# ")),
                p.stem,
            )

        out.append({"id": m_id.group(1), "type": m_type.group(1) if m_type else "note", "title": title})
        if len(out) >= cap:
            break
    return out


def collect_facts(cfg: Any, day: str) -> dict[str, Any]:
    """Deterministic day facts — no LLM. Each sub-source is best-effort."""
    from memo.dashboard_logs import read_grounding_log, read_recall_log
    from memo.resume._index import open_store
    from memo.token_ledger import consults_by_day_client, grounded_by_day

    episodes: list[dict[str, Any]] = []
    store = open_store(cfg)
    if store is not None:
        episodes = [
            e for e in store.recent(limit=200)
            if str(e.get("updated_at") or "").startswith(day)
        ]

    grounded = grounded_by_day(read_grounding_log(cfg.state_dir)).get(day, 0)
    consults = consults_by_day_client(read_recall_log(cfg.state_dir, limit=4000)).get(day, {})

    receipt_events: dict[str, int] = {}
    last = Path(cfg.state_dir) / "dream" / "last.json"
    if last.exists():
        try:
            data = json.loads(last.read_text(encoding="utf-8"))
            for key in _RECEIPT_KEYS:
                v = data.get(key)
                if isinstance(v, list) and v:
                    receipt_events[key] = len(v)
        except (ValueError, OSError):
            pass

    return {
        "episodes": episodes,
        "new_memories": _memories_created_on(cfg, day),
        "grounded": grounded,
        "consults": consults,
        "receipt_events": receipt_events,
    }


def fact_lines(facts: dict[str, Any]) -> tuple[list[str], set[str]]:
    """Citable bullets fed to the LLM. Every line ends with its [id8];
    the returned set is the provenance whitelist for filter_cited()."""
    lines: list[str] = []
    allowed: set[str] = set()
    for e in facts["episodes"]:
        sid = str(e.get("session_id") or "")[:8]
        if not sid:
            continue
        allowed.add(sid)
        lines.append(
            f"- session {sid} ({e.get('agent', '?')}, {e.get('turn_count', 0)} turns): "
            f"{str(e.get('summary') or '')[:200]} [{sid}]"
        )
    for m in facts["new_memories"]:
        mid = str(m["id"])[:8]
        allowed.add(mid)
        lines.append(f"- new {m['type']} memory: {str(m['title'])[:120]} [{mid}]")
    return lines, allowed
