"""Phase 3 — ASK: promote AT MOST ONE recurring unmet gap per session into an
explicit question on the SessionStart briefing.

The nightly ``dream_anticipate`` pass already writes recurring knowledge gaps
memo could NOT answer into the dream receipt (``anticipated.gaps``), sorted by
frequency-then-recency, but nothing surfaces them to the user. This module reads
that receipt and, when enabled, renders the single highest-value gap as a
QUESTION in the briefing. It NEVER fabricates — a gap is only ever a prompt memo
already failed to answer from real usage; asking it back is honest.

Report-only: shadow-logs what it WOULD ask regardless of the enable flag; a
human flips MEMO_ASK_GAPS_ENABLED after reviewing ``memo ask-gaps shadow``.
Deduped per session (a session asks at most one NEW gap and never repeats one).

Also home to the code-hub gap surface (``MEMO_GAPS_CODE_HUBS``, default ON):
top codegraph call-magnets no memory documents, via :func:`code_hub_gaps` —
the briefing carries at most ONE such line, read from the nightly dream
receipt (zero graph queries at SessionStart); the full live list is for the
CLI and the nightly code-drift pass.
"""

from __future__ import annotations

import json as _json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from memo.atomic_io import atomic_write_text

SHADOW_LOG = "ask_shadow.log"
_SHADOW_CAP = 1000
_SHADOW_SIZE_LIMIT = 1_000_000


def pick_gap(gaps: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The highest-value gap — the receipt is already sorted (count, recency)."""
    return gaps[0] if gaps else None


def phrase_question(gap: dict[str, Any]) -> str:
    """Re-ask an unmet gap as a question. Pure string; never an invented answer."""
    prompt = str(gap.get("prompt") or "").strip().rstrip("?")
    count = int(gap.get("count", 1) or 1)
    return (
        f"❓ **memo keeps hitting a gap** ({count}×): {prompt}? — "  # noqa: RUF001
        "save what you know so it stops asking."
    )


def _gap_key(gap: dict[str, Any]) -> str:
    return " ".join(str(gap.get("prompt") or "").lower().split())


def _marker_file(state_dir: Path, session_id: str) -> Path:
    from memo.session import validate_session_id

    # Internal fallback used by SessionStart when the client has no id yet.
    safe_id = session_id if session_id == "_no_session" else validate_session_id(session_id)
    state_root = Path(state_dir).resolve()
    marker_dir = state_root / ".ask_gaps_seen"
    path = marker_dir / f"{safe_id}.json"
    if (
        marker_dir.is_symlink()
        or path.is_symlink()
        or not path.resolve().is_relative_to(state_root)
    ):
        raise ValueError("session_id resolves to an unsafe ask-gaps marker path")
    return path


def _load_marker(state_dir: Path, session_id: str) -> dict[str, Any]:
    f = _marker_file(state_dir, session_id)
    if not f.is_file():
        return {}
    try:
        data = _json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _save_marker(state_dir: Path, session_id: str, marker: dict[str, Any]) -> None:
    f = _marker_file(state_dir, session_id)
    f.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(f, _json.dumps(marker))


def already_asked(state_dir: Path, session_id: str, key: str) -> bool:
    return key in set(_load_marker(state_dir, session_id).get("asked", []))


def note_asked(state_dir: Path, session_id: str, key: str) -> None:
    m = _load_marker(state_dir, session_id)
    asked = set(m.get("asked", []))
    asked.add(key)
    m["asked"] = sorted(asked)
    _save_marker(state_dir, session_id, m)


def shadow_record(gap: dict[str, Any], *, rendered: bool) -> dict[str, Any]:
    return {
        "ts": datetime.now(UTC).isoformat(timespec="seconds"),
        "prompt": str(gap.get("prompt") or "")[:200],
        "count": int(gap.get("count", 1) or 1),
        "rendered": bool(rendered),
    }


def _shadow_path(state_dir: Path) -> Path:
    return Path(state_dir) / SHADOW_LOG


def log_shadow(state_dir: Path, entry: dict[str, Any]) -> None:
    from memo.dashboard_logs import _write_jsonl_entry

    _write_jsonl_entry(
        _shadow_path(state_dir), entry, cap=_SHADOW_CAP, size_limit=_SHADOW_SIZE_LIMIT
    )


def read_shadow(state_dir: Path, *, limit: int = 1000) -> list[dict[str, Any]]:
    from memo.dashboard_logs import _read_jsonl

    return _read_jsonl(_shadow_path(state_dir), limit=limit, newest_first=True)


def _read_receipt_gaps(state_dir: Path) -> list[dict[str, Any]]:
    try:
        last = Path(state_dir) / "dream" / "last.json"
        if not last.is_file():
            return []
        data = _json.loads(last.read_text(encoding="utf-8"))
        gaps = ((data.get("anticipated") or {}).get("gaps")) or []
        return [g for g in gaps if isinstance(g, dict) and g.get("prompt")]
    except Exception:
        return []


# --- code-hub gaps (MEMO_GAPS_CODE_HUBS) ------------------------------------------

_HUB_TOP = 10
# Same base query `memo code-facts` mines call hubs with: in-degree of `calls`
# edges over src/ nodes only.
_HUB_QUERY = (
    "SELECT t.name, t.file_path, COUNT(*) AS n "
    "FROM edges e JOIN nodes t ON e.target = t.id "
    "WHERE e.kind = 'calls' AND t.file_path LIKE 'src/%' "
    "GROUP BY t.id ORDER BY n DESC, t.qualified_name LIMIT ?"
)


def code_hub_gaps(mem: Any, top: int = _HUB_TOP) -> list[str]:
    """Knowledge gaps on code hubs: top call-magnets no memory documents.

    The top-``top`` codegraph nodes by incoming ``calls`` edges (src/ only)
    that no non-reference memory cites, one ``hub sin memoria: …`` line each.
    Gated by ``MEMO_GAPS_CODE_HUBS``; on-demand graph query only (never the
    recall hook, never SessionStart). Fail-open: flag off / no index → [].
    """
    from memo import code_intel
    from memo.flags import flag_bool

    if not flag_bool("MEMO_GAPS_CODE_HUBS"):
        return []
    opened = code_intel.open_graph()
    if opened is None:
        return []
    graph, _db_repo_id = opened
    try:
        return _hub_gap_lines(mem.store._conn, graph, top=top)
    finally:
        graph.close()


def _hub_gap_lines(store_conn: Any, graph: Any, *, top: int) -> list[str]:
    """The ``hub sin memoria`` lines over an OPEN graph connection — shared by
    :func:`code_hub_gaps` and the nightly code-drift pass (which persists them
    to the receipt for the briefing). A failed hub query degrades to no lines."""
    from memo import code_intel

    try:
        rows = graph.execute(_HUB_QUERY, (int(top),)).fetchall()
    except sqlite3.Error:
        return []
    lines: list[str] = []
    for name, file_path, callers in rows:
        symbol = str(name or "")
        if not symbol:
            continue
        if code_intel.memories_citing(store_conn, symbols={symbol}, limit=1):
            continue
        lines.append(f"hub sin memoria: {symbol} ({callers} callers) — {file_path}")
    return lines


def _hub_briefing_lines(cfg: Any) -> list[str]:
    """At most ONE hub line for the briefing, read from the nightly dream
    receipt (``code_drift.hub_gaps``, written by the code-drift pass).

    SessionStart's budget is ZERO graph queries: no index open, no git, no
    store scan — one small json read. The live query belongs to
    :func:`code_hub_gaps` (on-demand) and the nightly pass. No receipt /
    corrupt receipt / flag off → []."""
    from memo.flags import flag_bool

    if not flag_bool("MEMO_GAPS_CODE_HUBS"):
        return []
    try:
        data = _json.loads(
            (Path(cfg.state_dir) / "dream" / "last.json").read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        return []
    section = data.get("code_drift") if isinstance(data, dict) else None
    gaps = section.get("hub_gaps") if isinstance(section, dict) else None
    if not isinstance(gaps, list):
        return []
    return [g for g in gaps if isinstance(g, str) and g][:1]


def briefing_lines(cfg: Any, *, session_id: str) -> list[str]:
    """SessionStart render: at most one NEW high-value gap as a question plus
    at most one undocumented code hub. Never raises, never fabricates.
    Shadow-logs whether or not the question renders."""
    lines: list[str] = []
    try:
        from memo.flags import flag_bool

        gaps = _read_receipt_gaps(cfg.state_dir)
        gap = pick_gap(gaps)
        if gap is not None:
            key = _gap_key(gap)
            if not already_asked(cfg.state_dir, session_id, key):
                render = flag_bool("MEMO_ASK_GAPS_ENABLED")
                log_shadow(cfg.state_dir, shadow_record(gap, rendered=render))
                if render:
                    note_asked(cfg.state_dir, session_id, key)
                    lines.extend([phrase_question(gap), ""])
        lines.extend(_hub_briefing_lines(cfg))
    except Exception:
        return lines
    return lines
