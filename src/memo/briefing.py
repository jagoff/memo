"""Synapse-aware briefing section composer.

`memo briefing` (SessionStart hook) and the `memo_unified_briefing`
MCP tool both want to surface unified state from synapse — handoffs +
attention queue + open conflicts — alongside memo's own local sections.

This module owns the "borrow from synapse" half. It is intentionally
small + side-effect-free:

* `synapse_briefing_lines(cwd)` returns a list of markdown lines (may
  be empty if synapse is unreachable or returned a degenerate packet).
* All shell-outs go through `memo.synapse_client.get_packet`, which
  already swallows subprocess errors.

Memo never imports synapse and the briefing degrades gracefully when
synapse is absent — keeps the single-Mac path zero-regression per the
plan's "graceful, opt-in" boundary.
"""

from __future__ import annotations

from typing import Any

from memo import synapse_client

_MAX_ITEMS = 3
_SNIPPET_CHARS = 160


def compact_text(text: str, *, max_chars: int = 480) -> str:
    """Collapse blank/indented lines and enforce a hard context-size cap."""
    if max_chars <= 0:
        return ""
    compact = "\n".join(line.strip() for line in str(text or "").splitlines() if line.strip())
    if len(compact) <= max_chars:
        return compact
    if max_chars == 1:
        return "…"
    return compact[: max_chars - 1].rstrip() + "…"


def synapse_briefing_lines(
    cwd: str | None = None,
    *,
    k: int = 5,
) -> list[str]:
    """Return markdown lines summarising synapse's unified consciousness.

    Empty list when synapse is unavailable or returns nothing useful.
    Sections (each only present if the packet had data for it):

    * `### Current state (Synapse)` — top present_state items
      (memflow focus / handoffs / current work).
    * `### Open conflicts` — top reality_conflicts.
    * `_Synapse: ready · trace=<short>_` — health footer.
    """
    if not synapse_client.is_available():
        return []
    query = (cwd or "").strip() or "current focus"
    packet = synapse_client.get_packet(query, k=k)
    if not packet:
        return []

    lines: list[str] = []
    present_lines = _present_state_section(packet.get("present_state"))
    conflict_lines = _conflicts_section(packet.get("reality_conflicts"))
    if not present_lines and not conflict_lines:
        return []

    lines.extend(present_lines)
    lines.extend(conflict_lines)

    status = str(packet.get("status") or "?").strip() or "?"
    trace = str(packet.get("trace_id") or "")
    trace_short = trace.rsplit("/", 1)[-1][:12] if trace else "—"
    lines.append("")
    lines.append(f"_Synapse: {status} · trace={trace_short}_")
    lines.append("")
    return lines


def _present_state_section(rows: Any) -> list[str]:
    items = _coerce_rows(rows)[:_MAX_ITEMS]
    if not items:
        return []
    out: list[str] = ["### Current state (Synapse)", ""]
    for i, item in enumerate(items, 1):
        source = str(item.get("source") or "?")
        title = str(item.get("title") or "—").strip() or "—"
        snippet = _clip(item.get("snippet") or "")
        out.append(f"{i}. **[{source}]** {title}")
        if snippet:
            out.append(f"   > {snippet}")
    out.append("")
    return out


def _conflicts_section(rows: Any) -> list[str]:
    items = _coerce_rows(rows)
    open_items = [
        r
        for r in items
        if str(r.get("lifecycle_state") or "detected").lower() in ("detected", "acknowledged")
    ][:_MAX_ITEMS]
    if not open_items:
        return []
    out: list[str] = ["### Open conflicts", ""]
    for i, c in enumerate(open_items, 1):
        cid = str(c.get("conflict_id") or "?")
        summary = str(c.get("summary") or c.get("title") or "—").strip()
        severity = str(c.get("severity") or "?")
        state = str(c.get("lifecycle_state") or "detected").lower()
        freeze = " ❄️" if c.get("freeze_write") else ""
        out.append(
            f"{i}. `{cid[:12]}` · {state} · sev={severity}{freeze} — {_clip(summary, limit=140)}"
        )
    out.append("")
    return out


def _coerce_rows(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [r for r in value if isinstance(r, dict)]


def _clip(text: str, *, limit: int = _SNIPPET_CHARS) -> str:
    text = str(text or "").strip().replace("\n", " ")
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


__all__ = ["compact_text", "synapse_briefing_lines"]
