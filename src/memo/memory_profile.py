"""Bounded, evidence-aware profile projection for agents.

The profile is a read-only view over Memo's existing markdown/SQLite state.
It deliberately does not introduce a second profile store: the stable section
comes from the dream-maintained profile documents and the active section from
recent durable records.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

SCHEMA = "memo.profile.v1"


def _confidence(record: Any) -> float:
    value = (getattr(record, "extra", {}) or {}).get("confidence")
    try:
        if value is not None:
            return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        pass
    state = getattr(getattr(record, "verification_state", None), "value", "unverified")
    return {"verified": 1.0, "stale": 0.5, "unverified": 0.7}.get(state, 0.7)


def _freshness(updated: str) -> str | None:
    if not updated:
        return None
    try:
        parsed = datetime.fromisoformat(updated.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed.isoformat()
    except ValueError:
        return updated


def build_memory_profile(
    memory: Any,
    *,
    scope: str = "current",
    cwd: str | None = None,
    limit: int = 8,
    budget_chars: int = 4000,
) -> dict[str, Any]:
    """Return a bounded stable/active profile envelope.

    ``scope`` is intentionally descriptive (``current``, ``user``,
    ``project`` or ``agent``); filtering remains governed by Memo's existing
    project/profile files and record provenance. This prevents callers from
    accidentally treating a client-provided scope as an authorization claim.
    """
    if scope not in {"current", "user", "project", "agent"}:
        raise ValueError("scope must be one of: current, user, project, agent")
    limit = max(0, min(int(limit), 50))
    budget_chars = max(256, min(int(budget_chars), 12000))

    from memo.briefing import profile_lines

    stable_text = "\n".join(profile_lines(memory.cfg, cwd=cwd)).strip()
    stable: list[dict[str, Any]] = []
    if stable_text:
        stable_truncated = len(stable_text) > budget_chars
        if stable_truncated:
            stable_text = stable_text[: max(0, budget_chars - 1)].rstrip() + "…"
        stable.append(
            {
                "scope": scope,
                "text": stable_text,
                "confidence": 0.8,
                "freshness": None,
                "evidence_ids": [],
                "source": "profile_document",
            }
        )

    active: list[dict[str, Any]] = []
    try:
        records = memory.list(limit=limit, include_forgotten=False)
    except Exception:
        records = []
    for record in records[:limit]:
        active.append(
            {
                "id": record.id,
                "id_short": record.id[:8],
                "scope": scope,
                "title": record.title,
                "type": record.type,
                "text": record.body[:800],
                "confidence": _confidence(record),
                "freshness": _freshness(record.updated),
                "evidence_ids": [record.id],
                "source": "memory_record",
            }
        )

    omissions: list[str] = []
    if not stable and not active:
        omissions.append("no stable or active memory is available")
    if stable_text and stable_text.endswith("…"):
        omissions.append("stable profile truncated to the requested budget")
    return {
        "schema": SCHEMA,
        "scope": scope,
        "available": bool(stable or active),
        "stable": stable,
        "active": active,
        "omissions": omissions,
        "budget_chars": budget_chars,
    }
