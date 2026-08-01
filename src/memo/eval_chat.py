"""Declarative regression checks for the chat pipeline (corpus rescued from synapse)."""

from __future__ import annotations

from typing import Any

from memo.chat.synthesis import REFUSAL


def apply_checks(query: dict[str, Any], done: dict[str, Any], total_ms: int) -> dict[str, Any]:
    checks_spec = query.get("checks") or {}
    answer = str(done.get("answer") or "")
    answer_lower = answer.lower()
    source_ids = [str(s.get("id") or "") for s in done.get("sources") or []]
    results: list[dict[str, Any]] = []

    for sub in checks_spec.get("require_substrings") or []:
        results.append({"check": f"require:{sub}", "passed": str(sub).lower() in answer_lower})
    for sub in checks_spec.get("forbid_substrings") or []:
        results.append({"check": f"forbid:{sub}", "passed": str(sub).lower() not in answer_lower})
    if checks_spec.get("forbid_refusal"):
        results.append({"check": "forbid_refusal", "passed": REFUSAL not in answer})
    min_sources = checks_spec.get("min_sources")
    if isinstance(min_sources, int):
        results.append(
            {"check": f"min_sources:{min_sources}", "passed": len(source_ids) >= min_sources}
        )
    expected = query.get("expected_source_ids") or []
    if expected:
        hit = any(e in sid or sid in e for e in map(str, expected) for sid in source_ids if sid)
        results.append({"check": "expected_source_hit", "passed": hit})

    return {
        "id": str(query.get("id") or "?"),
        "passed": all(r["passed"] for r in results) if results else True,
        "checks": results,
        "total_ms": total_ms,
    }
