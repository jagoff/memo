"""Read-only trust + adoption diagnostics for `memo usefulness doctor`."""

from __future__ import annotations

from typing import Any

from memo.config import Config
from memo.dashboard import (
    consult_breakdown,
    consumer_label,
    read_grounding_log,
    read_recall_log,
    recall_health,
)
from memo.memory import Memory

DiagnosticItem = dict[str, Any]
ActionItem = dict[str, Any]


def _item(
    *,
    id: str,
    severity: str,
    status: str,
    message: str,
    action: str,
    evidence: dict[str, Any] | None = None,
) -> DiagnosticItem:
    return {
        "id": id,
        "severity": severity,
        "status": status,
        "message": message,
        "evidence": evidence or {},
        "action": action,
    }


def _action(id: str, command: str, reason: str) -> ActionItem:
    return {"id": id, "command": command, "reason": reason}


def _adoption_items(cfg: Config, *, limit: int) -> tuple[list[DiagnosticItem], list[ActionItem]]:
    breakdown = consult_breakdown(cfg.state_dir, limit=limit)
    health = recall_health(cfg.state_dir, limit=limit)
    rows = read_recall_log(cfg.state_dir, limit=limit)
    items: list[DiagnosticItem] = []
    actions: list[ActionItem] = []

    silent = list(breakdown.get("silent") or [])
    consumers = list(breakdown.get("consumers") or [])
    if not consumers:
        items.append(
            _item(
                id="no_consults",
                severity="critical",
                status="silent",
                message="No memo consults were recorded in the sampled window.",
                evidence={"sampled": breakdown.get("sampled", 0)},
                action="Run a memo-enabled agent session or check MCP/hook installation.",
            )
        )
        actions.append(
            _action(
                "check_install",
                "memo doctor --strict-runtime",
                "No consult telemetry exists.",
            )
        )
    elif silent:
        items.append(
            _item(
                id="silent_consumers",
                severity="warning",
                status="degraded",
                message="Expected memo consumers have zero recent consults.",
                evidence={"silent": silent},
                action='Configure those clients to call memo and pass source="<client>".',
            )
        )
        actions.append(
            _action(
                "source_attribution",
                'Pass source="<client>" on memo read tool calls.',
                "Silent or unattributed consumers cannot prove adoption.",
            )
        )

    unattributed = [
        r
        for r in rows
        if consumer_label(r) in {"mcp:unknown", "unknown"}
        or ((r.get("via") or "").startswith("mcp:") and not (r.get("source") or r.get("client")))
    ]
    if unattributed:
        items.append(
            _item(
                id="unattributed_consults",
                severity="warning",
                status="degraded",
                message="Some memo consults lack explicit source attribution.",
                evidence={"count": len(unattributed), "sample": unattributed[:3]},
                action='Pass source="<client>" on memo read tool calls.',
            )
        )
        actions.append(
            _action(
                "add_source",
                'Pass source="<client>" on memo read tool calls.',
                "Unattributed consults collapse into generic consumers.",
            )
        )

    grounded_rate = health.get("grounded_rate")
    if consumers and grounded_rate is None:
        items.append(
            _item(
                id="grounding_unmeasured",
                severity="warning",
                status="unmeasured",
                message="Memo is consulted, but grounding has not been measured yet.",
                evidence={
                    "sampled": health.get("sampled"),
                    "measured_turns": health.get("measured_turns"),
                    "surfaced_turns": health.get("surfaced_turns"),
                },
                action="Let Stop-hook grounding accumulate or inspect grounding diagnostics.",
            )
        )
    elif isinstance(grounded_rate, (int, float)) and float(grounded_rate) < 0.1:
        items.append(
            _item(
                id="low_grounding",
                severity="warning",
                status="degraded",
                message="Memo is consulted, but few surfaced memories are grounded.",
                evidence={"grounded_rate": grounded_rate},
                action="Run memo debug-recall on recent low-quality prompts.",
            )
        )

    return items, actions


def _derive_verdict(adoption: list[DiagnosticItem], trust: list[DiagnosticItem]) -> str:
    items = adoption + trust
    if any(i["status"] == "silent" for i in items):
        return "silent"
    if any(i["severity"] == "critical" for i in items):
        return "untrusted"
    if any(i["severity"] == "warning" for i in items):
        return "degraded"
    if not items:
        return "healthy"
    return "healthy"


def _health_rows(mem: Memory) -> list[dict[str, Any]]:
    try:
        return list(mem.store.dump_signal().get("memory_health") or [])
    except Exception:
        return []


def _grounded_ids(cfg: Config, *, limit: int) -> set[str]:
    ids: set[str] = set()
    for row in read_grounding_log(cfg.state_dir, limit=limit):
        score = row.get("used_score")
        rid = row.get("recall_id")
        if rid and isinstance(score, (int, float)) and float(score) >= 0.8:
            ids.add(str(rid))
    return ids


def _trust_items(
    cfg: Config,
    *,
    limit: int,
) -> tuple[list[DiagnosticItem], list[ActionItem], dict[str, Any]]:
    items: list[DiagnosticItem] = []
    actions: list[ActionItem] = []
    summary: dict[str, Any] = {
        "memory_health_rows": 0,
        "support_count_positive": 0,
        "grounded_memory_ids": 0,
    }
    try:
        mem = Memory(cfg)
    except Exception as exc:
        items.append(
            _item(
                id="store_unavailable",
                severity="warning",
                status="unknown",
                message="Memory store could not be opened for trust checks.",
                evidence={"error": str(exc)[:200]},
                action="Run memo doctor.",
            )
        )
        actions.append(_action("doctor", "memo doctor", "Store trust checks failed."))
        return items, actions, summary

    health_rows = _health_rows(mem)
    support_positive = sum(1 for r in health_rows if int(r.get("support_count") or 0) > 0)
    summary["memory_health_rows"] = len(health_rows)
    summary["support_count_positive"] = support_positive
    if len(health_rows) >= 20 and support_positive == 0:
        items.append(
            _item(
                id="support_count_starvation",
                severity="warning",
                status="degraded",
                message="No memory_health rows have support_count > 0.",
                evidence={
                    "memory_health_rows": len(health_rows),
                    "support_count_positive": support_positive,
                },
                action="Verify corroboration bump sites and signal export/import.",
            )
        )
        actions.append(
            _action(
                "verify_support_count",
                "uv run --no-sync pytest tests/test_support_count.py -v",
                "support_count is not accumulating.",
            )
        )

    grounded_prefixes = _grounded_ids(cfg, limit=limit)
    summary["grounded_memory_ids"] = len(grounded_prefixes)
    if grounded_prefixes:
        full_ids = mem.store.all_ids()
        resolved = [id_ for id_ in full_ids if id_[:8] in grounded_prefixes]
        rows = mem.store.get_batch(resolved)
        health = mem.store.get_health_batch(resolved)
        bad: list[dict[str, Any]] = []
        for row in rows:
            rid = str(row["id"])
            tags = list(row.get("tags") or [])
            extra = row.get("extra") or {}
            conf = float((health.get(rid) or {}).get("confidence", 1.0))
            reasons: list[str] = []
            if "_invalidated" in tags or extra.get("invalidated_reason"):
                reasons.append("invalidated")
            if extra.get("superseded_by"):
                reasons.append("superseded")
            if conf < 0.5:
                reasons.append("low_confidence")
            if reasons:
                bad.append(
                    {
                        "id": rid[:8],
                        "title": row.get("title"),
                        "confidence": conf,
                        "reasons": reasons,
                    }
                )
        if bad:
            severity = (
                "critical"
                if any(
                    "invalidated" in memory["reasons"] or "superseded" in memory["reasons"]
                    for memory in bad
                )
                else "warning"
            )
            items.append(
                _item(
                    id="untrusted_memories_grounded",
                    severity=severity,
                    status="untrusted",
                    message="Grounded recall used memories with low-trust markers.",
                    evidence={"count": len(bad), "memories": bad[:10]},
                    action="Update stale memories, run contradiction triage, or undo incorrect invalidations.",
                )
            )
            actions.append(
                _action(
                    "triage_untrusted",
                    "memo contradict triage",
                    "Grounded memories include invalidated, superseded, or low-confidence records.",
                )
            )

    return items, actions, summary


def build_report(cfg: Config, *, limit: int = 500) -> dict[str, Any]:
    """Build a read-only trust + adoption report."""
    adoption, adoption_actions = _adoption_items(cfg, limit=limit)
    trust, trust_actions, trust_summary = _trust_items(cfg, limit=limit)
    verdict = _derive_verdict(adoption, trust)
    return {
        "verdict": verdict,
        "adoption": adoption,
        "trust": trust,
        "actions": adoption_actions + trust_actions,
        "summary": {
            "limit": int(limit),
            "adoption_items": len(adoption),
            "trust_items": len(trust),
            **trust_summary,
        },
    }


def format_text_report(report: dict[str, Any]) -> str:
    """Render a compact human report."""
    lines = [
        "memo trust + adoption doctor",
        "",
        f"verdict: {report.get('verdict', 'unknown')}",
        "",
        "adoption",
    ]
    adoption = list(report.get("adoption") or [])
    if not adoption:
        lines.append("  - healthy: consumers are attributed and grounding is not degraded")
    for item in adoption:
        lines.append(f"  - {item['id']}: {item['status']} ({item['severity']})")
        lines.append(f"    {item['message']}")
        lines.append(f"    action: {item['action']}")

    lines.extend(["", "trust"])
    trust = list(report.get("trust") or [])
    if not trust:
        lines.append("  - no trust warnings in sampled data")
    for item in trust:
        lines.append(f"  - {item['id']}: {item['status']} ({item['severity']})")
        lines.append(f"    {item['message']}")
        lines.append(f"    action: {item['action']}")

    return "\n".join(lines)
