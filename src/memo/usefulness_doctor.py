"""Read-only trust + adoption diagnostics for `memo usefulness doctor`."""

from __future__ import annotations

from typing import Any

from memo.config import Config
from memo.dashboard import (
    consult_breakdown,
    consumer_label,
    read_recall_log,
    recall_health,
)

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


def build_report(cfg: Config, *, limit: int = 500) -> dict[str, Any]:
    """Build a read-only trust + adoption report."""
    adoption, actions = _adoption_items(cfg, limit=limit)
    trust: list[DiagnosticItem] = []
    verdict = _derive_verdict(adoption, trust)
    return {
        "verdict": verdict,
        "adoption": adoption,
        "trust": trust,
        "actions": actions,
        "summary": {
            "limit": int(limit),
            "adoption_items": len(adoption),
            "trust_items": len(trust),
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
