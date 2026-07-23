"""MCP tools for Memo-native evidence and operational continuity."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from memo.server_annotations import READ_ONLY, WRITE, WRITE_IDEMPOTENT, annotated_tool


def _register_evidence_and_state_tools(server: Any, memory: Any) -> None:
    @annotated_tool(server, **READ_ONLY)
    def memo_evidence_pack(
        question: str,
        k: int = 8,
        max_chars: int = 12_000,
        min_coverage: float = 0.2,
        type: str | None = None,
        as_of: str | None = None,
    ) -> dict[str, Any]:
        """Return bounded, cited evidence or an explicit abstention."""
        return memory.evidence_pack(
            question,
            k=max(1, min(k, 50)),
            max_chars=max_chars,
            min_coverage=min_coverage,
            type_=type,
            as_of=as_of,
        ).to_dict()

    @annotated_tool(server, **READ_ONLY)
    def memo_operational_state(project: str | None = None) -> dict[str, Any]:
        """Read focus, handoffs, attention, conflicts, and outcomes."""
        return memory.operational.state(project=project)

    @annotated_tool(server, **READ_ONLY)
    def memo_federation_preview(
        principal: str,
    ) -> dict[str, Any]:
        """Preview the exact memories an ACL would allow into a signed bundle."""
        if principal.strip() == memory.cfg.device_id:
            return {
                "error": "owner_preview_requires_local_cli",
                "message": "owner-wide preview is available only from the local CLI",
            }
        return memory.federation.preview(
            principal=principal,
            owner_principal=memory.cfg.device_id,
        )

    @annotated_tool(server, **READ_ONLY)
    def memo_journal_verify() -> dict[str, Any]:
        """Verify every native operational hash chain."""
        return memory.operational.ledger.verify()


def _register_focus_and_handoff_tools(server: Any, memory: Any) -> None:
    @annotated_tool(server, **WRITE_IDEMPOTENT)
    def memo_focus_set(
        project: str,
        summary: str,
        actor_id: str = "memo",
    ) -> dict[str, Any]:
        """Set the current focus for a project."""
        from memo.contracts import ActorIdentity

        return asdict(
            memory.operational.set_focus(
                project=project,
                summary=summary,
                actor=ActorIdentity(actor_id=actor_id, actor_kind="agent"),
            )
        )

    @annotated_tool(server, **WRITE_IDEMPOTENT)
    def memo_focus_clear(project: str, actor_id: str = "memo") -> dict[str, Any]:
        """Clear a project's current focus."""
        from memo.contracts import ActorIdentity

        return {
            "cleared": memory.operational.clear_focus(
                project,
                actor=ActorIdentity(actor_id=actor_id, actor_kind="agent"),
            )
        }

    @annotated_tool(server, **WRITE)
    def memo_handoff_create(
        project: str,
        summary: str,
        from_actor: str,
        to_actor: str = "",
    ) -> dict[str, Any]:
        """Create a durable handoff for another agent or session."""
        return asdict(
            memory.operational.create_handoff(
                project=project,
                summary=summary,
                from_actor=from_actor,
                to_actor=to_actor,
            )
        )

    @annotated_tool(server, **WRITE_IDEMPOTENT)
    def memo_handoff_consume(id: str, actor_id: str = "memo") -> dict[str, Any]:
        """Mark a handoff consumed."""
        return {"consumed": memory.operational.consume_handoff(id, actor_id=actor_id)}


def _register_attention_and_conflict_tools(server: Any, memory: Any) -> None:
    @annotated_tool(server, **WRITE)
    def memo_attention_add(
        project: str,
        summary: str,
        severity: str = "medium",
    ) -> dict[str, Any]:
        """Add an item that must be surfaced to later agents."""
        return asdict(
            memory.operational.add_attention(
                project=project,
                summary=summary,
                severity=severity,
            )
        )

    @annotated_tool(server, **WRITE_IDEMPOTENT)
    def memo_attention_ack(id: str, actor_id: str = "memo") -> dict[str, Any]:
        """Acknowledge an attention item."""
        return {
            "acknowledged": memory.operational.acknowledge_attention(
                id,
                actor_id=actor_id,
            )
        }

    @annotated_tool(server, **WRITE)
    def memo_conflict_open(
        topic: str,
        summary: str,
        freeze_write: bool = True,
        evidence_uris: list[str] | None = None,
    ) -> dict[str, Any]:
        """Open a local, auditable reality conflict."""
        return asdict(
            memory.operational.open_conflict(
                topic=topic,
                summary=summary,
                freeze_write=freeze_write,
                evidence_uris=evidence_uris,
            )
        )

    @annotated_tool(server, **READ_ONLY)
    def memo_conflict_resolve(
        id: str,
    ) -> dict[str, Any]:
        """Report the local human action required to resolve a conflict."""
        return {
            "resolved": False,
            "id": id,
            "requires_human_cli": True,
            "command": f"memo operational conflict resolve {id} '<resolution>' --actor <human>",
        }


def _register_outcome_tools(server: Any, memory: Any) -> None:
    @annotated_tool(server, **WRITE_IDEMPOTENT)
    def memo_outcome_record(
        task_id: str,
        status: str,
        memory_ids: list[str],
        idempotency_key: str,
        actor_id: str = "memo",
        artifacts: list[str] | None = None,
    ) -> dict[str, Any]:
        """Record whether recalled memories helped a task succeed."""
        return memory.record_task_outcome(
            task_id=task_id,
            status=status,
            memory_ids=memory_ids,
            actor_id=actor_id,
            artifacts=artifacts,
            idempotency_key=idempotency_key,
        )

    @annotated_tool(server, **READ_ONLY)
    def memo_procedure_candidates(
        min_successes: int = 2,
        min_utility: float = 0.75,
        limit: int = 50,
    ) -> dict[str, Any]:
        """List outcome-backed memories ready for procedural promotion."""
        candidates = memory.procedure_candidates(
            min_successes=max(1, min_successes),
            min_utility=min(1.0, max(0.0, min_utility)),
            limit=max(1, min(limit, 500)),
        )
        return {"candidates": candidates, "count": len(candidates)}

    @annotated_tool(server, **WRITE)
    def memo_procedure_promote(
        memory_ids: list[str],
        title: str,
        kind: str = "procedure",
        content: str | None = None,
        reason: str = "outcome-backed promotion",
        actor_id: str = "memo",
    ) -> dict[str, Any]:
        """Promote grounded memories into a reusable procedure/failure pattern."""
        return memory.promote_learning(
            memory_ids,
            title=title,
            kind=kind,
            content=content,
            reason=reason,
            actor_id=actor_id,
        ).to_dict()


def register(server: Any, memory: Any) -> None:
    _register_evidence_and_state_tools(server, memory)
    _register_focus_and_handoff_tools(server, memory)
    _register_attention_and_conflict_tools(server, memory)
    _register_outcome_tools(server, memory)


__all__ = ["register"]
