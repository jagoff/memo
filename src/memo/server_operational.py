"""MCP tools for Memo-native evidence and operational continuity."""

from __future__ import annotations

from dataclasses import asdict
from typing import Annotated, Any

from pydantic import Field

from memo.server_annotations import READ_ONLY, WRITE, WRITE_IDEMPOTENT, annotated_tool

# Section -> the field its items are ordered by when the snapshot is trimmed.
_STATE_SECTION_ORDER = {
    "conflicts": "created_at",
    "handoffs": "created_at",
    "attention": "created_at",
    "outcomes": "recorded_at",
    "signals": "created_at",
}


def _bounded_state(state: dict[str, Any], *, limit: int) -> dict[str, Any]:
    """Trim each id-keyed section to its newest `limit` entries.

    `focus` is one row per project and stays whole; the growing sections are
    trimmed with their true sizes reported under `counts`, so a caller can see
    a backlog exists without paying to read all of it.
    """
    bounded = dict(state)
    counts: dict[str, int] = {}
    cap = max(0, limit)
    for section, order_key in _STATE_SECTION_ORDER.items():
        items = state.get(section)
        if not isinstance(items, dict):
            continue
        counts[section] = len(items)
        if len(items) <= cap:
            continue
        newest = sorted(
            items.items(),
            key=lambda kv: str((kv[1] or {}).get(order_key) or ""),
            reverse=True,
        )[:cap]
        bounded[section] = dict(newest)
    bounded["counts"] = counts
    bounded["limit"] = cap
    return bounded


def _register_evidence_and_state_tools(server: Any, memory: Any) -> None:
    @annotated_tool(server, **READ_ONLY)
    def memo_evidence_pack(
        question: Annotated[
            str,
            Field(description="Natural-language question to collect cited evidence for."),
        ],
        k: Annotated[
            int,
            Field(description="Maximum candidate memories to retrieve (clamped to 1-50)."),
        ] = 8,
        max_chars: Annotated[
            int,
            Field(description="Character budget for the packed evidence text."),
        ] = 12_000,
        min_coverage: Annotated[
            float,
            Field(
                description="Minimum retrieval coverage (0-1) required to answer; "
                "below it the pack abstains explicitly."
            ),
        ] = 0.2,
        type: Annotated[
            str | None,
            Field(
                description="Restrict evidence to one memory type "
                "(e.g. 'decision', 'fact'); None searches every type."
            ),
        ] = None,
        as_of: Annotated[
            str | None,
            Field(
                description="ISO date/datetime for time-travel: only memories "
                "valid at that moment are considered."
            ),
        ] = None,
    ) -> dict[str, Any]:
        """Return bounded, cited evidence for a question, or an explicit abstention.

        Read-only, no side effects. Every snippet carries a memo:// citation;
        when retrieval coverage stays under `min_coverage` the result is an
        explicit abstention, never a fabricated answer.
        """
        out = memory.evidence_pack(
            question,
            k=max(1, min(k, 50)),
            max_chars=max_chars,
            min_coverage=min_coverage,
            type_=type,
            as_of=as_of,
        ).to_dict()

        # Suppress items already emitted into this session's window. `items`
        # carry the emitted text under `snippet` (EvidenceItem.to_dict()), not
        # `body`. `confidence`/`coverage`/`token_estimate` above were computed
        # from the full (pre-suppression) items -- same as memo_ask's answer,
        # they describe what evidence_pack actually used, not what survives
        # here as citations.
        items = out.get("items")
        if isinstance(items, list):
            from memo.server_common import apply_ledger

            kept, ledger_extra = apply_ledger(
                memory,
                "memo_evidence_pack",
                items,
                text_of=lambda h: str(h.get("snippet") or ""),
            )
            out["items"] = kept
            out.update(ledger_extra)

        return out

    @annotated_tool(server, **READ_ONLY)
    def memo_operational_state(
        project: Annotated[
            str | None,
            Field(description="Project tag to scope the state to; None returns every project."),
        ] = None,
        include_closed: Annotated[
            bool,
            Field(
                description="Also return settled history: resolved conflicts, consumed "
                "handoffs, and acknowledged attention items. Off by default — that "
                "history only grows and can exceed the response budget."
            ),
        ] = False,
        limit: Annotated[
            int,
            Field(
                description="Newest items to return per section (conflicts, handoffs, "
                "attention, outcomes, signals). True totals come back under `counts`."
            ),
        ] = 20,
    ) -> dict[str, Any]:
        """Read current focus, handoffs, attention items, conflicts, and outcomes.

        Read-only snapshot of the operational journal's current state. Returns
        only what is still open unless ``include_closed`` is set, and only the
        newest ``limit`` entries per section — open items awaiting human triage
        accumulate faster than they settle, so an unbounded snapshot eventually
        costs more context than the memories it describes.
        """
        return _bounded_state(
            memory.operational.state(project=project, include_closed=include_closed),
            limit=limit,
        )

    @annotated_tool(server, **READ_ONLY)
    def memo_federation_preview(
        principal: Annotated[
            str,
            Field(
                description="Principal (device id) of the intended bundle recipient "
                "whose ACL-visible memories are previewed."
            ),
        ],
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

    @annotated_tool(server, **WRITE_IDEMPOTENT)
    def memo_signal_remember(
        marker: Annotated[
            str, Field(description="Stable idempotency marker for the watcher signal.")
        ],
        epoch: Annotated[
            int, Field(description="Monotonic watcher epoch; stale epochs are rejected.")
        ] = 0,
        fence: Annotated[str, Field(description="Optional leadership fence token.")] = "",
        payload: Annotated[
            dict[str, Any] | None, Field(description="Structured signal payload.")
        ] = None,
        actor_id: Annotated[str, Field(description="Agent writing the signal.")] = "memo",
    ) -> dict[str, Any]:
        """Remember a durable, idempotent operational watcher marker."""
        return asdict(
            memory.operational.remember_signal(
                marker=marker, epoch=epoch, fence=fence, payload=payload, actor_id=actor_id
            )
        )

    @annotated_tool(server, **READ_ONLY)
    def memo_signal_list(
        marker: Annotated[str | None, Field(description="Filter to one marker.")] = None,
        min_epoch: Annotated[
            int | None, Field(description="Return signals at or above this epoch.")
        ] = None,
        limit: Annotated[int, Field(description="Maximum number of markers.")] = 100,
    ) -> dict[str, Any]:
        """List durable watcher markers, newest epoch first."""
        rows = [
            asdict(item)
            for item in memory.operational.list_signals(
                marker=marker, min_epoch=min_epoch, limit=limit
            )
        ]
        return {"signals": rows, "count": len(rows)}


def _register_focus_and_handoff_tools(server: Any, memory: Any) -> None:
    @annotated_tool(server, **WRITE_IDEMPOTENT)
    def memo_focus_set(
        project: Annotated[
            str,
            Field(description="Project tag the focus belongs to."),
        ],
        summary: Annotated[
            str,
            Field(description="One-line description of the work currently in focus."),
        ],
        actor_id: Annotated[
            str,
            Field(description="Identifier of the agent setting the focus (journaled)."),
        ] = "memo",
    ) -> dict[str, Any]:
        """Set the current focus for a project.

        Replaces the project's previous focus; the change is journaled with
        the acting agent's identity. Safe to repeat with the same summary.
        """
        from memo.contracts import ActorIdentity

        return asdict(
            memory.operational.set_focus(
                project=project,
                summary=summary,
                actor=ActorIdentity(actor_id=actor_id, actor_kind="agent"),
            )
        )

    @annotated_tool(server, **WRITE_IDEMPOTENT)
    def memo_focus_clear(
        project: Annotated[
            str,
            Field(description="Project tag whose focus should be cleared."),
        ],
        actor_id: Annotated[
            str,
            Field(description="Identifier of the agent clearing the focus (journaled)."),
        ] = "memo",
    ) -> dict[str, Any]:
        """Clear a project's current focus.

        Idempotent: returns {'cleared': false} when no focus was set.
        """
        from memo.contracts import ActorIdentity

        return {
            "cleared": memory.operational.clear_focus(
                project,
                actor=ActorIdentity(actor_id=actor_id, actor_kind="agent"),
            )
        }

    @annotated_tool(server, **WRITE)
    def memo_handoff_create(
        project: Annotated[
            str,
            Field(description="Project tag the handoff belongs to."),
        ],
        summary: Annotated[
            str,
            Field(description="What the receiving agent needs to know to continue the work."),
        ],
        from_actor: Annotated[
            str,
            Field(description="Identifier of the agent creating the handoff."),
        ],
        to_actor: Annotated[
            str,
            Field(
                description="Target agent identifier; empty string leaves the "
                "handoff open to any agent."
            ),
        ] = "",
    ) -> dict[str, Any]:
        """Create a durable handoff for another agent or session.

        Writes a journaled handoff record that surfaces in
        memo_operational_state until some agent consumes it.
        """
        return asdict(
            memory.operational.create_handoff(
                project=project,
                summary=summary,
                from_actor=from_actor,
                to_actor=to_actor,
            )
        )

    @annotated_tool(server, **WRITE_IDEMPOTENT)
    def memo_handoff_consume(
        id: Annotated[
            str,
            Field(
                description="Id of the handoff to mark consumed "
                "(from memo_operational_state or memo_handoff_create)."
            ),
        ],
        actor_id: Annotated[
            str,
            Field(description="Identifier of the consuming agent (journaled)."),
        ] = "memo",
    ) -> dict[str, Any]:
        """Mark a handoff consumed so it stops surfacing to later agents.

        Idempotent: returns {'consumed': false} when the id is unknown or the
        handoff was already consumed; true only on the first consume.
        """
        return {"consumed": memory.operational.consume_handoff(id, actor_id=actor_id)}


def _register_attention_and_conflict_tools(server: Any, memory: Any) -> None:
    @annotated_tool(server, **WRITE)
    def memo_attention_add(
        project: Annotated[
            str,
            Field(description="Project tag the attention item belongs to."),
        ],
        summary: Annotated[
            str,
            Field(description="The item later agents must see before working on the project."),
        ],
        severity: Annotated[
            str,
            Field(description="One of 'low', 'medium', 'high', 'critical'."),
        ] = "medium",
    ) -> dict[str, Any]:
        """Add an item that must be surfaced to later agents.

        Writes a journaled attention record; severities outside
        low|medium|high|critical are rejected.
        """
        return asdict(
            memory.operational.add_attention(
                project=project,
                summary=summary,
                severity=severity,
            )
        )

    @annotated_tool(server, **WRITE_IDEMPOTENT)
    def memo_attention_ack(
        id: Annotated[
            str,
            Field(description="Id of the attention item to acknowledge."),
        ],
        actor_id: Annotated[
            str,
            Field(description="Identifier of the acknowledging agent (journaled)."),
        ] = "memo",
    ) -> dict[str, Any]:
        """Acknowledge an attention item so it stops being surfaced.

        Idempotent: returns {'acknowledged': false} when the id is unknown or
        already acknowledged; true only on the first acknowledgement.
        """
        return {
            "acknowledged": memory.operational.acknowledge_attention(
                id,
                actor_id=actor_id,
            )
        }

    @annotated_tool(server, **WRITE)
    def memo_conflict_open(
        topic: Annotated[
            str,
            Field(description="Short stable label for the disputed subject."),
        ],
        summary: Annotated[
            str,
            Field(description="Description of the conflicting claims or evidence."),
        ],
        freeze_write: Annotated[
            bool,
            Field(
                description="When true, mark the topic write-frozen until a "
                "human resolves the conflict."
            ),
        ] = True,
        evidence_uris: Annotated[
            list[str] | None,
            Field(description="memo:// URIs of the evidence supporting each side."),
        ] = None,
    ) -> dict[str, Any]:
        """Open a local, auditable reality conflict.

        Writes a 'detected' conflict record to the hash-chained journal.
        Resolution is human-only — memo_conflict_resolve reports the CLI
        command a human must run.
        """
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
        id: Annotated[
            str,
            Field(description="Id of the conflict to resolve."),
        ],
    ) -> dict[str, Any]:
        """Report the local human action required to resolve a conflict.

        Read-only: never resolves anything itself — it returns the
        `memo operational conflict resolve` command a human must run locally.
        """
        return {
            "resolved": False,
            "id": id,
            "requires_human_cli": True,
            "command": f"memo operational conflict resolve {id} '<resolution>' --actor <human>",
        }


def _register_outcome_tools(server: Any, memory: Any) -> None:
    @annotated_tool(server, **WRITE_IDEMPOTENT)
    def memo_outcome_record(
        task_id: Annotated[
            str,
            Field(description="Stable identifier of the task whose outcome is recorded."),
        ],
        status: Annotated[
            str,
            Field(description="One of 'success', 'failure', 'partial'."),
        ],
        memory_ids: Annotated[
            list[str],
            Field(description="Ids of the memories that were recalled or used for the task."),
        ],
        idempotency_key: Annotated[
            str,
            Field(
                description="Caller-chosen key that makes retries safe: the same "
                "key replays the stored outcome instead of double-counting; "
                "reusing it with a different payload is rejected."
            ),
        ],
        actor_id: Annotated[
            str,
            Field(description="Identifier of the reporting agent (journaled)."),
        ] = "memo",
        artifacts: Annotated[
            list[str] | None,
            Field(description="Optional URIs or paths of artifacts the task produced."),
        ] = None,
    ) -> dict[str, Any]:
        """Record whether recalled memories helped a task succeed.

        Feeds success/failure back into each cited memory's outcome stats —
        the signal behind procedure promotion. Idempotent per
        `idempotency_key`.
        """
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
        min_successes: Annotated[
            int,
            Field(description="Minimum successful outcomes a memory needs (floor 1)."),
        ] = 2,
        min_utility: Annotated[
            float,
            Field(description="Minimum outcome utility score, clamped to 0-1."),
        ] = 0.75,
        limit: Annotated[
            int,
            Field(description="Maximum candidates returned (clamped to 1-500)."),
        ] = 50,
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
        memory_ids: Annotated[
            list[str],
            Field(
                description="Source memories with recorded outcome evidence; "
                "every id must exist and qualify."
            ),
        ],
        title: Annotated[
            str,
            Field(description="Title of the new procedure or failure-pattern memory."),
        ],
        kind: Annotated[
            str,
            Field(
                description="'procedure' (needs >=2 successes and utility >=0.75 "
                "per source) or 'failure_pattern' (needs >=2 failures at >=50% "
                "failure rate)."
            ),
        ] = "procedure",
        content: Annotated[
            str | None,
            Field(
                description="Explicit body for the new memory; None concatenates the source bodies."
            ),
        ] = None,
        reason: Annotated[
            str,
            Field(description="Provenance note recorded with the promotion."),
        ] = "outcome-backed promotion",
        actor_id: Annotated[
            str,
            Field(description="Identifier of the promoting agent (journaled)."),
        ] = "memo",
    ) -> dict[str, Any]:
        """Promote grounded memories into a reusable procedure/failure pattern.

        Creates a new durable memory citing the sources; rejects sources whose
        outcome stats don't meet the `kind` threshold.
        """
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
