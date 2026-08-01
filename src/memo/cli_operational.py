"""CLI surfaces for evidence and operational continuity."""

from __future__ import annotations

import json
from dataclasses import asdict
from functools import wraps
from pathlib import Path
from typing import Any

import click

from memo.config import Config


def _with_memory(fn: Any) -> Any:
    @wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        from memo.memory import Memory

        memory = Memory(Config.from_env())
        try:
            return fn(memory, *args, **kwargs)
        finally:
            memory.close()

    return wrapper


def _json(value: Any) -> None:
    click.echo(json.dumps(value, ensure_ascii=False, indent=2, default=str))


@click.command(name="evidence")
@click.argument("question")
@click.option("-k", default=8, type=click.IntRange(1, 50))
@click.option("--max-chars", default=12_000, type=click.IntRange(256))
@click.option("--min-coverage", default=0.2, type=click.FloatRange(0.0, 1.0))
@click.option("--type", "type_", default=None)
@click.option("--as-of", default=None)
@_with_memory
def evidence_cmd(
    memory: Any,
    question: str,
    k: int,
    max_chars: int,
    min_coverage: float,
    type_: str | None,
    as_of: str | None,
) -> None:
    """Build a bounded EvidencePack or explicitly abstain."""
    _json(
        memory.evidence_pack(
            question,
            k=k,
            max_chars=max_chars,
            min_coverage=min_coverage,
            type_=type_,
            as_of=as_of,
        ).to_dict()
    )


@click.group(name="operational")
def operational_group() -> None:
    """Manage Memo's native continuity journal."""


@operational_group.command(name="state")
@click.option("--project", default=None)
@_with_memory
def operational_state(memory: Any, project: str | None) -> None:
    _json(memory.operational.state(project=project))


@operational_group.command(name="verify")
@_with_memory
def operational_verify(memory: Any) -> None:
    report = memory.operational.ledger.verify()
    _json(report)
    if not report["ok"]:
        raise click.ClickException("operational journal verification failed")


@operational_group.group(name="signal")
def signal_group() -> None:
    """Remember or inspect durable watcher markers."""


def _payload_object(raw: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise click.BadParameter("must be valid JSON", param_hint="--payload-json") from exc
    if not isinstance(payload, dict):
        raise click.BadParameter("must decode to a JSON object", param_hint="--payload-json")
    return payload


@signal_group.command(name="remember")
@click.option("--marker", required=True, help="Stable idempotency marker.")
@click.option("--epoch", default=0, type=click.IntRange(0), show_default=True)
@click.option("--fence", default="", help="Optional leadership fence token.")
@click.option("--payload-json", default="{}", help="Signal payload as a JSON object.")
@click.option("--actor-id", default="memo", show_default=True)
@_with_memory
def signal_remember(
    memory: Any,
    marker: str,
    epoch: int,
    fence: str,
    payload_json: str,
    actor_id: str,
) -> None:
    """Remember a durable idempotent signal and print its JSON record."""
    try:
        item = memory.operational.remember_signal(
            marker=marker,
            epoch=epoch,
            fence=fence,
            payload=_payload_object(payload_json),
            actor_id=actor_id,
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    _json(asdict(item))


@signal_group.command(name="list")
@click.option("--marker", default=None, help="Filter to one exact marker.")
@click.option("--min-epoch", default=None, type=click.IntRange(0))
@click.option("--limit", default=100, type=click.IntRange(1, 1000), show_default=True)
@_with_memory
def signal_list(
    memory: Any,
    marker: str | None,
    min_epoch: int | None,
    limit: int,
) -> None:
    """List durable watcher signals as a JSON envelope."""
    rows = memory.operational.list_signals(
        marker=marker,
        min_epoch=min_epoch,
        limit=limit,
    )
    _json({"signals": [asdict(item) for item in rows], "count": len(rows)})


@operational_group.group(name="focus")
def focus_group() -> None:
    """Set or clear the current project focus."""


@focus_group.command(name="set")
@click.argument("project")
@click.argument("summary")
@click.option("--actor", default="memo")
@_with_memory
def focus_set(memory: Any, project: str, summary: str, actor: str) -> None:
    from memo.contracts import ActorIdentity

    _json(
        asdict(
            memory.operational.set_focus(
                project=project,
                summary=summary,
                actor=ActorIdentity(actor_id=actor, actor_kind="agent"),
            )
        )
    )


@focus_group.command(name="clear")
@click.argument("project")
@click.option("--actor", default="memo")
@_with_memory
def focus_clear(memory: Any, project: str, actor: str) -> None:
    from memo.contracts import ActorIdentity

    _json(
        {
            "cleared": memory.operational.clear_focus(
                project,
                actor=ActorIdentity(actor_id=actor, actor_kind="agent"),
            )
        }
    )


@operational_group.group(name="handoff")
def handoff_group() -> None:
    """Create or consume agent handoffs."""


@handoff_group.command(name="create")
@click.argument("project")
@click.argument("summary")
@click.option("--from", "from_actor", required=True)
@click.option("--to", "to_actor", default="")
@_with_memory
def handoff_create(
    memory: Any,
    project: str,
    summary: str,
    from_actor: str,
    to_actor: str,
) -> None:
    _json(
        asdict(
            memory.operational.create_handoff(
                project=project,
                summary=summary,
                from_actor=from_actor,
                to_actor=to_actor,
            )
        )
    )


@handoff_group.command(name="consume")
@click.argument("id_")
@click.option("--actor", default="memo")
@_with_memory
def handoff_consume(memory: Any, id_: str, actor: str) -> None:
    _json({"consumed": memory.operational.consume_handoff(id_, actor_id=actor)})


@operational_group.group(name="attention")
def attention_group() -> None:
    """Create or acknowledge attention items."""


@attention_group.command(name="add")
@click.argument("project")
@click.argument("summary")
@click.option(
    "--severity",
    default="medium",
    type=click.Choice(["low", "medium", "high", "critical"]),
)
@_with_memory
def attention_add(memory: Any, project: str, summary: str, severity: str) -> None:
    _json(
        asdict(
            memory.operational.add_attention(
                project=project,
                summary=summary,
                severity=severity,
            )
        )
    )


@attention_group.command(name="ack")
@click.argument("id_")
@click.option("--actor", default="memo")
@_with_memory
def attention_ack(memory: Any, id_: str, actor: str) -> None:
    _json({"acknowledged": memory.operational.acknowledge_attention(id_, actor_id=actor)})


@operational_group.group(name="conflict")
def conflict_group() -> None:
    """Open or resolve write-freezing conflicts."""


@conflict_group.command(name="open")
@click.argument("topic")
@click.argument("summary")
@click.option("--freeze/--no-freeze", default=True)
@click.option("--evidence", "evidence_uris", multiple=True)
@_with_memory
def conflict_open(
    memory: Any,
    topic: str,
    summary: str,
    freeze: bool,
    evidence_uris: tuple[str, ...],
) -> None:
    _json(
        asdict(
            memory.operational.open_conflict(
                topic=topic,
                summary=summary,
                freeze_write=freeze,
                evidence_uris=list(evidence_uris),
            )
        )
    )


@conflict_group.command(name="resolve")
@click.argument("id_")
@click.argument("resolution")
@click.option("--actor", required=True)
@_with_memory
def conflict_resolve(memory: Any, id_: str, resolution: str, actor: str) -> None:
    from memo.contracts import ActorIdentity

    _json(
        {
            "resolved": memory.operational.resolve_conflict(
                id_,
                resolution=resolution,
                actor=ActorIdentity(actor_id=actor, actor_kind="human"),
            )
        }
    )


@operational_group.group(name="outcome")
def outcome_group() -> None:
    """Record task outcomes and update memory utility."""


@outcome_group.command(name="record")
@click.argument("task_id")
@click.argument("status", type=click.Choice(["success", "failure", "partial"]))
@click.option("--memory", "memory_ids", multiple=True, required=True)
@click.option("--artifact", "artifacts", multiple=True)
@click.option("--actor", default="memo")
@click.option("--idempotency-key", default="")
@_with_memory
def outcome_record(
    memory: Any,
    task_id: str,
    status: str,
    memory_ids: tuple[str, ...],
    artifacts: tuple[str, ...],
    actor: str,
    idempotency_key: str,
) -> None:
    _json(
        memory.record_task_outcome(
            task_id=task_id,
            status=status,
            memory_ids=list(memory_ids),
            artifacts=list(artifacts),
            actor_id=actor,
            idempotency_key=idempotency_key,
        )
    )


@operational_group.group(name="procedure")
def procedure_group() -> None:
    """Promote outcome-backed memories into reusable learnings."""


@procedure_group.command(name="candidates")
@click.option("--min-successes", default=2, type=click.IntRange(1))
@click.option("--min-utility", default=0.75, type=click.FloatRange(0.0, 1.0))
@click.option("--limit", default=50, type=click.IntRange(1, 500))
@_with_memory
def procedure_candidates(
    memory: Any,
    min_successes: int,
    min_utility: float,
    limit: int,
) -> None:
    _json(
        memory.procedure_candidates(
            min_successes=min_successes,
            min_utility=min_utility,
            limit=limit,
        )
    )


@procedure_group.command(name="promote")
@click.argument("title")
@click.option("--memory", "memory_ids", multiple=True, required=True)
@click.option(
    "--kind",
    type=click.Choice(["procedure", "failure_pattern"]),
    default="procedure",
)
@click.option("--content", default=None)
@click.option("--reason", default="outcome-backed promotion")
@click.option("--actor", default="memo")
@_with_memory
def procedure_promote(
    memory: Any,
    title: str,
    memory_ids: tuple[str, ...],
    kind: str,
    content: str | None,
    reason: str,
    actor: str,
) -> None:
    _json(
        memory.promote_learning(
            list(memory_ids),
            title=title,
            kind=kind,
            content=content,
            reason=reason,
            actor_id=actor,
        ).to_dict()
    )


__all__ = ["evidence_cmd", "operational_group"]


@click.command(name="migrate-independence")
@click.option("--write", is_flag=True, help="Apply the migration; default is dry-run.")
@click.option("--legacy-ledger", type=click.Path(path_type=Path), default=None)
@click.option("--config", "config_paths", multiple=True, type=click.Path(path_type=Path))
@_with_memory
def migrate_independence_cmd(
    memory: Any,
    write: bool,
    legacy_ledger: Path | None,
    config_paths: tuple[Path, ...],
) -> None:
    """Migrate legacy integration metadata into Memo-native contracts."""
    from memo.independence_migration import migrate_independence

    _json(
        migrate_independence(
            memory,
            write=write,
            legacy_ledger=legacy_ledger,
            config_paths=list(config_paths),
        )
    )


__all__.append("migrate_independence_cmd")
