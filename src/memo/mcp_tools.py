"""Declarative MCP tool registry for memo.

Uses a ToolSpec/Param pattern so new tools can be added as compact declarative
specs instead of boilerplate FastMCP server_*.py files. Existing server_*.py
modules stay unchanged and are NOT being migrated here — porting the ~160
tools already living in server_*.py was considered and explicitly deferred
indefinitely (housekeeping audit, 2026-08-07); this module is only the home
for *new* tools going forward.

Usage
-----
Add a new tool:
  1. Write a handler function: ``def _handle_my_tool(memory, args) -> dict``.
  2. Add a ``ToolSpec(name, description, params, handler)`` to ``_TOOL_SPECS``.
  3. Nothing else — ``register_all(server, memory)`` picks it up automatically.

This module is a leaf: it imports nothing from FastMCP at module level so it
can be imported in tests without the MCP runtime.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from memo.memory import Memory
from memo.server_annotations import READ_ONLY, WRITE_IDEMPOTENT

__all__ = [
    "Param",
    "ToolSpec",
    "register_all",
    "schema_from_spec",
]


# ---------------------------------------------------------------------------
# Declarative tool infrastructure
# ---------------------------------------------------------------------------


class _Missing:
    _instance: _Missing | None = None

    def __new__(cls) -> _Missing:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "MISSING"


MISSING = _Missing()

_PY_TYPE: dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "object": dict,
    "array": list,
}


@dataclass(frozen=True)
class Param:
    """One MCP tool parameter.

    ``default is MISSING`` marks a required parameter. ``nullable`` widens
    the JSON type to ``[type, "null"]``. ``item_type`` is the element type
    for array params.
    """

    name: str
    json_type: str
    default: Any = MISSING
    nullable: bool = False
    item_type: str = "string"
    description: str = ""

    @property
    def required(self) -> bool:
        return self.default is MISSING

    def annotation(self) -> Any:
        base: Any = list[str] if self.json_type == "array" else _PY_TYPE[self.json_type]
        return (base | None) if self.nullable else base

    def json_schema(self) -> dict[str, Any]:
        type_field: Any = [self.json_type, "null"] if self.nullable else self.json_type
        schema: dict[str, Any] = {"type": type_field}
        if self.json_type == "array":
            schema["items"] = {"type": self.item_type}
        if self.default is not MISSING:
            schema["default"] = self.default
        if self.description:
            schema["description"] = self.description
        return schema


@dataclass(frozen=True)
class ToolSpec:
    """One MCP tool: identity, schema, and dict-handler."""

    name: str
    description: str
    params: tuple[Param, ...]
    handler: Callable[[Memory, dict[str, Any]], Any]
    annotations: dict[str, Any] | None = None


def schema_from_spec(spec: ToolSpec) -> dict[str, Any]:
    """Build the JSON Schema advertised in ``tools/list`` for ``spec``."""
    properties = {p.name: p.json_schema() for p in spec.params}
    schema: dict[str, Any] = {"type": "object", "properties": properties}
    required = [p.name for p in spec.params if p.required]
    if required:
        schema["required"] = required
    return schema


def _signature_from_spec(spec: ToolSpec) -> tuple[inspect.Signature, dict[str, Any]]:
    """Build an inspect.Signature + annotations dict for FastMCP wrapper registration."""
    parameters: list[inspect.Parameter] = []
    annotations: dict[str, Any] = {}
    ordered = sorted(spec.params, key=lambda p: not p.required)
    for param in ordered:
        annotations[param.name] = param.annotation()
        default = (
            inspect.Parameter.empty
            if param.required
            else (None if param.default is None else param.default)
        )
        parameters.append(
            inspect.Parameter(
                param.name,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                default=default,
                annotation=param.annotation(),
            )
        )
    annotations["return"] = dict
    return inspect.Signature(parameters, return_annotation=dict), annotations


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------


def _handle_entity_search(memory: Memory, args: dict[str, Any]) -> dict[str, Any]:
    """Entity-aware search: extracts entities from query, boosts matching results."""
    query = str(args.get("query") or "").strip()
    limit = int(args.get("limit") or 8)
    mode = str(args.get("mode") or "hybrid")
    if not query:
        return {"error": "query is required", "results": []}
    # Enable entity retrieval for just this call via a kwarg — never mutate the
    # global env (concurrent FastMCP threads would clobber each other's flag).
    results = memory.search(query, limit=limit, mode=mode, recency=True, entity_boost=True)
    return {
        "query": query,
        "results": [r.to_dict() for r in results],
        "count": len(results),
        "entity_boost": True,
    }


def _handle_feedback_implicit(memory: Memory, args: dict[str, Any]) -> dict[str, Any]:
    """Record implicit feedback (click/ignore) without requiring a thumbs up/down decision."""
    source_id = str(args.get("source_id") or "").strip()
    query = str(args.get("query") or "").strip()
    signal = str(args.get("signal") or "click").strip().lower()
    if signal not in {"click", "ignore"}:
        signal = "click"
    if not source_id or not query:
        return {"error": "source_id and query are required"}
    return memory.feedback_record(source_id, query_text=query, rating=signal)


def _handle_version(memory: Memory, args: dict[str, Any]) -> dict[str, Any]:
    """Get memo version info — version string and backend protocol version."""
    from memo import __version__ as _memo_version
    from memo.memory import NATIVE_BACKEND_PROTOCOL_VERSION

    return {
        "version": _memo_version,
        "backend_protocol_version": NATIVE_BACKEND_PROTOCOL_VERSION,
    }


def _handle_event_bus_publish(memory: Memory, args: dict[str, Any]) -> dict[str, Any]:
    """Publish one `agent` event to the shared local event journal (multi-agent
    sync). Agent-id is this memory's own; the event lands in `memo events list`
    and is visible to other local agents via memo_event_poll."""
    from memo.agent_event_bus import AgentEventBus

    bus = AgentEventBus(memory.cfg.state_dir, agent_id="mcp")
    event = bus.publish(args["event_type"], args.get("data") or {})
    return {
        "published": True,
        "event": {
            "event_type": event.event_type,
            "agent_id": event.agent_id,
            "data": event.data,
            "timestamp": event.timestamp,
        },
    }


def _handle_event_poll(memory: Memory, args: dict[str, Any]) -> dict[str, Any]:
    """Read `agent` events written by OTHER local agents since last poll.

    Idempotent: each event id is delivered once per session and excluded from
    later polls. Empty list means no new cross-agent activity."""
    from memo.agent_event_bus import AgentEventBus

    bus = AgentEventBus(memory.cfg.state_dir, agent_id="mcp")
    events = bus.poll_new_events()
    return {
        "events": [
            {
                "event_type": e.event_type,
                "agent_id": e.agent_id,
                "data": e.data,
                "timestamp": e.timestamp,
            }
            for e in events
        ],
        "count": len(events),
    }


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

_TOOL_SPECS: tuple[ToolSpec, ...] = (
    ToolSpec(
        name="memo_entity_search",
        description=(
            "Search memories with entity-aware boosting. Extracts named entities "
            "(persons, technologies, projects) from the query and boosts results "
            "whose stored entities overlap. Useful for 'what do I know about "
            "Alice and React?' style queries."
        ),
        params=(
            Param("query", "string", description="Search query text"),
            Param("limit", "integer", default=8, description="Max results"),
            Param(
                "mode", "string", default="hybrid", description="Search mode: hybrid, vec, or bm25"
            ),
        ),
        handler=_handle_entity_search,
        annotations=READ_ONLY,
    ),
    ToolSpec(
        name="memo_version",
        description=(
            "Get memo version info — the package version and the backend protocol version. "
            "Use this to check which version of memo is running."
        ),
        params=(),
        handler=_handle_version,
        annotations=READ_ONLY,
    ),
    ToolSpec(
        name="memo_event_bus_publish",
        description=(
            "Publish a local multi-agent event to memo's shared event journal. "
            "Other agents (or their own memo_event_poll calls) can read it, and "
            "it appears in `memo events list`. Use for cross-agent coordination "
            "signals (e.g. 'refreshed the shared index', 'completed migration'). "
            "Gated by MEMO_EVENT_BUS_ENABLED (default on)."
        ),
        params=(
            Param("event_type", "string", description="Event type, e.g. 'sync.completed'"),
            Param(
                "data",
                "object",
                default=None,
                nullable=True,
                description="Optional structured payload for the event",
            ),
        ),
        handler=_handle_event_bus_publish,
        annotations=WRITE_IDEMPOTENT,
    ),
    ToolSpec(
        name="memo_event_poll",
        description=(
            "Read `agent` events published by OTHER local agents since the last "
            "poll in this process. Idempotent — each event is delivered once. "
            "Empty list means no new cross-agent activity."
        ),
        params=(),
        handler=_handle_event_poll,
        annotations=READ_ONLY,
    ),
)


def _register_spec(server: Any, memory: Memory, spec: ToolSpec) -> None:
    sig, annotations = _signature_from_spec(spec)

    def _make_wrapper(s: ToolSpec, sig: Any, annotations: dict[str, Any]) -> Callable[..., Any]:
        def _wrapper(**kwargs: Any) -> Any:
            return s.handler(memory, kwargs)

        _wrapper.__name__ = s.name
        _wrapper.__doc__ = s.description
        _wrapper.__signature__ = sig  # type: ignore[attr-defined]
        _wrapper.__annotations__ = annotations
        return _wrapper

    wrapper = _make_wrapper(spec, sig, annotations)
    if spec.annotations:
        from memo.server_annotations import annotated_tool

        annotated_tool(server, **spec.annotations)(wrapper)
    else:
        server.tool()(wrapper)


def register_version(server: Any, memory: Memory) -> None:
    """Register only memo_version — available in all profiles including core."""
    spec = next(s for s in _TOOL_SPECS if s.name == "memo_version")
    _register_spec(server, memory, spec)


def register_all(server: Any, memory: Memory) -> None:
    """Register all tools in _TOOL_SPECS with the FastMCP server.

    Each tool is wrapped in a typed function whose signature FastMCP can
    introspect, so the advertised JSON schema matches schema_from_spec.
    """
    for spec in _TOOL_SPECS:
        _register_spec(server, memory, spec)
