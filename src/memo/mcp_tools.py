"""Declarative MCP tool registry for memo.

Ports the ToolSpec/Param pattern from memflow so new tools can be added
as compact declarative specs instead of boilerplate FastMCP server_*.py
files. Existing server_*.py modules stay unchanged — this module is the
home for *new* tools going forward and a migration target for the old ones.

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
from memo.server_annotations import READ_ONLY, WRITE

__all__ = [
    "Param",
    "ToolSpec",
    "coerce_args",
    "register_all",
    "schema_from_spec",
]


# ---------------------------------------------------------------------------
# Infrastructure (adapted from memflow/mcp_tools.py)
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


def coerce_args(spec: ToolSpec, raw: dict[str, Any]) -> dict[str, Any]:
    """Coerce a raw arguments dict into typed kwargs."""
    kwargs: dict[str, Any] = {}
    for param in spec.params:
        if param.name not in raw or raw[param.name] is None:
            if param.required:
                raise ValueError(
                    f"required parameter '{param.name}' missing or None for tool '{spec.name}'"
                )
            if param.default is not MISSING:
                kwargs[param.name] = None if param.default is None else param.default
            continue
        kwargs[param.name] = _coerce_value(param, raw[param.name])
    return kwargs


def _coerce_value(param: Param, value: Any) -> Any:
    if param.json_type == "string":
        return str(value)
    if param.json_type == "integer":
        try:
            return int(value)
        except (TypeError, ValueError):
            return param.default if param.default is not MISSING else 0
    if param.json_type == "number":
        try:
            return float(value)
        except (TypeError, ValueError):
            return param.default if param.default is not MISSING else 0.0
    if param.json_type == "boolean":
        return bool(value)
    if param.json_type == "array":
        if isinstance(value, str):
            return [value]
        return list(value) if isinstance(value, (list, tuple)) else value
    return value


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
        name="memo_feedback_implicit",
        description=(
            "Record an implicit feedback signal (click or ignore) on a memory "
            "for a given query. Use 'click' when the user viewed/used this result; "
            "'ignore' when they skipped it. These are softer signals than "
            "memo_feedback_record's thumbs_up/thumbs_down."
        ),
        params=(
            Param("source_id", "string", description="Memory id or unique prefix"),
            Param("query", "string", description="Query text this feedback applies to"),
            Param(
                "signal",
                "string",
                default="click",
                description="'click' (implicit positive) or 'ignore' (implicit negative)",
            ),
        ),
        handler=_handle_feedback_implicit,
        annotations=WRITE,
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
