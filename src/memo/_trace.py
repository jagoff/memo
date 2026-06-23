"""Cross-backend trace propagation shim.

Re-exports the shared trace contextvar from ``consciousness-contracts`` so the
MCP server (which reads the ``x-synapse-trace-id`` header) and the write path
(which stamps ``synapse_trace_id`` onto saved memories) converge on one trace
id per request. When the optional contracts package is absent (CI / clean
installs) this degrades to inert no-ops: ``current_trace()`` returns ``""`` and
``trace_scope`` is a transparent context manager, so the env-var path still
works on its own.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

try:
    from consciousness_contracts import (  # type: ignore[assignment]
        TRACE_HEADER,
        current_trace,
        trace_scope,
    )

    HAS_TRACE_CONTEXT = True
except ImportError:  # graceful degradation — contracts is optional
    HAS_TRACE_CONTEXT = False
    TRACE_HEADER = "x-synapse-trace-id"

    def current_trace() -> str:  # type: ignore[misc]
        return ""

    @contextmanager
    def trace_scope(trace_id: str) -> Iterator[str]:  # type: ignore[misc]
        yield (trace_id or "").strip()


__all__ = ["HAS_TRACE_CONTEXT", "TRACE_HEADER", "current_trace", "trace_scope"]
