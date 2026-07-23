"""Process-local memo trace propagation.

HTTP/MCP requests and direct CLI calls share one stdlib context variable. A
legacy environment value is accepted only as a read-time migration alias;
Memo emits and documents the native name exclusively.
"""

from __future__ import annotations

import contextvars
import os
from collections.abc import Iterator
from contextlib import contextmanager

TRACE_HEADER = "x-memo-trace-id"
TRACE_ENV = "MEMO_TRACE_ID"
LEGACY_TRACE_ENV = "SYNAPSE_TRACE_ID"

_trace: contextvars.ContextVar[str] = contextvars.ContextVar("memo_trace_id", default="")


def current_trace() -> str:
    return _trace.get()


def ambient_trace() -> str:
    return (
        current_trace()
        or os.environ.get(TRACE_ENV, "").strip()
        or os.environ.get(LEGACY_TRACE_ENV, "").strip()
    )


@contextmanager
def trace_scope(trace_id: str) -> Iterator[None]:
    token = _trace.set(str(trace_id or "").strip())
    try:
        yield
    finally:
        _trace.reset(token)


__all__ = [
    "LEGACY_TRACE_ENV",
    "TRACE_ENV",
    "TRACE_HEADER",
    "ambient_trace",
    "current_trace",
    "trace_scope",
]
