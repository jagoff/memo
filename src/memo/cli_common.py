"""Shared CLI state + helpers for the memo command tree.

Extracted from cli.py so command groups can live in their own modules
(cli_graph.py, …) without importing the 9k-line cli.py back — which would
be a circular import. cli.py and every cli_*.py group import from here.
"""

from __future__ import annotations

import os
import sys
from typing import Any

import click
from rich.console import Console

from memo.config import Config

# One process-wide rich Console shared by every command for consistent output.
console = Console()


def get_memory(cfg: Config) -> Any:
    """Build a Memory instance. Deferred import keeps module load light and
    avoids a cli -> memory -> … import cycle at startup."""
    from memo.memory import Memory

    return Memory(cfg)


def _short(text: str, n: int = 120) -> str:
    """Collapse whitespace and truncate `text` to `n` chars with an ellipsis."""
    text = (text or "").strip().replace("\n", " ")
    return text if len(text) <= n else text[: n - 1] + "…"


def _parse_as_of_date(s: str) -> str:
    """Accept date-only (`2026-03-01`) or full ISO. Return ISO with
    a stable noon-UTC anchor for date-only inputs."""
    from datetime import UTC
    from datetime import datetime as _dt
    s = s.strip()
    if len(s) == 10:  # YYYY-MM-DD
        return f"{s}T23:59:59+00:00"  # end-of-day to be inclusive
    try:
        dt = _dt.fromisoformat(s.rstrip("Z"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt.isoformat()
    except ValueError as exc:
        raise click.ClickException(
            f"Could not parse --date {s!r}. Use YYYY-MM-DD or ISO 8601.",
        ) from exc


def _backend_native_trace_id(trace_id: str = "") -> str:
    return (trace_id or os.environ.get("SYNAPSE_TRACE_ID", "")).strip()


def _memo_backend_version() -> str:
    from memo import __version__

    return __version__


def _resolved(thunk: Any) -> Any:
    """Run `thunk()` translating `AmbiguousIdError` into a friendly print
    + exit code 2. Used by every CLI verb that takes an id-or-prefix
    argument (`get`, `update`, `delete`, `extract-entities`).
    """
    from memo.memory import AmbiguousIdError

    try:
        return thunk()
    except AmbiguousIdError as exc:
        console.print(f"[red]ambiguous id prefix[/red] {exc.prefix!r} matches:")
        for m in exc.matches[:8]:
            console.print(f"  · {m}")
        if len(exc.matches) > 8:
            console.print(f"  · …and {len(exc.matches) - 8} more")
        sys.exit(2)
