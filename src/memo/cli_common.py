"""Shared CLI state + helpers for the memo command tree.

Extracted from cli.py so command groups can live in their own modules
(cli_graph.py, …) without importing the 9k-line cli.py back — which would
be a circular import. cli.py and every cli_*.py group import from here.
"""

from __future__ import annotations

from typing import Any

from rich.console import Console

from memo.config import Config

# One process-wide rich Console shared by every command for consistent output.
console = Console()


def get_memory(cfg: Config) -> Any:
    """Build a Memory instance. Deferred import keeps module load light and
    avoids a cli -> memory -> … import cycle at startup."""
    from memo.memory import Memory

    return Memory(cfg)
