"""Base types for the MEMO_* flag registry.

Extracted from flags.py so domain spec files (flags_recall.py, flags_search.py, etc.)
can import FlagSpec + _spec without circular dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

FlagKind = Literal["bool", "int", "float", "str"]

# Truthy spellings accepted for bool flags. Mirrors the historical mix of
# `== "1"` and `.lower() in (...)` checks across the codebase.
_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off", ""}


@dataclass(frozen=True)
class FlagSpec:
    """One `MEMO_*` flag: how to parse it, its default, and what it does."""

    name: str
    kind: FlagKind
    default: Any
    group: str
    help: str
    # Some bools are checked with inverted polarity (`!= "1"` → default-on,
    # opt-out). Recorded so `config flags` can show the real default.
    opt_out: bool = False
    # Optional inclusive bounds for numeric flags. Enforced in _coerce().
    min_val: float | None = None
    max_val: float | None = None


def _spec(
    name: str,
    kind: FlagKind,
    default: Any,
    group: str,
    help: str,
    opt_out: bool = False,
    min_val: float | None = None,
    max_val: float | None = None,
) -> FlagSpec:
    return FlagSpec(name, kind, default, group, help, opt_out, min_val, max_val)
