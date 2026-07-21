"""Proactive engine package.

`ProactiveSuggester`/`Suggestion`/`SuggestionFeedback` re-export lazily via
`__getattr__` — `.suggester` imports `memo.llm` at module top, so an eager
re-export here would drag the LLM stack into every cheap `memo.proactive.*`
import (e.g. `memo.proactive.surfaces`, used on the digest/briefing hot path)
(I4 review fix).
"""

from __future__ import annotations

from typing import Any

__all__ = ["ProactiveSuggester", "Suggestion", "SuggestionFeedback"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from . import suggester

        return getattr(suggester, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
