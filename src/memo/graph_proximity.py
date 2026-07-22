"""Legacy raw-graph query entity matching for navigation compatibility.

Retrieval serving no longer uses this module; curated query resolution lives in
``GraphReadModel`` and is applied once by ``Memory.search``.
"""

from __future__ import annotations

import contextlib
import re
from typing import Any

_TOKEN_RE = re.compile(r"[a-záéíóúüñ0-9]+", re.IGNORECASE)


def extract_query_entities(prompt: str, graph: Any) -> list[str]:
    """Return regex entities plus exact unigram/bigram raw-vocabulary matches."""
    from memo.entity_extractor import extract_entities

    out = list(extract_entities(prompt))
    seen = {entity.lower() for entity in out}
    names: set[str] = set()
    with contextlib.suppress(Exception):
        names = graph.entity_names()
    if not names:
        return out
    tokens = [token for token in _TOKEN_RE.findall(prompt.lower()) if len(token) >= 3]
    candidates = set(tokens)
    candidates.update(f"{tokens[index]} {tokens[index + 1]}" for index in range(len(tokens) - 1))
    for candidate in sorted(candidates):
        if candidate in names and candidate not in seen:
            out.append(candidate)
            seen.add(candidate)
    return out


__all__ = ["extract_query_entities"]
