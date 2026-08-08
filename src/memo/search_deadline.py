"""Wall-clock budget for one search.

Verified 2026-08-06: no read path had a deadline -- zero occurrences of
`deadline`, `time_budget` or `monotonic()` across `memory/search_ops.py`,
`recall_logic.py` and `store/queries.py`. Under contention search stretched from
9.3s to over 300s, indistinguishable from a hang.

The contract is not "be fast". It is: finish inside the budget, and tell the
caller which stages were dropped to do it.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

from memo.flags import flag_int

# Stage cost estimates (ms) used by `afford`. Deliberately pessimistic: the cost
# of skipping a stage that would have fit is one slightly worse ranking; the
# cost of starting one that does not fit is the hang this module prevents.
COST_RERANK_MS = 4000.0
COST_EXPANSION_MS = 1500.0
COST_GRAPH_SIGNAL_MS = 500.0
COST_EMBED_MS = 2000.0


@dataclass(frozen=True)
class Deadline:
    """Monotonic time budget. `budget_ms <= 0` means unlimited."""

    budget_ms: int
    started: float

    @classmethod
    def start(cls, budget_ms: int | None = None) -> Deadline:
        if budget_ms is None:
            # `flag_int` is typed `int | None` because it is a generic accessor
            # over any flag -- but MEMO_SEARCH_BUDGET_MS is registered with a
            # concrete int default (30000), so the flag resolution chain
            # (env > markdown config > tuned overlay > built-in default) always
            # bottoms out on that default, never None. The assert documents the
            # invariant instead of re-implementing the fallback `flag()` already
            # guarantees.
            budget_ms = flag_int("MEMO_SEARCH_BUDGET_MS")
            assert budget_ms is not None, "MEMO_SEARCH_BUDGET_MS is registered with a default"
        return cls(budget_ms=budget_ms, started=time.monotonic())

    @property
    def unlimited(self) -> bool:
        return self.budget_ms <= 0

    def remaining_ms(self) -> float:
        if self.unlimited:
            return math.inf
        elapsed = (time.monotonic() - self.started) * 1000.0
        return self.budget_ms - elapsed

    @property
    def expired(self) -> bool:
        return not self.unlimited and self.remaining_ms() <= 0

    def afford(self, cost_ms: float) -> bool:
        """Room for a stage that typically costs `cost_ms`."""
        return self.remaining_ms() > cost_ms
