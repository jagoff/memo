"""Declarative, budgetable search pipeline for the post-candidate stages.

`Memory.search()` (in `search_ops.py`) builds a candidate pool, then chains a
long series of optional post-processing passes (fact surface, feedback,
health scores, cross-encoder rerank, recency decay, entity boost, graph
ordering, …). Each pass used to be an inline `if flag_bool(...)` block re-
iterating the result list — one ~670-line method with 26 flag reads.

This module turns that tail into data:

- `SearchStage` — one named pass with an optional latency budget. A stage
  is *skippable* when dropping it only degrades ranking/annotation quality,
  and *mandatory* when it is part of the caller contract (materialize,
  usage tracking, body resolution).
- `SearchBudget` — wall-clock budget checked between stages. When exhausted,
  remaining skippable stages are dropped (traced), so a latency-constrained
  caller (the 5 s recall hook) degrades gracefully instead of blowing its
  budget on nice-to-have passes.
- `run_search_stages` — threads the result list through the stages, honouring
  the budget and emitting a `trace` entry for every skipped pass.

Behaviour is byte-identical to the previous inline chain: stages run in the
same order, each stage still evaluates its own flag gates internally, and a
budget of `None` (the default) disables all budget logic.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from memo.search_deadline import Deadline

StageFn = Callable[[list[Any]], list[Any]]
TraceFn = Callable[..., None]


@dataclass
class SearchCtx:
    """Mutable per-search state threaded through the stage pipeline.

    Holds everything a post-candidate stage needs beyond the result list:
    the raw query parameters, the query embedding (shared between the
    feedback/entity stages), and the mutable flags that later stages both
    read and write (`reranker_will_run`, `health_applied`).
    """

    query: str
    limit: int
    mode: str
    type_: str | None = None
    load_bodies: bool = True
    disable_reranker: bool = False
    recency: bool = False
    include_forgotten: bool = False
    read_through: bool = False
    entity_boost: bool | None = None
    quality_rerank: bool | None = None
    track_usage: bool = True
    as_of: str | None = None
    #: Query embedding from the vec/hybrid candidate legs; None when the
    #: caller opted out of the bi-encoder (bm25/exact/fuzzy modes).
    emb: list[float] | None = None
    #: Computed by the `rerank_gate` stage; the rerank stage may flip it to
    #: False when the skip-confident-RRF decision fires.
    reranker_will_run: bool = False
    #: Set by the pre-rerank health pass so the post-pipeline health gate
    #: does not double-apply scores.
    health_applied: bool = False
    #: True when the candidate pool was fed from the FTS index (hybrid mode)
    #: and survivors need their canonical disk bodies re-resolved.
    bodies_from_fts: bool = False
    trace: TraceFn | None = None
    #: Whole-search wall-clock budget (see `memo.search_deadline.Deadline`).
    #: None means unlimited — stages that consult it (rerank, curated graph
    #: order) run unconditionally, matching pre-deadline behaviour.
    deadline: Deadline | None = None
    #: Out-parameter, same shape as `trace`. A stage a budget-aware caller
    #: shed to stay inside `deadline` appends its name here.
    degraded: list[str] | None = None


def emit(trace: TraceFn | None, stage: str, **data: Any) -> None:
    """Trace helper — no-op when no tracer is attached (unit-testable stages)."""
    if trace is not None:
        trace(stage, **data)


@dataclass
class SearchStage:
    """One named post-candidate pass.

    `run` receives the current result list and returns the (possibly
    transformed) result list. Stages are responsible for their own flag
    gates: an enabled=False stage should return its input unchanged.
    """

    name: str
    run: StageFn
    #: Whether the budget may drop this stage when time runs out.
    skippable: bool = True
    #: Optional per-stage wall-clock budget in milliseconds. When set, the
    #: stage is measured and skipped (if skippable) when it overruns — the
    #: only per-stage budget in the pipeline.
    budget_ms: float | None = None


@dataclass
class SearchBudget:
    """Wall-clock budget for the whole post-candidate pipeline.

    `None` deadline disables every budget check (default — current
    behaviour). The budget is checked BETWEEN stages, never mid-stage.
    """

    total_ms: float | None = None
    _started: float = field(default_factory=time.perf_counter)

    @property
    def remaining_ms(self) -> float | None:
        if self.total_ms is None:
            return None
        return max(0.0, self.total_ms - (time.perf_counter() - self._started) * 1000.0)

    @property
    def exhausted(self) -> bool:
        remaining = self.remaining_ms
        return remaining is not None and remaining <= 0.0


def run_search_stages(
    stages: list[SearchStage],
    initial: list[Any],
    *,
    budget: SearchBudget | None = None,
    trace: TraceFn | None = None,
) -> list[Any]:
    """Run `stages` over `initial`, honouring `budget` and tracing drops.

    Returns the final result list. Stages that are skipped due to budget
    exhaustion produce a `trace("stage_budget_skip", stage=name)` entry;
    stages that overrun their own `budget_ms` (and are skippable) produce a
    `trace("stage_overrun_skip", ...)` entry. Mandatory stages always run.
    """
    out = initial
    for stage in stages:
        if budget is not None and budget.exhausted and stage.skippable:
            if trace is not None:
                trace("stage_budget_skip", stage_name=stage.name, remaining_ms=budget.remaining_ms)
            continue
        if stage.budget_ms is not None:
            before = out
            started = time.perf_counter()
            out = stage.run(out)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            if stage.skippable and elapsed_ms > stage.budget_ms:
                # The stage overran its own budget: discard its result and
                # keep the pre-stage list (the pipeline degrades gracefully
                # instead of shipping a half-applied transform).
                out = before
                if trace is not None:
                    trace(
                        "stage_overrun_skip",
                        stage_name=stage.name,
                        elapsed_ms=round(elapsed_ms, 2),
                        budget_ms=stage.budget_ms,
                    )
            continue
        out = stage.run(out)
    return out
