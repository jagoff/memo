"""Unit tests for the declarative search stage pipeline (`search_pipeline.py`).

These cover the pure pipeline mechanics (ordering, budget enforcement,
per-stage overrun, trace output) without touching a real Memory/store —
the stages themselves are already exercised end-to-end through
`test_memory_search.py` and the recall/ask/chat suites.
"""

from __future__ import annotations

from memo.memory.search_pipeline import (
    SearchBudget,
    SearchStage,
    emit,
    run_search_stages,
)


def _stage(tag: str, *, skippable: bool = True, budget_ms: float | None = None) -> SearchStage:
    def run(_out: list[int]) -> list[int]:
        return [len(tag)]

    return SearchStage(tag, run, skippable=skippable, budget_ms=budget_ms)


def _exhausted_budget() -> SearchBudget:
    # Negative total ⇒ remaining_ms is immediately 0.
    return SearchBudget(total_ms=-1.0)


def test_no_budget_runs_every_stage_in_order() -> None:
    calls: list[str] = []

    def a(_out: list[int]) -> list[int]:
        calls.append("a")
        return [1, 2]

    def b(_out: list[int]) -> list[int]:
        calls.append("b")
        return [3]

    out = run_search_stages(
        [SearchStage("a", a, skippable=True), SearchStage("b", b, skippable=True)],
        [0],
        budget=None,
    )
    assert calls == ["a", "b"]
    assert out == [3]


def test_exhausted_budget_skips_remaining_skippable_stages() -> None:
    ran: list[str] = []

    def run(tag: str) -> SearchStage:
        def _s(_out: list[int]) -> list[int]:
            ran.append(tag)
            return [1]

        return SearchStage(tag, _s, skippable=True)

    out = run_search_stages(
        [run("a"), run("b")],
        initial=[0],
        budget=_exhausted_budget(),
        trace=lambda stage, **kw: None,
    )
    assert ran == []
    assert out == [0]


def test_mandatory_stages_run_even_under_exhausted_budget() -> None:
    ran: list[str] = []

    def mk(tag: str, skippable: bool) -> SearchStage:
        def run(_out: list[int]) -> list[int]:
            ran.append(tag)
            return [len(ran)]

        return SearchStage(tag, run, skippable=skippable)

    out = run_search_stages(
        [
            mk("a", skippable=False),
            mk("b", skippable=True),
            mk("c", skippable=False),
        ],
        initial=[0],
        budget=_exhausted_budget(),
    )
    assert ran == ["a", "c"]
    assert out == [2]


def test_budget_skip_emits_trace_entry() -> None:
    traces: list[tuple[str, object]] = []

    def _trace(stage: str, **data: object) -> None:
        traces.append((stage, data.get("stage_name")))

    out = run_search_stages(
        [_stage("x"), _stage("y")],
        initial=[0],
        budget=_exhausted_budget(),
        trace=_trace,
    )
    assert out == [0]
    assert traces == [("stage_budget_skip", "x"), ("stage_budget_skip", "y")]


def test_per_stage_overrun_skips_stage_and_traces() -> None:
    traces: list[tuple[str, str]] = []

    def slow_stage(_out: list[int]) -> list[int]:
        import time

        time.sleep(0.02)
        return [9]

    out = run_search_stages(
        [SearchStage("slow", slow_stage, skippable=True, budget_ms=0.001)],
        initial=[0],
        budget=None,
        trace=lambda stage, **kw: traces.append((stage, str(kw.get("stage_name")))),
    )
    # Per-stage overrun skips a SKIPPABLE stage; the run fn still produced a
    # value, but by contract we discard the result of an overrun stage.
    assert out == [0]
    assert traces == [("stage_overrun_skip", "slow")]


def test_emit_noop_without_tracer() -> None:
    emit(None, "anything", key=1)  # must not raise
