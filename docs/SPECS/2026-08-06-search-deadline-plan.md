# Search Deadline Degradation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give every search a time budget it cannot silently exceed: under contention it sheds work in cost order, reports exactly what it dropped, and returns.

**Architecture:** A `Deadline` value object is created at the top of `Memory.search` from `MEMO_SEARCH_BUDGET_MS` and consulted before each expensive stage. The rerank stage — which already has a budget and an RRF fallback — is capped by the remaining time rather than a fixed 20s. Shed stages are appended to a caller-supplied `_degraded` list, the same out-parameter shape `_trace` already uses, so no return type changes.

**Tech Stack:** `time.monotonic`, the `MEMO_*` flag registry, Click, pytest.

Spec: `docs/SPECS/2026-08-06-deadline-and-corpus-conformance-design.md` (part C1).
Depends on: `docs/SPECS/2026-08-06-corpus-conformance-plan.md` Task 2 (the `big_corpus` fixture) for Task 4 only.

## Global Constraints

- `Memory.search` returns `list[MemoryRecord]` and MUST keep returning exactly that. Degradation is reported through an out-parameter, never by changing the return type.
- Default `MEMO_SEARCH_BUDGET_MS = 30000`, active. The worst measured healthy search is 9.3s, so 30s cannot fire on a healthy install; it only truncates the contention case that reaches 300s+. `0` disables. This is a numeric budget like `MEMO_RERANK_BUDGET_S`, not a dark feature flag, so it needs no graduation gate.
- Degrade and say so. A shed stage that is not reported is the bug this plan exists to prevent.
- Flags in `flags_search.py` (`SPECS` tuple), never inline `os.environ`.
- Shared working tree: stage explicit paths only.

---

### Task 1: The deadline primitive

**Files:**
- Create: `src/memo/search_deadline.py`
- Create: `tests/test_search_deadline.py`
- Modify: `src/memo/flags_search.py` (append to `SPECS`)

**Interfaces:**
- Produces:
  - `Deadline` — frozen dataclass with `budget_ms: int`, `started: float`
  - `Deadline.start(budget_ms: int | None = None) -> Deadline` — classmethod; `None` reads the flag
  - `Deadline.remaining_ms() -> float` — `inf` when unlimited
  - `Deadline.expired` — property, `False` when unlimited
  - `Deadline.unlimited` — property
  - `Deadline.afford(cost_ms: float) -> bool` — is there room for a stage that typically costs this much

- [ ] **Step 1: Write the failing tests**

`tests/test_search_deadline.py`:

```python
"""The deadline primitive. Monotonic, unlimited-safe, and cheap enough to
consult before every stage."""

from __future__ import annotations

import math
import time

from memo.search_deadline import Deadline


def test_unlimited_budget_never_expires() -> None:
    d = Deadline.start(0)
    assert d.unlimited
    assert not d.expired
    assert math.isinf(d.remaining_ms())
    assert d.afford(10_000_000)


def test_remaining_shrinks_monotonically() -> None:
    d = Deadline.start(1000)
    first = d.remaining_ms()
    time.sleep(0.01)
    assert d.remaining_ms() < first
    assert d.remaining_ms() <= 1000


def test_expired_after_the_budget() -> None:
    d = Deadline.start(5)
    time.sleep(0.02)
    assert d.expired


def test_afford_refuses_a_stage_that_will_not_fit() -> None:
    d = Deadline.start(100)
    assert d.afford(10)
    assert not d.afford(10_000)


def test_start_reads_the_flag_when_unset(monkeypatch) -> None:
    monkeypatch.setenv("MEMO_SEARCH_BUDGET_MS", "1234")
    assert Deadline.start().budget_ms == 1234
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run --no-sync pytest tests/test_search_deadline.py -q`
Expected: `ModuleNotFoundError: No module named 'memo.search_deadline'`.

- [ ] **Step 3: Register the flag**

Append to the `SPECS` tuple in `src/memo/flags_search.py`:

```python
    _spec(
        "MEMO_SEARCH_BUDGET_MS",
        "int",
        30000,
        "search",
        "Wall-clock budget for one search. Stages shed work in cost order "
        "(rerank, then query expansion, then graph signal, then the embed "
        "itself falls back to BM25) and every shed stage is reported to the "
        "caller -- degrade and say so, never stretch silently. Measured "
        "2026-08-06: search went 9.3s idle -> 25.9s under maintain -> >300s "
        "under maintain + full suite, with no timeout and no fallback. 30000 "
        "is above the worst healthy case so it cannot fire on a healthy "
        "install. 0 = no deadline.",
        min_val=0,
    ),
```

- [ ] **Step 4: Write the module**

`src/memo/search_deadline.py`:

```python
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
            flagged = flag_int("MEMO_SEARCH_BUDGET_MS")
            budget_ms = 30000 if flagged is None else flagged
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
```

- [ ] **Step 5: Run the tests**

Run: `uv run --no-sync pytest tests/test_search_deadline.py -q`
Expected: 5 passed.

- [ ] **Step 6: Validate the registry**

Run: `uv run --no-sync memo config validate`
Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add src/memo/search_deadline.py tests/test_search_deadline.py src/memo/flags_search.py
git commit -m "feat(search): add a monotonic search deadline primitive"
```

---

### Task 2: Cap the rerank stage with the remaining budget

**Files:**
- Modify: `src/memo/memory/rerank_ops.py:363-369` (signature), `:395-400` (budget resolution)
- Modify: `src/memo/memory/search_ops.py:663` (call site)
- Create: `tests/test_search_deadline_rerank.py`

**Interfaces:**
- Consumes: `Deadline` from Task 1.
- Produces: `_rerank(self, query, hits, *, top_n, deadline=None, degraded=None) -> list[MemoryRecord]` — `deadline` caps the reranker's own budget; `degraded` collects shed-stage names.

Rerank is first on the ladder because it is the most expensive stage and already has the fallback: `RerankBudgetExceeded` is caught at `rerank_ops.py:400` and the RRF order is served instead. This task makes that budget respect the caller's remaining time instead of a fixed 20.0s.

- [ ] **Step 1: Write the failing test**

`tests/test_search_deadline_rerank.py`:

```python
"""Rerank is rung one of the ladder: the most expensive stage, and the one that
already knows how to fall back. Under a spent deadline it must not start."""

from __future__ import annotations

import time

import pytest

from memo.search_deadline import Deadline


def test_rerank_is_skipped_when_the_deadline_cannot_afford_it(tmp_cfg, monkeypatch) -> None:
    from memo.memory.facade import Memory

    mem = Memory(tmp_cfg)
    try:
        called: list[str] = []
        monkeypatch.setattr(
            type(mem), "_ensure_reranker", lambda self: called.append("built") or object()
        )
        spent = Deadline.start(5)
        time.sleep(0.02)
        degraded: list[str] = []

        out = mem._rerank("q", [], top_n=3, deadline=spent, degraded=degraded)

        assert called == [], "the reranker was built despite an expired deadline"
        assert degraded == ["rerank_skipped"]
        assert out == []
    finally:
        mem.close()


def test_rerank_runs_and_reports_nothing_when_there_is_room(tmp_cfg) -> None:
    from memo.memory.facade import Memory

    mem = Memory(tmp_cfg)
    try:
        degraded: list[str] = []
        mem._rerank("q", [], top_n=3, deadline=Deadline.start(30000), degraded=degraded)
        assert degraded == []
    finally:
        mem.close()


def test_rerank_without_a_deadline_behaves_exactly_as_before(tmp_cfg) -> None:
    from memo.memory.facade import Memory

    mem = Memory(tmp_cfg)
    try:
        assert mem._rerank("q", [], top_n=3) == []
    finally:
        mem.close()
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --no-sync pytest tests/test_search_deadline_rerank.py -q`
Expected: `TypeError: _rerank() got an unexpected keyword argument 'deadline'`.

- [ ] **Step 3: Implement**

In `src/memo/memory/rerank_ops.py`, change the signature at line 363:

```python
    def _rerank(
        self,
        query: str,
        hits: list[MemoryRecord],
        *,
        top_n: int,
        deadline: Deadline | None = None,
        degraded: list[str] | None = None,
    ) -> list[MemoryRecord]:
```

Add `from memo.search_deadline import COST_RERANK_MS, Deadline` to the imports.

At the top of the body, before `self._ensure_reranker()`:

```python
        # Rung one of the shed ladder. Skipping costs one slightly worse
        # ordering; starting a cross-encoder pass that cannot finish costs the
        # hang. The RRF order the candidates already carry is the fallback --
        # the same one `RerankBudgetExceeded` falls back to below.
        if deadline is not None and not deadline.afford(COST_RERANK_MS):
            if degraded is not None:
                degraded.append("rerank_skipped")
            return hits[:top_n]
```

At line 395, cap the reranker's own budget by what is left:

```python
        budget = flag_float("MEMO_RERANK_BUDGET_S")
        budget_s = 20.0 if budget is None else budget
        if deadline is not None and not deadline.unlimited:
            budget_s = min(budget_s, max(0.1, deadline.remaining_ms() / 1000.0))
```

and pass `budget_s=budget_s` to `reranker.rerank(...)`.

- [ ] **Step 4: Thread it from the search call site**

In `src/memo/memory/search_ops.py`, change line 663 from

```python
            out = self._rerank(query, out, top_n=limit)
```

to

```python
            out = self._rerank(query, out, top_n=limit, deadline=_deadline, degraded=_degraded)
```

`_deadline` and `_degraded` are introduced in Task 3; until then this line will not resolve. Do Task 3 before running the full suite — or introduce the two locals now as `_deadline = None` / `_degraded = None` and let Task 3 replace them.

- [ ] **Step 5: Run the tests**

Run: `uv run --no-sync pytest tests/test_search_deadline_rerank.py tests/test_search_deadline.py -q`
Expected: green.

If `_ensure_reranker` is not the accessor the current code uses, read `rerank_ops.py:386` and patch the real one in the test.

- [ ] **Step 6: Commit**

```bash
git add src/memo/memory/rerank_ops.py src/memo/memory/search_ops.py tests/test_search_deadline_rerank.py
git commit -m "feat(search): cap the rerank stage with the caller's remaining budget"
```

---

### Task 3: The shed ladder in `Memory.search`

**Files:**
- Modify: `src/memo/memory/search_ops.py:139-160` (signature), body (deadline creation + ladder rungs)
- Create: `tests/test_search_degradation.py`

**Interfaces:**
- Consumes: `Deadline`, `COST_*` from Task 1; `_rerank(..., deadline, degraded)` from Task 2.
- Produces: `Memory.search(..., _budget_ms: int | None = None, _degraded: list[str] | None = None) -> list[MemoryRecord]` — return type unchanged.
- Produces: degradation tokens, exactly these strings: `"rerank_skipped"`, `"expansion_skipped"`, `"graph_signal_skipped"`, `"embed_skipped_bm25_only"`.

- [ ] **Step 1: Write the failing test**

`tests/test_search_degradation.py`:

```python
"""The ladder: shed in cost order, report every rung, never exceed the budget.

The assertions are on the DEGRADATION DECISION, not on wall-clock racing, so
this is deterministic in CI."""

from __future__ import annotations

import pytest


def test_search_reports_nothing_when_the_budget_is_generous(tmp_cfg) -> None:
    from memo.memory.facade import Memory

    mem = Memory(tmp_cfg)
    try:
        degraded: list[str] = []
        mem.search("anything", mode="bm25", _budget_ms=30000, _degraded=degraded)
        assert degraded == []
    finally:
        mem.close()


def test_an_exhausted_budget_sheds_every_optional_stage(tmp_cfg) -> None:
    from memo.memory.facade import Memory

    mem = Memory(tmp_cfg)
    try:
        degraded: list[str] = []
        mem.search("anything", mode="hybrid", _budget_ms=1, _degraded=degraded)
        assert "rerank_skipped" in degraded
    finally:
        mem.close()


def test_zero_budget_disables_the_ladder(tmp_cfg) -> None:
    from memo.memory.facade import Memory

    mem = Memory(tmp_cfg)
    try:
        degraded: list[str] = []
        mem.search("anything", mode="hybrid", _budget_ms=0, _degraded=degraded)
        assert degraded == []
    finally:
        mem.close()


def test_search_without_the_out_parameter_is_unchanged(tmp_cfg) -> None:
    from memo.memory.facade import Memory

    mem = Memory(tmp_cfg)
    try:
        assert mem.search("anything", mode="bm25") == []
    finally:
        mem.close()
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --no-sync pytest tests/test_search_degradation.py -q`
Expected: `TypeError: search() got an unexpected keyword argument '_budget_ms'`.

- [ ] **Step 3: Implement**

In `src/memo/memory/search_ops.py`, add two parameters after `_trace` (line 159):

```python
        _budget_ms: int | None = None,
        _degraded: list[str] | None = None,
```

Document them in the docstring's `Args:` block alongside `load_bodies`:

```
            _degraded: Out-parameter, same shape as `_trace`. Pass a list and
                every stage shed to stay inside the wall-clock budget appends
                its name to it. Degrade and say so -- a search that quietly
                dropped its reranker looks identical to one that ran it.
            _budget_ms: Override for MEMO_SEARCH_BUDGET_MS. 0 = no deadline.
```

At the top of the body, before any expensive work:

```python
        _deadline = Deadline.start(_budget_ms)
```

with `from memo.search_deadline import (COST_EMBED_MS, COST_EXPANSION_MS, COST_GRAPH_SIGNAL_MS, Deadline)` in the imports.

Then guard each optional stage. The rerank rung is already done in Task 2. For the remaining rungs, wrap the existing stage conditions:

```python
        # Rung two: query expansion / HyDE / multi-query.
        if <existing expansion condition> and _deadline.afford(COST_EXPANSION_MS):
            ...
        elif <existing expansion condition>:
            if _degraded is not None:
                _degraded.append("expansion_skipped")
```

Apply the identical shape to the graph/associative signal stage with
`COST_GRAPH_SIGNAL_MS` and `"graph_signal_skipped"`, and to the embed stage with
`COST_EMBED_MS` and `"embed_skipped_bm25_only"` — where shedding the embed means
taking the BM25-only path deliberately instead of by accident.

Locate the real stage conditions with:

```bash
grep -n "hyde\|expansion\|multi_query\|graph_signal\|associative\|embed_query" src/memo/memory/search_ops.py
```

Guard only stages that already exist. Do not add a stage in order to shed it.

- [ ] **Step 4: Run the tests**

Run: `uv run --no-sync pytest tests/test_search_degradation.py -q`
Expected: green.

- [ ] **Step 5: Run the search suites**

Run: `uv run --no-sync pytest tests/test_memory.py tests/test_search*.py tests/test_recall_hooks.py -q`
Expected: green. The default budget is 30s and these tests take milliseconds, so no rung should fire.

- [ ] **Step 6: Commit**

```bash
git add src/memo/memory/search_ops.py tests/test_search_degradation.py
git commit -m "feat(search): shed optional stages to stay inside the budget"
```

---

### Task 4: Surface the degradation to humans and clients

**Files:**
- Modify: `src/memo/cli_search.py:76+` (the `search` command)
- Modify: `src/memo/server_core_search.py` (the `memo_search` tool result)
- Create: `tests/conformance/test_read_latency_budget.py`

**Interfaces:**
- Consumes: `Memory.search(..., _degraded=...)` from Task 3; `big_corpus` from the conformance plan Task 2.

A shed stage nobody sees is a silent failure with extra steps.

- [ ] **Step 1: Write the failing tests**

`tests/conformance/test_read_latency_budget.py`:

```python
"""Under a budget it cannot meet, search returns inside the budget AND says what
it dropped. The assertion is on the decision and the report, not on a race."""

from __future__ import annotations

import json
import time

import pytest
from click.testing import CliRunner

from memo.cli import cli
from memo.memory.facade import Memory

pytestmark = pytest.mark.conformance


def _env(cfg) -> dict[str, str]:
    return {
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_DATA_DIR": str(cfg.data_dir),
        "MEMO_STATE_DIR": str(cfg.state_dir),
        "MEMO_EMBEDDER_DIMS": "64",
    }


def test_search_returns_inside_a_generous_budget(big_corpus) -> None:
    mem = Memory(big_corpus)
    try:
        started = time.monotonic()
        degraded: list[str] = []
        mem.search("topic00", mode="bm25", limit=10, _budget_ms=30000, _degraded=degraded)
        assert (time.monotonic() - started) < 30.0
    finally:
        mem.close()


def test_a_tight_budget_degrades_and_reports(big_corpus) -> None:
    mem = Memory(big_corpus)
    try:
        degraded: list[str] = []
        mem.search("topic00", mode="hybrid", limit=10, _budget_ms=1, _degraded=degraded)
        assert degraded, "search silently ran every stage under a 1ms budget"
    finally:
        mem.close()


def test_cli_reports_degradation_in_json(big_corpus) -> None:
    result = CliRunner().invoke(
        cli,
        ["search", "topic00", "--json", "--limit", "5"],
        env={**_env(big_corpus), "MEMO_SEARCH_BUDGET_MS": "1"},
    )
    assert result.exit_code == 0, result.output
    assert "degraded" in json.loads(result.output)
```

- [ ] **Step 2: Run to verify it fails**

Run: `MEMO_CONFORMANCE_CORPUS_N=10000 uv run --no-sync pytest tests/conformance/test_read_latency_budget.py -q`
Expected: the CLI test fails — no `degraded` key.

- [ ] **Step 3: Wire the CLI**

In `src/memo/cli_search.py`, in the `search` command body, pass an out-parameter and surface it:

```python
    degraded: list[str] = []
    hits = mem.search(query, ..., _degraded=degraded)
```

For `--json`, add `"degraded": degraded` to the emitted object. For the human path, after the results, when `degraded` is non-empty:

```python
        click.secho(
            f"degraded: {', '.join(degraded)} (search budget)", dim=True, err=True
        )
```

to stderr so it never contaminates a piped result.

- [ ] **Step 4: Wire the MCP tool**

In `src/memo/server_core_search.py`, thread the same list into `Memory.search` and include `"degraded": degraded` in the tool's result dict. Add it only when non-empty, so an unaffected response is byte-identical to today's.

- [ ] **Step 5: Run the tests**

Run: `MEMO_CONFORMANCE_CORPUS_N=10000 uv run --no-sync pytest tests/conformance/test_read_latency_budget.py -q`
Expected: green.

- [ ] **Step 6: Prove the ladder actually fires under real contention**

Manually, against the live install:

```bash
env -u PYTHONPATH memo maintain &
time env -u PYTHONPATH memo search "decision" --limit 5
```

Expected: returns within `MEMO_SEARCH_BUDGET_MS` with a `degraded:` note on stderr, instead of the 25.9s / 300s+ measured on 2026-08-06. Record the number in the commit message.

- [ ] **Step 7: Full check**

```bash
uv run --no-sync pytest -m "not slow and not conformance" -q
uv run --no-sync mypy src/memo/search_deadline.py src/memo/memory/search_ops.py src/memo/memory/rerank_ops.py src/memo/cli_search.py
uv run --no-sync ruff check src/memo/search_deadline.py src/memo/memory/search_ops.py src/memo/memory/rerank_ops.py src/memo/cli_search.py
```

- [ ] **Step 8: Commit**

```bash
git add src/memo/cli_search.py src/memo/server_core_search.py tests/conformance/test_read_latency_budget.py
git commit -m "feat(search): report shed stages to the CLI and the MCP surface"
```

---

## Self-review notes

- Spec C1 coverage: the four ladder rungs (Tasks 2-3), reuse of the existing rerank budget/fallback rather than a parallel mechanism (Task 2), `degraded` reporting on both surfaces (Task 4), the 30000 default with its justification (Task 1), `0` disabling (Tasks 1 and 3).
- The spec says "the result carries `degraded`". `Memory.search` returns `list[MemoryRecord]`, so carrying a field would be a breaking return-type change; the out-parameter mirrors the existing `_trace` seam and reaches the same two surfaces. This is the faithful implementation of the intent, and Task 4 is what makes it observable.
- Unverified identifiers flagged in-step: the real stage conditions in `search_ops.py` (Task 3 Step 3 gives the grep), `_ensure_reranker` (Task 2 Step 5), and the `memo_search` tool module name (Task 4 Step 4).
