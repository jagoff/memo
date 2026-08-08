Status: partially shipped — neither `src/memo/search_deadline.py` (C1) nor `tests/conformance/` (C2) exist on master. #209 reused/fixed the pre-existing `MEMO_RERANK_BUDGET_S` → RRF fallback and fixed several individual conformance-class defects (symlinked-`/tmp` traceback, missing-parent-dir errors) ad hoc, without the designed `SearchDeadline` shed-ladder or the harness. #210 (merged 2026-08-07) separately added a narrower `SearchBudget` mechanism scoped only to the recall-hook path (`recall_search_budget_ms()`), not the `MEMO_SEARCH_BUDGET_MS`/`degraded`-list contract this design specifies for `Memory.search()` generally. A complete implementation matching this design exists on local branch `feat/conformance-budget-deadline-admission` (`search_deadline.py`, `tests/conformance/*`), not yet opened as a PR.

# Deadline degradation + corpus-scale conformance — design

**Date:** 2026-08-06
**Status:** proposed
**Scope:** `src/memo/search_deadline.py` (new), `src/memo/memory/search_ops.py`, `src/memo/memory/rerank_ops.py`, `tests/conformance/` (new), `pyproject.toml` markers, CI lane

## Problem

Two failures with one root: nothing in memo is verified against a real-sized
corpus under real load.

**No read path has a deadline.** Verified 2026-08-06: zero occurrences of
`deadline`, `time_budget` or `monotonic()` in `memory/search_ops.py`,
`recall_logic.py`, or `store/queries.py`. Measured degradation on the live
corpus:

| Condition | `memo search` |
|---|---|
| idle | 9.3s |
| with `memo maintain` in background | 25.9s |
| with `maintain` + full test suite | **>300s** |

There is no timeout and no fallback. The first search of a session is
indistinguishable from a hang. `--max-scan-seconds` bounds only maintain's scan
phase, not its runtime, and nothing bounds search at all.

**The gates do not see the surfaces.** 6,955 tests, mypy across 501 files, ruff,
`memo definitive check` and `memo journey-check` were all green while twelve
defects lived in shipped surfaces. Every one was found by hand against 11,383
real memories. The defects were not subtle logic errors; they were:

- internal caps returned to the caller as if complete (5 tools)
- a page size reported as the corpus total — `memo analytics summary` printed
  `9999` against 11,383, and derived a growth rate from it
- `atomic_write_text` rejecting any destination with a symlinked parent, which
  on macOS means all of `/tmp` — `memo graph mindmap -o /tmp/x.html` raised a
  raw traceback
- `memo links reindex` deleting the crossref index and rebuilding only the
  newest 10,000 — silent data loss
- `memo backup --out` / `memo export` raw-tracebacking on a missing parent dir

None are reachable with a 3-memory fixture. All are trivially reachable with
10,000.

## Goals

1. No read path exceeds its time budget. Under contention it degrades and
   **says so**.
2. A test harness at corpus scale that asserts payload size, wall-clock, and
   reported-total honesty per surface.
3. The harness reproduces the twelve defects before their fixes.

## Non-goals

- Making search faster. This is about bounded, honest behaviour under load, not
  throughput.
- Replacing `memo eval recall`. Retrieval *quality* stays gated there, against
  the live corpus. The conformance harness gates *shape* — size, time, honesty.

## Design

### C1 — deadline degradation

`SearchDeadline`: a monotonic-clock value object (`remaining_ms()`, `expired`),
created at the entry of `Memory.search` from `MEMO_SEARCH_BUDGET_MS` and passed
down. Stages consult the remaining budget and shed work in cost order:

| Order | Shed | Existing hook |
|---|---|---|
| 1 | rerank | `rerank_ops.py:395` already caps via `MEMO_RERANK_BUDGET_S` and falls back to RRF order on `RerankBudgetExceeded` — the deadline passes `min(budget, remaining)` instead of a fixed 20.0 |
| 2 | query expansion / HyDE / multi-query | flag-gated stages, skipped |
| 3 | graph / associative signal | already has `MEMO_GRAPH_SIGNAL_BUDGET_MS`, `MEMO_ASSOCIATIVE_BUDGET_MS` |
| 4 | MLX embed → BM25-only | the cold path that already exists; taken deliberately instead of by accident |

This extends the pattern already in the codebase rather than inventing one; the
rerank stage's budget-and-fall-back-to-RRF behaviour is the model.

Every shed stage appends to `degraded: list[str]` on the result. The CLI prints
it dim (`degraded: rerank skipped (budget)`); the MCP result carries the field.
The rule is **degrade and say so** — never stretch silently, never fail silently.

**Default.** `MEMO_SEARCH_BUDGET_MS` defaults to **30000**, on. The worst
measured healthy search is 9.3s, so a 30s cap cannot fire on a healthy install;
it only truncates the pathological contention case that currently reaches 300s+.
`0` disables it. This follows `MEMO_RERANK_BUDGET_S`, which is likewise a
numeric budget active by default — it is not a dark feature flag and does not
need a graduation gate.

### C2 — conformance harness

`tests/conformance/`, new pytest marker `conformance` in `pyproject.toml`
alongside the existing `slow` / `db_contract` / `concurrency` markers,
deselected from the default suite and run in its own CI lane.

**Fixture** (`tests/conformance/conftest.py`): session-scoped, seeds ~10,000
deterministic synthetic memories into an isolated `Config` (never the developer
vault — `tests/conftest.py` conventions apply). Vectors come from a deterministic
hash-to-float32 stub with `MEMO_EMBEDDER_DIMS` pinned to the stub's output dim,
per the MLX invariant. Seeding writes through `VecStore` in bulk; target build
under 60s, reused across the whole session. Corpus size is env-overridable so a
developer can iterate at 1,000.

**Modules and the invariant each asserts:**

| Module | Invariant | Defect class covered |
|---|---|---|
| `test_mcp_response_budget.py` | every registered tool ≤ its cap | the 5 payload defects (see the response-budget spec) |
| `test_read_latency_budget.py` | every read surface within budget; under injected contention it degrades, reports `degraded`, and never exceeds the deadline | search 9.3s → >300s |
| `test_reported_totals.py` | any surface reporting a total reports the real one, not a page size | `analytics summary` "9999", dashboard corpus panel |
| `test_output_paths.py` | every `-o`/`--out` CLI against `/tmp` (a symlink on macOS) and a missing parent dir → clean error or success, never a raw traceback | `graph mindmap`, `federation export`, `backup`, `export` |
| `test_index_rebuild_preserves.py` | a reindex/rebuild preserves what it does not own | `links reindex` dropping the crossref index |

**Discipline:** each test is validated by reverting its fix locally and
confirming the test fails. For the defects already fixed on
`fix/bounded-mcp-payloads`, that check happens during implementation and is
recorded in the plan — it is not shipped as a skipped or xfail test.

## Testing

The harness is the test. Its own correctness is established by the
revert-and-confirm-red step above, which is what distinguishes it from the 6,955
tests that were green through all twelve defects.

`SearchDeadline` additionally gets unit tests: monotonic accounting, each rung of
the ladder shed in order, `degraded` populated accurately, budget `0` disabling
the mechanism with byte-identical results.

## Success criteria

- Every one of the twelve sweep defects has a conformance test that fails
  against the pre-fix code.
- `memo search` under `maintain` + full suite returns within
  `MEMO_SEARCH_BUDGET_MS` with a populated `degraded` list, instead of 300s+.
- A new MCP tool with an unbounded elastic field, or a new surface reporting a
  page size as a total, fails CI.

## Risks

- **Fixture build cost.** Session-scoped and deterministic; if 10k proves too
  slow to build, the shape defects are reachable at 5k and the size is a
  parameter, not a constant.
- **A stub embedder does not validate semantic quality.** Deliberate. Quality
  stays with `memo eval recall` against the live corpus; this harness gates
  size, time and honesty, which are exactly the properties the live-corpus eval
  does not check.
- **Contention injection can be flaky in CI.** Model it as a deterministic
  bounded synthetic load (a writer holding `BEGIN IMMEDIATE`), not as
  wall-clock racing, and assert the *degradation decision*, not a specific
  timing.
- **A 30s default is high for an interactive CLI.** It is chosen to be
  unreachable on a healthy install so the change cannot regress anyone; once the
  latency lane has a baseline, lowering it is a measured follow-up, not a guess.
