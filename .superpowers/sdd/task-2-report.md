# Task 2: Wire Quality Rerank Into Explicit Search — Completion Report

Status: DONE  
Date: 2026-07-08  
Branch: `master`

## Outcome
- Wired Task 1's quality rerank into the explicit `Memory.search()` pipeline only.
- Kept the existing hybrid/vector/BM25 candidate generation path unchanged.
- Kept the feature default-off behind `MEMO_QUALITY_RERANK`.
- Did not change Task 1 APIs.

## Files
- Modified: `src/memo/memory/search_scoring_ops.py`
- Modified: `src/memo/memory/search_ops.py`
- Added: `tests/test_quality_search.py`

## Implementation
1. Added focused integration coverage in `tests/test_quality_search.py` for:
   - flag-gated behavior when `MEMO_QUALITY_RERANK=0` vs `1`
   - verified-hit promotion via quality rerank
2. Confirmed the red phase by running the new test first and observing:
   - `AttributeError: '_Harness' object has no attribute '_apply_quality_rerank'`
3. Added `_SearchScoringMixin._apply_quality_rerank(results)`:
   - reads the registered flag via `flag_bool("MEMO_QUALITY_RERANK")`
   - delegates to `memo.quality.apply_quality_rerank`
   - fails open with debug logging so malformed optional metadata never breaks search
4. Wired the helper into `src/memo/memory/search_ops.py`:
   - after the existing health-score pass
   - before co-recall/reference-floor stages
   - with trace emission as `quality_rerank`

## Verification
- `uv run --no-sync pytest tests/test_quality_search.py -v`
  - Result before implementation: `2 failed` with missing `_apply_quality_rerank`, as expected
- `uv run --no-sync pytest tests/test_quality.py tests/test_quality_search.py -v`
  - Result: `8 passed`
- `uv run --no-sync pytest tests/test_cli_debug_recall.py::test_rank_hits_explain_none_path_is_identical -v`
  - Result: `1 passed`

## Self-Review
- The new helper is double-gated intentionally:
  - direct helper calls remain safe and testable
  - the explicit search pipeline stays default-off without changing other paths
- No ambient recall hot path wiring was added.
- No project/scope compaction behavior was added.
- No markdown memories or storage behavior were changed.
- No issues found in the scoped diff during review.

## Commit
- Commit message: `feat: wire quality rerank into search`

## Review Fixes

Status: DONE  
Date: 2026-07-08

### Findings addressed
1. Moved quality rerank behind an explicit per-call opt-in on `Memory.search(...)`.
   - Added keyword-only `quality_rerank: bool | None = None` in `src/memo/memory/search_ops.py`.
   - The shared search path now applies the rerank stage only when both conditions hold:
     - the caller explicitly opted in with `quality_rerank=True`
     - `MEMO_QUALITY_RERANK=1`
   - Ambient recall and other existing `search()` callers remain unchanged by default.
2. Wired explicit surfaces instead of the shared ambient path.
   - `src/memo/cli_search.py` now opts `memo search` into quality rerank.
   - `src/memo/memory/ask_ops.py` now opts the ask retrieval path, including multi-round refinement searches, into quality rerank.
   - Recall hook / daemon call sites were not changed.
3. Replaced helper-only coverage with real pipeline coverage.
   - `tests/test_quality_search.py` now verifies:
     - `search_with_trace()` does not emit `quality_rerank` by default, even when the flag is on
     - `search_with_trace()` emits `quality_rerank` only when explicit opt-in and the flag are both on
     - the ask retrieval context calls `search(..., quality_rerank=True)`

### Verification
- `uv run --no-sync pytest tests/test_quality.py tests/test_quality_search.py -v`
  - Result: `10 passed`
- `uv run --no-sync pytest tests/test_cli_debug_recall.py::test_rank_hits_explain_none_path_is_identical -v`
  - Result: `1 passed`
- `uv run --no-sync ruff check src/memo/memory/search_ops.py src/memo/memory/ask_ops.py src/memo/cli_search.py tests/test_quality_search.py`
  - Result: `All checks passed`

## Review Fixes 2

Status: DONE  
Date: 2026-07-08

### Findings addressed
1. Wired explicit MCP search surfaces into the existing per-call quality-rerank opt-in.
   - `src/memo/server_core_search.py` now passes `quality_rerank=True` from:
     - `memo_search(...)` when calling `memory.search(...)`
     - `memo_search(...)` when calling `memory.search_by_file(...)`
     - `memo_search_trace(...)` when calling `memory.search_with_trace(...)`
   - Ambient recall remains unchanged because the opt-in is still only applied by explicit callers.
2. Preserved wrapper behavior in `Memory.search_by_file(...)` while carrying the opt-in through.
   - Added keyword-only `quality_rerank: bool | None = None` to `src/memo/memory/search_ops.py`.
   - Forwarded the value to the fallback plain-search path and the by-file candidate-pool search.

### Focused coverage
- Updated `tests/test_search_by_file.py` to verify:
  - MCP `memo_search(file=...)` forwards `quality_rerank=True` to `search_by_file(...)`
  - MCP `memo_search(...)` forwards `quality_rerank=True` to `search(...)`
  - MCP `memo_search_trace(...)` forwards `quality_rerank=True` to `search_with_trace(...)`
  - `Memory.search_by_file(...)` forwards `quality_rerank=True` to its internal `search(...)` call

### Verification
- `uv run --no-sync pytest tests/test_quality_search.py tests/test_search_by_file.py tests/test_server.py -v`
- `uv run --no-sync ruff check src/memo/server_core_search.py src/memo/memory/search_ops.py tests/test_search_by_file.py`
