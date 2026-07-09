# Task 1 Report — Quality Signals And Pure Reranker

## What changed
- Added `src/memo/quality.py` with:
  - `QualityDecision` dataclass
  - `classify_quality(hit)` signal classifier
  - `apply_quality_rerank(hits, explain=None)` pure rerank wrapper
- Added `tests/test_quality.py` with 4 focused RED/GREEN tests for classification and reranking behavior.
- Registered new default-off search flags in `src/memo/flags_search.py`:
  - `MEMO_QUALITY_RERANK`
  - `MEMO_CONTEXT_PACK`

## TDD evidence
- **RED**: before adding `memo.quality`, ran:
  - `uv run --no-sync pytest tests/test_quality.py -v`
  - collection failed with `ModuleNotFoundError: No module named 'memo.quality'`.
- **GREEN**: after implementing `memo.quality`, ran:
  - `uv run --no-sync pytest tests/test_quality.py -v`
  - `4 passed`.

## Tests and outputs
- `uv run --no-sync pytest tests/test_quality.py -v`:
  - 4 tests, 4 passed.
- `uv run --no-sync ruff check src/memo/quality.py src/memo/flags_search.py tests/test_quality.py`:
  - All checks passed.

## Files changed
- `src/memo/quality.py` (new)
- `src/memo/flags_search.py` (modified)
- `tests/test_quality.py` (new)

## Self-review
- Scope was kept to Task 1 files only; no existing behavior was altered and no new ranking path was wired.
- `MEMO_QUALITY_RERANK` and `MEMO_CONTEXT_PACK` are default-off, matching the task constraints.
- `VerificationState` imports in implementation/tests use `memo.tiers` as required.

## Concerns
- `apply_quality_rerank` mutates non-dataclass hit objects in a fallback branch (`hit.score = score`) when `dataclasses.replace` is not applicable; in practice, hit objects in current search paths are expected to be dataclasses.

## Reviewer Finding Fix (Quality Rerank)

### Fix summary
- Removed the in-place mutation fallback in `src/memo/quality.py:apply_quality_rerank`.
- Kept dataclass behavior via `dataclasses.replace`.
- Added strict validation for unsupported hit types and now raise `ValidationError` for non-dataclass or non-replaceable hits before side effects.
- Added `test_apply_quality_rerank_rejects_non_dataclass_hit_without_mutation` in `tests/test_quality.py` to assert non-dataclass hits raise the domain error and retain original `score`.

### Tests run and outputs
- `uv run --no-sync pytest tests/test_quality.py -v`: `5 passed`.
- `uv run --no-sync ruff check src/memo/quality.py tests/test_quality.py`: `All checks passed!`

### Files changed
- `src/memo/quality.py`
- `tests/test_quality.py`
- `.superpowers/sdd/task-1-report.md` (appended reviewer-finding fix section)

### Self-review
- Enforced a pure reranker contract without mutating unsupported shapes.
- Error path is domain-specific (`ValidationError`) and does not alter input objects.
