# Task 2: Source Audit Helper For Flags And Exceptions

Status: DONE
Date: 2026-07-09

## Outcome

- Added `memo.dev_audit` with standard-library AST scanners for raw `MEMO_*`
  environment reads and broad `except Exception` handlers.
- Added contract tests that require raw env reads and target broad-exception
  sites to be explicitly classified.
- Added the engineering policy doc for broad exceptions and raw env reads.

## Files Changed

- `src/memo/dev_audit.py`
- `tests/test_dev_audit.py`
- `docs/engineering/exception-policy.md`
- `.superpowers/sdd/task-2-report.md`

## Verification

- Red test first: `uv run --no-sync pytest tests/test_dev_audit.py -q`
  - Failed as expected with `ModuleNotFoundError: No module named 'memo.dev_audit'`.
- Target baseline check: `uv run --no-sync pytest tests/test_dev_audit.py::test_broad_exception_policy_targets_are_classified -q`
  - Passed.
- Focused tests: `uv run --no-sync pytest tests/test_dev_audit.py -q`
  - Passed, `3 passed`.
- Ruff: `uv run --no-sync ruff check src/memo/dev_audit.py tests/test_dev_audit.py`
  - Passed.

## Baseline

- Broad-exception baseline regeneration was not needed; the planned line set
  still matches this worktree.

## Commit

- Commit message: `test: audit memo env and exception policy`

## Concerns

- None.
