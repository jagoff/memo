# Task 2 Brief: Source Audit Helper For Flags And Exceptions

Plan: docs/superpowers/plans/2026-07-09-memo-deep-improvement-first-sprint.md

## Goal

Add a small standard-library source-audit helper and contract tests that make
raw `MEMO_*` environment reads and selected broad `except Exception` sites
explicitly classified.

## Global Constraints

- Do not rewrite retrieval ranking.
- Do not flip HyDE, MMR, graph, or capture defaults.
- Do not bulk-edit memory records.
- Do not delete or weaken memory records.
- Do not restructure the whole CLI/MCP surface.
- Do not chase coverage percentage with low-value tests.
- Do not eliminate every broad exception handler.
- Use `MemoError` subclasses for normal user-visible domain errors.
- Behavioral `MEMO_*` flags go through `src/memo/flags.py`; storage/model
  config remains in `src/memo/config.py`.
- Keep `mlx` / `mlx_lm` imports deferred.

## Files

- Create: `src/memo/dev_audit.py`
- Create: `tests/test_dev_audit.py`
- Create: `docs/engineering/exception-policy.md`

Do not edit other files unless a focused test failure proves the task cannot be
completed otherwise.

## Interfaces

- `RawMemoEnvRead`
- `BroadExceptionSite`
- `RAW_MEMO_ENV_ALLOWED`
- `BROAD_EXCEPTION_ALLOWED`
- `find_raw_memo_env_reads(root: Path) -> list[RawMemoEnvRead]`
- `find_broad_exception_sites(root: Path) -> list[BroadExceptionSite]`

`src/memo/dev_audit.py` must use only the Python standard library.

## Steps

1. Create `tests/test_dev_audit.py` exactly from the Task 2 section of the plan.
2. Run `uv run --no-sync pytest tests/test_dev_audit.py -q` and confirm it fails
   because `memo.dev_audit` does not exist.
3. Implement `src/memo/dev_audit.py` from the plan.
4. Add `docs/engineering/exception-policy.md` from the plan.
5. Run:

   ```bash
   uv run --no-sync pytest tests/test_dev_audit.py::test_broad_exception_policy_targets_are_classified -q
   ```

   If the broad-exception line baseline has shifted, regenerate only that set
   with the command in the plan and keep the target-file scope unchanged.
6. Run:

   ```bash
   uv run --no-sync pytest tests/test_dev_audit.py -q
   uv run --no-sync ruff check src/memo/dev_audit.py tests/test_dev_audit.py
   ```

7. Commit with message exactly:

   ```text
   test: audit memo env and exception policy
   ```

8. Write a report to `.superpowers/sdd/task-2-report.md` with status, commit,
   files changed, tests run, any regenerated baseline, and concerns.

## Expected Behavior

- Raw `os.environ.get("MEMO_*")` and `_os.environ.get("MEMO_*")` calls are found.
- Only allowed raw env reads are accepted by the test.
- Broad `except Exception` sites in the four target files listed by the plan are
  checked against the explicit baseline.
- The policy doc contains the phrases `hook hot path`, `user-visible CLI`, and
  `destructive write paths`.
