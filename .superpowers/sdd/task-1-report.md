# Task 1 Report - SQLite Resource Hygiene Guard

## Status
DONE

Implemented Task 1 only. The initial reproduction did not fail, so no product
cleanup fix was applied. The work adds a focused regression test file proving
`Memory.close()` releases SQLite connections and remains idempotent after lazy
store connection creation.

## Initial Reproduction
Command:

```bash
PYTHONTRACEMALLOC=1 uv run --no-sync pytest \
  tests/test_resume_episodes.py::test_mcp_episodes_search_tool \
  tests/test_runtime_isolation.py::test_install_slash_claude_proceeds_to_add_when_remove_fails \
  -q -W error::ResourceWarning
```

Result before implementation: PASS

Output summary:

```text
2 passed in 2.61s
```

Because the reproduction passed, the warning appears either already fixed or
dependent on broader suite interleaving. I continued with the focused guard
tests as required by the brief.

## Implementation
Created `tests/test_sqlite_resource_hygiene.py` with two focused tests:

- `test_memory_close_releases_sqlite_connections`
- `test_memory_close_is_idempotent_after_lazy_connections`

Both tests capture `ResourceWarning`, force garbage collection after
`Memory.close()`, and assert that no `unclosed database` warning was emitted.

The brief's sample fake embedding used a four-dimensional vector, but the
current isolated `tmp_cfg` uses `embedder_dims=1024`. The test therefore builds
the fake embedding from `tmp_cfg.embedder_dims` so it exercises the SQLite
lifecycle instead of failing at embedding validation.

## Files Changed
- `tests/test_sqlite_resource_hygiene.py` (new)
- `.superpowers/sdd/task-1-report.md` (updated for this task report)

No changes were made to:

- `src/memo/store/connection.py`
- `tests/conftest.py`
- `tests/test_runtime_isolation.py`
- `tests/test_resume_episodes.py`

## Tests Run
Focused guard:

```bash
uv run --no-sync pytest tests/test_sqlite_resource_hygiene.py -q -W error::ResourceWarning
```

Initial result after adding the brief's literal sample: FAIL

Failure reason:

```text
ValueError: embedding dim mismatch: got 4, want 1024
```

This was a test fixture mismatch, not a SQLite cleanup failure. After adapting
the fake embedding to `tmp_cfg.embedder_dims`, the same command passed:

```text
2 passed in 1.51s
```

Final reproduction rerun:

```bash
PYTHONTRACEMALLOC=1 uv run --no-sync pytest \
  tests/test_resume_episodes.py::test_mcp_episodes_search_tool \
  tests/test_runtime_isolation.py::test_install_slash_claude_proceeds_to_add_when_remove_fails \
  -q -W error::ResourceWarning
```

Result:

```text
2 passed in 2.45s
```

Lint:

```bash
uv run --no-sync ruff check tests/test_sqlite_resource_hygiene.py
```

Result:

```text
All checks passed!
```

## Commit
This report is included in the task commit. The requested commit message is:

```text
test: guard sqlite resource cleanup
```

## Concerns
- The original reproduction passed before implementation, so this task guards
  the lifecycle directly but does not prove the full-suite interleaving warning
  source.
- The working tree had pre-existing uncommitted edits in
  `.superpowers/sdd/progress.md` and `.superpowers/sdd/task-1-brief.md`; these
  were left untouched.
