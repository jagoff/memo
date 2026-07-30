# Task 6 TOCTOU correction report

## Status

The three requested retirement-audit TOCTOU findings are closed. The audit
remains read-only and retirement cleanup remains refusal-only.

## Changes

- Directory snapshots are now re-listed and every original entry is re-statted
  through the retained directory descriptor before recursion unwinds.
  Membership changes, identity replacements, and regular-file metadata changes
  fail closed.
- Regular files are opened with `O_NOFOLLOW` relative to their retained parent
  descriptor. Their entry observation, opened descriptor identity/metadata,
  post-read descriptor metadata, byte count, and final parent-relative
  observation must agree. Traversal consumers use the validated bytes directly
  and never re-resolve the yielded `Path`.
- Every absolute root component is observed and opened from `/`, one component
  at a time, with `O_DIRECTORY | O_NOFOLLOW`. All parent descriptors are
  retained and every component link is re-statted after the scan, so an
  ancestor replacement cannot leave a detached tree reported as the live root.

## Race regression evidence

Before the implementation, the five new race cases failed:

```text
5 failed, 22 deselected
```

The same cases now pass:

```text
5 passed, 22 deselected
```

They cover post-list membership insertion, post-list identity replacement,
file replacement between classification and open, in-place file metadata
change during reading, and root-ancestor replacement during the scan.

## Host portability

The macOS host reports descriptor-relative support for both `os.open` and
`os.stat`, `follow_symlinks=False` support for `os.stat`, and provides
`O_DIRECTORY` plus `O_NOFOLLOW`. `os.listdir(fd)`, `os.fstat`, and `os.read`
are also exercised by the passing regressions.

## Verification

```text
uv run --no-sync pytest tests/tools/test_retirement_audit.py -q
27 passed

uv run --no-sync pytest tests/tools/test_retirement_audit.py tests/tools/test_absorption_*.py -q
66 passed

uv run --no-sync pytest tests/tools -q
167 passed

uv run --no-sync ruff check tools/memflow_absorption tests/tools/test_retirement_audit.py
All checks passed!

uv run --no-sync ruff format --check tools/memflow_absorption/inventory.py tests/tools/test_retirement_audit.py
2 files already formatted

uv run --no-sync mypy tools/memflow_absorption
Success: no issues found in 15 source files

git diff --check
passed
```

## Scope

Only `tools/memflow_absorption/inventory.py`,
`tests/tools/test_retirement_audit.py`, and this report were changed. No
cleanup, deletion, service, LaunchAgent, runtime, configuration, state,
repository, release, or documentation mutation was performed.
