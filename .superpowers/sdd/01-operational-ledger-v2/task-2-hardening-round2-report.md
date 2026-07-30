# Task 2 hardening round 2 implementation report

Status: implementation delivered for independent re-review. This report does
not mark Task 2 complete or accepted.

## Revision

- Base: `123cd8f6`
- Technical commit: `ed454393`
- Commit subject: `fix(ledger): finalize crash-safe transactions`
- Frozen v1 paths were unchanged.

## RED evidence

The initial round-2 regression set was run before production edits:

```text
9 failed, 2 passed in 0.62s
```

The failures reproduced partial marker visibility, phase replay, unverified
APPLIED recovery, retained stage blobs, compaction epoch regression, missing
predecessor history, crossed authority roots, canonical mapping drift, and
authority descriptor loss. Additional cleanup-boundary, bounded-retention,
predecessor-staging, and retained roster/epoch tests were added while closing
the same findings.

## Implementation

- `create_bytes_exclusive` now fsyncs a temporary regular file, installs the
  final name atomically with no-replace hard-link semantics, removes the
  temporary name, and fsyncs the parent directory.
- Transaction markers carry a mandatory `committed` or `applied` phase.
  Recovery validates the expected phase, all staged bytes, and every published
  target digest and size.
- Verified transactions finalize to canonical receipts capped at 256 entries.
  Stage trees are retired descriptor-relatively, and recovery can complete a
  crash after receipt creation without replaying or leaking stage blobs.
- Every accepted anchor is retained as immutable history keyed by origin and
  anchor hash. Compaction and import transactions stage the exact predecessor
  history. Legacy recovery fully validates that predecessor and applies the
  shared monotonic transition check.
- The ledger rejects crossed roster and epoch roots before acquiring locks.
  Admission retains one authority-root descriptor; nested roster, epoch, pin,
  lock, read, and write paths reuse it across pathname rename or replacement.
- Canonical encoding now omits supported optional defaults identically for
  dataclasses and their public mapping/asdict representations without erasing
  unrelated payload fields.

## GREEN evidence

```text
Focused round-2 regressions:
11 passed in 0.44s
3 cleanup/retention/history tests passed in 0.49s

Authority/key/roster/epoch contracts:
61 passed in 1.24s

Task 1 + Task 2 contracts:
200 passed in 37.89s

Frozen v1:
3 passed in 0.43s
No diff in frozen v1 paths from 123cd8f6..ed454393

Ruff:
All checks passed

mypy:
Success: no issues found in 6 source files

Full non-slow:
6009 passed, 18 skipped, 19 warnings in 590.93s

git diff --check:
passed
```

The warnings are the existing Python/macOS warning for `fork()` from
multi-threaded xdist workers; all crash-boundary tests passed.

## Re-review

Independent specification, durability, and quality re-review is still
required on `123cd8f6..ed454393`. No acceptance is claimed here.
