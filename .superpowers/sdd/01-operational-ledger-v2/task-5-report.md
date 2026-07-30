# Task 5 implementation report — exactly-once durable promotion outbox

BASE: `20e16ba4`
Brief commit: `ae46ea83`
Technical commit: `24f7a406`

Status: **ACCEPTED**. Independent specification, durability, and quality
review returned `PASS` with no material defect and no open
`BLOCKER/HIGH/MEDIUM`.

## Delivered

- Replaced the preliminary two-event projection with the closed four-event
  authority: requested, retry-scheduled, completed, and rejected.
- Added immutable canonical promotion intents, stable operation keys and
  request hashes, deterministic retry timing, bounded worker runs, synchronous
  reconciliation, and aggregate outbox diagnostics.
- Made Markdown frontmatter the exactly-once authority through
  `Memory.save_operation`: same key/hash reuses the original record, changed
  requests and multiple claims fail closed, and missing SQLite rows are
  repaired without creating a second Markdown file.
- Required an idempotency key on CLI, MCP, and `promote_learning`; grounded
  each promotion in verified outcome event IDs and a source-derived stable
  promotion timestamp.
- Added monotonic SQLite reducers and deterministic rebuild for all four
  events. SQLite remains a rebuildable projection.
- Preserved the activation boundary: the production v1 facade uses the
  exactly-once Markdown operation seam directly, while the v2 worker is used
  only when Task 7 explicitly installs its capability.

## RED evidence

The new outbox and operation-identity suites were collected before the
production modules existed:

```text
uv run --no-sync pytest \
  tests/test_durable_outbox.py \
  tests/test_write_ops_operation_identity.py -q

2 collection errors
ModuleNotFoundError: No module named 'memo.durable_outbox'
```

The first full non-slow run found one additional policy regression:

```text
1 failed, 6078 passed, 18 skipped
tests/test_dev_audit.py::test_broad_exception_policy_targets_are_classified
```

The recovery seam was narrowed from `except Exception` to the typed
`StorageError` and `sqlite3.Error` failures. The policy test, operation
identity tests, Ruff, and mypy then passed before the full suite was repeated.

## GREEN evidence

```text
Focused final:
490 passed

Required focused matrix including operational idempotency:
500 passed

Task 1–5 cumulative operational/definitive matrix:
676 passed

Ruff over all 15 touched source/test paths:
All checks passed

mypy over all 7 touched source paths:
Success: no issues found

Full non-slow:
6079 passed, 18 skipped, 19 known fork/xdist warnings in 36.99s

git diff --check:
passed
```

Frozen v1 sources remained byte-identical to BASE:

```text
src/memo/operation_ledger.py
55d29af262c1e3547e058505da1f09693dc5eb950f462672ada827b2cb911d9c

src/memo/operational.py
ab607b4ade663c176b70ade04b9d957ea0170e12710c2b070ca6b701461d3702
```

## Compatibility and deferred activation

- No v1 journal, snapshot, or production backend selector changed.
- No v2 durable-outbox capability is installed by production construction.
- No dual-write path was introduced.
- Memflow runtime, launch agents, hooks, state, pending ACKs, and configuration
  were not modified.
- Task 7 still owns v2 activation after Tasks 2–6 pass independent review.

## Review package

Normative range:

```text
20e16ba4..24f7a406
```

Required independent checks:

1. Crash boundaries before/after Markdown save and completion commit.
2. Frontmatter scan, SQLite repair, collision behavior, and symlink/path
   assumptions.
3. Reducer monotonicity, exact event validation, retry determinism, and rebuild
   equivalence.
4. Outcome-event provenance and stable replay across v1 and dormant v2.
5. Frozen-v1 compatibility and absence of activation or Memflow mutation.

## Independent acceptance

Durable review record: Memo `99210fd486024444b3715ef395a24ff0`.

The independent review explicitly returned `Task 5 PASS`. Its only note was
missing symlink/path-confinement test coverage; it identified no material
defect. The normative task did not require that additional test, terminal
symlink/non-file candidates are already rejected, and operation identity uses
the fixed `promotion/<sha256>` namespace rather than caller-controlled paths.
The note is optional LOW test-quality coverage, not an acceptance gap.

Current focused verification after later shared-view changes:

```text
40 passed
```

P01-T05 therefore requires no implementation follow-up or re-review.
