# Task 6 Report: Quality Compaction Apply And Undo Receipt

## Status

DONE

## Scope

Implemented Task 6 in `/Users/fer/repos/memo` for explicit quality compaction apply flow with receipt persistence and undo integration, while preserving Task 5 preview behavior and safety boundaries.

## Changes

### `src/memo/quality_compact.py`

- Added `apply_quality_compaction(memory, proposals, *, dry_run=False)`.
- Archives proposal `source_ids` through `memory.lifecycle.archive_memory(...)`.
- Preserves the existing canonical pointer when available from record metadata, and falls back to the proposal id only when no canonical marker is present.
- Returns receipt fragments in the form:
  - `quality_compacted`: per-proposal archived ids
  - `errors`: per-source apply failures

### `src/memo/cli_maintain.py`

- Added `_prepare_quality_compact_receipt_paths(cfg)` to fail closed before mutation if receipt/undo persistence cannot be set up.
- Extended `_undo_targets(receipt)` to restore archived ids recorded under `quality_compacted`.
- Updated `maintain quality-compact` so:
  - `MEMO_QUALITY_COMPACT=1` is still required.
  - `--apply` is now supported.
  - `--preview` behavior is preserved.
  - `--preview --apply` remains rejected.
  - apply mode first computes the Task 5 preview receipt, then:
    - refuses to mutate when preview validation already produced errors
    - applies archivals through the lifecycle API
    - writes `last.json` and a timestamped run receipt containing `quality_compacted`
- Added rollback protection for late receipt-write failures:
  - if receipt persistence fails after archiving, the command restores the archived ids immediately and reindexes before surfacing the failure

### Tests

Updated focused coverage in:

- `tests/test_quality_compact.py`
  - apply mode JSON receipt shape
  - apply receipt persistence to `state/maintain/last.json`
  - mutually exclusive `--preview` and `--apply`
- `tests/test_maintain.py`
  - undo target extraction now includes `quality_compacted`
  - restore path covers compaction-archived ids

## Verification

Passed:

```bash
uv run --no-sync ruff check src/memo/quality_compact.py src/memo/cli_maintain.py tests/test_quality_compact.py tests/test_maintain.py
uv run --no-sync pytest tests/test_quality_compact.py tests/test_maintain.py::test_maintain_undo_cli_dry_run_reads_receipt tests/test_maintain.py::test_undo_targets_and_restore_from_inactive -v
```

## Safety Notes

- Apply remains explicit and gated by `MEMO_QUALITY_COMPACT=1`.
- Markdown memories are not deleted.
- Compaction still stays within Task 5's strict scope/project/sensitive filtering because apply only consumes preview-approved proposals.
- Undo restores archived ids from `quality_compacted` receipt entries.
- Receipt/undo setup is checked before mutation, and receipt write failure after mutation triggers rollback.

## Commit

- `aee103c feat: apply quality compaction with undo`

## Self-review

No blocking issues found in the implemented scope.
