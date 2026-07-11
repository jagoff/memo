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

---

## Review Fixes

Addressed the two follow-up findings from review:

- Receipt persistence for `maintain quality-compact --apply` is now published atomically from the user's point of view. The timestamped run receipt and `last.json` are both staged to temp files first, the run receipt is replaced first, and a later `last.json` replace failure now removes the new run receipt before surfacing the error. Rollback leaves the prior `last.json` content intact.
- Quality compaction apply receipts now record `attempted_ids` alongside `archived_ids`, and rollback uses those attempted ids instead of only successful archives. That closes the gap where `archive_memory()` could move a file and then raise before returning `True`.

### Additional Tests

- `test_quality_compact_apply_receipt_publish_is_atomic`
  - seeds a real compaction candidate
  - simulates the second receipt publish failure (`last.json`)
  - asserts the prior `last.json` survives, no timestamped run receipt remains, and archived memory is restored
- `test_quality_compact_apply_rolls_back_attempted_archive_ids`
  - simulates `archive_memory()` moving the file and then raising
  - simulates the later receipt publish failure
  - asserts rollback still restores the attempted source id from inactive

### Verification

Passed:

```bash
uv run --no-sync ruff check src/memo/quality_compact.py src/memo/cli_maintain.py tests/test_quality_compact.py tests/test_maintain.py
uv run --no-sync pytest tests/test_quality_compact.py tests/test_maintain.py::test_maintain_undo_cli_dry_run_reads_receipt tests/test_maintain.py::test_undo_targets_and_restore_from_inactive -v
```
# Task 6 Report

## Completed

- First-run setup now treats Markdown configuration as configured state and skips the picker when the Markdown index or config directory exists.
- The picker persists default Markdown config files through `write_default_config()`.
- `memo.setup` exports Markdown config helpers while retaining legacy TOML helpers for compatibility.
- Init tests now isolate `MEMO_CONFIG_DIR`, verify Markdown storage output, and cover the Markdown first-run gate.

## Verification

- `uv run --no-sync pytest tests/test_cli_init.py -v` (7 passed)
- `uv run --no-sync ruff check src/memo/cli.py src/memo/setup/__init__.py tests/test_cli_init.py` (passed)

## Follow-up

`memo init` overwrite confirmation remains implemented in `src/memo/runtime/install.py` and currently checks only the legacy TOML path. That file was outside Task 6 ownership, so the retained confirmation test is explicitly legacy-compatibility coverage; Markdown overwrite confirmation needs a coordinated follow-up there.
