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

---

# Task 6 Report: Continuity detector (open-loop nudges) — memo Proactive Engine

**Program:** memo Proactive Engine (`feat/proactive-engine`) — distinct from the
Quality Compaction report above (same task number, different SDD run/program).

## Status: DONE

Commit: `b81c38d2` — "feat(proactive): continuity detector (open-loop nudges)"

## Summary

Implemented `detect_continuity(mem, *, now, limit=5) -> list[Nudge]` in
`src/memo/proactive/detectors/continuity.py`, exactly per the brief: calls
`mem.open_loops(limit)`, and emits one `KIND_CONTINUITY` `Nudge.make(...)`
per `(memo_id, text)` pair, with `evidence=(mid,)`, `urgency=0.4`,
`value=0.7`, no `action`. Guarded: `try/except Exception -> []`, logged at
`_log.debug`. Same shape as Task 5's `detect_reliability`.

TDD: wrote `tests/test_proactive_continuity.py` with the brief's two tests
(fake mem with `open_loops(limit)`, and a `Boom` class raising in
`open_loops`) — confirmed `ModuleNotFoundError` (RED) before creating the
implementation, then both passed (GREEN): `test_continuity_wraps_open_loops`,
`test_continuity_guarded`.

## Real facade accessor: ADDED (not deferred)

Investigated per the brief's note: `src/memo/briefing.py:371-377` computes
"open loops" as

```python
cutoff = (datetime.now(tz=UTC) - timedelta(days=loops_days)).isoformat()
all_recent = mem.store.list_recent(limit=loops_n * 4, exclude_types={"reference", "secret"})
open_loops = [r for r in all_recent if (r.get("updated") or "") >= cutoff][:loops_n]
```

`mem.store` is `VecStore` (`src/memo/store/queries.py:1041`), and
`list_recent` already accepts an `updated_since` SQL parameter
(`clauses.append("coalesce(julianday(updated), -1e300) >= julianday(?)")`)
that does the same cutoff filtering server-side — briefing.py's
overfetch-then-filter (`limit * 4` then Python-side slice) predates that
parameter or simply never adopted it. Since the exact filter is already a
first-class SQL argument, a thin wrapper needs no 4x overfetch and no
Python-side date comparison:

```python
def open_loops(self, limit: int = 5, *, days: int = 7) -> list[tuple[str, str]]:
    cutoff = (datetime.now(tz=UTC) - timedelta(days=days)).isoformat()
    rows = self.store.list_recent(
        limit=limit, exclude_types={"reference", "secret"}, updated_since=cutoff,
    )
    return [(r.get("id") or "", r.get("title") or "—") for r in rows]
```

Added to `src/memo/memory/facade.py` (12 lines incl. docstring), right after
the existing `project` property — the facade's home for small methods that
don't belong to any single op mixin (per the module docstring). Unlike Task
5's `superseded_pairs()` (which needed a real plumbing decision — disk scan
of archived `.md` files vs. a new persistent index), this genuinely is the
"thin ≤20-line wrapper over an existing query" case the brief's escape hatch
anticipated, so it was added rather than deferred.

Added a third test, `test_real_facade_open_loops_wraps_recent_memory`, using
the real `Memory` facade via the shared `mock_memory` fixture (stubbed
embedder, isolated `tmp_cfg`): saves a memory, calls `mock_memory.open_loops(5)`,
asserts `(rec.id, rec.title)` is in the result. All 3 tests pass.

## Lint + type-check

```
$ uv run --no-sync ruff check src/memo/proactive/detectors/continuity.py src/memo/memory/facade.py tests/test_proactive_continuity.py
All checks passed!

$ uv run --no-sync ruff format --check src/memo/proactive/detectors/continuity.py src/memo/memory/facade.py tests/test_proactive_continuity.py
3 files already formatted

$ uv run --no-sync mypy src/memo/proactive/detectors/continuity.py src/memo/memory/facade.py
Success: no issues found in 2 source files

$ uv run --no-sync ruff check --select C901 src/memo/proactive/detectors/continuity.py src/memo/memory/facade.py
All checks passed!
```

## Regression checks

```
$ uv run --no-sync pytest tests/test_proactive_continuity.py tests/test_proactive_reliability.py tests/test_proactive_nudge.py tests/test_proactive_store.py tests/test_proactive_arbiter.py -q
12 passed in 0.16s

$ uv run --no-sync pytest tests/ -q -m "not slow" -k "facade or briefing" --ignore=tests/test_proactive.py
47 passed, 4 skipped, 4829 deselected in 1.50s
```

`tests/test_proactive.py` was excluded — it fails to collect
(`ImportError: cannot import name 'ProactiveSuggester' from 'memo.proactive'`)
on a pre-existing, unmodified file from an earlier unrelated feature
(`0732bbc2`/`44c243a6`, before this branch), confirmed via `git status
--short` (clean) and `git log` (no recent touch) — not caused by this task.

## Shared working tree discipline

- Staged only `src/memo/proactive/detectors/continuity.py
  tests/test_proactive_continuity.py src/memo/memory/facade.py` — verified
  via `git status --short` before commit that concurrent sessions' in-flight
  files (`.superpowers/sdd/progress.md`, `task-{1,2,3,4}-*.md`) were NOT
  swept in.
- No `git add -A`, no `ruff format src/`, no `reset`/`checkout`.
- This report file already contained an unrelated Task 6 report ("Quality
  Compaction Apply And Undo Receipt"). Per the established convention (see
  `task-5-report.md`), appended this section at the bottom with a clear
  separator/banner rather than overwriting — original content fully
  preserved above.

## Concerns

None blocking. `Memory.open_loops()` now has one caller path ready
(`detect_continuity`), but nothing in production calls `detect_continuity`
itself yet — that's Task 8 (engine) / Task 11 (dream). The pre-existing
`tests/test_proactive.py` collection failure (unrelated `ProactiveSuggester`
import) is a known gap in the shared tree, not introduced or fixed by this
task — flagged for whoever owns that file's cleanup.
