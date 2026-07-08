# Task 2: Corpus Trust Checks — Completion Report

Status: DONE  
Date: 2026-07-08  
Branch: `trust-adoption-doctor`

## Outcome
- Extended `memo/usefulness_doctor.py` with read-only trust diagnostics over existing store and grounding signals.
- Added focused TDD coverage in `tests/test_usefulness_doctor.py` for:
  - support-count starvation when `memory_health` rows exist but no `support_count` is positive.
  - grounded invalidated memories surfacing as critical trust findings.
- Preserved the existing report shape while adding trust summary counters:
  - `summary["memory_health_rows"]`
  - `summary["support_count_positive"]`
  - `summary["grounded_memory_ids"]`

## Files
- Modified: `src/memo/usefulness_doctor.py`
- Modified: `tests/test_usefulness_doctor.py`

## Validation
- `PYTHONPATH=src /Users/fer/repos/memo/.venv/bin/python -m pytest tests/test_usefulness_doctor.py -v -k "support_count or invalidated"`
  - Result: `2 failed` before implementation, as expected for TDD red phase.
- `PYTHONPATH=src /Users/fer/repos/memo/.venv/bin/python -m pytest tests/test_usefulness_doctor.py -v`
  - Result: `5 passed`
- `PYTHONPATH=src /Users/fer/repos/memo/.venv/bin/python -m pytest tests/test_usefulness_doctor.py tests/test_support_count.py tests/test_cli_invalidate.py -v`
  - Result: `25 passed`
- `/Users/fer/repos/memo/.venv/bin/ruff check src/memo/usefulness_doctor.py tests/test_usefulness_doctor.py`
  - Result: `All checks passed!`

## Commit
- Commit message: `feat(usefulness): report corpus trust diagnostics`

## Notes
- Trust checks remain read-only: they open `Memory(cfg)` for signal inspection only and do not mutate corpus state.
- Grounded-memory resolution intentionally maps the 8-char `grounding.log` prefixes back to full ids via `store.all_ids()` before fetching metadata and health rows.

## Review Fix Follow-up
- Reworked `src/memo/usefulness_doctor.py` to remove `from memo.memory import Memory` and all `Memory`/store constructor usage.
- Trust checks now open `cfg.db_path` through stdlib `sqlite3` with `file:{cfg.db_path}?mode=ro`, so the doctor stays read-only and does not bootstrap schema/state paths.
- Missing or unreadable DB now yields a partial report with `store_unavailable` in `unknown` status instead of creating state or failing the command.
- Grounded-memory trust checks now resolve directly from `meta` + `memory_health`, with defensive JSON parsing for `tags` and `extra_json`.
- Reused `memo.dashboard.GROUNDED_SCORE` instead of keeping a hardcoded grounding threshold.

## Review Fix Validation
- `PYTHONPATH=src /Users/fer/repos/memo/.venv/bin/python -m pytest tests/test_usefulness_doctor.py tests/test_support_count.py tests/test_cli_invalidate.py -v`
  - Result: `28 passed`
- `/Users/fer/repos/memo/.venv/bin/ruff check src/memo/usefulness_doctor.py tests/test_usefulness_doctor.py`
  - Result: `All checks passed!`

## Added Regression Coverage
- Doctor module source does not import `memo.memory`.
- `build_report()` with a missing DB leaves `state_dir` and `memvec.db` absent.
- Grounded-memory row parsing tolerates valid-but-wrong JSON shapes in `tags` / `extra_json`.
