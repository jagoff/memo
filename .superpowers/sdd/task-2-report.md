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
