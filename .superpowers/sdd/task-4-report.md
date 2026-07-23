### Task 4 report: Arbiter (score + budget + urgent gating)

STATUS: DONE
Commit: 7a3be4ce
Tests: `pytest tests/test_proactive_arbiter.py -v` — 3 passed (urgent-only-reliability-over-threshold-and-can-push, no-push-when-cannot-push, floored-multiplier-keeps-reliability-visible); ruff check + ruff format --check + mypy clean on both files.
Concerns: brief's Step-1 test snippet imports `score` alongside `route` but never calls it directly, which trips ruff F401 (unused import) — a global constraint of this task. Dropped the unused `score` import from the test file (lint-only change; all assertions/values kept verbatim). No other deviations from the brief.
