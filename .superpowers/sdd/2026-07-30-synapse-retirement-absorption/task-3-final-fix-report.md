# Task 3 final fix report — rendered briefing provenance

## Status

Complete. The unified-briefing fallback now keeps native and operational
rendering results separate and collects source IDs only for sections that
actually emitted lines.

## Regression

Added a focused case where the native section is empty while the operational
section renders a focus record. Before the fix, parity was blocked because the
fallback also reported an unrendered durable ID. After the fix, the row contains
only the rendered operational ID and passes.

## Verification

- `uv run --no-sync pytest tests/tools/test_synapse_parity.py tests/test_memory_ask.py tests/test_memory_evidence_pack.py -q`
  — 40 passed.
- `uv run --no-sync ruff check src/memo/memory/ask_ops.py src/memo/memory/evidence_ops.py src/memo/server_core_search.py tools/memflow_absorption tests`
  — passed.
- `uv run --no-sync mypy src/memo`
  — passed, 463 source files.
- `git diff --check`
  — passed.

## Concerns

None. Pre-existing edits to unrelated SDD ledgers and reports were preserved
and excluded from this change.
