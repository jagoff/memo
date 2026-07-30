# Git-derived improvements: Phase 0 admission receipt

Date: 2026-07-29
Base commit: `0c3224776b74e2115a21b41ca09434212dfefb69`
Design: `docs/superpowers/specs/2026-07-29-git-codebase-improvements-design.md`

## Baseline and frozen gates

- Environment: editable install with `dev` and `http` extras.
- Focused HTTP baseline: `13 passed in 3.82s`.
- Non-slow baseline: `6065 passed, 18 skipped in 41.38s`.
- Correctness: no existing passing test may regress.
- Retrieval quality: no expected ID may be lost from the committed recall labels.
- Retrieval latency: warm p50 may not regress by more than 5%; any result within the
  benchmark noise band is treated as no improvement.
- Write latency: warm p50 may not regress by more than 10% for a correctness fix.
- Storage: a narrowed history fix may add at most one compressed/current body snapshot
  per destructive event; a content-addressed revision experiment must publish its own
  measured amplification before admission.
- Maintainability: an admitted abstraction must consolidate or delete duplicated logic;
  a wrapper that leaves all previous mechanisms intact fails.
- Rollback: disabling or reverting a new path must not delete current Markdown or require
  a destructive downgrade.

Baseline note: a second full xdist run reached 100% test completion but stalled while
pytest recursively cleaned an old numbered temporary directory; it was interrupted during
`pytest_sessionfinish`. The earlier clean run above is the product baseline, and focused
reruns are used below so temporary-directory cleanup is not mistaken for a code failure.

## 1. Atomic mutation and quarantine

Decision: **Admit, narrowed**

Observed gap: a seeded failure on the second save leaves the first federation record in
current Markdown/searchable state. The current implementation then continues to the
operational-journal seam even though the durable-memory half of the import failed.

Smallest admitted slice: make one federation bundle import all-or-nothing in final current
state, keep operational events unapplied on memory failure, and emit an honest rollback
receipt. The full cross-process read-barrier transaction and general importer framework are
deferred until this slice proves value.

Primary gate: after a seeded failure at every imported item boundary, zero bundle records
are current/searchable and zero bundle operational events are imported.

Guardrails: successful and idempotent imports keep their current API; rollback failure is
typed and fail-closed; no second transaction framework or daemon is introduced.

Rollback: revert the federation-only rail; current Markdown remains readable and no schema
downgrade is required.
