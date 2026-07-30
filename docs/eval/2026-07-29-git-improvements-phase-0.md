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
