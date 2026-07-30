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

## 2. Immutable revisions, refs, verifier, and recovery

Decision: **Admit only the exact-delete-history slice; Defer the revision archive**

Observed gap: time-machine reconstruction before a later delete returns the record but marks
its body unavailable. Memo already has `HistoryStore`, `VersionStore`, and
`ContentAddressedArtifactStore`; adding a fourth historical authority before convergence
would violate the maintainability gate. The existing concurrent save/update/store baseline
passes (`3 passed`), so no general CAS rewrite is justified by Phase 0.

Smallest admitted slice: preserve the canonical pre-delete body and tags in the existing
append-only delete event and teach `time_machine.reconstruct` to consume that snapshot.

Primary gate: reconstruction immediately before a later delete returns the exact body and
tags, while old history rows without snapshots remain explicitly unavailable.

Guardrails: current Markdown authority is unchanged; no new database or public command;
legacy rows remain readable; hard-delete behavior and portable backup remain compatible.

Deferred slice: immutable revision envelopes, heads, reflog, `fsck`, lost-found, and
historical-store retirement require a separate shadow-write/storage-amplification plan.

Rollback: readers ignore the additive delete-event fields; old and new history databases
remain readable.

## 4. Retrieval planning and backend filter pushdown

Decision: **Admit, narrowed**

Observed gap: BM25/exact/fuzzy candidate generation spends its limit before applying common
date and excluded-tag filters. Both adversarial `limit=1` cases reproduced the crowd-out,
while vector search already widens or pushes the same predicates.

Smallest admitted slice: extend existing BM25/fuzzy store methods with the two common
predicates and over-fetch only when a backend cannot execute them before its own limit.
Do not add `SearchPlan`, `SearchFilter`, or a planner class.

Primary gate: all affected modes return the same eligible record in the adversarial matrix
at `limit=1`; candidate count remains bounded.

Guardrails: committed recall labels lose no expected ID; ranking is unchanged when filters
are absent; warm p50 is within 5% of baseline; no new flag is needed for a correctness fix.

Rollback: revert the extra store parameters; no data migration exists.

## 3. Explicit operation context and plumbing/porcelain

Decision: **Defer**

Evidence: `Config`, explicit `ActorIdentity`, native trace scope, project tags, and the
raw-environment AST ratchet already constrain the important boundaries. The audit passed
(`4 passed`), and no cross-vault, cross-principal, or signature-growth failure was
reproduced.

Re-entry gate: a concrete context-propagation bug or a measured reduction in repeated
parameters/ambient reads must be shown before introducing a context type.

## 5. Structured tracing plus perf/fuzz/fault discipline

Decision: **Defer runtime tracing; Admit test discipline inside admitted slices**

Evidence: Memo already propagates a native trace ID and `search_with_trace` exposes
candidate stages. The trace/search/maintenance baseline passed (`57 passed`). No diagnosis
incident was reproduced that requires a second event system, and no sampled-trace overhead
budget has been measured.

Re-entry gate: a fault that cannot be localized with existing receipts/traces plus a
disabled/sampled overhead benchmark. Admitted transaction, history, and retrieval slices
must still add deterministic fault and complexity tests.

## 6. Need-driven incremental maintenance

Decision: **Defer**

Evidence: maintenance entry points are fragmented across `maintain`, `dream`,
`maint-daemon`, session idle work, and the runtime sleep cycle, but Phase 0 has no receipt
proving incompatible overlap, avoidable work, or a lock/recovery failure. A registry now
would be a wrapper around live schedulers and fail the consolidation gate.

Re-entry gate: two existing paths must be shown scheduling the same or incompatible work,
with before/after code and execution counts that a single task descriptor can reduce.
