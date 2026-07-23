# Engram Learnings — P5 Graduation and Cleanup Plan

**Status:** completed on 2026-07-22; relation candidates and annotations
graduated to default-on dogfooding on 2026-07-23 after explicit approval.

**Goal:** Graduate only measured behavior, prove compatibility parity, and
remove duplicated legacy ownership without a flag-day migration.

## Contract

- Correctness invariants remain always on.
- Relation candidate/annotation defaults change only after their fixed corpus
  gate passes; unfaithfully measurable setup UX remains human-approved.
- Legacy contradiction storage is archived only after import parity and one
  compatibility window; old public commands remain aliases.
- Installer duplication is removed only after registry parity tests.

## Implementation

1. Commit relation candidate and annotation eval fixtures before tuning.
2. Record candidate recall/noise, save latency, coordinator wait/rejections,
   setup idempotency, and migration parity.
3. Register report-only graduation metrics; enable only passing defaults.
4. Convert the contradiction sidecar to a read/import compatibility adapter,
   remove remaining dual ownership, and document rollback.
5. Produce a P5 scorecard plus release notes and run the full required gates.

## Gates

- Ruff → mypy → full non-slow → slow suite.
- Recall eval stays flat-or-better with flat-or-lower noise.
- Relation corpus gate, migration parity, setup idempotency, queue load test,
  runtime/installer smoke, and final diff review.
