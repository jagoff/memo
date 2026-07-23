# Engram Learnings — P3 Lifecycle Convergence Plan

**Goal:** Separate truth validity, verification confidence, and review timing,
then expose explicit review operations without age-based invalidation.

## Contract

- `valid_at`/`invalid_at` alone describe whether a claim is true.
- `verification_state`/`verified_at` describe whether it was checked.
- `review_after` describes when another check would be useful and never
  invalidates a record.
- Automatic schedules: preference/config 90 days, decision 180 days,
  policy/architecture 365 days, otherwise none. Open conflicts are due now.
- `mark_reviewed` records evidence, marks verified, and schedules the next
  review without rewriting the content body.
- Verified records become stale only after their own `review_after`; records
  without a review policy do not decay from wall-clock age.

## Implementation

1. Make `review_after` a first-class `MemoryRecord` and store/reindex field.
2. Add review evidence as a signal table preserved by rebuild.
3. Add pure schedule policy and lifecycle operations:
   `list_due_reviews`, `mark_reviewed`, `invalidate`, and `supersede`.
4. Reconcile the verification transition pass with record-specific schedules.
5. Add CLI/MCP review/list/mark operations and maintain diagnostics; never
   auto-invalidate from maintenance.

## Gates

- Policy matrix and no-policy no-decay.
- Due review does not change validity.
- Evidence and next schedule survive reindex.
- Mark-reviewed idempotency and supersede/as-of compatibility.
