# Engram Learnings — P2 Relation Convergence Plan

**Goal:** Make `memory_relations` the only writable relation truth and complete
the save → candidate → judgment → recall loop without a service-side LLM call.

## Contract

- One deterministic row per unordered memory pair, with directional
  `source_id` → `target_id` semantics for `supersedes`.
- States are `pending`, `judged`, or `orphaned`; verbs are `supersedes`,
  `conflicts_with`, `compatible`, `scoped`, `related`, and `not_conflict`.
- Repeating a judgment is idempotent. A different second judgment raises a
  typed conflict without changing the audit row.
- New/revised eligible saves search at most three same-namespace/global
  candidates after the canonical write commits. Detection failure never
  changes the successful save outcome.
- A judged `supersedes` transition closes the target validity interval before
  the judgment commits. Deleted endpoints become orphaned.
- Legacy `contradictions.db` imports idempotently by deterministic migration
  key. It is read-only during the compatibility window; there is no dual write.
- Judged positive/conflict relations appear compactly in search, ask, and
  unified briefing. Pending rows appear only in review/diagnostic APIs.

## Implementation

1. Add schema-v6 relation provenance/idempotency columns and indexes without
   touching Markdown or rebuildable rows.
2. Add a narrow relation store mixin behind `VecStore`.
3. Add `memory/relation_ops.py` and compose it into `Memory`.
4. Wrap `Memory.save()` post-commit to generate bounded candidates for eligible
   durable records and return additive candidate metadata.
5. Upgrade `mem_judge`/`mem_compare` to delegate to `Memory`; add stable
   relation review/list MCP surfaces without duplicating the judgment tool.
6. Import the legacy contradiction sidecar and make existing scanners write
   through the canonical store only.
7. Annotate search/ask/briefing results and orphan rows on deletion.

## Gates

- Candidate cap, namespace matrix, no-LLM assertion, save-failure isolation.
- Judgment idempotency/conflict, supersede rollback, as-of behavior.
- Orphan handling, deterministic legacy import, save→judge→recall E2E.
- Existing contradiction/session-pattern compatibility suites.
