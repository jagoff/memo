# Task 5 — exactly-once durable promotion outbox

BASE: `20e16ba4`

Status: brief frozen; production edits and RED execution have not started.
Tasks 2, 3, and 4 remain technically green but independently unaccepted. v1
remains the production authority, v2 remains dormant, and Memflow remains live.

## Owned paths

Production:

- create `src/memo/durable_outbox.py`
- modify `src/memo/operational_event_types.py`
- modify `src/memo/operation_views.py`
- modify `src/memo/operation_view_schema.py` only for an idempotent ready-row
  index if required
- modify `src/memo/memory/write_ops.py`
- modify `src/memo/memory/outcome_feedback_ops.py`
- modify `src/memo/cli_operational.py`
- modify `src/memo/server_operational.py`

Tests:

- create `tests/test_durable_outbox.py`
- create `tests/test_write_ops_operation_identity.py`
- extend `tests/test_operational_event_types.py`
- extend `tests/test_operation_views_v2.py`
- extend `tests/test_definitive_memory.py`
- extend `tests/test_cli_mcp_surface_smoke.py`

Do not edit the frozen v1 ledger, migration format, activation selector,
federation, sessions, Memflow runtime, live state, or production
configuration.

## Authority and activation boundary

- The four v2 events below are the sole outbox authority. SQLite is a
  rebuildable projection and worker counters are never authoritative.
- This task does not activate v2 or create a second production authority.
  `DurableOutboxWorker` is explicitly constructed against a v2
  `OperationalStore`, `OperationalViewStore`, authenticated context factory,
  and immutable command authority.
- `Memory.promote_learning` requires an idempotency key now. When the v2
  durable-outbox capability is explicitly installed, it enqueues and
  synchronously reconciles the intent. Until Task 7 installs that capability,
  the existing v1 facade preserves synchronous behavior through
  `save_operation`; it does not emit pretend v1 outbox events. Task 7 owns
  activation and must prove the v2 capability is installed before selecting
  v2.
- The daemon only calls `run_once` to recover interrupted intents; a normal
  promotion call returns only after its requested intent is completed or
  permanently rejected.

## Stable identity and frozen request

- Normalize `idempotency_key` by stripping surrounding whitespace and reject
  empty values.
- `promotion_digest = sha256(idempotency_key.encode("utf-8")).hexdigest()`.
- `promotion_id = promotion_digest`.
- `operation_key = "promotion/" + promotion_digest`.
- Freeze the complete `Memory.save` keyword mapping by canonical JSON
  round-trip. Only string-keyed JSON values are admitted; booleans are not
  accepted where integers are required. The public intent exposes recursively
  immutable mappings/tuples and returns detached mutable kwargs only at the
  `Memory.save_operation` boundary.
- `request_hash = sha256(canonical_json_bytes(save_kwargs)).hexdigest()`.
- The promotion timestamp stored in `save_kwargs` is the latest qualifying
  source outcome timestamp. This keeps replay stable before v2 activation;
  it is never recomputed from the wall clock for the same evidence.
- `source_event_ids` are the canonically ordered IDs of verified outcome
  events that cite any source memory. They are persisted both in the requested
  intent and in the resulting memory provenance.
- `Memory.save_operation` owns `extra["_memo_operation"]` with exactly
  `operation_key` and `request_hash`; callers cannot override it.

## Closed event vocabulary and payloads

The preliminary `enqueued/completed` pair is replaced before activation by
the complete, fully qualified vocabulary:

- `memo.operational.durable.promotion.requested.v1`
- `memo.operational.durable.promotion.retry_scheduled.v1`
- `memo.operational.durable.promotion.completed.v1`
- `memo.operational.durable.promotion.rejected.v1`

Requested payload:

```text
promotion_id, idempotency_key, operation_key, request_hash,
save_kwargs, source_event_ids, created_at
```

Retry-scheduled payload:

```text
promotion_id, operation_key, request_hash, attempt_number,
failure_class, retry_at
```

Completed payload:

```text
promotion_id, operation_key, request_hash, memory_id
```

Rejected payload:

```text
promotion_id, operation_key, request_hash, failure_class, reason
```

Every hash is lowercase SHA-256, timestamps are timezone-aware ISO-8601, and
source event IDs are non-empty and duplicate-free. Terminal and retry events
must bind the exact operation key and request hash from the request.

Command idempotency keys are deterministic and disjoint:

- `durable-promotion/requested/<promotion_id>`
- `durable-promotion/retry/<promotion_id>/<attempt_number>`
- `durable-promotion/completed/<promotion_id>/<request_hash>`
- `durable-promotion/rejected/<promotion_id>/<request_hash>`

## Monotonic reducer

- requested → `pending`, attempts `0`
- pending/retry_scheduled → `retry_scheduled`, attempts exactly previous + 1
- pending/retry_scheduled → `completed`
- pending/retry_scheduled → `rejected`
- exact requested replay is a no-op; a changed request with the same promotion
  ID or operation key is an idempotency conflict and never overwrites the row
- a terminal row never transitions again
- retry timing is derived from frozen `created_at` plus
  `min(2 ** (attempt_number - 1), 3600)` seconds, not from failure wall time
- rebuild from the verified ledger produces the same row bytes and report

`FrozenPromotionIntent` contains `id`, `idempotency_key`, `operation_key`,
`request_hash`, immutable `save_kwargs`, ordered `source_event_ids`,
`created_at`, and `attempts`.

`OperationalViewStore.pending_outbox(limit=...)` returns only non-terminal,
due rows in deterministic `(retry_at-or-created_at, created_at, id)` order.
The optional worker clock is passed as an explicit `now` to the internal view
query; the public one-argument form uses UTC now.

`OperationalViewStore.outbox_report()` is an aggregate projection snapshot.
`DurableOutboxWorker.run_once()` returns a run-local `OutboxRunReport` so
`limit` cannot inflate `examined`; its `pending` field is the aggregate
remaining non-terminal count.

## Exactly-once write and recovery

Within the existing re-entrant memory authority lock:

1. Canonicalize and hash the complete save request again.
2. Reject a caller-supplied hash mismatch.
3. Scan exact Markdown frontmatter for `_memo_operation.operation_key`.
4. Return the existing record when its request hash matches.
5. Raise `IdentityConflictError(kind="durable_operation")` when the same key
   names a different request, or when multiple Markdown files claim the key.
6. If Markdown exists but SQLite is absent or `_memo_embed_pending`, rebuild
   the text row from allowlisted frontmatter/body without creating a second
   Markdown file. Embedding recovery remains the existing reindex contract.
7. Otherwise inject the owned operation metadata and call the normal save
   pipeline once.

Topic identity remains a convenience only. Exactly-once proof is the operation
key plus request hash in authoritative Markdown.

`IdentityConflictError`, invalid frozen requests, and native write-policy
refusals are permanent rejections. Storage/runtime failures schedule the next
deterministic retry and are re-raised after the retry event commits. A crash
after Markdown save but before completion is reconciled by the frontmatter
identity on the next run.

## RED-first contracts

1. The event registry accepts all four exact payloads and rejects malformed
   hashes, attempts, timestamps, lists, and missing fields.
2. The reducer is monotonic, detects changed requests, enforces attempt
   sequence, and rebuilds requested → retry/completed/rejected byte-for-byte.
3. Same operation key and hash returns the same memory ID.
4. Same operation key and different hash raises `IdentityConflictError`.
5. Multiple Markdown claims fail closed.
6. Markdown-without-vector and `_memo_embed_pending` recovery never creates a
   duplicate Markdown file.
7. Crash before save schedules one retry; crash after save then replay
   completes the original memory.
8. Permanent write-policy rejection commits rejected and does not retry.
9. A bounded batch examines at most `limit`; future retries are not attempted.
10. Frozen timestamps, request hash, source event ordering, and provenance are
    stable across replay.
11. Full SQLite rebuild between requested and completed still reconciles once.
12. CLI and MCP require and forward `idempotency_key`.
13. The v1 facade remains active and frozen v1 bytes remain identical.

## Gates

- Focused outbox, write identity, v2 event/view, outcome, idempotency, CLI/MCP,
  and definitive tests.
- Task 1–5 cumulative operational/definitive regression.
- Ruff and mypy over every touched path.
- Full non-slow suite.
- Frozen-v1 diff and activation-marker audit.
- Explicit-path technical commit and clean tracked worktree.
- Implementation report.
- Independent specification/durability review and PASS before acceptance.
