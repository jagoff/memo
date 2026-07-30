# Task 3 — transactional SQLite views and idempotent commit

BASE: `c9127f39`

Status: implementation delivered in `daf3bf36` and all gates are green;
independent acceptance is still required. Task 2 also awaits independent
acceptance. Task 3 did not activate v2 in the production facade or mutate v1
bytes.

## Owned paths

Production:

- create `src/memo/operation_view_schema.py`
- create `src/memo/operation_views.py`
- modify `src/memo/operational.py`
- modify `src/memo/config.py`

Tests:

- create `tests/test_operation_views_v2.py`
- create `tests/test_operational_idempotency.py`
- extend `tests/test_operational_memory.py` only for compatibility assertions

Do not edit the frozen v1 ledger, v2 ledger authority, activation selector,
facade construction, federation, migration, outbox worker, sessions, plans,
progress, or SDD reports while production code is in flight.

## Compatibility boundary

- Existing `OperationalStore(state_dir, device_id=...)` remains the v1 facade
  until Task 7 activates a verified v2 generation.
- Add an explicit v2 construction path that receives an
  `OperationLedgerV2`, `OperationalViewStore`, and `EpochFence`; tests may use
  it directly.
- There is no production dual-write.
- Existing v1 snapshots, journal bytes, convenience return shapes, and
  `Memory.operational` behavior remain unchanged.

## RED-first contracts

1. Schema creation is idempotent and rejects unsupported schema versions.
2. `apply_events` uses one `BEGIN IMMEDIATE` transaction; reducer failure
   leaves no applied-event, cursor, domain row, or idempotency residue.
3. Exact replay is a no-op; the same `(origin, sequence)` with different
   event/hash fails closed.
4. Focus, handoff, attention, conflict, outcome, and session rows reduce to
   the exact current public state shape.
5. Catch-up applies every verified ledger event in canonical order.
6. Rebuild into a fresh DB produces the same canonical state and digest.
7. Unknown/unsupported events are quarantined and never partially reduced.
8. Commit holds `operational-transactions`, verifies the supplied
   `CommitContext`, catches up, resolves idempotency, appends once, and applies
   the event plus stored result atomically.
9. A crash after append but before view application replays the existing event
   on retry and never appends a duplicate.
10. Reusing `(project, idempotency_key)` with a different canonical request
    hash raises `IDEMPOTENCY_CONFLICT`.
11. Missing/stale/future epoch, wrong control OID, actor/context mismatch, and
    internal direct-call bypass fail before append.
12. The default v1 constructor and all legacy operational tests remain
    byte-compatible.

## SQLite authority

`operational.db` is derived and rebuildable. Required tables:

- `view_meta`
- `applied_events`
- `origin_cursors`
- `idempotency`
- `focus`
- `handoffs`
- `attention`
- `conflicts`
- `outcomes`
- `sessions`
- `session_local_artifacts`
- `durable_outbox`
- `quarantined_events`

Use foreign-key checks, WAL, busy timeout, deterministic canonical JSON,
explicit column lists, and `BEGIN IMMEDIATE`. Never treat SQLite as ledger
authority.

## Idempotency recovery

The request hash is SHA-256 over canonical `OperationalCommand` bytes. Because
the event embeds that hash as `content_hash`, retry may locate an already
appended event by `(project, idempotency_key)`:

- identical request hash: catch the view up, persist/reuse its canonical
  result, return `CommandResult(replayed=True)`;
- different request hash: fail with `IDEMPOTENCY_CONFLICT`;
- absent key: append exactly once, then apply/store the result.

An empty idempotency key is rejected by the v2 mutation entry point.

## Gates

- Focused Task 3 tests.
- Full Task 1 + Task 2 + Task 3 contracts.
- Frozen v1 exact/no diff.
- Ruff and mypy over touched paths.
- Full non-slow.
- Explicit-path technical commit and clean tracked worktree.
- Implementation report.
- Independent specification/durability review and PASS before acceptance.
