# Task 3 implementation report — transactional operational views

BASE: `c9127f39`
Technical commit: `daf3bf36`

Status: implementation delivered and green; independent specification,
durability, and quality review is still required before acceptance.

## Delivered

- Added a versioned STRICT SQLite schema for rebuildable operational-v2 views.
- Added one-transaction reducers for focus, handoffs, attention, conflicts,
  outcomes, sessions, and the currently registered durable-promotion events.
- Added durable applied-event, per-origin cursor, idempotency-result, and
  quarantine records.
- Added deterministic catch-up and rebuild with a canonical state digest.
- Added normalized UTC ordering for the public global last-event hash.
- Added per-origin fail-closed quarantine: an unsupported event prevents later
  events from that origin from advancing until a reducer exists; successful
  replay clears stale quarantine rows.
- Added a dormant explicit `OperationalStore.for_v2` construction path and the
  nine-step guarded commit protocol without activating the production facade.
- Added append-before-view crash recovery, exact replay, request-hash conflict
  detection, same-origin interleaving catch-up, and rejection of local event
  types that have no active reducer.
- Kept the default v1 constructor authoritative and proved it does not create
  the dormant v2 generation.

## RED evidence

Initial collection:

```text
2 errors
ModuleNotFoundError: No module named 'memo.operation_views'
```

First crash-window implementation:

```text
1 failed, 10 passed
```

The failing test showed that its injected view failure also intercepted the
required pre-append empty catch-up. The fault injection was narrowed to fail
only when a persisted event was present, preserving the specified protocol.

## Final gates

```text
Focused Task 3:
22 passed

Task 1 + Task 2 + Task 3 + legacy operational contracts:
256 passed

Frozen v1:
3 passed
operation_ledger.py and contracts.py diff from BASE: empty

Ruff:
All checks passed

Mypy:
Success: no issues found in 3 source files

Full non-slow:
6032 passed, 18 skipped, 19 pre-existing fork/xdist warnings
```

The real `OperationLedgerV2` + `EpochFence` + `OperationalViewStore`
integration is exercised in the committed idempotency tests.

## Compatibility and deferred activation

- No v1 journal or snapshot encoding changed.
- No production v1/v2 dual-write was introduced.
- `Memory.operational`, CLI, MCP, and daemon surfaces still use the v1 facade.
- External authenticated epoch/control propagation remains an activation task:
  it must be wired and bypass-tested when Task 7 selects the verified v2
  generation. This task covers direct internal bypass, real stale/future epoch,
  wrong control OID, missing context, and actor/context mismatch before append.

## Review package

Review range:

```text
c9127f39..daf3bf36
```

Required independent checks:

1. SQLite transaction and crash durability.
2. Request-hash/idempotency recovery after append-before-view loss.
3. Per-origin quarantine and future reducer recovery.
4. Real ledger/fence lock composition and bypass resistance.
5. Exact v1 compatibility and dormant activation boundary.
