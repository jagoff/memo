# Task 6 implementation report — canonical operational sessions

BASE: `6b68a260`
Brief commit: `3eb454f4`
Technical commit: `ecf2b951`

Status: implementation delivered and green; independent specification,
durability, and quality review is still required before acceptance.

## Delivered

- Replaced the preliminary session status event with the closed three-event
  authority: checkpointed, recoverable, and terminated.
- Added exact portable payload validation and a monotonic reducer that rejects
  absent lifecycle events, identity/project/workspace changes, recoverable
  regression, duplicate transitions under new events, and every post-terminal
  mutation.
- Added `OperationalSessionService` with authenticated commands, shared
  transaction locking, view catch-up before reads and preconditions, exact
  idempotent replay, durable timestamp reuse when the caller omits a timestamp,
  and deterministic recoverable-session lookup.
- Added canonical session reads and local-artifact storage to the operational
  view. Ledger rebuilds reproduce portable state while preserving the local
  artifact table, which is excluded from state hashes and federation.
- Added a pure deterministic legacy merger for JSON checkpoints and legacy
  SQLite rows. It rejects incompatible identity boundaries and detaches every
  unknown/local field from the portable checkpoint.
- Added an explicit process-local runtime binding in `memo.session`. Without
  installation it preserves v1 JSON behavior; with installation it commits the
  canonical event first, stores local artifacts separately, and emits the JSON
  sidecar only as an atomic derived cache.
- Renamed the lifecycle MCP registrations to `memo_session_start` and
  `memo_session_end`; no lifecycle aliases remain. Existing session list/get
  and context rendering use the canonical capability when it is explicitly
  installed.

## RED evidence

The focused session matrix was collected before the service and final event
vocabulary existed:

```text
3 collection errors
ModuleNotFoundError: No module named 'memo.operational_sessions'
ImportError: cannot import name 'SESSION_RECOVERABLE'
```

## GREEN evidence

```text
Focused Task 6 matrix:
586 passed

Task 1–6 cumulative operational/definitive matrix:
817 passed in 109.69s

Ruff over all 13 touched source/test paths:
All checks passed

Ruff format check over all 13 touched source/test paths:
13 files already formatted

mypy over all 6 touched source paths:
Success: no issues found

Full non-slow:
6106 passed, 18 skipped, 19 known fork/xdist warnings in 59.97s

git diff --check:
passed
```

Frozen v1 sources remained byte-identical to BASE:

```text
src/memo/operation_ledger.py
55d29af262c1e3547e058505da1f09693dc5eb950f462672ada827b2cb911d9c

src/memo/operational.py
ab607b4ade663c176b70ade04b9d957ea0170e12710c2b070ca6b701461d3702
```

## Compatibility and deferred activation

- No activation marker exists and production source has no runtime installer
  call; Task 7 still owns explicit v2 activation and teardown.
- The v1 JSON session path remains the default while no runtime is installed.
- Memflow launchd remained running and port `127.0.0.1:18766` remained
  listening after the implementation and verification runs.
- No Memflow runtime, hooks, state, pending ACKs, or configuration were
  modified.

## Review package

Normative range:

```text
6b68a260..ecf2b951
```

Required independent checks:

1. Lifecycle monotonicity, immutable ownership boundaries, and exact payloads.
2. Append-before-view recovery, shared-lock preconditions, exact replay, and
   omitted-timestamp stability.
3. Local/portable separation, rebuild preservation, and state/federation
   exclusion.
4. Legacy merge determinism and derived-cache crash boundaries.
5. Dormant activation boundary, v1 compatibility, public MCP names, and
   absence of Memflow mutation.
