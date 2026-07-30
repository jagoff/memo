# Task 7 — activate the verified operational-v2 facade

BASE: `e4de56db`

Status: brief frozen; production edits are gated by independent review of
Tasks 2–6. The v1 backend remains authoritative, v2 remains dormant, and
Memflow remains live.

## Preconditions

- Every Task 2–6 `BLOCKER`, `HIGH`, and `MEDIUM` finding has a regression
  test, a focused green gate, and independent re-review `PASS`.
- The productive macOS signing provider can generate, sign, reopen, and
  destroy a non-exportable operational key. The current fail-closed
  placeholder is not an acceptable fresh-install implementation.
- No production activation marker exists before all gates pass.
- Frozen v1 bytes remain identical through every test and rehearsal.

Known blockers at brief freeze:

- `P01-T02 HIGH`: v1 manifest/anchor construction can reread bytes after
  verification (TOCTOU).
- `P01-T03 HIGH`: an older event imported from another origin can make an
  incremental projection diverge from a full rebuild.
- `P01-T07 BLOCKER`: `MacOSKeychainProvider` cannot perform any productive
  key operation.

## Owned paths

Production:

- modify `src/memo/operation_ledger.py`
- modify `src/memo/memory/facade.py`
- modify `src/memo/federation.py`
- modify `src/memo/server_operational.py`
- modify `src/memo/server_annotations.py`
- modify `src/memo/definitive.py`

Tests:

- extend `tests/test_operational_memory.py`
- extend `tests/test_definitive_memory.py`
- extend `tests/test_cli_mcp_surface_smoke.py`

Prerequisite hardening may modify Task 1–6 paths only in separately reviewed
commits. Do not mix those fixes into the activation commit.

## Closed backend selector

`open_operational_backend(cfg)` classifies the complete install before
constructing a mutable component:

- `fresh`: no v1 authority and no v2 artifact exists; create one empty v2
  generation.
- `legacy_only`: verified v1 authority exists and no v2 artifact exists; open
  the explicit legacy adapter.
- `activated_v2`: v1 may exist, but the complete signed activation record and
  every bound v2 authority artifact verify; open v2.
- every other partial, missing, mixed, corrupt, or rollback state: raise a
  typed non-retryable `OperationalError` without writing v1 or recreating v2.

The selector returns a closed `OperationalBackend` protocol, never a raw
v1/v2 ledger union. `LegacyOperationalBackend` exposes `backend_version=1`;
`V2OperationalBackend` exposes `backend_version=2`. Both preserve current
successful public response shapes.

## Fresh-install transaction

The v2 root is staged outside its final path and published atomically only
after all of the following verify:

1. create the productive non-exportable local signing key;
2. persist and pin the signed one-peer roster v1;
3. persist and verify the canonical empty anchor;
4. persist and pin the signed epoch-0 marker bound to roster and anchor;
5. create/rebuild the empty operational view;
6. write and verify a signed fresh-generation activation record;
7. fsync every authority file and directory;
8. atomically publish the complete generation.

Any preexisting final or staging artifact is classified as partial and fails
closed. A crash may leave cleanup candidates but never a selectable authority.

## Prepared migration activation

The activation record is signed under a dedicated registered signature
domain. Its canonical digest binds:

- the exact prepared-migration stamp and source manifest;
- target generation and parity roots;
- every genesis anchor and current position;
- verification roster version/hash;
- epoch marker authorization and control OID;
- operational event-registry hash;
- reducer and SQLite schema versions;
- canonical session/parity root;
- Memo runtime version.

Activation first re-verifies the prepared stamp, source manifest/parity,
ledger, anchors, roster, epoch, views, registry, and runtime. The marker is the
last write. A marker with any missing or changed dependency fails closed.

## Runtime composition

- `Memory` constructs exactly one selected backend.
- Only v2 constructs `OperationLedgerV2`, `OperationalViewStore`,
  `EpochFence`, native sessions, and durable-outbox runtime.
- Federation validates complete signed bundles before import and catches views
  up after an all-or-nothing import.
- MCP v2 mutations require a non-empty idempotency key and authenticated
  epoch/control context.
- Only typed operational exceptions map to `memo.error.v1`; unchanged success
  payloads remain byte-shape compatible.
- Definitive verification checks ledger, anchors, roster, epoch, view rebuild
  equality, activation binding, and absence of dual authority.

## RED-first contracts

1. A fresh state directory selects v2, creates an empty state, roster v1, and
   epoch 0.
2. A verified legacy-only directory selects backend v1 without creating any
   v2 path.
3. A fully prepared but unactivated migration fails closed.
4. A valid activation selects v2 and reproduces exact v1 state.
5. A corrupt/missing/extra activation dependency fails closed and leaves v1
   bytes unchanged.
6. A crash after each fresh-install step never yields a selectable partial
   generation.
7. Exact open is idempotent and does not rotate keys, rewrite authority, or
   append an event.
8. Federation rejects one invalid bundle before importing any bundle.
9. MCP rejects empty idempotency and stale/wrong authority with typed errors.
10. Definitive verification detects view drift, anchor drift, roster rollback,
    epoch rollback, and activation-binding drift.
11. Existing successful operational response shapes are unchanged.
12. `LegacyOperationLedger` is inaccessible from active v2 runtime paths.

## Gates

- Focused facade, activation, federation, MCP, definitive, and v1-compat
  tests.
- Complete Tasks 1–7 ledger matrix.
- Ruff and mypy over every touched path.
- Full non-slow suite.
- Frozen-v1 byte diff and live Memflow no-mutation audit.
- Explicit-path technical commit.
- Independent specification and continuous security review `PASS`.
- No live activation or Memflow cutover in Plan 01.
