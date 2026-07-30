# Task 4 — deterministic v1 genesis migration and parity gate

BASE: `42f458c0`

Status: in progress. v1 remains the production authority, v2 remains dormant,
and this task may write only to an isolated prepared generation. Task 2 and
Task 3 still require independent acceptance.

## Owned paths

Production:

- create `src/memo/operation_migration.py`
- modify `src/memo/operation_ledger_v2.py` only to admit a deterministic,
  migration-authorized event ID
- modify `src/memo/definitive.py`

Tests:

- create `tests/test_operation_migration_v2.py`
- extend `tests/test_operation_ledger_v2.py` only for the migration event-ID
  boundary
- extend `tests/test_definitive_memory.py`

Do not edit the frozen v1 reader, current v1 facade, activation selector,
federation, Memflow runtime, live state, outbox, sessions, or production
configuration.

## Migration authority and multi-origin model

- `plan_v1_migration` verifies every v1 origin and snapshots a canonical
  manifest, source state, source proofs, seed plan, and source heads without
  writing v1 or v2.
- The enrolled local v1 origin is the sole seed-event writer. Every verified
  v1 origin receives its own `memo_v1` genesis anchor; remote anchors remain at
  their v1 heads. Seed events written on the local origin may prove rows whose
  latest contributing event came from another origin.
- The migration attestor key is exclusive to the `migration_attestor` role.
  The normal origin key signs seed events. Every seed event embeds the same
  signed `MigrationOrigin`, bound to the source manifest and the authenticated
  Merkle root of the selected v1 source proofs.
- A non-empty source must contain the enrolled local origin. This avoids
  forging a remote origin through a local epoch fence. An empty source produces
  an empty prepared view and no fabricated origin.

## Deterministic seed contract

Reduce verified v1 events in memory; never read `operational-state.json` as
source. Emit one upsert seed for each final row, sorted by
`(domain, canonical id)`:

- focus → `FOCUS_SET`, keyed by project
- handoffs → `HANDOFF_CREATED`, keyed by id
- attention → `ATTENTION_ADDED`, keyed by id
- conflicts, including semantic-anomaly rows → `CONFLICT_OPENED`, keyed by id
- outcomes → `OUTCOME_RECORDED`, keyed by task id

Each seed payload is the exact final v1 row. Its source proof is the latest v1
event that contributed to that row. Event IDs and idempotency keys are exactly
`memo-v1/<manifest>/<domain>/<percent-encoded-id>`. Ordinary append keeps
random IDs; only an authenticated migration context may request the
deterministic ID, and it must match the command idempotency key and manifest
prefix.

All migration timestamps, attempt IDs, ordering, proofs, signatures, state
roots, and semantic generation digests are derived from frozen source and
authority inputs. No wall-clock value may enter a prepared generation.

## Fixed apply protocol

1. Recompute and compare the verified v1 manifest before creating the target.
2. Refuse a pre-existing non-identical target; accept an already prepared,
   byte/semantic-equivalent target as an idempotent replay.
3. Create a sibling staging directory.
4. Seal an empty pre-seed checkpoint and one v1 genesis anchor per origin.
5. Append deterministic, source-proven seed events under the signed migration
   origin.
6. Verify the v2 ledger and rebuild `operational.db` only from verified v2
   events.
7. Compare canonical v1/v2 domain state excluding `last_event_hash`,
   `journal_heads`, and v2-only sessions.
8. Sign and atomically write `migration-v1.json` only after exact parity.
9. Atomically install the prepared generation. Do not write an activation
   stamp and do not select v2.

Failures before install remove only the task-owned staging directory. Corrupt
v1 and source-manifest drift fail before any target write.

## RED-first contracts

1. Identical source and authority inputs produce identical seed IDs, event
   hashes, source proofs, parity digest, and semantic generation digest.
2. Reapplying an installed prepared generation returns zero inserted events.
3. A corrupt v1 chain leaves the requested target absent.
4. Any source change after planning is a hard manifest failure before target
   creation.
5. Every verified source origin has a `memo_v1` anchor at its exact v1 head.
6. Seed IDs follow the exact deterministic namespace and survive rebuild.
7. Every seed carries an authenticated original v1 `SourceProof` and the
   signed migration origin.
8. Focus, handoff, attention, conflict/anomaly, and outcome state is exactly
   equal after excluding authority metadata.
9. A forced mismatch writes neither `migration-v1.json` nor an activation
   marker.
10. Definitive readiness reports prepared-migration evidence separately and
    does not treat it as production activation.
11. Frozen v1 bytes remain identical.

## Gates

- Focused Task 4 tests, Task 2 ledger tests, Task 3 view tests, frozen-v1
  compatibility, and definitive tests.
- Ruff and mypy over every touched path.
- Full non-slow suite.
- Explicit-path technical commit and clean tracked worktree.
- Implementation report.
- Independent specification/durability review and PASS before acceptance.
