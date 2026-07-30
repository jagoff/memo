# Task 4 implementation report — deterministic v1 genesis migration

BASE: `42f458c0`
Brief commit: `5d346fe7`
Technical commit: `78764d74`

Status: implementation delivered and green; independent specification,
durability, and quality review is still required before acceptance.

## Delivered

- Added a deterministic, side-effect-free v1 migration planner that verifies
  every legacy origin, replays the frozen v1 reducer in memory, and treats
  `operational-state.json` only as an optional parity oracle.
- Added one authenticated `memo_v1` genesis anchor per verified source origin.
  The enrolled local origin is the sole v2 seed writer; remote-origin rows
  retain authenticated source proofs bound to anchors for the same manifest.
- Added one deterministic final-state seed per focus, handoff, attention,
  conflict/anomaly, and outcome row.
- Restricted deterministic event IDs to an authenticated memo-v1 migration
  context and the sealed
  `memo-v1/<manifest>/<domain>/<percent-encoded-id>` namespace.
- Added exact plan re-derivation before target creation and exact generation
  matching on replay, including authority, identity, workspace, payload,
  provenance, timestamp, and migration-origin fields.
- Added isolated sibling staging, final source fencing, exact v1/v2 state
  parity, signed `migration-v1.json`, and atomic prepared-generation install.
- Added structural prepared-migration evidence to definitive diagnostics
  without adding it to readiness or activation checks.
- Kept `operational-v2-activated.json` absent and the production v1 facade
  authoritative.

## RED evidence

Initial collection before the migration module existed:

```text
ModuleNotFoundError: No module named 'memo.operation_migration'
```

The final hardening regression also demonstrated that a prepared target must
reject the same source manifest under a different migration plan. The first
strict verifier run failed on timestamp representation; the verifier was
corrected to compare the authority time in the ledger's canonical
millisecond-UTC representation without relaxing any field checks.

## Final gates

```text
Focused Task 4:
10 passed

Required Task 4 matrix:
100 passed

Task 1–4 operational/definitive cumulative:
276 passed

Frozen v1:
operation_ledger.py, operation_ledger_v1.py, and contracts.py diff from BASE:
empty

Ruff:
All checks passed

Mypy:
Success: no issues found in 4 source files

Full non-slow:
6043 passed, 18 skipped, 7 deselected, 19 known fork warnings
```

## Compatibility and deferred activation

- No frozen v1 source file, journal encoding, or snapshot encoding changed.
- No production dual-write or v2 selection was introduced.
- A prepared stamp is informational and cannot activate the v2 facade.
- Memflow runtime, launch agents, hooks, state, and pending ACKs were not
  modified.
- Production activation remains owned by Task 7 and later readiness,
  active-state migration, atomic cutover, observation, and retirement gates.

## Review package

Normative range:

```text
42f458c0..78764d74
```

Technical-only range after the brief:

```text
5d346fe7..78764d74
```

Required independent checks:

1. Manifest re-derivation, source-fence coverage, and staging cleanup.
2. Multi-origin anchor/source-proof binding and migration-attestor isolation.
3. Exact deterministic generation replay and plan mismatch rejection.
4. State parity, prepared-stamp verification, and no activation path.
5. Frozen-v1 compatibility and absence of Memflow mutation.
