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

## Fix round — bind publish to the requested parent namespace

BASE reviewed: `23a64ccb9499474255a96c1215fd8b3534ea1452`

Technical commit:
`ce608c4253b844cfd4db0913a93e036911f5648f`

Status: implementation and proportional gates are green. Independent
specification/durability/security re-review is still required before
acceptance.

### RED evidence

The two required parent-swap regressions were added before the production
change and run against the current implementation:

```text
uv run --no-sync pytest \
  tests/test_operation_migration_v2.py::test_publish_rejects_parent_swap_before_exclusive_rename \
  tests/test_operation_migration_v2.py::test_publish_rejects_parent_swap_after_rename_before_parent_fsync \
  -q

2 failed in 3.34s
Failed: DID NOT RAISE OSError
Failed: DID NOT RAISE OSError
```

In both RED reproductions, `apply_v1_migration` returned success even though
the requested target path was absent. The prepared generation existed only
below the parent directory that had been renamed aside.

### Threat closure

- A secure descriptor for the trusted grandparent boundary and a no-follow
  descriptor for the exact requested parent name are retained before staging
  creation. The parent's initial `(st_dev, st_ino)` is bound to both
  descriptors.
- Staging allocation uses `mkdir(..., dir_fd=parent_fd)` with a private random
  name. Cleanup is also descriptor-relative and removes only the staging
  identity created by this attempt.
- Parent identity is re-resolved no-follow from the retained boundary across
  the build/verification phases and immediately before exclusive publish.
- Staging fsync, Darwin/Linux exclusive descriptor-relative rename, child
  identity validation, parent fsync, and durable child identity validation
  retain their original order.
- After parent fsync, parent and target are freshly re-opened/re-statted
  no-follow from the trusted boundary and must match the retained parent and
  prepared-generation identities. The same resolution is repeated
  immediately before returning success.
- A parent rename/replacement either before rename or at
  `after-rename-before-parent-fsync` now raises
  `prepared parent namespace identity changed`. A valid generation may remain
  in the displaced directory for crash/retry semantics, but it is never
  accepted as success at the requested pathname.
- The existing no-clobber path, replacement-generation rejection, stable-path
  crash retry, and task-owned staging cleanup remain green. The roster-selected
  prepared-stamp algorithm introduced in `23a64ccb` is preserved.
- Frozen v1 paths are unchanged and no activation marker or v2 selector was
  added.

### GREEN evidence and final gates

Required focused gate:

```text
uv run --no-sync pytest \
  tests/test_operation_migration_v2.py \
  tests/test_definitive_gate.py -q

19 passed in 22.36s
```

Proportional migration/ledger/view/frozen-v1/definitive matrix:

```text
uv run --no-sync pytest \
  tests/test_operation_migration_v2.py \
  tests/test_operation_ledger_v2.py \
  tests/test_operation_views_v2.py \
  tests/test_operation_ledger_v1_compat.py \
  tests/test_definitive_memory.py \
  tests/test_definitive_gate.py -q

123 passed in 31.06s
```

Static and repository gates:

```text
uv run --no-sync ruff check \
  src/memo/operation_migration.py \
  tests/test_operation_migration_v2.py
All checks passed!

uv run --no-sync mypy src/memo/operation_migration.py
Success: no issues found in 1 source file

git diff --exit-code 23a64ccb -- \
  src/memo/operation_ledger.py \
  src/memo/operation_ledger_v1.py \
  src/memo/contracts.py
exit 0, no output

git diff --check
exit 0, no output
```

Commit scope:

```text
git diff-tree --no-commit-id --name-only -r ce608c42
src/memo/operation_migration.py
tests/test_operation_migration_v2.py

## Fix round 2 — bind the complete boundary namespace chain

BASE lógico: `2f636240`

Technical commit:
`6e54600fbd4b197350db2830e20ad1906d4d6c5d`

Status: boundary-swap regressions and all required gates are green. Independent
security re-review remains required before acceptance.

### RED evidence

The new regressions were added before the production change and run against
the `ce608c42` implementation:

```text
uv run --no-sync pytest \
  tests/test_operation_migration_v2.py::test_publish_rejects_boundary_swap_before_exclusive_rename \
  tests/test_operation_migration_v2.py::test_publish_rejects_boundary_swap_after_rename_before_parent_fsync \
  -q

2 failed in 3.85s
Failed: DID NOT RAISE OSError
Failed: DID NOT RAISE OSError
```

Both RED cases renamed the grandparent/boundary aside, recreated the
requested boundary and parent names, and observed a success/parity result with
the prepared generation only under the displaced boundary.

### Threat closure

- The requested parent is now opened by walking every absolute component
  descriptor-relative from a retained no-follow filesystem-root FD. Each
  component is bound to its name and `(st_dev, st_ino)`; missing components are
  created with descriptor-relative `mkdir` and parent `fsync`.
- The retained parent FD, staging allocation/cleanup, target publication,
  child identity checks, and fsyncs remain descriptor-relative and no-follow.
  The accepted parent-swap protections from `ce608c42` are preserved.
- `_resolve_bound_parent` reopens the entire root-to-parent chain and requires
  every expected component identity before resolving the target. Boundary or
  ancestor replacement therefore fails closed even when the retained parent
  FD still points into a displaced directory.
- Final parent+target re-resolution is performed after parent fsync and again
  immediately before returning success. No activation marker or v2 selector is
  introduced; the roster-selected prepared-stamp algorithm from `23a64ccb`
  remains unchanged.
- Stable crash retry, no-clobber Darwin/Linux exclusive rename, staging
  replacement detection, parent-swap tests, and frozen-v1 bytes remain green.

### GREEN evidence and final gates

Focused Task 4 plus definitive gate:

```text
uv run --no-sync pytest tests/test_operation_migration_v2.py \
  tests/test_definitive_gate.py -q

21 passed in 25.78s
```

Proportional migration/ledger/view/frozen-v1/definitive matrix:

```text
uv run --no-sync pytest \
  tests/test_operation_migration_v2.py \
  tests/test_operation_ledger_v2.py \
  tests/test_operation_views_v2.py \
  tests/test_operation_ledger_v1_compat.py \
  tests/test_definitive_memory.py \
  tests/test_definitive_gate.py -q

125 passed in 34.31s
```

Boundary and immediate-parent regressions:

```text
4 passed in 5.98s
```

Static and repository gates:

```text
uv run --no-sync ruff check \
  src/memo/operation_migration.py \
  tests/test_operation_migration_v2.py
All checks passed!

uv run --no-sync mypy src/memo/operation_migration.py
Success: no issues found in 1 source file

git diff --exit-code 2f636240 -- \
  src/memo/operation_ledger.py \
  src/memo/operation_ledger_v1.py \
  src/memo/contracts.py
exit 0, no output

git diff --check
exit 0, no output
```

Commit scope:

```text
git diff-tree --no-commit-id --name-only -r 6e54600f
src/memo/operation_migration.py
tests/test_operation_migration_v2.py
```
```
