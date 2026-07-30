# Task 1 implementation report — safe snapshots and signed readiness manifests

Memo BASE: `90a7144e04366e36201e7d366d928151b5c80d17`

Task brief: `74ad2c5d`

Technical commits:

- `f6ca3fff3afb4d15ad44fa47bb0a76dee56d8ff4`
- `3c31a033`
- `26c90608a0401d82052a55ead8c12eb46e72f40c`
- `7858d540a06f0f705fa2db623c7a397b1dadfb6e`

Status: implementation delivered and green; independent specification,
durability, security, and quality review is still required before acceptance.

## Delivered

- Added an exact `memo/cutover/<attempt-id>` filesystem authority boundary.
  Broad targets, repositories, unresolved shell syntax, escapes, symlink
  components, and missing/mismatched canonical sentinels fail closed.
- Added explicit-file snapshots using descriptor-relative `O_NOFOLLOW` reads
  and exclusive, fsynced publication. Snapshot contents and canonical receipts
  are read-only; receipts preserve source size, mode, mtime, and SHA-256.
- Added immutable schemas for exact audit exclusions, per-Mac usage proofs,
  operation routes/maps, SLO baselines, capability rows/manifests, consumer
  inventories, Synapse retirement manifests, and cutover control records.
- Aligned the shared control schemas exactly with the normative cutover states
  and modes, including immutable fence, drain, final-fence, and
  verified-control-record types.
- Extended Memo's permanent operational signing contract with
  domain-separated cutover records and device/key/roster claim binding.
- Added a capability-manifest builder that joins a pinned Memflow source
  snapshot to signed evidence from exactly two Macs over the exact inclusive
  90-day window. It signs only a blocker-free manifest.
- The manifest fails closed on unknown or ambiguous traffic, missing/stale
  machine evidence, coverage gaps, evidence mismatches, mixed/duplicate
  dispositions, unsafe route predicates, incomplete deletion proof, and
  insufficient/lossy/duplicating SLO evidence.
- Canonical operation-map and SLO-baseline bytes are hashed into the signed
  manifest. Capability-level rows aggregate their exact source-operation
  mappings without discarding telemetry counts.
- Added read-only source/process/launchd consumer inventory without following
  symlinks. Clean inventories can be signed; inventories with scan blockers
  remain deliberately unsigned.
- Added a signed Synapse retirement manifest that enumerates Memflow-specific
  files, identifiers, tests, and goldens at a pinned source commit while
  binding a full-tree reference-scan digest.
- Added fresh-OID signed control-record verification that returns a typed
  `VerifiedControlRecord`, with positive monotonic sequence and
  predecessor-shape checks.
- Added repository-local CLI commands for `snapshot`, `manifest`, and
  `inventory`. They default to dry-run; snapshot apply is confined to a
  preexisting validated attempt root whose canonical sentinel is bound to the
  exact supplied manifest SHA-256. Missing or mismatched authority fails before
  any write.
- Added sanitized, non-production fixtures with an executable fixture digest.

## RED evidence

Before implementation, the exact five focused modules failed during
collection:

```text
ModuleNotFoundError: No module named 'tools'
5 errors during collection
```

## GREEN evidence

```text
Focused Task 1 tooling:
32 passed

Task 1 + operational signing/roster/atomic regression:
60 passed

Ruff over touched source and tests:
All checks passed

Mypy over all Task 1 production modules and operational signing:
Success: no issues found in 9 source files

Full non-slow suite at technical HEAD:
6138 passed, 18 skipped, 19 known fork/xdist warnings

git diff --check:
passed
```

Frozen Plan 01 v1 sources remained byte-identical:

```text
src/memo/operation_ledger.py
55d29af262c1e3547e058505da1f09693dc5eb950f462672ada827b2cb911d9c

src/memo/operational.py
ab607b4ade663c176b70ade04b9d957ea0170e12710c2b070ca6b701461d3702
```

The operational-v2 activation marker remains absent.

## Compatibility and deferred operation

- No production snapshot, capability manifest, inventory, exclusion record,
  usage proof, or cutover control record was created.
- All apply-path testing used pytest temporary roots and in-memory signing
  keys. The committed fixtures are synthetic and contain no local host,
  process, memory, credential, or production identity data.
- No LaunchAgent, process, hook, config, state root, port, service, or runtime
  was loaded, unloaded, stopped, replaced, or restarted.
- Memflow remains live and untouched. This task creates the authority tooling;
  it does not claim production capability parity or authorize cutover.
- A real frozen `CapabilityManifest` still requires fresh signed snapshots and
  complete 90-day evidence from both configured Macs.
- Plan 01 acceptance, independent Task 1 review, later parity gates, cutover
  observation, and explicit retirement authorization remain prerequisites to
  uninstalling Memflow.

## Review package

Normative implementation range:

```text
74ad2c5d..7858d540
```

Required independent checks:

1. Attempt-root authority, sentinel semantics, descriptor use, fsync
   boundaries, read-only publication, and symlink/escape rejection.
2. Signature domain separation and device/key/roster binding for exclusions,
   per-Mac usage, manifests, inventories, retirement manifests, and control
   records.
3. Exact inclusive 90-day evidence logic, freshness, exclusions, raw-event
   hashes, unknown/ambiguous traffic, and duplicate-count preservation.
4. One disposition per pinned source operation, route-language closure,
   fixture binding, deletion proof, capability aggregation, and canonical
   operation-map digest.
5. SLO coverage, minimum samples, distribution validity, recovery/error
   bounds, and zero loss/duplicates.
6. Consumer scan completeness, symlink handling, Synapse surface enumeration,
   reference-scan digest, and signature tamper resistance.
7. CLI dry-run purity and proof that no live Memflow or Synapse state changed.
