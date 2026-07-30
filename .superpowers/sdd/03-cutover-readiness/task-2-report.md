# Task 2 implementation report — Memflow mutation fencing

Memflow BASE: `5426e8e5fce83d8ccf98fa7ecba3bcc634531ae2`

Memflow workspace:
`/Users/fer/repos/memflow/.worktrees/memflow-absorption-fence`

Memflow branch: `feat/memflow-absorption-fence`

Technical commits:

- `22299647aa00f2d419d416d0b3d8bccdab8a4946`
- `2c6438636c7ef6ed21c5d7e9f59929812439fcc1`

Status: implementation delivered and green; independent specification,
durability, security, and quality review is still required before acceptance.

## Delivered

- Added a Memo-independent Ed25519 verifier for the exact
  `memo.cutover.fence.v1` signature domain and canonical JSON wire contract.
- Added immutable `FenceMarker`, `VerifiedControlRecord`, `DrainSnapshot`,
  `CutoverMode`, and typed fail-closed errors.
- Bound every installed marker to the expected Memflow commit, runtime digest,
  device, signer, attempt, control OID/sequence/predecessor, issuance/expiry,
  and compatible verified control state before publication.
- Added an atomic/fsynced `fence.json`, durable `fence-seen` sentinel, process
  lock, file lock, no-follow authority reads, and symlink rejection.
- Preserved exact pre-fence ACTIVE behavior. After the first fence,
  missing/corrupt/expired authority fails closed; an expired ACTIVE abort
  marker never restores writes.
- Made mode validation and in-flight increment one atomic admission operation.
  Nested work already admitted may finish in QUIESCING; all new mutations and
  public startup are rejected retryably.
- Made RETIRED non-retryable, monotonic, and permanent. ACTIVE can return only
  from a newer signature-verified ABORTED control and never after RETIRED.
- Kept read-only operations available in every mode. The private
  `abort_healthcheck` authorization requires a matching ABORTING control
  re-fetched and supplied again after process restart.
- Fenced channel/events/transcript capture, delivery/ACK, presence, sessions,
  usage telemetry, cursors, Git sync/push, capture-shim installation, and
  central MCP mutation dispatch before handlers execute.
- Added reentrant per-domain accounting for requests, event append, delivery,
  ACK, cursor, sync, Git push, autonomous loops, and writable handles.
- Added the required `cryptography` runtime dependency to packaging, both
  installer fallback paths, installer regression coverage, and technical
  documentation.
- Separately restored metrics to the tested opt-in default, accepted
  `Path | str` inputs, and removed the only nine pre-existing mypy errors in
  the Memflow source tree.

## RED evidence

The focused fence test was collected before the implementation existed:

```text
ModuleNotFoundError: No module named 'memflow.cutover_fence'
```

## GREEN evidence

```text
Fence + capture focused:
34 passed

Mutation/installer/metrics parity:
211 passed

Ruff over all Memflow source and touched tests:
All checks passed

Mypy over all Memflow source:
Success: no issues found in 145 source files

Full Memflow suite at technical HEAD:
1888 passed, 1 external Starlette/httpx deprecation warning

bash -n install.sh:
passed

git diff --check:
passed
```

## Production non-mutation proof

- No production marker or sentinel was installed.
- No live state, capture shim, hook, config, LaunchAgent, listener, or
  credential was changed.
- The live job remained `running`, PID `2046`, with program
  `/Users/fer/repos/memflow/.venv/bin/memflow-mcp`.
- The same process remained bound only to `127.0.0.1:18766`.
- The implementation worktree was clean after both commits.

## Deferred to Task 3 / Plan 05

- Task 3 owns loop-iteration tracking, observable drain waiting/fsync,
  temporary cutover CLI, and startup refusal before listener/thread creation.
- Plan 05 owns controller orchestration and the signed
  `RollbackHealthyReceipt` returned by the private read-only abort healthcheck.
- No cutover activation, rollback, retirement, or uninstall action is
  authorized by this task.

## Review package

Normative Memflow range:

```text
5426e8e5fce83d8ccf98fa7ecba3bcc634531ae2..2c6438636c7ef6ed21c5d7e9f59929812439fcc1
```

Required independent checks:

1. Signature protocol, canonical bytes, key/device/runtime/commit binding, and
   control-record binding.
2. Locking, fsync, no-follow publication, sentinel durability, expiry, and
   restart behavior.
3. Complete mutation-boundary coverage and MCP dynamic mutation predicates.
4. In-flight accounting, nested/exception cleanup, QUIESCING race safety, and
   permanent RETIRED behavior.
5. ACTIVE parity, packaging/install behavior, and absence of production
   mutation.
