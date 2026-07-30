# Task 5 report — Synapse activation-epoch retirement fence

## Delivered

- Extended the signed CAS control record with a monotonic Synapse retirement
  state, capability/consumer-plan authority digests, peer-vote evidence,
  active-state receipt and retirement-manifest digests, the one-time retirement
  epoch, and final independence receipt digest.
- Enforced only `PREPARING -> READY -> QUIESCED -> STAGED -> COMMITTED ->
  VERIFIED`, with `ABORTED` as the sole pre-commit failure branch. Coordinated
  and Synapse states must agree; skipped/stale transitions, unsigned transition
  inputs, changed authority, missing peer votes, incomplete staging evidence,
  and a second epoch fail closed.
- Added the offline retired-runtime request-fence contract. Read-only status
  remains available; stale epochs fail first, while startup, write, and
  fallback admissions at the committed epoch return
  `synapse.cutover.retired`. Production wiring is not delivered by this Memo
  repository; see the correction-round-2 boundary map below.
- Added final independence verification bound to the committed control,
  retirement manifest, and signed consumer inventory. Both post-stop and
  post-reboot scans plus complete process, port, LaunchAgent, MCP/gateway,
  shell/config, and state-root coverage are mandatory. Any active row or
  remaining reference blocks the receipt.
- Added inspection-only `synapse-preflight` and `synapse-verify` commands.
  Both require canonical JSON and a pinned verification roster, verify the
  relevant Ed25519 signatures, validate internal digests, and reject
  `--apply`. They do not inspect, start, stop, load, unload, or rewrite a live
  service.
- Added the Task 5 regression matrix for offline peers, digest mutation,
  skipped states, abort behavior, stale/second epochs, retired startup,
  resurrected processes and loaded LaunchAgents, missing reboot evidence,
  successful receipt binding, CLI signature tamper, and inspection-only
  behavior.

## Verification

```text
uv run --no-sync pytest tests/tools/test_absorption_control_record.py tests/tools/test_absorption_safety.py tests/tools/test_synapse_cutover.py -v
22 passed

uv run --no-sync ruff check tools/memflow_absorption tests/tools
All checks passed!

uv run --no-sync mypy tools/memflow_absorption
Success: no issues found in 14 source files

git diff --check
passed
```

## Scope / concerns

- No activation, listener, worker, LaunchAgent, runtime configuration, or real
  state mutation was executed. The new CLI surfaces are deliberately
  inspection-only.
- The Python interfaces named in the brief retain their exact three-argument
  contracts and consume typed artifacts after authority admission. The CLI
  boundary additionally verifies the serialized artifacts cryptographically
  against an explicitly supplied pinned roster.
- The broader `tests/tools` suite currently reports 115 passed and 3 failures
  in `tests/tools/test_synapse_data.py`. All three are caused by the concurrent
  epoch-fence change at HEAD requiring authenticated epoch context for
  operational writes; Task 5 does not own that Synapse-data write path or its
  fixtures.
- Concurrent transform-registry/source-receipt work was preserved and excluded
  from this task's files and commit.

## Correction round 1 — end-to-end cutover authority

The first independent review found that the original helpers represented
evidence but did not prove the authority chain. This correction closes all
eleven findings:

- The real consumer replacement builder now requires complete signed surface
  observations and populates `covered_surfaces` itself. Its plan binds the
  exact roster-verified inventory and capability-manifest signed bytes; a
  caller-created self-hash is insufficient.
- Capability and retirement manifests are cryptographically verified against
  the supplied roster before their digests enter a control transition.
- Peer votes are separate roster-signed envelopes from the exact two manifest
  devices, bound to attempt ID, freshly fetched control OID, combined authority
  digest, and `QUIESCED`.
- `prepare_synapse_retirement`, `advance_synapse_retirement`, and
  `commit_synapse_activation` now consume a freshly fetched
  `VerifiedControlRecord` plus a CAS adapter, roster, signer, and next OID.
  Every transition signs `sequence + 1` with the exact predecessor OID and
  commits via compare-and-swap. A stale reader or concurrent writer loses
  before state changes.
- `sign_control_record` validates the exact predecessor, next sequence, legal
  state edge, unchanged attempt/roster authority, immutable post-preflight
  digests, and a distinct next OID. The activation epoch is committed only by
  the atomic `STAGED -> COMMITTED` path and is roster-verified after CAS.
- Final independence requires two distinct signed scan receipts: `post_stop`
  and `post_reboot`. Each binds boot ID, timezone-aware capture time, complete
  six-surface observations, and their source digest. Reuse of one boot,
  missing/duplicated observations, active references, digest mutation, or
  signature mutation fails closed.
- The final inventory cryptographically binds both scan receipt digests. The
  resulting independence receipt is itself signed, strictly parseable, and
  verified against the committed control, final inventory, retirement
  manifest, and both scans.
- `synapse-preflight` now requires the signed source inventory used to derive
  the plan. `synapse-verify` now requires both scan receipts and the signed
  independence receipt. Both CLI paths verify the complete serialized
  authority chain against the pinned roster.
- Because this repository has no Synapse listener or worker implementation,
  no production call site can be modified here. Tooling-only adapters were
  added for listener start, worker start, writes, and fallback. Each invokes
  the retirement fence before its callback; tests prove a retired callback is
  never entered. No service, LaunchAgent, configuration, or real state was
  touched.

### Deliberate API incompatibility

The original brief's three-argument Python interfaces could not prove roster
authority, freshness, predecessor lineage, or atomic CAS. They were
intentionally replaced by security-complete contracts requiring verified
control, roster, signer, CAS, explicit next OID, and the relevant signed
artifacts. This is a tooling API change only; there are no production Synapse
callers in this repository.

### Correction verification

```text
uv run --no-sync pytest tests/tools/test_synapse_cutover.py tests/tools/test_absorption_control_record.py tests/tools/test_absorption_safety.py tests/tools/test_consumer_migration.py tests/tools/test_absorption_inventory.py -q
73 passed

uv run --no-sync ruff check src/memo/operational_signing.py tools/memflow_absorption tests/tools
All checks passed!

uv run --no-sync mypy tools/memflow_absorption
Success: no issues found in 15 source files

git diff --check
passed
```

The broader `tests/tools` result after this correction is 123 passed and the
same 3 unrelated `test_synapse_data.py` failures from the concurrent
authenticated-epoch requirement.

## Correction round 2 — committed evidence and admission freshness

The second review found nine remaining gaps between signed artifact presence
and end-to-end authority. Eight are now closed in this repository; the ninth
is documented as an explicit cross-repository cutover blocker:

- `synapse-preflight` parses the exact typed `ConsumerInventory` schema and
  runs `verify_consumer_inventory`. A raw envelope with a valid signature but
  an invalid row kind or shape is rejected before plan admission.
- `prepare_synapse_retirement` deterministically rebuilds the
  `ConsumerReplacementPlan` from the roster-verified inventory, capability
  manifest, and exact Memo binary. The supplied plan must match the rebuilt
  canonical authority bytes; self-consistent caller rows are not authority.
- Every signed replacement control record is roster-verified before
  compare-and-swap. An invalid signer or envelope leaves the CAS head
  unchanged.
- `COMMITTED -> VERIFIED` consumes the typed signed independence receipt,
  inventory, retirement manifest, and both scan receipts. It calls
  `verify_independence_receipt` before CAS and stores the exact verified
  receipt digest.
- `sign_control_record` freezes peer votes after `QUIESCED`, retirement
  manifest and active-state receipt after `STAGED`, retirement epoch after
  `COMMITTED`, and the independence receipt after `VERIFIED`, in addition to
  the preflight authority already frozen in round 1.
- `synapse-verify` accepts only a signed `VERIFIED` control. The receipt must
  bind the immediately preceding `COMMITTED` control OID, and its exact digest
  must equal `control.independence_receipt_sha256`. A fresh receipt presented
  with an unadvanced `COMMITTED` control is rejected.
- The public `verify_independence_receipt` repeats all cross-artifact
  invariants itself: distinct boot IDs, strictly later post-reboot capture,
  exact inventory scan-receipt binding and aggregate source digest, plus the
  control-bound retirement-manifest digest.
- Tooling admission adapters no longer accept a cached
  `VerifiedControlRecord`. Each listener, worker, write, or fallback admission
  reads the current CAS head, verifies its signature against the roster, and
  applies the epoch fence before invoking the callback.

### Production runtime boundary map and blocker

There is no production Synapse runtime boundary in this Memo repository.
Repository-wide symbol mapping found `before_listener_start`,
`before_worker_start`, `before_write`, and `before_fallback` only in
`tools/memflow_absorption/runtime_gate.py` and
`tests/tools/test_synapse_cutover.py`; there is no import or call from
`src/memo`, a package entrypoint, or a checked-in runtime configuration.
The listener/server/write/worker entrypoints under `src/memo` operate Memo,
not Synapse. Wiring the Synapse retirement fence into them would block the
replacement product and is therefore unsafe.

Production fence delivery is explicitly **not claimed**. Before a real
cutover, the tooling adapters must be wired at the actual listener, worker,
write, and fallback boundaries in the isolated Synapse worktree and verified
there. If that runtime is removed rather than started, the corresponding
entrypoints disappear with it. Until one of those two outcomes is evidenced,
production runtime-gate wiring remains a material cutover blocker. No service,
LaunchAgent, configuration, or live state was inspected or changed here.

### Correction-round-2 verification

```text
uv run --no-sync pytest tests/tools/test_synapse_cutover.py tests/tools/test_absorption_control_record.py tests/tools/test_absorption_safety.py tests/tools/test_consumer_migration.py tests/tools/test_absorption_inventory.py -q
84 passed

uv run --no-sync ruff check tools/memflow_absorption tests/tools/test_synapse_cutover.py
All checks passed!

uv run --no-sync mypy tools/memflow_absorption
Success: no issues found in 15 source files

uv run --no-sync pytest tests/tools -q
138 passed

git diff --check
passed
```
