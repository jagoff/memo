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
- Added the retired-runtime request fence. Read-only status remains available;
  stale epochs fail first, while startup, write, and fallback requests at the
  committed epoch return `synapse.cutover.retired`.
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
