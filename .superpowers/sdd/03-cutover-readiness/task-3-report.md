# Task 3 implementation report — observable Memflow drain and startup refusal

Memflow BASE: `2c6438636c7ef6ed21c5d7e9f59929812439fcc1`

Memflow workspace:
`/Users/fer/repos/memflow/.worktrees/memflow-absorption-fence`

Memflow branch: `feat/memflow-absorption-fence`

Status: implementation in progress.

## Scope and invariants

- Make in-flight admission observable across the running daemon and a separate
  operator CLI process; process-local zero must never claim a clean drain.
- Reclaim only entries whose owning PID is definitely dead. Ambiguity remains
  fail-closed.
- Require a fresh, matching verified QUIESCING control record and exact
  control OID before drain.
- Track mutation-capable autonomous-loop iterations and stop scheduling new
  iterations when the durable marker becomes QUIESCING or RETIRED.
- Include Git locks, dirty/unpushed state, and writable handles in the final
  zero proof.
- Fsync journals/cursors after all counters reach zero, then recheck every
  blocker before returning `clean=true`.
- Add temporary read-only `memflow cutover status --json` and controlled
  `memflow cutover drain --control-oid OID --timeout SECONDS`.
- Reject daemon and launchd startup before listener bind or thread creation.
- Preserve unfenced ACTIVE behavior and existing daemon/CLI contracts.
- Do not install a production marker, quiesce production, restart/unload a
  service, alter live configuration, or perform cutover activation.

## Required evidence

- Deterministic RED for cross-process counts, timeout/stuck-loop behavior,
  fsync-before-clean, exact control binding, and pre-bind startup refusal.
- Focused cutover/loop/daemon/launchd/Git tests.
- Full Memflow mypy, Ruff, installer syntax, diff check, and full suite.
- Read-only proof that the live Memflow process, binary, PID, port, and
  configuration remain unchanged.
