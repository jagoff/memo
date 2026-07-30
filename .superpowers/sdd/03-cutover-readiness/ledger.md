# SDD ledger — Plan 03: cutover readiness

- Normative plan:
  `/Users/fer/repos/memo/.worktrees/memflow-absorption/docs/superpowers/plans/2026-07-29-memflow-absorption/03-cutover-readiness.md`
- Memo program workspace:
  `/Users/fer/repos/memo/.worktrees/memflow-absorption`
- Synapse Task 4 workspace:
  `/Users/fer/repos/synapse/.worktrees/memflow-absorption-runtime`
- Synapse Task 4 branch: `feat/memflow-absorption-runtime`
- Synapse Task 4 BASE: `45c146d5b5f4548528f4e6bbcd6909f2e0983b3b`
- Task 4: technical implementation delivered at `933445fd`; focused
  isolation/runtime-policy matrix `18 passed`; safety-adjacent matrix `20
  passed`; full Synapse suite `2326 passed, 5 skipped`; Ruff and strict mypy
  clean. A real commit+lock-versioned runtime built successfully, verified
  Synapse and its pinned shared-contract dependency, rendered three isolated
  plists, and returned the same digest on idempotent retry. No LaunchAgent was
  loaded/unloaded/replaced, and Memflow remained live and untouched.
  Independent review is still required before acceptance.
- Task 1: safe snapshot, signed capability-manifest, consumer-inventory,
  Synapse-retirement, and fresh-control-record tooling delivered at
  `f6ca3fff..7858d540`. Shared states/modes and fence/drain/final-fence/control
  schemas match the normative contract, and every CLI apply path is bound to a
  preexisting sentinel with the exact manifest SHA-256. Focused tooling `32
  passed`; signing/roster/atomic regression `60 passed`; full Memo non-slow
  `6138 passed, 18 skipped`; Ruff and mypy clean; frozen v1 hashes unchanged.
  Only temporary roots, in-memory keys, and sanitized fixtures were used. No
  live state changed and Memflow remains active. Independent review is still
  required before acceptance or any production evidence capture.
- Task 3: observable cross-process drain, strict Git/lock probes, final
  fsync-and-recheck, autonomous-loop admission, temporary operator CLI, and
  pre-listener/pre-thread startup refusal delivered in Memflow at
  `a3a6070912`. The required focused matrix is `224 passed`; full Memflow is
  `1906 passed` with one external deprecation warning; Ruff, mypy, installer
  syntax, and diff checks are clean. Signed ACTIVE rollback recovery and
  explicit-environment isolation are covered. No production marker, service,
  configuration, or state changed; the live job remains PID `2046` on
  `127.0.0.1:18766`. Independent review is still required before acceptance.
