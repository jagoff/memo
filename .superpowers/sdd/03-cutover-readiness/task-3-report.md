# Task 3 implementation report — observable Memflow drain and startup refusal

Memflow BASE: `2c6438636c7ef6ed21c5d7e9f59929812439fcc1`

Memflow workspace:
`/Users/fer/repos/memflow/.worktrees/memflow-absorption-fence`

Memflow branch: `feat/memflow-absorption-fence`

Memflow technical commit:
`a3a607091228b2ee0194302355a06d6b5c173f44`

Status: implementation delivered and green; independent specification,
durability, security, and quality review is still required before acceptance.

## Delivered

- Replaced process-local drain counters with a canonical, atomically published
  and fsynced cross-process in-flight ledger under the existing authority lock.
  Every entry is bound to a process token, PID, start timestamp, and named
  mutation domains.
- Kept ordinary status conservative: stale entries remain blockers. Only a
  drain holding the exact verified QUIESCING control OID may reap PIDs that are
  definitely dead; permission ambiguity and PID reuse stay fail-closed.
- Bound the final fsync proof to the exact attempt and control OID. A
  QUIESCING zero is never clean before that proof, and an old or mismatched
  proof cannot satisfy a new attempt.
- Added `memflow cutover status --json` and
  `memflow cutover drain --control-oid OID --timeout SECONDS`. The drain never
  changes mode: it requires an unexpired, signed QUIESCING marker and matching
  freshly supplied verified control, waits for every blocker, fsyncs, and
  rechecks before returning canonical `clean=true` JSON.
- Added strict external blocker probes for the live sync lock, Git transaction
  markers, dirty Memflow events, exact upstream-ahead commits, and probe
  failures. Unverifiable state is invalid, never zero.
- Added recursive no-follow fsync coverage for event journals and local
  cursor/runtime state, rejecting symlinks and unsupported file types.
- Wrapped every mutation-capable daemon-loop iteration in durable
  `autonomous_loops` admission. An already-admitted iteration may finish, but
  no new iteration is scheduled after QUIESCING or RETIRED.
- Added startup authorization before MCP/FastMCP construction, listener bind,
  updater/loop creation, and launchd bootstrap. The launchd path reconstructs
  authority solely from the installed plist rather than the caller shell.
- Persisted only immutable cutover bindings and verification keys in an
  explicitly configured plist. Fresh verified-control envelopes are never
  persisted.
- Proved a newer signed ACTIVE/ABORTED rollback restores launchd startup while
  QUIESCING remains retryably refused and RETIRED remains permanent.
- Preserved pre-fence ACTIVE behavior and existing public daemon/CLI
  contracts.

## RED evidence

The focused Task 3 test was collected before the drain runtime existed:

```text
ModuleNotFoundError: No module named 'memflow.cutover_runtime'
```

## GREEN evidence

```text
Cutover/fence/loop/daemon/launchd/Git/delivery/session matrix:
224 passed in 8.91s

Ruff over Memflow source and touched tests:
All checks passed

Mypy over all Memflow source:
Success: no issues found in 147 source files

Full Memflow suite:
1906 passed, 1 external Starlette/httpx deprecation warning in 94.42s

bash -n install.sh:
passed

git diff --check:
passed
```

Coverage includes cross-process visibility, dead-entry reaping authority,
stuck-loop timeout, fsync-before-clean, expired/wrong/unverified control
rejection, shell-authority isolation, dirty/ahead/live-lock Git blockers,
symlink refusal, loop completion then self-stop, MCP pre-construction refusal,
launchd pre-bootstrap refusal, and signed rollback recovery.

## Production non-mutation proof

- No production marker, sentinel, control envelope, or drain proof was
  installed.
- No live config, capture shim, hook, LaunchAgent, listener, credential, or
  state was changed.
- The live job remained `running`, PID `2046`, with program
  `/Users/fer/repos/memflow/.venv/bin/memflow-mcp`.
- The same process remained bound only to `127.0.0.1:18766`.
- The technical worktree was clean after commit.

## Deferred to Plan 05

- This task does not install a production marker, quiesce Memflow, activate
  Memo, commit a new epoch, retire a runtime, or uninstall anything.
- Plan 05 alone owns the signed controller sequence, production drain,
  rollback healthcheck, final retirement fence, observation window, and
  eventual removal.

## Review package

Normative Task 3 range:

```text
2c6438636c7ef6ed21c5d7e9f59929812439fcc1..a3a607091228b2ee0194302355a06d6b5c173f44
```

Required independent checks:

1. Cross-process ledger locking, crash cleanup, PID ambiguity, nested
   admission, and corruption behavior.
2. Exact attempt/OID/control/expiry binding and fsync-proof durability.
3. Git/lock/writable-handle coverage and fail-closed probe behavior.
4. Loop scheduling boundaries, legacy join behavior, and startup refusal
   before every listener/thread path.
5. ACTIVE parity, rollback recovery, launchd environment authority, and
   absence of production mutation.
