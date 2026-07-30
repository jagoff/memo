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

## State-root and trusted-Git authority follow-up

Review follow-up status: delivered in Memflow commit
`45b22d6c64fe8c13b18eae4ad119986dd6cb2e27`.

The follow-up closes the independent-review BLOCKER and MEDIUM findings:

- A drain now resolves one absolute, canonical, non-symlink project worktree
  and its exact Git dir with a trusted absolute Git executable. Its durable
  state root is always `<trusted-git-dir>/memflow`; the separate signed-control
  repository and commit cannot select it.
- Authority Git calls use a closed environment containing only deterministic
  locale, no-replace, no-system-config, global-null-config, and no-optional-lock
  settings. Inherited `PATH`, `HOME`, XDG, loader, Git config, object,
  alternate-object, replace, and every other process variable are absent.
- Exact control transport remains a raw 40-hex commit whose tree contains only
  mode-`100644` `control.json`. Canonical JSON, logical control OID,
  Ed25519/roster, issuance/freshness, attempt, sequence/predecessor, signer,
  marker-mode bindings, and environment-JSON refusal remain enforced.
- Ordinary pre-fence writers retain the established non-Git fallback, including
  first-write creation of a missing project root. A drain carrying an expected
  control OID instead requires the exact Git worktree and fails closed.

RED evidence before the fix:

```text
3 failed
- hostile GIT_DIR selected alternate.git/memflow and hid real delivery=1
- fake git first in PATH impersonated a non-Git control repository
- authority command was relative and inherited PATH/HOME/XDG/loader state
```

GREEN evidence after the fix:

```text
Critical regressions: 3 passed
Cutover/fence suites: 59 passed
Focused cutover/daemon/launchd/delivery/Git/MCP/session matrix: 292 passed
Ruff: All checks passed
Mypy: Success: no issues found in 147 source files
git diff --check: passed
```

No production marker, drain, state root, service, LaunchAgent, listener, or
runtime configuration was read for mutation or changed by this follow-up.

## Git-entry authority BLOCKER follow-up

Independent review finding `6c2f6cc82d534be8a1730edf994912ac` is resolved
in Memflow commit
`8ebc663091f9b3e8cebf1b8e26b4938cfd83c47f`.

The trusted project authority now:

- opens the project `.git` entry with no-follow semantics and accepts exactly
  either a directory or a canonical regular gitfile;
- binds the entry's device, inode, mode, and, for a gitfile, its exact bytes;
- binds the resolved Git directory and the device, inode, and mode of every
  directory in its absolute component chain;
- revalidates those bindings before and after authority operations; and
- runs every later trusted absolute-Git operation against the pinned
  `--git-dir` and `--work-tree`, without repeating `git -C` discovery.

RED evidence before the fix:

```text
Authority regressions: 2 failed, 1 passed
- project/.git symlink selected alternate/.git/memflow instead of refusing
- a directory swap between the two discovery calls was not detected
- a legitimate linked-worktree gitfile remained accepted
```

GREEN evidence after the fix:

```text
Authority regressions, including exact gitfile-byte binding: 4 passed
Cutover/fence suites: 63 passed
Focused cutover/daemon/launchd/delivery/Git/MCP/session matrix: 296 passed
Ruff: All checks passed
Mypy: Success: no issues found in 147 source files
git diff --check: passed
```

No production marker, drain, restart, state root, service, LaunchAgent,
listener, or runtime configuration was read for mutation or changed by this
follow-up.

## Descriptor-bound state authority BLOCKER follow-up

Residual review finding `700cebd1f3904536b1ca3e4867b2f9b5` is resolved
in Memflow commit
`3c5b91ebdb7c1073713c0d13d6a6acee88871d78`.

The gate now retains its authority for its entire lifetime:

- canonical worktree, `.git` entry, resolved Git directory, state root, and
  cutover directory are opened with no-follow, close-on-exec descriptors;
- every root-to-leaf directory link is bound by device, inode, and mode and
  revalidated through `stat(..., dir_fd=parent_fd, follow_symlinks=False)`
  against the retained child descriptor;
- canonical worktree gitfiles additionally retain their exact inode and bytes;
- lock, marker, sentinel, in-flight ledger, and drain-fsync reads and writes
  are descriptor-relative and never use the diagnostic `Path` properties;
- the admission lock rejects unlink/recreate split brain after acquiring the
  flock and again after the protected operation;
- atomic writes use a unique no-follow `O_EXCL` temp, write-all, `fchmod`,
  file fsync, identity-checked `renameat`, target identity check, and directory
  fsync;
- gate caching and registration bind the Git/worktree/state/cutover identities,
  close losing candidates, and never close a cached gate with active
  admissions; and
- explicit close is idempotent, serialized, and refuses to orphan a live
  admission context.

RED evidence before the fix:

```text
5 failed
- post-build project/.git replacement hid real delivery=1 as clean=True
- state-root replacement hid real delivery=1 as clean=True
- cutover-directory replacement hid real delivery=1 as clean=True
- admission.lock unlink/recreate after flock was accepted
- FenceGate had no descriptor lifecycle close
```

GREEN evidence after the fix:

```text
Descriptor authority regressions: 10 passed
Cutover/fence suites: 73 passed
Focused cutover/daemon/launchd/delivery/Git/MCP/session matrix: 306 passed
Ruff: All checks passed
Mypy: Success: no issues found in 147 source files
git diff --check: passed
```

The additional GREEN cases cover worktree gitfile and resolved-target
replacement after construction, true cutover-name ABA during an atomic write,
partial-build and cache-loser FD cleanup, cache identity displacement, and
close refusal during an active admission. A concurrent mkdirat warning found
during the first integrated run was fixed with safe create-or-open plus parent
fsync; the final integrated run completed without warnings.

No production marker, drain, restart, state root, service, LaunchAgent,
listener, or runtime configuration was read for mutation or changed by this
follow-up.
