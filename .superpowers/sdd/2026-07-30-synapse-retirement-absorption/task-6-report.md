# Task 6 report — retirement audit tooling only

## Status

Tooling and regression coverage are complete. Operational retirement cleanup,
post-reboot proof, and release/docs cleanup were deliberately **not** executed.
No live `VERIFIED` control artifact was supplied, and Task 5 records the
production Synapse runtime-gate wiring as a material blocker.

## Delivered

- Added `build_independence_receipt(...)` as a deterministic, read-only
  filesystem negative scan. It rejects active `synapse`, `SYNAPSE_*`, and
  Memflow-runtime references; rejects symlink blind spots; hashes every scanned
  regular file; binds the scanned roots and manifest digest into the scan
  digest; and permits archived provenance only below explicitly separate
  archive roots, after roster verification of the signed manifest, and at exact
  source/test/golden paths listed by that manifest.
- Added the read-only `retirement-audit` CLI. It first reuses the complete
  signed `synapse-verify` authority chain (VERIFIED control, final manifest and
  inventory, post-stop/post-reboot scans, and independence receipt), then scans
  only explicitly named installed/archive roots. It never accepts `--apply`.
- Added a refusal-only `retirement-cleanup` CLI and
  `assert_retirement_cleanup_authority(...)`. They require:
  - coordinated and Synapse states both `VERIFIED`;
  - a positive retirement epoch;
  - complete lowercase SHA-256 values for the control, retirement manifest,
    consumer replacement authority, bounded data receipt, and independence
    receipt;
  - exact observed/expected digest equality;
  - control-bound manifest, consumer-plan, and independence digests; and
  - absolute, resolved, non-broad, non-repository, non-symlinked, distinct,
    non-overlapping cleanup paths.
- The cleanup validator has intentionally no success path. Even after all
  available checks pass, it reports the missing production runtime-gate
  evidence and signed exact-path deletion plan. It contains no unlink, remove,
  launchctl, process, socket, configuration, or service mutation primitive.
- Added regressions for unlisted references, clean roots, exact manifest-listed
  archive provenance, symlink blind spots, non-VERIFIED control, incomplete and
  mismatched digests, broad/unresolved paths, CLI audit reporting, and proof
  that refusal leaves candidate files untouched.

## Verification

```text
uv run --no-sync pytest tests/tools/test_retirement_audit.py -q
11 passed

uv run --no-sync pytest tests/tools/test_retirement_audit.py tests/tools/test_absorption_*.py -q
50 passed

uv run --no-sync pytest tests/tools -q
151 passed

uv run --no-sync ruff check tools/memflow_absorption/inventory.py tools/memflow_absorption/safety.py tools/memflow_absorption/__main__.py tests/tools/test_retirement_audit.py
All checks passed!

uv run --no-sync mypy tools/memflow_absorption
Success: no issues found in 15 source files

git diff --check
passed
```

## Operational steps still pending

These steps require a later operator-controlled window and new signed
authority; none was performed in this task:

1. Wire and verify the retirement fence at the real Synapse listener, worker,
   write, and fallback boundaries in the isolated Synapse worktree, or prove
   those entrypoints disappeared with an authorized runtime removal.
2. Produce a signed exact-path deletion plan. The current control record binds
   the consumer plan, retirement manifest, and independence receipt, but does
   not authorize filesystem deletion targets. The bounded data receipt is also
   not bound into the current control schema.
3. Reach and independently fetch/verify the terminal `VERIFIED` control on both
   peers, then supply the exact artifact digests required by the refusal-only
   command.
4. In an approved maintenance window: disable KeepAlive, close/reconnect
   clients, stop Synapse services, and remove only exact signed-plan targets.
5. Reboot both peers and capture new signed process, port, LaunchAgent, MCP
   registration, shell/config, state-root, source/runtime/wrapper, and package
   metadata observations.
6. Run `memo doctor --strict-runtime`, Memo MCP tool-list smoke, and the
   cross-Mac handoff/delivery/ACK/presence/continuity smoke; build and commit the
   final independence receipt.
7. Only after the audit is genuinely green, update Memo install/runtime docs
   and release manifests and remove/archive authorized Synapse files.

## Scope confirmation

No cleanup Step 4 action, reboot, service stop, LaunchAgent/config/state change,
Synapse repository modification/removal, documentation change, or release
manifest change was executed. Concurrent work was preserved.

## Correction round 1 — read-only and negative-proof hardening

The first review found five ways an inspection-only audit could overclaim
authority or completeness. All five now fail closed:

- `retirement-audit` compares the typed manifest digest from its second read
  and the completed negative-scan receipt's manifest digest to the manifest
  digest already proven by the terminal signed authority chain. A mismatch or
  concurrent manifest replacement is rejected.
- Every CLI roster read now uses a strictly read-only loader. It does not call
  `VerificationRoster.load`, because that API performs crash recovery and can
  write roster files or finish a pending pin. The replacement reads the
  existing Keychain binding without creating one, rejects pending roster/epoch
  recovery, verifies the complete signed roster history, compares pin state
  before and after, and rejects a concurrent authority change. Tests compare
  both the authority files and pin snapshot before/after successful and
  pending-recovery reads.
- `_safe_files` supplies an `os.walk` error callback and converts traversal
  permission/I/O failures into `InventoryError`; an omitted subtree can no
  longer produce `verified`.
- Mixed/binary files retain ASCII bytes via surrogate-escape decoding.
  `SYNAPSE_*`, Synapse, and Memflow-runtime references embedded beside invalid
  UTF-8 bytes remain visible to the negative scan.
- Cleanup now requires an explicit absolute authority root before validating
  candidate paths. Broad roots, missing/non-directory authorities,
  repository descendants, paths outside the boundary, missing targets,
  symlinks, FIFOs/sockets/devices, duplicates, and overlaps are rejected before
  the deliberate runtime-gate blocker.

Correction-round-1 verification:

```text
uv run --no-sync pytest tests/tools/test_retirement_audit.py -q
17 passed

uv run --no-sync pytest tests/tools/test_retirement_audit.py tests/tools/test_absorption_*.py -q
56 passed

uv run --no-sync pytest tests/tools -q
157 passed

uv run --no-sync ruff check tools/memflow_absorption tests/tools
All checks passed!

uv run --no-sync mypy tools/memflow_absorption
Success: no issues found in 15 source files

git diff --check
passed
```

The operational blockers and refusal-only behavior are unchanged. No runtime,
service, LaunchAgent, configuration, state, repository, release, or cleanup
mutation was performed.

## Correction round 2 — CLI roster path and entry classification

The second review identified two remaining fail-open edges. Both now reject
without recovery or mutation:

- `synapse-manifest` now obtains its roster through the same strictly
  read-only `_verification_roster` path as the other CLI commands; it no
  longer calls `VerificationRoster.load`. The production-style Keychain test
  stages a pending roster update, snapshots every authority file and the pin,
  invokes the CLI handler, verifies the read-only rejection, and proves both
  snapshots are unchanged.
- `_safe_files` now calls `lstat` for every directory and file entry returned
  by `os.walk`. A vanished entry, denied/stat failure, unsupported special
  file, or traversal-time type change becomes `InventoryError`; no
  `Path.is_file`/`Path.is_symlink` false result can silently omit an entry.

Correction-round-2 verification:

```text
uv run --no-sync pytest tests/tools/test_retirement_audit.py -q
20 passed

uv run --no-sync pytest tests/tools/test_retirement_audit.py tests/tools/test_absorption_*.py -q
59 passed

uv run --no-sync pytest tests/tools -q
160 passed

uv run --no-sync ruff check tools/memflow_absorption tests/tools/test_retirement_audit.py
All checks passed!

uv run --no-sync ruff format --check tools/memflow_absorption/inventory.py tools/memflow_absorption/__main__.py tests/tools/test_retirement_audit.py
3 files already formatted

uv run --no-sync mypy tools/memflow_absorption
Success: no issues found in 15 source files

git diff --check
passed
```

Cleanup remains refusal-only. No runtime, service, LaunchAgent,
configuration, state, repository, release, or cleanup mutation was performed.
