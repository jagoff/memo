# Task 4 implementation report — isolate Synapse runtime from Memflow

Synapse BASE: `45c146d5b5f4548528f4e6bbcd6909f2e0983b3b`
Synapse technical commit: `933445fd3ca909cdc661087d992341d9e169f4c5`

Status: implementation delivered and green; independent specification,
durability, and quality review is still required before acceptance.

## Delivered

- Added a Synapse-owned runtime installer whose identity is the exact Git
  commit plus SHA-256 of `uv.lock`.
- Default execution is a read-only dry-run. `--apply` requires an exact
  expected commit and dependency-lock digest, plus a clean source tree.
- Pinned and locked the previously implicit `consciousness-contracts`
  dependency, so the isolated runtime can import the package without borrowing
  Memflow's environment.
- Builds the environment with an independent Python >=3.11, locked `uv sync`,
  non-editable project installation, and copied dependency artifacts.
- Verifies the installed Synapse version and imports both `synapse.cli` and
  `consciousness_contracts` before publication.
- Renders dashboard, runtime-loop, and watcher templates with the versioned
  runtime Python and an absolute Memo CLI. The rendered plists contain no
  Memflow interpreter, root, binary, or environment dependency.
- Hashes paths, modes, symlink targets, and file contents into a runtime
  digest; records the digest and immutable identity in `manifest.json`.
- Fsyncs staged artifacts, atomically renames the complete version, and
  atomically updates a small `current` file. Failed builds preserve the
  previously current version.
- Reuses a verified existing version without rebuilding or rewriting the
  current pointer.
- Contains no service activation surface: no `launchctl`, live LaunchAgent
  path, load, unload, bootstrap, or bootout operation.

## RED evidence

The focused Task 4 test was collected before the installer existed:

```text
ModuleNotFoundError: No module named 'synapse.runtime_install'
1 error during collection
```

## GREEN evidence

```text
Task 4 isolation + existing runtime policy:
18 passed

Task 4 + runtime policy + environment-registry safety:
20 passed in 1.51s

Ruff over all Synapse source, tests, and scripts:
All checks passed

Ruff format over the three touched Python paths:
3 files already formatted

Strict mypy over the full Synapse source/test tree:
Success: no issues found in 281 source files

Full Synapse suite:
2326 passed, 5 skipped in 182.35s

git diff --check:
passed
```

## Real runtime proof

The committed source built in a disposable state root with:

```text
commit:
933445fd3ca909cdc661087d992341d9e169f4c5

dependency-lock SHA-256:
1b63a148bb6c26fa47aa3c44ede42b989ccc5f7b7ca73fc5e7adfe48fea5dbb5

runtime digest:
571c36bac24986099db6f6a42ba76bb41969341293e2518c2523d40d1370e932

installed Synapse version:
0.1.0

second apply:
reused=true, same runtime digest
```

All three rendered plists passed `plutil -lint`; their `ProgramArguments[0]`
resolved to the versioned runtime Python. The venv Python resolved to the
independent Homebrew interpreter, not a Memflow path. Published plists and
runtime metadata had the required executable/0644/0755 modes.

## Compatibility and deferred activation

- The build proof used only a disposable state root, which was then moved to
  Trash. No production runtime path was created.
- Existing live Synapse plists were inspected read-only and remain unchanged.
- No LaunchAgent was loaded, unloaded, replaced, or restarted.
- Memflow's service, hooks, state, pending ACKs, port, and configuration were
  not modified.
- Task 5 still owns Memo-only backend routing. Plan 05 still owns live config
  replacement and eventual Memflow retirement.

## Review package

Normative Synapse range:

```text
45c146d5b5f4548528f4e6bbcd6909f2e0983b3b..933445fd3ca909cdc661087d992341d9e169f4c5
```

Required independent checks:

1. Commit/lock identity, clean-tree enforcement, and dependency pinning.
2. Build isolation, installed-version/import verification, and forbidden-path
   detection.
3. Runtime digest coverage, fsync/rename boundaries, current-pointer
   atomicity, and retry idempotency.
4. Template completeness, modes, exact program paths, and absence of Memflow
   runtime/environment dependencies.
5. Dry-run purity and absence of every service activation mutation.

## P03-T04 adversarial fix-loop — 2026-07-30

Synapse fix commit:
`f32e789` (`fix: harden immutable runtime installation`)

Status: DONE. The follow-up preserves the pre-review implementation and closes
the 2 BLOCKER and 2 HIGH findings without activating services or touching live
plists.

### RED evidence

Four focused adversarial regressions failed against the uncommitted pre-review
state:

```text
4 failed, 25 deselected in 46.72s
```

- mutate -> capture -> restore built hostile transient source bytes instead of
  the expected commit bytes;
- recursive gitlinks were empty and the pinned child payload was absent;
- locally modified runtime bytes plus a locally resealed manifest were accepted
  when the caller supplied the recalculated digest;
- a first `hdiutil detach` timeout escaped without force-detach or cleanup.

### Delivered fixes

- Replaced detached-worktree trust with Git plumbing materialization. Commit,
  tree, and blob payloads are independently content-address verified; the
  expected complete source digest is computed directly from those verified
  objects, then checked against both the mutable materialization and the
  kernel-read-only mounted image.
- Recursively materialized every gitlink from its pinned submodule object
  database, required exact `.gitmodules` agreement, rejected missing object
  databases/cycles/unsafe paths, and included every submodule directory, file,
  mode, type, symlink, and payload in the complete digest.
- Replaced the caller-only runtime digest with a frozen external attestation
  bound to attestation schema, commit OID, dependency-lock SHA-256, runtime
  SHA-256, runtime payload schema, and project version. CLI attestations must be
  canonical regular files on a read-only filesystem or carry a kernel immutable
  flag. Apply and reuse both require the full tuple.
- Made detach and force-detach exception-safe, based cleanup on independently
  observed mount state, protected active image roots with an advisory lease,
  and added next-start recovery for orphan mounts and candidates.
- Preserved the full tree digest, read-only source mount, fsync/atomic rename,
  atomic `current`, reuse idempotency, dry-run purity, path safety, and the
  complete absence of service activation.

### GREEN evidence

```text
focused runtime isolation + runtime policy:
43 passed in 38.02s

strict mypy, full Synapse source/test tree:
Success: no issues found in 281 source files

Ruff, full Synapse source/test tree:
All checks passed

Ruff format, owned paths:
2 files already formatted

git diff --check, owned paths:
passed
```

A temp-only real macOS repro materialized and mounted
`933445fd3ca909cdc661087d992341d9e169f4c5` from Git objects, matched committed
`uv.lock` bytes exactly, verified the mounted complete digest, exposed no
`.git`, and left no `synapse-runtime-image.*` candidate. All adversarial
submodule, attestation, detach, and orphan-recovery repros used pytest temporary
roots. No runtime service, LaunchAgent, plist, Memo code, Memflow code, or
ledger was changed.

## P03-T04 provenance follow-up — 2026-07-30

Status: DONE. Review found that a caller could supply an in-memory
`RuntimeAttestation` rather than the immutable external artifact required by
the CLI. The public dataclass constructor now always fails closed, and the
installer requires an exact internal provenance token in addition to the full
attestation tuple. Only the canonical immutable-file loader and the private
test-fixture constructor can mint a usable attestation. `dataclasses.replace()`
and public forged construction fail closed before installation.

Validation: focused runtime isolation plus runtime policy passed (45 tests);
the new constructor/replace regression subset passed (8 tests). Strict mypy
passed over 281 files; Ruff, format check, and `git diff --check` passed. No
service activation or live plist mutation occurred. The Synapse commit contains
only `src/synapse/runtime_install.py` and `tests/test_runtime_isolation.py`.
