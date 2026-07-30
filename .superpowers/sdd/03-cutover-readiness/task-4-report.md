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
