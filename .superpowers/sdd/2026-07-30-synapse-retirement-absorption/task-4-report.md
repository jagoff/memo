# Task 4 report — Memo-owned Synapse consumer replacements

## Delivered

- Added a deterministic replacement plan for active Synapse LaunchAgents. The
  plan maps WhatsApp ingest, watcher, recall daemon, nightly/dream, digest, and
  vault jobs to existing Memo CLI entrypoints. Dashboard, relay, gateway, and
  MCP routes produce an explicit client close/reconnect action instead of a
  compatibility service.
- Extended the signed consumer inventory rows with their machine-actionable
  label and captured launchd loaded state. Plain Synapse references are
  recognized for process/launchd observations without broadening source scans
  into false runnable consumers.
- Added an operator-only renderer that writes beneath the supplied staging
  root, pins `MEMO_NONINTERACTIVE=1`, emits `KeepAlive` only for persistent
  native jobs, collapses byte-identical Memo targets, and never edits or loads
  production LaunchAgents.
- Added fail-closed checks for inventory/manifest blockers, unmapped active
  jobs, duplicate old labels, plan digest mutation, unsafe/colliding labels,
  symlinked staging components or outputs, missing/mutated rendered files, and
  Synapse/Memflow references in staged paths or contents.
- Updated Memo's existing watcher renderer to pin
  `MEMO_NONINTERACTIVE=1`. No runtime wrapper or new CLI entrypoint was needed:
  every admitted replacement in this task already has a Memo-owned command.

## Verification

```text
uv run --no-sync pytest tests/tools/test_consumer_migration.py tests/test_runtime_isolation.py tests/test_watcher.py tests/test_whatsapp_ingest.py -v
52 passed

uv run --no-sync pytest tests/tools/test_consumer_migration.py tests/tools/test_absorption_inventory.py -q
15 passed

uv run --no-sync ruff check tools/memflow_absorption src/memo/runtime src/memo/watcher.py src/memo/cli_import.py src/memo/cli_dream.py src/memo/cli_ingest_daemon.py tests
All checks passed!

uv run --no-sync mypy src/memo
Success: no issues found in 463 source files

uv run --no-sync mypy tools/memflow_absorption/consumer_migration.py tools/memflow_absorption/inventory.py tools/memflow_absorption/schemas.py src/memo/watcher.py
Success: no issues found in 4 source files

git diff --check
passed
```

The brief's combined Mypy command lists both `src/memo/runtime` and its parent
`src/memo`, so Mypy aborts with a duplicate-module error before analysis.
Removing the duplicate path exposes pre-existing snapshot receipt loader typing
errors in `tools/memflow_absorption/snapshot.py`; that file is outside Task 4
and was not included in this commit.

## Scope / concerns

- This is a staged renderer only. It does not unload, overwrite, bootstrap, or
  delete any production configuration.
- Source/process inventory rows remain evidence, not launchd replacement rows;
  an active LaunchAgent without an admitted mapping blocks the plan.
- During work on the shared tree, concurrent snapshot commit `d14755bc`
  incorporated the Task 4 schema types (`ConsumerInventoryRow` extensions plus
  `ConsumerReplacement`/`ConsumerReplacementPlan`) together with its own
  SnapshotReceipt changes. This commit does not rewrite that shared history; it
  contains only the remaining Task 4 implementation, tests, and report.
- Concurrent unrelated snapshot/manifest tooling, secure-enclave work, and
  prior SDD ledgers/reports were preserved and excluded from this commit.
