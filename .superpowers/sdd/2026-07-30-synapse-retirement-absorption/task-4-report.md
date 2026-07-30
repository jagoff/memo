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

## Correction round 1

The consumer stage is now authority-bound and executable rather than a
best-effort label translation:

- `_mapping` is an exact reviewed-label table. Unknown labels, including names
  that merely contain an admitted substring, block the plan.
- `build_consumer_replacement_plan` requires a `VerificationRoster`, verifies
  both inventory and capability-manifest signatures/digests, and requires one
  admitted operation plus capability mapping (live evidence, parity route, and
  Memo target) for every active row.
- WhatsApp preserves only a signed allowlisted configuration. It requires
  exactly one `--all-chats` or `--include-chat` scope, rejects unknown,
  duplicate, conflicting, or relative-path options, and emits an executable
  absolute Memo command.
- Live process rows must correlate to the exact command of one active
  LaunchAgent or the plan blocks for manual operator resolution.
- Inventory and replacement rows now bind `RunAtLoad`, `KeepAlive`,
  `StartInterval`, `StartCalendarInterval`, `WatchPaths`, and
  `ThrottleInterval`. Periodic jobs without an authoritative trigger and
  persistent jobs without `KeepAlive` block instead of receiving invented
  defaults.
- Rendered plists use a resolved absolute executable outside project virtual
  environments and a Memo-owned environment containing only
  `MEMO_NONINTERACTIVE=1`.
- The renderer rejects `~/Library`, its ancestors/descendants, and a root named
  `LaunchAgents`. It uses descriptor-relative `O_NOFOLLOW` directory access and
  atomic writes, preflights hardlinks/symlinks before any write, and requires
  the root and `LaunchAgents` directory to contain exactly the planned outputs.
  Verification rereads those exact regular single-link files and byte-compares
  each one to the plan.

Correction verification:

```text
uv run --no-sync pytest tests/tools/test_consumer_migration.py tests/test_runtime_isolation.py tests/test_watcher.py tests/test_whatsapp_ingest.py -v
58 passed

uv run --no-sync pytest tests/tools/test_consumer_migration.py tests/tools/test_absorption_inventory.py -q
22 passed

uv run --no-sync mypy src/memo
Success: no issues found in 463 source files

uv run --no-sync mypy --follow-imports=skip tools/memflow_absorption/consumer_migration.py tools/memflow_absorption/inventory.py tools/memflow_absorption/schemas.py
Success: no issues found in 3 source files

uv run --no-sync ruff check tools/memflow_absorption/consumer_migration.py tools/memflow_absorption/inventory.py tools/memflow_absorption/schemas.py tests/tools/test_consumer_migration.py tests/tools/test_absorption_inventory.py src/memo/watcher.py tests/test_watcher.py
All checks passed!

git diff --check
passed
```

The full Ruff scope remains red on concurrent, unrelated source-receipt and
snapshot edits: unsorted imports and an unused descriptor snapshot in
`manifest.py`, one unused descriptor snapshot in `snapshot.py`, and an unsorted
`source_receipt.py.__all__`. The deduplicated full Mypy scope likewise reports
three pre-existing dynamic `SnapshotReceipt(**dict[str, object])` loader errors
in `snapshot.py`. Those files were not modified by Task 4. Shared-worktree
commit `4187276a` incorporated the schema/inventory and most planner/renderer
correction hunks while committing source-receipt work; this correction commit
preserves that history and contains the remaining Task 4 hardening,
regressions, and report.

## Correction round 2

The replacement stage now binds every executable detail to signed authority
without losing valid launchd configuration:

- WhatsApp's single-value options (`--retention-days`, `--since`,
  `--notes-dir`, and `--db`) reject duplicates. Retention is a canonical
  positive integer, `--since` is a canonical ISO date, and both filesystem
  options require absolute paths.
- Each reviewed replacement names its exact admitted CLI route. Planning
  requires one matching `OperationRoute.memo_cli`, verifies that the selected
  route's Memo method is in the capability target, and verifies that the full
  capability target exactly matches the methods of its signed nested routes.
  The route id and target are included in the replacement config digest.
- Signed `MEMO_*` environment is preserved, `MEMO_NONINTERACTIVE=1` is forced,
  unrelated variables are discarded, and `PATH` accepts only absolute,
  non-retired entries. The isolated Memo binary directory is first in `PATH`,
  and the binary must be named `memo`, so WhatsApp's nested `memo ingest`
  invocation resolves to the same isolated runtime.
- Staging rejects `/Library`, `/System/Library`, the current user's Library,
  other `/Users/<name>/Library` and `/home/<name>/Library` roots, and their
  ancestors or descendants.
- `KeepAlive` now preserves valid boolean or dictionary policies, including
  state dictionaries. `StartCalendarInterval` preserves one dictionary or an
  array of multiple canonical dictionaries, with key and range validation.

Correction verification:

```text
uv run --no-sync pytest tests/tools/test_consumer_migration.py tests/tools/test_absorption_inventory.py -q
34 passed

uv run --no-sync pytest tests/tools/test_consumer_migration.py tests/test_runtime_isolation.py tests/test_watcher.py tests/test_whatsapp_ingest.py -q
70 passed

uv run --no-sync pytest tests/tools -q
100 passed

uv run --no-sync mypy src/memo
Success: no issues found in 463 source files

uv run mypy --follow-imports=skip tools/memflow_absorption/schemas.py tools/memflow_absorption/inventory.py tools/memflow_absorption/consumer_migration.py
Success: no issues found in 3 source files

uv run ruff check tools/memflow_absorption/schemas.py tools/memflow_absorption/inventory.py tools/memflow_absorption/consumer_migration.py tests/tools/test_consumer_migration.py tests/tools/test_absorption_inventory.py
All checks passed!

git diff --check
passed
```

The brief's full Ruff scope remains red only on the concurrent, unrelated
`published_stat` unused local in `tools/memflow_absorption/snapshot.py`.
Task 4 does not modify that file.

Shared-worktree commit `e3152027` incorporated the round-2 implementation,
schemas, regressions, and this report while committing concurrent source
receipt tests. History was not rewritten; the dedicated Task 4 correction
commit records this handoff and verification outcome.
