# Memo Configuration TUI Design

**Date:** 2026-07-11
**Status:** Approved design
**Scope:** Replace manual configuration discovery and editing with a terminal-native configuration center while retaining Markdown as the editable source of truth.

## Context

Memo has just moved persistent configuration to Markdown files:

```text
~/.config/memo/
  memo-config.md
  config/
    storage-config.md
    models-config.md
    search-config.md
    recall-config.md
    capture-config.md
    graph-config.md
    hooks-config.md
    advanced-config.md
```

The files contain fenced TOML blocks. Environment variables override Markdown,
Markdown overrides the tuned overlay, and the legacy `config.toml` remains a
fallback during migration.

This structure is portable and inspectable, but it is not a practical primary
configuration experience. Memo currently exposes 368 registered flags across
30 internal groups plus structured `Config` fields. Users have to know keys,
types, ranges, dependencies, and runtime effects before changing them.

The approved direction is a terminal-native TUI opened by `memo config`. The
TUI becomes the primary editor, while Markdown remains independently readable,
editable, and authoritative.

## Goals

- Open a terminal-native configuration center with `memo config`.
- Make common settings understandable without exposing internal implementation
  details first.
- Cover every persistible setting and surface non-persistible runtime state.
- Preserve the existing configuration precedence and Markdown source of truth.
- Provide a short first-run wizard when Markdown configuration does not exist.
- Support keyboard and mouse interaction in a real terminal.
- Stage edits, validate them, review a diff, and apply them safely as one batch.
- Explain environment overrides, derived values, platform restrictions, and
  service restart impact.
- Retain scriptable CLI subcommands for automation and headless environments.
- Avoid model loading or other heavyweight imports on TUI startup.

## Non-Goals

- Do not create a web UI, local HTTP server, WebView, or browser-delivered app.
- Do not use `textual serve` or add Textual Web deployment support.
- Do not replace Markdown with a database-owned configuration store.
- Do not persist session ids, hook context, CI controls, or other ephemeral
  `MEMO_*` values.
- Do not let the TUI modify the parent process environment.
- Do not silently install dependencies or invoke `brew`, `apt`, or `sudo`.
- Do not silently restart daemons, hooks, or background services.
- Do not change retrieval defaults or runtime behavior merely because the
  configuration surface changes.
- Do not remove the existing `memo config` subcommands.

## Fixed Product Decisions

- `memo config` opens the TUI when stdin and stdout are interactive TTYs and
  `MEMO_NONINTERACTIVE` is not enabled.
- With no TTY or with `MEMO_NONINTERACTIVE=1`, `memo config` prints concise
  command help and exits normally.
- Existing subcommands such as `show`, `set`, `unset`, `validate`, `migrate`,
  `path`, and `flags` remain available and scriptable.
- Missing Markdown configuration launches a four-step first-run wizard.
- The normal interface is a domain-based configuration center.
- The interface copy is English. Stable keys and filenames remain English.
- Edits are deferred in memory until the user reviews and applies them.
- Common settings appear first; advanced and experimental settings use
  progressive disclosure. Global search always sees the full catalog.
- Environment overrides are read-only and clearly labeled, but the underlying
  Markdown value remains editable.
- Applying changes may offer to restart affected services, but requires an
  explicit confirmation and shows the exact impact first.

## Framework Choice

Use Textual as a required runtime dependency:

```toml
"textual>=8.2.8,<9"
```

Textual supplies terminal-native screens, focus management, inputs, switches,
select controls, scrolling, keyboard and mouse events, responsive layout, and
headless `Pilot` testing. Memo will run it only as a terminal application.

The alternatives were rejected:

- Extending the current Rich `Live` dashboard would require memo to implement
  focus, forms, validation presentation, mouse input, scrolling, and most test
  infrastructure itself.
- Questionary is useful for sequential prompts but does not provide the
  persistent, searchable domain center required here.
- Gum and shell wrappers would add an external executable and bypass memo's
  typed Python configuration APIs.

## Inspiration Boundary

The design borrows useful interaction patterns from
[`jagoff/universal-shell-gui-framework`](https://github.com/jagoff/universal-shell-gui-framework):

- explicit TTY detection;
- consistent contracts for menu, input, confirmation, progress, and status;
- state-aware defaults;
- semantic success, warning, error, and information feedback;
- a safe sequence of inspect, confirm, backup, write, validate, report, and
  rollback;
- clear cancellation and finish states.

It does not copy the framework's shell/Gum implementation, dependency
auto-installation, direct file mutation, monolithic example structure, or
emoji-heavy presentation.

## Architecture

The architecture has four layers.

### 1. CLI Entry

`src/memo/cli_config.py` remains Click wiring. The `config` group uses
`invoke_without_command=True` and opens the TUI only when no subcommand was
provided, both input and output are TTYs, and the registered noninteractive
flag is disabled.

The entrypoint must not import Textual until it decides to launch the app. This
keeps `memo config show`, other memo commands, hooks, and headless calls free of
the Textual import cost.

### 2. Textual Presentation

New focused modules under `src/memo/config_tui/` own presentation only:

```text
config_tui/
  __init__.py
  app.py
  screens.py
  widgets.py
  controls.py
  styles.tcss
```

Screens consume view models and emit user intents. They do not import
filesystem helpers, parse Markdown, resolve environment precedence, or restart
services.

### 3. Configuration Domain

Focused non-UI modules own reusable configuration behavior:

```text
config_catalog.py
config_session.py
config_apply.py
config_impact.py
```

- `config_catalog.py` defines the complete setting catalog.
- `config_session.py` loads configured/effective state and owns the draft.
- `config_apply.py` validates and applies a transactional batch.
- `config_impact.py` maps changed settings to explicit activation actions.

The CLI subcommands may reuse these services so the TUI and CLI do not develop
different validation or write semantics.

### 4. Existing Adapters And Runtime

- `config_md.py` remains the Markdown parser and storage adapter.
- `Config.from_env()` and `flags.flag()` remain the effective runtime seams.
- Environment variables and the tuned overlay remain read-only inputs.
- Legacy `config.toml` remains a migration fallback.
- Runtime install/daemon/hook helpers execute only confirmed impact actions.

## Configuration Catalog

### SettingSpec

Every setting exposed to the configuration system has one typed definition:

```python
@dataclass(frozen=True)
class SettingSpec:
    key: str
    label: str
    description: str
    domain: str
    section: str
    kind: SettingKind
    default: object
    env_name: str | None
    config_field: str | None
    minimum: float | None
    maximum: float | None
    choices: tuple[SettingChoice, ...]
    visibility: Visibility
    policy: PersistencePolicy
    risk: RiskLevel
    platforms: frozenset[str]
    requires: tuple[str, ...]
    conflicts: tuple[str, ...]
    restart_targets: tuple[str, ...]
    sensitive: bool
```

The exact Python shape may use enums or smaller nested dataclasses, but these
semantics are required.

### Required Classifications

Each setting has an explicit persistence policy:

- `persistent`: safe to write to Markdown.
- `runtime-only`: ephemeral environment/session/hook state, visible read-only
  under diagnostics.
- `secret`: value is masked and stored through memo's encrypted secret storage,
  never as plaintext Markdown.
- `derived`: computed from platform or runtime state, visible read-only.

Each setting also has a visibility level:

- `common`: shown directly in its domain.
- `advanced`: hidden behind the domain's advanced disclosure.
- `experimental`: separated, warned, and never enabled implicitly.

Catalog construction starts from the existing `Config` fields and `FlagSpec`
registry, then applies explicit user-facing metadata. It must not infer whether
a flag is persistible from its name. CI fails if a registered config field or
flag lacks a policy, if public keys collide, or if a persistent key cannot map
to the Markdown adapter.

The existing `_FIELD_PATHS` and flag path mapping in `config_md.py` become
catalog-owned mappings so parsing, CLI output, validation, and the TUI share one
definition.

## Setting State And Drafts

### SettingState

The TUI receives a `SettingState` that separates:

- configured Markdown value;
- effective value;
- effective source;
- default value;
- active environment override;
- platform availability;
- validation result;
- dirty state.

This separation prevents the UI from presenting a persistent edit as already
active when an environment variable still overrides it.

### ConfigDraft

The session service owns an in-memory `ConfigDraft` containing set and unset
operations. Editing never writes directly to disk. Reverting a field removes
its draft operation; discarding clears the whole draft.

An `unset` operation previews the value and source that will become effective
after the Markdown override is removed.

### ApplyPlan

Review builds an immutable `ApplyPlan` containing:

- changed keys with old and new configured/effective values;
- affected Markdown files;
- validation errors and warnings;
- source-shadow warnings;
- required or optional runtime activation actions;
- a fingerprint of every source file read by the session.

## Terminal Experience

### First-Run Wizard

When Markdown configuration is absent, `memo config` opens a four-step wizard:

1. Data and vault locations.
2. Detected hardware and recommended model profile.
3. Integrations, hooks, and automatic recall.
4. Privacy/capture choices, final summary, and validation.

The wizard uses detected current state as defaults and explains deviations from
the recommendation. Completing it writes Markdown through the same draft and
apply transaction used by the normal configuration center, then opens the
normal overview.

Cancellation leaves no partial config files.

### Main Configuration Center

The normal screen has:

- a compact header with current validity and pending-change state;
- a stable domain navigation area;
- a global search input;
- the selected domain's settings;
- source and status badges;
- actions to discard or review pending changes.

User-facing domains are:

```text
Overview
Storage
Models
Search
Recall
Capture
Graph
Hooks
Maintenance
Advanced
```

Internal registry groups map into these user-facing domains. The UI does not
expose 30 implementation groups as primary navigation.

### Progressive Disclosure

- Each domain shows roughly 5 to 12 common settings first.
- Advanced settings are collapsed by default.
- Experimental settings are separate and include a risk explanation.
- Global search includes all catalog entries, even collapsed entries.
- Search matches label, description, stable key, environment name, and domain.
- Runtime-only and derived values live under `Advanced > Runtime diagnostics`
  and remain read-only.

### Typed Controls

A control factory maps catalog kinds to reusable widgets:

- booleans use switches;
- enumerations use select/radio controls;
- bounded integers and floats use validated numeric inputs with increment and
  decrement actions;
- strings use validated inputs;
- paths use inputs with expansion, existence/access checks, and a normalized
  preview;
- secrets use masked inputs and the encrypted secret adapter;
- derived/runtime-only values use read-only rows.

Every editable row shows a friendly label and short description. The stable key
appears as secondary technical information, not as the primary label.

### Environment Overrides

When a `MEMO_*` environment variable wins:

- the row shows the effective value with an `ENV` source badge;
- the underlying Markdown value remains editable;
- review warns that the persistent edit will not become effective until the
  environment override is removed;
- the TUI never edits or unsets the parent environment.

### Review And Apply

Review shows:

- key-level old/new values;
- whether effective behavior changes immediately;
- warnings and validation results;
- files that will be written;
- affected daemons, hooks, or services.

The user chooses between `Save only` and `Apply and restart affected services`.
No service action runs without this confirmation.

### Responsive Terminal Layout

- At 100 columns or wider, use the full sidebar and settings pane.
- Below 100 columns, replace the sidebar with a compact domain selector.
- At 80x24, use single-column forms and full-screen modal screens.
- Below the supported minimum, show a clear size requirement and point to the
  scriptable CLI subcommands.

The UI must not overlap, truncate controls into unusability, or resize rows when
badges and validation messages appear.

## Data Flow

### Load

1. The TTY-aware entrypoint launches `ConfigApp`.
2. The session service builds the catalog and snapshots config file hashes plus
   mtimes.
3. The Markdown adapter loads configured values.
4. The effective resolver uses the existing precedence:

```text
explicit kwargs
environment
Markdown
tuned overlay
legacy config.toml
defaults
```

5. The session service produces source-aware view models for Textual.

### Edit

1. A widget emits a typed set/unset intent.
2. The session service updates the draft.
3. Field validation runs immediately.
4. Cross-setting validation runs against the whole projected configuration.
5. The TUI updates pending state without touching disk.

### Apply

1. Build and display the `ApplyPlan`.
2. Re-read source fingerprints before commit.
3. If files changed externally, rebase independent keys. If the same key
   changed externally and in the draft, require explicit conflict resolution.
4. Render complete affected TOML blocks while preserving Markdown outside
   fenced TOML blocks byte-for-byte.
5. Write temporary files in the target directories.
6. Parse and validate the temporary files using the production loader.
7. Create a timestamped backup and transaction manifest.
8. Replace each file atomically. A multi-file failure restores the whole batch.
9. Clear config caches and resolve effective state again.
10. Run only the confirmed impact actions and report each result separately.

Atomicity is guaranteed per file. Multi-file behavior is transactional through
the manifest, backups, rollback, and startup recovery.

## Markdown Preservation

The TUI must preserve user ownership of Markdown:

- Text outside fenced TOML blocks remains byte-identical.
- Unrelated TOML blocks remain unchanged.
- Affected TOML tables may be rendered canonically; exact whitespace and TOML
  comments inside an affected block are not guaranteed.
- Unknown or invalid blocks are never silently discarded.
- Hand-edited Markdown wins after reload and participates in conflict detection.

## Runtime Impact

`config_impact.py` maps keys to explicit targets such as recall daemon, watcher,
hooks, or no restart. It produces descriptions and callables but does not run
them during planning.

After a successful save:

- `Save only` reports which services still use old state.
- `Apply and restart affected services` invokes existing lifecycle helpers in a
  deterministic order.
- A restart failure does not pretend the file transaction failed. The UI reports
  `saved but not fully activated` and offers retry or restore-from-backup.

## Error Handling And Recovery

Use `MemoError` subclasses for configuration domain failures, including catalog
coverage, validation, external conflict, transaction, and activation failures.
Click and Textual translate those errors into their own presentation.

### Invalid Configuration At Startup

Open a recovery screen that reports file, block, key, and cause. Available
actions are:

- open the file in `$EDITOR`;
- restore the latest known-good backup;
- continue in read-only mode;
- exit without changes.

The TUI never overwrites a malformed source automatically.

### Invalid Draft

Show the error beside the control and keep review/apply disabled until all
blocking field and cross-setting errors are resolved.

### Platform Restrictions

Unavailable settings remain visible but disabled, with a concrete platform or
dependency requirement. Missing optional capabilities are diagnosed; the TUI
does not install them.

### External Changes

Independent external changes rebase automatically. Same-key conflicts open a
comparison screen with current disk, session baseline, and draft values. The
user must choose before applying.

### Transaction Failure

Restore the complete previous batch, keep the draft in memory, and show the
failed stage and path. A transaction manifest left by interruption is detected
on the next launch and recovery runs before normal editing.

### Cancellation And Terminal Cleanup

Cancel, Escape, or SIGINT never writes a pending draft. Textual teardown must
restore the terminal even when an exception occurs.

### Secrets

Sensitive values are masked in controls and excluded from logs, diffs,
exceptions, notifications, and test snapshots. Plaintext secret values never
enter Markdown.

## Performance Requirements

- Do not import `mlx`, `mlx_lm`, embedding models, rerankers, or LLM helpers on
  startup.
- Lazy-import Textual only for the interactive no-subcommand path.
- Build the catalog deterministically without filesystem or model work.
- Load Markdown once per session and invalidate through source fingerprints.
- Filter the full catalog locally; typing in search must not trigger disk reads.
- Run service health detection asynchronously so it cannot block initial paint.

## Testing Strategy

### Catalog Tests

- Every `Config` field and registered `FlagSpec` has exactly one policy.
- Every persistent setting has a stable public key and Markdown mapping.
- Keys and environment names are unique.
- Choices, defaults, bounds, dependencies, and conflicts are internally valid.
- Runtime-only, secret, and derived settings cannot use the persistent writer.

### Session And Resolution Tests

- Configured and effective values remain distinct.
- Existing precedence is unchanged.
- Environment badges and shadow warnings are correct.
- Unset previews the correct fallback source and value.
- Platform availability and derived values are read-only.
- Field and cross-setting validation update draft state correctly.

### Markdown And Transaction Tests

- Parse/render round trips preserve outside-Markdown bytes.
- Affected TOML blocks are canonical and valid.
- Temporary files are validated before replacement.
- Fault injection at each transaction stage restores the complete prior batch.
- Interrupted manifests recover on next startup.
- External edits rebase or conflict at key granularity.
- Backups and caches update only after a successful commit.

### Textual Tests

Use `App.run_test()` and `Pilot` to cover:

- first-run wizard completion and cancellation;
- domain navigation by keyboard and mouse;
- global search and progressive disclosure;
- every control kind;
- inline validation;
- source badges and environment warnings;
- review, discard, apply, conflict, recovery, and activation-result screens;
- clean exit and terminal teardown.

Add snapshots at 80x24, 100x30, and 140x45. Sensitive values must be absent
from all snapshots.

### CLI And Integration Tests

- `memo config` launches only with interactive stdin/stdout.
- Headless invocation prints help and exits without hanging.
- Existing subcommands and JSON output remain compatible.
- Tests isolate `MEMO_CONFIG_DIR`, `MEMO_DATA_DIR`, `MEMO_STATE_DIR`, and
  `MEMO_NONINTERACTIVE`.
- Impact actions use fakes; tests never restart real user services.

### Verification Commands

Follow repository CI order:

```bash
uv run --no-sync ruff check src/ tests/
uv run --no-sync mypy src/memo
uv run --no-sync pytest -m "not slow" -n auto --timeout=120 --cov=memo --cov-report=term-missing
```

Also run focused config/TUI suites and the macOS runtime smoke. Retrieval evals
are not required unless implementation changes ranking defaults or behavior.

## Migration And Compatibility

- Existing Markdown installs open directly in the normal configuration center.
- Legacy-only installs launch the first-run experience with a migration option
  that reuses `memo config migrate` semantics and creates a backup.
- Existing Markdown values classified as runtime-only are reported as
  actionable validation errors. The recovery screen can remove them only after
  explicit confirmation; they are never migrated silently.
- Existing config subcommands keep their names and machine-readable output.
- `memo init` and `memo config init` continue to work for automation.
- Runtime code continues consuming `Config.from_env()` and typed flag accessors;
  the TUI does not introduce a second runtime configuration path.

## Acceptance Criteria

The feature is complete when:

1. `memo config` opens a terminal-native Textual app on a TTY, respects
   `MEMO_NONINTERACTIVE`, and never starts a web server.
2. Missing Markdown config launches the approved four-step wizard.
3. Every persistent setting is editable and every other registered setting is
   explicitly classified and visible where appropriate.
4. Common, advanced, experimental, runtime-only, secret, and derived states are
   distinguishable.
5. Configured, effective, default, and environment-override values cannot be
   confused in the UI.
6. Changes remain pending until review and explicit apply.
7. Multi-file apply validates before commit, backs up previous files, rolls back
   on failure, and recovers interrupted transactions.
8. External Markdown edits are never silently overwritten.
9. Service restarts are impact-aware, visible, and explicitly confirmed.
10. The interface remains usable at 80x24 and adapts at wider sizes.
11. Existing headless config commands remain compatible and tests never touch
    the real vault or state directory.
12. Ruff, mypy, focused suites, CI-parity pytest, and macOS smoke pass.

## Implementation Order

The later implementation plan should preserve these boundaries and stage work
in this order:

1. Add Textual dependency and catalog types/coverage enforcement.
2. Centralize config field/flag path mappings in the catalog.
3. Build source-aware session, draft, validation, and apply-plan services.
4. Add preserving Markdown batch rendering and transaction recovery.
5. Add impact planning and confirmed activation adapters.
6. Build reusable Textual controls and the main configuration center.
7. Build first-run, review, conflict, recovery, and result screens.
8. Wire the TTY-aware Click entrypoint while preserving subcommands.
9. Add tests, snapshots, documentation, and runtime smoke coverage.
