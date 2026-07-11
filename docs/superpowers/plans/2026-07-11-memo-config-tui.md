# Memo Configuration TUI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a terminal-native `memo config` application that exposes every memo setting safely while retaining Markdown as the editable source of truth and all existing config subcommands for automation.

**Architecture:** Consolidate all terminal UI implementation under `src/memo/tui/`. The Textual configuration app consumes a typed catalog and a source-aware draft session; a preserving Markdown adapter and transactional apply service own writes, while impact actions remain explicit and injectable. Top-level `cli_config.py` and `cli_tui.py` stay wiring-only and lazy-import TUI entrypoints.

**Tech Stack:** Python 3.13+, Textual 8.2.8 (`>=8.2.8,<9`), Click, Rich, Pydantic v2, stdlib `tomllib`, existing `tomli-w`, pytest, pytest-asyncio 1.4, pytest-textual-snapshot 1.1, Textual `Pilot`, ruff, mypy.

## Global Constraints

- Perform every implementation commit on the `tui` branch; never merge or
  cherry-pick implementation commits into `master` during this plan.
- The feature is a terminal TUI only: no HTTP server, WebView, browser delivery, `textual serve`, or Textual Web support.
- All widget, layout, theme, input-loop, terminal-rendering, catalog, draft, transaction, and impact code introduced for this feature lives below `src/memo/tui/`.
- `src/memo/cli_config.py` and `src/memo/cli_tui.py` remain top-level wiring-only modules per repo convention.
- Existing `dashboard_tui.py`, `dashboard_panels.py`, and `setup/picker.py` implementation moves below `src/memo/tui/`; compatibility modules contain imports only.
- Markdown files remain the persistent editable source of truth.
- Existing precedence remains: explicit kwargs > environment > Markdown > tuned overlay > legacy TOML > defaults.
- `MEMO_NONINTERACTIVE=1` and non-TTY invocations never launch Textual.
- Existing `memo config show/set/unset/validate/migrate/path/flags` behavior and JSON output remain compatible.
- Every `Config` field and registered `FlagSpec` receives a `persistent`, `runtime-only`, `secret`, or `derived` policy.
- Runtime-only, secret, and derived values cannot reach the persistent Markdown writer.
- Common settings appear first; advanced settings are folded; experimental settings are isolated and warned; global search sees all catalog entries.
- The TUI never edits environment variables, installs dependencies, loads MLX/models, or restarts a service without explicit confirmation.
- Text outside affected fenced TOML blocks remains byte-identical.
- Affected TOML blocks may be rendered canonically; comments inside an affected block are not guaranteed.
- File replacement is atomic per file; multi-file consistency uses a manifest, backups, rollback, and startup recovery.
- Keep UI copy, stable keys, and filenames in English.
- Tests must isolate `MEMO_CONFIG_DIR`, `MEMO_CONFIG_FILE`, `MEMO_DATA_DIR`, `MEMO_STATE_DIR`, and `MEMO_NONINTERACTIVE`.
- Use `MemoError` subclasses for domain failures.
- CI order remains ruff, mypy, pytest.

---

## Execution Precondition

Before Task 1, run:

```bash
git branch --show-current
git merge-base --is-ancestor master HEAD
```

Expected: the first command prints `tui`; the second exits 0. If either check
fails, stop without editing files.

---

## File Structure

### New Package

```text
src/memo/tui/
  __init__.py                 # Stable terminal-UI entrypoint exports only
  common.py                   # Semantic theme constants and TTY helpers
  picker.py                   # Existing questionary picker implementation
  dashboard/
    __init__.py               # Dashboard exports
    app.py                    # Existing Rich Live dashboard loop
    panels.py                 # Existing dashboard renderables
  config/
    __init__.py               # run_config_tui export
    catalog.py                # SettingSpec catalog and config/flag bindings
    session.py                # SettingState, ConfigDraft, validation, source resolution
    apply.py                  # Source snapshots, preserving render, transactions
    impact.py                 # Impact planning and confirmed command execution
    controls.py               # Typed Textual control factory
    widgets.py                # Domain nav, setting rows, badges, summaries
    screens.py                # Wizard/review/conflict/recovery/result screens
    app.py                    # ConfigApp and main configuration center
    styles.tcss               # Textual terminal stylesheet
```

### Compatibility And Shared Files

- Modify `src/memo/dashboard_tui.py` into a re-export shim.
- Modify `src/memo/dashboard_panels.py` into a re-export shim.
- Modify `src/memo/setup/picker.py` into a re-export shim.
- Modify `src/memo/dashboard.py` to import from `memo.tui.dashboard` directly.
- Modify `src/memo/setup/__init__.py` to import picker types from `memo.tui.picker`.
- Modify `src/memo/config_md.py` to consume catalog bindings and expose preserving batch helpers.
- Modify `src/memo/cli_config.py` for TTY-aware no-subcommand launch only.
- Modify `src/memo/cli.py` so `config` bypasses the old first-run picker.
- Modify `pyproject.toml` for Textual and TUI snapshot test dependencies.

### Tests

```text
tests/test_tui_package.py
tests/test_config_catalog.py
tests/test_config_session.py
tests/test_config_apply.py
tests/test_config_impact.py
tests/test_config_tui_controls.py
tests/test_config_tui_app.py
tests/test_config_tui_screens.py
tests/test_cli_config.py
tests/test_config_md.py
tests/test_cli_init.py
```

---

### Task 1: Consolidate Existing Terminal UI Under `src/memo/tui/`

**Files:**
- Create: `src/memo/tui/__init__.py`
- Create: `src/memo/tui/picker.py`
- Create: `src/memo/tui/dashboard/__init__.py`
- Create: `src/memo/tui/dashboard/app.py`
- Create: `src/memo/tui/dashboard/panels.py`
- Modify: `src/memo/dashboard_tui.py`
- Modify: `src/memo/dashboard_panels.py`
- Modify: `src/memo/dashboard.py`
- Modify: `src/memo/setup/picker.py`
- Modify: `src/memo/setup/__init__.py`
- Create: `tests/test_tui_package.py`

**Interfaces:**
- Produces: `memo.tui.picker.PickerResult`, `memo.tui.picker.run_picker`, `memo.tui.dashboard.render`, `memo.tui.dashboard.run_tui`.
- Preserves: imports from `memo.setup.picker`, `memo.dashboard_tui`, and `memo.dashboard_panels`.
- Consumes: existing Rich dashboard and questionary picker behavior unchanged.

- [ ] **Step 1: Write compatibility tests**

Create `tests/test_tui_package.py`:

```python
from __future__ import annotations


def test_picker_compatibility_exports_are_identical() -> None:
    from memo.setup.picker import PickerResult as old_result
    from memo.setup.picker import run_picker as old_picker
    from memo.tui.picker import PickerResult as new_result
    from memo.tui.picker import run_picker as new_picker

    assert old_result is new_result
    assert old_picker is new_picker


def test_dashboard_compatibility_exports_are_identical() -> None:
    from memo.dashboard_tui import render as old_render
    from memo.dashboard_tui import run_tui as old_run
    from memo.tui.dashboard import render as new_render
    from memo.tui.dashboard import run_tui as new_run

    assert old_render is new_render
    assert old_run is new_run


def test_top_level_tui_modules_are_compatibility_shims() -> None:
    from pathlib import Path
    import memo.dashboard_panels
    import memo.dashboard_tui
    import memo.setup.picker

    for module in (memo.dashboard_panels, memo.dashboard_tui, memo.setup.picker):
        body = Path(module.__file__).read_text(encoding="utf-8")
        assert "from memo.tui" in body
        assert "def run_tui" not in body
        assert "def run_picker" not in body
```

- [ ] **Step 2: Run the new tests and verify the package is missing**

Run:

```bash
uv run --no-sync pytest tests/test_tui_package.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'memo.tui'`.

- [ ] **Step 3: Move implementations and add compatibility shims**

Move the current file bodies without behavioral edits:

```bash
mkdir -p src/memo/tui/dashboard
git mv src/memo/dashboard_tui.py src/memo/tui/dashboard/app.py
git mv src/memo/dashboard_panels.py src/memo/tui/dashboard/panels.py
git mv src/memo/setup/picker.py src/memo/tui/picker.py
```

Update imports in the moved files from `memo.dashboard_panels` to
`memo.tui.dashboard.panels`. Create `src/memo/tui/dashboard/__init__.py`:

```python
"""Terminal dashboard implementation."""

from memo.tui.dashboard.app import render, run_tui

__all__ = ["render", "run_tui"]
```

Create `src/memo/dashboard_tui.py`:

```python
"""Compatibility exports for the terminal dashboard."""

from memo.tui.dashboard.app import render, run_tui

__all__ = ["render", "run_tui"]
```

Create `src/memo/setup/picker.py`:

```python
"""Compatibility exports for the first-run terminal picker."""

from memo.tui.picker import DEFAULT_VAULT_SUBDIR, PickerResult, run_picker

__all__ = ["DEFAULT_VAULT_SUBDIR", "PickerResult", "run_picker"]
```

Create `src/memo/dashboard_panels.py` as an explicit re-export shim for every
symbol currently imported from it by `memo.dashboard` and tests. Change
`src/memo/dashboard.py` and `src/memo/setup/__init__.py` to import new package
paths directly.

- [ ] **Step 4: Run focused dashboard, picker, and package tests**

Run:

```bash
uv run --no-sync pytest tests/test_tui_package.py tests/test_cli_init.py tests/test_tui_recall_quality.py tests/test_daemon_lever_parity.py -v
```

Expected: PASS.

- [ ] **Step 5: Run targeted static checks**

Run:

```bash
uv run --no-sync ruff check src/memo/tui src/memo/dashboard.py src/memo/dashboard_tui.py src/memo/dashboard_panels.py src/memo/setup tests/test_tui_package.py
uv run --no-sync mypy src/memo/tui src/memo/setup
```

Expected: PASS.

- [ ] **Step 6: Commit the package consolidation**

```bash
git add src/memo/tui src/memo/dashboard.py src/memo/dashboard_tui.py src/memo/dashboard_panels.py src/memo/setup tests/test_tui_package.py
git commit -m "refactor(tui): consolidate terminal UI package"
```

---

### Task 2: Add Textual And The Complete Typed Setting Catalog

**Files:**
- Modify: `pyproject.toml`
- Create: `src/memo/tui/config/__init__.py`
- Create: `src/memo/tui/config/catalog.py`
- Create: `tests/test_config_catalog.py`

**Interfaces:**
- Produces enums `SettingKind`, `Visibility`, `PersistencePolicy`, `RiskLevel`.
- Produces dataclasses `SettingChoice`, `FieldBinding`, `SettingSpec`.
- Produces `build_catalog() -> tuple[SettingSpec, ...]`.
- Produces `catalog_by_key() -> dict[str, SettingSpec]`.
- Produces `path_to_env() -> dict[str, str]`, `path_to_field() -> dict[str, str]`, and `domain_file_for_key(key: str) -> str`.
- Consumes `memo.flags.REGISTRY`, `memo.config.Config.model_fields`, and existing user-facing config paths.

- [ ] **Step 1: Write catalog coverage tests**

Create `tests/test_config_catalog.py`:

```python
from __future__ import annotations

from memo.config import Config
from memo.flags import REGISTRY
from memo.tui.config.catalog import (
    PersistencePolicy,
    Visibility,
    build_catalog,
    catalog_by_key,
    path_to_env,
    path_to_field,
)


def test_catalog_covers_every_config_field_and_flag_once() -> None:
    catalog = build_catalog()
    assert len({spec.key for spec in catalog}) == len(catalog)
    assert {spec.config_field for spec in catalog if spec.config_field} == set(Config.model_fields)
    assert {spec.env_name for spec in catalog if spec.env_name} >= set(REGISTRY)


def test_every_setting_has_explicit_policy_and_visibility() -> None:
    for spec in build_catalog():
        assert isinstance(spec.policy, PersistencePolicy)
        assert isinstance(spec.visibility, Visibility)
        assert spec.label
        assert spec.description


def test_runtime_controls_are_not_persistent() -> None:
    by_key = catalog_by_key()
    assert by_key["misc.noninteractive"].policy is PersistencePolicy.RUNTIME_ONLY
    assert by_key["session.agent_tty"].policy is PersistencePolicy.RUNTIME_ONLY


def test_catalog_owns_markdown_mappings() -> None:
    assert path_to_field()["storage.data_dir"] == "data_dir"
    assert path_to_env()["recall.top_k"] == "MEMO_RECALL_TOP_K"
```

- [ ] **Step 2: Run the tests and verify failure**

Run:

```bash
uv run --no-sync pytest tests/test_config_catalog.py -v
```

Expected: FAIL because `memo.tui.config.catalog` does not exist.

- [ ] **Step 3: Add dependencies**

Add to core dependencies in `pyproject.toml`:

```toml
"textual>=8.2.8,<9",
```

Add to dev dependencies:

```toml
"pytest-asyncio>=1.4,<2",
"pytest-textual-snapshot>=1.1,<2",
```

Run `uv pip install -e '.[dev]'` after editing so the shared runtime contains
the new dependency.

- [ ] **Step 4: Implement catalog types and explicit bindings**

Create `src/memo/tui/config/catalog.py` with these exact public types:

```python
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class SettingKind(StrEnum):
    BOOL = "bool"
    INT = "int"
    FLOAT = "float"
    STR = "str"
    PATH = "path"
    SECRET = "secret"
    ENUM = "enum"


class Visibility(StrEnum):
    COMMON = "common"
    ADVANCED = "advanced"
    EXPERIMENTAL = "experimental"


class PersistencePolicy(StrEnum):
    PERSISTENT = "persistent"
    RUNTIME_ONLY = "runtime-only"
    SECRET = "secret"
    DERIVED = "derived"


class RiskLevel(StrEnum):
    NORMAL = "normal"
    CAUTION = "caution"
    DESTRUCTIVE = "destructive"


@dataclass(frozen=True)
class SettingChoice:
    label: str
    value: str
    description: str = ""


@dataclass(frozen=True)
class FieldBinding:
    key: str
    field: str
    env_name: str
    domain: str
    kind: SettingKind


@dataclass(frozen=True)
class SettingSpec:
    key: str
    label: str
    description: str
    domain: str
    section: str
    kind: SettingKind
    default: Any
    env_name: str | None = None
    config_field: str | None = None
    minimum: float | None = None
    maximum: float | None = None
    choices: tuple[SettingChoice, ...] = ()
    visibility: Visibility = Visibility.ADVANCED
    policy: PersistencePolicy = PersistencePolicy.PERSISTENT
    risk: RiskLevel = RiskLevel.NORMAL
    platforms: frozenset[str] = frozenset()
    requires: tuple[str, ...] = ()
    conflicts: tuple[str, ...] = ()
    restart_targets: tuple[str, ...] = ()
    sensitive: bool = False
```

Declare all 20 `Config.model_fields` in an explicit `FIELD_BINDINGS` tuple.
Use the paths currently in `_FIELD_PATHS`, including `storage.*`, `models.*`,
and `search.*`. Mark `memory_subdir` advanced/deprecated. Give
`models.model_profile` choices `light`, `balanced`, and `quality`; give
`models.embedder_backend` choices `auto`, `mlx`, and `st`.

Map every current registry group with this complete table:

```python
GROUP_TO_DOMAIN = {
    "behavior": "Advanced", "bench": "Advanced", "briefing": "Recall",
    "cache": "Maintenance", "capture": "Capture", "cli": "Advanced",
    "dream": "Maintenance", "embedder": "Models", "entity": "Graph",
    "feedback": "Recall", "graph": "Graph", "ingest": "Capture",
    "links": "Graph", "maintain": "Maintenance", "mcp": "Hooks",
    "misc": "Advanced", "outcome": "Recall", "privacy": "Capture",
    "recall": "Recall", "repo": "Search", "retrieval": "Search",
    "roi": "Recall", "search": "Search", "secret": "Advanced",
    "session": "Recall", "store": "Storage", "synapse": "Hooks",
    "sync": "Hooks", "temporal": "Search", "update": "Maintenance",
    "whatsapp": "Capture",
}
```

Use explicit sets for `COMMON_ENV_NAMES`, `EXPERIMENTAL_ENV_NAMES`,
`RUNTIME_ONLY_ENV_NAMES = {"MEMO_NONINTERACTIVE", "MEMO_AGENT_TTY"}`, and
`SENSITIVE_ENV_NAMES`. All other registered flags receive persistent/advanced
classification in the resulting `SettingSpec`; no flag is omitted. Derive a
friendly fallback label from the stable key, while explicit overrides supply
labels, choices, risks, platforms, dependencies, conflicts, and restart targets
for common settings.

Preserve the current stable Markdown path algorithm from `config_md`: the table
prefix remains the registry group (`misc.noninteractive`,
`session.agent_tty`, `update.hook_selfheal`), while `SettingSpec.domain` is the
friendlier navigation domain. Map storage/models/search field bindings to their
existing files; map recall/search/capture/graph/hooks registry groups to their
matching domain files; map every other registry group to
`advanced-config.md`. This expands writability without renaming existing keys.

- [ ] **Step 5: Run catalog tests**

Run:

```bash
uv run --no-sync pytest tests/test_config_catalog.py -v
```

Expected: PASS and report catalog coverage for all current fields and flags.

- [ ] **Step 6: Run static checks and commit**

```bash
uv run --no-sync ruff check src/memo/tui/config/catalog.py tests/test_config_catalog.py
uv run --no-sync mypy src/memo/tui/config/catalog.py
git add pyproject.toml src/memo/tui/config tests/test_config_catalog.py
git commit -m "feat(config): add typed TUI setting catalog"
```

Expected: checks PASS and commit succeeds.

---

### Task 3: Make `config_md` Consume Catalog Bindings And Enforce Policies

**Files:**
- Modify: `src/memo/config_md.py`
- Modify: `tests/test_config_md.py`
- Modify: `tests/test_cli_config.py`

**Interfaces:**
- Consumes `path_to_env()`, `path_to_field()`, `domain_file_for_key()`, and `catalog_by_key()` from Task 2.
- Preserves all existing public `config_md` functions.
- Produces `configured_values(env) -> dict[str, ConfigValue]` as the canonical source-aware read surface.
- Rejects runtime-only, secret, and derived keys in Markdown validation and write paths.

- [ ] **Step 1: Add failing policy and mapping tests**

Append to `tests/test_config_md.py`:

```python
def test_runtime_only_key_is_rejected_in_markdown(tmp_path: Path) -> None:
    home = tmp_path / "memo-home"
    cfg = home / "config"
    cfg.mkdir(parents=True)
    (cfg / "advanced-config.md").write_text(
        "```toml\n[misc]\nnoninteractive = \"on\"\n```\n",
        encoding="utf-8",
    )
    problems = config_md.validate_markdown_config({"MEMO_CONFIG_DIR": str(home)})
    assert any("runtime-only" in problem.error for problem in problems)


def test_set_value_refuses_runtime_only_key(tmp_path: Path) -> None:
    env = {"MEMO_CONFIG_DIR": str(tmp_path / "memo-home")}
    with pytest.raises(ValueError, match="runtime-only"):
        config_md.set_value("misc.noninteractive", "on", env)


def test_configured_values_keeps_source_file(tmp_path: Path) -> None:
    home = tmp_path / "memo-home"
    cfg = home / "config"
    cfg.mkdir(parents=True)
    path = cfg / "recall-config.md"
    path.write_text("```toml\n[recall]\ntop_k = 7\n```\n", encoding="utf-8")
    value = config_md.configured_values({"MEMO_CONFIG_DIR": str(home)})["recall.top_k"]
    assert value.value == 7
    assert value.file == str(path)
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
uv run --no-sync pytest tests/test_config_md.py -v
```

Expected: FAIL because policy validation and `configured_values` do not exist.

- [ ] **Step 3: Replace private mapping ownership**

Remove `_FIELD_PATHS`, `_flag_path_map()`, and `_DOMAIN_FOR_PREFIX` from
`config_md.py`. Import catalog functions lazily inside parser/write functions
to preserve the existing `Config -> config_md -> flags` import safety.

Implement:

```python
def configured_values(env: Mapping[str, str] | None = None) -> dict[str, ConfigValue]:
    return dict(load_values(env))
```

Before accepting or writing a known key, resolve its `SettingSpec`. Raise or
return `ConfigProblem` with one of these exact messages:

```text
runtime-only setting cannot be persisted
derived setting cannot be persisted
secret setting must use encrypted secret storage
```

Keep environment, Markdown, and tuned-overlay resolution semantics unchanged.

- [ ] **Step 4: Run config parser and CLI tests**

```bash
uv run --no-sync pytest tests/test_config_md.py tests/test_cli_config.py tests/test_flags.py tests/test_config.py -v
```

Expected: PASS.

- [ ] **Step 5: Run static checks and commit**

```bash
uv run --no-sync ruff check src/memo/config_md.py src/memo/tui/config/catalog.py tests/test_config_md.py tests/test_cli_config.py
uv run --no-sync mypy src/memo/config_md.py src/memo/tui/config/catalog.py
git add src/memo/config_md.py src/memo/tui/config/catalog.py tests/test_config_md.py tests/test_cli_config.py
git commit -m "refactor(config): centralize Markdown bindings in TUI catalog"
```

---

### Task 4: Add Source-Aware Sessions, Drafts, And Validation

**Files:**
- Create: `src/memo/tui/config/session.py`
- Create: `tests/test_config_session.py`

**Interfaces:**
- Produces `ValueSource`, `ValidationIssue`, `SettingState`, `DraftOperation`, `ConfigDraft`, and `ConfigSession`.
- Produces `ConfigSession.open(env: Mapping[str, str] | None = None) -> ConfigSession`.
- Produces `set_value(key: str, raw: object)`, `unset_value(key: str)`, `discard()`, `states()`, `issues()`, and `review()`.
- `review()` produces the immutable `ApplyPlan` defined in `tui/config/apply.py` in Task 5; until Task 5, define the shared `PlannedChange` and `ApplyPlan` dataclasses in `session.py`, then move them without API change.

- [ ] **Step 1: Write draft/source tests**

Create `tests/test_config_session.py`:

```python
from pathlib import Path

from memo.tui.config.session import ConfigSession, ValueSource


def test_session_distinguishes_markdown_and_effective_env(tmp_path: Path) -> None:
    home = tmp_path / "memo-home"
    cfg = home / "config"
    cfg.mkdir(parents=True)
    (cfg / "recall-config.md").write_text(
        "```toml\n[recall]\ntop_k = 7\n```\n", encoding="utf-8"
    )
    env = {
        "MEMO_CONFIG_DIR": str(home),
        "MEMO_RECALL_TOP_K": "2",
        "MEMO_DATA_DIR": str(tmp_path / "data"),
        "MEMO_STATE_DIR": str(tmp_path / "state"),
    }
    state = ConfigSession.open(env).state("recall.top_k")
    assert state.configured_value == 7
    assert state.effective_value == 2
    assert state.source is ValueSource.ENV
    assert state.env_override == "2"


def test_draft_set_and_unset_never_write(tmp_path: Path) -> None:
    home = tmp_path / "memo-home"
    env = {"MEMO_CONFIG_DIR": str(home), "MEMO_DATA_DIR": str(tmp_path / "data")}
    session = ConfigSession.open(env)
    session.set_value("recall.top_k", 9)
    assert session.state("recall.top_k").pending_value == 9
    assert not home.exists()
    session.unset_value("recall.top_k")
    assert session.state("recall.top_k").pending_unset is True
    assert not home.exists()


def test_invalid_numeric_value_blocks_review(tmp_path: Path) -> None:
    session = ConfigSession.open({"MEMO_CONFIG_DIR": str(tmp_path / "memo-home")})
    session.set_value("recall.top_k", -1)
    assert any(issue.blocking and issue.key == "recall.top_k" for issue in session.issues())
```

- [ ] **Step 2: Run tests and verify failure**

```bash
uv run --no-sync pytest tests/test_config_session.py -v
```

Expected: FAIL because `memo.tui.config.session` does not exist.

- [ ] **Step 3: Implement immutable state and draft models**

Use these public shapes in `session.py`:

```python
class ValueSource(StrEnum):
    ENV = "env"
    MARKDOWN = "markdown"
    OVERLAY = "overlay"
    LEGACY = "legacy"
    DEFAULT = "default"
    DERIVED = "derived"


@dataclass(frozen=True)
class ValidationIssue:
    key: str
    message: str
    blocking: bool = True


@dataclass(frozen=True)
class SettingState:
    spec: SettingSpec
    configured_value: object | None
    effective_value: object | None
    source: ValueSource
    default_value: object | None
    env_override: str | None = None
    pending_value: object | None = None
    pending_unset: bool = False
    available: bool = True
    availability_reason: str = ""
    issues: tuple[ValidationIssue, ...] = ()
```

`ConfigDraft` stores a `dict[str, DraftOperation]`. Coerce values through the
same boolean/numeric bounds used by `FlagSpec` and Pydantic fields. Add explicit
cross-setting validators for `memories_in_vault -> vault_path`,
`embedder_model <-> embedder_dims`, and platform-gated reranking. Runtime-only,
derived, and secret policies reject normal `set_value`.

- [ ] **Step 4: Run session tests and static checks**

```bash
uv run --no-sync pytest tests/test_config_session.py -v
uv run --no-sync ruff check src/memo/tui/config/session.py tests/test_config_session.py
uv run --no-sync mypy src/memo/tui/config/session.py
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/memo/tui/config/session.py tests/test_config_session.py
git commit -m "feat(config): add source-aware TUI draft session"
```

---

### Task 5: Add Preserving Markdown Rendering And Transaction Recovery

**Files:**
- Create: `src/memo/tui/config/apply.py`
- Modify: `src/memo/config_md.py`
- Create: `tests/test_config_apply.py`
- Modify: `tests/test_config_md.py`
- Modify: `src/memo/errors.py`

**Interfaces:**
- Produces `FileFingerprint`, `SourceSnapshot`, `PlannedChange`, `ApplyPlan`, `TransactionReceipt`, and `ConfigTransaction`.
- Produces `snapshot_sources(env) -> SourceSnapshot`.
- Produces `render_draft(plan, env) -> dict[Path, str]`.
- Produces `ConfigTransaction.commit(rendered, snapshot) -> TransactionReceipt`.
- Produces `recover_interrupted_transaction(config_home: Path) -> TransactionReceipt | None`.
- Consumes `ConfigDraft`, catalog bindings, and production Markdown validation.

- [ ] **Step 1: Write preservation, rollback, and conflict tests**

Create `tests/test_config_apply.py` with these cases:

```python
def test_render_preserves_markdown_outside_toml(tmp_path: Path) -> None:
    path = _write_recall_config(tmp_path, top_k=3, intro="Keep this prose exactly.\n")
    plan = _plan_set(tmp_path, "recall.top_k", 5)
    rendered = render_draft(plan, _env(tmp_path))[path]
    assert rendered.startswith("Keep this prose exactly.\n")
    assert "top_k = 5" in rendered


def test_commit_rolls_back_every_file_on_second_replace_failure(tmp_path: Path, monkeypatch) -> None:
    rendered, snapshot = _two_file_render(tmp_path)
    original = {path: path.read_text(encoding="utf-8") for path in rendered}
    real_replace = os.replace
    calls = 0

    def fail_second(src, dst):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected replace failure")
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", fail_second)
    with pytest.raises(ConfigTransactionError, match="injected replace failure"):
        ConfigTransaction(_env(tmp_path)).commit(rendered, snapshot)
    assert {path: path.read_text(encoding="utf-8") for path in rendered} == original


def test_same_key_external_edit_raises_conflict(tmp_path: Path) -> None:
    plan = _plan_set(tmp_path, "recall.top_k", 5)
    path = next(iter(plan.snapshot.files))
    path.write_text(path.read_text().replace("top_k = 3", "top_k = 4"), encoding="utf-8")
    with pytest.raises(ConfigConflictError) as exc:
        render_draft(plan, _env(tmp_path))
    assert exc.value.keys == ("recall.top_k",)
```

The helper functions in the test create isolated Markdown files and an
`ApplyPlan`; they must not patch the user's home directory.

- [ ] **Step 2: Run tests and verify failure**

```bash
uv run --no-sync pytest tests/test_config_apply.py -v
```

Expected: FAIL because apply/transaction types do not exist.

- [ ] **Step 3: Add domain errors**

Add to `src/memo/errors.py`:

```python
class ConfigConflictError(MemoError, RuntimeError):
    def __init__(self, keys: tuple[str, ...]) -> None:
        self.keys = keys
        super().__init__(f"configuration changed externally: {', '.join(keys)}")


class ConfigTransactionError(StorageError):
    """A staged configuration batch could not commit or roll back cleanly."""


class ConfigActivationError(MemoError, RuntimeError):
    """Configuration saved but one or more confirmed activation actions failed."""
```

- [ ] **Step 4: Implement preserving render and transaction protocol**

Use SHA-256 plus mtime/size fingerprints. `render_draft` performs a key-level
three-way comparison between the session baseline, current disk, and draft.
Independent external edits are retained; same-key edits raise
`ConfigConflictError`.

For each affected file:

1. Render only its affected fenced TOML table.
2. Write a sibling `.<name>.memo-tmp-<transaction-id>` file.
3. Parse and validate all temporary content before replacement.
4. Copy originals under `<config_home>/.transactions/<id>/backup/`.
5. Write `manifest.json` with `prepared`, then `committing`, then `complete`.
6. Replace each file with `os.replace`.
7. On failure, restore every backup and keep the manifest as `rolled_back`.

Use `tomli_w.dumps` for canonical affected tables. Never rewrite prose outside
the matched TOML fence.

- [ ] **Step 5: Run apply/config tests and static checks**

```bash
uv run --no-sync pytest tests/test_config_apply.py tests/test_config_md.py -v
uv run --no-sync ruff check src/memo/tui/config/apply.py src/memo/config_md.py src/memo/errors.py tests/test_config_apply.py tests/test_config_md.py
uv run --no-sync mypy src/memo/tui/config/apply.py src/memo/config_md.py
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/memo/tui/config/apply.py src/memo/config_md.py src/memo/errors.py tests/test_config_apply.py tests/test_config_md.py
git commit -m "feat(config): add transactional Markdown apply"
```

---

### Task 6: Add Impact Planning And Confirmed Activation

**Files:**
- Create: `src/memo/tui/config/impact.py`
- Modify: `src/memo/tui/config/catalog.py`
- Create: `tests/test_config_impact.py`

**Interfaces:**
- Produces `ImpactTarget`, `ImpactAction`, `ImpactResult`, and `ImpactController`.
- Produces `plan_impacts(changes: tuple[PlannedChange, ...]) -> tuple[ImpactAction, ...]`.
- Produces `ImpactController.execute(actions) -> tuple[ImpactResult, ...]`.
- Default execution uses argv lists and `subprocess.run(..., shell=False)`; tests inject an executor.

- [ ] **Step 1: Write impact tests**

Create `tests/test_config_impact.py`:

```python
def test_model_change_requires_recall_daemon_restart() -> None:
    actions = plan_impacts((_change("models.model_profile", "light", "balanced"),))
    assert [action.target for action in actions] == [ImpactTarget.RECALL_DAEMON]


def test_hook_change_reports_hook_rewire() -> None:
    actions = plan_impacts((_change("update.hook_selfheal", False, True),))
    assert ImpactTarget.HOOKS in {action.target for action in actions}


def test_execute_uses_injected_executor_and_reports_partial_failure() -> None:
    seen: list[tuple[str, ...]] = []

    def executor(argv: tuple[str, ...]) -> tuple[int, str]:
        seen.append(argv)
        return (1, "failed")

    result = ImpactController(executor=executor).execute(
        (ImpactAction(ImpactTarget.RECALL_DAEMON, "Restart recall daemon", ("memo", "recall-daemon", "restart")),)
    )
    assert seen == [("memo", "recall-daemon", "restart")]
    assert result[0].success is False
```

- [ ] **Step 2: Run tests and verify failure**

```bash
uv run --no-sync pytest tests/test_config_impact.py -v
```

Expected: FAIL because `impact.py` does not exist.

- [ ] **Step 3: Implement explicit targets and commands**

Define:

```python
class ImpactTarget(StrEnum):
    RECALL_DAEMON = "recall-daemon"
    WATCHER = "watcher"
    HOOKS = "hooks"
    REINDEX = "reindex"
```

Catalog metadata maps model/embedder changes to recall-daemon plus reindex
warning where dimensions/model change, watcher-owned storage/capture paths to
watcher, and hook settings to hook rewire. Deduplicate actions in stable order.
Execution is never called while building or displaying a plan.

- [ ] **Step 4: Run tests, static checks, and commit**

```bash
uv run --no-sync pytest tests/test_config_impact.py tests/test_config_catalog.py -v
uv run --no-sync ruff check src/memo/tui/config/impact.py src/memo/tui/config/catalog.py tests/test_config_impact.py
uv run --no-sync mypy src/memo/tui/config/impact.py
git add src/memo/tui/config/impact.py src/memo/tui/config/catalog.py tests/test_config_impact.py
git commit -m "feat(config): plan explicit runtime activation impacts"
```

---

### Task 7: Build Typed Textual Controls And The Domain Configuration Center

**Files:**
- Create: `src/memo/tui/common.py`
- Create: `src/memo/tui/config/controls.py`
- Create: `src/memo/tui/config/widgets.py`
- Create: `src/memo/tui/config/app.py`
- Create: `src/memo/tui/config/styles.tcss`
- Modify: `src/memo/tui/config/__init__.py`
- Create: `tests/test_config_tui_controls.py`
- Create: `tests/test_config_tui_app.py`

**Interfaces:**
- Produces `control_for(state: SettingState) -> Widget`.
- Produces `DomainNav`, `SourceBadge`, `SettingRow`, `ValidationSummary`.
- Produces `ConfigApp(session: ConfigSession)` and `run_config_tui(env=None) -> int`.
- Consumes session intents only; UI modules do not import `config_md`, `Path.write_text`, `os.replace`, or lifecycle commands.

- [ ] **Step 1: Write control and main-screen tests**

Create tests using `App.run_test()` and `Pilot`:

```python
@pytest.mark.asyncio
async def test_boolean_setting_uses_switch(tmp_path: Path) -> None:
    app = ConfigApp(_session(tmp_path))
    async with app.run_test(size=(120, 36)):
        row = app.query_one("#setting-recall-disable", SettingRow)
        assert row.query_one(Switch).value is False


@pytest.mark.asyncio
async def test_search_finds_folded_advanced_setting(tmp_path: Path) -> None:
    app = ConfigApp(_session(tmp_path))
    async with app.run_test(size=(120, 36)) as pilot:
        search = app.query_one("#setting-search", Input)
        search.focus()
        await pilot.press(*"intra dedup threshold")
        await pilot.pause()
        assert app.query("SettingRow").first().setting_key == "recall.intra_dedup_threshold"


@pytest.mark.asyncio
async def test_env_override_badge_keeps_markdown_editable(tmp_path: Path) -> None:
    app = ConfigApp(_session(tmp_path, MEMO_RECALL_TOP_K="2"))
    async with app.run_test(size=(120, 36)):
        row = app.query_one("#setting-recall-top-k", SettingRow)
        assert row.query_one(SourceBadge).source == ValueSource.ENV
        assert row.query_one(Input).disabled is False


@pytest.mark.parametrize("terminal_size", [(80, 24), (100, 30), (140, 45)])
def test_config_center_snapshots(snap_compare, tmp_path: Path, terminal_size) -> None:
    assert snap_compare(ConfigApp(_session(tmp_path)), terminal_size=terminal_size)
```

- [ ] **Step 2: Run tests and verify failure**

```bash
uv run --no-sync pytest tests/test_config_tui_controls.py tests/test_config_tui_app.py -v
```

Expected: FAIL because controls/app do not exist.

- [ ] **Step 3: Implement reusable controls and widgets**

Map `SettingKind` to Textual `Switch`, `Select`, and validated `Input`
subclasses. Each control posts one `SettingChanged(key, value)` message.
`SettingRow` has stable height constraints for label, description, badges,
control, and validation text so state changes do not shift unrelated rows.

Use semantic colors in `common.py` and `styles.tcss`: green success, yellow
warning, red error, cyan information, neutral foreground/background. Avoid
emoji and web-specific constructs.

- [ ] **Step 4: Implement `ConfigApp` main center**

`ConfigApp.compose()` yields a compact header, `DomainNav`, global search,
scrollable setting list, pending summary, and `Discard`/`Review` actions.
Use the approved domains in catalog order. At widths below 100, hide the
sidebar and show a `Select` domain control. At 80x24, use a single-column list.
Below 80x24, show a size requirement screen without starting background work.

`run_config_tui` creates `ConfigSession.open(env)`, runs `ConfigApp`, and returns
the app exit code. It must not import Textual Web packages or open sockets.

- [ ] **Step 5: Run UI tests and snapshots**

```bash
uv run --no-sync pytest tests/test_config_tui_controls.py tests/test_config_tui_app.py -v
uv run --no-sync pytest tests/test_config_tui_app.py --snapshot-update
```

Expected: functional tests PASS and approved snapshots exist for 80x24,
100x30, and 140x45.

- [ ] **Step 6: Run static checks and commit**

```bash
uv run --no-sync ruff check src/memo/tui tests/test_config_tui_controls.py tests/test_config_tui_app.py
uv run --no-sync mypy src/memo/tui/config
git add src/memo/tui tests/test_config_tui_controls.py tests/test_config_tui_app.py
git commit -m "feat(config): add Textual domain configuration center"
```

---

### Task 8: Add Wizard, Review, Conflict, Recovery, And Result Screens

**Files:**
- Create: `src/memo/tui/config/screens.py`
- Modify: `src/memo/tui/config/app.py`
- Modify: `src/memo/tui/config/styles.tcss`
- Create: `tests/test_config_tui_screens.py`

**Interfaces:**
- Produces `FirstRunWizard`, `ReviewScreen`, `ConflictScreen`, `RecoveryScreen`, and `ApplyResultScreen`.
- Consumes `ConfigSession`, `ApplyPlan`, `ConfigTransaction`, and `ImpactController`.
- Uses Textual workers for service health and activation commands; file commit itself runs through the synchronous domain service and returns a receipt.

- [ ] **Step 1: Write screen-flow tests**

Cover these exact flows:

```python
@pytest.mark.asyncio
async def test_missing_config_opens_four_step_wizard(tmp_path: Path) -> None:
    app = ConfigApp(_session(tmp_path, no_markdown=True))
    async with app.run_test(size=(120, 36)):
        wizard = app.screen
        assert isinstance(wizard, FirstRunWizard)
        assert wizard.step_count == 4


@pytest.mark.asyncio
async def test_review_blocks_invalid_draft(tmp_path: Path) -> None:
    app = ConfigApp(_session(tmp_path))
    async with app.run_test(size=(120, 36)):
        app.session.set_value("recall.top_k", -1)
        app.action_review()
        assert not isinstance(app.screen, ReviewScreen)
        assert app.query_one(ValidationSummary).blocking_count == 1


@pytest.mark.asyncio
async def test_apply_requires_restart_confirmation(tmp_path: Path) -> None:
    controller = FakeImpactController()
    app = ConfigApp(_session_with_model_change(tmp_path), impact_controller=controller)
    async with app.run_test(size=(120, 36)) as pilot:
        app.action_review()
        await pilot.click("#apply-save-only")
        assert controller.executed == []
```

Also test recovery actions, same-key conflict choice, cancel preserving no files,
masked secrets absent from rendered text, partial activation failure state, and
a legacy-only install offering migration through the existing
`memo config migrate` semantics before the four-step wizard.

- [ ] **Step 2: Run tests and verify failure**

```bash
uv run --no-sync pytest tests/test_config_tui_screens.py -v
```

Expected: FAIL because screen classes do not exist.

- [ ] **Step 3: Implement the four-step wizard**

Steps are exactly: storage/vault, hardware/model profile, integrations/hooks/
recall, privacy/capture plus summary. Use `platform_detect` without importing
MLX. Completion creates a draft and calls the same review/apply path; cancel
creates no files.

- [ ] **Step 4: Implement review and apply screens**

Review displays configured/effective old/new values, shadowed ENV warnings,
affected files, and impact actions. Provide `Save only` and
`Apply and restart affected services`; no activation action runs by default.

- [ ] **Step 5: Implement recovery and conflict screens**

Recovery displays exact file/block/key errors and offers `$EDITOR`, latest
backup, read-only, or exit. Conflict displays baseline/disk/draft values per
same-key conflict and requires a choice. `$EDITOR` is invoked as an argv list
using `shlex.split(os.environ["EDITOR"]) + [path]`, never through a shell.

- [ ] **Step 6: Run screen and app tests, static checks, and commit**

```bash
uv run --no-sync pytest tests/test_config_tui_screens.py tests/test_config_tui_app.py -v
uv run --no-sync ruff check src/memo/tui/config tests/test_config_tui_screens.py tests/test_config_tui_app.py
uv run --no-sync mypy src/memo/tui/config
git add src/memo/tui/config tests/test_config_tui_screens.py tests/test_config_tui_app.py
git commit -m "feat(config): add safe TUI setup and apply flows"
```

---

### Task 9: Wire `memo config` With TTY And Headless Compatibility

**Files:**
- Modify: `src/memo/cli_config.py`
- Modify: `src/memo/cli.py`
- Modify: `tests/test_cli_config.py`
- Modify: `tests/test_cli_init.py`

**Interfaces:**
- Consumes `memo.tui.config.run_config_tui` via lazy import.
- Preserves every existing config subcommand.
- Root first-run gate skips `config`; the config app owns its wizard.

- [ ] **Step 1: Write TTY/headless tests**

Add to `tests/test_cli_config.py`:

```python
def test_bare_config_launches_tui_on_tty(tmp_path: Path) -> None:
    runner = CliRunner()
    env = {**_env(tmp_path), "MEMO_NONINTERACTIVE": ""}
    with (
        patch("sys.stdin.isatty", return_value=True),
        patch("sys.stdout.isatty", return_value=True),
        patch("memo.tui.config.run_config_tui", return_value=0) as run,
    ):
        result = runner.invoke(cli, ["config"], env=env)
    assert result.exit_code == 0
    run.assert_called_once()


def test_bare_config_prints_help_without_tty(tmp_path: Path) -> None:
    runner = CliRunner()
    with patch("memo.tui.config.run_config_tui") as run:
        result = runner.invoke(cli, ["config"], env=_env(tmp_path))
    assert result.exit_code == 0
    assert "show" in result.output
    run.assert_not_called()


def test_noninteractive_blocks_tui_even_with_tty(tmp_path: Path) -> None:
    runner = CliRunner()
    with (
        patch("sys.stdin.isatty", return_value=True),
        patch("sys.stdout.isatty", return_value=True),
        patch("memo.tui.config.run_config_tui") as run,
    ):
        result = runner.invoke(cli, ["config"], env=_env(tmp_path))
    assert result.exit_code == 0
    run.assert_not_called()
```

- [ ] **Step 2: Run tests and verify failure**

```bash
uv run --no-sync pytest tests/test_cli_config.py tests/test_cli_init.py -v
```

Expected: bare `config` does not yet invoke Textual.

- [ ] **Step 3: Add TTY-aware config callback**

Change the group to `invoke_without_command=True`, pass context, and implement:

```python
@click.group(name="config", invoke_without_command=True)
@click.pass_context
def config_group(ctx: click.Context) -> None:
    """Inspect, validate, and edit memo configuration."""
    if ctx.invoked_subcommand is not None:
        return
    from memo.flags import flag_bool

    if flag_bool("MEMO_NONINTERACTIVE") or not (sys.stdin.isatty() and sys.stdout.isatty()):
        click.echo(ctx.get_help())
        return
    from memo.tui.config import run_config_tui

    raise click.exceptions.Exit(run_config_tui())
```

Add `config` to `_FIRST_RUN_GATE_SKIP_COMMANDS` in `cli.py`. Keep Textual imports
out of module scope.

- [ ] **Step 4: Run CLI compatibility and import-smoke tests**

```bash
uv run --no-sync pytest tests/test_cli_config.py tests/test_cli_init.py tests/test_flags.py tests/test_config.py -v
uv run --no-sync python -X importtime -c "import memo.cli" 2> /tmp/memo-cli-importtime.txt
rg "mlx|mlx_lm|textual" /tmp/memo-cli-importtime.txt
```

Expected: pytest PASS; import trace contains no `mlx`/`mlx_lm` and does not
import Textual for a plain module import.

- [ ] **Step 5: Run static checks and commit**

```bash
uv run --no-sync ruff check src/memo/cli_config.py src/memo/cli.py tests/test_cli_config.py tests/test_cli_init.py
uv run --no-sync mypy src/memo/cli_config.py src/memo/cli.py
git add src/memo/cli_config.py src/memo/cli.py tests/test_cli_config.py tests/test_cli_init.py
git commit -m "feat(config): open terminal TUI from memo config"
```

---

### Task 10: Documentation, Full Verification, And Final Regression Gate

**Files:**
- Modify: `docs/reference.md`
- Modify: `README.md`
- Modify: `docs/install-new-mac.md`
- Modify: `CHANGELOG.md`
- Modify: tests/snapshots created in Tasks 7-8

**Interfaces:**
- Documents terminal-only behavior, key workflows, headless subcommands, source precedence, and recovery.
- Does not advertise browser or Textual Web operation.

- [ ] **Step 1: Update user documentation**

Document:

```text
memo config                 # terminal configuration center (TTY only)
memo config show --effective
memo config validate
memo config set recall.top_k 5
memo config unset recall.top_k
```

State that Markdown remains editable, `MEMO_*` overrides are temporary, the
first-run wizard has four steps, and headless automation keeps using subcommands.
Include recovery and backup locations without exposing secret values.

- [ ] **Step 2: Run focused feature suites**

```bash
uv run --no-sync pytest tests/test_tui_package.py tests/test_config_catalog.py tests/test_config_session.py tests/test_config_apply.py tests/test_config_impact.py tests/test_config_tui_controls.py tests/test_config_tui_app.py tests/test_config_tui_screens.py tests/test_config_md.py tests/test_cli_config.py tests/test_cli_init.py tests/test_flags.py tests/test_config.py -v
```

Expected: PASS.

- [ ] **Step 3: Run CI-parity checks in required order**

```bash
uv run --no-sync ruff check src/ tests/
uv run --no-sync mypy src/memo
uv run --no-sync pytest -m "not slow" -n auto --timeout=120 --cov=memo --cov-report=term-missing
```

Expected: PASS with coverage at or above the configured floor.

- [ ] **Step 4: Run runtime and terminal smoke checks**

```bash
MEMO_NONINTERACTIVE=1 uv run --no-sync memo config
uv run --no-sync memo config show --effective --json
uv run --no-sync memo config validate
uv run --no-sync memo doctor --strict-runtime
```

Expected: first command prints help and exits; JSON parses; validation passes;
runtime doctor reports one consistent `memo`/`memo-mcp` environment.

Run an interactive smoke in a real terminal:

```bash
uv run --no-sync memo config
```

Verify first paint, domain navigation, search, edit, discard, review, save-only,
and clean quit at 80x24 and at a wide terminal. Do not apply service restart
actions during the smoke.

- [ ] **Step 5: Inspect final diff and commit documentation**

```bash
git diff --check
git status --short
git diff --stat master...HEAD
git add README.md CHANGELOG.md docs/reference.md docs/install-new-mac.md tests
git commit -m "docs(config): document terminal configuration center"
```

Expected: no whitespace errors, no generated `.superpowers/brainstorm` files in
the commit, and documentation commit succeeds.

- [ ] **Step 6: Record durable outcome**

Save a memo outcome containing the final commit range, verification commands,
any intentionally deferred behavior, and the invariant that the app is
terminal-only under `src/memo/tui/`.
