# memo Deep Improvement First Sprint Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Clean memo's first layer of verification noise and make eval/flag/error policy explicit before any retrieval or capture behavior changes.

**Architecture:** This sprint adds small, testable guardrails instead of broad rewrites: resource lifecycle tests/fixes, source-audit tests for raw `MEMO_*` reads and broad exception policy, and named recall-eval profiles. No ranking, capture, corpus, or default behavior changes ship in this sprint.

**Tech Stack:** Python 3.13+, Click, pytest, pytest-xdist, sqlite3/sqlite-vec, existing `memo.eval_recall`, existing `memo.flags`, local git pre-push hook.

## Global Constraints

- Do not rewrite retrieval ranking.
- Do not flip HyDE, MMR, graph, or capture defaults.
- Do not bulk-edit memory records.
- Do not delete or weaken memory records without a separate cleanup plan.
- Do not restructure the whole CLI/MCP surface.
- Do not chase coverage percentage with low-value tests.
- Do not eliminate every broad exception handler.
- Use `MemoError` subclasses for normal user-visible domain errors.
- Behavioral `MEMO_*` flags go through `src/memo/flags.py`; storage/model config remains in `src/memo/config.py`.
- Keep `mlx` / `mlx_lm` imports deferred.
- Verification for this sprint includes ruff, mypy, full non-slow pytest, targeted warning-as-error tests, isolated `memo doctor --strict-runtime`, `memo eval recall --quick`, and push-time pre-push gate.

---

## File Structure

Create:

- `docs/engineering/exception-policy.md`
  - Human-readable policy for broad exception handling by layer.
- `src/memo/dev_audit.py`
  - Pure source-scanning helpers for implementation-policy tests.
- `tests/test_dev_audit.py`
  - Contract tests for raw `MEMO_*` reads and broad exception policy annotations.
- `tests/test_sqlite_resource_hygiene.py`
  - Focused tests that force sqlite cleanup to happen while warnings are captured.

Modify:

- `src/memo/eval_recall.py`
  - Add named eval profile helpers: `EvalProfile`, `profile_configs()`.
- `src/memo/cli_eval.py`
  - Add `--profile` to `memo eval recall`; keep current default behavior unchanged.
- `tests/test_eval_recall.py`
  - Cover profile membership, CLI help, and interaction between `--profile`, `--quick`, and explicit `--config`.
- `.git/hooks/pre-push`
  - Use `memo eval recall --profile pre-push` instead of spelling out A/B/E/F/G/H/I.
- `src/memo/store/connection.py`
  - Only if Task 1's warning-as-error reproduction points at holder/connection cleanup.
- `tests/conftest.py`, `tests/test_runtime_isolation.py`, or `tests/test_resume_episodes.py`
  - Only if Task 1's warning-as-error reproduction points at fixture/test cleanup.

Do not create runtime dependencies. `src/memo/dev_audit.py` must use only the standard library.

---

### Task 1: SQLite Resource Hygiene Guard

**Files:**
- Create: `tests/test_sqlite_resource_hygiene.py`
- Modify if needed: `src/memo/store/connection.py`
- Modify if needed: `tests/conftest.py`
- Modify if needed: `tests/test_runtime_isolation.py`
- Modify if needed: `tests/test_resume_episodes.py`

**Interfaces:**
- Consumes: `memo.memory.Memory.close()`, `memo.store.connection._ConnectionMixin.close()`
- Produces: targeted tests proving sqlite connections are closed or failures are reproducible under `ResourceWarning` as error

- [ ] **Step 1: Reproduce the current warning with tracebacks**

Run:

```bash
PYTHONTRACEMALLOC=1 uv run --no-sync pytest \
  tests/test_resume_episodes.py::test_mcp_episodes_search_tool \
  tests/test_runtime_isolation.py::test_install_slash_claude_proceeds_to_add_when_remove_fails \
  -q -W error::ResourceWarning
```

Expected before the fix: either FAIL with `ResourceWarning: unclosed database` or PASS if the warning only appears under full-suite interleaving. If it passes, continue with Step 2 anyway so future regressions are guarded.

- [ ] **Step 2: Write a focused sqlite cleanup test file**

Create `tests/test_sqlite_resource_hygiene.py`:

```python
from __future__ import annotations

import gc
import warnings

import pytest

from memo.config import Config
from memo.memory import Memory


def _sqlite_resource_warnings(caught: list[warnings.WarningMessage]) -> list[warnings.WarningMessage]:
    return [
        w
        for w in caught
        if issubclass(w.category, ResourceWarning)
        and "unclosed database" in str(w.message).lower()
    ]


def test_memory_close_releases_sqlite_connections(tmp_cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "memo.embedder.MLXEmbedder.embed",
        lambda self, inputs: [[1.0, 0.0, 0.0, 0.0] for _ in inputs],
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ResourceWarning)
        mem = Memory(tmp_cfg)
        mem.save(content="sqlite cleanup probe", title="SQLite Cleanup Probe")
        assert mem.search("sqlite cleanup probe", mode="bm25", limit=1)
        mem.close()
        del mem
        gc.collect()

    assert _sqlite_resource_warnings(caught) == []


def test_memory_close_is_idempotent_after_lazy_connections(
    tmp_cfg: Config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "memo.embedder.MLXEmbedder.embed",
        lambda self, inputs: [[1.0, 0.0, 0.0, 0.0] for _ in inputs],
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ResourceWarning)
        mem = Memory(tmp_cfg)
        _ = mem.store.get("missing-id")
        mem.close()
        mem.close()
        del mem
        gc.collect()

    assert _sqlite_resource_warnings(caught) == []
```

- [ ] **Step 3: Run the focused tests**

Run:

```bash
uv run --no-sync pytest tests/test_sqlite_resource_hygiene.py -q -W error::ResourceWarning
```

Expected before implementation: PASS if `Memory.close()` already covers the focused lifecycle; FAIL if the leak is in the core connection lifecycle.

- [ ] **Step 4: If Step 1 failed in runtime/resume tests, apply the smallest cleanup fix**

If `test_mcp_episodes_search_tool` is the failing path, make sure the test always closes memory and forces garbage collection before returning. Patch only the test finalizer block:

```python
    finally:
        mem.close()
        import gc

        gc.collect()
```

If `test_install_slash_claude_proceeds_to_add_when_remove_fails` is the failing path, add a post-test garbage collection finalizer to the existing `_sandbox_home` autouse fixture in `tests/test_runtime_isolation.py`:

```python
@pytest.fixture(autouse=True)
def _sandbox_home(monkeypatch, tmp_path_factory):
    # existing setup unchanged
    home = tmp_path_factory.mktemp("home")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(shims_mod, "_DEFAULT_BIN_DIR", home / ".memo" / "bin")

    real_write = install_mod.write_mandates_for_clients

    def _sandboxed_write(*args, **kwargs):
        if not kwargs.get("dry_run"):
            kwargs["cwd"] = home
        return real_write(*args, **kwargs)

    monkeypatch.setattr(install_mod, "write_mandates_for_clients", _sandboxed_write)
    yield home

    import gc

    gc.collect()
```

If the traceback points to `_ConnectionHolder`, keep `src/memo/store/connection.py` behavior idempotent and explicit by changing `close()` to clear references before closing, then preserve the existing `suppress(BaseException)` behavior:

```python
class _ConnectionHolder:
    """Close a thread-local connection when its owning thread exits."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn: sqlite3.Connection | None = conn

    def close(self) -> None:
        conn = self.conn
        self.conn = None
        if conn is not None:
            with suppress(BaseException):
                conn.close()
```

- [ ] **Step 5: Re-run warning-as-error checks**

Run:

```bash
uv run --no-sync pytest tests/test_sqlite_resource_hygiene.py -q -W error::ResourceWarning
PYTHONTRACEMALLOC=1 uv run --no-sync pytest \
  tests/test_resume_episodes.py::test_mcp_episodes_search_tool \
  tests/test_runtime_isolation.py::test_install_slash_claude_proceeds_to_add_when_remove_fails \
  -q -W error::ResourceWarning
```

Expected: PASS with no `ResourceWarning: unclosed database`.

- [ ] **Step 6: Commit**

```bash
git add tests/test_sqlite_resource_hygiene.py src/memo/store/connection.py tests/conftest.py tests/test_runtime_isolation.py tests/test_resume_episodes.py
git commit -m "test: guard sqlite resource cleanup"
```

Only add files that changed.

---

### Task 2: Source Audit Helper For Flags And Exceptions

**Files:**
- Create: `src/memo/dev_audit.py`
- Create: `tests/test_dev_audit.py`
- Create: `docs/engineering/exception-policy.md`

**Interfaces:**
- Produces: `RawMemoEnvRead`, `BroadExceptionSite`, `find_raw_memo_env_reads(root: Path) -> list[RawMemoEnvRead]`, `find_broad_exception_sites(root: Path) -> list[BroadExceptionSite]`
- Consumes: standard-library `ast`, source tree under `src/memo`

- [ ] **Step 1: Write failing audit tests**

Create `tests/test_dev_audit.py`:

```python
from __future__ import annotations

from pathlib import Path

from memo.dev_audit import (
    BROAD_EXCEPTION_ALLOWED,
    RAW_MEMO_ENV_ALLOWED,
    find_broad_exception_sites,
    find_raw_memo_env_reads,
)


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "memo"


def test_raw_memo_env_reads_are_classified() -> None:
    found = find_raw_memo_env_reads(SRC)
    unclassified = [
        f"{site.path}:{site.line}:{site.name}"
        for site in found
        if (site.relpath, site.name) not in RAW_MEMO_ENV_ALLOWED
    ]
    assert unclassified == []


def test_broad_exception_policy_targets_are_classified() -> None:
    found = find_broad_exception_sites(SRC)
    target_files = {
        "recall_logic.py",
        "memory/write_ops.py",
        "cli_recall_hook.py",
        "store/queries.py",
    }
    unclassified = [
        f"{site.path}:{site.line}"
        for site in found
        if site.relpath in target_files and (site.relpath, site.line) not in BROAD_EXCEPTION_ALLOWED
    ]
    assert unclassified == []


def test_exception_policy_doc_exists() -> None:
    policy = ROOT / "docs" / "engineering" / "exception-policy.md"
    text = policy.read_text(encoding="utf-8")
    assert "hook hot path" in text
    assert "user-visible CLI" in text
    assert "destructive write paths" in text
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
uv run --no-sync pytest tests/test_dev_audit.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'memo.dev_audit'`.

- [ ] **Step 3: Implement the audit helper**

Create `src/memo/dev_audit.py`:

```python
from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RawMemoEnvRead:
    path: Path
    relpath: str
    line: int
    name: str


@dataclass(frozen=True)
class BroadExceptionSite:
    path: Path
    relpath: str
    line: int


# Allowed raw env reads are grouped by exact file and env var. Reasons live in
# docs/engineering/exception-policy.md and in local source comments.
RAW_MEMO_ENV_ALLOWED: set[tuple[str, str]] = {
    ("config.py", "MEMO_MODEL_PROFILE"),
    ("config.py", "MEMO_RERANKER_ENABLED"),
    ("config.py", "MEMO_MEMORIES_IN_VAULT"),
    ("config.py", "MEMO_SINGLE_DB"),
    ("config.py", "MEMO_DATA_DIR"),
    ("config.py", "MEMO_STATE_DIR"),
    ("config.py", "MEMO_VAULT_PATH"),
    ("config.py", "MEMO_MEMORY_SUBDIR"),
    ("config.py", "MEMO_EMBEDDER_MODEL"),
    ("config.py", "MEMO_EMBEDDER_DIMS"),
    ("store/schema.py", "MEMO_SKIP_MODEL_VERSION_CHECK"),
    ("memory/facade.py", "MEMO_EMBEDDER_VIA_DAEMON"),
    ("mlx_gpu.py", "MEMO_GPU_LOCK_PATH"),
    ("mlx_gpu.py", "MEMO_GPU_XPROC_LOCK"),
    ("setup/config_io.py", "MEMO_CONFIG_FILE"),
    ("embed_protocol.py", "MEMO_STATE_DIR"),
    ("runtime/autoupdate.py", "MEMO_AUTO_UPDATE"),
    ("embedder.py", "MEMO_QUERY_CACHE_SIZE"),
    ("cli.py", "MEMO_DATA_DIR"),
    ("cli.py", "MEMO_VAULT_PATH"),
    ("cli.py", "MEMO_MEMORY_SUBDIR"),
}


# First sprint only classifies high-risk target files. These lines are a
# baseline inventory, not blanket approval for future sites.
BROAD_EXCEPTION_ALLOWED: set[tuple[str, int]] = {
    ("cli_recall_hook.py", 85),
    ("cli_recall_hook.py", 110),
    ("cli_recall_hook.py", 140),
    ("cli_recall_hook.py", 154),
    ("cli_recall_hook.py", 190),
    ("cli_recall_hook.py", 214),
    ("cli_recall_hook.py", 250),
    ("cli_recall_hook.py", 253),
    ("cli_recall_hook.py", 264),
    ("cli_recall_hook.py", 320),
    ("cli_recall_hook.py", 342),
    ("cli_recall_hook.py", 355),
    ("cli_recall_hook.py", 384),
    ("cli_recall_hook.py", 421),
    ("cli_recall_hook.py", 439),
    ("cli_recall_hook.py", 473),
    ("cli_recall_hook.py", 488),
    ("cli_recall_hook.py", 522),
    ("cli_recall_hook.py", 562),
    ("cli_recall_hook.py", 579),
    ("cli_recall_hook.py", 590),
    ("cli_recall_hook.py", 597),
    ("memory/write_ops.py", 126),
    ("memory/write_ops.py", 146),
    ("memory/write_ops.py", 176),
    ("memory/write_ops.py", 358),
    ("memory/write_ops.py", 428),
    ("memory/write_ops.py", 449),
    ("memory/write_ops.py", 467),
    ("memory/write_ops.py", 500),
    ("memory/write_ops.py", 635),
    ("memory/write_ops.py", 683),
    ("memory/write_ops.py", 694),
    ("memory/write_ops.py", 873),
    ("memory/write_ops.py", 929),
    ("memory/write_ops.py", 1005),
    ("recall_logic.py", 457),
    ("recall_logic.py", 566),
    ("recall_logic.py", 599),
    ("recall_logic.py", 611),
    ("recall_logic.py", 774),
    ("recall_logic.py", 895),
    ("recall_logic.py", 1028),
    ("recall_logic.py", 1056),
    ("recall_logic.py", 1079),
    ("recall_logic.py", 1180),
    ("recall_logic.py", 1198),
    ("recall_logic.py", 1206),
    ("store/queries.py", 160),
    ("store/queries.py", 256),
    ("store/queries.py", 405),
    ("store/queries.py", 800),
    ("store/queries.py", 811),
    ("store/queries.py", 826),
    ("store/queries.py", 849),
    ("store/queries.py", 871),
    ("store/queries.py", 894),
}


def _rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _constant_str(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _is_os_environ_get(call: ast.Call) -> bool:
    func = call.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr == "get"
        and isinstance(func.value, ast.Attribute)
        and func.value.attr == "environ"
        and isinstance(func.value.value, ast.Name)
        and func.value.value.id in {"os", "_os"}
    )


def find_raw_memo_env_reads(root: Path) -> list[RawMemoEnvRead]:
    out: list[RawMemoEnvRead] = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        relpath = _rel(path, root)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not _is_os_environ_get(node) or not node.args:
                continue
            name = _constant_str(node.args[0])
            if name and name.startswith("MEMO_"):
                out.append(RawMemoEnvRead(path=path, relpath=relpath, line=node.lineno, name=name))
    return out


def find_broad_exception_sites(root: Path) -> list[BroadExceptionSite]:
    out: list[BroadExceptionSite] = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        relpath = _rel(path, root)
        for node in ast.walk(tree):
            if not isinstance(node, ast.ExceptHandler) or node.type is None:
                continue
            if isinstance(node.type, ast.Name) and node.type.id == "Exception":
                out.append(BroadExceptionSite(path=path, relpath=relpath, line=node.lineno))
    return out
```

- [ ] **Step 4: Add the exception policy doc**

Create `docs/engineering/exception-policy.md`:

```markdown
# memo Exception And Env Flag Policy

## Broad Exception Policy

Broad `except Exception` handlers are allowed only when the layer explicitly
needs fault isolation. Each allowed site should have one of these intents:

- `hook hot path`: never block user work; log structured debug evidence.
- `daemon or maintenance best effort`: capture warning, receipt, or debug context.
- `optional dependency`: degrade gracefully when an optional package is absent.
- `cleanup path`: preserve the primary exception and avoid raising during cleanup.

Broad handlers are not acceptable for normal user-visible domain failures. Those
should raise or wrap `memo.errors.MemoError` subclasses so CLI/MCP callers get a
clear message.

Destructive write paths must not silently swallow failures unless a rollback,
receipt, or explicit recovery path is present.

## Raw `MEMO_*` Env Reads

Normal behavioral flags must use `memo.flags`.

Raw `os.environ.get("MEMO_*")` is allowed only for:

- bootstrap/config discovery before the flags registry can safely be used
- storage/model configuration that belongs in `Config`
- tri-state checks where `flag_bool()` would collapse unset into `False`
- low-level cross-process setup where importing higher layers creates a cycle

Each allowed raw read is classified in `memo.dev_audit.RAW_MEMO_ENV_ALLOWED`.
New raw reads must either use `memo.flags`, move into `Config`, or add a clear
classification with a source comment.
```

- [ ] **Step 5: Confirm the broad exception baseline is exact**

Run:

```bash
uv run --no-sync pytest tests/test_dev_audit.py::test_broad_exception_policy_targets_are_classified -q
```

Expected: PASS. If line numbers changed because earlier tasks edited one of the
four target files, regenerate the set with this command and replace the full
`BROAD_EXCEPTION_ALLOWED` set in `src/memo/dev_audit.py`:

```bash
uv run --no-sync python - <<'PY'
from pathlib import Path
from memo.dev_audit import find_broad_exception_sites
root = Path("src/memo")
targets = {"recall_logic.py", "memory/write_ops.py", "cli_recall_hook.py", "store/queries.py"}
for site in find_broad_exception_sites(root):
    if site.relpath in targets:
        print(f'    ("{site.relpath}", {site.line}),')
PY
```

- [ ] **Step 6: Run tests**

Run:

```bash
uv run --no-sync pytest tests/test_dev_audit.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/memo/dev_audit.py tests/test_dev_audit.py docs/engineering/exception-policy.md
git commit -m "test: audit memo env and exception policy"
```

---

### Task 3: Eval Recall Profile Source Of Truth

**Files:**
- Modify: `src/memo/eval_recall.py`
- Modify: `src/memo/cli_eval.py`
- Modify: `tests/test_eval_recall.py`
- Modify: `.git/hooks/pre-push`

**Interfaces:**
- Produces: `EvalProfile = Literal["quick", "default", "pre-push", "matrix", "expensive"]`
- Produces: `profile_configs(profile: EvalProfile) -> list[Cfg]`
- Consumes: existing `default_configs()`, `tuning_configs()`, `extra_configs()`, `select_configs()`

- [ ] **Step 1: Write failing profile tests**

Append to `tests/test_eval_recall.py` near the existing config-grid tests:

```python
def test_profile_configs_name_eval_roles() -> None:
    assert [c.name for c in eval_recall.profile_configs("quick")] == ["A vec/0.60/keep"]
    assert [c.name for c in eval_recall.profile_configs("default")] == [
        "A vec/0.60/keep",
        "B vec/0.72/excl",
        "C hyb/0.40/excl",
        "D hyb/0.40/ctx",
    ]
    assert [c.name for c in eval_recall.profile_configs("pre-push")] == [
        "A vec/0.60/keep",
        "B vec/0.72/excl",
        "E mmr/0.3",
        "F mmr/0.5",
        "G mmr/0.7",
        "H synth/0.05",
        "I synth/0.10",
    ]
    assert [c.name for c in eval_recall.profile_configs("matrix")] == [
        "A vec/0.60/keep",
        "B vec/0.72/excl",
        "C hyb/0.40/excl",
        "D hyb/0.40/ctx",
        "E mmr/0.3",
        "F mmr/0.5",
        "G mmr/0.7",
        "H synth/0.05",
        "I synth/0.10",
    ]
    assert [c.name for c in eval_recall.profile_configs("expensive")] == ["J hyb/0.40/hyde"]


def test_cli_eval_recall_profile_pre_push_selects_named_subset(tmp_path: Path, monkeypatch):
    labels_path = tmp_path / "labels.json"
    labels_path.write_text(
        json.dumps({"prompts": [{"text": "where is memo", "relevant": True}]}),
        encoding="utf-8",
    )
    labels = LabelSet(prompts=[Prompt("where is memo", relevant=True)])
    seen: list[str] = []

    monkeypatch.setattr("memo.cli_eval._get_memory", lambda cfg: object())
    monkeypatch.setattr(eval_recall, "load_labels", lambda path: labels)
    monkeypatch.setattr(eval_recall, "fingerprint_corpus", lambda mem: "corpus")

    def _evaluate(mem, *, k, labels, configs, progress=None):
        seen.extend(c.name for c in configs)
        return [eval_recall.Row(config=configs[0].name, precision_at_k=1.0, noise_at_k=0.0)]

    monkeypatch.setattr(eval_recall, "evaluate", _evaluate)

    result = CliRunner().invoke(
        cli,
        ["eval", "recall", "--labels", str(labels_path), "--profile", "pre-push", "--force"],
        env=_env(tmp_path),
    )

    assert result.exit_code == 0, result.output
    assert seen == [
        "A vec/0.60/keep",
        "B vec/0.72/excl",
        "E mmr/0.3",
        "F mmr/0.5",
        "G mmr/0.7",
        "H synth/0.05",
        "I synth/0.10",
    ]
```

Update `test_cli_eval_recall_help_lists_options` to assert:

```python
    assert "--profile" in result.output
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
uv run --no-sync pytest tests/test_eval_recall.py::test_profile_configs_name_eval_roles tests/test_eval_recall.py::test_cli_eval_recall_profile_pre_push_selects_named_subset -q
```

Expected: FAIL because `profile_configs` and `--profile` do not exist.

- [ ] **Step 3: Implement profiles in `src/memo/eval_recall.py`**

Add imports near the top:

```python
from typing import Literal
```

Add after `extra_configs()`:

```python
EvalProfile = Literal["quick", "default", "pre-push", "matrix", "expensive"]


def profile_configs(profile: EvalProfile) -> list[Cfg]:
    """Named eval profiles with explicit cost/coverage contracts."""
    if profile == "quick":
        return select_configs(quick=True)
    if profile == "default":
        return default_configs()
    if profile == "pre-push":
        return select_configs(["A", "B", "E", "F", "G", "H", "I"])
    if profile == "matrix":
        return [*default_configs(), *tuning_configs()]
    if profile == "expensive":
        return select_configs(["J"])
    raise ValueError(f"unknown recall eval profile: {profile}")
```

- [ ] **Step 4: Wire `--profile` in `src/memo/cli_eval.py`**

Add a Click option before `--config`:

```python
@click.option(
    "--profile",
    type=click.Choice(["quick", "default", "pre-push", "matrix", "expensive"]),
    default=None,
    help="Named config profile. Explicit --config values override this.",
)
```

Update the function signature to include:

```python
    profile: str | None,
```

Replace config selection logic with:

```python
    config_names = list(config or [])
    if config_names:
        selected_configs = eval_recall.select_configs(config_names, quick=quick)
    elif profile is not None:
        selected_configs = eval_recall.profile_configs(profile)  # type: ignore[arg-type]
    else:
        selected_configs = eval_recall.select_configs(None, quick=quick)
```

Keep `--quick` behavior unchanged when no profile or explicit config is supplied.

- [ ] **Step 5: Update pre-push hook**

Edit `.git/hooks/pre-push` to use the named profile:

```zsh
#!/bin/zsh
# memo retrieval regression gate (machine-local, not committed).
# Fast vec-only profile (~20-30s); hybrid C/D excluded (minutes).
# Baseline: state_dir/eval/recall_baseline.json (seeded with this profile).
# Bypass for emergencies: git push --no-verify
cd "$(git rev-parse --show-toplevel)" || exit 0
if command -v memo >/dev/null 2>&1 && [ -f eval/regression_labels.json ]; then
  echo "pre-push: memo eval recall --gate --profile pre-push..."
  if ! memo eval recall --labels eval/regression_labels.json --k 5 --force --gate --profile pre-push; then
    echo "✗ memo eval gate FAILED — retrieval regression vs saved baseline."
    echo "  Inspect: memo eval recall --labels eval/regression_labels.json --k 5 --force --profile pre-push"
    echo "  Bypass (emergencies): git push --no-verify"
    exit 1
  fi
fi
exit 0
```

- [ ] **Step 6: Run focused tests**

Run:

```bash
uv run --no-sync pytest tests/test_eval_recall.py::test_cli_eval_recall_help_lists_options tests/test_eval_recall.py::test_profile_configs_name_eval_roles tests/test_eval_recall.py::test_cli_eval_recall_profile_pre_push_selects_named_subset -q
```

Expected: PASS.

- [ ] **Step 7: Run smoke eval profile**

Run:

```bash
uv run --no-sync memo eval recall --labels eval/regression_labels.json --k 5 --force --quick --max-prompts 1
```

Expected: PASS and output starts with `Running recall eval: 1 config(s) x 1 prompt(s) = 1 search(es).`

- [ ] **Step 8: Commit**

```bash
git add src/memo/eval_recall.py src/memo/cli_eval.py tests/test_eval_recall.py .git/hooks/pre-push
git commit -m "feat: name recall eval profiles"
```

---

### Task 4: Baseline Audit Report

**Files:**
- Create: `docs/superpowers/reports/2026-07-09-memo-deep-improvement-baseline.md`

**Interfaces:**
- Consumes: commands from the design baseline
- Produces: committed baseline report for future before/after comparison

- [ ] **Step 1: Create the reports directory and report file**

Create `docs/superpowers/reports/2026-07-09-memo-deep-improvement-baseline.md`:

```markdown
# memo Deep Improvement Baseline

Date: 2026-07-09
Source spec: `docs/superpowers/specs/2026-07-09-memo-deep-improvement-roadmap-design.md`

## Verification

- Full non-slow suite: 3,534 passed, 29 skipped, 6 warnings.
- Coverage: 73.35%.
- Current coverage floor: 68%.
- Known warning class: sqlite `ResourceWarning: unclosed database`.

## Operational Health

- `memo health`: 5,982 memories, 949 archived, 693.5MB corpus, no warnings.
- Isolated doctor: passed with `/Users/fer/.local/bin/memo doctor --strict-runtime`.
- Dev-mode doctor: `uv run --no-sync memo doctor --strict-runtime` reports project `.venv` mode by design.

## Memory Utility

- Consults sampled: 437.
- Recall-hook hit rate: 98.7%.
- Strong-hit rate: 97.4%.
- Grounded rate: 0.371.
- Referenced rate: 0.009.
- Known gaps: 1.

## Corpus Lint

- `legacy_extra`: 0.
- `few_tags`: 2,635.
- `body_skinny`: 242.
- `untitled`: 0.

## Latency

- Daemon p50: 655ms.
- Daemon p95: 6,803ms.
- Daemon p99: 8,152ms.
- Subprocess p50: 8,896ms.
- Subprocess p95: 10,668ms.
- Subprocess p99: 11,572ms.

## Code Surface

- `src/memo`: about 86k Python lines.
- Top-level CLI/server modules: 113.
- CLI modules: 74.
- Server modules: 39.

## Source-Debt Counts

- Broad `except Exception`: 516 sites across 142 source files.
- Silent `pass`: 79 sites across 55 source files.
- Raw `os.environ.get("MEMO_*")`: 17 sites across 9 source files.
- todo/fixme/hack markers: 7 sites across 4 source files.

## Low-Coverage Risk Areas

- `src/memo/semantic_relations.py`: 0%.
- `src/memo/runtime/daemon.py`: 20%.
- `src/memo/memory/secret_ops.py`: 23%.
- `src/memo/cli_contradict.py`: 24%.
- `src/memo/cli_transcripts.py`: 24%.
- `src/memo/synapse_backend.py`: 24%.
- `src/memo/server_session_patterns.py`: 33%.
- `src/memo/server_idle_capture.py`: 36%.
- `src/memo/llm.py`: 37%.
- `src/memo/runtime/update.py`: 44%.

## Eval Gate

- Latest pre-push gate: 238 searches.
- Precision gate: `prec@k 0.884 >= 0.877`.
- Noise gate: `noise@k 0.000 <= 0.000`.
```

- [ ] **Step 2: Check the report for accidental draft markers**

Run:

```bash
rg -n 'TB''D|TO''DO|FIX''ME|place''holder|\\?\\?' docs/superpowers/reports/2026-07-09-memo-deep-improvement-baseline.md
```

Expected: no output.

- [ ] **Step 3: Commit**

```bash
git add -f docs/superpowers/reports/2026-07-09-memo-deep-improvement-baseline.md
git commit -m "docs: record memo improvement baseline"
```

---

### Task 5: Sprint Verification

**Files:**
- No source files unless verification reveals a small doc/test correction.

**Interfaces:**
- Consumes: all prior tasks
- Produces: final verified branch ready for push

- [ ] **Step 1: Run formatting/lint**

Run:

```bash
uv run --no-sync ruff check src/ tests/
```

Expected: `All checks passed!`

- [ ] **Step 2: Run type checking**

Run:

```bash
uv run --no-sync mypy src/memo
```

Expected: `Success: no issues found`.

- [ ] **Step 3: Run focused sprint tests**

Run:

```bash
uv run --no-sync pytest \
  tests/test_sqlite_resource_hygiene.py \
  tests/test_dev_audit.py \
  tests/test_eval_recall.py \
  -q
```

Expected: PASS.

- [ ] **Step 4: Run targeted warning-as-error tests**

Run:

```bash
PYTHONTRACEMALLOC=1 uv run --no-sync pytest \
  tests/test_sqlite_resource_hygiene.py \
  tests/test_resume_episodes.py::test_mcp_episodes_search_tool \
  tests/test_runtime_isolation.py::test_install_slash_claude_proceeds_to_add_when_remove_fails \
  -q -W error::ResourceWarning
```

Expected: PASS with no sqlite `ResourceWarning`.

- [ ] **Step 5: Run quick recall eval**

Run:

```bash
uv run --no-sync memo eval recall --labels eval/regression_labels.json --k 5 --force --quick --max-prompts 1
```

Expected: PASS and a recommendation table.

- [ ] **Step 6: Run full non-slow suite**

Run:

```bash
uv run --no-sync pytest -m "not slow" -n auto --timeout=120 --cov=memo --cov-report=term-missing
```

Expected: PASS. Warning count must not increase above the baseline 6 warnings; if Task 1 fixed sqlite warnings globally, warning count should decrease.

- [ ] **Step 7: Run isolated doctor**

Run:

```bash
/Users/fer/.local/bin/memo doctor --strict-runtime
```

Expected: all checks green except no expected github-sync dirtiness.

- [ ] **Step 8: Commit any verification-only corrections**

If verification required small corrections:

```bash
git add <changed-files>
git commit -m "fix: close memo improvement sprint verification"
```

If no files changed, skip this commit.

- [ ] **Step 9: Push to master**

Run:

```bash
git push origin master
```

Expected: pre-push hook runs `memo eval recall --gate --profile pre-push` and passes.

---

## Self-Review Checklist

- Spec coverage:
  - SQLite Resource Hygiene: Task 1.
  - Exception Policy Audit: Task 2.
  - Flag Access Audit: Task 2.
  - Eval Profile Naming: Task 3.
  - Audit Report Baseline: Task 4.
  - Full validation contract: Task 5.
- No ranking/capture/corpus defaults are changed.
- The plan intentionally does not implement Phase 3 or Phase 4; those need later specs/plans after this sprint.
- All new helper code uses only the standard library.
- All tasks end with a focused test cycle and commit.
