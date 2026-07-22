# Testing System Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make memo's PR suite detect resource leaks, stateful vector-store failures, untested branches, and uncovered changed lines while scheduled jobs expose order-dependent flakes and weak assertions.

**Architecture:** Keep deterministic enforcement in the existing Linux PR workflow and isolate expensive or probabilistic checks in two scheduled workflows. Put resource-warning capture in an opt-in pytest plugin, model VecStore behavior with a Hypothesis state machine, and express every CI promise through repository-level workflow contract tests.

**Tech Stack:** Python 3.13/3.14, pytest 8, pytest-xdist, pytest-cov/coverage.py, Hypothesis, diff-cover, pytest-randomly, pytest-repeat, mutmut 3, SQLite/sqlite-vec, GitHub Actions, uv.

## Global Constraints

- Tests must never access the real vault, state directory, daemon socket, or model configuration.
- Markdown remains source of truth and SQLite remains rebuildable derived state.
- Real MLX tests stay behind `requires_mlx`; new database tests use deterministic four-dimensional vectors and no model downloads.
- Preserve every existing dirty-worktree change and stage only files named by the active task.
- Production behavior changes use RED/GREEN/REFACTOR; CI/configuration changes start with failing contract tests.
- Standard PR CI keeps the order `ruff -> mypy -> quality gate -> pytest`.
- Deterministic resource, Hypothesis, branch, and diff gates block PRs; random/repeat and mutation jobs are scheduled/manual.
- Do not add automatic test reruns.

---

### Task 1: Testing dependencies, markers, and generated-artifact boundaries

**Files:**
- Create: `tests/test_testing_tooling_contract.py`
- Modify: `pyproject.toml:60-80`
- Modify: `pyproject.toml:200-207`
- Modify: `.gitignore:1-15`
- Modify: `uv.lock`

**Interfaces:**
- Consumes: existing `[project.optional-dependencies]` and strict pytest marker configuration.
- Produces: `dev`, `test-stability`, and `test-mutation` dependency extras; markers `db_contract`, `resource_hygiene`, and `concurrency`.

- [ ] **Step 1: Write the failing tooling contract**

```python
# tests/test_testing_tooling_contract.py
from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _pyproject() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def test_testing_dependencies_are_scoped_by_ci_cost() -> None:
    optional = _pyproject()["project"]["optional-dependencies"]
    assert "hypothesis>=6.158,<7" in optional["dev"]
    assert "diff-cover>=9,<10" in optional["dev"]
    assert optional["test-stability"] == [
        "pytest-randomly>=4.1,<5",
        "pytest-repeat>=0.9.4,<1",
    ]
    assert optional["test-mutation"] == ["mutmut>=3,<4"]


def test_testing_markers_are_strictly_registered() -> None:
    markers = "\n".join(_pyproject()["tool"]["pytest"]["ini_options"]["markers"])
    for marker in ("db_contract", "resource_hygiene", "concurrency"):
        assert f"{marker}:" in markers


def test_generated_testing_artifacts_are_ignored() -> None:
    ignored = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for entry in (".hypothesis/", "coverage.xml", "mutation-results.txt", "mutants/"):
        assert entry in ignored
```

- [ ] **Step 2: Run the contract and verify RED**

Run: `uv run --no-sync pytest tests/test_testing_tooling_contract.py -v`

Expected: FAIL because the extras, markers, and ignore entries do not exist.

- [ ] **Step 3: Add bounded dependencies and strict markers**

Add to `dev` in `pyproject.toml`:

```toml
    "hypothesis>=6.158,<7",
    "diff-cover>=9,<10",
```

Add adjacent optional extras:

```toml
test-stability = [
    "pytest-randomly>=4.1,<5",
    "pytest-repeat>=0.9.4,<1",
]
test-mutation = [
    "mutmut>=3,<4",
]
```

Append to `tool.pytest.ini_options.markers`:

```toml
    "db_contract: real SQLite/sqlite-vec behavioral contract.",
    "resource_hygiene: owns native resources and participates in the serial leak lane.",
    "concurrency: exercises threads, processes, sockets, locks, WAL, or interleavings.",
```

Append to the Python/testing section of `.gitignore`:

```gitignore
.hypothesis/
coverage.xml
mutation-results.txt
mutants/
```

- [ ] **Step 4: Regenerate and validate the universal lock**

Run: `uv lock`

Expected: exit 0 and `uv.lock` contains `hypothesis`, `diff-cover`, `pytest-randomly`, `pytest-repeat`, and `mutmut` for compatible Python environments.

Run: `uv lock --check`

Expected: exit 0.

- [ ] **Step 5: Verify GREEN**

Run: `uv run --no-sync pytest tests/test_testing_tooling_contract.py -v`

Expected: 3 passed.

- [ ] **Step 6: Commit the dependency boundary**

```bash
git add pyproject.toml uv.lock .gitignore tests/test_testing_tooling_contract.py
git commit -m "test: add hardening toolchain boundaries"
```

---

### Task 2: Opt-in resource-hygiene pytest plugin

**Files:**
- Create: `tests/resource_hygiene_plugin.py`
- Create: `tests/test_resource_hygiene_plugin.py`
- Modify: `tests/conftest.py:18-65`

**Interfaces:**
- Consumes: pytest's `request.config.getoption()` and Python `warnings`/`gc`.
- Produces: CLI flag `--resource-hygiene`; helper `resource_warning_messages(caught) -> list[str]`; autouse fixture `_enforce_resource_hygiene` loaded through `pytest_plugins`.

- [ ] **Step 1: Write failing unit and integration contracts**

```python
# tests/test_resource_hygiene_plugin.py
from __future__ import annotations

import warnings
from types import SimpleNamespace

import pytest

from tests.resource_hygiene_plugin import resource_warning_messages

pytest_plugins = ["pytester"]


def test_resource_warning_filter_rejects_only_unclosed_resources() -> None:
    caught = [
        SimpleNamespace(message=ResourceWarning("unclosed database in <sqlite3.Connection>")),
        SimpleNamespace(message=ResourceWarning("unclosed <MemoryObjectReceiveStream>")),
        SimpleNamespace(message=ResourceWarning("harmless advisory")),
        SimpleNamespace(message=DeprecationWarning("old API")),
    ]
    assert resource_warning_messages(caught) == [
        "unclosed database in <sqlite3.Connection>",
        "unclosed <MemoryObjectReceiveStream>",
    ]


def test_resource_hygiene_flag_fails_an_unclosed_sqlite_test(pytester) -> None:
    pytester.makepyfile(
        """
        import sqlite3
        import pytest

        @pytest.mark.resource_hygiene
        def test_leak():
            sqlite3.connect(\":memory:\")
        """
    )
    result = pytester.runpytest(
        "-p", "tests.resource_hygiene_plugin", "--resource-hygiene", "-q"
    )
    result.assert_outcomes(errors=1)
    result.stdout.fnmatch_lines(["*unclosed database*"])


def test_resource_hygiene_plugin_is_inert_without_flag(pytester) -> None:
    pytester.makepyfile(
        """
        import sqlite3
        import warnings

        def test_leak_is_not_gated_by_default():
            with warnings.catch_warnings():
                warnings.simplefilter(\"ignore\", ResourceWarning)
                sqlite3.connect(\":memory:\")
        """
    )
    result = pytester.runpytest("-p", "tests.resource_hygiene_plugin", "-q")
    result.assert_outcomes(passed=1)
```

- [ ] **Step 2: Verify RED**

Run: `uv run --no-sync pytest tests/test_resource_hygiene_plugin.py -v`

Expected: collection ERROR because `tests.resource_hygiene_plugin` does not exist.

- [ ] **Step 3: Implement the opt-in plugin**

```python
# tests/resource_hygiene_plugin.py
from __future__ import annotations

import gc
import warnings
from collections.abc import Iterable
from typing import Any

import pytest

_UNCLOSED_RESOURCE_MARKERS = (
    "unclosed database",
    "unclosed <memoryobject",
    "unclosed socket",
    "unclosed file",
)


def resource_warning_messages(caught: Iterable[Any]) -> list[str]:
    messages: list[str] = []
    for warning in caught:
        if not isinstance(warning.message, ResourceWarning):
            continue
        text = str(warning.message)
        if any(marker in text.lower() for marker in _UNCLOSED_RESOURCE_MARKERS):
            messages.append(text)
    return messages


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.getgroup("memo testing").addoption(
        "--resource-hygiene",
        action="store_true",
        default=False,
        help="fail selected tests that leak SQLite, stream, socket, or file resources",
    )


@pytest.fixture(autouse=True)
def _enforce_resource_hygiene(request: pytest.FixtureRequest):
    if not request.config.getoption("--resource-hygiene"):
        yield
        return

    gc.collect()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ResourceWarning)
        yield
        gc.collect()

    leaked = resource_warning_messages(caught)
    if leaked:
        pytest.fail("unclosed resources:\n" + "\n".join(f"- {item}" for item in leaked))
```

At the top level of `tests/conftest.py`, after importing pytest, add:

```python
pytest_plugins = ["tests.resource_hygiene_plugin"]
```

- [ ] **Step 4: Verify GREEN and default-suite inertness**

Run: `uv run --no-sync pytest tests/test_resource_hygiene_plugin.py -v`

Expected: 3 passed.

Run: `uv run --no-sync pytest tests/test_store.py::test_store_roundtrip -q`

Expected: selected test passes without requiring `--resource-hygiene`.

- [ ] **Step 5: Commit the plugin**

```bash
git add tests/conftest.py tests/resource_hygiene_plugin.py tests/test_resource_hygiene_plugin.py
git commit -m "test: add opt-in resource leak detector"
```

---

### Task 3: Clean current resource owners and broad-exception baseline

**Files:**
- Modify: `src/memo/dev_audit.py:54-92`
- Modify: `tests/test_proactive_acted.py`
- Modify: `tests/test_sqlite_cleanup.py`
- Modify: `tests/test_http_auth.py`
- Modify: `tests/test_vector_database_contracts.py`
- Modify: `tests/test_housekeeping_contracts.py`

**Interfaces:**
- Consumes: `ProactiveStore.__enter__/__exit__`, `VecStore.close()`, `--resource-hygiene`, and `BROAD_EXCEPTION_ALLOWED` lexical ordinals.
- Produces: a green exception audit and a non-empty serial `resource_hygiene` selection with explicit resource ownership.

- [ ] **Step 1: Reproduce both baseline failures**

Run:

```bash
uv run --no-sync pytest tests/test_dev_audit.py::test_broad_exception_policy_targets_are_classified -v
```

Expected: FAIL with `cli_recall_hook.py:820:recall_hook:22` unclassified.

Run:

```bash
uv run --no-sync pytest tests/test_proactive_acted.py tests/test_sqlite_cleanup.py tests/test_http_auth.py -n 0 --resource-hygiene -q
```

Expected: ERROR/FAIL with one or more unclosed SQLite resource messages.

- [ ] **Step 2: Classify the intentional hook fail-open boundary**

Add immediately after ordinal 21 in `BROAD_EXCEPTION_ALLOWED`:

```python
    # Proactive urgent rendering is optional hook-hot-path work. Store reads,
    # timestamp parsing, or rendering failures must degrade to no urgent line
    # and must never block the recall payload.
    ("cli_recall_hook.py", "recall_hook", 22),
```

- [ ] **Step 3: Make ProactiveStore ownership explicit**

For every direct store in `tests/test_proactive_acted.py`, replace:

```python
s = ProactiveStore(tmp_path / "p.db")
s.put_candidates([...])
# assertions using s
```

with:

```python
with ProactiveStore(tmp_path / "p.db") as store:
    store.put_candidates([...])
    # calls and assertions use store inside this block
```

For the CLI feedback tests, keep the store open only while seeding and reading:

```python
with ProactiveStore(tmp_path / "state" / "proactive.db") as store:
    store.put_candidates([...])
    out = runner.invoke(cli, ["get", memory_id], env=env)
    assert out.exit_code == 0, out.output
    assert store.kind_multipliers(floor=0.2)[KIND_RELIABILITY] >= 1.0
```

- [ ] **Step 4: Mark the focused resource/database suites**

Add at module scope where pytest is already imported:

```python
pytestmark = pytest.mark.resource_hygiene
```

Apply `resource_hygiene` to `test_proactive_acted.py`, `test_sqlite_cleanup.py`,
and `test_http_auth.py`. Apply both markers to the contract files:

```python
pytestmark = [pytest.mark.db_contract, pytest.mark.resource_hygiene]
```

Do not add a blanket warning ignore to `test_http_auth.py`.

- [ ] **Step 5: Verify GREEN serially and next to HTTP auth**

Run:

```bash
uv run --no-sync pytest tests/test_dev_audit.py::test_broad_exception_policy_targets_are_classified -v
uv run --no-sync pytest tests/test_proactive_acted.py tests/test_sqlite_cleanup.py tests/test_http_auth.py tests/test_vector_database_contracts.py tests/test_housekeeping_contracts.py -n 0 --resource-hygiene -q
```

Expected: both commands exit 0 with no resource-hygiene teardown errors.

- [ ] **Step 6: Commit the clean baseline**

```bash
git add src/memo/dev_audit.py tests/test_proactive_acted.py tests/test_sqlite_cleanup.py tests/test_http_auth.py tests/test_vector_database_contracts.py tests/test_housekeeping_contracts.py
git commit -m "test: enforce resource ownership in database suites"
```

---

### Task 4: Stateful Hypothesis model for VecStore

**Files:**
- Create: `tests/test_vector_store_state_machine.py`

**Interfaces:**
- Consumes: `VecStore.upsert`, `delete`, `hard_delete`, `clear_memory_index`, `get`, `count`, `list_soft_deleted`, `list_by_tag`, `search`, `search_bm25`, signal writes, `connection`, and `close`.
- Produces: `_assert_store_matches_model(store, model, purged) -> None`; `VecStoreStateMachine.TestCase`; PR Hypothesis profile of 25 examples × 30 steps.

- [ ] **Step 1: Write an oracle test before the helper exists**

Start `tests/test_vector_store_state_machine.py` with imports, a model record, and this test:

```python
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pytest
from hypothesis import HealthCheck, settings, strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, rule

from memo.store import VecStore

pytestmark = [pytest.mark.db_contract, pytest.mark.resource_hygiene]


@dataclass
class ModelRow:
    slot: int
    version: int
    deleted: bool = False


def _rank_value(slot: int, version: int) -> float:
    return (slot * 10 + version + 1) / 100


def _body_token(slot: int, version: int) -> str:
    return f"slot{slot}version{version}"


def _write(store: VecStore, slot: int, version: int) -> None:
    memory_id = f"memory-{slot}"
    embedding = [1.0, _rank_value(slot, version), 0.0, 0.0]
    store.upsert(
        id_=memory_id,
        path=f"memory/{memory_id}.md",
        title=f"Memory {slot} v{version}",
        type_="decision" if version % 2 else "note",
        tags=[f"slot-{slot}", f"version-{version}"],
        created="2026-01-01T00:00:00+00:00",
        updated=f"2026-01-{version + 1:02d}T00:00:00+00:00",
        body_hash=f"hash-{slot}-{version}",
        embedding=embedding,
        body_text=f"body {_body_token(slot, version)}",
    )


def test_model_oracle_detects_a_missing_vector(tmp_path: Path) -> None:
    store = VecStore(tmp_path / "vectors.db", dims=4, vec_quant="off")
    try:
        _write(store, 1, 1)
        store.connection.execute("DELETE FROM vec WHERE id = ?", ("memory-1",))
        store.connection.commit()
        with pytest.raises(AssertionError, match="vector row count"):
            _assert_store_matches_model(
                store,
                {"memory-1": ModelRow(slot=1, version=1)},
                purged=set(),
            )
    finally:
        store.close()
```

- [ ] **Step 2: Verify RED**

Run:

```bash
uv run --no-sync pytest tests/test_vector_store_state_machine.py::test_model_oracle_detects_a_missing_vector -v
```

Expected: ERROR/FAIL because `_assert_store_matches_model` is undefined.

- [ ] **Step 3: Implement the model oracle**

Add above the oracle test:

```python
def _assert_store_matches_model(
    store: VecStore,
    model: dict[str, ModelRow],
    purged: set[str],
) -> None:
    active = {memory_id for memory_id, row in model.items() if not row.deleted}
    deleted = {memory_id for memory_id, row in model.items() if row.deleted}
    assert store.count() == len(active)
    assert set(store.list_soft_deleted()) == deleted

    for memory_id, expected in model.items():
        row = store.get(memory_id)
        vector_count = store.connection.execute(
            "SELECT COUNT(*) FROM vec WHERE id = ?", (memory_id,)
        ).fetchone()[0]
        fts_count = store.connection.execute(
            "SELECT COUNT(*) FROM fts WHERE id = ?", (memory_id,)
        ).fetchone()[0]
        if expected.deleted:
            assert row is None
            assert vector_count == 0, "vector row count for tombstone"
            assert fts_count == 0, "FTS row count for tombstone"
            assert store.search_bm25(
                _body_token(expected.slot, expected.version), limit=20
            ) == []
            continue

        assert row is not None
        assert row["title"] == f"Memory {expected.slot} v{expected.version}"
        assert row["type"] == ("decision" if expected.version % 2 else "note")
        assert row["tags"] == [
            f"slot-{expected.slot}",
            f"version-{expected.version}",
        ]
        assert vector_count == 1, "vector row count for active memory"
        assert fts_count == 1, "FTS row count for active memory"
        body = store.get_fts_body_by_path(f"memory/{memory_id}.md")
        assert body == f"body {_body_token(expected.slot, expected.version)}"
        assert [
            hit["id"]
            for hit in store.search_bm25(
                _body_token(expected.slot, expected.version), limit=20
            )
        ] == [memory_id]
        for old_version in range(9):
            if old_version != expected.version:
                assert store.search_bm25(
                    _body_token(expected.slot, old_version), limit=20
                ) == []

    expected_vector_order = [
        memory_id
        for memory_id, row in sorted(
            (
                (memory_id, row)
                for memory_id, row in model.items()
                if not row.deleted
            ),
            key=lambda item: _rank_value(item[1].slot, item[1].version),
        )
    ]
    vector_ids = [
        hit["id"] for hit in store.search([1.0, 0.0, 0.0, 0.0], limit=20)
    ]
    assert vector_ids == expected_vector_order

    for type_ in ("decision", "note"):
        expected_type_ids = {
            memory_id
            for memory_id, row in model.items()
            if not row.deleted
            and ("decision" if row.version % 2 else "note") == type_
        }
        assert {
            hit["id"]
            for hit in store.search(
                [1.0, 0.0, 0.0, 0.0], limit=20, type_=type_
            )
        } == expected_type_ids

    for slot in range(6):
        expected_tag_ids = {
            memory_id
            for memory_id, row in model.items()
            if not row.deleted and row.slot == slot
        }
        assert {row["id"] for row in store.list_by_tag(f"slot-{slot}")} == (
            expected_tag_ids
        )

    for memory_id in purged:
        for table, column in (
            ("meta", "id"),
            ("vec", "id"),
            ("fts", "id"),
            ("access", "id"),
            ("memory_health", "id"),
            ("source_feedback", "source_id"),
            ("source_feedback_vec", "source_id"),
        ):
            count = store.connection.execute(
                f"SELECT COUNT(*) FROM {table} WHERE {column} = ?",  # noqa: S608
                (memory_id,),
            ).fetchone()[0]
            assert count == 0, f"{table} rows for hard-deleted memory"
```

- [ ] **Step 4: Verify the oracle test is GREEN**

Run: `uv run --no-sync pytest tests/test_vector_store_state_machine.py::test_model_oracle_detects_a_missing_vector -v`

Expected: PASS because the oracle catches the injected divergence.

- [ ] **Step 5: Add the complete state machine**

```python
class VecStoreStateMachine(RuleBasedStateMachine):
    def __init__(self) -> None:
        super().__init__()
        self._tmp = TemporaryDirectory(prefix="memo-vec-state-")
        self._env = patch.dict(
            os.environ,
            {"MEMO_SOFT_DELETE": "1", "MEMO_TANTIVY_ENABLED": "0"},
        )
        self._env.start()
        self._db_path = Path(self._tmp.name) / "vectors.db"
        self.store = VecStore(self._db_path, dims=4, vec_quant="off")
        self.model: dict[str, ModelRow] = {}
        self.purged: set[str] = set()

    @rule(slot=st.integers(min_value=0, max_value=5), version=st.integers(0, 8))
    def upsert(self, slot: int, version: int) -> None:
        _write(self.store, slot, version)
        memory_id = f"memory-{slot}"
        self.model[memory_id] = ModelRow(slot=slot, version=version)
        self.purged.discard(memory_id)

    @rule(slot=st.integers(min_value=0, max_value=5))
    def soft_delete(self, slot: int) -> None:
        memory_id = f"memory-{slot}"
        existed = memory_id in self.model
        assert self.store.delete(memory_id) is existed
        if existed:
            self.model[memory_id].deleted = True

    @rule(slot=st.integers(min_value=0, max_value=5))
    def restore_by_upsert(self, slot: int) -> None:
        memory_id = f"memory-{slot}"
        current = self.model.get(memory_id)
        if current is None or not current.deleted:
            return
        version = (current.version + 1) % 9
        _write(self.store, slot, version)
        self.model[memory_id] = ModelRow(slot=slot, version=version)
        self.purged.discard(memory_id)

    @rule(slot=st.integers(min_value=0, max_value=5))
    def hard_delete(self, slot: int) -> None:
        memory_id = f"memory-{slot}"
        existed = memory_id in self.model
        if existed:
            self.store.touch([memory_id], ts="2026-02-01T00:00:00+00:00")
            self.store.set_confidence_batch([(memory_id, 0.42)])
            self.store.record_source_feedback(
                source_id=memory_id,
                query_text=f"feedback for {memory_id}",
                query_emb=[0.0, 1.0, 0.0, 0.0],
                rating=1,
                feedback_id=f"feedback-{memory_id}",
            )
        assert self.store.hard_delete(memory_id) is existed
        self.model.pop(memory_id, None)
        self.purged.add(memory_id)

    @rule()
    def reopen(self) -> None:
        self.store.close()
        self.store = VecStore(self._db_path, dims=4, vec_quant="off")

    @rule()
    def clear_derived_index(self) -> None:
        assert self.store.clear_memory_index() == len(self.model)
        self.model.clear()

    @invariant()
    def database_matches_reference_model(self) -> None:
        _assert_store_matches_model(self.store, self.model, self.purged)

    def teardown(self) -> None:
        self.store.close()
        self._env.stop()
        self._tmp.cleanup()


settings.register_profile(
    "memo_pr",
    max_examples=25,
    stateful_step_count=30,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.register_profile(
    "ci_extended",
    max_examples=100,
    stateful_step_count=75,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "memo_pr"))

TestVecStoreStateMachine = VecStoreStateMachine.TestCase
```

- [ ] **Step 6: Run the state machine and existing contracts**

Run:

```bash
uv run --no-sync pytest tests/test_vector_store_state_machine.py tests/test_vector_database_contracts.py tests/test_store.py -q
```

Expected: all selected tests pass with no state-machine invariant failure.

- [ ] **Step 7: Commit the stateful contract**

```bash
git add tests/test_vector_store_state_machine.py
git commit -m "test(store): add stateful sqlite-vec model"
```

---

### Task 5: Branch-aware aggregate and changed-lines coverage gates

**Files:**
- Modify: `pyproject.toml:267-280`
- Modify: `.github/workflows/test.yml:12-60`
- Modify: `tests/test_quality_gate.py:101-120`
- Create: `tests/test_testing_workflows.py`

**Interfaces:**
- Consumes: pytest-cov XML output, coverage.py branch measurement, `diff-cover coverage.xml`.
- Produces: branch-aware `fail_under >= 74`, `coverage.xml`, serial leak check, and PR-only 90% diff coverage.

- [ ] **Step 1: Write failing configuration and workflow contracts**

Update `test_quality_baseline_schema_and_configuration`:

```python
    parsed = tomllib.loads(pyproject)
    coverage_run = parsed["tool"]["coverage"]["run"]
    coverage_report = parsed["tool"]["coverage"]["report"]
    assert coverage_run["branch"] is True
    assert coverage_report["fail_under"] >= 74
```

Add imports `tomllib` and this file:

```python
# tests/test_testing_workflows.py
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"


def test_pr_workflow_enforces_resource_and_diff_coverage_gates() -> None:
    workflow = (WORKFLOWS / "test.yml").read_text(encoding="utf-8")
    assert "fetch-depth: 0" in workflow
    assert "--resource-hygiene" in workflow
    assert '-m "resource_hygiene"' in workflow
    assert "--cov-report=xml" in workflow
    assert "diff-cover coverage.xml" in workflow
    assert "--compare-branch=origin/master" in workflow
    assert "--fail-under=90" in workflow
    assert "github.event_name == 'pull_request'" in workflow
```

- [ ] **Step 2: Verify RED**

Run:

```bash
uv run --no-sync pytest tests/test_quality_gate.py::test_quality_baseline_schema_and_configuration tests/test_testing_workflows.py::test_pr_workflow_enforces_resource_and_diff_coverage_gates -v
```

Expected: FAIL because branch coverage, resource lane, XML, and diff-cover are absent.

- [ ] **Step 3: Enable branch coverage with an initial safe ratchet**

Change coverage configuration to:

```toml
[tool.coverage.run]
source = ["memo"]
omit = ["*/tests/*"]
branch = true

[tool.coverage.report]
# Branch-aware baseline ratchet. Re-measure before raising; never reduce below 74.
fail_under = 74
```

- [ ] **Step 4: Add deterministic gates to the Linux matrix job**

In the first checkout of `.github/workflows/test.yml`, add:

```yaml
        with:
          persist-credentials: false
          fetch-depth: 0
```

After the quality budget and before the full suite, add:

```yaml
      - name: Resource hygiene (serial ownership gate)
        run: >-
          .venv/bin/python -m pytest -m "resource_hygiene" -n 0
          --timeout=120 --resource-hygiene
```

Extend the coverage command with:

```text
--cov-report=xml
```

Add immediately after it:

```yaml
      - name: Changed-lines coverage
        if: github.event_name == 'pull_request'
        run: >-
          .venv/bin/diff-cover coverage.xml
          --compare-branch=origin/master --fail-under=90 --show-uncovered
```

- [ ] **Step 5: Verify focused GREEN**

Run:

```bash
uv run --no-sync pytest tests/test_quality_gate.py tests/test_testing_workflows.py tests/test_release_workflows.py -q
```

Expected: all selected tests pass.

- [ ] **Step 6: Measure branch coverage and apply the exact ratchet formula**

Run:

```bash
uv run --no-sync pytest -m "not slow" -n auto --timeout=120 --cov=memo --cov-report=term --cov-report=xml
```

Expected: exit 0 at the initial 74% gate. Read the unrounded total from the final report. Compute `max(74, floor(total) - 1)` and set `tool.coverage.report.fail_under` to that integer. Re-run the same command; it must remain green. If the first result is below 74%, add focused tests for uncovered stable/core branches before changing the floor; never lower the gate.

- [ ] **Step 7: Commit coverage enforcement**

```bash
git add pyproject.toml .github/workflows/test.yml tests/test_quality_gate.py tests/test_testing_workflows.py
git commit -m "ci: enforce branch and changed-lines coverage"
```

---

### Task 6: Scheduled randomized and repeated stability workflow

**Files:**
- Create: `.github/workflows/test-stability.yml`
- Modify: `tests/test_testing_workflows.py`
- Modify: `tests/test_history.py`
- Modify: `tests/test_store.py`
- Modify: `tests/test_capture_hooks.py`
- Modify: `tests/test_token_ledger.py`

**Interfaces:**
- Consumes: `test-stability` extra, markers `concurrency`/`resource_hygiene`, pytest-randomly replay seeds, pytest-repeat session scope.
- Produces: nightly/manual shuffled suite plus ten-session focused repetition and JUnit artifact.

- [ ] **Step 1: Add the failing workflow contract**

```python
def test_stability_workflow_is_replayable_and_never_masks_flakes() -> None:
    workflow = (WORKFLOWS / "test-stability.yml").read_text(encoding="utf-8")
    assert "schedule:" in workflow and "workflow_dispatch:" in workflow
    assert "timeout-minutes: 30" in workflow
    assert "--extra test-stability" in workflow
    assert "--randomly-seed" in workflow
    assert "github.run_id" in workflow
    assert '-m "concurrency or resource_hygiene"' in workflow
    assert "--count=10" in workflow
    assert "--repeat-scope=session" in workflow
    assert "-x" in workflow
    assert "pytest-rerunfailures" not in workflow
    assert "upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a" in workflow
    assert "if: always()" in workflow
```

- [ ] **Step 2: Verify RED**

Run: `uv run --no-sync pytest tests/test_testing_workflows.py::test_stability_workflow_is_replayable_and_never_masks_flakes -v`

Expected: FAIL with `FileNotFoundError` for `test-stability.yml`.

- [ ] **Step 3: Mark the existing concurrency suites**

Add `pytestmark = pytest.mark.concurrency` to `tests/test_history.py`,
`tests/test_capture_hooks.py`, and `tests/test_token_ledger.py`. In
`tests/test_store.py`, decorate the explicit thread/migration interleaving tests
with `@pytest.mark.concurrency` rather than marking the whole module.

- [ ] **Step 4: Create the stability workflow**

```yaml
name: test-stability

on:
  schedule:
    - cron: "30 4 * * *"
  workflow_dispatch:

permissions:
  contents: read

jobs:
  stability:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - uses: actions/checkout@9c091bb21b7c1c1d1991bb908d89e4e9dddfe3e0  # v7.0.0
        with:
          persist-credentials: false
      - uses: actions/setup-python@ece7cb06caefa5fff74198d8649806c4678c61a1  # v6
        with:
          python-version: "3.13"
      - name: Install uv
        run: pip install uv==0.11.21
      - name: Install stability environment
        run: uv sync --frozen --extra dev --extra cpu --extra http --extra test-stability --python 3.13
      - name: Full non-slow suite in replayable random order
        env:
          RANDOM_SEED: ${{ github.run_id }}
        run: |
          echo "Replay: pytest -m 'not slow' -n 0 --randomly-seed=$RANDOM_SEED"
          .venv/bin/python -m pytest -m "not slow" -n 0 --timeout=120 \
            --randomly-seed="$RANDOM_SEED" --junitxml=stability-full.xml
      - name: Repeat concurrency and resource ownership sessions
        env:
          RANDOM_SEED: ${{ github.run_id }}
        run: >-
          .venv/bin/python -m pytest -m "concurrency or resource_hygiene"
          -n 0 -x --timeout=120 --randomly-seed="$RANDOM_SEED"
          --count=10 --repeat-scope=session --resource-hygiene
          --junitxml=stability-repeat.xml
      - name: Upload stability reports
        if: always()
        uses: actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a  # v7.0.1
        with:
          name: stability-results-${{ github.run_id }}
          path: stability-*.xml
          if-no-files-found: warn
```

- [ ] **Step 5: Verify marker selection and workflow GREEN**

Run:

```bash
uv run --no-sync pytest --collect-only -m "concurrency or resource_hygiene" -q
uv run --no-sync pytest tests/test_testing_workflows.py tests/test_release_workflows.py -q
```

Expected: collection lists at least one test and workflow contracts pass.

- [ ] **Step 6: Commit the stability lane**

```bash
git add .github/workflows/test-stability.yml tests/test_testing_workflows.py tests/test_history.py tests/test_store.py tests/test_capture_hooks.py tests/test_token_ledger.py
git commit -m "ci: add replayable stability testing"
```

---

### Task 7: Scoped scheduled mutation testing

**Files:**
- Modify: `pyproject.toml` after coverage configuration
- Create: `.github/workflows/mutation-tests.yml`
- Create: `scripts/check_mutation_results.py`
- Create: `tests/test_mutation_result_gate.py`
- Modify: `tests/test_testing_tooling_contract.py`
- Modify: `tests/test_testing_workflows.py`

**Interfaces:**
- Consumes: `test-mutation` extra, mutmut 3 `[tool.mutmut]`, focused storage/retrieval/housekeeping tests.
- Produces: weekly/manual mutation job, bounded mutation surface, `mutation-results.txt`, uploaded mutants directory, and a repository-owned hard-failure gate for surviving or incomplete mutants.

- [ ] **Step 1: Write failing mutmut configuration and workflow contracts**

Append to `tests/test_testing_tooling_contract.py`:

```python
def test_mutation_scope_is_bounded_to_covered_core_paths() -> None:
    config = _pyproject()["tool"]["mutmut"]
    assert config["mutate_only_covered_lines"] is True
    assert set(config["only_mutate"]) == {
        "src/memo/store/vec_base.py",
        "src/memo/memory/search_scoring_ops.py",
        "src/memo/session.py",
        "src/memo/sqlite_snapshot.py",
    }
    selected = set(config["pytest_add_cli_args_test_selection"])
    assert "tests/test_vector_database_contracts.py" in selected
    assert "tests/test_search_scoring_ops_unit.py" in selected
    assert "tests/test_housekeeping_contracts.py" in selected
```

Append to `tests/test_testing_workflows.py`:

```python
def test_mutation_workflow_is_scoped_scheduled_and_retains_results() -> None:
    workflow = (WORKFLOWS / "mutation-tests.yml").read_text(encoding="utf-8")
    assert "schedule:" in workflow and "workflow_dispatch:" in workflow
    assert "timeout-minutes: 30" in workflow
    assert "--extra test-mutation" in workflow
    assert "mutmut run" in workflow
    assert "mutmut results" in workflow
    assert "scripts/check_mutation_results.py mutants" in workflow
    assert "PIPESTATUS[0]" in workflow
    assert "mutation-results.txt" in workflow
    assert "if: always()" in workflow
    assert "upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a" in workflow
```

- [ ] **Step 2: Verify RED**

Run:

```bash
uv run --no-sync pytest tests/test_testing_tooling_contract.py::test_mutation_scope_is_bounded_to_covered_core_paths tests/test_testing_workflows.py::test_mutation_workflow_is_scoped_scheduled_and_retains_results -v
```

Expected: FAIL because `[tool.mutmut]` and the workflow do not exist.

- [ ] **Step 3: Add bounded mutmut configuration**

```toml
[tool.mutmut]
source_paths = ["src/memo"]
only_mutate = [
    "src/memo/store/vec_base.py",
    "src/memo/memory/search_scoring_ops.py",
    "src/memo/session.py",
    "src/memo/sqlite_snapshot.py",
]
pytest_add_cli_args_test_selection = [
    "tests/test_vector_database_contracts.py",
    "tests/test_store.py",
    "tests/test_search_scoring_ops_unit.py",
    "tests/test_housekeeping_contracts.py",
    "tests/test_session.py",
    "tests/test_sqlite_cleanup.py",
]
mutate_only_covered_lines = true
use_setproctitle = false
```

- [ ] **Step 4: Write the failing survivor-gate tests**

```python
# tests/test_mutation_result_gate.py
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.check_mutation_results import blocking_mutants, main


def _write_meta(root: Path, results: dict[str, int | None]) -> None:
    path = root / "src" / "memo" / "example.py.meta"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"exit_code_by_key": results}), encoding="utf-8")


def test_blocking_mutants_reports_survivors_and_incomplete_results(tmp_path: Path) -> None:
    _write_meta(
        tmp_path,
        {
            "memo.example.x_killed__mutmut_1": 1,
            "memo.example.x_survived__mutmut_2": 0,
            "memo.example.x_unchecked__mutmut_3": None,
            "memo.example.x_no_tests__mutmut_4": 33,
        },
    )
    assert blocking_mutants(tmp_path) == {
        "memo.example.x_no_tests__mutmut_4": "no-tests",
        "memo.example.x_survived__mutmut_2": "survived",
        "memo.example.x_unchecked__mutmut_3": "not-checked",
    }


def test_gate_exits_nonzero_for_survivor(tmp_path: Path) -> None:
    _write_meta(tmp_path, {"memo.example.x_value__mutmut_1": 0})
    assert main([str(tmp_path)]) == 1


def test_gate_rejects_missing_metadata(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="no mutmut metadata"):
        blocking_mutants(tmp_path)
```

Run: `uv run --no-sync pytest tests/test_mutation_result_gate.py -v`

Expected: collection ERROR because `scripts.check_mutation_results` does not exist.

- [ ] **Step 5: Implement the repository-owned survivor gate**

```python
# scripts/check_mutation_results.py
from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

_BLOCKING_EXIT_CODES = {
    None: "not-checked",
    0: "survived",
    5: "no-tests",
    33: "no-tests",
    35: "suspicious",
}


def blocking_mutants(root: Path) -> dict[str, str]:
    metadata = sorted(root.rglob("*.meta"))
    if not metadata:
        raise RuntimeError(f"no mutmut metadata found under {root}")

    blocked: dict[str, str] = {}
    for path in metadata:
        payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        results = payload.get("exit_code_by_key", {})
        for mutant, exit_code in results.items():
            reason = _BLOCKING_EXIT_CODES.get(exit_code)
            if reason is not None:
                blocked[str(mutant)] = reason
    return dict(sorted(blocked.items()))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, nargs="?", default=Path("mutants"))
    args = parser.parse_args(argv)
    blocked = blocking_mutants(args.root)
    if not blocked:
        print("mutation gate: no surviving or incomplete mutants")
        return 0
    for mutant, reason in blocked.items():
        print(f"{reason}: {mutant}")
    print(f"mutation gate failed: {len(blocked)} blocking mutant(s)")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
```

Run: `uv run --no-sync pytest tests/test_mutation_result_gate.py -v`

Expected: 3 passed.

- [ ] **Step 6: Create the mutation workflow**

```yaml
name: mutation-tests

on:
  schedule:
    - cron: "0 5 * * 0"
  workflow_dispatch:

permissions:
  contents: read

jobs:
  mutate-core:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - uses: actions/checkout@9c091bb21b7c1c1d1991bb908d89e4e9dddfe3e0  # v7.0.0
        with:
          persist-credentials: false
      - uses: actions/setup-python@ece7cb06caefa5fff74198d8649806c4678c61a1  # v6
        with:
          python-version: "3.13"
      - name: Install uv
        run: pip install uv==0.11.21
      - name: Install mutation environment
        run: uv sync --frozen --extra dev --extra cpu --extra test-mutation --python 3.13
      - name: Prove focused baseline is green
        run: >-
          .venv/bin/python -m pytest tests/test_vector_database_contracts.py
          tests/test_store.py tests/test_search_scoring_ops_unit.py
          tests/test_housekeeping_contracts.py tests/test_session.py
          tests/test_sqlite_cleanup.py -q
      - name: Run scoped mutations
        id: mutate
        shell: bash
        run: |
          set +e
          .venv/bin/mutmut run 2>&1 | tee mutation-results.txt
          status=${PIPESTATUS[0]}
          .venv/bin/mutmut results 2>&1 | tee -a mutation-results.txt
          if [ "$status" -ne 0 ]; then
            exit "$status"
          fi
          .venv/bin/python scripts/check_mutation_results.py mutants \
            2>&1 | tee -a mutation-results.txt
      - name: Upload mutation results
        if: always()
        uses: actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a  # v7.0.1
        with:
          name: mutation-results-${{ github.run_id }}
          path: |
            mutation-results.txt
            mutants/
          if-no-files-found: warn
```

- [ ] **Step 7: Verify contracts, survivor gate, and mutmut discovery**

Run:

```bash
uv run --no-sync pytest tests/test_testing_tooling_contract.py tests/test_testing_workflows.py tests/test_mutation_result_gate.py tests/test_release_workflows.py -q
uv run --extra test-mutation mutmut --help
```

Expected: workflow/tooling tests pass and mutmut prints help with exit 0.

- [ ] **Step 8: Commit the mutation lane**

```bash
git add pyproject.toml uv.lock .github/workflows/mutation-tests.yml scripts/check_mutation_results.py tests/test_mutation_result_gate.py tests/test_testing_tooling_contract.py tests/test_testing_workflows.py
git commit -m "ci: add scoped mutation testing"
```

---

### Task 8: End-to-end verification and ratchet documentation

**Files:**
- Modify: `docs/superpowers/specs/2026-07-22-testing-system-hardening-design.md` only if the measured final coverage floor differs from 74.
- Modify: `docs/superpowers/plans/2026-07-22-testing-system-hardening.md` to check completed steps during execution.

**Interfaces:**
- Consumes: every task's committed surface.
- Produces: fresh evidence for dependency reproducibility, resource hygiene, stateful storage behavior, static checks, workflow contracts, branch coverage, and int8 compatibility.

- [x] **Step 1: Validate dependency resolution**

Run:

```bash
uv lock --check
uv sync --frozen --extra dev --extra cpu --extra http --extra test-stability --extra test-mutation
```

Expected: both commands exit 0 without changing `uv.lock`.

- [x] **Step 2: Run focused new gates**

Run:

```bash
uv run --no-sync pytest tests/test_resource_hygiene_plugin.py tests/test_testing_tooling_contract.py tests/test_testing_workflows.py tests/test_vector_store_state_machine.py tests/test_dev_audit.py tests/test_quality_gate.py tests/test_release_workflows.py -q
uv run --no-sync pytest -m "resource_hygiene" -n 0 --timeout=120 --resource-hygiene -q
```

Expected: both commands exit 0, the marker selection is non-empty, and no leak teardown fails.

- [x] **Step 3: Run CI static gates in repository order**

Run:

```bash
uv run --no-sync ruff format --check .
uv run --no-sync ruff check src/ tests/ scripts/
uv run --no-sync mypy src/memo
uv run --no-sync python scripts/quality_gate.py
```

Expected: every command exits 0.

- [x] **Step 4: Run the complete branch-aware PR suite**

Run:

```bash
uv run --no-sync pytest -m "not slow" -n auto --timeout=120 --cov=memo --cov-report=term-missing --cov-report=xml
```

Expected: zero failures, branch-aware total at or above configured `fail_under`, and `coverage.xml` created.

- [x] **Step 5: Run the shipped-default int8 compatibility lane**

Run:

```bash
MEMO_VEC_QUANTIZE=int8 uv run --no-sync pytest -m "not slow and not float32_precision" -n auto --timeout=120
```

Expected: zero failures.

- [x] **Step 6: Exercise scheduled commands locally at bounded scale**

Run:

```bash
uv run --extra test-stability pytest -m "concurrency or resource_hygiene" -n 0 -x --timeout=120 --randomly-seed=20260722 --count=2 --repeat-scope=session --resource-hygiene -q
uv run --extra test-mutation mutmut --help
```

Expected: repeated focused tests pass twice with the fixed seed; mutmut help exits 0. Do not run the full mutation campaign locally unless explicitly monitoring its runtime—the workflow owns that bounded 30-minute job.

- [x] **Step 7: Verify the worktree and commits contain no unrelated files**

Run:

```bash
git status --short
git log --oneline --stat -8
git diff --check HEAD~7..HEAD
```

Expected: pre-existing dirty files remain present but uncommitted unless they were explicitly named by a task; each hardening commit contains only its declared files; diff check is clean.

- [x] **Step 8: Commit any measured-floor documentation adjustment**

If Task 5 selected a floor other than 74, update the numeric example/status in the approved specification and commit only that document:

```bash
git add -f docs/superpowers/specs/2026-07-22-testing-system-hardening-design.md
git commit -m "docs(testing): record branch coverage ratchet"
```

If the floor remained 74, no final documentation commit is needed.
