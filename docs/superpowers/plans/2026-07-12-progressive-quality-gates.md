# Progressive Quality Gates Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent complexity, broad-exception, typing, and coverage debt from increasing while allowing incremental cleanup.

**Architecture:** A standalone checker compares current measurements to a versioned JSON baseline. Ruff supplies C901 metrics, AST supplies per-file broad exception counts, mypy strictness is expanded for touched modules, and CI runs the checker before pytest.

**Tech Stack:** Python 3.13, Ruff JSON output, AST, JSON, mypy, coverage.py, GitHub Actions, pytest.

## Global Constraints

- Existing debt may remain but cannot increase.
- Baseline mutation requires explicit `--update`.
- Coverage floor is exactly 72%.
- CI order is ruff, mypy, quality gate, pytest.

---

### Task 1: Quality measurement and comparison

**Files:**
- Create: `scripts/quality_gate.py`
- Create: `tests/test_quality_gate.py`

**Interfaces:**
- Produces: `collect_complexity(root: Path) -> dict[str, int]` keyed by `path::function`.
- Produces: `collect_broad_exceptions(root: Path) -> dict[str, int]` keyed by source-relative path.
- Produces: `compare(current: dict[str, dict[str, int]], baseline: dict[str, dict[str, int]]) -> list[str]`.

- [ ] **Step 1: Add failing unit tests for new/increased debt, accepted reductions/deletions, and per-file exception budgets**

```python
assert compare({"complexity": {"a.py::f": 12}}, {"complexity": {}})
assert compare({"complexity": {"a.py::f": 13}}, {"complexity": {"a.py::f": 12}})
assert compare({"complexity": {"a.py::f": 11}}, {"complexity": {"a.py::f": 12}}) == []
assert compare({"broad_exceptions": {"a.py": 2}}, {"broad_exceptions": {"a.py": 1}})
```

- [ ] **Step 2: Run `uv run --no-sync pytest tests/test_quality_gate.py -v` and observe missing module failure**

- [ ] **Step 3: Implement Ruff JSON parsing, AST counting, comparison, CLI diagnostics, and explicit update mode**

```python
def compare(current: Metrics, baseline: Metrics) -> list[str]:
    issues = []
    for metric in ("complexity", "broad_exceptions"):
        for key, value in current[metric].items():
            allowed = baseline[metric].get(key, 0)
            if value > allowed:
                issues.append(f"{metric} {key}: current {value} > baseline {allowed}")
    return issues
```

- [ ] **Step 4: Run checker unit tests and lint the new script**

Run: `uv run --no-sync ruff check scripts/quality_gate.py tests/test_quality_gate.py`

- [ ] **Step 5: Commit checker and tests**

```bash
git add scripts/quality_gate.py tests/test_quality_gate.py
git commit -m "feat: add progressive quality checker"
```

### Task 2: Baseline, typing, coverage, and CI

**Files:**
- Create: `eval/quality_baseline.json`
- Modify: `pyproject.toml`
- Modify: `.github/workflows/test.yml`
- Test: `tests/test_quality_gate.py`

**Interfaces:**
- Consumes: `scripts/quality_gate.py --update` output.
- Produces: CI quality budget and strict typing for `memo.cli_release`, `memo.runtime.autoupdate`, `memo.server`, `memo.surface`, and `memo.capture_core`.

- [ ] **Step 1: Add failing configuration tests for baseline schema, 72% floor, strict modules, and CI order**

```python
assert baseline["version"] == 1
assert coverage["fail_under"] == 72
assert required_modules <= strict_modules
assert steps.index("ruff") < steps.index("mypy") < steps.index("quality") < steps.index("pytest")
```

- [ ] **Step 2: Run the configuration tests and observe missing baseline and old floor failures**

- [ ] **Step 3: Generate the baseline, expand mypy overrides, raise coverage, and insert CI quality checking**

Run: `uv run --no-sync python scripts/quality_gate.py --update`

- [ ] **Step 4: Fix all strict-module mypy errors and run the local gate sequence**

Run: `uv run --no-sync ruff check src/ tests/ scripts/`

Run: `uv run --no-sync mypy src/memo`

Run: `uv run --no-sync python scripts/quality_gate.py`

- [ ] **Step 5: Commit the ratchets**

```bash
git add eval/quality_baseline.json pyproject.toml .github/workflows/test.yml tests/test_quality_gate.py src/memo
git commit -m "ci: ratchet code quality budgets"
```

### Task 3: Full verification

**Files:**
- Verify: all files changed by the four implementation plans.

**Interfaces:**
- Produces: release-ready evidence for lint, types, tests, packages, and MCPB parity.

- [ ] **Step 1: Run ruff, mypy, and quality gate in CI order**
- [ ] **Step 2: Run non-slow tests with 72% coverage**
- [ ] **Step 3: Run slow/MLX tests serially**
- [ ] **Step 4: Build wheel/sdist, run `memo release check`, and compare two MCPB hashes**
- [ ] **Step 5: Inspect final diff, confirm user changes remain preserved, and record verification results**
