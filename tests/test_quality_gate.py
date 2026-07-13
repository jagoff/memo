from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "quality_gate.py"


def _quality_gate() -> ModuleType:
    assert SCRIPT.is_file(), "scripts/quality_gate.py must exist"
    spec = importlib.util.spec_from_file_location("memo_quality_gate", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_compare_rejects_new_and_increased_complexity() -> None:
    gate = _quality_gate()
    baseline = {
        "complexity": {"src/memo/a.py::existing": 12},
        "broad_exceptions": {},
    }
    current = {
        "complexity": {
            "src/memo/a.py::existing": 13,
            "src/memo/b.py::new_complex": 11,
        },
        "broad_exceptions": {},
    }

    issues = gate.compare(current, baseline)

    assert any("existing" in issue and "13 > baseline 12" in issue for issue in issues)
    assert any("new_complex" in issue and "11 > baseline 0" in issue for issue in issues)


def test_compare_accepts_complexity_reductions_and_deletions() -> None:
    gate = _quality_gate()
    baseline = {
        "complexity": {
            "src/memo/a.py::reduced": 15,
            "src/memo/a.py::deleted": 11,
        },
        "broad_exceptions": {},
    }
    current = {
        "complexity": {"src/memo/a.py::reduced": 12},
        "broad_exceptions": {},
    }

    assert gate.compare(current, baseline) == []


def test_compare_budgets_broad_exceptions_per_file() -> None:
    gate = _quality_gate()
    baseline = {
        "complexity": {},
        "broad_exceptions": {"src/memo/a.py": 1},
    }
    current = {
        "complexity": {},
        "broad_exceptions": {"src/memo/a.py": 2, "src/memo/new.py": 1},
    }

    issues = gate.compare(current, baseline)

    assert any("src/memo/a.py" in issue and "2 > baseline 1" in issue for issue in issues)
    assert any("src/memo/new.py" in issue and "1 > baseline 0" in issue for issue in issues)


def test_parse_ruff_complexity_keys_by_path_and_function(tmp_path: Path) -> None:
    gate = _quality_gate()
    source = tmp_path / "src" / "memo" / "sample.py"
    source.parent.mkdir(parents=True)
    source.write_text("def work():\n    return 1\n", encoding="utf-8")
    payload = [
        {
            "code": "C901",
            "filename": str(source),
            "message": "`work` is too complex (17 > 10)",
        }
    ]

    assert gate.parse_ruff_complexity(payload, tmp_path) == {
        "src/memo/sample.py::work": 17
    }


def test_collect_broad_exceptions_counts_per_file(tmp_path: Path) -> None:
    gate = _quality_gate()
    source = tmp_path / "src" / "memo" / "sample.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "try:\n    pass\nexcept Exception:\n    pass\n"
        "try:\n    pass\nexcept ValueError:\n    pass\n",
        encoding="utf-8",
    )

    assert gate.collect_broad_exceptions(tmp_path) == {"src/memo/sample.py": 1}


def test_quality_baseline_schema_and_configuration() -> None:
    gate = _quality_gate()
    baseline = json.loads((ROOT / "eval" / "quality_baseline.json").read_text(encoding="utf-8"))
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    workflow = (ROOT / ".github" / "workflows" / "test.yml").read_text(encoding="utf-8")

    assert baseline["version"] == 1
    assert set(baseline) == {"version", "complexity", "broad_exceptions"}
    assert "fail_under = 72" in pyproject
    for module in gate.STRICT_MODULES:
        assert module in pyproject
    assert workflow.index("ruff check") < workflow.index("mypy src/memo")
    assert workflow.index("mypy src/memo") < workflow.index("quality_gate.py")
    assert workflow.index("quality_gate.py") < workflow.index("pytest")
