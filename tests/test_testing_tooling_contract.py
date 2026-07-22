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


def test_mutation_scope_is_bounded_to_covered_core_paths() -> None:
    config = _pyproject()["tool"]["mutmut"]
    assert config["mutate_only_covered_lines"] is True
    assert set(config["only_mutate"]) == {
        "src/memo/store/vec_base.py",
        "src/memo/memory/search_scoring_ops.py",
        "src/memo/session.py",
        "src/memo/sqlite_snapshot.py",
    }
    assert config["pytest_add_cli_args_test_selection"] == [
        "tests/test_vector_database_contracts.py",
        "tests/test_store.py",
        "tests/test_search_scoring_ops_unit.py",
        "tests/test_housekeeping_contracts.py",
        "tests/test_session.py",
        "tests/test_sqlite_cleanup.py",
    ]
