from __future__ import annotations

import re
import tomllib
from pathlib import Path

import yaml

from memo.cli import cli

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"


def test_dependabot_covers_every_shipped_dependency_surface() -> None:
    config = yaml.safe_load((ROOT / ".github" / "dependabot.yml").read_text(encoding="utf-8"))

    assert config["version"] == 2
    updates = config["updates"]
    assert {entry["package-ecosystem"] for entry in updates} == {
        "docker",
        "github-actions",
        "uv",
    }
    assert all(entry["directory"] == "/" for entry in updates)
    assert all(entry["schedule"]["interval"] == "weekly" for entry in updates)


def test_dependency_security_workflow_is_frozen_and_enforcing() -> None:
    workflow = (WORKFLOWS / "dependency-security.yml").read_text(encoding="utf-8")

    assert "uv export --frozen" in workflow
    assert "pip-audit==2.10.1" in workflow
    assert "pip-audit --strict" in workflow
    assert "actions/dependency-review-action@a1d282b36b6f3519aa1f3fc636f609c47dddb294" in workflow
    assert "pull_request:" in workflow
    assert "schedule:" in workflow


def test_contract_ci_has_public_gate_and_clearly_optional_private_integration() -> None:
    workflow = (WORKFLOWS / "test.yml").read_text(encoding="utf-8")

    assert "contract-stub-compatibility:" in workflow
    assert "tests/test_contract_stub_compatibility.py" in workflow
    assert "private-contract-integration:" in workflow
    assert "CONTRACTS_TOKEN" in workflow
    assert "optional private integration" in workflow.lower()


def test_security_policy_matches_current_release_and_opt_in_surfaces() -> None:
    policy = (ROOT / "SECURITY.md").read_text(encoding="utf-8")
    version = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"][
        "version"
    ]
    release_line = ".".join(version.split(".")[:2]) + ".x"

    assert release_line in policy
    assert "2.12.x" not in policy
    assert "No credentials" not in policy
    assert "MEMO_SECRET_STORAGE_ENABLED=1" in policy
    assert "Authorization: Bearer" in policy


def test_readme_surface_counts_and_top_level_command_inventory_are_exact() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    heading = f"### All {len(cli.commands)} top-level CLI commands"
    start = readme.index(heading)
    block = readme[start : readme.index("</details>", start)]
    documented = set(re.findall(r"`([a-z][a-z0-9-]*)`", block))

    assert documented == set(cli.commands)
    assert "| `full` / `default` | 131 |" in readme
    assert "versus **131 tools / ~15k tokens**" in readme
    assert "default MCP surface is 14 tools, not 131" in readme
