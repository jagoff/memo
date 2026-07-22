from __future__ import annotations

import re
import tomllib
from pathlib import Path

import yaml

from memo.cli import cli

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"


def _workflow_configs() -> list[tuple[Path, dict[str, object]]]:
    return [
        (path, yaml.safe_load(path.read_text(encoding="utf-8")))
        for path in sorted(WORKFLOWS.glob("*.yml"))
    ]


def test_dependabot_covers_every_shipped_dependency_surface() -> None:
    config = yaml.safe_load((ROOT / ".github" / "dependabot.yml").read_text(encoding="utf-8"))

    assert config["version"] == 2
    updates = config["updates"]
    assert {entry["package-ecosystem"] for entry in updates} == {
        "docker",
        "github-actions",
        "npm",
        "uv",
    }
    assert {(entry["package-ecosystem"], entry["directory"]) for entry in updates} == {
        ("docker", "/"),
        ("github-actions", "/"),
        ("npm", "/editors/vscode"),
        ("uv", "/"),
    }
    assert all(entry["schedule"]["interval"] == "weekly" for entry in updates)


def test_dependabot_respects_mlx_and_test_dependency_caps() -> None:
    config = yaml.safe_load((ROOT / ".github" / "dependabot.yml").read_text(encoding="utf-8"))
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]

    uv_update = next(entry for entry in config["updates"] if entry["package-ecosystem"] == "uv")
    ignored_versions = {
        entry["dependency-name"]: entry.get("versions", []) for entry in uv_update["ignore"]
    }

    assert ">=5.13" in ignored_versions["transformers"]
    assert ">=9" in ignored_versions["pytest"]
    assert any(
        dependency.startswith("transformers<5.13;") for dependency in project["dependencies"]
    )
    assert "pytest>=9.0.3,<10" in project["optional-dependencies"]["dev"]


def test_starlette_test_client_uses_the_supported_http_transport() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]

    assert "httpx2>=2.5,<3" in project["optional-dependencies"]["dev"]


def test_dependency_security_workflow_is_frozen_and_enforcing() -> None:
    workflow = (WORKFLOWS / "dependency-security.yml").read_text(encoding="utf-8")

    assert "uv export --frozen" in workflow
    assert "pip-audit==2.10.1" in workflow
    assert "pip-audit --strict" in workflow
    assert "actions/dependency-review-action@a1d282b36b6f3519aa1f3fc636f609c47dddb294" in workflow
    assert "pull_request:" in workflow
    assert "schedule:" in workflow


def test_slow_test_workflow_installs_test_collection_dependencies() -> None:
    workflow = (WORKFLOWS / "slow-tests.yml").read_text(encoding="utf-8")

    assert "uv sync --frozen --extra dev --extra http" in workflow


def test_every_workflow_job_declares_explicit_permissions() -> None:
    jobs_without_permissions: list[str] = []

    for path, config in _workflow_configs():
        workflow_permissions = config.get("permissions")
        jobs = config.get("jobs", {})
        assert isinstance(jobs, dict)
        for job_name, job in jobs.items():
            assert isinstance(job, dict)
            if workflow_permissions is None and "permissions" not in job:
                jobs_without_permissions.append(f"{path.name}:{job_name}")

    assert jobs_without_permissions == []


def test_checkout_never_persists_github_credentials() -> None:
    insecure_checkouts: list[str] = []

    for path, config in _workflow_configs():
        jobs = config.get("jobs", {})
        assert isinstance(jobs, dict)
        for job_name, job in jobs.items():
            assert isinstance(job, dict)
            steps = job.get("steps", [])
            assert isinstance(steps, list)
            for step in steps:
                assert isinstance(step, dict)
                if str(step.get("uses", "")).startswith("actions/checkout@"):
                    options = step.get("with", {})
                    assert isinstance(options, dict)
                    if options.get("persist-credentials") is not False:
                        insecure_checkouts.append(f"{path.name}:{job_name}")

    assert insecure_checkouts == []


def test_third_party_actions_are_pinned_to_full_commit_shas() -> None:
    unpinned_actions: list[str] = []

    for path, config in _workflow_configs():
        jobs = config.get("jobs", {})
        assert isinstance(jobs, dict)
        for job_name, job in jobs.items():
            assert isinstance(job, dict)
            steps = job.get("steps", [])
            assert isinstance(steps, list)
            for step in steps:
                assert isinstance(step, dict)
                action = str(step.get("uses", ""))
                if not action or action.startswith("./"):
                    continue
                _name, separator, revision = action.rpartition("@")
                if separator != "@" or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
                    unpinned_actions.append(f"{path.name}:{job_name}:{action}")

    assert unpinned_actions == []


def test_macos_model_cache_survives_tooling_only_project_changes() -> None:
    workflow = (WORKFLOWS / "macos-smoke.yml").read_text(encoding="utf-8")

    assert "key: macos-hf-${{ runner.os }}-${{ hashFiles('pyproject.toml') }}" in workflow
    assert "restore-keys: |" in workflow
    assert "macos-hf-${{ runner.os }}-" in workflow


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
    assert "/security/advisories/new" in policy
    assert "private vulnerability report" in policy.lower()


def test_readme_surface_counts_and_top_level_command_inventory_are_exact() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    heading = f"### All {len(cli.commands)} top-level CLI commands"
    start = readme.index(heading)
    block = readme[start : readme.index("</details>", start)]
    documented = set(re.findall(r"`([a-z][a-z0-9-]*)`", block))

    assert documented == set(cli.commands)
    assert "| `full` / `default` | 133 |" in readme
    assert "versus **133 tools / ~15k tokens**" in readme
    assert "default MCP surface is 14 tools, not 133" in readme
