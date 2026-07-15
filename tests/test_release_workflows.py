from __future__ import annotations

import json
import re
import subprocess
import tarfile
import tomllib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"
SHA_REF = re.compile(r"^[^@\s]+@[0-9a-f]{40}$")


def test_all_workflow_actions_use_immutable_commit_shas() -> None:
    mutable: list[str] = []
    for path in sorted(WORKFLOWS.glob("*.yml")):
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            match = re.search(r"\buses:\s*([^\s#]+)", line)
            if match is None or match.group(1).startswith("./"):
                continue
            if SHA_REF.fullmatch(match.group(1)) is None:
                mutable.append(f"{path.name}:{line_number}:{match.group(1)}")
    assert mutable == []


def test_publish_workflow_pins_and_verifies_mcp_publisher() -> None:
    publish = (WORKFLOWS / "publish.yml").read_text(encoding="utf-8")
    checksum = (ROOT / ".github" / "mcp-publisher.sha256").read_text(encoding="utf-8")

    assert "/download/v1.7.9/mcp-publisher_linux_amd64.tar.gz" in publish
    assert "ab128162b0616090b47cf245afe0a23f3ef08936fdce19074f5ba0a4469281ac" in checksum
    assert "sha256sum --check" in publish
    assert "/releases/latest/" not in publish


def test_publish_workflow_fails_when_pypi_never_propagates() -> None:
    publish = (WORKFLOWS / "publish.yml").read_text(encoding="utf-8")

    assert 'echo "available=1" >> "$GITHUB_OUTPUT"' in publish
    assert 'if [[ "$available" != "1" ]]' in publish
    assert "exit 1" in publish


def test_python_ci_uses_committed_frozen_uv_lock() -> None:
    assert (ROOT / "uv.lock").is_file()
    for workflow in (
        "test.yml",
        "slow-tests.yml",
        "macos-smoke.yml",
        "linux-cpu-smoke.yml",
    ):
        text = (WORKFLOWS / workflow).read_text(encoding="utf-8")
        assert "uv sync --frozen" in text, workflow


def test_publish_workflow_builds_from_project_metadata_with_uv() -> None:
    publish = (WORKFLOWS / "publish.yml").read_text(encoding="utf-8")

    assert "uv build --out-dir dist" in publish
    assert "python -m build" not in publish
    assert "uv sync --frozen --no-dev" in publish


def test_built_distributions_only_ship_release_allowlist(tmp_path: Path) -> None:
    subprocess.run(
        ["uv", "build", "--out-dir", str(tmp_path)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    version = project["project"]["version"]
    sdist = tmp_path / f"mlx_memo-{version}.tar.gz"
    wheel = tmp_path / f"mlx_memo-{version}-py3-none-any.whl"
    assert sdist.is_file()
    assert wheel.is_file()

    with tarfile.open(sdist, "r:gz") as archive:
        sdist_names = archive.getnames()
    sdist_prefix = f"mlx_memo-{version}/"
    sdist_roots = {
        name.removeprefix(sdist_prefix).split("/", 1)[0]
        for name in sdist_names
        if name.startswith(sdist_prefix)
    }
    assert sdist_roots <= {
        ".agents",
        ".claude-plugin",
        ".gitignore",
        "CHANGELOG.md",
        "LICENSE",
        "PKG-INFO",
        "README.md",
        "commands",
        "hooks",
        "plugins",
        "pyproject.toml",
        "server.json",
        "skills",
        "src",
        "statusline",
    }

    with zipfile.ZipFile(wheel) as archive:
        wheel_names = archive.namelist()
    assert all(
        name.startswith("memo/") or name.startswith(f"mlx_memo-{version}.dist-info/")
        for name in wheel_names
    )
    forbidden_parts = {
        ".claude",
        ".devin",
        ".env",
        ".superpowers",
        ".worktrees",
        "__pycache__",
        "build",
        "dist",
        "worktrees",
    }
    assert not {
        name
        for name in (*sdist_names, *wheel_names)
        if forbidden_parts.intersection(Path(name).parts)
        or name.endswith((".db", ".envrc", ".pyc", ".sqlite", ".sqlite3"))
    }

    force_include = project["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"]
    assert all((ROOT / source).is_file() for source in force_include)


def test_publish_workflow_gates_tag_version_master_and_release_metadata() -> None:
    publish = (WORKFLOWS / "publish.yml").read_text(encoding="utf-8")

    assert "GITHUB_REF_TYPE" in publish
    assert "GITHUB_REF_NAME" in publish
    assert "v${version}" in publish
    assert 'git merge-base --is-ancestor "$GITHUB_SHA" origin/master' in publish
    assert "memo release check" in publish
    assert "fetch-depth: 0" in publish


def test_publish_oidc_is_limited_to_publish_jobs_after_verification() -> None:
    publish = (WORKFLOWS / "publish.yml").read_text(encoding="utf-8")

    assert "needs: verify-release" in publish
    assert publish.count("id-token: write") == 2
    assert not re.search(r"^permissions:\n(?:  .+\n)*  id-token: write", publish, re.MULTILINE)
    assert "skip-existing: true" not in publish


def test_private_contract_secret_job_never_runs_for_pull_requests() -> None:
    workflow = (WORKFLOWS / "test.yml").read_text(encoding="utf-8")

    assert (
        "  private-contract-integration:\n"
        "    if: github.event_name == 'push' && github.ref == 'refs/heads/master'\n"
    ) in workflow


def test_docker_publish_requires_a_versioned_tag_from_master() -> None:
    workflow = (WORKFLOWS / "docker-publish.yml").read_text(encoding="utf-8")

    assert "workflow_dispatch:" not in workflow
    assert "fetch-depth: 0" in workflow
    assert 'if [[ "$GITHUB_REF_TYPE" != "tag" ]]' in workflow
    assert 'if [[ "$GITHUB_REF_NAME" != "v${version}" ]]' in workflow
    assert 'git merge-base --is-ancestor "$GITHUB_SHA" origin/master' in workflow


def test_slow_test_job_has_a_hard_runtime_limit() -> None:
    slow = (WORKFLOWS / "slow-tests.yml").read_text(encoding="utf-8")

    assert "timeout-minutes:" in slow


def test_release_metadata_and_lock_versions_match() -> None:
    project_version = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]["version"]
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    local_package = next(
        package
        for package in lock["package"]
        if package["name"] == "mlx-memo" and package.get("source") == {"editable": "."}
    )
    server = json.loads((ROOT / "server.json").read_text(encoding="utf-8"))

    assert local_package["version"] == project_version, "run `uv lock` after a version bump"
    assert server["version"] == project_version, "update server.json version"
    assert server["packages"][0]["version"] == project_version, "update server.json package version"
