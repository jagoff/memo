from __future__ import annotations

import json
import re
import subprocess
import tarfile
import tomllib
import zipfile
from pathlib import Path

import yaml

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
    ):
        text = (WORKFLOWS / workflow).read_text(encoding="utf-8")
        assert "uv sync --frozen" in text, workflow

    linux = (WORKFLOWS / "linux-cpu-smoke.yml").read_text(encoding="utf-8")
    assert "uv tool install --python 3.13" in linux
    assert "--find-links https://download.pytorch.org/whl/cpu/torch/" in linux
    assert '"$TOOL_PYTHON"' in linux
    assert "MEMO_TOOL_PYTHON" not in linux


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
        # The curated regression labels the tuner's no-regression gate reads.
        # They must reach the sdist too: a wheel built from the sdist resolves
        # the force-include from these sources, and without them the gate
        # silently fails open on the installed runtime.
        "eval",
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
    assert len(wheel_names) == len(set(wheel_names)), "wheel contains duplicate archive entries"
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


def test_generated_agent_assets_cannot_shadow_force_included_release_assets() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    build_excludes = project["tool"]["hatch"]["build"]["exclude"]
    dockerignore = (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()

    assert "/src/memo/agent_assets/" in build_excludes
    assert "src/memo/agent_assets/" in dockerignore


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

    assert "needs: [verify-release, quality-gate]" in publish
    assert publish.count("id-token: write") == 2
    assert not re.search(r"^permissions:\n(?:  .+\n)*  id-token: write", publish, re.MULTILINE)
    assert "skip-existing: true" not in publish


def test_publish_jobs_run_full_qa_on_the_exact_tagged_sha() -> None:
    quality = (WORKFLOWS / "release-quality.yml").read_text(encoding="utf-8")
    publish = (WORKFLOWS / "publish.yml").read_text(encoding="utf-8")
    docker = (WORKFLOWS / "docker-publish.yml").read_text(encoding="utf-8")
    publish_jobs = yaml.safe_load(publish)["jobs"]
    docker_jobs = yaml.safe_load(docker)["jobs"]

    assert "workflow_call:" in quality
    assert 'test "$(git rev-parse HEAD)" = "$GITHUB_SHA"' in quality
    assert "ruff format --check ." in quality
    assert "ruff check src/ tests/ scripts/" in quality
    assert "mypy src/memo" in quality
    assert "scripts/quality_gate.py" in quality
    assert 'pytest -m "not slow"' in quality
    assert "--cov=memo" in quality
    assert quality.count("uses: ./.github/workflows/linux-cpu-smoke.yml") == 1
    assert quality.count("uses: ./.github/workflows/macos-smoke.yml") == 1

    assert publish_jobs["quality-gate"]["uses"] == "./.github/workflows/release-quality.yml"
    for job_name in ("github-release", "build"):
        assert set(publish_jobs[job_name]["needs"]) == {"verify-release", "quality-gate"}
    assert set(publish_jobs["pypi"]["needs"]) == {"build", "github-release"}
    assert docker_jobs["quality-gate"]["uses"] == "./.github/workflows/release-quality.yml"
    assert docker_jobs["build-and-push"]["needs"] == "quality-gate"


def test_release_quality_and_publish_chain_fail_closed() -> None:
    """Required smoke jobs cannot be skipped while downstream publish still runs."""
    quality_jobs = yaml.safe_load((WORKFLOWS / "release-quality.yml").read_text(encoding="utf-8"))[
        "jobs"
    ]
    publish_jobs = yaml.safe_load((WORKFLOWS / "publish.yml").read_text(encoding="utf-8"))["jobs"]

    assert set(quality_jobs) == {"quality", "linux-cpu-smoke", "macos-mlx-smoke"}
    assert all("if" not in job for job in quality_jobs.values())
    for job_name in ("github-release", "build", "pypi", "mcp-registry"):
        assert "if" not in publish_jobs[job_name]


def test_publish_has_one_automatic_release_trigger() -> None:
    workflow = (WORKFLOWS / "publish.yml").read_text(encoding="utf-8")
    config = yaml.safe_load(workflow)
    triggers = config.get("on", config.get(True))

    assert triggers["push"] == {"tags": ["v*"]}
    assert "release" not in triggers


def test_reusable_linux_smoke_cannot_cancel_sibling_release_gates() -> None:
    workflow = (WORKFLOWS / "linux-cpu-smoke.yml").read_text(encoding="utf-8")
    config = yaml.safe_load(workflow)
    triggers = config.get("on", config.get(True))

    assert "workflow_call" in triggers
    assert "push" not in triggers
    assert "${{ github.workflow }}" in config["concurrency"]["group"]
    assert config["concurrency"]["cancel-in-progress"] is True


def test_runtime_independence_replaces_private_contract_ci() -> None:
    workflow = (WORKFLOWS / "test.yml").read_text(encoding="utf-8")

    assert "  runtime-independence:\n" in workflow
    assert "tests/test_independence.py" in workflow
    assert "memo definitive benchmark" in workflow
    assert "private-contract-integration" not in workflow
    assert "CONTRACTS_TOKEN" not in workflow


def test_docker_publish_requires_a_versioned_tag_from_master() -> None:
    workflow = (WORKFLOWS / "docker-publish.yml").read_text(encoding="utf-8")

    assert "workflow_dispatch:" not in workflow
    assert "fetch-depth: 0" in workflow
    assert 'if [[ "$GITHUB_REF_TYPE" != "tag" ]]' in workflow
    assert 'if [[ "$GITHUB_REF_NAME" != "v${version}" ]]' in workflow
    assert 'git merge-base --is-ancestor "$GITHUB_SHA" origin/master' in workflow


def test_installer_defaults_to_the_versioned_release_without_destructive_pre_uninstall() -> None:
    installer = (ROOT / "install.sh").read_text(encoding="utf-8")
    project_version = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]["version"]

    assert f'DEFAULT_VERSION="{project_version}"' in installer
    assert 'printf \'%s==%s\\n\' "$PYPI_SPEC" "$DEFAULT_VERSION"' in installer
    assert 'tool uninstall "$APP_NAME"' not in installer
    assert 'run_pipx uninstall "$APP_NAME"' not in installer
    assert 'rm -rf "$venvs_dir/$APP_NAME"' not in installer
    assert 'run_pipx install "$spec" --force' in installer


def test_public_installer_commands_pin_the_script_to_the_release_tag() -> None:
    project_version = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]["version"]
    expected = f"raw.githubusercontent.com/jagoff/memo/v{project_version}/install.sh"
    for relative in ("README.md", "docs/install-new-mac.md", "docs/reference.md", "docs/docker.md"):
        text = (ROOT / relative).read_text(encoding="utf-8")
        assert "/memo/master/install.sh" not in text, relative
        if "install.sh" in text:
            assert expected in text, relative


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
