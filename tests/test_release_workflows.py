from __future__ import annotations

import re
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
