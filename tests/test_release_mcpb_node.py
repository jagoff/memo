from __future__ import annotations

import json
import re
import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_BOOTSTRAP = _REPO_ROOT / "packaging" / "mcpb-node" / "bootstrap.js"
_NODE_MANIFEST = _REPO_ROOT / "packaging" / "mcpb-node" / "manifest.json"
_PYTHON_MANIFEST = _REPO_ROOT / "packaging" / "mcpb" / "manifest.json"


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_bootstrap_js_syntax() -> None:
    result = subprocess.run(
        ["node", "--check", str(_BOOTSTRAP)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_bootstrap_js_is_zero_dep() -> None:
    source = _BOOTSTRAP.read_text(encoding="utf8")
    requires = re.findall(r"require\(\s*[\"']([^\"']+)[\"']\s*\)", source)
    assert requires, "expected at least one require() call in bootstrap.js"
    for module_name in requires:
        assert module_name.startswith("node:"), (
            f"require({module_name!r}) is not stdlib-prefixed — bootstrap.js must be zero-dep"
        )


def test_bootstrap_reads_pin_from_manifest() -> None:
    source = _BOOTSTRAP.read_text(encoding="utf8")
    assert "manifest.json" in source
    assert ".version" in source


def test_node_manifest_required_fields() -> None:
    manifest = json.loads(_NODE_MANIFEST.read_text(encoding="utf8"))
    assert manifest["privacy_policies"], "privacy_policies must be non-empty (MCPB validation)"
    assert manifest["author"]["email"], "author.email is required (MCPB validation)"
    assert manifest["tools_generated"] is True
    assert manifest["server"]["type"] == "node"
    assert manifest["server"]["entry_point"] == "bootstrap.js"
    assert manifest["server"]["mcp_config"]["command"] == "node"
    assert manifest["server"]["mcp_config"]["args"] == ["${__dirname}/bootstrap.js"]


def test_manifest_versions_in_sync() -> None:
    pyproject = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text(encoding="utf8"))
    project_version = pyproject["project"]["version"]
    python_manifest = json.loads(_PYTHON_MANIFEST.read_text(encoding="utf8"))
    node_manifest = json.loads(_NODE_MANIFEST.read_text(encoding="utf8"))
    assert python_manifest["version"] == project_version, (
        "packaging/mcpb/manifest.json version out of sync with pyproject.toml"
    )
    assert node_manifest["version"] == project_version, (
        "packaging/mcpb-node/manifest.json version out of sync with pyproject.toml"
    )


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_bootstrap_pin_matches_manifest() -> None:
    result = subprocess.run(
        [
            "node",
            "-e",
            f"process.stdout.write(require({str(_BOOTSTRAP)!r}).readPin())",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    manifest = json.loads(_NODE_MANIFEST.read_text(encoding="utf8"))
    assert result.stdout == manifest["version"]
