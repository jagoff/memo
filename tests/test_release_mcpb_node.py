from __future__ import annotations

import json
import re
import shutil
import subprocess
import tomllib
import zipfile
from pathlib import Path

import pytest

from memo.release_mcpb import build_mcpb, build_mcpb_node

_REPO_ROOT = Path(__file__).resolve().parent.parent
_BOOTSTRAP = _REPO_ROOT / "packaging" / "mcpb-node" / "bootstrap.js"
_NODE_MANIFEST = _REPO_ROOT / "packaging" / "mcpb-node" / "manifest.json"
_PYTHON_MANIFEST = _REPO_ROOT / "packaging" / "mcpb" / "manifest.json"
_PYTHON_STUB = _REPO_ROOT / "packaging" / "mcpb" / "server" / "main.py"


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


def test_bootstrap_never_downloads_and_executes_a_remote_shell_script() -> None:
    source = _BOOTSTRAP.read_text(encoding="utf8")
    assert "curl" not in source
    assert "wget" not in source
    assert '["-c"' not in source


def test_python_stub_never_recommends_remote_shell_execution() -> None:
    source = _PYTHON_STUB.read_text(encoding="utf8")
    assert "curl" not in source
    assert "wget" not in source
    assert "| sh" not in source
    assert "docs.astral.sh/uv/getting-started/installation" in source


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_bootstrap_path_lookup_does_not_depend_on_external_which(tmp_path: Path) -> None:
    executable = tmp_path / "memo-mcp"
    executable.write_text("#!/bin/sh\n", encoding="utf8")
    executable.chmod(0o755)
    result = subprocess.run(
        [
            "node",
            "-e",
            f"process.env.PATH={str(tmp_path)!r}; process.stdout.write(require({str(_BOOTSTRAP)!r}).which('memo-mcp') || '')",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == str(executable)


def test_node_manifest_required_fields() -> None:
    manifest = json.loads(_NODE_MANIFEST.read_text(encoding="utf8"))
    assert manifest["privacy_policies"], "privacy_policies must be non-empty (MCPB validation)"
    assert manifest["author"]["email"], "author.email is required (MCPB validation)"
    assert manifest["tools_generated"] is True
    assert manifest["server"]["type"] == "node"
    assert manifest["server"]["entry_point"] == "bootstrap.js"
    assert manifest["server"]["mcp_config"]["command"] == "node"
    # ${__dirname} is official MCPB substitution syntax (anthropics/mcpb MANIFEST.md:
    # "replaced with the absolute path to the extension's directory").
    assert manifest["server"]["mcp_config"]["args"] == ["${__dirname}/bootstrap.js"]


def test_pin_chain_in_sync() -> None:
    """The whole pin chain moves together: pyproject == both manifests == uvx pin.

    The node bundle's install pin IS its manifest version (bootstrap.js readPin),
    so a bump that misses either manifest ships a stale pin. Each assert names
    the exact file to update.
    """
    pyproject = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text(encoding="utf8"))
    project_version = pyproject["project"]["version"]
    python_manifest = json.loads(_PYTHON_MANIFEST.read_text(encoding="utf8"))
    node_manifest = json.loads(_NODE_MANIFEST.read_text(encoding="utf8"))

    assert python_manifest["version"] == project_version, (
        f"packaging/mcpb/manifest.json version {python_manifest['version']!r} != "
        f"pyproject.toml {project_version!r} — update packaging/mcpb/manifest.json"
    )
    assert node_manifest["version"] == project_version, (
        f"packaging/mcpb-node/manifest.json version {node_manifest['version']!r} != "
        f"pyproject.toml {project_version!r} — update packaging/mcpb-node/manifest.json"
    )

    pins = [
        match.group(1)
        for arg in python_manifest["server"]["mcp_config"]["args"]
        if isinstance(arg, str) and (match := re.fullmatch(r"mlx-memo==([^,\s]+)", arg))
    ]
    assert pins, (
        "packaging/mcpb/manifest.json server.mcp_config.args has no mlx-memo pin — "
        "expected an exact 'mlx-memo==X' arg"
    )
    for pin in pins:
        assert pin == project_version, (
            f"packaging/mcpb/manifest.json mlx-memo pin {pin!r} != pyproject.toml "
            f"{project_version!r} — update the mlx-memo==X arg in "
            "packaging/mcpb/manifest.json server.mcp_config.args"
        )


def test_bare_mcpb_pin_closes_linux_cpu_embedder_dependency() -> None:
    """Both Linux-advertising MCPBs install the bare project pin."""
    project = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text(encoding="utf8"))["project"]
    manifests = [
        json.loads(path.read_text(encoding="utf8")) for path in (_PYTHON_MANIFEST, _NODE_MANIFEST)
    ]
    sentence_transformers = [
        dependency
        for dependency in project["dependencies"]
        if dependency.startswith("sentence-transformers>=3.0")
    ]

    assert sentence_transformers == ["sentence-transformers>=3.0; sys_platform == 'linux'"]
    assert project["optional-dependencies"]["cpu"] == []
    assert all("linux" in manifest["compatibility"]["platforms"] for manifest in manifests)


def _scaffold_repo(root: Path) -> Path:
    (root / "packaging" / "mcpb" / "server").mkdir(parents=True)
    (root / "packaging" / "mcpb" / "manifest.json").write_text('{"version": "1.2.3"}\n')
    (root / "packaging" / "mcpb" / "icon.png").write_bytes(b"fake-icon")
    (root / "packaging" / "mcpb" / "server" / "main.py").write_text('print("stub")\n')
    (root / "packaging" / "mcpb-node").mkdir()
    (root / "packaging" / "mcpb-node" / "manifest.json").write_text('{"version": "1.2.3"}\n')
    (root / "packaging" / "mcpb-node" / "bootstrap.js").write_text("// stub\n")
    return root


def test_build_mcpb_node_members_and_determinism(tmp_path: Path) -> None:
    repo = _scaffold_repo(tmp_path)
    built = build_mcpb_node(repo)
    assert built == repo / "packaging" / "memo-node.mcpb"
    with zipfile.ZipFile(built) as archive:
        assert tuple(archive.namelist()) == ("icon.png", "manifest.json", "bootstrap.js")
    first_bytes = built.read_bytes()
    second_bytes = build_mcpb_node(repo).read_bytes()
    assert first_bytes == second_bytes


def test_build_mcpb_node_icon_falls_back_to_python_bundle(tmp_path: Path) -> None:
    repo = _scaffold_repo(tmp_path)  # no icon.png under mcpb-node/
    built = build_mcpb_node(repo)
    with zipfile.ZipFile(built) as archive:
        assert archive.read("icon.png") == b"fake-icon"


def test_build_mcpb_node_prefers_local_member_over_fallback(tmp_path: Path) -> None:
    repo = _scaffold_repo(tmp_path)
    (repo / "packaging" / "mcpb-node" / "icon.png").write_bytes(b"node-icon")
    built = build_mcpb_node(repo)
    with zipfile.ZipFile(built) as archive:
        assert archive.read("icon.png") == b"node-icon"


def test_build_mcpb_node_missing_member_in_both_dirs_raises(tmp_path: Path) -> None:
    repo = _scaffold_repo(tmp_path)
    (repo / "packaging" / "mcpb-node" / "bootstrap.js").unlink()  # absent in mcpb/ too
    with pytest.raises(FileNotFoundError):
        build_mcpb_node(repo)


def test_build_mcpb_python_bundle_unchanged(tmp_path: Path) -> None:
    """Guard the DRY refactor: build_mcpb keeps members, metadata and determinism."""
    repo = _scaffold_repo(tmp_path)
    out = tmp_path / "memo.mcpb"
    build_mcpb(repo, output=out)
    with zipfile.ZipFile(out) as archive:
        assert tuple(archive.namelist()) == ("icon.png", "manifest.json", "server/main.py")
        for info in archive.infolist():
            assert info.date_time == (1980, 1, 1, 0, 0, 0)
            assert info.compress_type == zipfile.ZIP_DEFLATED
            assert info.create_system == 3
            assert info.external_attr == 0o100644 << 16
    first_bytes = out.read_bytes()
    build_mcpb(repo, output=out)
    assert out.read_bytes() == first_bytes


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


def test_bootstrap_executes_mcp_from_exact_pin_instead_of_path() -> None:
    """The checked package pin and the command that runs must be identical."""
    source = _BOOTSTRAP.read_text(encoding="utf-8")
    assert '["tool", "run", "--from", `mlx-memo==${pin}`, "memo-mcp"]' in source
    assert "memoMcpBin" not in source
    assert "spawn(bin" not in source
