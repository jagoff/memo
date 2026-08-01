from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import pytest
from click.testing import CliRunner

from memo.cli_release import (
    _REPO_ROOT,
    _atomic_write_all,
    bump_version,
    plan_release_edits,
    release_check_report,
    release_group,
)
from memo.release_mcpb import build_mcpb


def test_bump_version_levels() -> None:
    assert bump_version("1.2.3", "patch") == "1.2.4"
    assert bump_version("1.2.3", "minor") == "1.3.0"
    assert bump_version("1.2.3", "major") == "2.0.0"


def test_bump_version_rejects_non_semver() -> None:
    with pytest.raises(ValueError):
        bump_version("1.2", "patch")


def _fake_repo(root: Path, version: str) -> Path:
    (root / "pyproject.toml").write_text(
        f'[project]\nname = "mlx-memo"\nversion = "{version}"\n', encoding="utf-8"
    )
    (root / ".claude-plugin").mkdir()
    (root / ".claude-plugin" / "plugin.json").write_text(
        f'{{\n  "name": "memo",\n  "version": "{version}"\n}}\n', encoding="utf-8"
    )
    (root / "plugins" / "memo" / ".codex-plugin").mkdir(parents=True)
    (root / "plugins" / "memo" / ".codex-plugin" / "plugin.json").write_text(
        f'{{\n  "name": "memo",\n  "version": "{version}"\n}}\n', encoding="utf-8"
    )
    (root / "server.json").write_text(
        f'{{\n  "version": "{version}",\n  "packages": [\n    {{\n      "version": "{version}"\n    }}\n  ]\n}}\n',
        encoding="utf-8",
    )
    (root / "install.sh").write_text(
        f'#!/usr/bin/env bash\nDEFAULT_VERSION="{version}"\n', encoding="utf-8"
    )
    (root / "packaging" / "mcpb").mkdir(parents=True)
    (root / "packaging" / "mcpb" / "manifest.json").write_text(
        f'{{\n  "manifest_version": "0.3",\n  "version": "{version}",\n'
        f'  "server": {{\n    "mcp_config": {{\n'
        f'      "args": ["--from", "mlx-memo=={version}", "memo-mcp"]\n'
        f"    }}\n  }}\n}}\n",
        encoding="utf-8",
    )
    (root / "packaging" / "mcpb-node").mkdir(parents=True)
    (root / "packaging" / "mcpb-node" / "manifest.json").write_text(
        f'{{\n  "manifest_version": "0.3",\n  "version": "{version}",\n'
        f'  "server": {{\n    "type": "node",\n    "entry_point": "bootstrap.js"\n  }}\n}}\n',
        encoding="utf-8",
    )
    (root / "packaging" / "mcpb-node" / "bootstrap.js").write_text(
        "// test bootstrap\n",
        encoding="utf-8",
    )
    (root / "packaging" / "mcpb" / "icon.png").write_bytes(b"fake-icon")
    (root / "packaging" / "mcpb" / "server").mkdir()
    (root / "packaging" / "mcpb" / "server" / "main.py").write_text(
        "from memo.server import main\nmain()\n",
        encoding="utf-8",
    )
    with zipfile.ZipFile(root / "packaging" / "memo.mcpb", "w") as archive:
        for member in ("icon.png", "manifest.json", "server/main.py"):
            archive.write(root / "packaging" / "mcpb" / member, arcname=member)
    with zipfile.ZipFile(root / "packaging" / "memo-node.mcpb", "w") as archive:
        archive.write(root / "packaging" / "mcpb" / "icon.png", arcname="icon.png")
        for member in ("manifest.json", "bootstrap.js"):
            archive.write(root / "packaging" / "mcpb-node" / member, arcname=member)
    (root / "CHANGELOG.md").write_text(
        f"# Changelog\n\n## [Unreleased]\n\n## [{version}] - 2026-01-01\n\n- prior\n",
        encoding="utf-8",
    )
    return root


def test_release_mcpb_build_is_reproducible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _fake_repo(tmp_path, "1.2.3")
    monkeypatch.setenv("MEMO_DEV_REPO", str(repo))

    first = CliRunner().invoke(release_group, ["mcpb"])
    first_hash = hashlib.sha256((repo / "packaging" / "memo.mcpb").read_bytes()).hexdigest()
    second = CliRunner().invoke(release_group, ["mcpb"])
    second_hash = hashlib.sha256((repo / "packaging" / "memo.mcpb").read_bytes()).hexdigest()

    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    assert first_hash == second_hash


def test_release_mcpb_builds_python_and_node_archives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Longer than Rich's default captured-console width: artifact filenames
    # must remain intact even when their absolute paths are very long.
    long_root = tmp_path / ("long-release-path-" * 6)
    long_root.mkdir()
    repo = _fake_repo(long_root, "1.2.3")
    (repo / "packaging" / "memo.mcpb").unlink()
    (repo / "packaging" / "memo-node.mcpb").unlink()
    monkeypatch.setenv("MEMO_DEV_REPO", str(repo))

    result = CliRunner().invoke(release_group, ["mcpb"])

    assert result.exit_code == 0, result.output
    assert (repo / "packaging" / "memo.mcpb").is_file()
    assert (repo / "packaging" / "memo-node.mcpb").is_file()
    assert "memo.mcpb" in result.output
    assert "memo-node.mcpb" in result.output


def test_release_mcpb_ignores_predictable_temporary_symlink(tmp_path: Path) -> None:
    repo = _fake_repo(tmp_path, "1.2.3")
    destination = repo / "packaging" / "memo.mcpb"
    destination.unlink()
    victim = tmp_path / "victim.txt"
    victim.write_text("do not overwrite", encoding="utf-8")
    destination.with_name(f"{destination.name}.tmp").symlink_to(victim)

    build_mcpb(repo, destination)

    assert victim.read_text(encoding="utf-8") == "do not overwrite"
    assert destination.is_file()
    assert not destination.is_symlink()


def test_release_mcpb_rejects_symlinked_source_member(tmp_path: Path) -> None:
    repo = _fake_repo(tmp_path, "1.2.3")
    source = repo / "packaging" / "mcpb" / "server" / "main.py"
    outside = tmp_path / "outside.py"
    outside.write_text("raise SystemExit(9)\n", encoding="utf-8")
    source.unlink()
    source.symlink_to(outside)

    with pytest.raises(ValueError, match="symlink"):
        build_mcpb(repo)


def test_atomic_write_all_ignores_predictable_temporary_symlink(tmp_path: Path) -> None:
    target = tmp_path / "pyproject.toml"
    target.write_text("old", encoding="utf-8")
    victim = tmp_path / "victim.txt"
    victim.write_text("do not overwrite", encoding="utf-8")
    target.with_name(f"{target.name}.tmp").symlink_to(victim)

    _atomic_write_all({target: "new"})

    assert victim.read_text(encoding="utf-8") == "do not overwrite"
    assert target.read_text(encoding="utf-8") == "new"
    assert not target.is_symlink()


def test_atomic_write_all_rejects_symlinked_target(tmp_path: Path) -> None:
    victim = tmp_path / "victim.txt"
    victim.write_text("do not overwrite", encoding="utf-8")
    target = tmp_path / "pyproject.toml"
    target.symlink_to(victim)

    with pytest.raises(ValueError, match="regular file"):
        _atomic_write_all({target: "new"})

    assert victim.read_text(encoding="utf-8") == "do not overwrite"


def test_release_check_report_rejects_archived_member_drift(tmp_path: Path) -> None:
    repo = _fake_repo(tmp_path, "1.2.3")
    (repo / "packaging" / "mcpb" / "server" / "main.py").write_text(
        "raise SystemExit(9)\n",
        encoding="utf-8",
    )

    report = release_check_report(repo)

    assert report.ok is False
    assert any("server/main.py" in issue for issue in report.issues)


def test_release_check_report_rejects_node_archived_member_drift(tmp_path: Path) -> None:
    repo = _fake_repo(tmp_path, "1.2.3")
    (repo / "packaging" / "mcpb-node" / "manifest.json").write_text(
        '{"manifest_version":"0.3","version":"1.2.4"}\n',
        encoding="utf-8",
    )

    report = release_check_report(repo)

    assert report.ok is False
    assert any(
        "packaging/memo-node.mcpb" in issue and "manifest.json" in issue for issue in report.issues
    )


def test_plan_release_edits_bumps_mcpb_manifest(tmp_path: Path) -> None:
    repo = _fake_repo(tmp_path, "1.2.3")
    edits = plan_release_edits(repo, "1.2.3", "1.2.4", "2026-06-25")

    manifest = edits[repo / "packaging" / "mcpb" / "manifest.json"]
    assert '"version": "1.2.4"' in manifest
    assert "mlx-memo==1.2.4" in manifest
    # manifest_version is a schema version, not a release version.
    assert '"manifest_version": "0.3"' in manifest


def test_release_check_report_rejects_node_manifest_drift(tmp_path: Path) -> None:
    repo = _fake_repo(tmp_path, "1.2.3")
    (repo / "packaging" / "mcpb-node" / "manifest.json").write_text(
        '{\n  "manifest_version": "0.3",\n  "version": "1.0.0"\n}\n',
        encoding="utf-8",
    )

    report = release_check_report(repo)

    assert report.ok is False
    assert any("packaging/mcpb-node/manifest.json" in issue for issue in report.issues)


def test_release_check_report_tolerates_missing_node_manifest(tmp_path: Path) -> None:
    repo = _fake_repo(tmp_path, "1.2.3")
    (repo / "packaging" / "mcpb-node" / "manifest.json").unlink()

    report = release_check_report(repo)

    assert not any("mcpb-node" in issue for issue in report.issues)


def test_plan_release_edits_bumps_mcpb_node_manifest(tmp_path: Path) -> None:
    repo = _fake_repo(tmp_path, "1.2.3")
    edits = plan_release_edits(repo, "1.2.3", "1.2.4", "2026-06-25")

    manifest = edits[repo / "packaging" / "mcpb-node" / "manifest.json"]
    assert '"version": "1.2.4"' in manifest
    # manifest_version is a schema version, not a release version.
    assert '"manifest_version": "0.3"' in manifest


def test_plan_release_edits_tolerates_missing_mcpb_node_manifest(tmp_path: Path) -> None:
    repo = _fake_repo(tmp_path, "1.2.3")
    (repo / "packaging" / "mcpb-node" / "manifest.json").unlink()

    edits = plan_release_edits(repo, "1.2.3", "1.2.4", "2026-06-25")

    assert repo / "packaging" / "mcpb-node" / "manifest.json" not in edits
    assert 'version = "1.2.4"' in edits[repo / "pyproject.toml"]


def test_plan_release_edits_tolerates_missing_mcpb_manifest(tmp_path: Path) -> None:
    repo = _fake_repo(tmp_path, "1.2.3")
    (repo / "packaging" / "mcpb" / "manifest.json").unlink()

    edits = plan_release_edits(repo, "1.2.3", "1.2.4", "2026-06-25")

    assert repo / "packaging" / "mcpb" / "manifest.json" not in edits
    assert 'version = "1.2.4"' in edits[repo / "pyproject.toml"]


def test_plan_release_edits_syncs_versioned_release_files(tmp_path: Path) -> None:
    repo = _fake_repo(tmp_path, "1.2.3")
    edits = plan_release_edits(repo, "1.2.3", "1.2.4", "2026-06-25")

    assert 'version = "1.2.4"' in edits[repo / "pyproject.toml"]
    assert '"version": "1.2.4"' in edits[repo / ".claude-plugin" / "plugin.json"]
    assert (
        '"version": "1.2.4"' in edits[repo / "plugins" / "memo" / ".codex-plugin" / "plugin.json"]
    )
    # server.json has TWO version occurrences — both must move.
    assert edits[repo / "server.json"].count('"version": "1.2.4"') == 2
    assert edits[repo / "server.json"].count('"version": "1.2.3"') == 0
    assert 'DEFAULT_VERSION="1.2.4"' in edits[repo / "install.sh"]
    # CHANGELOG gains a new section right under Unreleased, above the old one.
    cl = edits[repo / "CHANGELOG.md"]
    assert "## [1.2.4] - 2026-06-25" in cl
    assert cl.index("## [1.2.4]") < cl.index("## [1.2.3]")


def test_release_check_report_rejects_installer_default_version_drift(tmp_path: Path) -> None:
    repo = _fake_repo(tmp_path, "1.2.3")
    (repo / "install.sh").write_text(
        '#!/usr/bin/env bash\nDEFAULT_VERSION="1.2.2"\n', encoding="utf-8"
    )

    report = release_check_report(repo)

    assert report.ok is False
    assert any("install.sh install pin" in issue for issue in report.issues)


def test_plan_release_edits_realigns_drifted_manifest(tmp_path: Path) -> None:
    repo = _fake_repo(tmp_path, "1.2.3")
    codex_plugin = repo / "plugins" / "memo" / ".codex-plugin" / "plugin.json"
    codex_plugin.write_text(
        '{\n  "name": "memo",\n  "version": "1.0.2"\n}\n',
        encoding="utf-8",
    )

    edits = plan_release_edits(repo, "1.2.3", "1.2.4", "2026-06-25")

    assert '"version": "1.2.4"' in edits[codex_plugin]
    assert '"version": "1.0.2"' not in edits[codex_plugin]


def test_release_bump_dry_run_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _fake_repo(tmp_path, "1.2.3")
    monkeypatch.setenv("MEMO_DEV_REPO", str(repo))
    result = CliRunner().invoke(release_group, ["bump", "patch", "--dry-run"])
    assert result.exit_code == 0
    assert 'version = "1.2.3"' in (repo / "pyproject.toml").read_text()


def test_atomic_write_all_writes_and_leaves_no_temps(tmp_path: Path) -> None:
    a, b = tmp_path / "a.txt", tmp_path / "b.txt"
    _atomic_write_all({a: "AAA", b: "BBB"})
    assert a.read_text() == "AAA"
    assert b.read_text() == "BBB"
    # No staged *.tmp siblings survive a successful write.
    assert list(tmp_path.glob("*.tmp")) == []


def test_atomic_write_all_aborts_without_touching_files(tmp_path: Path) -> None:
    good = tmp_path / "good.txt"
    good.write_text("OLD", encoding="utf-8")
    # A target whose parent dir is missing makes staging raise mid-batch.
    bad = tmp_path / "missing_dir" / "bad.txt"
    with pytest.raises(OSError):
        _atomic_write_all({good: "NEW", bad: "NEW"})
    # The pre-existing file is untouched and no temp leaks behind.
    assert good.read_text() == "OLD"
    assert list(tmp_path.glob("*.tmp")) == []


def test_release_bump_writes_all_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _fake_repo(tmp_path, "1.2.3")
    monkeypatch.setenv("MEMO_DEV_REPO", str(repo))
    result = CliRunner().invoke(release_group, ["bump", "minor", "--date", "2026-06-25"])
    assert result.exit_code == 0, result.output
    assert 'version = "1.3.0"' in (repo / "pyproject.toml").read_text()
    assert (
        '"version": "1.3.0"'
        in (repo / "plugins" / "memo" / ".codex-plugin" / "plugin.json").read_text()
    )
    assert (repo / "server.json").read_text().count('"version": "1.3.0"') == 2
    assert "## [1.3.0] - 2026-06-25" in (repo / "CHANGELOG.md").read_text()


def test_release_sync_realigns_files_to_pyproject(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _fake_repo(tmp_path, "1.2.3")
    codex_plugin = repo / "plugins" / "memo" / ".codex-plugin" / "plugin.json"
    codex_plugin.write_text(
        '{\n  "name": "memo",\n  "version": "1.0.2"\n}\n',
        encoding="utf-8",
    )
    (repo / "server.json").write_text(
        '{\n  "version": "1.0.2",\n  "packages": [{"version": "1.0.2"}]\n}\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("MEMO_DEV_REPO", str(repo))

    result = CliRunner().invoke(release_group, ["sync"])

    assert result.exit_code == 0, result.output
    assert '"version": "1.2.3"' in codex_plugin.read_text()
    assert (repo / "server.json").read_text().count('"version": "1.2.3"') == 2


def test_release_check_report_passes_synced_release(tmp_path: Path) -> None:
    repo = _fake_repo(tmp_path, "1.2.3")

    report = release_check_report(repo)

    assert report.ok is True
    assert report.version == "1.2.3"
    assert report.versions["plugins/memo/.codex-plugin/plugin.json"] == "1.2.3"
    assert report.issues == []


def test_release_check_real_checkout_versions_are_synced() -> None:
    if not (_REPO_ROOT / "pyproject.toml").is_file():
        pytest.skip("not running from a source checkout")

    report = release_check_report(_REPO_ROOT)

    # This working tree is shared by concurrent sessions (see CLAUDE.md): a
    # bump in flight legitimately leaves the TODO placeholder — skip, don't red.
    if any("TODO" in issue for issue in report.issues):
        pytest.skip("release bump in flight (CHANGELOG TODO placeholder)")
    assert report.ok is True, report.issues
    assert report.versions["plugins/memo/.codex-plugin/plugin.json"] == report.version


def test_release_check_report_rejects_codex_plugin_bundle_drift(tmp_path: Path) -> None:
    repo = _fake_repo(tmp_path, "1.2.3")
    (repo / "plugins" / "memo" / ".codex-plugin" / "plugin.json").write_text(
        '{\n  "name": "memo",\n  "version": "1.0.2"\n}\n',
        encoding="utf-8",
    )

    report = release_check_report(repo)

    assert report.ok is False
    assert any("plugins/memo/.codex-plugin/plugin.json" in issue for issue in report.issues)


def test_release_check_report_rejects_mcpb_manifest_drift(tmp_path: Path) -> None:
    repo = _fake_repo(tmp_path, "1.2.3")
    manifest = repo / "packaging" / "mcpb" / "manifest.json"
    manifest.write_text(
        '{"version":"1.2.2","server":{"mcp_config":{"args":["--from","mlx-memo>=1.2.2","memo-mcp"]}}}\n',
        encoding="utf-8",
    )

    report = release_check_report(repo)

    assert report.ok is False
    assert any("packaging/mcpb/manifest.json" in issue for issue in report.issues)


def test_release_check_report_rejects_non_exact_mcpb_dependency(tmp_path: Path) -> None:
    repo = _fake_repo(tmp_path, "1.2.3")
    manifest = repo / "packaging" / "mcpb" / "manifest.json"
    manifest.write_text(
        manifest.read_text(encoding="utf-8").replace("mlx-memo==1.2.3", "mlx-memo>=1.2.3"),
        encoding="utf-8",
    )

    report = release_check_report(repo)

    assert report.ok is False
    assert any("exact mlx-memo==" in issue for issue in report.issues)


def test_release_check_report_rejects_changelog_todo(tmp_path: Path) -> None:
    repo = _fake_repo(tmp_path, "1.2.3")
    (repo / "CHANGELOG.md").write_text(
        "# Changelog\n\n## [Unreleased]\n\n## [1.2.3] - 2026-01-01\n\n- TODO: fill this\n",
        encoding="utf-8",
    )

    report = release_check_report(repo)

    assert report.ok is False
    assert any("CHANGELOG" in issue and "TODO" in issue for issue in report.issues)


def test_release_check_cli_json_fails_on_version_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _fake_repo(tmp_path, "1.2.3")
    (repo / "server.json").write_text(
        '{\n  "version": "1.2.9",\n  "packages": [{"version": "1.2.9"}]\n}\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("MEMO_DEV_REPO", str(repo))

    result = CliRunner().invoke(release_group, ["check", "--json"])

    assert result.exit_code == 1
    assert '"ok": false' in result.output
    assert "server.json" in result.output


def test_release_check_reports_homebrew_formula_drift(tmp_path: Path) -> None:
    repo = _fake_repo(tmp_path, "1.2.3")
    formula = repo / "docs" / "homebrew" / "mlx-memo.rb"
    formula.parent.mkdir(parents=True)
    formula.write_text(
        'url "https://files.pythonhosted.org/packages/source/m/mlx-memo/mlx_memo-1.2.2.tar.gz"\n',
        encoding="utf-8",
    )

    report = release_check_report(repo)
    strict = release_check_report(repo, strict_docs=True)

    assert report.ok is True
    assert any("mlx-memo.rb" in warning for warning in report.warnings)
    assert strict.ok is False
    assert any("mlx-memo.rb" in issue for issue in strict.issues)


def test_release_check_reports_homebrew_formula_audit_drift(tmp_path: Path) -> None:
    repo = _fake_repo(tmp_path, "1.2.3")
    formula = repo / "docs" / "homebrew" / "mlx-memo.rb"
    formula.parent.mkdir(parents=True)
    formula.write_text(
        "\n".join(
            [
                'url "https://github.com/jagoff/memo/archive/refs/tags/v1.2.3.tar.gz"',
                "depends_on :macos",
                "depends_on arch: :arm64",
                "",
            ]
        ),
        encoding="utf-8",
    )

    report = release_check_report(repo)
    strict = release_check_report(repo, strict_docs=True)

    assert report.ok is True
    assert any("PyPI source distribution" in warning for warning in report.warnings)
    assert any("dependency order" in warning for warning in report.warnings)
    assert strict.ok is False
    assert any("PyPI source distribution" in issue for issue in strict.issues)
    assert any("dependency order" in issue for issue in strict.issues)


def test_release_check_reports_homebrew_formula_missing_virtualenv_mixin(tmp_path: Path) -> None:
    repo = _fake_repo(tmp_path, "1.2.3")
    formula = repo / "docs" / "homebrew" / "mlx-memo.rb"
    formula.parent.mkdir(parents=True)
    formula.write_text(
        "\n".join(
            [
                "class MlxMemo < Formula",
                '  url "https://files.pythonhosted.org/packages/aa/bb/mlx_memo-1.2.3.tar.gz"',
                "  depends_on arch: :arm64",
                "  depends_on :macos",
                "  def install",
                '    virtualenv_create(libexec, "python3.13")',
                "  end",
                "end",
                "",
            ]
        ),
        encoding="utf-8",
    )

    report = release_check_report(repo)
    strict = release_check_report(repo, strict_docs=True)

    assert report.ok is True
    assert any("Language::Python::Virtualenv" in warning for warning in report.warnings)
    assert strict.ok is False
    assert any("Language::Python::Virtualenv" in issue for issue in strict.issues)


def test_release_check_reports_homebrew_formula_no_deps_helper(tmp_path: Path) -> None:
    repo = _fake_repo(tmp_path, "1.2.3")
    formula = repo / "docs" / "homebrew" / "mlx-memo.rb"
    formula.parent.mkdir(parents=True)
    formula.write_text(
        "\n".join(
            [
                "class MlxMemo < Formula",
                "  include Language::Python::Virtualenv",
                '  url "https://files.pythonhosted.org/packages/aa/bb/mlx_memo-1.2.3.tar.gz"',
                "  depends_on arch: :arm64",
                "  depends_on :macos",
                "  def install",
                '    venv = virtualenv_create(libexec, "python3.13")',
                "    venv.pip_install_and_link buildpath",
                "  end",
                "end",
                "",
            ]
        ),
        encoding="utf-8",
    )

    report = release_check_report(repo)
    strict = release_check_report(repo, strict_docs=True)

    assert report.ok is True
    assert any("--no-deps" in warning for warning in report.warnings)
    assert strict.ok is False
    assert any("--no-deps" in issue for issue in strict.issues)


def test_release_check_reports_homebrew_formula_missing_preserve_rpath(tmp_path: Path) -> None:
    repo = _fake_repo(tmp_path, "1.2.3")
    formula = repo / "docs" / "homebrew" / "mlx-memo.rb"
    formula.parent.mkdir(parents=True)
    formula.write_text(
        "\n".join(
            [
                "class MlxMemo < Formula",
                "  include Language::Python::Virtualenv",
                '  url "https://files.pythonhosted.org/packages/aa/bb/mlx_memo-1.2.3.tar.gz"',
                "  depends_on arch: :arm64",
                "  depends_on :macos",
                "  def install",
                '    virtualenv_create(libexec, "python3.13")',
                '    system "python3.13", "-m", "pip", "--python=#{libexec}/bin/python",',
                '           "install", buildpath',
                "  end",
                "end",
                "",
            ]
        ),
        encoding="utf-8",
    )

    report = release_check_report(repo)
    strict = release_check_report(repo, strict_docs=True)

    assert report.ok is True
    assert any("preserve_rpath" in warning for warning in report.warnings)
    assert strict.ok is False
    assert any("preserve_rpath" in issue for issue in strict.issues)


def test_release_check_reports_blocking_homebrew_mcp_test(tmp_path: Path) -> None:
    repo = _fake_repo(tmp_path, "1.2.3")
    formula = repo / "docs" / "homebrew" / "mlx-memo.rb"
    formula.parent.mkdir(parents=True)
    formula.write_text(
        "\n".join(
            [
                "class MlxMemo < Formula",
                '  url "https://files.pythonhosted.org/packages/aa/bb/mlx_memo-1.2.3.tar.gz"',
                "  depends_on arch: :arm64",
                "  depends_on :macos",
                "  test do",
                '    system bin/"memo-mcp", "--help"',
                "  end",
                "end",
                "",
            ]
        ),
        encoding="utf-8",
    )

    report = release_check_report(repo)
    strict = release_check_report(repo, strict_docs=True)

    assert report.ok is True
    assert any("blocks waiting for input" in warning for warning in report.warnings)
    assert strict.ok is False
    assert any("blocks waiting for input" in issue for issue in strict.issues)
