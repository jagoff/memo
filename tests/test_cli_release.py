from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from memo.cli_release import (
    _atomic_write_all,
    bump_version,
    plan_release_edits,
    release_check_report,
    release_group,
)


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
    (root / "server.json").write_text(
        f'{{\n  "version": "{version}",\n  "packages": [\n    {{\n      "version": "{version}"\n    }}\n  ]\n}}\n',
        encoding="utf-8",
    )
    (root / "CHANGELOG.md").write_text(
        f"# Changelog\n\n## [Unreleased]\n\n## [{version}] - 2026-01-01\n\n- prior\n",
        encoding="utf-8",
    )
    return root


def test_plan_release_edits_syncs_four_files(tmp_path: Path) -> None:
    repo = _fake_repo(tmp_path, "1.2.3")
    edits = plan_release_edits(repo, "1.2.3", "1.2.4", "2026-06-25")

    assert 'version = "1.2.4"' in edits[repo / "pyproject.toml"]
    assert '"version": "1.2.4"' in edits[repo / ".claude-plugin" / "plugin.json"]
    # server.json has TWO version occurrences — both must move.
    assert edits[repo / "server.json"].count('"version": "1.2.4"') == 2
    assert edits[repo / "server.json"].count('"version": "1.2.3"') == 0
    # CHANGELOG gains a new section right under Unreleased, above the old one.
    cl = edits[repo / "CHANGELOG.md"]
    assert "## [1.2.4] - 2026-06-25" in cl
    assert cl.index("## [1.2.4]") < cl.index("## [1.2.3]")


def test_release_bump_dry_run_writes_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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
    assert (repo / "server.json").read_text().count('"version": "1.3.0"') == 2
    assert "## [1.3.0] - 2026-06-25" in (repo / "CHANGELOG.md").read_text()


def test_release_check_report_passes_synced_release(tmp_path: Path) -> None:
    repo = _fake_repo(tmp_path, "1.2.3")

    report = release_check_report(repo)

    assert report.ok is True
    assert report.version == "1.2.3"
    assert report.issues == []


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
