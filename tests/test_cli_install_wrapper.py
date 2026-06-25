"""`memo install-shell-wrapper` — print mode + idempotent --write.

The shell snippet itself isn't unit-testable from Python (would require
a zsh harness), so we cover the install-side behaviour: snippet
content shape, file creation, idempotent rc append, and force/conflict
handling.

`$HOME` is monkeypatched to `tmp_path` so the test never touches the
real `~/.zshrc`. `Path.home()` reads from `$HOME` on POSIX, so the
patch is sufficient — no need to mock `Path.home()` directly.
"""

from __future__ import annotations

from click.testing import CliRunner

from memo.cli import cli


def test_install_shell_wrapper_print_default():
    """No flags → emit snippet, no fs write."""
    runner = CliRunner()
    result = runner.invoke(cli, ["install-shell-wrapper"])
    assert result.exit_code == 0
    assert "function claude()" in result.output
    assert "command claude" in result.output
    assert "MEMO_CLAUDE_EXTRA_ARGS" in result.output


def test_install_shell_wrapper_explicit_print():
    runner = CliRunner()
    result = runner.invoke(cli, ["install-shell-wrapper", "--print"])
    assert result.exit_code == 0
    assert "function claude()" in result.output


def test_install_shell_wrapper_write_creates_files(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    runner = CliRunner()
    result = runner.invoke(cli, ["install-shell-wrapper", "--write"])
    assert result.exit_code == 0, result.output

    wrapper = tmp_path / ".zsh" / "memo-wrapper.zsh"
    assert wrapper.is_file()
    assert "function claude()" in wrapper.read_text(encoding="utf-8")

    rc = tmp_path / ".zshrc"
    assert rc.is_file()
    assert f"source {wrapper}" in rc.read_text(encoding="utf-8")


def test_install_shell_wrapper_idempotent_write(tmp_path, monkeypatch):
    """Calling --write twice produces exactly one source line."""
    monkeypatch.setenv("HOME", str(tmp_path))
    runner = CliRunner()
    runner.invoke(cli, ["install-shell-wrapper", "--write"])
    runner.invoke(cli, ["install-shell-wrapper", "--write"])

    rc = tmp_path / ".zshrc"
    body = rc.read_text(encoding="utf-8")
    occurrences = body.count("source")
    # The snippet adds at most one `source` line; if any other line in
    # rc happens to contain "source" we'd over-count, but rc is empty
    # before our test so this is exact.
    assert occurrences == 1


def test_install_shell_wrapper_conflict_without_force(tmp_path, monkeypatch):
    """If wrapper file exists with different content and --force is not
    set, the install fails with exit 2 — never silently overwrites."""
    monkeypatch.setenv("HOME", str(tmp_path))
    wrapper_dir = tmp_path / ".zsh"
    wrapper_dir.mkdir()
    wrapper = wrapper_dir / "memo-wrapper.zsh"
    wrapper.write_text("# user's hand-edited content\n", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(cli, ["install-shell-wrapper", "--write"])
    assert result.exit_code == 2
    # Original content untouched.
    assert wrapper.read_text(encoding="utf-8") == "# user's hand-edited content\n"


def test_install_shell_wrapper_force_overwrites(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    wrapper_dir = tmp_path / ".zsh"
    wrapper_dir.mkdir()
    wrapper = wrapper_dir / "memo-wrapper.zsh"
    wrapper.write_text("# stale\n", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(
        cli, ["install-shell-wrapper", "--write", "--force"],
    )
    assert result.exit_code == 0
    assert "function claude()" in wrapper.read_text(encoding="utf-8")


def test_install_shell_wrapper_warns_on_existing_alias(tmp_path, monkeypatch):
    """If the rc already has `alias claude=...`, the install command
    surfaces a warning so the user migrates to MEMO_CLAUDE_EXTRA_ARGS
    rather than silently shadowing it with the new function."""
    monkeypatch.setenv("HOME", str(tmp_path))
    rc = tmp_path / ".zshrc"
    rc.write_text(
        "alias claude='claude --dangerously-skip-permissions'\n",
        encoding="utf-8",
    )
    runner = CliRunner()
    result = runner.invoke(cli, ["install-shell-wrapper", "--write"])
    assert result.exit_code == 0
    import re
    stripped = re.sub(r'\x1b\[[0-9;]*m', '', result.output)
    assert "MEMO_CLAUDE_EXTRA_ARGS" in stripped
    assert "alias claude" in stripped
