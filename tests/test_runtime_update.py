"""`memo update` self-updater guards.

Regression: `memo update` run from an editable/dev `.venv` silently targeted the
SIBLING isolated uv-tool install (`_detect_install_method` returns "uv" because
`uv tool list` shows mlx-memo), so it installed a tag over the isolated runtime
and left the running editable checkout on its stale version — "memo update
doesn't update 1.1.11". The updater must detect an editable install and refuse,
pointing at git + uv sync instead.
"""

from __future__ import annotations

import base64
import json

from click.testing import CliRunner

import memo.runtime.update as upd
from memo.cli import cli


class _Rc:
    def __init__(self, rc: int) -> None:
        self.returncode = rc


def _editable_direct_url() -> str:
    return json.dumps({"url": "file:///Users/dev/repo", "dir_info": {"editable": True}})


def _vcs_direct_url() -> str:
    return json.dumps(
        {
            "url": "https://github.com/jagoff/memo.git",
            "vcs_info": {"vcs": "git", "requested_revision": "v2.1.0"},
        }
    )


def test_running_install_is_editable_true(monkeypatch):
    monkeypatch.setattr(upd, "_read_direct_url", _editable_direct_url)
    assert upd._running_install_is_editable() is True


def test_running_install_is_editable_false_for_vcs(monkeypatch):
    monkeypatch.setattr(upd, "_read_direct_url", _vcs_direct_url)
    assert upd._running_install_is_editable() is False


def test_running_install_is_editable_false_when_no_direct_url(monkeypatch):
    monkeypatch.setattr(upd, "_read_direct_url", lambda: None)
    assert upd._running_install_is_editable() is False


def test_editable_source_path_strips_file_scheme(monkeypatch):
    monkeypatch.setattr(upd, "_read_direct_url", _editable_direct_url)
    assert upd._editable_source_path() == "/Users/dev/repo"


def test_self_update_refuses_editable_and_does_not_touch_isolated_runtime(monkeypatch):
    """From an editable/dev .venv, `memo update` must NOT install a git tag over
    the sibling isolated tool — it must report the editable install and explain
    how to update the checkout instead."""
    monkeypatch.setattr(upd, "_running_install_is_editable", lambda: True)
    monkeypatch.setattr(upd, "_editable_source_path", lambda: "/Users/dev/repo")

    called: list = []
    monkeypatch.setattr(upd.subprocess, "run", lambda *a, **k: called.append(a) or _Rc(0))
    # Defensive: if the guard is missing, this would be reached and "succeed".
    monkeypatch.setattr("memo.runtime.autoupdate.latest_remote_tag", lambda *a, **k: "v999.0.0")

    result = CliRunner().invoke(cli, ["update"])

    assert result.exit_code == 0, result.output
    assert "editable" in result.output.lower()
    assert "uv pip install -e ." in result.output
    assert called == []  # no pipx/uv install attempted


def test_self_update_proceeds_for_isolated_install(monkeypatch):
    """Guard must not over-block: a real isolated (uv-tool) install still updates
    to a newer tag."""
    monkeypatch.setattr(upd, "_running_install_is_editable", lambda: False)
    monkeypatch.setattr("memo.runtime.autoupdate.latest_remote_tag", lambda *a, **k: "v999.0.0")
    monkeypatch.setattr(upd, "_detect_install_method", lambda: "uv")
    monkeypatch.setattr(upd, "_find_uv", lambda: "uv")
    monkeypatch.setattr(upd, "_clear_update_notify", lambda: None)
    monkeypatch.setattr(upd, "_notify_codex_plugin_updated", lambda: False)
    monkeypatch.setattr(upd, "_refresh_agent_artifacts", lambda: False)
    monkeypatch.setattr(upd, "_prewarm_after_update", lambda: None)

    calls: list = []
    monkeypatch.setattr(upd.subprocess, "run", lambda *a, **k: calls.append(a[0]) or _Rc(0))

    result = CliRunner().invoke(cli, ["update"])

    assert result.exit_code == 0, result.output
    assert any("tool" in c and "install" in c for c in calls), calls


def test_codex_plugin_update_notification_uses_notify_protocol(tmp_path, monkeypatch):
    tty = tmp_path / "tty"
    tty.touch()
    monkeypatch.setenv("MEMO_AGENT_TTY", str(tty))

    assert upd._notify_codex_plugin_updated() is True

    raw = tty.read_bytes()
    title = base64.b64encode(b"Plugin updated: memo").decode("ascii")
    body = base64.b64encode(b"Run /reload_plugins to apply").decode("ascii")
    assert raw.startswith(b"\x1b]3008;start=codex;kind=notify;")
    assert f"title={title}".encode("ascii") in raw
    assert f"body={body}".encode("ascii") in raw
    assert raw.endswith(b"\x1b\\")


def test_codex_plugin_update_notification_noops_without_tty(tmp_path, monkeypatch):
    monkeypatch.delenv("MEMO_AGENT_TTY", raising=False)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "missing-data-home"))

    assert upd._notify_codex_plugin_updated() is False


def test_refresh_agent_artifacts_updates_static_agent_surfaces(monkeypatch, tmp_path):
    def fake_which(name: str) -> str | None:
        if name in {"claude", "codex", "devin"}:
            return f"/usr/bin/{name}"
        return None

    monkeypatch.setattr(upd.shutil, "which", fake_which)
    monkeypatch.setattr("memo.runtime.mcp._agent_asset_root", lambda: tmp_path)
    monkeypatch.setattr(
        upd,
        "_devin_skill_path",
        lambda: tmp_path / ".config" / "devin" / "skills" / "memo" / "SKILL.md",
    )

    copied: list[tuple] = []
    installed: list[tuple] = []
    commands: list[tuple] = []
    monkeypatch.setattr("memo.runtime.codex._codex_home", lambda: tmp_path / "codex-home")
    monkeypatch.setattr(
        "memo.runtime.codex._copy_slash_skill",
        lambda root, dst, *, dry_run: copied.append((root, dst, dry_run)),
    )
    monkeypatch.setattr(
        "memo.runtime.codex._install_codex_plugin",
        lambda root, *, dry_run: installed.append((root, dry_run)),
    )
    monkeypatch.setattr(
        "memo.runtime.mcp._run_agent_command",
        lambda args, **kwargs: commands.append((args, kwargs)),
    )

    assert upd._refresh_agent_artifacts() is True
    assert (tmp_path, tmp_path / "codex-home" / "skills" / "memo" / "SKILL.md", False) in copied
    assert any(
        str(dst).endswith(".config/devin/skills/memo/SKILL.md") for _root, dst, _dry in copied
    )
    assert installed == [(tmp_path, False)]
    assert [call[0][:3] for call in commands] == [
        ["claude", "plugin", "marketplace"],
        ["claude", "plugin", "install"],
    ]


def test_refresh_agent_artifacts_skips_agents_without_static_assets(monkeypatch, tmp_path):
    monkeypatch.setattr(
        upd.shutil,
        "which",
        lambda name: f"/usr/bin/{name}" if name in {"opencode", "devin-desktop"} else None,
    )
    # Pin the devin-skill probe off the developer's real HOME: on a machine
    # where memo installed ~/.config/devin/skills/memo/SKILL.md, has_devin would
    # otherwise be True and the "skip" path under test would never run.
    monkeypatch.setattr(
        upd,
        "_devin_skill_path",
        lambda: tmp_path / ".config" / "devin" / "skills" / "memo" / "SKILL.md",
    )
    monkeypatch.setattr("memo.runtime.mcp._agent_asset_root", lambda: tmp_path)
    # Recording stubs, not raising ones: _refresh_agent_artifacts wraps each
    # branch in `except Exception`, so a raised AssertionError would be
    # swallowed and the test would pass for the wrong reason.
    copied: list[tuple] = []
    installed: list[tuple] = []
    commands: list[tuple] = []
    monkeypatch.setattr(
        "memo.runtime.codex._copy_slash_skill",
        lambda *args, **kwargs: copied.append(args),
    )
    monkeypatch.setattr(
        "memo.runtime.codex._install_codex_plugin",
        lambda *args, **kwargs: installed.append(args),
    )
    monkeypatch.setattr(
        "memo.runtime.mcp._run_agent_command",
        lambda *args, **kwargs: commands.append(args),
    )

    assert upd._refresh_agent_artifacts() is False
    assert copied == []
    assert installed == []
    assert commands == []


def test_finish_successful_update_refreshes_codex_plugin_before_notify(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(upd, "_clear_update_notify", lambda: calls.append("clear"))
    monkeypatch.setattr(upd, "_refresh_agent_artifacts", lambda: calls.append("refresh"))
    monkeypatch.setattr(upd, "_notify_codex_plugin_updated", lambda: calls.append("notify"))

    upd._finish_successful_update()

    assert calls == ["clear", "refresh", "notify"]


def test_self_update_to_tag_notifies_codex_after_success(monkeypatch):
    monkeypatch.setattr(upd, "_running_install_is_editable", lambda: False)
    monkeypatch.setattr(upd, "_detect_install_method", lambda: "uv")
    monkeypatch.setattr(upd, "_find_uv", lambda: "uv")
    monkeypatch.setattr(upd, "_clear_update_notify", lambda: None)
    monkeypatch.setattr(upd, "_refresh_agent_artifacts", lambda: False)
    monkeypatch.setattr(upd, "_prewarm_after_update", lambda: None)
    notified: list[bool] = []
    monkeypatch.setattr(
        upd, "_notify_codex_plugin_updated", lambda: notified.append(True), raising=False
    )

    calls: list = []
    monkeypatch.setattr(upd.subprocess, "run", lambda *a, **k: calls.append(a[0]) or _Rc(0))

    result = CliRunner().invoke(cli, ["update", "--to-tag", "v9.9.9"])

    assert result.exit_code == 0, result.output
    assert any("tool" in c and "install" in c for c in calls), calls
    assert notified == [True]
