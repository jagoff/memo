"""`memo update` self-updater guards.

Regression: `memo update` run from an editable/dev `.venv` silently targeted the
SIBLING isolated uv-tool install (`_detect_install_method` returns "uv" because
`uv tool list` shows mlx-memo), so it installed a tag over the isolated runtime
and left the running editable checkout on its stale version — "memo update
doesn't update 1.1.11". The updater must detect an editable install and refuse,
pointing at git + uv sync instead.
"""

from __future__ import annotations

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
    monkeypatch.setattr(upd, "_prewarm_after_update", lambda: None)

    calls: list = []
    monkeypatch.setattr(upd.subprocess, "run", lambda *a, **k: calls.append(a[0]) or _Rc(0))

    result = CliRunner().invoke(cli, ["update"])

    assert result.exit_code == 0, result.output
    assert any("tool" in c and "install" in c for c in calls), calls
