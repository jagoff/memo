import os
import shutil
import subprocess

import pytest

from memo.ops_launchd import (
    install_chat,
    parse_launchctl_list,
    render_chat_plist,
    uninstall_chat,
)


def test_render_chat_plist_contents() -> None:
    plist = render_chat_plist("/usr/local/bin/memo", "/Users/tester", port=8765, dist="/x/dist")
    assert "<key>Label</key>" in plist and "com.memo.chat" in plist
    assert "/usr/local/bin/memo" in plist
    assert "serve" in plist and "8765" in plist and "/x/dist" in plist
    assert "/Users/tester/Library/Logs/memo/chat.log" in plist
    assert "KeepAlive" in plist


def test_render_without_dist_omits_flag() -> None:
    plist = render_chat_plist("/bin/memo", "/Users/t", port=8765, dist=None)
    assert "--dist" not in plist


def test_parse_launchctl_list() -> None:
    raw = "PID\tStatus\tLabel\n50864\t0\tcom.memo.recall-daemon\n-\t0\tcom.memo.nightly\n123\t0\tcom.other.thing\n"
    rows = parse_launchctl_list(raw)
    labels = {r["label"] for r in rows}
    assert labels == {"com.memo.recall-daemon", "com.memo.nightly"}
    recall = next(r for r in rows if r["label"] == "com.memo.recall-daemon")
    assert recall["pid"] == 50864 and recall["last_exit"] == 0


def test_render_chat_plist_escapes_xml_ampersand(tmp_path) -> None:
    plist = render_chat_plist("/usr/local/bin/memo", "/Users/tester", port=8765, dist="/x/A&B/dist")
    assert "A&amp;B" in plist
    assert "/x/A&B/dist" not in plist  # raw unescaped ampersand must not survive

    plist_path = tmp_path / "escaped.plist"
    plist_path.write_text(plist, encoding="utf-8")
    if shutil.which("plutil") is not None:
        result = subprocess.run(
            ["plutil", "-lint", str(plist_path)], capture_output=True, text=True, check=False
        )
        assert result.returncode == 0, result.stdout + result.stderr


def test_render_chat_plist_forwards_memo_env_vars(tmp_path, monkeypatch) -> None:
    # launchd agents don't inherit the shell env — without forwarding MEMO_*
    # vars the installed daemon falls back to the default 0.6B/1024 embedder
    # against a 2560-dim index (broken retrieval + poisoned vote embeddings).
    monkeypatch.setenv("MEMO_EMBEDDER_DIMS", "2560")
    monkeypatch.setenv("MEMO_VAULT_PATH", "/Users/tester/A&B")
    monkeypatch.delenv("NOT_MEMO_UNRELATED", raising=False)
    monkeypatch.setenv("NOT_MEMO_UNRELATED", "should-not-forward")

    plist = render_chat_plist("/usr/local/bin/memo", "/Users/tester", port=8765, dist=None)

    assert "<key>MEMO_EMBEDDER_DIMS</key>" in plist
    assert "<string>2560</string>" in plist
    assert "<key>MEMO_VAULT_PATH</key>" in plist
    assert "A&amp;B" in plist
    assert "/Users/tester/A&B" not in plist  # raw unescaped ampersand must not survive
    assert "NOT_MEMO_UNRELATED" not in plist

    plist_path = tmp_path / "env.plist"
    plist_path.write_text(plist, encoding="utf-8")
    if shutil.which("plutil") is not None:
        result = subprocess.run(
            ["plutil", "-lint", str(plist_path)], capture_output=True, text=True, check=False
        )
        assert result.returncode == 0, result.stdout + result.stderr


def test_install_chat_writes_plist_and_calls_bootout_then_bootstrap(tmp_path, monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(os, "getuid", lambda: 501)

    path = install_chat("/usr/local/bin/memo", tmp_path, port=9999, dist=None)

    assert path == tmp_path / "Library" / "LaunchAgents" / "com.memo.chat.plist"
    assert path.exists()
    content = path.read_text(encoding="utf-8")
    assert "9999" in content
    assert "/usr/local/bin/memo" in content

    launchctl_calls = [c for c in calls if c[0] == "launchctl"]
    assert len(launchctl_calls) == 2
    assert launchctl_calls[0][:3] == ["launchctl", "bootout", "gui/501"]
    assert launchctl_calls[1][:3] == ["launchctl", "bootstrap", "gui/501"]


def test_install_chat_bootstrap_failure_raises_runtime_error(tmp_path, monkeypatch) -> None:
    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        if cmd[:2] == ["launchctl", "bootstrap"]:
            raise subprocess.CalledProcessError(1, cmd, output="", stderr="bootstrap boom")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(os, "getuid", lambda: 501)

    with pytest.raises(RuntimeError, match="bootstrap boom"):
        install_chat("/usr/local/bin/memo", tmp_path, port=8765, dist=None)

    # plist is written before the (failed) bootstrap call
    assert (tmp_path / "Library" / "LaunchAgents" / "com.memo.chat.plist").exists()


def test_uninstall_chat_removes_file_and_returns_bool(tmp_path, monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(os, "getuid", lambda: 501)

    plist_dir = tmp_path / "Library" / "LaunchAgents"
    plist_dir.mkdir(parents=True)
    plist_path = plist_dir / "com.memo.chat.plist"
    plist_path.write_text("stub", encoding="utf-8")

    assert uninstall_chat(tmp_path) is True
    assert not plist_path.exists()
    assert calls[0][:3] == ["launchctl", "bootout", "gui/501"]

    assert uninstall_chat(tmp_path) is False
