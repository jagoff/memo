"""Auto-update on memo-mcp start — pure logic + gating (no network, no MLX)."""

from __future__ import annotations

import subprocess

import pytest

from memo.runtime import autoupdate as au


@pytest.mark.parametrize(
    "tag,expected",
    [
        ("v1.2.3", (1, 2, 3)),
        ("1.0.0", (1, 0, 0)),
        ("v2.10.4", (2, 10, 4)),
        ("v1.2.3-rc1", (1, 2, 3)),
        ("v1.2", None),
        ("latest", None),
        ("v1.x.0", None),
    ],
)
def test_parse_semver(tag, expected):
    assert au._parse_semver(tag) == expected


@pytest.mark.parametrize(
    "remote,local,expected",
    [
        ("v1.0.1", "1.0.0", True),
        ("v1.0.0", "1.0.0", False),
        ("v1.0.0", "1.0.1", False),
        ("v2.0.0", "1.9.9", True),
        ("garbage", "1.0.0", False),
        ("v1.0.1", "garbage", False),
    ],
)
def test_is_newer(remote, local, expected):
    assert au.is_newer(remote, local) is expected


def test_latest_remote_tag_picks_highest(monkeypatch):
    stdout = (
        "abc123\trefs/tags/v0.9.0\n"
        "def456\trefs/tags/v1.0.0\n"
        "aaa111\trefs/tags/v1.2.0\n"
        "bbb222\trefs/tags/not-a-version\n"
    )

    def fake_run(*a, **k):
        return subprocess.CompletedProcess(a, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(au.subprocess, "run", fake_run)
    assert au.latest_remote_tag("https://example/repo.git") == "v1.2.0"


def test_latest_remote_tag_none_on_failure(monkeypatch):
    def boom(*a, **k):
        raise FileNotFoundError("git missing")

    monkeypatch.setattr(au.subprocess, "run", boom)
    assert au.latest_remote_tag("https://example/repo.git") is None


def test_throttle_first_check_then_blocked(tmp_cfg):
    assert au._should_check(tmp_cfg, 3600, now=1000.0) is True
    au._record_check(tmp_cfg, now=1000.0)
    assert au._should_check(tmp_cfg, 3600, now=1000.0 + 60) is False  # within window
    assert au._should_check(tmp_cfg, 3600, now=1000.0 + 4000) is True  # past window


def test_maybe_auto_update_enabled_by_default(tmp_cfg, monkeypatch):
    monkeypatch.delenv("MEMO_AUTO_UPDATE", raising=False)
    monkeypatch.setattr(au, "latest_remote_tag", lambda *a, **k: None)
    spawned = au.maybe_auto_update(tmp_cfg)
    # default is now True; returns False only because no newer tag was found
    assert spawned is False


def test_maybe_auto_update_spawns_when_newer_tag(tmp_cfg, monkeypatch):
    monkeypatch.setenv("MEMO_AUTO_UPDATE", "1")
    monkeypatch.setattr(au, "latest_remote_tag", lambda *a, **k: "v999.0.0")
    calls = {"n": 0}

    def fake_popen(*a, **k):
        calls["n"] += 1

        class _P:
            pid = 4242

        return _P()

    monkeypatch.setattr(au.subprocess, "Popen", fake_popen)
    spawned = au.maybe_auto_update(tmp_cfg)
    assert spawned is True
    assert calls["n"] == 1
    # throttle recorded → a second call in the same window does not respawn
    assert au.maybe_auto_update(tmp_cfg) is False
    assert calls["n"] == 1


def test_maybe_auto_update_no_spawn_when_not_newer(tmp_cfg, monkeypatch):
    monkeypatch.setenv("MEMO_AUTO_UPDATE", "1")
    monkeypatch.setattr(au, "latest_remote_tag", lambda *a, **k: "v0.0.1")
    monkeypatch.setattr(
        au.subprocess, "Popen", lambda *a, **k: pytest.fail("should not spawn")
    )
    assert au.maybe_auto_update(tmp_cfg) is False
