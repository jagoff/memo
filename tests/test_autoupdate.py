"""Auto-update on memo-mcp start — pure logic + gating (no network, no MLX)."""

from __future__ import annotations

import json
import subprocess
import threading
from pathlib import Path

import pytest

from memo.runtime import autoupdate as au
from memo.runtime import autoupdate_worker as au_worker

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "args",
    [
        [],
        ["state", "v1.2.3", "not-a-pid", "1.0"],
        ["state", "v1.2.3", "123", "not-a-time"],
    ],
)
def test_autoupdate_worker_rejects_invalid_arguments(args):
    assert au_worker.main(args) == 2


def test_autoupdate_worker_returns_nonzero_when_lease_is_lost(tmp_path, monkeypatch):
    seen = {}

    def reject_lease(cfg, tag, *, parent_pid, child_pid, started_at):
        seen.update(
            cfg=cfg,
            tag=tag,
            parent_pid=parent_pid,
            child_pid=child_pid,
            started_at=started_at,
        )
        return False

    monkeypatch.setattr(au_worker, "_claim_spawned_lease", reject_lease)
    monkeypatch.setattr(au_worker.os, "getpid", lambda: 456)

    assert au_worker.main([str(tmp_path), "v4.4.3", "123", "12.5"]) == 1
    assert seen["cfg"].state_dir == tmp_path.resolve()
    assert seen["tag"] == "v4.4.3"
    assert seen["parent_pid"] == 123
    assert seen["child_pid"] == 456
    assert seen["started_at"] == 12.5


def test_autoupdate_worker_execs_trusted_interpreter_after_claim(tmp_path, monkeypatch):
    class ExecCalled(Exception):
        pass

    seen = {}
    monkeypatch.setattr(au_worker, "_claim_spawned_lease", lambda *args, **kwargs: True)

    def fake_execv(executable, argv):
        seen.update(executable=executable, argv=argv)
        raise ExecCalled

    monkeypatch.setattr(au_worker.os, "execv", fake_execv)

    with pytest.raises(ExecCalled):
        au_worker.main([str(tmp_path), "v4.4.3", "123", "12.5"])
    assert seen == {
        "executable": au_worker.sys.executable,
        "argv": [
            au_worker.sys.executable,
            "-m",
            "memo.cli",
            "update",
            "--to-tag",
            "v4.4.3",
        ],
    }


@pytest.mark.parametrize(
    "tag,expected",
    [
        ("v1.2.3", (1, 2, 3)),
        ("1.0.0", (1, 0, 0)),
        ("v2.10.4", (2, 10, 4)),
        ("v1.2.3-rc1", None),
        ("v1.2.3+build.4", None),
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


def test_latest_remote_tag_ignores_prereleases_and_build_metadata(monkeypatch):
    stdout = "abc123\trefs/tags/v2.0.0-rc1\ndef456\trefs/tags/v2.0.0+build.4\n"

    def fake_run(*a, **k):
        return subprocess.CompletedProcess(a, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(au.subprocess, "run", fake_run)

    assert au.latest_remote_tag("https://example/repo.git") is None


def test_latest_remote_tag_none_on_failure(monkeypatch):
    def boom(*a, **k):
        raise FileNotFoundError("git missing")

    monkeypatch.setattr(au.subprocess, "run", boom)
    assert au.latest_remote_tag("https://example/repo.git") is None


def test_tag_provenance_requires_successful_fetch_and_master_ancestry(monkeypatch):
    returncodes = iter((0, 0, 0))
    monkeypatch.setattr(
        au.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], next(returncodes), stdout="", stderr=""
        ),
    )
    assert au.tag_is_on_remote_master("https://example/repo.git", "v1.2.3") is True


def test_tag_provenance_rejects_tag_outside_master(monkeypatch):
    returncodes = iter((0, 0, 1))
    monkeypatch.setattr(
        au.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], next(returncodes), stdout="", stderr=""
        ),
    )
    assert au.tag_is_on_remote_master("https://example/repo.git", "v1.2.3") is False


class _FakeResp:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> _FakeResp:
        return self

    def __exit__(self, *a) -> None:
        return None


def test_anon_id_is_hashed_stable_and_not_raw(tmp_cfg):
    raw = str(tmp_cfg.device_id)
    anon = au._anon_id(tmp_cfg)
    assert anon and anon != raw  # hashed, not the raw id
    assert len(anon) == 16 and all(c in "0123456789abcdef" for c in anon)
    assert au._anon_id(tmp_cfg) == anon  # stable across calls
    import hashlib

    assert anon == hashlib.sha256(raw.encode()).hexdigest()[:16]


def test_latest_tag_via_endpoint_parses_and_sends_anon_params(monkeypatch):
    seen: dict[str, str] = {}

    def fake_urlopen(req, *a, **k):
        seen["url"] = req.full_url
        return _FakeResp(json.dumps({"latest": "v3.2.1"}).encode())

    monkeypatch.setattr(au.urllib.request, "urlopen", fake_urlopen)
    tag = au.latest_tag_via_endpoint(
        "https://tel.example/v1/latest",
        anon_id="deadbeefdeadbeef",
        version="3.0.0",
        os_name="Darwin",
    )
    assert tag == "v3.2.1"
    assert "id=deadbeefdeadbeef" in seen["url"]
    assert "v=3.0.0" in seen["url"]
    assert "os=Darwin" in seen["url"]


def test_latest_tag_via_endpoint_rejects_non_semver_and_failures(monkeypatch):
    monkeypatch.setattr(
        au.urllib.request,
        "urlopen",
        lambda *a, **k: _FakeResp(json.dumps({"latest": "not-a-version"}).encode()),
    )
    assert (
        au.latest_tag_via_endpoint("https://x/y", anon_id="a", version="1.0.0", os_name="Linux")
        is None
    )

    def boom(*a, **k):
        raise OSError("network down")

    monkeypatch.setattr(au.urllib.request, "urlopen", boom)
    assert (
        au.latest_tag_via_endpoint("https://x/y", anon_id="a", version="1.0.0", os_name="Linux")
        is None
    )


def test_resolve_latest_tag_uses_git_when_endpoint_unset(tmp_cfg, monkeypatch):
    monkeypatch.delenv("MEMO_UPDATE_ENDPOINT", raising=False)
    monkeypatch.setattr(au, "latest_remote_tag", lambda *a, **k: "v1.4.0")
    monkeypatch.setattr(
        au,
        "latest_tag_via_endpoint",
        lambda *a, **k: pytest.fail("endpoint must not be called when unset"),
    )
    assert au.resolve_latest_tag(tmp_cfg, "https://example/repo.git") == "v1.4.0"


def test_resolve_latest_tag_prefers_endpoint_then_falls_back_to_git(tmp_cfg, monkeypatch):
    monkeypatch.setenv("MEMO_UPDATE_ENDPOINT", "https://tel.example/v1/latest")

    # endpoint reachable → its tag wins, git not consulted
    monkeypatch.setattr(au, "latest_tag_via_endpoint", lambda *a, **k: "v9.9.9")
    monkeypatch.setattr(
        au, "latest_remote_tag", lambda *a, **k: pytest.fail("git probe should be skipped")
    )
    assert au.resolve_latest_tag(tmp_cfg, "https://example/repo.git") == "v9.9.9"

    # endpoint fails → git fallback
    monkeypatch.setattr(au, "latest_tag_via_endpoint", lambda *a, **k: None)
    monkeypatch.setattr(au, "latest_remote_tag", lambda *a, **k: "v1.0.0")
    assert au.resolve_latest_tag(tmp_cfg, "https://example/repo.git") == "v1.0.0"


def test_throttle_first_check_then_blocked(tmp_cfg):
    assert au._should_check(tmp_cfg, 3600, now=1000.0) is True
    au._record_check(tmp_cfg, now=1000.0)
    assert au._should_check(tmp_cfg, 3600, now=1000.0 + 60) is False  # within window
    assert au._should_check(tmp_cfg, 3600, now=1000.0 + 4000) is True  # past window


def test_runtime_flag_defaults(monkeypatch):
    # memo v4.1.0+: MEMO_AUTO_UPDATE defaults ON (memo keeps itself current);
    # the check/self-heal flags stay opt-in (default off).
    from memo.flags import flag_bool

    names = (
        "MEMO_UPDATE_CHECK_ENABLED",
        "MEMO_AUTO_UPDATE",
        "MEMO_STATUSLINE_SELFHEAL",
        "MEMO_HOOK_SELFHEAL",
    )
    for name in names:
        monkeypatch.delenv(name, raising=False)

    assert {name: flag_bool(name) for name in names} == {
        "MEMO_UPDATE_CHECK_ENABLED": False,
        "MEMO_AUTO_UPDATE": True,
        "MEMO_STATUSLINE_SELFHEAL": False,
        "MEMO_HOOK_SELFHEAL": False,
    }


def test_maybe_auto_update_enabled_by_default(tmp_cfg, monkeypatch):
    # memo v4.1.0+: MEMO_AUTO_UPDATE defaults ON. With no explicit env var (the
    # conftest hermetic pin removed), a memo-mcp start checks for a newer tag and
    # spawns the background updater. The isolated MEMO_CONFIG_DIR means no
    # markdown config leaks in, so this exercises the real built-in default.
    monkeypatch.delenv("MEMO_AUTO_UPDATE", raising=False)
    monkeypatch.setattr(au, "latest_remote_tag", lambda *a, **k: "v999.0.0")
    monkeypatch.setattr(au, "tag_is_on_remote_master", lambda *a, **k: True)
    monkeypatch.setattr(au, "_process_identity", lambda pid: f"test:{pid}")
    monkeypatch.setattr(au.subprocess, "Popen", lambda *a, **k: _FakeUpdateProc(4242))

    assert au.maybe_auto_update(tmp_cfg) is True


def test_maybe_auto_update_respects_explicit_optout(tmp_cfg, monkeypatch):
    # Setting =0 keeps startup fully offline: no tag check, no spawn.
    monkeypatch.setenv("MEMO_AUTO_UPDATE", "0")
    monkeypatch.setattr(
        au,
        "latest_remote_tag",
        lambda *a, **k: pytest.fail("opted-out auto-update must not access the network"),
    )

    assert au.maybe_auto_update(tmp_cfg) is False


def test_generated_mcp_environment_does_not_force_background_work(monkeypatch):
    from memo.runtime import mcp

    for name in (
        "MEMO_UPDATE_CHECK_ENABLED",
        "MEMO_AUTO_UPDATE",
        "MEMO_STATUSLINE_SELFHEAL",
        "MEMO_HOOK_SELFHEAL",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(mcp, "_actual_embedder_config", lambda: {})

    env = mcp._mcp_server_env()

    assert env.get("MEMO_AUTO_UPDATE") != "1"
    assert env.get("MEMO_UPDATE_CHECK_ENABLED") != "1"


def test_bundled_plugin_does_not_force_auto_update() -> None:
    plugin_mcp = (ROOT / "plugins" / "memo" / ".mcp.json").read_text(encoding="utf-8")

    assert '"MEMO_AUTO_UPDATE": "1"' not in plugin_mcp


class _FakeUpdateProc:
    def __init__(self, pid: int, returncode: int | None = None) -> None:
        self.pid = pid
        self.returncode = returncode
        self.terminated = False

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = -15

    def wait(self, timeout: int | None = None) -> int | None:
        del timeout
        return self.returncode


def _enable_newer_update(monkeypatch) -> None:
    monkeypatch.setenv("MEMO_AUTO_UPDATE", "1")
    monkeypatch.setattr(au, "latest_remote_tag", lambda *a, **k: "v999.0.0")
    monkeypatch.setattr(au, "tag_is_on_remote_master", lambda *a, **k: True)
    monkeypatch.setattr(au, "_process_identity", lambda pid: f"test:{pid}")


def test_maybe_auto_update_offers_but_does_not_spawn_on_homebrew(tmp_cfg, monkeypatch):
    # Homebrew is user-managed: auto-update writes the notify (the banner then
    # offers `brew upgrade mlx-memo`) but must NOT run brew unattended.
    _enable_newer_update(monkeypatch)
    monkeypatch.setattr("memo.runtime.detect.is_homebrew_install", lambda: True)
    monkeypatch.setattr(
        au.subprocess,
        "Popen",
        lambda *a, **k: pytest.fail("homebrew auto-update must not spawn a background install"),
    )

    assert au.maybe_auto_update(tmp_cfg) is False
    assert (tmp_cfg.state_dir / au._NOTIFY_FILE).read_text().strip() == "v999.0.0"


def test_maybe_auto_update_deduplicates_while_child_is_active(tmp_cfg, monkeypatch):
    _enable_newer_update(monkeypatch)
    proc = _FakeUpdateProc(4242)
    attempts = 0

    def fake_popen(*a, **k):
        nonlocal attempts
        attempts += 1
        return proc

    monkeypatch.setattr(au.subprocess, "Popen", fake_popen)

    assert au.maybe_auto_update(tmp_cfg) is True
    monkeypatch.setattr(au, "_ACTIVE_UPDATES", {})
    monkeypatch.setattr(au, "_process_is_alive", lambda _pid: True)
    assert au.maybe_auto_update(tmp_cfg) is False
    assert attempts == 1
    lease = json.loads((tmp_cfg.state_dir / au._SPAWNED_STAMP).read_text())
    assert lease["tag"] == "v999.0.0"
    assert lease["pid"] == 4242
    assert lease["state"] == "running"


def test_maybe_auto_update_retries_after_child_nonzero_exit(tmp_cfg, monkeypatch):
    _enable_newer_update(monkeypatch)
    procs = [_FakeUpdateProc(4242), _FakeUpdateProc(4243)]
    attempts = 0

    def fake_popen(*a, **k):
        nonlocal attempts
        proc = procs[attempts]
        attempts += 1
        return proc

    monkeypatch.setattr(au.subprocess, "Popen", fake_popen)

    assert au.maybe_auto_update(tmp_cfg) is True
    procs[0].returncode = 1
    assert au.maybe_auto_update(tmp_cfg) is True
    assert attempts == 2


def test_maybe_auto_update_retries_after_child_is_aborted(tmp_cfg, monkeypatch):
    _enable_newer_update(monkeypatch)
    procs = [_FakeUpdateProc(4242), _FakeUpdateProc(4243)]
    attempts = 0

    def fake_popen(*a, **k):
        nonlocal attempts
        proc = procs[attempts]
        attempts += 1
        return proc

    monkeypatch.setattr(au.subprocess, "Popen", fake_popen)

    assert au.maybe_auto_update(tmp_cfg) is True
    # Simulate a new memo-mcp process after the detached child was killed:
    # there is no in-memory Popen left and the persisted PID is no longer live.
    monkeypatch.setattr(au, "_ACTIVE_UPDATES", {}, raising=False)
    monkeypatch.setattr(au, "_process_is_alive", lambda _pid: False, raising=False)
    assert au.maybe_auto_update(tmp_cfg) is True
    assert attempts == 2


def test_live_persisted_child_never_expires_by_wall_clock(tmp_cfg, monkeypatch):
    monkeypatch.setattr(au, "_process_identity", lambda pid: f"test:{pid}")
    au._write_spawned_lease(
        tmp_cfg,
        "v999.0.0",
        4242,
        "running",
        started_at=1.0,
        process_identity="test:4242",
    )
    monkeypatch.setattr(au, "_ACTIVE_UPDATES", {})
    monkeypatch.setattr(au, "_process_is_alive", lambda pid: pid == 4242)

    assert au._spawn_lease_blocks_update(tmp_cfg, "v999.0.0", now=10_000_000.0) is True
    assert (tmp_cfg.state_dir / au._SPAWNED_STAMP).exists()


def test_running_lease_with_reused_pid_is_recovered(tmp_cfg, monkeypatch):
    au._write_spawned_lease(
        tmp_cfg,
        "v999.0.0",
        4242,
        "running",
        started_at=1.0,
        process_identity="old-process",
    )
    monkeypatch.setattr(au, "_ACTIVE_UPDATES", {})
    monkeypatch.setattr(au, "_process_is_alive", lambda _pid: True)
    monkeypatch.setattr(au, "_process_identity", lambda _pid: "reused-process")

    assert au._spawn_lease_blocks_update(tmp_cfg, "v999.0.0", now=2.0) is False
    assert not (tmp_cfg.state_dir / au._SPAWNED_STAMP).exists()


def test_stale_starting_lease_is_recovered_even_if_owner_pid_is_live(tmp_cfg, monkeypatch):
    au._write_spawned_lease(
        tmp_cfg,
        "v999.0.0",
        4242,
        "starting",
        started_at=1.0,
        process_identity="test:4242",
    )
    monkeypatch.setattr(au, "_process_is_alive", lambda _pid: True)
    monkeypatch.setattr(au, "_process_identity", lambda pid: f"test:{pid}")

    assert au._spawn_lease_blocks_update(tmp_cfg, "v999.0.0", now=62.0) is False
    assert not (tmp_cfg.state_dir / au._SPAWNED_STAMP).exists()


def test_fresh_starting_lease_blocks_when_parent_died(tmp_cfg, monkeypatch):
    au._write_spawned_lease(
        tmp_cfg,
        "v999.0.0",
        4242,
        "starting",
        started_at=10.0,
        process_identity="dead-parent",
    )
    monkeypatch.setattr(au, "_process_is_alive", lambda _pid: False)

    assert au._spawn_lease_blocks_update(tmp_cfg, "v999.0.0", now=11.0) is True


def test_stale_starting_lease_for_other_tag_does_not_block(tmp_cfg, monkeypatch):
    au._write_spawned_lease(
        tmp_cfg,
        "v998.0.0",
        4242,
        "starting",
        started_at=1.0,
        process_identity="test:4242",
    )
    monkeypatch.setattr(au, "_process_is_alive", lambda _pid: True)
    monkeypatch.setattr(au, "_process_identity", lambda pid: f"test:{pid}")

    assert au._spawn_lease_blocks_update(tmp_cfg, "v999.0.0", now=62.0) is False
    assert not (tmp_cfg.state_dir / au._SPAWNED_STAMP).exists()


def test_spawned_worker_claims_starting_lease_after_parent_crash(tmp_cfg, monkeypatch):
    started_at = 10.0
    parent_pid = 4242
    child_pid = 4343
    monkeypatch.setattr(au, "_process_identity", lambda pid: f"test:{pid}")
    assert au._acquire_spawned_lease(tmp_cfg, "v999.0.0", started_at=started_at)
    lease_path = tmp_cfg.state_dir / au._SPAWNED_STAMP
    payload = json.loads(lease_path.read_text())
    payload["pid"] = parent_pid
    payload["process_identity"] = "test:4242"
    lease_path.write_text(json.dumps(payload))

    assert au._claim_spawned_lease(
        tmp_cfg,
        "v999.0.0",
        parent_pid=parent_pid,
        child_pid=child_pid,
        started_at=started_at,
    )
    lease = au._read_spawned_lease(tmp_cfg)
    assert lease is not None
    assert (lease.state, lease.pid, lease.process_identity) == (
        "running",
        child_pid,
        "test:4343",
    )


def test_stale_cleanup_cannot_delete_a_concurrently_acquired_lease(tmp_cfg, monkeypatch):
    au._write_spawned_lease(tmp_cfg, "v1.0.0", 111, "running", started_at=1.0)
    monkeypatch.setattr(au, "_ACTIVE_UPDATES", {})
    monkeypatch.setattr(au, "_process_is_alive", lambda _pid: False)
    cleanup_entered = threading.Event()
    allow_cleanup = threading.Event()
    acquire_finished = threading.Event()
    original_clear = au._clear_spawned_stamp_unlocked

    def delayed_clear(*args, **kwargs):
        cleanup_entered.set()
        assert allow_cleanup.wait(timeout=2.0)
        return original_clear(*args, **kwargs)

    monkeypatch.setattr(au, "_clear_spawned_stamp_unlocked", delayed_clear)
    cleaner = threading.Thread(
        target=lambda: au._spawn_lease_blocks_update(tmp_cfg, "v2.0.0", now=2.0)
    )
    acquired: list[bool] = []

    def acquire() -> None:
        acquired.append(au._acquire_spawned_lease(tmp_cfg, "v2.0.0", started_at=2.0))
        acquire_finished.set()

    cleaner.start()
    assert cleanup_entered.wait(timeout=2.0)
    acquirer = threading.Thread(target=acquire)
    acquirer.start()
    assert not acquire_finished.wait(timeout=0.1), "acquire bypassed the cleanup file lock"
    allow_cleanup.set()
    cleaner.join(timeout=2.0)
    acquirer.join(timeout=2.0)

    assert acquired == [True]
    lease = au._read_spawned_lease(tmp_cfg)
    assert lease is not None and lease.tag == "v2.0.0"


def test_generated_mcp_environment_rejects_invalid_persisted_profile(tmp_path, monkeypatch) -> None:
    from memo.config_md import invalidate_cache
    from memo.errors import ValidationError
    from memo.runtime import mcp

    home = tmp_path / "memo-home"
    config_dir = home / "config"
    config_dir.mkdir(parents=True)
    (config_dir / "hooks-config.md").write_text(
        '```toml\n[mcp]\nprofile = "typo"\n```\n', encoding="utf-8"
    )
    monkeypatch.setenv("MEMO_CONFIG_DIR", str(home))
    monkeypatch.delenv("MEMO_MCP_PROFILE", raising=False)
    invalidate_cache()

    with pytest.raises(ValidationError, match="MEMO_MCP_PROFILE"):
        mcp._mcp_server_env()


def test_maybe_auto_update_keeps_successful_child_deduplicated(tmp_cfg, monkeypatch):
    _enable_newer_update(monkeypatch)
    proc = _FakeUpdateProc(4242)
    attempts = 0

    def fake_popen(*a, **k):
        nonlocal attempts
        attempts += 1
        return proc

    monkeypatch.setattr(au.subprocess, "Popen", fake_popen)

    assert au.maybe_auto_update(tmp_cfg) is True
    au._mark_spawned_success(tmp_cfg, "v999.0.0", pid=proc.pid)
    monkeypatch.setattr(au, "_ACTIVE_UPDATES", {})
    assert au.maybe_auto_update(tmp_cfg) is False
    assert attempts == 1
    lease = json.loads((tmp_cfg.state_dir / au._SPAWNED_STAMP).read_text())
    assert lease["tag"] == "v999.0.0"
    assert lease["pid"] == 4242
    assert lease["state"] == "succeeded"


def test_maybe_auto_update_retries_after_popen_failure_without_latching(tmp_cfg, monkeypatch):
    _enable_newer_update(monkeypatch)
    attempts = 0

    def failed_popen(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        raise OSError("cannot spawn")

    monkeypatch.setattr(au.subprocess, "Popen", failed_popen)

    assert au.maybe_auto_update(tmp_cfg) is False
    assert au.maybe_auto_update(tmp_cfg) is False
    assert attempts == 2
    assert not (tmp_cfg.state_dir / au._SPAWNED_STAMP).exists()


def test_maybe_auto_update_retries_when_running_lease_write_fails(tmp_cfg, monkeypatch):
    _enable_newer_update(monkeypatch)
    procs = [_FakeUpdateProc(4242), _FakeUpdateProc(4243)]
    attempts = 0
    original_write = au._write_spawned_lease
    fail_transition = True

    def fake_popen(*args, **kwargs):
        nonlocal attempts
        del args, kwargs
        proc = procs[attempts]
        attempts += 1
        return proc

    def flaky_write(*args, **kwargs):
        nonlocal fail_transition
        if args[3] == "running" and fail_transition:
            fail_transition = False
            return False
        return original_write(*args, **kwargs)

    monkeypatch.setattr(au.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(au, "_write_spawned_lease", flaky_write)

    assert au.maybe_auto_update(tmp_cfg) is False
    assert procs[0].terminated is True
    assert not (tmp_cfg.state_dir / au._SPAWNED_STAMP).exists()
    assert au.maybe_auto_update(tmp_cfg) is True
    assert attempts == 2


def test_maybe_auto_update_no_spawn_when_not_newer(tmp_cfg, monkeypatch):
    monkeypatch.setenv("MEMO_AUTO_UPDATE", "1")
    monkeypatch.setattr(au, "latest_remote_tag", lambda *a, **k: "v0.0.1")
    monkeypatch.setattr(au, "tag_is_on_remote_master", lambda *a, **k: True)
    monkeypatch.setattr(au.subprocess, "Popen", lambda *a, **k: pytest.fail("should not spawn"))
    assert au.maybe_auto_update(tmp_cfg) is False
