"""Wiring Claude Code at the local proxy is a HARD dependency: with
ANTHROPIC_BASE_URL pointed at a loopback port where nothing listens, the CLI
fails exactly like a dead network. These tests pin the two rules that keep
that from happening by accident."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from memo.proxy_wiring import settings_path, unwire, wire, wired_port


def _settings(tmp_path, payload):
    p = settings_path(tmp_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


def test_wire_writes_the_base_url_into_a_fresh_settings_file(tmp_path):
    assert wire(tmp_path, 8768) is True
    got = json.loads(settings_path(tmp_path).read_text(encoding="utf-8"))
    assert got["env"]["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8768"


def test_wire_preserves_every_other_key(tmp_path):
    _settings(tmp_path, {"model": "opus", "env": {"FOO": "1"}, "hooks": {"Stop": []}})
    wire(tmp_path, 8768)
    got = json.loads(settings_path(tmp_path).read_text(encoding="utf-8"))
    assert got["model"] == "opus"
    assert got["env"]["FOO"] == "1"
    assert got["hooks"] == {"Stop": []}


def test_wire_is_idempotent(tmp_path):
    assert wire(tmp_path, 8768) is True
    assert wire(tmp_path, 8768) is False


def test_wire_refuses_to_hijack_a_real_gateway(tmp_path):
    """Someone routing through a corporate LLM gateway must not silently have
    their traffic rerouted to a local process. A non-loopback base URL is a
    deliberate choice and is left exactly as found."""
    _settings(tmp_path, {"env": {"ANTHROPIC_BASE_URL": "https://gateway.corp.example"}})
    assert wire(tmp_path, 8768) is False
    got = json.loads(settings_path(tmp_path).read_text(encoding="utf-8"))
    assert got["env"]["ANTHROPIC_BASE_URL"] == "https://gateway.corp.example"


def test_unwire_removes_only_a_loopback_url(tmp_path):
    _settings(tmp_path, {"env": {"ANTHROPIC_BASE_URL": "http://127.0.0.1:8768", "K": "v"}})
    assert unwire(tmp_path) is True
    got = json.loads(settings_path(tmp_path).read_text(encoding="utf-8"))
    assert "ANTHROPIC_BASE_URL" not in got["env"]
    assert got["env"]["K"] == "v"


def test_unwire_leaves_a_real_gateway_alone(tmp_path):
    _settings(tmp_path, {"env": {"ANTHROPIC_BASE_URL": "https://gateway.corp.example"}})
    assert unwire(tmp_path) is False
    got = json.loads(settings_path(tmp_path).read_text(encoding="utf-8"))
    assert got["env"]["ANTHROPIC_BASE_URL"] == "https://gateway.corp.example"


def test_wired_port_reads_back_what_wire_wrote(tmp_path):
    assert wired_port(tmp_path) is None
    wire(tmp_path, 9001)
    assert wired_port(tmp_path) == 9001


def test_a_corrupt_settings_file_never_raises(tmp_path):
    p = settings_path(tmp_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{not json", encoding="utf-8")
    assert wire(tmp_path, 8768) is False
    assert unwire(tmp_path) is False
    assert wired_port(tmp_path) is None


def test_wait_until_listening_is_false_for_a_dead_port():
    """The gate in front of `wire`. A port nothing listens on must report
    False quickly rather than block the install."""
    from memo.ops_launchd import wait_until_listening

    assert wait_until_listening(1, timeout_s=0.4, interval_s=0.05) is False


def test_wait_until_listening_finds_a_real_listener():
    import socket

    from memo.ops_launchd import wait_until_listening

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        assert wait_until_listening(srv.getsockname()[1], timeout_s=2.0) is True


def test_ops_install_proxy_honours_the_port_option(monkeypatch):
    """`install_proxy` used to be called without `port`, so `--port` was
    silently ignored for the proxy -- while the port-in-use error told the
    user to "pick another with --port". With the proxy installed by default,
    that dead escape hatch is the difference between a one-flag fix and a
    user whose client cannot start."""
    import click.testing

    from memo import cli_ops

    seen: dict = {}

    def _fake_install(memo_bin, home, *, port):
        seen["port"] = port
        return home / "fake.plist"

    monkeypatch.setattr("memo.ops_launchd.install_proxy", _fake_install)
    monkeypatch.setattr("shutil.which", lambda _n: "/usr/bin/memo")
    res = click.testing.CliRunner().invoke(
        cli_ops.ops_group, ["install", "proxy", "--port", "9999"]
    )
    assert res.exit_code == 0, res.output
    assert seen["port"] == 9999


def test_ops_install_proxy_defaults_to_the_proxy_port_not_the_chat_one(monkeypatch):
    """A shared --port default of 8765 is the chat port; the proxy's is 8768.
    Defaulting the proxy to 8765 would collide with chat on every machine
    that runs both."""
    import click.testing

    from memo import cli_ops

    seen: dict = {}

    def _fake_install(memo_bin, home, *, port):
        seen["port"] = port
        return home / "fake.plist"

    monkeypatch.setattr("memo.ops_launchd.install_proxy", _fake_install)
    monkeypatch.setattr("shutil.which", lambda _n: "/usr/bin/memo")
    res = click.testing.CliRunner().invoke(cli_ops.ops_group, ["install", "proxy"])
    assert res.exit_code == 0, res.output
    assert seen["port"] == 8768


def test_ops_install_proxy_reports_failure_when_the_agent_never_answers(monkeypatch):
    """The installer must not claim success for a proxy that did not start.

    `install_proxy` bootstraps the agent and then gates the settings.json
    wiring on `wait_until_listening`. That gate correctly protects the client
    -- ANTHROPIC_BASE_URL is never written at a dead port -- but nothing told
    the CALLER, so `memo ops install proxy` exited 0 and install.sh printed
    "proxy installed — Claude Code now routes through it" over a launchd agent
    that was crashlooping. A silent half-failure is worse than a loud one: the
    user has no reason to look.
    """
    import click.testing

    from memo import cli_ops

    monkeypatch.setattr(
        "memo.ops_launchd.install_proxy",
        lambda *a, **k: (_ for _ in ()).throw(
            RuntimeError("proxy agent bootstrapped but never answered on 127.0.0.1:8768")
        ),
    )
    monkeypatch.setattr("shutil.which", lambda _n: "/usr/bin/memo")
    res = click.testing.CliRunner().invoke(cli_ops.ops_group, ["install", "proxy"])
    assert res.exit_code != 0
    assert "never answered" in res.output
    assert "Traceback" not in res.output


def test_install_proxy_raises_when_the_listener_never_comes_up(tmp_path, monkeypatch):
    """The unit behind it: a bootstrap that 'succeeded' but produced no
    listener is a failed install, not a quiet one."""
    import memo.ops_launchd as launchd

    monkeypatch.setattr(launchd, "_port_owner", lambda _p: None)
    monkeypatch.setattr(launchd, "_label_loaded", lambda _l: False)
    monkeypatch.setattr(
        launchd.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0, stderr="")
    )
    monkeypatch.setattr(launchd, "wait_until_listening", lambda *a, **k: False)
    wired = []
    monkeypatch.setattr(launchd.proxy_wiring, "wire", lambda *a, **k: wired.append(a) or True)

    with pytest.raises(RuntimeError, match="never answered"):
        launchd.install_proxy("/usr/bin/memo", tmp_path, port=8768)
    assert wired == [], "settings.json must not be touched when the proxy never came up"


def test_install_proxy_wires_the_client_once_the_listener_answers(tmp_path, monkeypatch):
    """The success half of the gate: a proxy that came up DOES get wired,
    and at the port it was actually installed on."""
    import memo.ops_launchd as launchd

    monkeypatch.setattr(launchd, "_port_owner", lambda _p: None)
    monkeypatch.setattr(launchd, "_label_loaded", lambda _l: False)
    monkeypatch.setattr(
        launchd.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0, stderr="")
    )
    monkeypatch.setattr(launchd, "wait_until_listening", lambda *a, **k: True)
    seen: dict = {}
    monkeypatch.setattr(
        launchd.proxy_wiring,
        "wire",
        lambda claude_dir, port: seen.update(dir=claude_dir, port=port),
    )

    path = launchd.install_proxy("/usr/bin/memo", tmp_path, port=9123)

    assert path.exists()
    assert seen["port"] == 9123
    assert seen["dir"] == tmp_path / ".claude"


def test_proxy_serve_serves_the_app_build_app_returned(tmp_path, monkeypatch):
    """`uvicorn.run` must receive the object `build_app` produced.

    This is the line the fix moved: `build_app(upstream)` used to be
    evaluated in the argument list, outside the ImportError guard. Pinning
    that uvicorn is handed exactly that object keeps the guard from being
    "fixed" back into an inline call.
    """
    import sys

    import click.testing

    from memo import cli_proxy

    sentinel = object()
    calls: dict = {}

    fake_uvicorn = SimpleNamespace(run=lambda app, **kw: calls.update(app=app, **kw))
    monkeypatch.setitem(sys.modules, "uvicorn", fake_uvicorn)
    monkeypatch.setattr("memo.proxy.server.build_app", lambda upstream: sentinel)

    res = click.testing.CliRunner().invoke(
        cli_proxy.proxy_group, ["serve", "--port", "9124"], env={"MEMO_NONINTERACTIVE": "1"}
    )

    assert res.exit_code == 0, res.output
    assert calls["app"] is sentinel
    assert calls["port"] == 9124
