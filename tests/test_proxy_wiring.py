"""Wiring Claude Code at the local proxy is a HARD dependency: with
ANTHROPIC_BASE_URL pointed at a loopback port where nothing listens, the CLI
fails exactly like a dead network. These tests pin the two rules that keep
that from happening by accident."""

from __future__ import annotations

import json

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
