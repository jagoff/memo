from __future__ import annotations

# These tests intentionally exercise rejection/acknowledgement of wildcard binds.
# ruff: noqa: S104
import pytest
from click.testing import CliRunner

from memo.cli_http import http_api

_TOKEN = "test-token-" + ("x" * 32)


def test_http_cli_rejects_unacknowledged_non_loopback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("MEMO_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("MEMO_HTTP_API_TOKEN", _TOKEN)
    called = []
    monkeypatch.setattr("memo.server_http.run_server", lambda **kwargs: called.append(kwargs))

    result = CliRunner().invoke(http_api, ["--host", "0.0.0.0"])

    assert result.exit_code != 0
    assert "non-loopback" in result.output
    assert called == []


def test_http_cli_forwards_explicit_security_options(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("MEMO_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("MEMO_HTTP_API_TOKEN", _TOKEN)
    called = []
    monkeypatch.setattr("memo.server_http.run_server", lambda **kwargs: called.append(kwargs))

    result = CliRunner().invoke(
        http_api,
        ["--host", "0.0.0.0", "--port", "9090", "--allow-non-loopback"],
    )

    assert result.exit_code == 0, result.output
    assert called == [
        {
            "port": 9090,
            "host": "0.0.0.0",
            "allow_no_auth": False,
            "allow_non_loopback": True,
        }
    ]


def test_http_cli_allows_no_auth_only_on_loopback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("MEMO_STATE_DIR", str(tmp_path / "state"))
    called = []
    monkeypatch.setattr("memo.server_http.run_server", lambda **kwargs: called.append(kwargs))

    local = CliRunner().invoke(http_api, ["--allow-no-auth"])
    exposed = CliRunner().invoke(
        http_api,
        ["--host", "0.0.0.0", "--allow-no-auth", "--allow-non-loopback"],
    )

    assert local.exit_code == 0, local.output
    assert called == [
        {
            "port": 8080,
            "host": "127.0.0.1",
            "allow_no_auth": True,
            "allow_non_loopback": False,
        }
    ]
    assert exposed.exit_code != 0
    assert "cannot run without authentication" in exposed.output
