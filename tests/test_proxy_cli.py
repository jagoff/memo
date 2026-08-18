# tests/test_proxy_cli.py
from click.testing import CliRunner

from memo.cli_proxy import proxy_group
from memo.ops_launchd import PROXY_LABEL, render_proxy_plist


def _env(tmp_path):
    return {
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_DATA_DIR": str(tmp_path / "data"),
        "MEMO_STATE_DIR": str(tmp_path / "state"),
    }


def test_plist_uses_the_proxy_label_and_port():
    xml = render_proxy_plist("/usr/local/bin/memo", "/Users/x", port=8768)
    assert f"<string>{PROXY_LABEL}</string>" in xml
    assert "<string>8768</string>" in xml
    assert "<key>KeepAlive</key>" in xml


def test_plist_never_embeds_an_api_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-not-appear")
    xml = render_proxy_plist("/usr/local/bin/memo", "/Users/x")
    assert "sk-should-not-appear" not in xml


def test_status_reports_not_running_without_a_daemon(tmp_path):
    result = CliRunner().invoke(proxy_group, ["status"], env=_env(tmp_path))
    assert result.exit_code == 0
    assert "not running" in result.output.lower()


def test_off_writes_the_markdown_config_not_just_the_env(tmp_path):
    runner = CliRunner()
    result = runner.invoke(proxy_group, ["off"], env=_env(tmp_path))
    assert result.exit_code == 0
    assert "proxy.enabled" in result.output or "MEMO_PROXY_ENABLED" in result.output


def test_serve_without_the_http_extra_is_a_clean_cli_error(tmp_path, monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name in ("fastapi", "uvicorn", "httpx"):
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    result = CliRunner().invoke(proxy_group, ["serve"], env=_env(tmp_path))
    assert result.exit_code != 0
    assert "pip install" in result.output or "extra" in result.output.lower()
    assert "Traceback" not in result.output