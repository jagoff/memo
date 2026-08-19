# tests/test_proxy_cli.py
import json
from pathlib import Path

from click.testing import CliRunner

from memo.cli_proxy import proxy_group
from memo.ops_launchd import PROXY_LABEL, render_proxy_plist


def _env(tmp_path):
    return {
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_DATA_DIR": str(tmp_path / "data"),
        "MEMO_STATE_DIR": str(tmp_path / "state"),
        # `config_md.set_value` (used by `proxy on`/`proxy off`) resolves its
        # config home independently via MEMO_CONFIG_DIR — MEMO_DATA_DIR and
        # MEMO_STATE_DIR do not isolate it. Without this, `proxy off` writes
        # `proxy.enabled = false` into the developer's real
        # ~/.config/memo/config/advanced-config.md. See
        # test_off_never_touches_the_real_config_home below, which enforces
        # this instead of assuming it.
        "MEMO_CONFIG_DIR": str(tmp_path / "config"),
    }


def _snapshot(root: Path) -> dict[str, str]:
    """Contents of every file under `root`, keyed by relative path.

    Used to prove a CLI invocation did not touch the developer's real config
    home — a mtime check would miss a same-second write, a same-size check
    would miss a same-length value change; comparing full content catches
    both.
    """
    if not root.is_dir():
        return {}
    return {
        str(p.relative_to(root)): p.read_text(encoding="utf-8", errors="replace")
        for p in root.rglob("*")
        if p.is_file()
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
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        free_port = s.getsockname()[1]
    env = _env(tmp_path)
    env["MEMO_PROXY_PORT"] = str(free_port)
    result = CliRunner().invoke(proxy_group, ["status"], env=env)
    assert result.exit_code == 0
    assert "not running" in result.output.lower()


def test_off_writes_the_markdown_config_not_just_the_env(tmp_path):
    env = _env(tmp_path)
    runner = CliRunner()
    result = runner.invoke(proxy_group, ["off"], env=env)
    assert result.exit_code == 0
    assert "proxy.enabled" in result.output or "MEMO_PROXY_ENABLED" in result.output

    written = list((Path(env["MEMO_CONFIG_DIR"]) / "config").glob("*.md"))
    assert written, "proxy off must persist under the isolated MEMO_CONFIG_DIR"
    assert any("proxy" in p.read_text(encoding="utf-8") for p in written)


def test_off_never_touches_the_real_config_home(tmp_path):
    """Regression for the test-isolation defect: `config_md.set_value` reads
    MEMO_CONFIG_DIR from the environment independently of MEMO_DATA_DIR /
    MEMO_STATE_DIR. An env missing MEMO_CONFIG_DIR makes `proxy off` write
    `proxy.enabled = false` into the developer's real
    ~/.config/memo/config/advanced-config.md — silently disabling the proxy
    on the real machine and corrupting state that outlives the test process.
    This snapshots the real config home before and after the CLI call and
    asserts it is byte-for-byte unchanged, so isolation is enforced rather
    than assumed.
    """
    from memo.config_md import config_home

    real_home = config_home()  # no env override: the developer's real path
    before = _snapshot(real_home)

    env = _env(tmp_path)
    result = CliRunner().invoke(proxy_group, ["off"], env=env)
    assert result.exit_code == 0

    after = _snapshot(real_home)
    assert before == after, "proxy off must never write to the real config home"


def _tool(name: str) -> dict:
    return {
        "name": name,
        "description": f"description of {name} " * 10,
        "input_schema": {"type": "object", "properties": {}},
    }


def _write_payload(path: Path, tool_names: list[str]) -> None:
    payload = {
        "model": "claude-x",
        "tools": [_tool(n) for n in tool_names],
        "messages": [{"role": "user", "content": "hi"}],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_tool_savings_reports_how_many_tools_a_payload_would_lose(tmp_path):
    """The predictive counterpart to `memo tokens` (which reports what a
    proxy already measured from real traffic): this answers "what would
    ToolSchemas do to THIS payload" without sending anything anywhere."""
    payload_path = tmp_path / "payload.json"
    _write_payload(payload_path, ["Read", "Bash", "memo_search", "mcp__octocode__ghSearchCode"])
    env = _env(tmp_path)
    result = CliRunner().invoke(proxy_group, ["tool-savings", str(payload_path)], env=env)
    assert result.exit_code == 0, result.output
    assert "4" in result.output  # total tools
    assert "kept" in result.output.lower()
    assert "saved" in result.output.lower()


def test_tool_savings_never_makes_the_payload_more_expensive(tmp_path):
    payload_path = tmp_path / "payload.json"
    _write_payload(payload_path, ["Read", "Bash", "memo_search"])
    env = _env(tmp_path)
    result = CliRunner().invoke(proxy_group, ["tool-savings", str(payload_path)], env=env)
    assert result.exit_code == 0, result.output
    assert "-" not in result.output.split("saved")[-1].split("\n")[0]


def test_tool_savings_on_a_missing_file_is_a_clean_cli_error(tmp_path):
    env = _env(tmp_path)
    result = CliRunner().invoke(proxy_group, ["tool-savings", str(tmp_path / "nope.json")], env=env)
    assert result.exit_code != 0
    assert "Traceback" not in result.output


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
