"""F4: the `memo ask` / `memo chat-ask` CLIs default `--snippet-chars` to the
`None` sentinel and forward it unchanged, so `MEMO_ASK_SNIPPET_CHARS` resolution
stays in `Memory`. An explicit `--snippet-chars` is forwarded verbatim.
"""

from __future__ import annotations

from types import SimpleNamespace

from click.testing import CliRunner

from memo.cli_search import ask, chat_ask
from memo.config import Config


def _env(tmp_cfg: Config) -> dict[str, str]:
    return {
        "MEMO_DATA_DIR": str(tmp_cfg.data_dir),
        "MEMO_STATE_DIR": str(tmp_cfg.state_dir),
        "MEMO_NONINTERACTIVE": "1",
    }


def _install_fake_mem(monkeypatch, captured: dict) -> None:
    fake = SimpleNamespace()

    def _ask(question, **kw):
        captured["ask"] = kw.get("snippet_chars", "MISSING")
        return {"answer": "an answer", "sources": []}

    def _chat_ask(question, **kw):
        captured["chat_ask"] = kw.get("snippet_chars", "MISSING")
        return {"answer": "an answer", "sources": []}

    fake.ask = _ask
    fake.chat_ask = _chat_ask
    monkeypatch.setattr("memo.cli_search._get_memory", lambda cfg: fake)
    monkeypatch.setattr("memo.cli_search.log_cli_consult", lambda *a, **k: None)


def test_ask_cli_forwards_none_when_snippet_chars_unset(tmp_cfg: Config, monkeypatch):
    captured: dict = {}
    _install_fake_mem(monkeypatch, captured)
    res = CliRunner().invoke(ask, ["question"], env=_env(tmp_cfg))
    assert res.exit_code == 0, res.output
    assert captured["ask"] is None


def test_ask_cli_forwards_explicit_snippet_chars(tmp_cfg: Config, monkeypatch):
    captured: dict = {}
    _install_fake_mem(monkeypatch, captured)
    res = CliRunner().invoke(ask, ["question", "--snippet-chars", "250"], env=_env(tmp_cfg))
    assert res.exit_code == 0, res.output
    assert captured["ask"] == 250


def test_chat_ask_cli_forwards_none_when_snippet_chars_unset(tmp_cfg: Config, monkeypatch):
    captured: dict = {}
    _install_fake_mem(monkeypatch, captured)
    res = CliRunner().invoke(chat_ask, ["question"], env=_env(tmp_cfg))
    assert res.exit_code == 0, res.output
    assert captured["chat_ask"] is None


def test_chat_ask_cli_forwards_explicit_snippet_chars(tmp_cfg: Config, monkeypatch):
    captured: dict = {}
    _install_fake_mem(monkeypatch, captured)
    res = CliRunner().invoke(chat_ask, ["question", "--snippet-chars", "175"], env=_env(tmp_cfg))
    assert res.exit_code == 0, res.output
    assert captured["chat_ask"] == 175
