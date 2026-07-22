"""`memo chat ask` (the cli_chat group twin) must log a consult exactly like
cli_search's `chat-ask` twin — otherwise trinity layers that shell out to
`memo chat ask` show up as silent in `memo usefulness` even though they read
memo. Mirrors test_cli_consult_attribution for the chat group surface."""

from __future__ import annotations

from types import SimpleNamespace

from click.testing import CliRunner

from memo.cli_chat import chat_group
from memo.config import Config
from memo.dashboard import read_recall_log


def _envelope() -> dict:
    return {
        "answer": "an answer",
        "sources": [{"id": "abc12345", "id_short": "abc12345", "title": "a note", "score": 1.1}],
    }


def _install_fake_mem(monkeypatch) -> None:
    fake = SimpleNamespace()
    fake.chat_ask = lambda *a, **k: _envelope()
    fake.chat_ask_stream = lambda *a, **k: iter(
        [
            {"event": "context", "sources": _envelope()["sources"]},
            {"event": "done", "answer": "an answer"},
        ]
    )
    monkeypatch.setattr("memo.cli_chat._get_memory", lambda cfg: fake)


def _env(tmp_cfg: Config) -> dict[str, str]:
    return {
        "MEMO_DATA_DIR": str(tmp_cfg.data_dir),
        "MEMO_STATE_DIR": str(tmp_cfg.state_dir),
        "MEMO_NONINTERACTIVE": "1",
    }


def test_chat_ask_logs_consult_nonstream(tmp_cfg: Config, monkeypatch) -> None:
    _install_fake_mem(monkeypatch)
    res = CliRunner().invoke(chat_group, ["ask", "what?", "--source", "synapse"], env=_env(tmp_cfg))
    assert res.exit_code == 0, res.output
    rows = read_recall_log(tmp_cfg.state_dir, limit=10)
    assert len(rows) == 1
    assert rows[0]["source"] == "synapse"
    assert rows[0]["via"] == "cli:chat_ask"
    assert len(rows[0]["hits"]) == 1


def test_chat_ask_logs_consult_stream(tmp_cfg: Config, monkeypatch) -> None:
    _install_fake_mem(monkeypatch)
    res = CliRunner().invoke(
        chat_group, ["ask", "--stream", "what?", "--source", "memflow"], env=_env(tmp_cfg)
    )
    assert res.exit_code == 0, res.output
    rows = read_recall_log(tmp_cfg.state_dir, limit=10)
    assert len(rows) == 1
    assert rows[0]["source"] == "memflow"
    assert rows[0]["via"] == "cli:chat_ask"
    assert len(rows[0]["hits"]) == 1


def test_chat_ask_silent_without_source(tmp_cfg: Config, monkeypatch) -> None:
    monkeypatch.delenv("MEMO_SOURCE", raising=False)
    _install_fake_mem(monkeypatch)
    res = CliRunner().invoke(chat_group, ["ask", "what?"], env=_env(tmp_cfg))
    assert res.exit_code == 0, res.output
    assert read_recall_log(tmp_cfg.state_dir, limit=10) == []
