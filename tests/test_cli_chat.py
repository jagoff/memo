"""`memo chat ask` (the cli_chat group twin) must log a consult exactly like
cli_search's `chat-ask` twin — otherwise trinity layers that shell out to
`memo chat ask` show up as silent in `memo usefulness` even though they read
memo. Mirrors test_cli_consult_attribution for the chat group surface."""

from __future__ import annotations

import json
from types import SimpleNamespace

from click.testing import CliRunner

from memo.chat.sessions import SessionStore
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


class _CrystallizeChatBackend:
    def chat(self, model, messages, options=None):
        return {
            "message": {
                "content": json.dumps(
                    {"title": "Preview title", "body": "preview body", "tags": ["x"]}
                )
            }
        }


class _CrystallizeMemory:
    """Fake memory for `chat crystallize` CLI tests — save() must not be called
    under --dry-run."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.save_calls: list[dict] = []

    def _ensure_chat(self):
        return _CrystallizeChatBackend()

    def save(self, **kwargs):
        self.save_calls.append(kwargs)
        return SimpleNamespace(id="mem-cli")


def test_chat_crystallize_dry_run_does_not_save(tmp_cfg: Config, monkeypatch) -> None:
    store = SessionStore(tmp_cfg.state_dir / "chat" / "sessions")
    store.append_turn("s1", "user", "hola")
    store.append_turn("s1", "assistant", "respuesta")

    fake_memory = _CrystallizeMemory(tmp_cfg)
    monkeypatch.setattr("memo.cli_chat._get_memory", lambda cfg: fake_memory)

    res = CliRunner().invoke(chat_group, ["crystallize", "s1", "--dry-run"], env=_env(tmp_cfg))

    assert res.exit_code == 0, res.output
    assert fake_memory.save_calls == []
    payload = json.loads(res.output)
    assert payload["ok"] is True
    assert payload["dry_run"] is True
    assert payload["crystal"]["title"] == "Preview title"
    assert payload["memory_id"] is None


def test_chat_crystallize_without_dry_run_saves(tmp_cfg: Config, monkeypatch) -> None:
    store = SessionStore(tmp_cfg.state_dir / "chat" / "sessions")
    store.append_turn("s1", "user", "hola")
    store.append_turn("s1", "assistant", "respuesta")

    fake_memory = _CrystallizeMemory(tmp_cfg)
    monkeypatch.setattr("memo.cli_chat._get_memory", lambda cfg: fake_memory)

    res = CliRunner().invoke(chat_group, ["crystallize", "s1"], env=_env(tmp_cfg))

    assert res.exit_code == 0, res.output
    assert len(fake_memory.save_calls) == 1
    payload = json.loads(res.output)
    assert payload["ok"] is True
    assert payload["memory_id"] == "mem-cli"
