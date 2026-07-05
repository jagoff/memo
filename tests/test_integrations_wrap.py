"""memo.integrations.wrap — transparent memory for raw LLM SDK clients."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest


def _fake_openai():
    calls: list = []

    class _Completions:
        def create(self, **kwargs):
            calls.append(kwargs)
            msg = SimpleNamespace(content="answer text")
            return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

    return SimpleNamespace(chat=SimpleNamespace(completions=_Completions())), calls


def _fake_anthropic():
    calls: list = []

    class _Messages:
        def create(self, **kwargs):
            calls.append(kwargs)
            block = SimpleNamespace(type="text", text="answer text")
            return SimpleNamespace(content=[block])

    return SimpleNamespace(messages=_Messages()), calls


def test_wrap_rejects_unknown_client(tmp_cfg):
    from memo.integrations import wrap

    with pytest.raises(TypeError, match="unsupported client"):
        wrap(object(), cfg=tmp_cfg)


def test_openai_precall_injects_recall_block(tmp_cfg, monkeypatch):
    from memo.integrations import wrap

    canned = {"results": [{"id": "a" * 32, "title": "Known fact", "body": "the port is 9090"}]}
    monkeypatch.setattr(
        "memo.integrations.wrap.connect_and_send",
        lambda state_dir, payload, timeout=2.0: json.dumps(canned),
    )
    client, calls = _fake_openai()
    wrap(client, cfg=tmp_cfg, capture=False)
    client.chat.completions.create(
        model="gpt-x", messages=[{"role": "user", "content": "what port?"}]
    )
    sent = calls[0]["messages"]
    assert sent[0]["role"] == "system"
    assert "Known fact" in sent[0]["content"]
    assert sent[1] == {"role": "user", "content": "what port?"}


def test_anthropic_precall_appends_to_system(tmp_cfg, monkeypatch):
    from memo.integrations import wrap

    canned = {"results": [{"id": "b" * 32, "title": "Pref", "body": "user prefers rg"}]}
    monkeypatch.setattr(
        "memo.integrations.wrap.connect_and_send",
        lambda state_dir, payload, timeout=2.0: json.dumps(canned),
    )
    client, calls = _fake_anthropic()
    wrap(client, cfg=tmp_cfg, capture=False)
    client.messages.create(
        model="claude-x", system="be terse", messages=[{"role": "user", "content": "search cmd?"}]
    )
    assert calls[0]["system"].startswith("be terse")
    assert "Pref" in calls[0]["system"]


def test_wrap_fails_open_when_daemon_down(tmp_cfg, monkeypatch):
    from memo.integrations import wrap

    monkeypatch.setattr("memo.integrations.wrap.connect_and_send", lambda *a, **k: None)
    client, calls = _fake_openai()
    wrap(client, cfg=tmp_cfg, capture=False)
    client.chat.completions.create(model="m", messages=[{"role": "user", "content": "hola"}])
    assert calls[0]["messages"] == [{"role": "user", "content": "hola"}]
