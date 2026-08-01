from pathlib import Path

from memo.chat.config import ChatConfig


def test_defaults_match_production(tmp_path: Path) -> None:
    cfg = ChatConfig.load(tmp_path)
    assert cfg.base_k == 20
    assert cfg.relevance_floor == 0.25
    assert cfg.vote_boost == 1.5
    assert cfg.semantic_threshold == 0.75
    assert cfg.multi_query is True
    assert cfg.multi_query_n == 2
    assert cfg.fulldoc is True
    assert cfg.answer_max_tokens == 1200
    assert cfg.synth_head == 8
    assert cfg.feedback_dir == tmp_path / "chat" / "feedback"
    assert cfg.sessions_dir == tmp_path / "chat" / "sessions"
    assert cfg.whatsapp_live is True
    assert cfg.contacts_dir is None


def test_env_overrides(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MEMO_CHAT_BASE_K", "5")
    monkeypatch.setenv("MEMO_CHAT_MULTI_QUERY", "0")
    monkeypatch.setenv("MEMO_CHAT_RELEVANCE_FLOOR", "0.4")
    monkeypatch.setenv("MEMO_CHAT_WHATSAPP_LIVE", "0")
    monkeypatch.setenv("MEMO_CHAT_CONTACTS_DIR", "/tmp/contacts")
    cfg = ChatConfig.load(tmp_path)
    assert cfg.base_k == 5
    assert cfg.multi_query is False
    assert cfg.relevance_floor == 0.4
    assert cfg.whatsapp_live is False
    assert cfg.contacts_dir == Path("/tmp/contacts")
