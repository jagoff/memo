import pytest

from memo.chat.sessions import SessionStore


def test_roundtrip_and_listing(tmp_path) -> None:
    store = SessionStore(tmp_path)
    store.append_turn("s1", "user", "hola")
    store.append_turn("s1", "assistant", "respuesta")
    store.append_turn("s2", "user", "otra consulta")
    sessions = store.list_sessions()
    assert len(sessions) == 2
    assert sessions[0]["session_id"] == "s2"  # más reciente primero
    turns = store.get("s1")
    assert [t["role"] for t in turns] == ["user", "assistant"]
    assert store.recent_queries() == ["otra consulta", "hola"]


def test_delete(tmp_path) -> None:
    store = SessionStore(tmp_path)
    store.append_turn("s1", "user", "x")
    assert store.delete("s1") is True
    assert store.delete("s1") is False
    store.append_turn("a", "user", "1")
    store.append_turn("b", "user", "2")
    assert store.delete_all() == 2


def test_invalid_session_id_rejected(tmp_path) -> None:
    store = SessionStore(tmp_path)
    with pytest.raises(ValueError):
        store.append_turn("../evil", "user", "x")
    with pytest.raises(ValueError):
        store.get("a/b")
