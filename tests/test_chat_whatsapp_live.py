import sqlite3
from pathlib import Path

import pytest

from memo.chat import whatsapp_live as wa


def _make_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE chats (jid TEXT PRIMARY KEY, name TEXT)")
    conn.execute(
        "CREATE TABLE messages "
        "(chat_jid TEXT, sender TEXT, is_from_me INTEGER, content TEXT, timestamp TEXT)"
    )
    conn.commit()
    conn.close()


def _add_chat(path: Path, jid: str, name: str) -> None:
    conn = sqlite3.connect(path)
    conn.execute("INSERT INTO chats (jid, name) VALUES (?, ?)", (jid, name))
    conn.commit()
    conn.close()


def _add_message(
    path: Path, jid: str, *, sender: str | None, is_from_me: bool, content: str, ts: str
) -> None:
    conn = sqlite3.connect(path)
    conn.execute(
        "INSERT INTO messages (chat_jid, sender, is_from_me, content, timestamp) "
        "VALUES (?, ?, ?, ?, ?)",
        (jid, sender, int(is_from_me), content, ts),
    )
    conn.commit()
    conn.close()


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    path = tmp_path / "messages.db"
    _make_db(path)
    return path


# ── bridge_db_path ───────────────────────────────────────────────────────


def test_bridge_db_path_defaults_to_bridge_store(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MEMO_WHATSAPP_DB", raising=False)
    assert (
        wa.bridge_db_path()
        == Path("~/repos/whatsapp-mcp/whatsapp-bridge/store/messages.db").expanduser()
    )


def test_bridge_db_path_honors_env_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    custom = tmp_path / "custom.db"
    monkeypatch.setenv("MEMO_WHATSAPP_DB", str(custom))
    assert wa.bridge_db_path() == custom


# ── resolve_chats ────────────────────────────────────────────────────────


def test_resolve_chats_contacts_index_takes_priority_over_name_match(db_path: Path) -> None:
    # A decoy chat also contains the token "monica" in its name; if the
    # contacts index weren't checked first, name-token matching would return
    # (or add) the decoy too.
    _add_chat(db_path, "correct-jid@lid", "Monica 🕉️")
    _add_chat(db_path, "decoy-jid@s.whatsapp.net", "Monica Random Group")
    contacts_index = {"monica": "correct-jid@lid"}

    result = wa.resolve_chats("qué me dijo monica", db_path, contacts_index)

    assert result == [("correct-jid@lid", "Monica 🕉️")]


def test_resolve_chats_matches_by_significant_chat_name_token(db_path: Path) -> None:
    _add_chat(db_path, "ana@s.whatsapp.net", "Ana Lopez")
    _add_chat(db_path, "bob@s.whatsapp.net", "Bob Smith")

    result = wa.resolve_chats("qué me dijo ana hoy?", db_path, {})

    assert result == [("ana@s.whatsapp.net", "Ana Lopez")]


def test_resolve_chats_excludes_busy_group(db_path: Path) -> None:
    _add_chat(db_path, "team@g.us", "Team Group")
    _add_message(
        db_path,
        "team@g.us",
        sender="111@s.whatsapp.net",
        is_from_me=False,
        content="hi",
        ts="2020-01-01 00:00:00",
    )
    _add_message(
        db_path,
        "team@g.us",
        sender="222@s.whatsapp.net",
        is_from_me=False,
        content="hey",
        ts="2020-01-01 00:01:00",
    )

    result = wa.resolve_chats("qué me dijo team", db_path, {})

    assert result == []


def test_resolve_chats_includes_group_with_single_other_sender(db_path: Path) -> None:
    _add_chat(db_path, "duo@g.us", "Duo Group")
    _add_message(
        db_path,
        "duo@g.us",
        sender="111@s.whatsapp.net",
        is_from_me=False,
        content="hi",
        ts="2020-01-01 00:00:00",
    )
    _add_message(
        db_path,
        "duo@g.us",
        sender="111@s.whatsapp.net",
        is_from_me=False,
        content="hey",
        ts="2020-01-01 00:01:00",
    )

    result = wa.resolve_chats("mensajes de duo", db_path, {})

    assert result == [("duo@g.us", "Duo Group")]


def test_resolve_chats_returns_empty_when_nothing_matches(db_path: Path) -> None:
    _add_chat(db_path, "ana@s.whatsapp.net", "Ana Lopez")

    assert wa.resolve_chats("qué me dijo carlos", db_path, {}) == []


def test_resolve_chats_query_stopwords_do_not_match_a_chat_name_token(db_path: Path) -> None:
    # Regression: "Mensajes Colegio" has the significant-looking token
    # "colegio" AND "mensajes" — but "mensajes" is a query stopword (it's
    # part of how a recency query talks about itself, not a person/place
    # name) and must never, on its own, match a chat whose name happens to
    # contain it. Without the full stopword set this chat wrongly matched
    # "cuáles son los últimos mensajes de Ana" via the "mensajes" token.
    _add_chat(db_path, "colegio@g.us", "Mensajes Colegio")

    result = wa.resolve_chats("cuáles son los últimos mensajes de Ana", db_path, {})

    assert result == []


def test_resolve_chats_stopwords_do_not_kill_legitimate_name_tokens(db_path: Path) -> None:
    # "colegio" itself is NOT a stopword — a chat legitimately named after a
    # place/topic must still match when that word is the one doing the work.
    _add_chat(db_path, "colegio-norte@s.whatsapp.net", "Colegio Norte")

    result = wa.resolve_chats("qué dijeron en el colegio?", db_path, {})

    assert result == [("colegio-norte@s.whatsapp.net", "Colegio Norte")]


def test_resolve_chats_missing_db_returns_empty_list(tmp_path: Path) -> None:
    missing = tmp_path / "nope.db"

    assert wa.resolve_chats("ana", missing, {}) == []


# ── last_messages ────────────────────────────────────────────────────────


def test_last_messages_returns_chronological_order(db_path: Path) -> None:
    jid = "ana@s.whatsapp.net"
    _add_chat(db_path, jid, "Ana")
    _add_message(
        db_path, jid, sender=None, is_from_me=True, content="first", ts="2020-01-01 10:00:00"
    )
    _add_message(
        db_path, jid, sender=None, is_from_me=False, content="second", ts="2020-01-01 10:05:00"
    )
    _add_message(
        db_path, jid, sender=None, is_from_me=True, content="third", ts="2020-01-01 10:10:00"
    )

    result = wa.last_messages(db_path, jid, limit=10)

    assert [m["content"] for m in result] == ["first", "second", "third"]
    assert result[0]["is_from_me"] is True
    assert result[1]["is_from_me"] is False


def test_last_messages_limit_keeps_newest_n_in_chronological_order(db_path: Path) -> None:
    jid = "ana@s.whatsapp.net"
    _add_chat(db_path, jid, "Ana")
    for i, content in enumerate(["a", "b", "c", "d"]):
        _add_message(
            db_path,
            jid,
            sender=None,
            is_from_me=False,
            content=content,
            ts=f"2020-01-01 10:0{i}:00",
        )

    result = wa.last_messages(db_path, jid, limit=2)

    assert [m["content"] for m in result] == ["c", "d"]


def test_last_messages_today_only_filters_to_current_local_date(db_path: Path) -> None:
    jid = "ana@s.whatsapp.net"
    _add_chat(db_path, jid, "Ana")
    _add_message(
        db_path, jid, sender=None, is_from_me=False, content="old", ts="2000-01-01 00:00:00"
    )
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO messages (chat_jid, sender, is_from_me, content, timestamp) "
        "VALUES (?, NULL, 0, 'fresh', datetime('now'))",
        (jid,),
    )
    conn.commit()
    conn.close()

    result = wa.last_messages(db_path, jid, limit=10, today_only=True)

    assert [m["content"] for m in result] == ["fresh"]


def test_last_messages_clamps_limit_to_max_100_when_not_today_only(db_path: Path) -> None:
    jid = "ana@s.whatsapp.net"
    _add_chat(db_path, jid, "Ana")
    conn = sqlite3.connect(db_path)
    conn.executemany(
        "INSERT INTO messages (chat_jid, sender, is_from_me, content, timestamp) "
        "VALUES (?, NULL, 0, ?, ?)",
        [(jid, f"m{i}", f"2020-01-01 {i // 60:02d}:{i % 60:02d}:00") for i in range(150)],
    )
    conn.commit()
    conn.close()

    result = wa.last_messages(db_path, jid, limit=500)

    assert len(result) == 100


def test_last_messages_clamps_limit_to_minimum_one(db_path: Path) -> None:
    jid = "ana@s.whatsapp.net"
    _add_chat(db_path, jid, "Ana")
    _add_message(
        db_path, jid, sender=None, is_from_me=False, content="only", ts="2020-01-01 00:00:00"
    )

    result = wa.last_messages(db_path, jid, limit=0)

    assert len(result) == 1


def test_last_messages_excludes_blank_content(db_path: Path) -> None:
    jid = "ana@s.whatsapp.net"
    _add_chat(db_path, jid, "Ana")
    _add_message(
        db_path, jid, sender=None, is_from_me=False, content="   ", ts="2020-01-01 00:00:00"
    )
    _add_message(
        db_path, jid, sender=None, is_from_me=False, content="real", ts="2020-01-01 00:01:00"
    )

    result = wa.last_messages(db_path, jid)

    assert [m["content"] for m in result] == ["real"]


def test_last_messages_missing_db_returns_empty_list(tmp_path: Path) -> None:
    missing = tmp_path / "nope.db"

    assert wa.last_messages(missing, "ana@s.whatsapp.net") == []


# ── format_transcript ────────────────────────────────────────────────────


def test_format_transcript_labels_sender_and_me() -> None:
    msgs = [
        {"ts": "2020-01-01 10:00:00", "is_from_me": False, "content": "hola"},
        {"ts": "2020-01-01 10:01:00", "is_from_me": True, "content": "hey"},
    ]

    text = wa.format_transcript("Ana", msgs)

    assert text == "[2020-01-01 10:00:00] Ana: hola\n[2020-01-01 10:01:00] yo: hey"


def test_format_transcript_empty_messages_is_empty_string() -> None:
    assert wa.format_transcript("Ana", []) == ""


# ── recency_conversation_intent ─────────────────────────────────────────


@pytest.mark.parametrize(
    "q",
    [
        "cuáles son los últimos mensajes de Ana",
        "última conversación con Ana",
        "qué me dijo Ana",
        "qué dijo Ana",
        "conversación con Ana",
        "last messages from Ana",
        "what did Ana say about the trip",
    ],
)
def test_recency_conversation_intent_true_cases(q: str) -> None:
    assert wa.recency_conversation_intent(q) is True


@pytest.mark.parametrize("q", ["cuál es el clima hoy", "resumime el proyecto memo"])
def test_recency_conversation_intent_false_cases(q: str) -> None:
    assert wa.recency_conversation_intent(q) is False


def test_recency_conversation_intent_handles_none() -> None:
    assert wa.recency_conversation_intent(None) is False  # type: ignore[arg-type]


# ── singular_last_intent ─────────────────────────────────────────────────


def test_singular_last_intent_true_for_singular_phrasing() -> None:
    assert wa.singular_last_intent("cuál es el último mensaje de Ana") is True
    assert wa.singular_last_intent("last message from Ana") is True


def test_singular_last_intent_false_when_plural_marker_present() -> None:
    assert wa.singular_last_intent("cuáles son los últimos mensajes") is False
    assert wa.singular_last_intent("last messages from Ana") is False


def test_singular_last_intent_false_when_no_singular_phrase() -> None:
    assert wa.singular_last_intent("qué dijo Ana hoy") is False


# ── today_only_intent ─────────────────────────────────────────────────────


def test_today_only_intent_true_for_hoy_and_today() -> None:
    assert wa.today_only_intent("qué me dijo Ana hoy") is True
    assert wa.today_only_intent("messages from today") is True


def test_today_only_intent_false_without_marker() -> None:
    assert wa.today_only_intent("qué me dijo Ana") is False
