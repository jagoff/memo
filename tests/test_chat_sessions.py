import json

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


@pytest.mark.parametrize(
    ("role", "text"),
    [
        (1, "text"),
        ("user", 1),
        ("\ud800", "text"),
        ("user", "\ud800"),
    ],
)
def test_append_rejects_invalid_text_before_touching_file(tmp_path, role, text) -> None:
    store = SessionStore(tmp_path)

    with pytest.raises(ValueError):
        store.append_turn("s1", role, text)

    assert not store._path("s1").exists()


def test_append_exchange_validates_both_turns_before_touching_file(tmp_path) -> None:
    store = SessionStore(tmp_path)

    with pytest.raises(ValueError):
        store.append_exchange("s1", "valid question", "invalid \ud800 answer")

    assert not store._path("s1").exists()


def test_append_exchange_persists_adjacent_turns(tmp_path) -> None:
    store = SessionStore(tmp_path)

    store.append_exchange("s1", "question", "answer")

    turns = store.get("s1")
    assert [(turn["role"], turn["text"]) for turn in turns] == [
        ("user", "question"),
        ("assistant", "answer"),
    ]


def test_get_skips_non_dict_lines(tmp_path) -> None:
    store = SessionStore(tmp_path)
    store.append_turn("s1", "user", "hola")
    path = store._path("s1")
    # A shape-corrupt line (valid JSON, not an object) must be skipped, not
    # returned as a turn — downstream code assumes every turn is a dict.
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(["not", "a", "dict"]) + "\n")
        fh.write(json.dumps("also not a dict") + "\n")
    store.append_turn("s1", "assistant", "respuesta")

    turns = store.get("s1")
    assert [t["role"] for t in turns] == ["user", "assistant"]


@pytest.mark.parametrize(
    "corrupt_turn",
    [
        {},
        {"role": "user", "ts": 1.0},
        {"text": "x", "ts": 1.0},
        {"role": [], "text": "x", "ts": 1.0},
        {"role": "user", "text": {}, "ts": 1.0},
        {"role": "user", "text": "x", "ts": "yesterday"},
        {"role": "user", "text": "x", "ts": True},
        {"role": "user", "text": "x", "ts": float("nan")},
        {"role": "user", "text": "x", "ts": -1},
        {"role": "user", "text": "x", "ts": 10**20},
        {"role": "\ud800", "text": "x", "ts": 1.0},
        {"role": "user", "text": "\ud800", "ts": 1.0},
    ],
)
def test_readers_skip_shape_corrupt_turn_objects(tmp_path, corrupt_turn) -> None:
    store = SessionStore(tmp_path)
    store.append_turn("s1", "user", "before")
    with store._path("s1").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(corrupt_turn, ensure_ascii=True) + "\n")
    store.append_turn("s1", "assistant", "after")

    assert [turn["text"] for turn in store.get("s1")] == ["before", "after"]
    assert [turn["text"] for turn in store.get_recent("s1", limit=2)] == ["before", "after"]


def test_get_recent_reads_tail_and_skips_malformed_lines(tmp_path) -> None:
    store = SessionStore(tmp_path)
    for index in range(20):
        store.append_turn("s1", "user", f"turn-{index}")
    with store._path("s1").open("a", encoding="utf-8") as fh:
        fh.write("not-json\n")
    store.append_turn("s1", "assistant", "last")

    turns = store.get_recent("s1", limit=3)

    assert [turn["text"] for turn in turns] == ["turn-18", "turn-19", "last"]


def test_get_recent_parses_each_long_tail_line_once(tmp_path, monkeypatch) -> None:
    store = SessionStore(tmp_path)
    for index in range(30):
        store.append_turn("s1", "user", f"turn-{index}-" + "x" * 9000)
    original_parse = SessionStore._parse_turn
    parse_calls = 0

    def spy_parse(raw_line):
        nonlocal parse_calls
        parse_calls += 1
        return original_parse(raw_line)

    monkeypatch.setattr(SessionStore, "_parse_turn", staticmethod(spy_parse))

    turns = store.get_recent("s1", limit=12)

    assert [turn["text"].split("-", 2)[:2] for turn in turns] == [
        ["turn", str(index)] for index in range(18, 30)
    ]
    assert parse_calls <= 13


def test_readers_skip_json_integer_beyond_interpreter_limit(tmp_path) -> None:
    store = SessionStore(tmp_path)
    store.append_turn("s1", "user", "before")
    huge_integer = "9" * 5000
    with store._path("s1").open("a", encoding="utf-8") as fh:
        fh.write(f'{{"role":"user","text":"bad","ts":{huge_integer}}}\n')
    store.append_turn("s1", "assistant", "after")

    assert [turn["text"] for turn in store.get("s1")] == ["before", "after"]
    assert [turn["text"] for turn in store.get_recent("s1", limit=2)] == ["before", "after"]


def test_listing_skips_invalid_filename_and_per_file_io_errors(tmp_path, monkeypatch) -> None:
    store = SessionStore(tmp_path)
    store.append_turn("valid", "user", "keep")
    (tmp_path / "bad id!.jsonl").write_text("{}\n", encoding="utf-8")
    (tmp_path / "unreadable.jsonl").write_text("{}\n", encoding="utf-8")
    (tmp_path / "unstatable.jsonl").touch()
    original_get = SessionStore.get
    original_stat = type(tmp_path).stat

    def sometimes_unreadable(self, session_id):
        if session_id == "unreadable":
            raise OSError("unreadable")
        return original_get(self, session_id)

    def sometimes_unstatable(path, *args, **kwargs):
        if path.name == "unstatable.jsonl":
            raise OSError("unstatable")
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(SessionStore, "get", sometimes_unreadable)
    monkeypatch.setattr(type(tmp_path), "stat", sometimes_unstatable)

    sessions = store.list_sessions()

    assert [row["session_id"] for row in sessions] == ["valid"]


@pytest.mark.parametrize("session_id", [None, 1, False, [], {}, "../evil", "a/b", ""])
def test_validate_id_is_strict_and_has_no_io(tmp_path, session_id) -> None:
    store = SessionStore(tmp_path)

    with pytest.raises(ValueError):
        store.validate_id(session_id)

    assert list(tmp_path.iterdir()) == []


def test_invalid_session_id_rejected(tmp_path) -> None:
    store = SessionStore(tmp_path)
    with pytest.raises(ValueError):
        store.append_turn("../evil", "user", "x")
    with pytest.raises(ValueError):
        store.get("a/b")
