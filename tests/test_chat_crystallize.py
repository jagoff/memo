import json
from types import SimpleNamespace

import pytest

from memo.chat.crystallize import crystallize_session
from memo.chat.sessions import SessionStore


class _FakeChatBackend:
    def __init__(self, content: str | None = None, raises: bool = False) -> None:
        self._content = content
        self._raises = raises
        self.calls: list = []

    def chat(self, model, messages, options=None):
        self.calls.append((model, messages, options))
        if self._raises:
            raise RuntimeError("mlx down")
        return {"message": {"content": self._content}}


class _FakeMemory:
    def __init__(self, tmp_path, chat_backend=None) -> None:
        self.cfg = SimpleNamespace(llm_model="fake-model", state_dir=tmp_path)
        self._chat_backend = chat_backend
        self.save_calls: list[dict] = []

    def _ensure_chat(self):
        return self._chat_backend

    def save(self, **kwargs):
        self.save_calls.append(kwargs)
        return SimpleNamespace(id="mem12345")


def _sessions_dir(tmp_path):
    return tmp_path / "chat" / "sessions"


def _seed_session(tmp_path, session_id="s1") -> None:
    store = SessionStore(_sessions_dir(tmp_path))
    store.append_turn(session_id, "user", "how do we ship the release?")
    store.append_turn(session_id, "assistant", "we tag, bump versions, and push.")


def test_valid_json_backend_saves_with_sections_and_tags(tmp_path) -> None:
    _seed_session(tmp_path)
    crystal_json = json.dumps(
        {
            "title": "Release process",
            "situation": "Discussed shipping the release.",
            "decisions": ["Tag before bumping versions"],
            "learnings": ["CHANGELOG must be updated first"],
            "goal_progress": ["release: cut and pushed"],
            "body": "We worked through the release checklist end to end.",
            "tags": ["release", "process"],
        }
    )
    backend = _FakeChatBackend(content=crystal_json)
    mem = _FakeMemory(tmp_path, chat_backend=backend)

    result = crystallize_session(mem, "s1")

    assert result["ok"] is True
    assert result["memory_id"] == "mem12345"
    assert len(mem.save_calls) == 1
    call = mem.save_calls[0]
    assert call["type_"] == "decision"
    assert "**Decisions:**" in call["content"]
    assert "- Tag before bumping versions" in call["content"]
    assert "**Learnings:**" in call["content"]
    assert "- CHANGELOG must be updated first" in call["content"]
    assert "**Goal progress:**" in call["content"]
    assert "- release: cut and pushed" in call["content"]
    assert call["tags"] == ["release", "process", "session-crystal"]
    assert len(call["content"]) <= 3000
    # LLM was called with the configured model + low-temp/short-output options.
    assert backend.calls[0][0] == "fake-model"
    assert backend.calls[0][2]["temperature"] == 0.1
    assert backend.calls[0][2]["max_tokens"] == 600


def test_tags_capped_at_eight_from_llm_plus_fixed_tag(tmp_path) -> None:
    _seed_session(tmp_path)
    crystal_json = json.dumps(
        {
            "title": "T",
            "situation": "",
            "decisions": [],
            "learnings": [],
            "goal_progress": [],
            "body": "body",
            "tags": [f"tag{i}" for i in range(10)],
        }
    )
    mem = _FakeMemory(tmp_path, chat_backend=_FakeChatBackend(content=crystal_json))

    result = crystallize_session(mem, "s1")

    assert result["ok"] is True
    tags = mem.save_calls[0]["tags"]
    assert tags == [f"tag{i}" for i in range(8)] + ["session-crystal"]


def test_backend_raises_falls_back_to_heuristic(tmp_path) -> None:
    _seed_session(tmp_path)
    mem = _FakeMemory(tmp_path, chat_backend=_FakeChatBackend(raises=True))

    result = crystallize_session(mem, "s1")

    assert result["ok"] is True
    crystal = result["crystal"]
    assert crystal["title"].startswith("Session s1")
    assert "2 turns" in crystal["body"]
    assert mem.save_calls[0]["tags"] == ["session", "session-crystal"]


def test_backend_returns_malformed_json_falls_back_to_heuristic(tmp_path) -> None:
    _seed_session(tmp_path)
    mem = _FakeMemory(tmp_path, chat_backend=_FakeChatBackend(content="not json at all"))

    result = crystallize_session(mem, "s1")

    assert result["ok"] is True
    assert result["crystal"]["title"].startswith("Session s1")


def test_dedup_window_skips_second_call_for_same_session(tmp_path) -> None:
    _seed_session(tmp_path)
    crystal_json = json.dumps({"title": "T", "body": "b", "tags": []})
    mem = _FakeMemory(tmp_path, chat_backend=_FakeChatBackend(content=crystal_json))

    first = crystallize_session(mem, "s1")
    assert first["ok"] is True
    assert len(mem.save_calls) == 1

    second = crystallize_session(mem, "s1")
    assert second == {
        "ok": False,
        "dry_run": False,
        "session_id": "s1",
        "crystal": None,
        "memory_id": None,
        "skipped": True,
        "error": None,
    }
    assert len(mem.save_calls) == 1  # no second save

    dedup_path = tmp_path / "chat" / "crystallize_last.json"
    assert dedup_path.exists()
    state = json.loads(dedup_path.read_text())
    assert "s1" in state


def test_dedup_window_does_not_block_a_different_session(tmp_path) -> None:
    _seed_session(tmp_path, session_id="s1")
    _seed_session(tmp_path, session_id="s2")
    crystal_json = json.dumps({"title": "T", "body": "b", "tags": []})
    mem = _FakeMemory(tmp_path, chat_backend=_FakeChatBackend(content=crystal_json))

    assert crystallize_session(mem, "s1")["ok"] is True
    second = crystallize_session(mem, "s2")
    assert second["ok"] is True
    assert len(mem.save_calls) == 2


def test_missing_session_returns_clean_error(tmp_path) -> None:
    mem = _FakeMemory(tmp_path, chat_backend=_FakeChatBackend(content="{}"))

    result = crystallize_session(mem, "does-not-exist")

    assert result["ok"] is False
    assert result["skipped"] is False
    assert result["error"] is not None
    assert "does-not-exist" in result["error"]
    assert mem.save_calls == []


def test_malformed_session_id_returns_clean_error(tmp_path) -> None:
    """A session id with characters SessionStore rejects (e.g. a path separator)
    must produce a clean error dict, not an uncaught ValueError."""
    mem = _FakeMemory(tmp_path, chat_backend=_FakeChatBackend(content="{}"))

    result = crystallize_session(mem, "../evil")

    assert result["ok"] is False
    assert result["error"] is not None
    assert mem.save_calls == []


def test_no_session_id_and_no_sessions_returns_clean_error(tmp_path) -> None:
    mem = _FakeMemory(tmp_path, chat_backend=_FakeChatBackend(content="{}"))

    result = crystallize_session(mem, None)

    assert result["ok"] is False
    assert result["error"] is not None
    assert mem.save_calls == []


def test_no_session_id_picks_most_recent_session(tmp_path) -> None:
    _seed_session(tmp_path, session_id="older")
    store = SessionStore(_sessions_dir(tmp_path))
    store.append_turn("newer", "user", "most recent question")
    crystal_json = json.dumps({"title": "T", "body": "b", "tags": []})
    mem = _FakeMemory(tmp_path, chat_backend=_FakeChatBackend(content=crystal_json))

    result = crystallize_session(mem, None)

    assert result["ok"] is True
    assert result["session_id"] == "newer"


def test_dry_run_does_not_save(tmp_path) -> None:
    _seed_session(tmp_path)
    crystal_json = json.dumps({"title": "Preview", "body": "preview body", "tags": ["x"]})
    mem = _FakeMemory(tmp_path, chat_backend=_FakeChatBackend(content=crystal_json))

    result = crystallize_session(mem, "s1", dry_run=True)

    assert result["ok"] is True
    assert result["dry_run"] is True
    assert result["crystal"]["title"] == "Preview"
    assert result["memory_id"] is None
    assert mem.save_calls == []
    dedup_path = tmp_path / "chat" / "crystallize_last.json"
    assert not dedup_path.exists()


def test_n_turns_limits_transcript_window(tmp_path) -> None:
    store = SessionStore(_sessions_dir(tmp_path))
    for i in range(5):
        store.append_turn("s1", "user", f"message {i}")
    backend = _FakeChatBackend(content=json.dumps({"title": "T", "body": "b", "tags": []}))
    mem = _FakeMemory(tmp_path, chat_backend=backend)

    crystallize_session(mem, "s1", n_turns=2)

    prompt = backend.calls[0][1][0]["content"]
    assert "message 3" in prompt
    assert "message 4" in prompt
    assert "message 0" not in prompt


@pytest.mark.parametrize("session_id", ["s1"])
def test_invalid_llm_backend_none_raises_falls_back(tmp_path, session_id) -> None:
    """_ensure_chat() itself raising must also fall back to the heuristic."""
    _seed_session(tmp_path, session_id=session_id)

    class _BoomMemory(_FakeMemory):
        def _ensure_chat(self):
            raise RuntimeError("no mlx runtime")

    mem = _BoomMemory(tmp_path)
    result = crystallize_session(mem, session_id)

    assert result["ok"] is True
    assert result["crystal"]["title"].startswith(f"Session {session_id}")
