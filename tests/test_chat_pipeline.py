from types import SimpleNamespace

from memo.chat import whatsapp_live
from memo.chat.pipeline import chat_stream


class _FakeRecord(SimpleNamespace):
    pass


class _FakeChatBackend:
    def chat(self, model, messages, options=None):  # multi_query expansion
        return {"message": {"content": '{"variants": []}'}}

    def chat_stream(self, model, messages, options=None):
        yield "respuesta "
        yield "sintetizada"


class _FakeEmbedder:
    def embed_query(self, q):
        return [1.0, 0.0]


class _FakeMemory:
    def __init__(self, tmp_path):
        self.cfg = SimpleNamespace(llm_model="fake-model", state_dir=tmp_path)
        self.embedder = _FakeEmbedder()

    def search(self, query, *, limit=None, mode="hybrid", **kw):
        return [
            _FakeRecord(
                id="m1",
                title="Nota uno",
                type="note",
                score=0.9,
                body="cuerpo de la nota uno",
                path="notes/uno.md",
            ),
        ]

    def repo_search(self, query, *, limit=10, **kw):
        return [
            SimpleNamespace(
                id="r1",
                repo_name="vault",
                path="docs/dos.md",
                score=0.7,
                text="texto del vault",
                locator="repo:vault:docs/dos.md:1-10@abcd1234",
            ),
        ]

    def repo_get_file(self, repo, path, *, start=None, end=None):
        return None

    def _ensure_chat(self):
        return _FakeChatBackend()


def test_event_sequence_and_shapes(tmp_path) -> None:
    events = list(chat_stream(_FakeMemory(tmp_path), "qué sabés de la nota uno?"))
    kinds = [e["type"] for e in events]
    assert kinds[0] == "stage"
    assert "context" in kinds and "token" in kinds
    assert kinds[-1] == "done"

    # web-chat/src/api.ts StreamEvent expects stage events shaped
    # {name: StageEventName, phase: "start"|"done", ms?}, not {stage: str}.
    stages = [e for e in events if e["type"] == "stage"]
    assert stages[0]["name"] == "memo_retrieval"
    assert stages[0]["phase"] == "start"
    retrieval_done = next(
        s for s in stages if s["name"] == "memo_retrieval" and s["phase"] == "done"
    )
    assert retrieval_done["ms"] >= 0
    streaming_start = next(s for s in stages if s["name"] == "streaming" and s["phase"] == "start")
    assert streaming_start["phase"] == "start"
    streaming_done = next(s for s in stages if s["name"] == "streaming" and s["phase"] == "done")
    assert streaming_done["ms"] >= 0

    context = next(e for e in events if e["type"] == "context")
    ids = {s["id"] for s in context["sources"]}
    assert {"m1", "r1"} <= ids
    for s in context["sources"]:
        assert {"source", "id", "title", "score", "snippet"} <= set(s)
        assert "normalized_score" in s
    done = events[-1]
    assert done["answer"] == "respuesta sintetizada"
    assert done["total_ms"] >= 0
    assert done["synthesis_source"] == "memo.chat"


def test_retrieval_error_yields_error_event_and_stops(tmp_path) -> None:
    class _BoomMemory(_FakeMemory):
        def search(self, query, *, limit=None, mode="hybrid", **kw):
            raise RuntimeError("index corrupted")

    events = list(chat_stream(_BoomMemory(tmp_path), "pregunta"))
    assert events[-1]["type"] == "error"
    assert events[-1]["message"] == "retrieval failed"
    assert not any(e["type"] == "context" for e in events)
    assert not any(e["type"] == "done" for e in events)


def test_synthesis_error_yields_error_event(tmp_path) -> None:
    class _Boom(_FakeChatBackend):
        def chat_stream(self, model, messages, options=None):
            yield "parcial"
            raise RuntimeError("mlx died")

    mem = _FakeMemory(tmp_path)
    mem._ensure_chat = lambda: _Boom()  # type: ignore[method-assign]
    events = list(chat_stream(mem, "pregunta simple"))
    assert events[-1]["type"] == "error"
    assert events[-1]["answer_partial"] == "parcial"


class _NoSearchMemory(_FakeMemory):
    """search()/repo_search() must never be called on the WA-live exclusive path."""

    def search(self, query, *, limit=None, mode="hybrid", **kw):
        raise AssertionError("memory.search must not be called on the WA-live path")

    def repo_search(self, query, *, limit=10, **kw):
        raise AssertionError("memory.repo_search must not be called on the WA-live path")


def _assert_not_called(*_args, **_kwargs):
    raise AssertionError("must not be called")


def test_whatsapp_live_recency_query_uses_exclusive_wa_source(tmp_path, monkeypatch) -> None:
    last_messages_calls: list[dict] = []

    monkeypatch.setattr(
        whatsapp_live,
        "resolve_chats",
        lambda query, db, contacts_index: [("549@s.whatsapp.net", "Ana")],
    )

    def _fake_last_messages(db, jid, *, limit=10, today_only=False):
        last_messages_calls.append({"jid": jid, "limit": limit, "today_only": today_only})
        return [{"ts": "2026-07-31 09:00:00", "is_from_me": False, "content": "todo bien, y vos?"}]

    monkeypatch.setattr(whatsapp_live, "last_messages", _fake_last_messages)

    events = list(chat_stream(_NoSearchMemory(tmp_path), "qué me dijo Ana hoy?"))

    context = next(e for e in events if e["type"] == "context")
    assert len(context["sources"]) == 1
    src = context["sources"][0]
    assert src == {
        "id": "wa-live:ana",
        "source": "memory",
        "type": "whatsapp_live",
        "title": "WhatsApp · Ana — 2026-07-31",
        "snippet": "[2026-07-31 09:00:00] Ana: todo bien, y vos?",
        "score": 0.99,
        "normalized_score": 0.99,
    }
    assert last_messages_calls == [{"jid": "549@s.whatsapp.net", "limit": 200, "today_only": True}]

    done = events[-1]
    assert done["type"] == "done"
    assert done["sources"] == [src]


def test_non_recency_query_uses_normal_semantic_flow(tmp_path, monkeypatch) -> None:
    # A non-recency query must never even attempt the WA-live path.
    monkeypatch.setattr(whatsapp_live, "resolve_chats", _assert_not_called)

    events = list(chat_stream(_FakeMemory(tmp_path), "qué sabés de la nota uno?"))

    context = next(e for e in events if e["type"] == "context")
    ids = {s["id"] for s in context["sources"]}
    assert {"m1", "r1"} <= ids
    assert not any(str(s["id"]).startswith("wa-live:") for s in context["sources"])


def test_wa_live_exception_falls_back_to_normal_flow(tmp_path, monkeypatch) -> None:
    def _boom(*_args, **_kwargs):
        raise RuntimeError("bridge db locked")

    monkeypatch.setattr(whatsapp_live, "resolve_chats", _boom)

    events = list(chat_stream(_FakeMemory(tmp_path), "qué me dijo Ana hoy?"))

    context = next(e for e in events if e["type"] == "context")
    ids = {s["id"] for s in context["sources"]}
    assert {"m1", "r1"} <= ids


def test_whatsapp_live_default_contacts_dir_derived_from_vault_path(tmp_path, monkeypatch) -> None:
    # No MEMO_CHAT_CONTACTS_DIR set — must fall back to <vault>/Obsidian/Contacts
    # derived from memory.cfg.vault_path.
    contacts_dir = tmp_path / "Obsidian" / "Contacts"
    contacts_dir.mkdir(parents=True)
    (contacts_dir / "Ana.md").write_text("- **wa_jid**: 549@s.whatsapp.net\n", encoding="utf-8")

    captured_index: dict = {}

    def _fake_resolve_chats(query, db, contacts_index):
        captured_index.update(contacts_index)
        return [("549@s.whatsapp.net", "Ana")]

    monkeypatch.setattr(whatsapp_live, "resolve_chats", _fake_resolve_chats)
    monkeypatch.setattr(
        whatsapp_live,
        "last_messages",
        lambda db, jid, *, limit=10, today_only=False: [
            {"ts": "2026-07-31 09:00:00", "is_from_me": False, "content": "todo bien"}
        ],
    )

    mem = _NoSearchMemory(tmp_path)
    mem.cfg.vault_path = tmp_path

    events = list(chat_stream(mem, "qué me dijo Ana hoy?"))

    # The contacts index passed to resolve_chats came from the vault-derived
    # dir, not an empty fallback — proves the contact (Ana → jid) resolved.
    assert captured_index
    context = next(e for e in events if e["type"] == "context")
    assert context["sources"][0]["id"] == "wa-live:ana"


def test_whatsapp_live_no_vault_path_yields_empty_contacts_index(tmp_path, monkeypatch) -> None:
    # memory.cfg has no vault_path attribute at all (like a fake/stub memory) —
    # must degrade to an empty contacts index, not crash.
    captured_index: dict = {}

    def _fake_resolve_chats(query, db, contacts_index):
        captured_index.update(contacts_index)
        return []

    monkeypatch.setattr(whatsapp_live, "resolve_chats", _fake_resolve_chats)

    events = list(chat_stream(_FakeMemory(tmp_path), "qué me dijo Ana hoy?"))

    assert captured_index == {}
    context = next(e for e in events if e["type"] == "context")
    ids = {s["id"] for s in context["sources"]}
    assert {"m1", "r1"} <= ids


def test_whatsapp_live_disabled_flag_skips_wa_branch(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MEMO_CHAT_WHATSAPP_LIVE", "0")
    monkeypatch.setattr(whatsapp_live, "resolve_chats", _assert_not_called)

    events = list(chat_stream(_FakeMemory(tmp_path), "qué me dijo Ana hoy?"))

    context = next(e for e in events if e["type"] == "context")
    ids = {s["id"] for s in context["sources"]}
    assert {"m1", "r1"} <= ids
