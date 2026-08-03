from types import SimpleNamespace

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


class _FakeGraph:
    def memory_entities(self, memory_id):
        if memory_id in ("m1", "m2"):
            return [{"name": "proyecto-omega", "type": "topic", "mention_count": 1}]
        return []

    def total_indexed_memories(self):
        return 10

    def entity_doc_freqs(self, names):
        return {"proyecto-omega": 1.0} if "proyecto-omega" in names else {}


class _TwoNoteMemory(_FakeMemory):
    def __init__(self, tmp_path):
        super().__init__(tmp_path)
        self.graph = _FakeGraph()

    def search(self, query, *, limit=None, mode="hybrid", **kw):
        return [
            _FakeRecord(
                id="m1", title="Nota uno", type="note", score=0.9,
                body="cuerpo uno sobre proyecto omega", path="notes/uno.md",
            ),
            _FakeRecord(
                id="m2", title="Nota dos", type="note", score=0.85,
                body="cuerpo dos sobre proyecto omega", path="notes/dos.md",
            ),
        ]


def test_graph_compact_collapses_related_sources_when_enabled(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MEMO_CHAT_GRAPH_COMPACT", "1")

    events = list(chat_stream(_TwoNoteMemory(tmp_path), "qué sabés del proyecto omega?"))
    context = next(e for e in events if e["type"] == "context")
    ids = [s["id"] for s in context["sources"]]

    assert ids.count("m1") + ids.count("m2") == 1  # one survivor, not both
    assert "r1" in ids
    survivor = next(s for s in context["sources"] if s["id"] in ("m1", "m2"))
    assert len(survivor.get("related_ids") or []) == 1


def test_graph_compact_noop_when_disabled(tmp_path) -> None:
    events = list(chat_stream(_TwoNoteMemory(tmp_path), "qué sabés del proyecto omega?"))
    context = next(e for e in events if e["type"] == "context")
    ids = {s["id"] for s in context["sources"]}
    assert {"m1", "m2", "r1"} <= ids
