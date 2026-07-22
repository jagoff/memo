from __future__ import annotations

from memo.memory import Memory


def test_history_logs_save_update_delete(mem_with_stub: Memory):
    rec = mem_with_stub.save(content="x", title="A", type_="note")
    mem_with_stub.update(rec.id, title="B")
    mem_with_stub.delete(rec.id)
    events = mem_with_stub.history.list_recent(limit=10)
    ops = [e["op"] for e in events]
    assert ops == ["delete", "update", "save"]
    upd = next(e for e in events if e["op"] == "update")
    assert upd["delta"]["title"] == ["A", "B"]
    assert "updated" in upd["delta"]  # always recorded for time-machine rewind


def test_history_filter_by_record_id(mem_with_stub: Memory):
    a = mem_with_stub.save(content="x", title="A")
    mem_with_stub.save(content="y", title="B")
    mem_with_stub.update(a.id, title="A2")
    events = mem_with_stub.history.list_recent(limit=10, record_id=a.id)
    assert all(e["record_id"] == a.id for e in events)
    assert {e["op"] for e in events} == {"save", "update"}


def test_extract_entities_writes_graph(mem_with_stub: Memory, monkeypatch):
    monkeypatch.setenv("MEMO_ENTITY_EXTRACT_ON_SAVE", "0")
    rec = mem_with_stub.save(
        content="Decidí migrar obsidian-rag a MLX con Qwen3-Embedding.",
        title="MLX migration",
    )

    def _stub_chat(self, model, messages, options=None):
        return {
            "message": {
                "content": '{"entities": [{"name": "obsidian-rag", "type": "project"}, {"name": "mlx", "type": "technology"}, {"name": "qwen3-embedding", "type": "technology"}]}'
            }
        }

    monkeypatch.setattr("memo.llm.MLXChat.chat", _stub_chat)
    counts = mem_with_stub.extract_entities(ids=[rec.id])
    assert counts["processed"] == 1
    assert counts["entities_extracted"] == 3
    assert counts["links_written"] == 3

    top = mem_with_stub.graph.top_entities(limit=10)
    names = {e["name"] for e in top}
    assert {"obsidian-rag", "mlx", "qwen3-embedding"}.issubset(names)

    ids = mem_with_stub.graph.entity_memories("mlx")
    assert ids == [rec.id]


def test_extract_entities_skip_already_indexed(mem_with_stub: Memory, monkeypatch):
    rec = mem_with_stub.save(content="x", title="X")
    calls = [0]

    def _stub_chat(self, model, messages, options=None):
        calls[0] += 1
        return {"message": {"content": '{"entities": [{"name": "x", "type": "concept"}]}'}}

    monkeypatch.setattr("memo.llm.MLXChat.chat", _stub_chat)
    mem_with_stub.extract_entities(ids=[rec.id])
    assert calls[0] == 1


def test_extract_entities_upgrades_regex_only_membership(mem_with_stub: Memory, monkeypatch):
    rec = mem_with_stub.save(
        content="FastAPI powers the service architecture.",
        title="FastAPI service",
    )
    assert mem_with_stub.graph.memory_extraction_provenance(rec.id) == {"regex"}

    calls = [0]

    def _stub_chat(self, model, messages, options=None):
        calls[0] += 1
        return {
            "message": {
                "content": (
                    '{"entities": '
                    '[{"name": "FastAPI", "type": "technology"}]}'
                )
            }
        }

    monkeypatch.setattr("memo.llm.MLXChat.chat", _stub_chat)

    first = mem_with_stub.extract_entities(ids=[rec.id], skip_already_indexed=True)
    second = mem_with_stub.extract_entities(ids=[rec.id], skip_already_indexed=True)

    assert first["processed"] == 1
    assert second["processed"] == 0
    assert calls[0] == 1
    assert mem_with_stub.graph.memory_extraction_provenance(rec.id) == {"llm"}
    assert mem_with_stub.graph.memory_entities(rec.id)[0]["type"] == "technology"
    counts = mem_with_stub.extract_entities(ids=[rec.id])
    assert counts["processed"] == 0
    assert calls[0] == 1


def test_delete_drops_graph_edges(mem_with_stub: Memory, monkeypatch):
    rec = mem_with_stub.save(content="x", title="X")

    def _stub_chat(self, model, messages, options=None):
        return {"message": {"content": '{"entities": [{"name": "foo", "type": "concept"}]}'}}

    monkeypatch.setattr("memo.llm.MLXChat.chat", _stub_chat)
    mem_with_stub.extract_entities(ids=[rec.id])
    assert mem_with_stub.graph.entity_memories("foo") == [rec.id]

    mem_with_stub.delete(rec.id)
    assert mem_with_stub.graph.entity_memories("foo") == []
