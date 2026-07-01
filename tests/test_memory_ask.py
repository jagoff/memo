from __future__ import annotations

from memo.memory import Memory, MemoryRecord


def test_consolidate_clusters_near_duplicates(mem_with_stub: Memory, monkeypatch):
    monkeypatch.setattr(
        "memo.embedder.MLXEmbedder.embed",
        lambda self, inputs: [[1.0, 0.0, 0.0, 0.0] for _ in inputs],
    )
    rec_a = mem_with_stub.save(content="alpha body uno", title="A1")
    rec_b = mem_with_stub.save(content="alpha body dos", title="A2")

    captured: list[list[dict]] = []

    def _stub_chat(self, model, messages, options=None):
        captured.append(messages)
        return {"message": {"content": '{"summary": "Both notes describe the same alpha concept.", "relationship": "duplicate", "rationale": "Body strings differ by one character only."}'}}

    monkeypatch.setattr("memo.llm.MLXChat.chat", _stub_chat)
    clusters = mem_with_stub.consolidate(threshold=0.99)
    assert len(clusters) >= 1
    found = False
    for c in clusters:
        ids = {m["id"] for m in c["members"]}
        if {rec_a.id, rec_b.id}.issubset(ids):
            found = True
            assert c["relationship"] == "duplicate"
            assert "alpha" in c["summary"].lower()
    assert found
    assert len(captured) == 1


def test_consolidate_drops_singletons(mem_with_stub: Memory):
    mem_with_stub.save(content="solo memoria", title="Lonely")
    clusters = mem_with_stub.consolidate(threshold=0.5)
    assert clusters == []


def test_ask_synthesises_with_citations(mem_with_stub: Memory, monkeypatch):
    rec_a = mem_with_stub.save(content="alpha body", title="Alpha")
    rec_b = mem_with_stub.save(content="beta body", title="Beta")

    captured: dict = {}

    def _stub_chat(self, model, messages, options=None):
        captured["model"] = model
        captured["messages"] = messages
        return {"message": {"content": f"Respuesta corta sobre alpha [{rec_a.id[:8]}]."}}

    monkeypatch.setattr("memo.llm.MLXChat.chat", _stub_chat)
    out = mem_with_stub.ask("¿qué hay sobre alpha?", k=2)

    assert "alpha" in out["answer"].lower() or rec_a.id[:8] in out["answer"]
    assert len(out["sources"]) >= 1
    user_msg = captured["messages"][-1]["content"]
    assert f"[{rec_a.id[:8]}]" in user_msg or f"[{rec_b.id[:8]}]" in user_msg
    assert "7B" in captured["model"] or "Qwen2.5" in captured["model"]


def test_ask_dedups_repo_hits_against_vault_and_intra_repo(mem_with_stub: Memory, monkeypatch):
    from memo.repo_index import RepoSearchHit

    rec = mem_with_stub.save(
        content="Sos la sal de este mar",
        title="01-Projects/Album-Muros-Fractales/Letra - Sal.md",
    )
    vault_path = rec.path

    def _fake_repo_hit(line_start: int) -> RepoSearchHit:
        return RepoSearchHit(
            id=f"chunk-{line_start}", repo_id="r1", repo_name="obsidian-personal",
            url="https://example.com/r.git", ref="HEAD", commit_sha="abcdef12",
            file_id="f1", path=vault_path, language="markdown", line_start=line_start,
            line_end=line_start + 10, text="Sos la sal de este mar", score=0.5, match_type="hybrid",
        )

    monkeypatch.setattr(
        Memory, "repo_search",
        lambda self, q, **kw: [_fake_repo_hit(1), _fake_repo_hit(20), _fake_repo_hit(40)],
    )
    monkeypatch.setattr(
        type(mem_with_stub.store), "list_repo_sources",
        lambda self, **kw: [{"name": "obsidian-personal"}],
    )

    _, sources, _, _ = mem_with_stub._build_ask_context(
        "sal de este mar", k=5, type_=None, snippet_chars=200, include_repos=True,
    )

    assert [s["source"] for s in sources] == ["memory"]
    assert sources[0]["title"] == rec.title


def test_ask_dedups_repo_chunks_when_no_vault_overlap(mem_with_stub: Memory, monkeypatch):
    from memo.repo_index import RepoSearchHit

    def _hit(line_start: int) -> RepoSearchHit:
        return RepoSearchHit(
            id=f"c-{line_start}", repo_id="r2", repo_name="code-repo",
            url="x", ref="HEAD", commit_sha="deadbeef", file_id="f", path="src/foo.py",
            language="python", line_start=line_start, line_end=line_start + 5,
            text="def foo(): pass", score=0.8, match_type="hybrid",
        )

    monkeypatch.setattr(Memory, "repo_search", lambda self, q, **kw: [_hit(1), _hit(50), _hit(100)])
    monkeypatch.setattr(type(mem_with_stub.store), "list_repo_sources", lambda self, **kw: [{"name": "code-repo"}])

    _, sources, _, _ = mem_with_stub._build_ask_context(
        "foo", k=5, type_=None, snippet_chars=200, include_repos=True,
    )

    repo_sources = [s for s in sources if s["source"] == "repo"]
    assert len(repo_sources) == 1
    assert repo_sources[0]["line_start"] == 1


def test_ask_returns_no_answer_when_no_hits(mem_with_stub: Memory):
    out = mem_with_stub.ask("pregunta sin contexto")
    assert "couldn't find" in out["answer"].lower()
    assert out["sources"] == []


def test_recency_helpers():
    from memo.memory import (
        _is_conversation_query,
        _is_recency_query,
        _is_whatsapp_hit,
        _recency_key,
    )

    assert _is_recency_query("qué fue lo último que dijo Grecia por whatsapp")
    assert _is_recency_query("what did Maria last say")
    assert _is_recency_query("su mensaje más reciente")
    assert not _is_recency_query("quién es Grecia")
    assert not _is_recency_query("resumen del proyecto memo")
    assert _is_conversation_query("mostrame el chat con Grecia")
    assert _is_conversation_query("qué me escribió Grecia")
    assert _is_conversation_query("show me my messages with Maria")
    assert not _is_conversation_query("quién es Grecia")
    assert not _is_conversation_query("cuándo es el cumple de Grecia")
    assert not _is_conversation_query("resumen del proyecto memo")

    wa = MemoryRecord(
        id="b" * 32, path="x/WhatsApp · Grecia.md", title="WhatsApp · Grecia 🩷",
        type="reference", tags=["whatsapp", "chat"],
        created="2026-05-18T00:00:00", updated="2026-05-18T00:00:00",
        body="## 2026-05-17\n- **Grecia 🩷** (16:26): Jajaja está linda",
    )
    contact = MemoryRecord(
        id="a" * 32, path="Contacts/Grecia.md", title="Grecia",
        type="reference", tags=["Obsidian", "Contacts"],
        created="2026-05-30T00:00:00", updated="2026-05-30T00:00:00",
        body="Grecia Ferrari, 15 años, Santa Fe.",
    )
    meta = MemoryRecord(
        id="c" * 32, path="AI/memory/whatsapp-note.md", title="Maria y Grecia completados",
        type="fact", tags=["whatsapp", "contacts", "vault"],
        created="2026-05-30T00:00:00", updated="2026-05-30T00:00:00",
        body="Se completaron los JID de Maria y Grecia.",
    )
    assert _is_whatsapp_hit(wa)
    assert not _is_whatsapp_hit(contact)
    assert not _is_whatsapp_hit(meta)
    assert _recency_key(wa) == "2026-05-17 16:26"
    assert _recency_key(contact) == "2026-05-30"


def test_ask_recency_floats_whatsapp_transcript_over_contact_card(mem_with_stub: Memory, monkeypatch):
    contact = MemoryRecord(
        id="a" * 32, path="Contacts/Grecia.md", title="Grecia",
        type="reference", tags=["Obsidian", "Contacts"],
        created="2026-05-30T00:00:00", updated="2026-05-30T00:00:00",
        body="Grecia Ferrari, 15 años, Santa Fe.", score=0.92,
    )
    transcript = MemoryRecord(
        id="b" * 32, path="AI/Whatsapp/Grecia.md", title="WhatsApp · Grecia 🩷",
        type="reference", tags=["whatsapp", "chat"],
        created="2026-05-18T00:00:00", updated="2026-05-18T00:00:00",
        body="## 2026-05-17\n- **Grecia 🩷** (16:26): Jajaja está linda", score=0.90,
    )
    group = MemoryRecord(
        id="d" * 32, path="AI/Whatsapp/Grecia's group.md", title="WhatsApp · Grecia's group",
        type="reference", tags=["whatsapp", "chat"],
        created="2026-05-19T00:00:00", updated="2026-05-19T00:00:00",
        body="## 2026-05-19\n- **alguien** (10:00): Jajajajaj", score=0.91,
    )
    monkeypatch.setattr(Memory, "search", lambda self, q, **kw: [contact, group, transcript])

    _, sources, _, _ = mem_with_stub._build_ask_context(
        "qué fue lo último que dijo Grecia por whatsapp", k=5, type_=None, snippet_chars=200, include_repos=False,
    )
    assert sources[0]["title"] == "WhatsApp · Grecia's group"
    assert sources[1]["title"] == "WhatsApp · Grecia 🩷"
    assert sources[2]["title"] == "Grecia"

    _, sources_c, _, _ = mem_with_stub._build_ask_context(
        "mostrame el chat con Grecia", k=5, type_=None, snippet_chars=200, include_repos=False,
    )
    assert sources_c[0]["title"] == "WhatsApp · Grecia 🩷"

    _, sources2, _, _ = mem_with_stub._build_ask_context(
        "quién es Grecia", k=5, type_=None, snippet_chars=200, include_repos=False,
    )
    assert sources2[0]["title"] == "Grecia"


def test_ask_conversation_intent_without_whatsapp_preserves_order(mem_with_stub: Memory, monkeypatch):
    top = MemoryRecord(
        id="a" * 32, path="AI/memory/decision.md", title="Decisión arquitectura",
        type="fact", tags=["memo"],
        created="2026-05-10T00:00:00", updated="2026-05-10T00:00:00",
        body="Migramos a MLX.", score=0.80,
    )
    older_but_newer_date = MemoryRecord(
        id="b" * 32, path="AI/memory/note.md", title="Nota suelta",
        type="note", tags=["memo"],
        created="2026-05-25T00:00:00", updated="2026-05-25T00:00:00",
        body="## 2026-05-25\nalgo", score=0.40,
    )
    monkeypatch.setattr(Memory, "search", lambda self, q, **kw: [top, older_but_newer_date])

    _, sources, _, _ = mem_with_stub._build_ask_context(
        "resumen de la conversación sobre arquitectura", k=5, type_=None, snippet_chars=200, include_repos=False,
    )
    assert sources[0]["title"] == "Decisión arquitectura"


def test_chat_ask_v2_uses_history_and_context(mem_with_stub: Memory, monkeypatch):
    rec = mem_with_stub.save(content="alpha architectural decision", title="Alpha")
    captured: dict = {}

    def _stub_chat(self, model, messages, options=None):
        captured["user"] = messages[-1]["content"]
        return {"message": {"content": f"Alpha answer [{rec.id[:8]}]."}}

    monkeypatch.setattr("memo.llm.MLXChat.chat", _stub_chat)

    out = mem_with_stub.chat_ask(
        "what did we decide?",
        k=2,
        history=[{"role": "user", "text": "previous alpha question"}],
        context={"packet_status": "ready", "route_decision": {"target": "memo"}},
    )

    assert out["schema"] == "memo.chat_ask.v2"
    assert out["synthesis_status"] == "ok"
    assert out["answer"].startswith("Alpha answer")
    assert out["history_turns_used"] == 1
    assert out["context_keys"] == ["packet_status", "route_decision"]
    assert out["citations"][0]["source"] == "memo"
    assert out["retrieval_trace"][0]["stage"] == "memo.chat_ask"
    assert "previous alpha question" in captured["user"]
    assert "packet_status" in captured["user"]
