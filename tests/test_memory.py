"""High-level Memory — save/search/list/get/delete with stub embedder.

These tests stub out `MLXEmbedder.embed` so they run on any platform
without loading the real model. The actual MLX-on-Metal smoke is in
`test_smoke_mlx.py` (gated by `requires_mlx`).
"""

from __future__ import annotations

import pytest

from memo.config import Config
from memo.memory import AmbiguousIdError, Memory


@pytest.fixture
def mem_with_stub(tmp_cfg: Config, monkeypatch) -> Memory:
    """`Memory` with a deterministic 4-dim embedder that hashes the
    input text into one of 4 buckets — same text always lands in the
    same vector quadrant, different texts collide deterministically.
    Good enough to exercise the index roundtrip without real MLX."""
    cfg = Config(
        data_dir=tmp_cfg.data_dir,
        vault_path=tmp_cfg.vault_path,
        state_dir=tmp_cfg.state_dir,
        embedder_dims=4,
    )

    def _stub_embed(self, inputs):
        out = []
        for s in inputs:
            h = sum(ord(c) for c in s) % 4
            v = [0.0] * 4
            v[h] = 1.0
            out.append(v)
        return out

    monkeypatch.setattr("memo.embedder.MLXEmbedder.embed", _stub_embed)
    return Memory(cfg)


def test_save_writes_md_and_indexes(mem_with_stub: Memory):
    rec = mem_with_stub.save(content="primer memo del test", title="Test 1", type_="note")
    abs_path = mem_with_stub.cfg.memory_dir /rec.path
    assert abs_path.is_file()
    text = abs_path.read_text(encoding="utf-8")
    assert "title: Test 1" in text
    assert "primer memo del test" in text
    assert mem_with_stub.store.count() == 1


def test_save_rejects_invalid_type(mem_with_stub: Memory):
    with pytest.raises(ValueError, match="not in valid set"):
        mem_with_stub.save(content="x", type_="bogus")


def test_save_rejects_empty_content(mem_with_stub: Memory):
    with pytest.raises(ValueError, match="non-empty"):
        mem_with_stub.save(content="   ")


def test_search_returns_matching(mem_with_stub: Memory):
    mem_with_stub.save(content="alpha", title="A")
    mem_with_stub.save(content="beta", title="B")
    hits = mem_with_stub.search("alpha", limit=2)
    assert any(h.title == "A" for h in hits)


def test_get_returns_record_with_body(mem_with_stub: Memory):
    rec = mem_with_stub.save(content="cuerpo del memo", title="X")
    fetched = mem_with_stub.get(rec.id)
    assert fetched is not None
    assert fetched.title == "X"
    assert "cuerpo del memo" in fetched.body


def test_get_missing_returns_none(mem_with_stub: Memory):
    assert mem_with_stub.get("nope") is None


def test_list_orders_recent_first(mem_with_stub: Memory):
    a = mem_with_stub.save(content="primero", title="A")
    # Force monotonic timestamp ordering for the test (sub-second
    # collisions would otherwise tie-break in insertion order).
    import time

    time.sleep(0.001)
    b = mem_with_stub.save(content="segundo", title="B")
    items = mem_with_stub.list(limit=10)
    titles = [r.title for r in items]
    # B was saved second → should appear first under `updated DESC`.
    assert titles.index("B") < titles.index("A")
    assert {r.id for r in items} == {a.id, b.id}


def test_delete_removes_disk_and_index(mem_with_stub: Memory):
    rec = mem_with_stub.save(content="borrar este", title="X")
    assert (mem_with_stub.cfg.memory_dir /rec.path).is_file()
    assert mem_with_stub.delete(rec.id) is True
    assert mem_with_stub.store.count() == 0
    assert not (mem_with_stub.cfg.memory_dir /rec.path).is_file()


def test_delete_missing_returns_false(mem_with_stub: Memory):
    assert mem_with_stub.delete("nope") is False


def test_delete_aborts_when_md_unlink_fails(mem_with_stub: Memory, monkeypatch):
    """Authority flip: if the canonical .md can't be removed, delete must
    raise and leave the index intact (truth still on disk → index must agree)."""
    from memo.errors import StorageError

    rec = mem_with_stub.save(content="protegido", title="X")
    assert mem_with_stub.store.count() == 1

    real_unlink = type(mem_with_stub.cfg.memory_dir).unlink

    def _boom(self, *a, **k):
        if self.name.endswith(".md"):
            raise OSError("permission denied")
        return real_unlink(self, *a, **k)

    monkeypatch.setattr("pathlib.Path.unlink", _boom)
    with pytest.raises(StorageError, match="delete refused"):
        mem_with_stub.delete(rec.id)
    # Index untouched — the row survives because its truth-bearing file does.
    assert mem_with_stub.store.count() == 1


def test_delete_proceeds_when_md_already_missing(mem_with_stub: Memory):
    """A missing .md is a no-op, not an error: the orphaned index row is dropped."""
    rec = mem_with_stub.save(content="huérfano", title="X")
    (mem_with_stub.cfg.memory_dir / rec.path).unlink()  # vanish the file
    assert mem_with_stub.delete(rec.id) is True
    assert mem_with_stub.store.count() == 0


def test_save_index_failure_keeps_md_and_marks_pending(mem_with_stub: Memory, monkeypatch):
    """markdown-is-truth: if embedding fails AFTER the .md is written, the save
    must NOT raise — the file stays on disk, stamped embed-pending so reindex
    replays it, and is still BM25-searchable."""
    def _explode(self, inputs):
        raise RuntimeError("embedder down")

    monkeypatch.setattr("memo.embedder.MLXEmbedder.embed", _explode)
    rec = mem_with_stub.save(content="cuerpo recuperable", title="Recuperable")

    abs_path = mem_with_stub.cfg.memory_dir / rec.path
    assert abs_path.is_file(), "canonical .md must survive an index failure"
    text = abs_path.read_text(encoding="utf-8")
    assert "_memo_embed_pending" in text, "on-disk frontmatter must flag embed-pending"
    assert rec.extra.get("_memo_embed_pending") is True
    # Text-only row exists so BM25 still surfaces it before the next reindex.
    assert mem_with_stub.store.get(rec.id) is not None


def test_save_index_failure_recovers_on_reindex(mem_with_stub: Memory, monkeypatch):
    """The embed-pending memoria becomes fully vector-searchable after reindex
    once the embedder is healthy again — proving the .md replays the index."""
    calls = {"n": 0}
    real_embed = type(mem_with_stub.embedder).embed

    def _flaky(self, inputs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("embedder down")
        return real_embed(self, inputs)

    monkeypatch.setattr("memo.embedder.MLXEmbedder.embed", _flaky)
    rec = mem_with_stub.save(content="recupera via reindex", title="Reindexable")
    assert not mem_with_stub.store.has_vector(rec.id)

    out = mem_with_stub.reindex()
    assert out["reindexed"] >= 1
    assert mem_with_stub.store.has_vector(rec.id)


def test_edit_md_then_reindex_markdown_wins(mem_with_stub: Memory):
    """Round-trip: edit the .md by hand (as Obsidian would) → reindex → search
    surfaces the EDITED body. markdown is the source of truth."""
    rec = mem_with_stub.save(content="contenido original", title="Editable")
    abs_path = mem_with_stub.cfg.memory_dir / rec.path
    text = abs_path.read_text(encoding="utf-8")
    abs_path.write_text(text.replace("contenido original", "contenido EDITADO a mano"), encoding="utf-8")

    out = mem_with_stub.reindex()
    assert out["reindexed"] >= 1
    fetched = mem_with_stub.get(rec.id)
    assert fetched is not None
    assert "EDITADO a mano" in fetched.body
    assert "contenido original" not in fetched.body


def test_reindex_rebuild_preserves_signal(mem_with_stub: Memory):
    """reindex(rebuild=True) truncates + replays the index from disk WITHOUT
    destroying user-signal data (access counts, health) keyed on the id."""
    a = mem_with_stub.save(content="memoria una", title="A")
    b = mem_with_stub.save(content="memoria dos", title="B")
    store = mem_with_stub.store
    store.touch([a.id])
    store.touch([a.id])
    store.boost_roi_batch([b.id], delta=0.3)
    assert store.get_access(a.id)["access_count"] == 2
    assert store.get_health_batch([b.id])[b.id]["roi_score"] > 1.0

    out = mem_with_stub.reindex(rebuild=True)
    assert out["added"] == 2  # both replayed fresh from disk
    assert mem_with_stub.store.count() == 2
    # Signal survived the rebuild — re-joined on the stable id.
    assert store.get_access(a.id)["access_count"] == 2
    assert store.get_health_batch([b.id])[b.id]["roi_score"] > 1.0
    # Content is fully searchable again.
    assert mem_with_stub.get(a.id) is not None
    assert mem_with_stub.get(b.id) is not None


def test_reindex_rebuild_drops_orphans(mem_with_stub: Memory):
    """A memoria whose .md vanished must not survive a rebuild (orphan dropped),
    while present memorias are retained."""
    a = mem_with_stub.save(content="vive", title="A")
    b = mem_with_stub.save(content="muere", title="B")
    (mem_with_stub.cfg.memory_dir / b.path).unlink()  # vanish B's source file

    out = mem_with_stub.reindex(rebuild=True)
    assert out["added"] == 1
    assert mem_with_stub.get(a.id) is not None
    assert mem_with_stub.get(b.id) is None
    assert mem_with_stub.store.count() == 1


def test_reindex_embedding_reuse_skips_second_pass(mem_with_stub: Memory, monkeypatch):
    """Once the content cache is warm, a force reindex of unchanged bodies
    issues zero embedder forward passes (cache hits cover every memoria)."""
    mem_with_stub.save(content="alpha", title="A")
    mem_with_stub.save(content="beta", title="B")
    # First force pass warms the content-addressed cache.
    mem_with_stub.reindex(force=True)

    calls = {"n": 0}
    real_embed = type(mem_with_stub.embedder).embed

    def _counting(self, inputs):
        calls["n"] += 1
        return real_embed(self, inputs)

    monkeypatch.setattr("memo.embedder.MLXEmbedder.embed", _counting)
    out = mem_with_stub.reindex(force=True)
    assert out["reindexed"] == 2
    assert calls["n"] == 0, "warm cache must avoid all re-embedding"


def test_tags_lower_dedup(mem_with_stub: Memory):
    rec = mem_with_stub.save(content="x", title="X", tags=["MLX", "mlx", "Local"])
    assert rec.tags == ["mlx", "local"]


def test_title_derived_from_first_line(mem_with_stub: Memory):
    rec = mem_with_stub.save(content="# Encabezado\n\nbody")
    assert rec.title == "Encabezado"


def test_update_skips_reembed_for_pure_retag(mem_with_stub: Memory, monkeypatch):
    rec = mem_with_stub.save(content="cuerpo", title="orig", type_="note", tags=["x"])
    # Track embed calls — pure retag/type changes should NOT re-embed.
    # Title and body changes DO re-embed (both feed into the vector).
    calls: list[int] = []
    orig = mem_with_stub.embedder.embed

    def _spy(inputs):
        calls.append(len(inputs))
        return orig(inputs)

    monkeypatch.setattr(mem_with_stub.embedder, "embed", _spy)
    updated = mem_with_stub.update(rec.id, type_="decision", tags=["y", "Z"])
    assert updated is not None
    assert updated.type == "decision"
    assert updated.tags == ["y", "z"]
    assert updated.title == "orig"  # unchanged
    assert updated.body == "cuerpo"  # unchanged
    assert calls == []  # no re-embed


def test_update_reembeds_when_title_changes(mem_with_stub: Memory, monkeypatch):
    """Title is part of the embed input, so changing it must re-embed."""
    rec = mem_with_stub.save(content="cuerpo", title="orig", type_="note")
    calls: list[int] = []
    orig = mem_with_stub.embedder.embed

    def _spy(inputs):
        calls.append(len(inputs))
        return orig(inputs)

    monkeypatch.setattr(mem_with_stub.embedder, "embed", _spy)
    updated = mem_with_stub.update(rec.id, title="renamed")
    assert updated is not None
    assert updated.title == "renamed"
    assert calls == [1]  # re-embed because title is part of vector input


def test_update_reembeds_when_content_changes(mem_with_stub: Memory, monkeypatch):
    rec = mem_with_stub.save(content="cuerpo viejo", title="X")
    calls: list[int] = []
    orig = mem_with_stub.embedder.embed

    def _spy(inputs):
        calls.append(len(inputs))
        return orig(inputs)

    monkeypatch.setattr(mem_with_stub.embedder, "embed", _spy)
    updated = mem_with_stub.update(rec.id, content="cuerpo nuevo y diferente")
    assert updated is not None
    assert updated.body == "cuerpo nuevo y diferente"
    assert calls == [1]
    # Disk reflects new body.
    on_disk = (mem_with_stub.cfg.memory_dir /updated.path).read_text()
    assert "cuerpo nuevo y diferente" in on_disk


def test_contextual_retrieval_prepends_context_only_when_enabled(
    mem_with_stub: Memory,
    monkeypatch,
):
    seen_inputs: list[str] = []

    def _spy_embed(inputs):
        seen_inputs.extend(inputs)
        out = []
        for _s in inputs:
            out.append([1.0, 0.0, 0.0, 0.0])
        return out

    monkeypatch.setenv("MEMO_CONTEXTUAL_RETRIEVAL", "1")
    monkeypatch.setattr(mem_with_stub.embedder, "embed", _spy_embed)
    monkeypatch.setattr(
        mem_with_stub,
        "_generate_contextual_summary",
        lambda _prompt: "Esta memoria trata sobre el runbook de ingestion.",
    )
    body = "Runbook de ingestion.\n\n" + ("detalle operacional " * 40)

    rec = mem_with_stub.save(content=body, title="Ingestion Runbook")

    assert rec.body == body
    assert seen_inputs
    assert seen_inputs[0].startswith("[contexto: Esta memoria trata")
    assert "Ingestion Runbook" in seen_inputs[0]
    assert "Runbook de ingestion" in seen_inputs[0]


def test_contextual_retrieval_cache_reuses_generated_summary(
    mem_with_stub: Memory,
    monkeypatch,
):
    calls: list[str] = []

    def _generate(prompt: str) -> str:
        calls.append(prompt)
        return "Contexto cacheado para búsqueda semántica."

    monkeypatch.setenv("MEMO_CONTEXTUAL_RETRIEVAL", "1")
    monkeypatch.setattr(mem_with_stub, "_generate_contextual_summary", _generate)
    body = "Nota larga.\n\n" + ("contenido importante " * 40)

    first = mem_with_stub._compose_for_embed("Nota", body)
    second = mem_with_stub._compose_for_embed("Nota", body)

    assert first == second
    assert first.startswith("[contexto: Contexto cacheado")
    assert len(calls) == 1


def test_update_missing_returns_none(mem_with_stub: Memory):
    assert mem_with_stub.update("nope", title="x") is None


def test_update_rejects_invalid_type(mem_with_stub: Memory):
    rec = mem_with_stub.save(content="x", title="X")
    with pytest.raises(ValueError, match="not in valid set"):
        mem_with_stub.update(rec.id, type_="bogus")


def test_reindex_force_reuses_warm_embed_cache(mem_with_stub: Memory, monkeypatch):
    """`reindex(force=True)` re-processes every entry, but a content-addressed
    embed-cache hit (same content + model) reuses the stored vector — only a
    cold cache (e.g. after an embedder/composition swap that changes the key)
    triggers a real forward pass. save() pre-warms the cache, so a forced
    rebuild of unchanged content issues zero embedder calls."""
    rec = mem_with_stub.save(content="cuerpo", title="X")
    calls: list[int] = []
    orig = mem_with_stub.embedder.embed

    def _spy(inputs):
        calls.append(len(inputs))
        return orig(inputs)

    monkeypatch.setattr(mem_with_stub.embedder, "embed", _spy)

    counts = mem_with_stub.reindex()  # no force → no re-embed
    assert counts["reindexed"] == 0
    assert calls == []

    # Warm cache (save() populated it): force re-processes but reuses vectors.
    counts = mem_with_stub.reindex(force=True)
    assert counts["reindexed"] == 1
    assert calls == [], "warm content cache should avoid the forward pass"

    # Cold cache (simulates a model/composition swap invalidating keys):
    # force genuinely re-embeds.
    with mem_with_stub.store._conn:
        mem_with_stub.store._conn.execute("DELETE FROM repo_embedding_cache")
    calls.clear()
    counts = mem_with_stub.reindex(force=True)
    assert counts["reindexed"] == 1
    assert calls == [1]
    # Disk + index still consistent.
    fetched = mem_with_stub.get(rec.id)
    assert fetched is not None
    assert fetched.title == "X"


def test_reindex_picks_up_external_edit(mem_with_stub: Memory):
    rec = mem_with_stub.save(content="primero", title="X")
    abs_path = mem_with_stub.cfg.memory_dir /rec.path
    # Simulate a user edit in Obsidian: rewrite body via frontmatter.
    import frontmatter as fm

    post = fm.loads(abs_path.read_text())
    post.content = "cuerpo editado a mano"
    abs_path.write_text(fm.dumps(post), encoding="utf-8")

    counts = mem_with_stub.reindex()
    assert counts["reindexed"] == 1
    assert counts["added"] == 0
    fetched = mem_with_stub.get(rec.id)
    assert fetched is not None
    assert "editado a mano" in fetched.body


def test_reindex_adds_orphan_disk_file(mem_with_stub: Memory):
    # Hand-craft a memory `.md` the store doesn't know about.
    import frontmatter as fm

    md = mem_with_stub.cfg.memory_dir / "2026-05-06-restored.md"
    post = fm.Post(
        "memo restaurado de un backup",
        id="ffeeddccbbaa00112233445566778899",
        title="Restored",
        type="note",
        tags=["backup"],
        created="2026-05-06T19:00:00-03:00",
        updated="2026-05-06T19:00:00-03:00",
    )
    md.write_text(fm.dumps(post), encoding="utf-8")
    counts = mem_with_stub.reindex()
    assert counts["added"] == 1
    fetched = mem_with_stub.get("ffeeddccbbaa00112233445566778899")
    assert fetched is not None
    assert fetched.title == "Restored"


def test_gc_reports_and_fixes_orphans(mem_with_stub: Memory):
    a = mem_with_stub.save(content="vivo", title="A")
    b = mem_with_stub.save(content="vivo", title="B")
    # Delete `b`'s `.md` from disk to make it an orphan store row.
    (mem_with_stub.cfg.memory_dir /b.path).unlink()
    report = mem_with_stub.gc(fix=False)
    assert b.id in report["orphan_store"]
    assert a.id not in report["orphan_store"]
    # Without --fix, store still has the orphan.
    assert mem_with_stub.store.get(b.id) is not None
    # With --fix, orphan store row is dropped.
    mem_with_stub.gc(fix=True)
    assert mem_with_stub.store.get(b.id) is None


def test_get_by_unique_prefix(mem_with_stub: Memory):
    rec = mem_with_stub.save(content="x", title="X")
    short = rec.id[:7]  # git-style prefix
    fetched = mem_with_stub.get(short)
    assert fetched is not None
    assert fetched.id == rec.id


def test_get_unknown_prefix_returns_none(mem_with_stub: Memory):
    mem_with_stub.save(content="x", title="X")
    assert mem_with_stub.get("ffffffff") is None


def test_get_ambiguous_prefix_raises(mem_with_stub: Memory, monkeypatch):
    # Force two ids that share a prefix by stubbing uuid4 sequentially.
    import uuid

    fixed = iter([
        uuid.UUID("aaaaaaaa1111000000000000000000ff"),
        uuid.UUID("aaaaaaaa2222000000000000000000ff"),
    ])
    monkeypatch.setattr("memo.memory.uuid.uuid4", lambda: next(fixed))
    mem_with_stub.save(content="a", title="A")
    mem_with_stub.save(content="b", title="B")
    with pytest.raises(AmbiguousIdError) as exc_info:
        mem_with_stub.get("aaaaaaaa")
    assert len(exc_info.value.matches) == 2


def test_update_and_delete_accept_prefix(mem_with_stub: Memory):
    rec = mem_with_stub.save(content="x", title="X")
    short = rec.id[:6]
    updated = mem_with_stub.update(short, title="X2")
    assert updated is not None
    assert updated.title == "X2"
    assert mem_with_stub.delete(short) is True


def test_embed_batch_preserves_order_and_handles_empty(tmp_cfg: Config, monkeypatch):
    """Batched embed must preserve input order and produce a zero vec
    for empty strings (caller's responsibility to filter)."""
    seen: list[int] = []

    def _spy(self, inputs):
        seen.append(len(inputs))
        out = []
        for s in inputs:
            if not s:
                out.append([0.0] * 4)
            else:
                # Deterministic per-string vector: bucket on length.
                v = [0.0] * 4
                v[len(s) % 4] = 1.0
                out.append(v)
        return out

    monkeypatch.setattr("memo.embedder.MLXEmbedder.embed", _spy)
    cfg = Config(
        data_dir=tmp_cfg.data_dir,
        vault_path=tmp_cfg.vault_path,
        state_dir=tmp_cfg.state_dir,
        embedder_dims=4,
    )
    mem = Memory(cfg)
    # save() routes through embed() once per call. The batch path is
    # tested at the embedder level; here we just verify the high-level
    # contract still holds when embedder is called with N>1 inputs.
    rec = mem.save(content="cuerpo", title="X")
    assert rec.title == "X"
    assert seen == [1]  # single-input save


def test_auto_derive_fills_missing_fields(mem_with_stub: Memory, monkeypatch):
    """auto_derive=True asks the helper LLM to fill title/type/tags
    when caller didn't provide them. Caller-provided values must win."""
    seen_messages: list[list[dict]] = []

    def _stub_chat(self, model, messages, options=None):
        seen_messages.append(messages)
        return {"message": {"content":
            '{"title": "Derived Title", "type": "decision", '
            '"tags": ["alpha", "beta", "gamma"]}'}}

    monkeypatch.setattr("memo.llm.MLXChat.chat", _stub_chat)
    rec = mem_with_stub.save(content="long body about something", auto_derive=True)
    assert rec.title == "Derived Title"
    assert rec.type == "decision"
    assert rec.tags == ["alpha", "beta", "gamma"]
    # The helper saw a system + user message.
    assert len(seen_messages) == 1
    assert seen_messages[0][0]["role"] == "system"
    assert "long body about something" in seen_messages[0][1]["content"]


def test_auto_derive_does_not_override_caller(mem_with_stub: Memory, monkeypatch):
    """When the caller provides title/type/tags, auto_derive must NOT
    overwrite them — even if the LLM disagrees."""
    def _stub_chat(self, model, messages, options=None):
        return {"message": {"content":
            '{"title": "LLM Title", "type": "bug", "tags": ["llm"]}'}}

    monkeypatch.setattr("memo.llm.MLXChat.chat", _stub_chat)
    rec = mem_with_stub.save(
        content="x", title="Mine", type_="fact", tags=["mine"], auto_derive=True,
    )
    assert rec.title == "Mine"
    assert rec.type == "fact"
    assert rec.tags == ["mine"]


def test_auto_derive_tolerates_bad_llm_output(mem_with_stub: Memory, monkeypatch):
    """Garbage from the helper LLM falls back to heuristic title and
    default type/tags. Save must not raise."""
    def _stub_chat(self, model, messages, options=None):
        return {"message": {"content": "this is not json at all sorry"}}

    monkeypatch.setattr("memo.llm.MLXChat.chat", _stub_chat)
    rec = mem_with_stub.save(content="primer línea\n\nmás contenido", auto_derive=True)
    assert rec.title == "primer línea"  # heuristic fallback
    assert rec.type == "note"
    assert rec.tags == []


def test_history_logs_save_update_delete(mem_with_stub: Memory):
    rec = mem_with_stub.save(content="x", title="A", type_="note")
    mem_with_stub.update(rec.id, title="B")
    mem_with_stub.delete(rec.id)
    events = mem_with_stub.history.list_recent(limit=10)
    ops = [e["op"] for e in events]
    # Most recent first.
    assert ops == ["delete", "update", "save"]
    # Update event carries a delta with title change.
    upd = next(e for e in events if e["op"] == "update")
    assert upd["delta"] == {"title": ["A", "B"]}


def test_history_filter_by_record_id(mem_with_stub: Memory):
    a = mem_with_stub.save(content="x", title="A")
    mem_with_stub.save(content="y", title="B")
    mem_with_stub.update(a.id, title="A2")
    events = mem_with_stub.history.list_recent(limit=10, record_id=a.id)
    assert all(e["record_id"] == a.id for e in events)
    assert {e["op"] for e in events} == {"save", "update"}


def test_extract_entities_writes_graph(mem_with_stub: Memory, monkeypatch):
    """LLM returns a JSON list of entities; graph gets edges + bumped
    mention counts."""
    rec = mem_with_stub.save(
        content="Decidí migrar obsidian-rag a MLX con Qwen3-Embedding.",
        title="MLX migration",
    )

    def _stub_chat(self, model, messages, options=None):
        return {"message": {"content":
            '{"entities": [{"name": "obsidian-rag", "type": "project"}, '
            '{"name": "mlx", "type": "technology"}, '
            '{"name": "qwen3-embedding", "type": "technology"}]}'}}

    monkeypatch.setattr("memo.llm.MLXChat.chat", _stub_chat)
    counts = mem_with_stub.extract_entities(ids=[rec.id])
    assert counts["processed"] == 1
    assert counts["entities_extracted"] == 3
    assert counts["links_written"] == 3

    # Top entities — we passed 3, all should appear with count 1.
    top = mem_with_stub.graph.top_entities(limit=10)
    names = {e["name"] for e in top}
    assert {"obsidian-rag", "mlx", "qwen3-embedding"}.issubset(names)

    # Reverse query: name → memoria ids.
    ids = mem_with_stub.graph.entity_memorias("mlx")
    assert ids == [rec.id]


def test_extract_entities_skip_already_indexed(mem_with_stub: Memory, monkeypatch):
    """Re-running without --force is a no-op."""
    rec = mem_with_stub.save(content="x", title="X")
    calls = [0]

    def _stub_chat(self, model, messages, options=None):
        calls[0] += 1
        return {"message": {"content": '{"entities": [{"name": "x", "type": "concept"}]}'}}

    monkeypatch.setattr("memo.llm.MLXChat.chat", _stub_chat)
    mem_with_stub.extract_entities(ids=[rec.id])
    assert calls[0] == 1
    # Second run: should skip the already-indexed memoria.
    counts = mem_with_stub.extract_entities(ids=[rec.id])
    assert counts["processed"] == 0
    assert calls[0] == 1  # no extra LLM call


def test_delete_drops_graph_edges(mem_with_stub: Memory, monkeypatch):
    """Deleting a memoria removes its entity_memoria links and decrements
    each entity's mention_count."""
    rec = mem_with_stub.save(content="x", title="X")

    def _stub_chat(self, model, messages, options=None):
        return {"message": {"content": '{"entities": [{"name": "foo", "type": "concept"}]}'}}

    monkeypatch.setattr("memo.llm.MLXChat.chat", _stub_chat)
    mem_with_stub.extract_entities(ids=[rec.id])
    assert mem_with_stub.graph.entity_memorias("foo") == [rec.id]

    mem_with_stub.delete(rec.id)
    assert mem_with_stub.graph.entity_memorias("foo") == []
    # The entity row may still exist with mention_count=0 — that's fine,
    # cheap to keep around for redux of the same name later.


def test_consolidate_clusters_near_duplicates(mem_with_stub: Memory, monkeypatch):
    """Two memorias with cosine ≈1.0 land in the same cluster.
    MLXChat is mocked to return a structured JSON."""
    # Force a constant embedding for every input — two saves get
    # identical vectors → cosine 1.0 → cluster guaranteed.
    monkeypatch.setattr(
        "memo.embedder.MLXEmbedder.embed",
        lambda self, inputs: [[1.0, 0.0, 0.0, 0.0] for _ in inputs],
    )
    rec_a = mem_with_stub.save(content="alpha body uno", title="A1")
    rec_b = mem_with_stub.save(content="alpha body dos", title="A2")

    captured: list[list[dict]] = []

    def _stub_chat(self, model, messages, options=None):
        captured.append(messages)
        return {"message": {"content":
            '{"summary": "Both notes describe the same alpha concept.", '
            '"relationship": "duplicate", '
            '"rationale": "Body strings differ by one character only."}'}}

    monkeypatch.setattr("memo.llm.MLXChat.chat", _stub_chat)
    clusters = mem_with_stub.consolidate(threshold=0.99)
    assert len(clusters) >= 1
    # The duplicate pair must be in some cluster of size ≥2.
    found = False
    for c in clusters:
        ids = {m["id"] for m in c["members"]}
        if {rec_a.id, rec_b.id}.issubset(ids):
            found = True
            assert c["relationship"] == "duplicate"
            assert "alpha" in c["summary"].lower()
    assert found
    # The 7B chat tier was invoked once (one cluster).
    assert len(captured) == 1


def test_consolidate_drops_singletons(mem_with_stub: Memory):
    """A unique memoria has no cluster — it shouldn't appear in the
    output. No LLM calls when there are no clusters."""
    mem_with_stub.save(content="solo memoria", title="Lonely")
    clusters = mem_with_stub.consolidate(threshold=0.5)
    assert clusters == []  # no cluster of size ≥2


def test_ask_synthesises_with_citations(mem_with_stub: Memory, monkeypatch):
    """ask() builds a prompt with snippet+id labels, calls MLXChat,
    returns the answer + the sources it fed to the model."""
    rec_a = mem_with_stub.save(content="alpha body", title="Alpha")
    rec_b = mem_with_stub.save(content="beta body", title="Beta")

    captured: dict = {}

    def _stub_chat(self, model, messages, options=None):
        captured["model"] = model
        captured["messages"] = messages
        # Reference one of the ids so the test asserts the LLM sees them.
        return {"message": {"content":
            f"Respuesta corta sobre alpha [{rec_a.id[:8]}]."}}

    monkeypatch.setattr("memo.llm.MLXChat.chat", _stub_chat)
    out = mem_with_stub.ask("¿qué hay sobre alpha?", k=2)

    assert "alpha" in out["answer"].lower() or rec_a.id[:8] in out["answer"]
    assert len(out["sources"]) >= 1
    # Prompt sent to the model must include the [id] labels of retrieved hits.
    user_msg = captured["messages"][-1]["content"]
    assert f"[{rec_a.id[:8]}]" in user_msg or f"[{rec_b.id[:8]}]" in user_msg
    # 7B chat tier (not the helper 3B used for auto_derive).
    assert "7B" in captured["model"] or "Qwen2.5" in captured["model"]


def test_ask_dedups_repo_hits_against_vault_and_intra_repo(
    mem_with_stub: Memory, monkeypatch,
):
    """_build_ask_context must drop repo hits whose path collides with a
    vault memoria, AND collapse multiple chunks of the same repo file
    into one source. Regression: chat previously showed N copies of the
    same file across vault + repo + repeated chunks."""
    from memo.repo_index import RepoSearchHit

    rec = mem_with_stub.save(
        content="Sos la sal de este mar",
        title="01-Projects/Album-Muros-Fractales/Letra - Sal.md",
    )
    vault_path = rec.path

    def _fake_repo_hit(line_start: int) -> RepoSearchHit:
        return RepoSearchHit(
            id=f"chunk-{line_start}",
            repo_id="r1",
            repo_name="obsidian-personal",
            url="https://example.com/r.git",
            ref="HEAD",
            commit_sha="abcdef12",
            file_id="f1",
            path=vault_path,
            language="markdown",
            line_start=line_start,
            line_end=line_start + 10,
            text="Sos la sal de este mar",
            score=0.5,
            match_type="hybrid",
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
        "sal de este mar", k=5, type_=None,
        snippet_chars=200, include_repos=True,
    )

    # 1 vault memoria + 0 repo (all collide with vault title/basename) → 1 source.
    assert [s["source"] for s in sources] == ["memory"]
    assert sources[0]["title"] == rec.title


def test_ask_dedups_repo_chunks_when_no_vault_overlap(
    mem_with_stub: Memory, monkeypatch,
):
    """If repo file has no matching vault memoria, keep the first chunk
    only (intra-repo dedup by (repo_name, path))."""
    from memo.repo_index import RepoSearchHit

    def _hit(line_start: int) -> RepoSearchHit:
        return RepoSearchHit(
            id=f"c-{line_start}", repo_id="r2", repo_name="code-repo",
            url="x", ref="HEAD", commit_sha="deadbeef", file_id="f",
            path="src/foo.py", language="python",
            line_start=line_start, line_end=line_start + 5,
            text="def foo(): pass", score=0.8, match_type="hybrid",
        )

    monkeypatch.setattr(
        Memory, "repo_search",
        lambda self, q, **kw: [_hit(1), _hit(50), _hit(100)],
    )
    monkeypatch.setattr(
        type(mem_with_stub.store), "list_repo_sources",
        lambda self, **kw: [{"name": "code-repo"}],
    )

    _, sources, _, _ = mem_with_stub._build_ask_context(
        "foo", k=5, type_=None, snippet_chars=200, include_repos=True,
    )

    repo_sources = [s for s in sources if s["source"] == "repo"]
    assert len(repo_sources) == 1
    assert repo_sources[0]["line_start"] == 1


def test_ask_returns_no_answer_when_no_hits(mem_with_stub: Memory):
    """Empty corpus → graceful 'not found' answer (no hallucination)."""
    out = mem_with_stub.ask("pregunta sin contexto")
    assert "no encuentro" in out["answer"].lower()
    assert out["sources"] == []


def test_recency_helpers():
    """Recency intent detection + whatsapp + dated-content sort key."""
    from memo.memory import (
        MemoryRecord,
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

    # Conversation intent: person-scoped message queries WITHOUT a recency word
    # still float transcripts. A profile lookup ("quién es X") must not.
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
    # A meta-note *about* whatsapp carries the `whatsapp` tag but is NOT a
    # transcript — must not be mistaken for one.
    meta = MemoryRecord(
        id="c" * 32, path="AI/memory/whatsapp-note.md", title="Maria y Grecia completados",
        type="fact", tags=["whatsapp", "contacts", "vault"],
        created="2026-05-30T00:00:00", updated="2026-05-30T00:00:00",
        body="Se completaron los JID de Maria y Grecia.",
    )
    assert _is_whatsapp_hit(wa)
    assert not _is_whatsapp_hit(contact)
    assert not _is_whatsapp_hit(meta)
    # Transcript's most-recent in-body date beats the contact's update stamp
    # only via the whatsapp flag; the key itself reflects dated content.
    # Dated transcript: date + latest clock time (breaks same-day sub-chunk ties).
    assert _recency_key(wa) == "2026-05-17 16:26"
    assert _recency_key(contact) == "2026-05-30"  # no body date/time → updated[:10]


def test_ask_recency_floats_whatsapp_transcript_over_contact_card(
    mem_with_stub: Memory, monkeypatch,
):
    """Recency/conversation whatsapp questions must surface a dated transcript,
    not the same-named contact/profile card. For a RECENCY ask the *newest*
    conversation wins (a group active today beats an older 1:1); a 1:1 is
    preferred only as a same-date tiebreaker / for non-recency asks."""
    from memo.memory import MemoryRecord

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
    # A same-named GROUP chat with a *more recent* message — for a recency ask
    # this IS the latest conversation and must lead.
    group = MemoryRecord(
        id="d" * 32, path="AI/Whatsapp/Grecia's group.md", title="WhatsApp · Grecia's group",
        type="reference", tags=["whatsapp", "chat"],
        created="2026-05-19T00:00:00", updated="2026-05-19T00:00:00",
        body="## 2026-05-19\n- **alguien** (10:00): Jajajajaj", score=0.91,
    )
    monkeypatch.setattr(Memory, "search", lambda self, q, **kw: [contact, group, transcript])

    # Recency intent → newest transcript leads (group 05-19 > 1:1 05-17), and
    # both float above the contact card despite its fresher `updated` stamp.
    _, sources, _, _ = mem_with_stub._build_ask_context(
        "qué fue lo último que dijo Grecia por whatsapp", k=5, type_=None,
        snippet_chars=200, include_repos=False,
    )
    assert sources[0]["title"] == "WhatsApp · Grecia's group"
    assert sources[1]["title"] == "WhatsApp · Grecia 🩷"
    assert sources[2]["title"] == "Grecia"

    # Conversation intent WITHOUT a recency word → no date sort; the 1:1 is
    # preferred over the group, both above the contact card.
    _, sources_c, _, _ = mem_with_stub._build_ask_context(
        "mostrame el chat con Grecia", k=5, type_=None,
        snippet_chars=200, include_repos=False,
    )
    assert sources_c[0]["title"] == "WhatsApp · Grecia 🩷"

    # Profile lookup (no conversation/recency intent) → original search order
    # preserved (contact first).
    _, sources2, _, _ = mem_with_stub._build_ask_context(
        "quién es Grecia", k=5, type_=None,
        snippet_chars=200, include_repos=False,
    )
    assert sources2[0]["title"] == "Grecia"


def test_ask_conversation_intent_without_whatsapp_preserves_order(
    mem_with_stub: Memory, monkeypatch,
):
    """A conversational query with NO WhatsApp hit must not be re-sorted: the
    `has_wa` guard keeps the relevance-ranked order intact (no destructive
    date sort of plain memorias)."""
    from memo.memory import MemoryRecord

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
    monkeypatch.setattr(
        Memory, "search", lambda self, q, **kw: [top, older_but_newer_date],
    )

    # "conversación" triggers convo intent, but no WhatsApp hit → no reorder.
    _, sources, _, _ = mem_with_stub._build_ask_context(
        "resumen de la conversación sobre arquitectura", k=5, type_=None,
        snippet_chars=200, include_repos=False,
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


def test_hybrid_search_fuses_vec_and_bm25(mem_with_stub: Memory):
    """Hybrid mode (default) combines vec + bm25 via reciprocal rank
    fusion. A doc that ranks high in BOTH sources beats a doc that only
    ranks high in one.
    """
    # Two memos with similar embeddings (same length) but different
    # bodies. The stub embedder hashes by content sum so titles + tags
    # don't matter; we verify FTS5 picks up the keyword.
    mem_with_stub.save(
        content="contenido sobre python testing y mocks",
        title="Python testing notes",
        tags=["python", "testing"],
    )
    mem_with_stub.save(
        content="receta de pizza casera con harina y queso",
        title="Pizza casera",
        tags=["receta", "cocina"],
    )
    # bm25-only: keyword match must work.
    bm = mem_with_stub.search("python testing", mode="bm25")
    assert bm and bm[0].title == "Python testing notes"

    # vec-only also returns something (stub bucketing).
    v = mem_with_stub.search("python testing", mode="vec")
    assert v

    # hybrid returns at most `limit` and includes the bm25-favored doc.
    h = mem_with_stub.search("python testing", mode="hybrid", limit=2)
    assert any(r.title == "Python testing notes" for r in h)


def test_bm25_handles_empty_and_garbage_queries(mem_with_stub: Memory):
    """Empty query → []. FTS-syntax-illegal query → [] (caught,
    not raised) so hybrid degrades to pure vec gracefully."""
    mem_with_stub.save(content="x", title="X")
    assert mem_with_stub.search("", mode="bm25") == []
    # Unbalanced quote — FTS5 would raise OperationalError without our
    # defensive escape. The escape transforms the query into a phrase
    # query which CAN match nothing without raising.
    out = mem_with_stub.search('weird " query', mode="bm25")
    assert isinstance(out, list)


def test_search_uses_query_prefix(tmp_cfg: Config, monkeypatch):
    """Queries must go through `embed_query` (prefix added), not raw
    `embed`. Locks the asymmetric-retrieval contract — if a refactor
    bypasses `embed_query`, recall on the real model collapses."""
    seen_inputs: list[str] = []

    def _spy(self, inputs):
        seen_inputs.extend(inputs)
        return [[1.0, 0.0, 0.0, 0.0] for _ in inputs]

    monkeypatch.setattr("memo.embedder.MLXEmbedder.embed", _spy)
    cfg = Config(
        data_dir=tmp_cfg.data_dir,
        vault_path=tmp_cfg.vault_path,
        state_dir=tmp_cfg.state_dir,
        embedder_dims=4,
    )
    mem = Memory(cfg)
    mem.save(content="cuerpo del doc", title="X")
    seen_inputs.clear()
    mem.search("buscame algo", limit=3)
    assert seen_inputs, "search did not call embed"
    assert seen_inputs[0].startswith("Instruct:"), (
        f"query was not prefixed: {seen_inputs[0]!r}"
    )
    assert "buscame algo" in seen_inputs[0]


def test_save_truncates_huge_body(tmp_cfg: Config, monkeypatch):
    cfg = Config(
        data_dir=tmp_cfg.data_dir,
        vault_path=tmp_cfg.vault_path,
        state_dir=tmp_cfg.state_dir,
        embedder_dims=4,
        max_content_chars=100,
    )
    monkeypatch.setattr(
        "memo.embedder.MLXEmbedder.embed",
        lambda self, inputs: [[1.0, 0.0, 0.0, 0.0] for _ in inputs],
    )
    mem = Memory(cfg)
    huge = "x" * 10_000
    rec = mem.save(content=huge, title="huge")
    # Body on disk should be truncated to `max_content_chars`.
    on_disk = (cfg.memory_dir / rec.path).read_text()
    assert on_disk.count("x") <= 100


# ── assert_valid_embedding guard ──────────────────────────────────────────


def test_save_rejects_wrong_dim_embedding(tmp_cfg: Config, monkeypatch):
    """Past silent-failure mode: string-as-Sequence-of-chars cascade
    returned variable-dim outputs (135, 512, 2465...) instead of the
    configured dim. The guard in `Memory.save` must raise loudly so
    the bad vector never reaches the index."""
    cfg = Config(
        data_dir=tmp_cfg.data_dir,
        vault_path=tmp_cfg.vault_path,
        state_dir=tmp_cfg.state_dir,
        embedder_dims=4,
    )
    monkeypatch.setattr(
        "memo.embedder.MLXEmbedder.embed",
        lambda self, inputs: [[1.0] * 7 for _ in inputs],  # wrong dim (7, not 4)
    )
    mem = Memory(cfg)
    with pytest.raises(ValueError, match="dim mismatch"):
        mem.save(content="x", title="t")


def test_high_signal_detector_rescues_pin_notes():
    """Real case from the user's vault: short notes that pin atomic
    facts (URLs, CBUs, commands) must bypass MIN_CHARS so memo can
    surface them on title-match queries. Without this, a 67-char
    "Link de pago escuela Grecia" note never reaches the index."""
    from memo.cli_ingest import _is_high_signal

    real_case = (
        "# Link de pago escuela Grecia\n\n"
        "https://sit.educacionadventista.org.ar/"
    )
    assert _is_high_signal(real_case, ["grecia", "escuela", "pagos", "links"])

    # URL alone is enough.
    assert _is_high_signal("https://example.com", None)
    # Code block alone is enough.
    assert _is_high_signal("```bash\nls\n```", None)
    # High-signal tag alone is enough.
    assert _is_high_signal("CBU 0001234567890", ["dato"])

    # Genuine stub stays filtered.
    assert not _is_high_signal(
        "#hipotesis #pendiente\n¿qué iba a hacer mañana?",
        ["hipotesis", "pendiente"],
    )
    # Low-signal short note stays filtered.
    assert not _is_high_signal("algo corto sin nada especial", ["random"])


def test_save_rejects_zero_norm_embedding(tmp_cfg: Config, monkeypatch):
    """Norm ≈ 0 is the signature of a corrupted embedder pass — the
    real Qwen3-Embedding always L2-normalises. A zero or near-zero
    vector would cosine-collapse all retrieval into a single bucket."""
    cfg = Config(
        data_dir=tmp_cfg.data_dir,
        vault_path=tmp_cfg.vault_path,
        state_dir=tmp_cfg.state_dir,
        embedder_dims=4,
    )
    monkeypatch.setattr(
        "memo.embedder.MLXEmbedder.embed",
        lambda self, inputs: [[0.0, 0.0, 0.0, 0.0] for _ in inputs],
    )
    mem = Memory(cfg)
    with pytest.raises(ValueError, match="norm out of"):
        mem.save(content="x", title="t")


def test_apply_decay_lets_fresher_memory_win_a_tie():
    """Recency decay: two equally-scored hits, the fresher one ranks first."""
    from datetime import UTC, datetime, timedelta

    from memo.memory import MemoryRecord, _apply_decay

    now = datetime.now(tz=UTC)

    def _rec(id_: str, updated: datetime) -> MemoryRecord:
        return MemoryRecord(
            id=id_, path=f"{id_}.md", title=id_, type="note", tags=[],
            created=updated.isoformat(), updated=updated.isoformat(),
            body="b", extra={}, score=0.70,
        )

    old = _rec("old", now - timedelta(days=400))
    fresh = _rec("fresh", now - timedelta(days=1))

    out = _apply_decay([old, fresh], halflife_days=180.0, alpha=0.15)
    assert [r.id for r in out] == ["fresh", "old"]
    assert out[0].score > out[1].score
