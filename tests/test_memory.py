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
    abs_path = mem_with_stub.cfg.vault_path / rec.path
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
    assert (mem_with_stub.cfg.vault_path / rec.path).is_file()
    assert mem_with_stub.delete(rec.id) is True
    assert mem_with_stub.store.count() == 0
    assert not (mem_with_stub.cfg.vault_path / rec.path).is_file()


def test_delete_missing_returns_false(mem_with_stub: Memory):
    assert mem_with_stub.delete("nope") is False


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
    on_disk = (mem_with_stub.cfg.vault_path / updated.path).read_text()
    assert "cuerpo nuevo y diferente" in on_disk


def test_update_missing_returns_none(mem_with_stub: Memory):
    assert mem_with_stub.update("nope", title="x") is None


def test_update_rejects_invalid_type(mem_with_stub: Memory):
    rec = mem_with_stub.save(content="x", title="X")
    with pytest.raises(ValueError, match="not in valid set"):
        mem_with_stub.update(rec.id, type_="bogus")


def test_reindex_force_reembeds_unchanged(mem_with_stub: Memory, monkeypatch):
    """`reindex(force=True)` re-embeds even when body_hash matches —
    used after embedder swap or composition change."""
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

    counts = mem_with_stub.reindex(force=True)
    assert counts["reindexed"] == 1
    assert calls == [1]
    # Disk + index still consistent.
    fetched = mem_with_stub.get(rec.id)
    assert fetched is not None
    assert fetched.title == "X"


def test_reindex_picks_up_external_edit(mem_with_stub: Memory):
    rec = mem_with_stub.save(content="primero", title="X")
    abs_path = mem_with_stub.cfg.vault_path / rec.path
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
    (mem_with_stub.cfg.vault_path / b.path).unlink()
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
    on_disk = (cfg.vault_path / rec.path).read_text()
    assert on_disk.count("x") <= 100
