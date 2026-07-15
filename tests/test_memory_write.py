"""High-level Memory write-path tests with stub embedder."""

from __future__ import annotations

import sqlite3
import threading
from contextlib import closing

import frontmatter
import pytest

from memo.config import Config
from memo.memory import Memory
from memo.store.queries import serialize_float32


def test_save_writes_md_and_indexes(mem_with_stub: Memory):
    rec = mem_with_stub.save(content="primer memo del test", title="Test 1", type_="note")
    abs_path = mem_with_stub.cfg.memory_dir / rec.path
    assert abs_path.is_file()
    text = abs_path.read_text(encoding="utf-8")
    assert "title: Test 1" in text
    assert "primer memo del test" in text
    assert mem_with_stub.store.count() == 1


def test_two_memory_instances_serialize_same_title_path_allocation(tmp_cfg: Config):
    """The path lock must be shared by independent Memory instances."""
    first_selected = threading.Event()
    release_first = threading.Event()
    second_done = threading.Event()
    records = []
    errors: list[Exception] = []
    first = Memory(tmp_cfg)
    second = Memory(tmp_cfg)
    original_build = first._build_rel_path

    def _pause_after_first_probe(title, now_iso, tags=None):
        candidate = original_build(title, now_iso, tags)
        first_selected.set()
        if not release_first.wait(timeout=5):
            raise TimeoutError("test did not release first path allocation")
        return candidate

    first._build_rel_path = _pause_after_first_probe  # type: ignore[method-assign]

    def _save(memory: Memory, content: str, done: threading.Event | None = None) -> None:
        try:
            records.append(
                memory.save(
                    content=content,
                    title="Concurrent title",
                    defer_embed=True,
                    auto_project=False,
                )
            )
        except Exception as exc:
            errors.append(exc)
        finally:
            if done is not None:
                done.set()

    one = threading.Thread(target=_save, args=(first, "first body"), daemon=True)
    two = threading.Thread(
        target=_save,
        args=(second, "second body", second_done),
        daemon=True,
    )
    try:
        one.start()
        assert first_selected.wait(timeout=5)
        two.start()
        # Under the old per-instance threading.Lock, the second save completes
        # while the first is paused after selecting the same free filename.
        # A shared data-dir flock keeps it blocked here.
        second_done.wait(timeout=0.5)
        release_first.set()
        one.join(timeout=5)
        two.join(timeout=5)

        assert not one.is_alive() and not two.is_alive()
        assert errors == []
        assert len(records) == 2
        assert len({record.path for record in records}) == 2
        assert first.store.count() == 2
        for record in records:
            post = frontmatter.loads((tmp_cfg.memory_dir / record.path).read_text(encoding="utf-8"))
            assert post.metadata["id"] == record.id
            assert post.content.strip() == record.body
    finally:
        release_first.set()
        first.close()
        second.close()


def test_two_memory_instances_do_not_duplicate_topic_key(tmp_cfg: Config):
    first = Memory(tmp_cfg)
    second = Memory(tmp_cfg)
    first_reservation = threading.Event()
    release_first = threading.Event()
    second_done = threading.Event()
    records = []
    errors: list[Exception] = []
    original_upsert = first.store.upsert_text_only

    def _pause_before_first_index(**kwargs):
        first_reservation.set()
        if not release_first.wait(timeout=5):
            raise TimeoutError("test did not release first topic reservation")
        return original_upsert(**kwargs)

    first.store.upsert_text_only = _pause_before_first_index  # type: ignore[method-assign]

    def _save(memory: Memory, body: str, done: threading.Event | None = None) -> None:
        try:
            records.append(
                memory.save(
                    content=body,
                    title="Shared topic",
                    topic_key="shared-topic-key",
                    defer_embed=True,
                    auto_project=False,
                )
            )
        except Exception as exc:
            errors.append(exc)
        finally:
            if done is not None:
                done.set()

    one = threading.Thread(target=_save, args=(first, "first"), daemon=True)
    two = threading.Thread(target=_save, args=(second, "second", second_done), daemon=True)
    try:
        one.start()
        assert first_reservation.wait(timeout=5)
        two.start()
        second_done.wait(timeout=0.5)
        release_first.set()
        one.join(timeout=5)
        two.join(timeout=5)

        assert not one.is_alive() and not two.is_alive()
        assert errors == []
        assert len(records) == 2
        assert records[0].id == records[1].id
        assert first.store.count() == 1
    finally:
        release_first.set()
        first.close()
        second.close()


def test_concurrent_topic_key_save_keeps_markdown_fts_and_vector_coherent(tmp_cfg: Config):
    first = Memory(tmp_cfg)
    second = Memory(tmp_cfg)
    first_embedding_started = threading.Event()
    release_first_embedding = threading.Event()
    errors: list[Exception] = []
    records = []

    first_vector = [0.0] * tmp_cfg.embedder_dims
    first_vector[0] = 1.0
    second_vector = [0.0] * tmp_cfg.embedder_dims
    second_vector[1] = 1.0

    def _slow_first_embed(_inputs):
        first_embedding_started.set()
        if not release_first_embedding.wait(timeout=5):
            raise TimeoutError("test did not release first embedding")
        return [first_vector]

    first.embedder.embed = _slow_first_embed
    second.embedder.embed = lambda _inputs: [second_vector]

    def _save(memory: Memory, body: str) -> None:
        try:
            records.append(
                memory.save(
                    content=body,
                    title="Shared topic",
                    topic_key="coherent-topic-key",
                    auto_project=False,
                )
            )
        except Exception as exc:
            errors.append(exc)

    one = threading.Thread(target=_save, args=(first, "FIRST"), daemon=True)
    two = threading.Thread(target=_save, args=(second, "SECOND"), daemon=True)
    try:
        one.start()
        assert first_embedding_started.wait(timeout=5)
        two.start()
        two.join(timeout=5)
        assert not two.is_alive()
        release_first_embedding.set()
        one.join(timeout=5)

        assert not one.is_alive()
        assert errors == []
        assert len(records) == 2
        assert records[0].id == records[1].id
        record_id = records[0].id
        current = first.get(record_id)
        assert current is not None
        post = frontmatter.loads((tmp_cfg.memory_dir / current.path).read_text(encoding="utf-8"))
        assert post.content.strip() == "SECOND"
        assert first.store.get_fts_body(record_id) == "SECOND"
        assert first.store.get_embedding_blob(record_id) == serialize_float32(second_vector)
    finally:
        release_first_embedding.set()
        first.close()
        second.close()


def test_two_memory_instances_do_not_lose_concurrent_appends(
    tmp_cfg: Config, monkeypatch: pytest.MonkeyPatch
):
    cfg = Config(
        data_dir=tmp_cfg.data_dir,
        vault_path=tmp_cfg.vault_path,
        state_dir=tmp_cfg.state_dir,
        embedder_dims=4,
        reranker_enabled=False,
    )
    first = Memory(cfg)
    second = Memory(cfg)
    first_ready = threading.Event()
    release_first = threading.Event()
    second_done = threading.Event()
    errors: list[Exception] = []

    def _vector(_text, *, ctx):
        return [1.0, 0.0, 0.0, 0.0]

    def _pause_after_first_read(_text, *, ctx):
        first_ready.set()
        if not release_first.wait(timeout=5):
            raise TimeoutError("test did not release first append")
        return [1.0, 0.0, 0.0, 0.0]

    monkeypatch.setattr(first, "_embed_cached", _vector)
    monkeypatch.setattr(second, "_embed_cached", _vector)
    rec = first.save(content="base", title="Concurrent append", auto_project=False)
    monkeypatch.setattr(first, "_embed_cached", _pause_after_first_read)

    def _append(memory: Memory, text: str, done: threading.Event | None = None) -> None:
        try:
            memory.update(rec.id, append=text)
        except Exception as exc:
            errors.append(exc)
        finally:
            if done is not None:
                done.set()

    one = threading.Thread(target=_append, args=(first, "A"), daemon=True)
    two = threading.Thread(target=_append, args=(second, "B", second_done), daemon=True)
    try:
        one.start()
        assert first_ready.wait(timeout=5)
        two.start()
        second_done.wait(timeout=0.5)
        release_first.set()
        one.join(timeout=5)
        two.join(timeout=5)

        assert not one.is_alive() and not two.is_alive()
        assert errors == []
        body = first._read_body(rec.path)
        assert "A" in body
        assert "B" in body
    finally:
        release_first.set()
        first.close()
        second.close()


def test_successful_update_clears_pending_embed_marker(
    mem_with_stub: Memory, monkeypatch: pytest.MonkeyPatch
):
    rec = mem_with_stub.save(
        content="pending body",
        title="Pending update",
        defer_embed=True,
    )
    path = mem_with_stub.cfg.memory_dir / rec.path
    assert "_memo_embed_pending" in path.read_text(encoding="utf-8")

    monkeypatch.setattr(
        mem_with_stub,
        "_embed_cached",
        lambda _text, *, ctx: [1.0, 0.0, 0.0, 0.0],
    )
    updated = mem_with_stub.update(rec.id, content="successfully embedded body")

    assert updated is not None
    assert "_memo_embed_pending" not in updated.extra
    assert "_memo_embed_pending" not in path.read_text(encoding="utf-8")
    assert "_memo_embed_pending" not in (mem_with_stub.store.get(rec.id)["extra"] or {})
    assert mem_with_stub.store.has_vector(rec.id) is True


def test_update_preserves_topic_and_normalized_hash_identity(mem_with_stub: Memory):
    rec = mem_with_stub.save(
        content="before",
        title="Stable identity",
        topic_key="stable-topic",
        normalized_hash="stable-hash",
        auto_project=False,
    )

    mem_with_stub.update(rec.id, content="after")

    assert mem_with_stub.store.get_dedup_keys(rec.id) == (
        "stable-topic",
        "stable-hash",
    )
    post = frontmatter.loads((mem_with_stub.cfg.memory_dir / rec.path).read_text(encoding="utf-8"))
    assert post.metadata["topic_key"] == "stable-topic"
    assert post.metadata["normalized_hash"] == "stable-hash"
    same = mem_with_stub.save(
        content="same topic after update",
        title="Stable identity",
        topic_key="stable-topic",
        auto_project=False,
    )
    assert same.id == rec.id
    assert mem_with_stub.store.count() == 1


def test_delete_serializes_against_inflight_update(
    tmp_cfg: Config, monkeypatch: pytest.MonkeyPatch
):
    cfg = Config(
        data_dir=tmp_cfg.data_dir,
        vault_path=tmp_cfg.vault_path,
        state_dir=tmp_cfg.state_dir,
        embedder_dims=4,
        reranker_enabled=False,
    )
    updater = Memory(cfg)
    deleter = Memory(cfg)
    update_ready = threading.Event()
    release_update = threading.Event()
    delete_done = threading.Event()
    errors: list[Exception] = []

    monkeypatch.setattr(
        updater,
        "_embed_cached",
        lambda _text, *, ctx: [1.0, 0.0, 0.0, 0.0],
    )
    monkeypatch.setattr(
        deleter,
        "_embed_cached",
        lambda _text, *, ctx: [1.0, 0.0, 0.0, 0.0],
    )
    rec = updater.save(content="base", title="Update delete race", auto_project=False)

    def _pause_embed(_text, *, ctx):
        update_ready.set()
        if not release_update.wait(timeout=5):
            raise TimeoutError("test did not release update")
        return [1.0, 0.0, 0.0, 0.0]

    monkeypatch.setattr(updater, "_embed_cached", _pause_embed)

    def _update() -> None:
        try:
            updater.update(rec.id, append="updated")
        except Exception as exc:
            errors.append(exc)

    def _delete() -> None:
        try:
            deleter.delete(rec.id)
        except Exception as exc:
            errors.append(exc)
        finally:
            delete_done.set()

    update_thread = threading.Thread(target=_update, daemon=True)
    delete_thread = threading.Thread(target=_delete, daemon=True)
    try:
        update_thread.start()
        assert update_ready.wait(timeout=5)
        delete_thread.start()
        delete_done.wait(timeout=0.5)
        release_update.set()
        update_thread.join(timeout=5)
        delete_thread.join(timeout=5)

        assert not update_thread.is_alive() and not delete_thread.is_alive()
        assert errors == []
        assert updater.store.get(rec.id) is None
        assert not (cfg.memory_dir / rec.path).exists()
    finally:
        release_update.set()
        updater.close()
        deleter.close()


def test_update_atomic_markdown_replace_failure_preserves_original(
    mem_with_stub: Memory, monkeypatch
):
    rec = mem_with_stub.save(
        content="original body",
        title="Atomic update",
        defer_embed=True,
    )
    path = mem_with_stub.cfg.memory_dir / rec.path
    original = path.read_text(encoding="utf-8")

    def _replace_fails(_source, _destination):
        raise OSError("replace interrupted")

    monkeypatch.setattr("memo.memory.write_ops.os.replace", _replace_fails)

    with pytest.raises(OSError, match="replace interrupted"):
        mem_with_stub.update(rec.id, content="new body")

    assert path.read_text(encoding="utf-8") == original
    assert not list(path.parent.glob(f".{path.name}.*.tmp"))


def test_save_indexes_entities_in_graph_db(mem_with_stub: Memory):
    rec = mem_with_stub.save(
        content="MLX and MCP share retrieval context.",
        title="MLX MCP Graph",
        type_="fact",
    )

    entity_names = {ent["name"] for ent in mem_with_stub.graph.memory_entities(rec.id)}
    assert {"mlx", "mcp"} <= entity_names

    neighbors = mem_with_stub.navigator.get_neighbors("mlx")
    assert "mcp" in neighbors.direct_neighbors
    assert rec.id in neighbors.neighbor_memories["mcp"]


def test_memory_uses_all_default_sqlite_databases(mem_with_stub: Memory):
    rec = mem_with_stub.save(
        content="MLX references [[target-memory]] and MCP.",
        title="Storage Smoke",
        type_="fact",
    )
    mem_with_stub.crossref.index_wikilinks(rec.id, "See [[target-memory]] for details")
    assert mem_with_stub.contradict_store.stats() == {}

    cfg = mem_with_stub.cfg
    expected_files = [
        cfg.db_path,
        cfg.history_db,
        cfg.graph_db,
        cfg.contradictions_db,
        cfg.crossref_db,
    ]
    assert all(path.is_file() for path in expected_files)

    checks = {
        cfg.db_path: ("meta", "id = ?", (rec.id,)),
        cfg.history_db: ("events", "record_id = ? AND op = 'save'", (rec.id,)),
        cfg.graph_db: ("entity_memory", "memory_id = ?", (rec.id,)),
        cfg.contradictions_db: ("pairs", "1 = 1", ()),
        cfg.crossref_db: ("backlinks", "source_id = ?", (rec.id,)),
    }
    for db_path, (table, where, params) in checks.items():
        with closing(sqlite3.connect(db_path)) as conn:
            count = conn.execute(
                f"SELECT COUNT(*) FROM {table} WHERE {where}",  # noqa: S608
                params,
            ).fetchone()[0]
        if table == "pairs":
            assert count == 0
        else:
            assert count > 0


def test_save_rejects_invalid_type(mem_with_stub: Memory):
    with pytest.raises(ValueError, match="not in valid set"):
        mem_with_stub.save(content="x", type_="bogus")


def test_save_rejects_empty_content(mem_with_stub: Memory):
    with pytest.raises(ValueError, match="non-empty"):
        mem_with_stub.save(content="   ")


def test_save_index_failure_keeps_md_and_marks_pending(mem_with_stub: Memory, monkeypatch):
    def _explode(self, inputs):
        raise RuntimeError("embedder down")

    monkeypatch.setattr("memo.embedder.MLXEmbedder.embed", _explode)
    rec = mem_with_stub.save(content="cuerpo recuperable", title="Recuperable")

    abs_path = mem_with_stub.cfg.memory_dir / rec.path
    assert abs_path.is_file()
    text = abs_path.read_text(encoding="utf-8")
    assert "_memo_embed_pending" in text
    assert rec.extra.get("_memo_embed_pending") is True
    assert mem_with_stub.store.get(rec.id) is not None


def test_tags_lower_dedup(mem_with_stub: Memory):
    rec = mem_with_stub.save(content="x", title="X", tags=["MLX", "mlx", "Local"])
    assert rec.tags == ["mlx", "local"]


def test_title_derived_from_first_line(mem_with_stub: Memory):
    rec = mem_with_stub.save(content="# Encabezado\n\nbody")
    assert rec.title == "Encabezado"


def test_embed_batch_preserves_order_and_handles_empty(tmp_cfg: Config, monkeypatch):
    seen: list[int] = []

    def _spy(self, inputs):
        seen.append(len(inputs))
        out = []
        for s in inputs:
            if not s:
                out.append([0.0] * 4)
            else:
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
    rec = mem.save(content="cuerpo", title="X")
    assert rec.title == "X"
    assert seen == [1]


def test_auto_derive_fills_missing_fields(mem_with_stub: Memory, monkeypatch):
    seen_messages: list[list[dict]] = []

    def _stub_chat(self, model, messages, options=None):
        seen_messages.append(messages)
        return {
            "message": {
                "content": '{"title": "Derived Title", "type": "decision", "tags": ["alpha", "beta", "gamma"]}'
            }
        }

    monkeypatch.setattr("memo.llm.MLXChat.chat", _stub_chat)
    rec = mem_with_stub.save(content="long body about something", auto_derive=True)
    assert rec.title == "Derived Title"
    assert rec.type == "decision"
    assert rec.tags == ["alpha", "beta", "gamma"]
    assert len(seen_messages) == 1
    assert seen_messages[0][0]["role"] == "system"
    assert "long body about something" in seen_messages[0][1]["content"]


def test_auto_derive_does_not_override_caller(mem_with_stub: Memory, monkeypatch):
    def _stub_chat(self, model, messages, options=None):
        return {"message": {"content": '{"title": "LLM Title", "type": "bug", "tags": ["llm"]}'}}

    monkeypatch.setattr("memo.llm.MLXChat.chat", _stub_chat)
    rec = mem_with_stub.save(
        content="x",
        title="Mine",
        type_="fact",
        tags=["mine"],
        auto_derive=True,
    )
    assert rec.title == "Mine"
    assert rec.type == "fact"
    assert rec.tags == ["mine"]


def test_auto_derive_tolerates_bad_llm_output(mem_with_stub: Memory, monkeypatch):
    def _stub_chat(self, model, messages, options=None):
        return {"message": {"content": "this is not json at all sorry"}}

    monkeypatch.setattr("memo.llm.MLXChat.chat", _stub_chat)
    rec = mem_with_stub.save(content="primer línea\n\nmás contenido", auto_derive=True)
    assert rec.title == "primer línea"
    assert rec.type == "note"
    assert rec.tags == []


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
    on_disk = (cfg.memory_dir / rec.path).read_text()
    assert on_disk.count("x") <= 110


def test_save_rejects_wrong_dim_embedding(tmp_cfg: Config, monkeypatch):
    cfg = Config(
        data_dir=tmp_cfg.data_dir,
        vault_path=tmp_cfg.vault_path,
        state_dir=tmp_cfg.state_dir,
        embedder_dims=4,
    )
    monkeypatch.setattr(
        "memo.embedder.MLXEmbedder.embed",
        lambda self, inputs: [[1.0] * 7 for _ in inputs],
    )
    mem = Memory(cfg)
    with pytest.raises(ValueError, match="dim mismatch"):
        mem.save(content="x", title="t")


def test_high_signal_detector_rescues_pin_notes():
    from memo.cli_ingest import _is_high_signal

    real_case = "# Link de pago escuela Grecia\n\nhttps://sit.educacionadventista.org.ar/"
    assert _is_high_signal(real_case, ["grecia", "escuela", "pagos", "links"])
    assert _is_high_signal("https://example.com", None)
    assert _is_high_signal("```bash\nls\n```", None)
    assert _is_high_signal("CBU 0001234567890", ["dato"])
    assert not _is_high_signal(
        "#hipotesis #pendiente\n¿qué iba a hacer mañana?",
        ["hipotesis", "pendiente"],
    )
    assert not _is_high_signal("algo corto sin nada especial", ["random"])


def test_save_rejects_zero_norm_embedding(tmp_cfg: Config, monkeypatch):
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
