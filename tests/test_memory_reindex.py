from __future__ import annotations

from contextlib import contextmanager

import frontmatter
import pytest

from memo.config import Config
from memo.errors import StorageError
from memo.identity import normalized_content_hash
from memo.memory import Memory
from memo.store import VecStore
from tests.operational_authority import authorize_test_config


def _index_snapshot(mem: Memory, ids: list[str]) -> dict[str, tuple[dict, object, str]]:
    return {
        id_: (
            mem.store.get(id_) or {},
            mem.store.get_embedding_blob(id_),
            mem.store.get_fts_body(id_),
        )
        for id_ in ids
    }


def test_save_index_failure_recovers_on_reindex(mem_with_stub: Memory, monkeypatch):
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
    rec = mem_with_stub.save(content="contenido original", title="Editable")
    abs_path = mem_with_stub.cfg.memory_dir / rec.path
    text = abs_path.read_text(encoding="utf-8")
    abs_path.write_text(
        text.replace("contenido original", "contenido EDITADO a mano"), encoding="utf-8"
    )

    out = mem_with_stub.reindex()
    assert out["reindexed"] >= 1
    fetched = mem_with_stub.get(rec.id)
    assert fetched is not None
    assert "EDITADO a mano" in fetched.body
    assert "contenido original" not in fetched.body


def test_reindex_sanitizes_derived_index_without_rewriting_markdown(
    mem_with_stub: Memory,
) -> None:
    token = "ghp_" + "a" * 32 + "WXYZ"
    rec = mem_with_stub.save(content="safe initial", title="Editable")
    path = mem_with_stub.cfg.memory_dir / rec.path
    post = frontmatter.load(str(path))
    post.content = f"hand edited {token} <private>private note</private>"
    post["extra"] = {"nested": token}
    path.write_text(frontmatter.dumps(post), encoding="utf-8")

    out = mem_with_stub.reindex(force=True)

    assert out["reindexed"] >= 1
    assert token in path.read_text(encoding="utf-8")  # Markdown source is untouched.
    assert token not in mem_with_stub.store.get_fts_body(rec.id)
    indexed = mem_with_stub.store.get(rec.id)
    assert indexed is not None
    assert token not in repr(indexed["extra"])
    assert "_redacted" in indexed["tags"]


def test_reindex_rederives_namespaces_topics_and_hand_edited_content(
    mem_with_stub: Memory,
) -> None:
    project = mem_with_stub.save(
        content="project original",
        title="Project",
        tags=["project:alpha"],
        topic_key="project-topic",
    )
    global_record = mem_with_stub.save(
        content="global original",
        title="Global",
        auto_project=False,
        topic_key="global-topic",
    )
    unscoped = mem_with_stub.save(
        content="unscoped original",
        title="Unscoped",
        auto_project=True,
        topic_key="unscoped-topic",
    )

    project_path = mem_with_stub.cfg.memory_dir / project.path
    post = frontmatter.load(str(project_path))
    post.content = "project hand edited\nwith trailing space   "
    post["topic_key"] = "  PROJECT   TÓPIC  "
    project_path.write_text(frontmatter.dumps(post), encoding="utf-8")

    mem_with_stub.reindex(rebuild=True)

    project_identity = mem_with_stub.store.get_identity_keys(project.id)
    assert project_identity["namespace"] == "project:alpha"
    assert project_identity["topic_key"] == "project tópic"
    assert project_identity["normalized_content_hash"] == normalized_content_hash(
        "project hand edited\nwith trailing space"
    )
    assert mem_with_stub.store.get_identity_keys(global_record.id)["namespace"] == "_global"
    assert mem_with_stub.store.get_identity_keys(unscoped.id)["namespace"] == "_unscoped"


def test_incremental_reindex_honors_removed_topic_key(mem_with_stub: Memory) -> None:
    record = mem_with_stub.save(
        content="topic may be removed by hand",
        title="Removable topic",
        topic_key="remove-me",
        auto_project=False,
    )
    path = mem_with_stub.cfg.memory_dir / record.path
    post = frontmatter.load(str(path))
    del post["topic_key"]
    path.write_text(frontmatter.dumps(post), encoding="utf-8")

    mem_with_stub.reindex()

    assert mem_with_stub.store.get_identity_keys(record.id)["topic_key"] is None


def test_rebuild_keeps_ambiguous_namespaces_readable(mem_with_stub: Memory) -> None:
    record = mem_with_stub.save(
        content="historical ambiguous tags",
        title="Ambiguous tags",
        tags=["project:one"],
    )
    path = mem_with_stub.cfg.memory_dir / record.path
    post = frontmatter.load(str(path))
    post["tags"] = ["project:one", "project:two"]
    path.write_text(frontmatter.dumps(post), encoding="utf-8")

    mem_with_stub.reindex(rebuild=True)

    assert mem_with_stub.store.get(record.id) is not None
    assert mem_with_stub.store.get_identity_keys(record.id)["namespace"] is None
    assert mem_with_stub.store.identity_diagnostics()["legacy_identity_rows"] == 1


def test_rebuild_blocks_then_reenables_topic_constraint(mem_with_stub: Memory) -> None:
    records = [
        mem_with_stub.save(
            content=f"collision body {index}",
            title=f"Collision {index}",
            tags=["project:alpha"],
            topic_key=f"original-{index}",
        )
        for index in range(2)
    ]
    for record in records:
        path = mem_with_stub.cfg.memory_dir / record.path
        post = frontmatter.load(str(path))
        post["topic_key"] = "hand-collision"
        path.write_text(frontmatter.dumps(post), encoding="utf-8")

    mem_with_stub.reindex(rebuild=True)

    diagnostics = mem_with_stub.store.identity_diagnostics()
    assert diagnostics["identity_constraint"] == "blocked"
    assert diagnostics["topic_collision_groups"] == 1
    assert all(mem_with_stub.store.get(record.id) is not None for record in records)

    second_path = mem_with_stub.cfg.memory_dir / records[1].path
    second_post = frontmatter.load(str(second_path))
    second_post["topic_key"] = "collision-fixed"
    second_path.write_text(frontmatter.dumps(second_post), encoding="utf-8")
    mem_with_stub.reindex(rebuild=True)

    repaired = mem_with_stub.store.identity_diagnostics()
    assert repaired["identity_constraint"] == "enabled"
    assert repaired["topic_collision_groups"] == 0


def test_reindex_folds_valid_at_from_markdown(mem_with_stub: Memory):
    """A hand-edited `valid_at` in the canonical markdown wins on reindex
    (disk overwrites the index), like the other allowlisted frontmatter keys."""
    rec = mem_with_stub.save(content="prod db is postgres", title="Fact", type_="fact")
    assert rec.valid_at is not None
    abs_path = mem_with_stub.cfg.memory_dir / rec.path
    post = frontmatter.load(str(abs_path))
    post["valid_at"] = "2020-01-01T00:00:00"
    abs_path.write_text(frontmatter.dumps(post), encoding="utf-8")

    mem_with_stub.reindex()
    fetched = mem_with_stub.get(rec.id)
    assert fetched is not None
    assert fetched.valid_at == "2020-01-01T00:00:00"


def test_reindex_keeps_index_invalid_at_when_markdown_omits_it(mem_with_stub: Memory):
    """`invalid_at` is written to frontmatter ONLY when non-None (open intervals
    omit it, like `verified_at`). So a missing `invalid_at` key on reindex must
    mean "leave the existing index value as-is", never "clear it" — while a
    present `valid_at` still folds from disk."""
    rec = mem_with_stub.save(content="prod db is postgres", title="Fact", type_="fact")
    assert mem_with_stub.get(rec.id).invalid_at is None

    # Simulate an index-only closed interval (e.g. a contradiction-supersede that
    # set invalid_at directly) while the markdown — an OPEN interval by the
    # frontmatter convention — omits the key entirely.
    with mem_with_stub.store._tx() as cx:
        cx.execute(
            "UPDATE meta SET invalid_at = ? WHERE id = ?",
            ("2030-01-01T00:00:00", rec.id),
        )
    assert mem_with_stub.get(rec.id).invalid_at == "2030-01-01T00:00:00"

    # Edit valid_at in the canonical markdown; leave invalid_at absent.
    abs_path = mem_with_stub.cfg.memory_dir / rec.path
    post = frontmatter.load(str(abs_path))
    post["valid_at"] = "2019-05-05T00:00:00"
    assert "invalid_at" not in post.metadata
    abs_path.write_text(frontmatter.dumps(post), encoding="utf-8")

    mem_with_stub.reindex()
    fetched = mem_with_stub.get(rec.id)
    assert fetched is not None
    assert fetched.valid_at == "2019-05-05T00:00:00"  # markdown wins
    assert fetched.invalid_at == "2030-01-01T00:00:00"  # absent key = leave as-is


def test_reindex_rebuild_preserves_signal(mem_with_stub: Memory):
    a = mem_with_stub.save(content="memoria una", title="A")
    b = mem_with_stub.save(content="memoria dos", title="B")
    store = mem_with_stub.store
    store.touch([a.id])
    store.touch([a.id])
    store.boost_roi_batch([b.id], delta=0.3)
    assert store.get_access(a.id)["access_count"] == 2
    assert store.get_health_batch([b.id])[b.id]["roi_score"] > 1.0

    out = mem_with_stub.reindex(rebuild=True)
    assert out["added"] == 2
    assert mem_with_stub.store.count() == 2
    assert store.get_access(a.id)["access_count"] == 2
    assert store.get_health_batch([b.id])[b.id]["roi_score"] > 1.0
    assert mem_with_stub.get(a.id) is not None
    assert mem_with_stub.get(b.id) is not None


def test_reindex_rebuild_migrates_vector_dimensions_and_model_identity(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    """The documented model/profile migration works across FLOAT[N] widths.

    The rebuild must atomically recreate every model-owned vec0 table, preserve
    non-vector feedback rows, re-embed their queries, and leave schema_meta in a
    state that a normal (non-bypassed) next process can open.
    """
    data_dir = tmp_path / "data"
    state_dir = tmp_path / "state"
    data_dir.mkdir()
    state_dir.mkdir()
    record_id = "a" * 32
    old = VecStore(state_dir / "memvec.db", dims=4, embedder_model="vendor/old-model")
    old.upsert(
        id_=record_id,
        path="memory.md",
        title="Old",
        type_="note",
        tags=[],
        created="2026-01-01T00:00:00+00:00",
        updated="2026-01-01T00:00:00+00:00",
        body_hash="old",
        embedding=[1.0, 0.0, 0.0, 0.0],
        body_text="old body",
    )
    feedback_id = old.record_source_feedback(
        source_id=record_id,
        query_text="old query",
        query_emb=[1.0, 0.0, 0.0, 0.0],
        rating=1,
    )
    old.close()
    from memo.store.episode_store import EpisodeStore
    from memo.store.hype_store import HypeStore

    hype = HypeStore(
        state_dir / "memvec.db",
        dims=4,
        embedder_model="vendor/old-model",
    )
    hype.replace_for_memory(
        record_id,
        "old",
        "helper-model",
        [("old question", [1.0, 0.0, 0.0, 0.0])],
    )
    hype.close()
    episodes = EpisodeStore(
        state_dir / "memvec.db",
        dims=4,
        embedder_model="vendor/old-model",
    )
    episodes.upsert(
        agent="claude",
        session_id="old-session",
        content_hash="old-hash",
        embedding=[1.0, 0.0, 0.0, 0.0],
        cwd="/repo",
        updated_at="2026-01-01T00:00:00+00:00",
        summary="old episode",
        resume_command=[],
        turn_count=1,
    )
    episodes.close()

    post = frontmatter.Post(
        "new body",
        id=record_id,
        title="New",
        type="note",
        tags=[],
        created="2026-01-01T00:00:00+00:00",
        updated="2026-01-02T00:00:00+00:00",
    )
    (data_dir / "memory.md").write_text(frontmatter.dumps(post), encoding="utf-8")

    monkeypatch.setenv("MEMO_SKIP_MODEL_VERSION_CHECK", "1")
    cfg = authorize_test_config(
        Config(
            data_dir=data_dir,
            state_dir=state_dir,
            embedder_backend="mlx",
            embedder_model="vendor/new-model",
            embedder_revision="a" * 40,
            embedder_dims=5,
            reranker_enabled=False,
        )
    )
    memory = Memory(cfg)
    memory.embedder.embed = lambda inputs: [  # type: ignore[method-assign]
        [1.0, 0.0, 0.0, 0.0, 0.0] for _ in inputs
    ]
    try:
        counts = memory.reindex(rebuild=True)
        assert counts["added"] == 1
    finally:
        memory.close()
    monkeypatch.delenv("MEMO_SKIP_MODEL_VERSION_CHECK")

    new_identity = f"vendor/new-model@{'a' * 40}"
    reopened = VecStore(
        state_dir / "memvec.db",
        dims=5,
        embedder_model=new_identity,
    )
    try:
        assert reopened.get_fts_body(record_id) == "new body"
        assert [
            reopened._vec_table_dims(table)
            for table in (
                "vec",
                "repo_vec",
                "source_feedback_vec",
                "hype_vec",
                "episode_vec",
            )
        ] == [5, 5, 5, 5, 5]
        schema = {
            str(row["key"]): str(row["value"])
            for row in reopened.connection.execute("SELECT key, value FROM schema_meta").fetchall()
        }
        assert schema["embedder_model"] == new_identity
        assert schema["embedder_dims"] == "5"
        assert reopened.list_source_feedback(source_id=record_id)[0]["id"] == feedback_id
        assert reopened.find_feedback_for_source(
            record_id,
            [1.0, 0.0, 0.0, 0.0, 0.0],
            threshold=0.99,
        )
        assert reopened.connection.execute("SELECT COUNT(*) FROM hype_questions").fetchone()[0] == 0
        assert reopened.connection.execute("SELECT COUNT(*) FROM episode_meta").fetchone()[0] == 0
        assert (
            reopened.connection.execute(
                "SELECT value FROM hype_schema_meta WHERE key = 'embedder_model'"
            ).fetchone()[0]
            == new_identity
        )
        assert (
            reopened.connection.execute(
                "SELECT value FROM episode_schema_meta WHERE key = 'embedder_model'"
            ).fetchone()[0]
            == new_identity
        )
    finally:
        reopened.close()


def test_reindex_dimension_migration_rolls_back_ddl_on_row_failure(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    data_dir = tmp_path / "data"
    state_dir = tmp_path / "state"
    data_dir.mkdir()
    state_dir.mkdir()
    record_id = "b" * 32
    old = VecStore(state_dir / "memvec.db", dims=4, embedder_model="vendor/old-model")
    old.upsert(
        id_=record_id,
        path="old.md",
        title="Old",
        type_="note",
        tags=[],
        created="2026-01-01T00:00:00+00:00",
        updated="2026-01-01T00:00:00+00:00",
        body_hash="old",
        embedding=[1.0, 0.0, 0.0, 0.0],
        body_text="old body",
    )
    old.close()
    post = frontmatter.Post(
        "new body",
        id=record_id,
        title="New",
        type="note",
        tags=[],
        created="2026-01-01T00:00:00+00:00",
        updated="2026-01-02T00:00:00+00:00",
    )
    (data_dir / "new.md").write_text(frontmatter.dumps(post), encoding="utf-8")

    monkeypatch.setenv("MEMO_SKIP_MODEL_VERSION_CHECK", "1")
    memory = Memory(
        Config(
            data_dir=data_dir,
            state_dir=state_dir,
            embedder_backend="mlx",
            embedder_model="vendor/new-model",
            embedder_dims=5,
            reranker_enabled=False,
        )
    )
    memory.embedder.embed = lambda inputs: [  # type: ignore[method-assign]
        [1.0, 0.0, 0.0, 0.0, 0.0] for _ in inputs
    ]
    monkeypatch.setattr(
        memory.store,
        "_upsert_memory_row",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("injected row failure")),
    )
    try:
        with pytest.raises(StorageError, match="atomic index replace failed"):
            memory.reindex(rebuild=True)
        assert memory.store._vec_table_dims("vec") == 4
        assert memory.store.get_fts_body(record_id) == "old body"
        row = memory.store.connection.execute(
            "SELECT value FROM schema_meta WHERE key = 'embedder_model'"
        ).fetchone()
        assert row is not None and row["value"] == "vendor/old-model"
    finally:
        memory.close()


def test_reindex_rebuild_upgrades_legacy_st_index_identity(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    """3.5.1 stamped ST-built vectors with the unrelated MLX config model."""
    data_dir = tmp_path / "data"
    state_dir = tmp_path / "state"
    data_dir.mkdir()
    state_dir.mkdir()
    legacy_model = "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ"
    VecStore(state_dir / "memvec.db", dims=4, embedder_model=legacy_model).close()
    post = frontmatter.Post(
        "legacy ST body",
        id="c" * 32,
        title="Legacy ST",
        type="note",
        tags=[],
        created="2026-01-01T00:00:00+00:00",
        updated="2026-01-01T00:00:00+00:00",
    )
    (data_dir / "legacy.md").write_text(frontmatter.dumps(post), encoding="utf-8")
    cfg = authorize_test_config(
        Config(
            data_dir=data_dir,
            state_dir=state_dir,
            embedder_backend="st",
            st_embedder_model="Qwen/Qwen3-Embedding-0.6B",
            st_embedder_revision="b" * 40,
            embedder_dims=4,
            reranker_enabled=False,
        )
    )

    monkeypatch.setenv("MEMO_SKIP_MODEL_VERSION_CHECK", "1")
    memory = Memory(cfg)
    memory.embedder.embed = lambda inputs: [  # type: ignore[method-assign]
        [1.0, 0.0, 0.0, 0.0] for _ in inputs
    ]
    try:
        memory.reindex(rebuild=True)
    finally:
        memory.close()
    monkeypatch.delenv("MEMO_SKIP_MODEL_VERSION_CHECK")

    reopened = Memory(cfg)
    try:
        assert reopened.store.embedder_model == f"Qwen/Qwen3-Embedding-0.6B@{'b' * 40}"
    finally:
        reopened.close()


def test_reindex_rebuild_refused_when_data_dir_empty(mem_with_stub: Memory):
    """Rebuild against a data_dir with 0 .md must refuse, not wipe the index.
    Markdown is the only source that can repopulate it — if it vanished (deleted
    dir / half-broken clone), truncating the index destroys the last copy."""
    import pytest

    from memo.errors import StorageError

    a = mem_with_stub.save(content="no me borres", title="A")
    b = mem_with_stub.save(content="ni a mi", title="B")
    for md in mem_with_stub.cfg.memory_dir.rglob("*.md"):
        md.unlink()
    assert next(mem_with_stub.cfg.memory_dir.rglob("*.md"), None) is None
    assert mem_with_stub.store.count() == 2

    with pytest.raises(StorageError, match="refused"):
        mem_with_stub.reindex(rebuild=True)

    # index untouched — both memorias still recoverable
    assert mem_with_stub.store.count() == 2
    assert mem_with_stub.get(a.id) is not None
    assert mem_with_stub.get(b.id) is not None


def test_reindex_rebuild_refused_when_only_invalid_markdown_remains(mem_with_stub: Memory):
    rec = mem_with_stub.save(content="no me borres", title="Canonical")
    (mem_with_stub.cfg.memory_dir / rec.path).unlink()
    invalid = frontmatter.Post(
        "not a canonical memory",
        id="../../invalid",
        title="Invalid",
        type="note",
    )
    (mem_with_stub.cfg.memory_dir / "invalid.md").write_text(
        frontmatter.dumps(invalid), encoding="utf-8"
    )

    with pytest.raises(StorageError, match="refused"):
        mem_with_stub.reindex(rebuild=True)

    assert mem_with_stub.store.count() == 1
    assert mem_with_stub.get(rec.id) is not None


def test_reindex_rebuild_drops_orphans(mem_with_stub: Memory):
    a = mem_with_stub.save(content="vive", title="A")
    b = mem_with_stub.save(content="muere", title="B")
    (mem_with_stub.cfg.memory_dir / b.path).unlink()

    out = mem_with_stub.reindex(rebuild=True)
    assert out["added"] == 1
    assert mem_with_stub.get(a.id) is not None
    assert mem_with_stub.get(b.id) is None
    assert mem_with_stub.store.count() == 1


def test_reindex_rebuild_does_not_resurrect_legacy_archived_directory(
    mem_with_stub: Memory,
):
    active = mem_with_stub.save(content="sigue activa", title="Active")
    archived = mem_with_stub.save(content="ya fue archivada", title="Archived")
    source = mem_with_stub.cfg.memory_dir / archived.path
    archive_dir = mem_with_stub.cfg.memory_dir / "archived"
    archive_dir.mkdir()
    source.rename(archive_dir / source.name)

    out = mem_with_stub.reindex(rebuild=True)

    assert out["skipped"] == 1
    assert mem_with_stub.get(active.id) is not None
    assert mem_with_stub.get(archived.id) is None
    assert mem_with_stub.store.count() == 1
    assert mem_with_stub._gc_disk_orphans() == []


def test_reindex_rebuild_embed_failure_keeps_previous_index(mem_with_stub: Memory, monkeypatch):
    a = mem_with_stub.save(content="alpha survives", title="A")
    b = mem_with_stub.save(content="beta survives", title="B")
    before = _index_snapshot(mem_with_stub, [a.id, b.id])
    with mem_with_stub.store._conn:
        mem_with_stub.store._conn.execute("DELETE FROM repo_embedding_cache")

    real_embed = mem_with_stub.embedder.embed
    calls = {"n": 0}

    def _fail_second(inputs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("second embed failed (injected)")
        return real_embed(inputs)

    monkeypatch.setattr(mem_with_stub.embedder, "embed", _fail_second)

    with pytest.raises(StorageError, match="rebuild preflight failed"):
        mem_with_stub.reindex(rebuild=True)

    assert _index_snapshot(mem_with_stub, [a.id, b.id]) == before
    assert mem_with_stub.store.count() == 2


def test_reindex_rebuild_parse_failure_keeps_previous_index(mem_with_stub: Memory):
    a = mem_with_stub.save(content="alpha survives", title="A")
    b = mem_with_stub.save(content="beta survives", title="B")
    before = _index_snapshot(mem_with_stub, [a.id, b.id])
    (mem_with_stub.cfg.memory_dir / b.path).write_bytes(b"\xff\xfeinvalid utf8")

    with pytest.raises(StorageError, match="rebuild preflight failed"):
        mem_with_stub.reindex(rebuild=True)

    assert _index_snapshot(mem_with_stub, [a.id, b.id]) == before
    assert mem_with_stub.store.count() == 2


def test_reindex_rebuild_row_failure_rolls_back_previous_index(mem_with_stub: Memory, monkeypatch):
    a = mem_with_stub.save(content="alpha survives", title="A")
    b = mem_with_stub.save(content="beta survives", title="B")
    before = _index_snapshot(mem_with_stub, [a.id, b.id])
    real_upsert_row = getattr(mem_with_stub.store, "_upsert_memory_row", None)
    calls = {"n": 0}

    def _fail_second_row(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("second row failed (injected)")
        assert real_upsert_row is not None
        return real_upsert_row(*args, **kwargs)

    monkeypatch.setattr(
        mem_with_stub.store,
        "_upsert_memory_row",
        _fail_second_row,
        raising=False,
    )

    with pytest.raises(StorageError, match="atomic index replace failed"):
        mem_with_stub.reindex(rebuild=True)

    assert calls["n"] == 2
    assert _index_snapshot(mem_with_stub, [a.id, b.id]) == before
    assert mem_with_stub.store.count() == 2


def test_reindex_embedding_reuse_skips_second_pass(mem_with_stub: Memory, monkeypatch):
    mem_with_stub.save(content="alpha", title="A")
    mem_with_stub.save(content="beta", title="B")
    mem_with_stub.reindex(force=True)

    calls = {"n": 0}
    real_embed = type(mem_with_stub.embedder).embed

    def _counting(self, inputs):
        calls["n"] += 1
        return real_embed(self, inputs)

    monkeypatch.setattr("memo.embedder.MLXEmbedder.embed", _counting)
    out = mem_with_stub.reindex(force=True)
    assert out["reindexed"] == 2
    assert calls["n"] == 0


def test_reindex_force_reuses_warm_embed_cache(mem_with_stub: Memory, monkeypatch):
    rec = mem_with_stub.save(content="cuerpo", title="X")
    calls: list[int] = []
    orig = mem_with_stub.embedder.embed

    def _spy(inputs):
        calls.append(len(inputs))
        return orig(inputs)

    monkeypatch.setattr(mem_with_stub.embedder, "embed", _spy)

    counts = mem_with_stub.reindex()
    assert counts["reindexed"] == 0
    assert calls == []

    counts = mem_with_stub.reindex(force=True)
    assert counts["reindexed"] == 1
    assert calls == []

    with mem_with_stub.store._conn:
        mem_with_stub.store._conn.execute("DELETE FROM repo_embedding_cache")
    calls.clear()
    counts = mem_with_stub.reindex(force=True)
    assert counts["reindexed"] == 1
    assert calls == [1]
    fetched = mem_with_stub.get(rec.id)
    assert fetched is not None
    assert fetched.title == "X"


def test_reindex_picks_up_external_edit(mem_with_stub: Memory):
    rec = mem_with_stub.save(content="primero", title="X")
    abs_path = mem_with_stub.cfg.memory_dir / rec.path
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


def test_reindex_rebuilds_declared_fact_edges_from_frontmatter(mem_with_stub: Memory):
    import frontmatter as fm

    rec = mem_with_stub.save(content="placeholder", title="Fact Source", type_="note")
    md = mem_with_stub.cfg.memory_dir / rec.path
    post = fm.loads(md.read_text(encoding="utf-8"))
    post.metadata["fact_edges"] = [
        {
            "subject": "memo",
            "predicate": "stores",
            "object": "temporal facts",
            "valid_at": "2026-01-01T00:00:00+00:00",
        }
    ]
    md.write_text(fm.dumps(post), encoding="utf-8")

    counts = mem_with_stub.reindex()
    rows = mem_with_stub.fact_edges.query(
        subject="memo",
        as_of="2026-02-01T00:00:00+00:00",
    )

    assert counts["facts"] == 1
    assert len(rows) == 1
    assert rows[0]["source_record_id"] == rec.id
    assert rows[0]["predicate"] == "stores"
    assert rows[0]["object"] == "temporal facts"


def test_reindex_updates_fact_edges_when_frontmatter_changes(mem_with_stub: Memory):
    import frontmatter as fm

    rec = mem_with_stub.save(
        content="memo backend fact",
        title="Backend",
        type_="note",
        extra={
            "fact_edges": [
                {
                    "subject": "memo",
                    "predicate": "backend",
                    "object": "old",
                    "valid_at": "2026-01-01T00:00:00+00:00",
                }
            ]
        },
    )
    assert mem_with_stub.fact_edges.query(subject="memo", as_of="2026-02-01T00:00:00+00:00")

    md = mem_with_stub.cfg.memory_dir / rec.path
    post = fm.loads(md.read_text(encoding="utf-8"))
    extra = dict(post.metadata.get("extra") or {})
    extra["fact_edges"] = [
        {
            "subject": "memo",
            "predicate": "backend",
            "object": "new",
            "valid_at": "2026-01-01T00:00:00+00:00",
        }
    ]
    post.metadata["extra"] = extra
    md.write_text(fm.dumps(post), encoding="utf-8")

    counts = mem_with_stub.reindex()
    rows = mem_with_stub.fact_edges.query(subject="memo", as_of="2026-02-01T00:00:00+00:00")

    assert counts["facts"] == 1
    assert [r["object"] for r in rows] == ["new"]


def test_reindex_path_collision_embed_failure_preserves_previous_row(
    mem_with_stub: Memory, monkeypatch: pytest.MonkeyPatch
):
    previous = mem_with_stub.save(content="old searchable body", title="Path owner")
    mem_with_stub.store.touch([previous.id])
    before = _index_snapshot(mem_with_stub, [previous.id])
    md = mem_with_stub.cfg.memory_dir / previous.path
    replacement_id = "d" * 32
    post = frontmatter.loads(md.read_text(encoding="utf-8"))
    post["id"] = replacement_id
    post.content = "new replacement body"
    md.write_text(frontmatter.dumps(post), encoding="utf-8")

    real_embed = mem_with_stub.embedder.embed
    monkeypatch.setattr(
        mem_with_stub.embedder,
        "embed",
        lambda _inputs: (_ for _ in ()).throw(RuntimeError("injected embed failure")),
    )

    failed = mem_with_stub.reindex()

    assert failed["skipped"] == 1
    assert _index_snapshot(mem_with_stub, [previous.id]) == before
    assert mem_with_stub.store.get(replacement_id) is None
    assert mem_with_stub.store.get_access(previous.id)["access_count"] == 1

    monkeypatch.setattr(mem_with_stub.embedder, "embed", real_embed)
    recovered = mem_with_stub.reindex()

    assert recovered["added"] == 1
    assert mem_with_stub.store.get(previous.id) is None
    replacement = mem_with_stub.store.get(replacement_id)
    assert replacement is not None and replacement["path"] == previous.path
    assert mem_with_stub.store.get_fts_body(replacement_id) == "new replacement body"


def test_reindex_rebuild_drops_fact_edges_for_deleted_markdown(mem_with_stub: Memory):
    mem_with_stub.save(content="keep this markdown", title="Keep")
    rec = mem_with_stub.save(
        content="memo backend fact",
        title="Backend",
        type_="note",
        extra={
            "fact_edges": [
                {
                    "subject": "memo",
                    "predicate": "backend",
                    "object": "sqlite",
                    "valid_at": "2026-01-01T00:00:00+00:00",
                }
            ]
        },
    )
    assert mem_with_stub.fact_edges.query(subject="memo", as_of="2026-02-01T00:00:00+00:00")
    (mem_with_stub.cfg.memory_dir / rec.path).unlink()

    counts = mem_with_stub.reindex(rebuild=True)

    assert counts["facts"] == 0
    assert (
        mem_with_stub.fact_edges.query(
            subject="memo",
            as_of="2026-02-01T00:00:00+00:00",
        )
        == []
    )


def test_reindex_reclaims_path_held_by_soft_deleted_tombstone(mem_with_stub: Memory, monkeypatch):
    # Prod incident 2026-07-05: with MEMO_SOFT_DELETE on, a deleted memory
    # leaves a tombstone row that still occupies UNIQUE(meta.path). When a
    # new file (new id) reclaims the same path, reindex used to fail the
    # INSERT ("UNIQUE constraint failed: meta.path") because the collision
    # guard's lookup excluded soft-deleted rows and its soft delete kept
    # the path occupied.
    import frontmatter as fm

    monkeypatch.setenv("MEMO_SOFT_DELETE", "1")
    rec = mem_with_stub.save(content="primera encarnación", title="Reclaimable")
    rel = rec.path
    mem_with_stub.delete(rec.id)
    assert mem_with_stub.get(rec.id) is None

    new_id = "aabbccdd00112233445566778899ffee"
    md = mem_with_stub.cfg.memory_dir / rel
    md.parent.mkdir(parents=True, exist_ok=True)
    post = fm.Post(
        "segunda encarnación, mismo path",
        id=new_id,
        title="Reclaimable",
        type="note",
        created="2026-07-05T19:00:00-03:00",
        updated="2026-07-05T19:00:00-03:00",
    )
    md.write_text(fm.dumps(post), encoding="utf-8")

    counts = mem_with_stub.reindex()
    assert counts["added"] == 1
    assert counts["skipped"] == 0
    fetched = mem_with_stub.get(new_id)
    assert fetched is not None
    assert "segunda encarnación" in fetched.body
    # The tombstone was purged — the path now belongs to the new id only.
    row = mem_with_stub.store.get_by_path(rel, include_deleted=True)
    assert row is not None and row["id"] == new_id


def test_reindex_indexes_project_bucket_named_like_lifecycle_dir(mem_with_stub: Memory):
    """A memory whose project bucket would slugify to `inactive`/`archived` is
    remapped to `_inactive`/`_archived` (project.py), so reindex must NOT skip
    it — the lifecycle-archive skip only covers the bare dirs. Regression for
    the collision the inactive/-skip fix introduced."""
    rec = mem_with_stub.save(
        content="memoria en proyecto inactive", title="P", tags=["project:inactive"]
    )
    assert rec.path.startswith("_inactive/"), rec.path  # remapped bucket
    out = mem_with_stub.reindex(rebuild=True)
    assert out["added"] == 1  # NOT skipped
    assert mem_with_stub.get(rec.id) is not None
    # gc must not flag it as a disk orphan either.
    assert rec.id not in mem_with_stub.gc(fix=False)["orphan_disk"]


def test_gc_reports_and_fixes_orphans(mem_with_stub: Memory):
    a = mem_with_stub.save(content="vivo", title="A")
    b = mem_with_stub.save(content="vivo", title="B")
    (mem_with_stub.cfg.memory_dir / b.path).unlink()
    report = mem_with_stub.gc(fix=False)
    assert b.id in report["orphan_store"]
    assert a.id not in report["orphan_store"]
    assert mem_with_stub.store.get(b.id) is not None
    mem_with_stub.gc(fix=True)
    assert mem_with_stub.store.get(b.id) is None


def _unit(dims: int) -> list[float]:
    v = 1.0 / dims**0.5
    return [v] * dims


def test_gc_never_deletes_labeled_ingest_reference_rows(mem_with_stub: Memory, tmp_path):
    """Labeled ingest rows store `label/rel` paths that never resolve under
    memory_dir/vault — gc must check their recorded abs_path (or skip when
    unverifiable), not mass-delete every labeled reference row."""
    src = tmp_path / "notes" / "doc.md"
    src.parent.mkdir(parents=True)
    src.write_text("contenido de referencia", encoding="utf-8")
    live = "ref-live-0001"
    mem_with_stub.store.upsert(
        id_=live,
        path="mylabel/notes/doc.md",
        title="ref doc",
        type_="reference",
        tags=["vault:mylabel"],
        created="2026-01-01T00:00:00+00:00",
        updated="2026-01-01T00:00:00+00:00",
        body_hash="x" * 16,
        embedding=_unit(mem_with_stub.cfg.embedder_dims),
        extra={"source": "vault", "vault": "mylabel", "abs_path": str(src)},
    )
    legacy = "ref-legacy-01"
    mem_with_stub.store.upsert(
        id_=legacy,
        path="oldlabel/notes/gone.md",
        title="legacy ref",
        type_="reference",
        tags=["vault:oldlabel"],
        created="2026-01-01T00:00:00+00:00",
        updated="2026-01-01T00:00:00+00:00",
        body_hash="y" * 16,
        embedding=_unit(mem_with_stub.cfg.embedder_dims),
        extra={"source": "vault", "vault": "oldlabel"},
    )
    report = mem_with_stub.gc(fix=True)
    # Source file exists -> not an orphan. No provenance -> unverifiable, skipped.
    assert live not in report["orphan_store"]
    assert legacy not in report["orphan_store"]
    assert mem_with_stub.store.get(live) is not None
    assert mem_with_stub.store.get(legacy) is not None
    # A labeled row whose recorded source file is GONE is a real orphan.
    src.unlink()
    report = mem_with_stub.gc(fix=True)
    assert live in report["orphan_store"]
    assert mem_with_stub.store.get(live) is None


def test_reindex_updates_store_path_after_markdown_move_without_embedding(
    mem_with_stub: Memory, monkeypatch
):
    rec = mem_with_stub.save(content="same body", title="Moved")
    old_path = mem_with_stub.cfg.memory_dir / rec.path
    new_path = mem_with_stub.cfg.memory_dir / "new-bucket" / old_path.name
    new_path.parent.mkdir(parents=True)
    old_path.rename(new_path)
    new_rel = str(new_path.relative_to(mem_with_stub.cfg.memory_dir))
    with mem_with_stub.store._conn:
        mem_with_stub.store._conn.execute("DELETE FROM repo_embedding_cache")
    embed_calls: list[list[str]] = []

    def _unexpected_embed(inputs):
        embed_calls.append(inputs)
        raise AssertionError("path-only reindex must preserve the existing vector")

    monkeypatch.setattr(mem_with_stub.embedder, "embed", _unexpected_embed)

    counts = mem_with_stub.reindex()

    row = mem_with_stub.store.get(rec.id)
    assert row is not None
    assert row["path"] == new_rel
    assert counts["reindexed"] == 1
    assert embed_calls == []
    assert rec.id not in mem_with_stub.gc(fix=True)["orphan_store"]
    assert mem_with_stub.store.get(rec.id) is not None


def test_reindex_never_decreases_schema_user_version(mem_with_stub: Memory):
    mem_with_stub.save(content="versioned", title="Versioned")
    before = mem_with_stub.store.get_user_version()
    assert before >= 4

    mem_with_stub.reindex()

    assert mem_with_stub.store.get_user_version() == before


def test_reindex_title_only_change_refreshes_embedding(mem_with_stub: Memory, monkeypatch):
    rec = mem_with_stub.save(content="unchanged body", title="Old title")
    before_blob = mem_with_stub.store.get_embedding_blob(rec.id)
    md_path = mem_with_stub.cfg.memory_dir / rec.path
    post = frontmatter.loads(md_path.read_text(encoding="utf-8"))
    post["title"] = "New title"
    md_path.write_text(frontmatter.dumps(post), encoding="utf-8")

    real_embed = mem_with_stub.embedder.embed
    embed_inputs: list[str] = []

    def _spy(inputs):
        embed_inputs.extend(inputs)
        return real_embed(inputs)

    monkeypatch.setattr(mem_with_stub.embedder, "embed", _spy)

    counts = mem_with_stub.reindex()

    row = mem_with_stub.store.get(rec.id)
    assert row is not None and row["title"] == "New title"
    assert counts["reindexed"] == 1
    assert any("New title" in text for text in embed_inputs)
    assert mem_with_stub.store.get_embedding_blob(rec.id) != before_blob


def test_reindex_pending_marker_cleanup_does_not_overwrite_concurrent_edit(
    mem_with_stub: Memory, monkeypatch: pytest.MonkeyPatch
):
    rec = mem_with_stub.save(
        content="old pending body",
        title="Pending race",
        defer_embed=True,
        auto_project=False,
    )
    path = mem_with_stub.cfg.memory_dir / rec.path
    entered = False

    @contextmanager
    def _edit_while_cleanup_locks():
        nonlocal entered
        if not entered:
            entered = True
            current = frontmatter.loads(path.read_text(encoding="utf-8"))
            current.content = "NEW concurrent body"
            current["updated"] = "2026-07-15T12:00:00+00:00"
            path.write_text(frontmatter.dumps(current), encoding="utf-8")
        yield

    monkeypatch.setattr(mem_with_stub, "_data_dir_write_lock", _edit_while_cleanup_locks)
    mem_with_stub._reindex_locked(force=True)

    current = frontmatter.loads(path.read_text(encoding="utf-8"))
    assert current.content.strip() == "NEW concurrent body"
    assert (current.metadata.get("extra") or {}).get("_memo_embed_pending") is True


@pytest.mark.parametrize("rebuild", [False, True])
def test_reindex_preserves_topic_and_hash_identity(mem_with_stub: Memory, rebuild: bool):
    rec = mem_with_stub.save(
        content="identity body",
        title="Identity",
        topic_key="identity-topic",
        normalized_hash="identity-hash",
        auto_project=False,
    )

    mem_with_stub.reindex(force=True, rebuild=rebuild)

    assert mem_with_stub.store.get_dedup_keys(rec.id) == (
        "identity-topic",
        "identity-hash",
    )
    same = mem_with_stub.save(
        content="updated identity body",
        title="Identity",
        topic_key="identity-topic",
        auto_project=False,
    )
    assert same.id == rec.id
    assert mem_with_stub.store.count() == 1


def test_reindex_rebuild_preserves_verification_state(mem_with_stub: Memory):
    rec = mem_with_stub.save(content="verified body", title="Verified")
    verified_at = 1_752_600_000
    with mem_with_stub.store._conn:
        mem_with_stub.store._conn.execute(
            "UPDATE meta SET verification_state = 'verified', verified_at = ? WHERE id = ?",
            (verified_at, rec.id),
        )
    path = mem_with_stub.cfg.memory_dir / rec.path
    post = frontmatter.loads(path.read_text(encoding="utf-8"))
    post["verification_state"] = "verified"
    post["verified_at"] = verified_at
    path.write_text(frontmatter.dumps(post), encoding="utf-8")

    mem_with_stub.reindex(rebuild=True)

    current = mem_with_stub.get(rec.id)
    assert current is not None
    assert current.verification_state.value == "verified"
    assert current.verified_at == verified_at


def test_reindex_rejects_noncanonical_memory_id(mem_with_stub: Memory):
    malicious_id = "../../outside"
    md_path = mem_with_stub.cfg.memory_dir / "malicious-id.md"
    post = frontmatter.Post(
        "untrusted body",
        id=malicious_id,
        title="Malicious id",
        type="note",
        tags=["test"],
        created="2026-07-15T00:00:00+00:00",
        updated="2026-07-15T00:00:00+00:00",
    )
    md_path.write_text(frontmatter.dumps(post), encoding="utf-8")

    counts = mem_with_stub.reindex()

    assert counts["added"] == 0
    assert counts["skipped"] == 1
    assert mem_with_stub.store.get(malicious_id) is None
    assert mem_with_stub.resolve_id(malicious_id) is None


def test_reindex_refuses_symlinked_markdown(mem_with_stub: Memory, tmp_path):
    outside = tmp_path / "outside.md"
    post = frontmatter.Post(
        "EXTERNAL SECRET MARKER",
        id="aabbccdd00112233445566778899ffee",
        title="External",
        type="note",
    )
    outside.write_text(frontmatter.dumps(post), encoding="utf-8")
    (mem_with_stub.cfg.memory_dir / "linked.md").symlink_to(outside)

    counts = mem_with_stub.reindex()
    assert counts["skipped"] == 1
    assert mem_with_stub.store.get("aabbccdd00112233445566778899ffee") is None
    with pytest.raises(StorageError, match="symlinked canonical path"):
        mem_with_stub.reindex(rebuild=True)


def test_reindex_rebuild_rejects_duplicate_canonical_ids(mem_with_stub: Memory):
    rec = mem_with_stub.save(content="canonical A", title="A")
    original = frontmatter.loads(
        (mem_with_stub.cfg.memory_dir / rec.path).read_text(encoding="utf-8")
    )
    original.content = "conflicting B"
    (mem_with_stub.cfg.memory_dir / "zz-duplicate.md").write_text(
        frontmatter.dumps(original), encoding="utf-8"
    )

    with pytest.raises(StorageError, match="duplicate canonical id"):
        mem_with_stub.reindex(rebuild=True)

    assert mem_with_stub.store.count() == 1
    assert mem_with_stub.get(rec.id) is not None


def test_reindex_skips_chronicle_bucket_silently(mem_with_stub: Memory, caplog):
    chronicle = mem_with_stub.cfg.memory_dir / "_chronicle"
    chronicle.mkdir(parents=True, exist_ok=True)
    (chronicle / "2026-07-18.md").write_text(
        "# Chronicle — 2026-07-18\n\nDiario nocturno sin frontmatter id.\n",
        encoding="utf-8",
    )

    with caplog.at_level("WARNING", logger="memo.memory.record"):
        mem_with_stub.reindex()

    assert "invalid memory id" not in caplog.text
    assert mem_with_stub.store.count() == 0
