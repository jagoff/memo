"""Hermetic tests for the derived-index health diagnostic.

Every test builds a real isolated ``Memory`` (via the ``mock_memory`` fixture,
which stubs the embedder) so the SQL-level checks run against a genuine
``memvec.db`` round-trip. The developer's real vault is never touched.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager

from memo.sqlite_compat import import_sqlite_vec
from memo.store.index_health import check_index_health

_serialize_float32 = import_sqlite_vec().serialize_float32


def _unit_vector(dims: int) -> list[float]:
    vec = [0.0] * dims
    vec[0] = 1.0
    return vec


def _insert_orphan_vector(mem, vec_id: str) -> None:
    """Insert a vec row with no matching meta row (a derived orphan)."""
    store = mem.store
    bind = store._vec_bind_new()
    with store._tx() as cx:
        cx.execute(
            f"INSERT INTO vec (id, embedding, type) VALUES (?, {bind}, ?)",  # noqa: S608
            (vec_id, _serialize_float32(_unit_vector(store.dims)), "note"),
        )


def test_clean_store_reports_ok(mock_memory):
    # Arrange: a store with two well-formed memories.
    mock_memory.save(content="prod db is postgres", title="DB choice", type_="fact")
    mock_memory.save(content="deploy runs at 03:00", title="Deploy window", type_="note")

    # Act
    result = check_index_health(mock_memory.cfg, mock_memory)

    # Assert: no divergence, and the informational check is still present.
    assert result["status"] == "ok"
    assert result["errors"] == []
    for name in ("orphan_vectors", "fts_missing_body", "missing_markdown", "md_divergence"):
        assert result["checks"][name]["count"] == 0
    assert "stale_caches" in result["checks"]


def test_orphan_vector_detected_and_repaired(mock_memory):
    # Arrange: one real memory + one injected orphan vector.
    rec = mock_memory.save(content="a durable fact", title="Fact", type_="fact")
    md_path = mock_memory.cfg.memory_dir / rec.path
    assert md_path.is_file()
    _insert_orphan_vector(mock_memory, "deadbeefdeadbeef")

    # Act: detect only.
    detected = check_index_health(mock_memory.cfg, mock_memory)

    # Assert: the orphan is flagged and status escalates.
    assert detected["status"] == "issues"
    orphan = detected["checks"]["orphan_vectors"]
    assert orphan["count"] == 1
    assert "deadbeefdeadbeef" in orphan["sample"]
    assert detected["repaired"] == {}

    # Act: repair.
    repaired = check_index_health(mock_memory.cfg, mock_memory, repair=True)

    # Assert: derived orphan removed, canonical .md untouched.
    assert repaired["repaired"].get("orphan_vectors") == 1
    assert md_path.is_file()
    after = check_index_health(mock_memory.cfg, mock_memory)
    assert after["checks"]["orphan_vectors"]["count"] == 0


def test_null_fts_body_detected(mock_memory):
    # Arrange: blank out an indexed FTS body under the memory row.
    rec = mock_memory.save(content="has a real body", title="Body", type_="note")
    with mock_memory.store._tx() as cx:
        cx.execute("UPDATE fts SET body = '' WHERE id = ?", (rec.id,))

    # Act
    result = check_index_health(mock_memory.cfg, mock_memory)

    # Assert
    check = result["checks"]["fts_missing_body"]
    assert check["count"] >= 1
    assert any(entry["id"] == rec.id for entry in check["sample"])


def test_missing_markdown_detected_but_not_repaired(mock_memory):
    # Arrange: delete the canonical .md on disk, leaving the index row.
    rec = mock_memory.save(content="canonical body", title="Canonical", type_="fact")
    md_path = mock_memory.cfg.memory_dir / rec.path
    md_path.unlink()

    # Act: detect.
    detected = check_index_health(mock_memory.cfg, mock_memory)
    assert detected["checks"]["missing_markdown"]["count"] >= 1
    assert any(e["id"] == rec.id for e in detected["checks"]["missing_markdown"]["sample"])

    # Act: repair must NOT delete a memory whose .md is (only) unreachable.
    repaired = check_index_health(mock_memory.cfg, mock_memory, repair=True)
    assert "missing_markdown" not in repaired["repaired"]
    assert mock_memory.store.get(rec.id) is not None


def test_md_divergence_detected_on_hand_edit(mock_memory):
    # Arrange: a hand-edit to the canonical file, leaving the stale index.
    rec = mock_memory.save(content="original body", title="Edited", type_="note")
    md_path = mock_memory.cfg.memory_dir / rec.path
    original = md_path.read_text(encoding="utf-8")
    md_path.write_text(original + "\n\nAppended by hand in Obsidian.\n", encoding="utf-8")

    # Act
    result = check_index_health(mock_memory.cfg, mock_memory)

    # Assert
    check = result["checks"]["md_divergence"]
    assert check["count"] >= 1
    assert any(entry["id"] == rec.id for entry in check["sample"])


def test_never_raises_on_empty_or_malformed_db(tmp_cfg):
    # Arrange: a fake store over an empty in-memory DB (no memo tables).
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row

    class _FakeStore:
        connection = conn
        embedder_model = ""
        dims = tmp_cfg.embedder_dims

        def _vec_table_dims(self, table: str):
            return None

        @contextmanager
        def _tx(self):
            yield conn

    class _FakeMem:
        store = _FakeStore()

    fake = _FakeMem()

    # Act + Assert: must return a dict, never raise, even with repair=True.
    for repair in (False, True):
        result = check_index_health(tmp_cfg, fake, repair=repair)
        assert isinstance(result, dict)
        assert result["status"] in {"ok", "issues", "error"}
        assert set(result) == {"status", "checks", "repaired", "errors"}

    conn.close()
