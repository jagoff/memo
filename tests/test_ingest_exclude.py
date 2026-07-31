"""Tests for the vault-ingest tombstone store (ported from synapse)."""

from memo.ingest_exclude import IngestExcludeStore


def test_add_globs_remove_roundtrip(tmp_path):
    store = IngestExcludeStore(state_dir=tmp_path)
    assert store.globs("notes") == []
    assert store.add(vault_label="notes", rel_path="a/b.md") is True
    assert store.add(vault_label="notes", rel_path="a/b.md") is False  # idempotent
    store.add(vault_label="notes", rel_path="c.md")
    assert store.globs("notes") == ["a/b.md", "c.md"]
    assert store.all_labels() == ["notes"]
    assert store.remove(vault_label="notes", rel_path="a/b.md") is True
    assert store.globs("notes") == ["c.md"]
    assert store.remove(vault_label="notes", rel_path="zzz") is False


def test_label_sanitized(tmp_path):
    store = IngestExcludeStore(state_dir=tmp_path)
    store.add(vault_label="Mi Vault!", rel_path="x.md")
    assert (tmp_path / "ingest_excludes" / "mi-vault.txt").exists()


def test_globs_skips_comments_and_dedupes(tmp_path):
    store = IngestExcludeStore(state_dir=tmp_path)
    path = tmp_path / "ingest_excludes" / "notes.txt"
    path.parent.mkdir(parents=True)
    path.write_text("# comment\na.md\n\na.md\nb.md\n", encoding="utf-8")
    assert store.globs("notes") == ["a.md", "b.md"]


def test_add_empty_raises(tmp_path):
    store = IngestExcludeStore(state_dir=tmp_path)
    try:
        store.add(vault_label="notes", rel_path="  ")
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")
