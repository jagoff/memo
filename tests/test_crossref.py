"""Tests for cross-reference module."""

import pytest

from memo.crossref import (
    Backlink,
    CrossReferenceIndex,
    LinkSuggester,
    LinkSuggestion,
    Wikilink,
)


@pytest.fixture
def crossref_index(tmp_cfg):
    """Fixture providing CrossReferenceIndex instance."""
    index = CrossReferenceIndex(tmp_cfg.crossref_db)
    yield index
    index.close()


@pytest.fixture
def link_suggester(mock_memory):
    """Fixture providing LinkSuggester instance."""
    from memo.crossref import CrossReferenceIndex

    crossref = CrossReferenceIndex(mock_memory.cfg.crossref_db)
    suggester = LinkSuggester(mock_memory, crossref)
    yield suggester
    crossref.close()


def test_crossref_index_init(crossref_index):
    """Test CrossReferenceIndex initialization."""
    assert crossref_index.db_path.is_file()


def test_crossref_index_wikilinks(crossref_index):
    """Test wikilink detection and indexing."""
    content = "See [[memoria-id]] for details and [[other-id|Other Memory]] for more."
    wikilinks = crossref_index.index_wikilinks("test-id", content)

    assert len(wikilinks) == 2
    assert wikilinks[0].target == "memoria-id"
    assert wikilinks[0].alias is None
    assert wikilinks[1].target == "other-id"
    assert wikilinks[1].alias == "Other Memory"


def test_crossref_index_wikilinks_no_links(crossref_index):
    """Test wikilink detection with no links."""
    content = "This has no wikilinks at all."
    wikilinks = crossref_index.index_wikilinks("test-id", content)

    assert len(wikilinks) == 0


def test_crossref_get_backlinks(crossref_index):
    """Test getting backlinks for a memoria."""
    # Index some links
    crossref_index.index_wikilinks("source1", "See [[target]] for details")
    crossref_index.index_wikilinks("source2", "Also [[target]] here")

    backlinks = crossref_index.get_backlinks("target")

    assert len(backlinks) == 2
    assert backlinks[0].target_id == "target"
    assert backlinks[0].source_id in ("source1", "source2")


def test_crossref_get_backlinks_none(crossref_index):
    """Test getting backlinks for a memoria with no backlinks."""
    backlinks = crossref_index.get_backlinks("nonexistent")
    assert backlinks == []


def test_crossref_get_backlinks_source_title_default_empty(crossref_index):
    """Without a title_resolver, source_title stays '' (crossref stores ids only)."""
    crossref_index.index_wikilinks("source1", "See [[target]]")
    backlinks = crossref_index.get_backlinks("target")
    assert backlinks[0].source_title == ""


def test_crossref_get_backlinks_populates_source_title(crossref_index):
    """A batched title_resolver populates source_title from source ids."""
    crossref_index.index_wikilinks("source1", "See [[target]]")
    crossref_index.index_wikilinks("source2", "Also [[target]]")

    titles = {"source1": "First Source", "source2": "Second Source"}
    seen_ids: list[list[str]] = []

    def _resolver(ids):
        seen_ids.append(ids)  # one batched call, not per-row
        return {i: titles.get(i, "") for i in ids}

    backlinks = crossref_index.get_backlinks("target", title_resolver=_resolver)

    assert len(seen_ids) == 1
    got = {b.source_id: b.source_title for b in backlinks}
    assert got == {"source1": "First Source", "source2": "Second Source"}


def test_crossref_get_backlinks_source_titles_via_helper(crossref_index):
    """source_titles_via wraps a single-id get callable into a batched resolver."""
    from types import SimpleNamespace

    from memo.crossref import source_titles_via

    crossref_index.index_wikilinks("source1", "See [[target]]")
    store = {"source1": SimpleNamespace(title="Resolved Title")}
    resolver = source_titles_via(lambda sid: store.get(sid))

    backlinks = crossref_index.get_backlinks("target", title_resolver=resolver)
    assert backlinks[0].source_title == "Resolved Title"


def test_crossref_get_outlinks(crossref_index):
    """Test getting outlinks from a memoria."""
    crossref_index.index_wikilinks("source", "See [[target1]] and [[target2]]")

    outlinks = crossref_index.get_outlinks("source")

    assert len(outlinks) == 2
    assert any(ol.target == "target1" for ol in outlinks)
    assert any(ol.target == "target2" for ol in outlinks)


def test_crossref_remove_memoria(crossref_index):
    """Test removing all links for a memoria."""
    # Index links
    crossref_index.index_wikilinks("source", "See [[target]]")
    crossref_index.index_wikilinks("other", "Also [[target]]")

    # Verify links exist
    assert len(crossref_index.get_backlinks("target")) == 2

    # Remove
    crossref_index.remove_memoria("target")

    # Verify removed
    assert len(crossref_index.get_backlinks("target")) == 0


def test_crossref_persistence(tmp_cfg):
    """Test that crossref index persists across instances."""
    db_path = tmp_cfg.crossref_db

    # Create first instance and index links
    index1 = CrossReferenceIndex(db_path)
    try:
        index1.index_wikilinks("source", "See [[target]]")

        # Create second instance and verify data persisted
        index2 = CrossReferenceIndex(db_path)
        try:
            backlinks = index2.get_backlinks("target")

            assert len(backlinks) == 1
            assert backlinks[0].source_id == "source"
        finally:
            index2.close()
    finally:
        index1.close()


def test_link_suggester_init(link_suggester):
    """Test LinkSuggester initialization."""
    assert link_suggester.memory is not None
    assert link_suggester.crossref is not None


def test_link_suggester_suggest_links(link_suggester, mock_memory):
    """Test link suggestions based on content."""
    # Create test memorias
    mock_memory.save(
        content="Memo about MLX performance",
        title="MLX Performance",
        tags=["mlx"],
    )
    mock_memory.save(
        content="Memo about Qwen models",
        title="Qwen Models",
        tags=["qwen"],
    )

    # Suggest links for similar content
    suggestions = link_suggester.suggest_links(
        content="MLX optimization techniques",
        title="MLX",
        tags=["mlx"],
        limit=5,
    )

    assert isinstance(suggestions, list)
    # May have suggestions or not depending on similarity
    if suggestions:
        assert all(isinstance(s, LinkSuggestion) for s in suggestions)


def test_link_suggester_suggest_links_empty(link_suggester):
    """Test link suggestions with empty corpus."""
    suggestions = link_suggester.suggest_links(
        content="Test content",
        title="Test",
        tags=[],
        limit=5,
    )

    assert suggestions == []


def test_link_suggester_format_wikilink(link_suggester):
    """Test formatting wikilinks."""
    # Without title
    wikilink1 = link_suggester.format_wikilink("abc123")
    assert wikilink1 == "[[abc123]]"

    # With title
    wikilink2 = link_suggester.format_wikilink("abc123", "My Memory")
    assert wikilink2 == "[[abc123|My Memory]]"


def test_wikilink_dataclass():
    """Test Wikilink dataclass structure."""
    link = Wikilink(
        target="memoria-id",
        alias="Display Name",
        position=100,
    )
    assert link.target == "memoria-id"
    assert link.alias == "Display Name"
    assert link.position == 100


def test_backlink_dataclass():
    """Test Backlink dataclass structure."""
    bl = Backlink(
        source_id="source",
        source_title="Source Title",
        target_id="target",
        link_type="wikilink",
        context="See [[target]]",
    )
    assert bl.source_id == "source"
    assert bl.target_id == "target"
    assert bl.link_type == "wikilink"


def test_link_suggestion_dataclass():
    """Test LinkSuggestion dataclass structure."""
    s = LinkSuggestion(
        memory_id="abc123",
        title="Test Title",
        similarity=0.85,
        reason="High semantic similarity",
    )
    assert s.memory_id == "abc123"
    assert s.similarity == 0.85
    assert "similarity" in s.reason.lower()


def test_wikilink_pattern():
    """Test the wikilink regex pattern."""
    from memo.crossref import _WIKILINK_PATTERN

    # Test simple wikilink
    matches = list(_WIKILINK_PATTERN.finditer("See [[target]]"))
    assert len(matches) == 1
    assert matches[0].group(1) == "target"

    # Test wikilink with alias
    matches = list(_WIKILINK_PATTERN.finditer("See [[target|Alias]]"))
    assert len(matches) == 1
    assert matches[0].group(1) == "target|Alias"

    # Test multiple wikilinks
    content = "[[a]] and [[b|B]] and [[c]]"
    matches = list(_WIKILINK_PATTERN.finditer(content))
    assert len(matches) == 3


def test_parse_links_typed_grammar():
    from memo.crossref import parse_links

    content = (
        "Decided to use sqlite-vec.\n"
        "- supersedes [[aaaaaaaa1111000000000000000000ff]]\n"
        "- caused_by [[bbbbbbbb2222000000000000000000ff|the OOM bug]]\n"
        "See also [[cccccccc3333000000000000000000ff]].\n"
    )
    links = parse_links(content)
    by_type = {(link.link_type, link.target) for link in links}
    assert ("supersedes", "aaaaaaaa1111000000000000000000ff") in by_type
    assert ("caused_by", "bbbbbbbb2222000000000000000000ff") in by_type
    assert ("wikilink", "cccccccc3333000000000000000000ff") in by_type
    # typed lines must NOT be double-counted as bare wikilinks
    assert len(links) == 3


def test_index_source_replaces_stale_rows(tmp_path):
    from memo.crossref import CrossReferenceIndex

    idx = CrossReferenceIndex(tmp_path / "crossref.db")
    idx.index_source("src11111", "- supersedes [[tgt11111aaaaaaaaaaaaaaaaaaaaaaaa]]")
    idx.index_source("src11111", "now links elsewhere [[tgt22222bbbbbbbbbbbbbbbbbbbbbbbb]]")
    outlinks = idx.get_outlinks("src11111")
    assert [o.target for o in outlinks] == ["tgt22222bbbbbbbbbbbbbbbbbbbbbbbb"]
    assert outlinks[0].link_type == "wikilink"
    idx.close()


def test_referencing_sources_matches_id_prefix(tmp_path):
    from memo.crossref import CrossReferenceIndex

    idx = CrossReferenceIndex(tmp_path / "crossref.db")
    full_id = "deadbeefcafe000000000000000000ff"
    # hand-authored links usually use a short prefix, not the full 32-char id
    idx.index_source("src11111", f"- supersedes [[{full_id[:12]}]]")
    refs = idx.referencing_sources(full_id)
    assert [r.source_id for r in refs] == ["src11111"]
    assert refs[0].link_type == "supersedes"
    idx.close()


def test_save_indexes_typed_links_when_flag_on(mock_memory, monkeypatch):
    monkeypatch.setenv("MEMO_CROSSREF_INDEX", "1")
    target = mock_memory.save(content="target body", title="Target")
    src = mock_memory.save(content=f"decision text\n- supersedes [[{target.id}]]", title="Source")
    refs = mock_memory.crossref.referencing_sources(target.id)
    assert [r.source_id for r in refs] == [src.id]
    assert refs[0].link_type == "supersedes"


def test_save_does_not_index_links_by_default(mock_memory):
    target = mock_memory.save(content="t", title="T-default")
    mock_memory.save(content=f"[[{target.id}]]", title="S-default")
    assert mock_memory.crossref.referencing_sources(target.id) == []


def test_update_reindexes_links_and_delete_removes_them(mock_memory, monkeypatch):
    monkeypatch.setenv("MEMO_CROSSREF_INDEX", "1")
    t1 = mock_memory.save(content="one", title="T1")
    t2 = mock_memory.save(content="two", title="T2")
    src = mock_memory.save(content=f"- relates_to [[{t1.id}]]", title="Src")
    mock_memory.update(src.id, content=f"- relates_to [[{t2.id}]]")
    assert mock_memory.crossref.referencing_sources(t1.id) == []
    assert [r.source_id for r in mock_memory.crossref.referencing_sources(t2.id)] == [src.id]
    mock_memory.delete(src.id)
    assert mock_memory.crossref.referencing_sources(t2.id) == []


def test_memo_delete_warns_on_inbound_links(mock_memory, monkeypatch):
    monkeypatch.setenv("MEMO_CROSSREF_INDEX", "1")
    from memo.server_core_records import register

    target = mock_memory.save(content="t", title="Cascade target")
    src = mock_memory.save(content=f"- supersedes [[{target.id}]]", title="Cascade source")

    tools: dict = {}

    class _Srv:
        def tool(self, *a, **k):
            def wrap(fn):
                tools[fn.__name__] = fn
                return fn

            return wrap

    register(_Srv(), mock_memory)
    import asyncio

    out = asyncio.run(tools["memo_delete"](id=target.id))
    assert out["deleted"] is True
    assert out["referenced_by"] == [src.id]
    assert "dangle" in out["cascade_warning"]
