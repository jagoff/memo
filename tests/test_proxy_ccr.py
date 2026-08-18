from memo.proxy import ccr


def test_stash_then_recover_roundtrips(tmp_path):
    key = ccr.stash(tmp_path, "the original content")
    assert ccr.recover(tmp_path, key) == "the original content"


def test_stash_is_content_addressed(tmp_path):
    assert ccr.stash(tmp_path, "same") == ccr.stash(tmp_path, "same")
    assert ccr.stash(tmp_path, "same") != ccr.stash(tmp_path, "other")


def test_recover_returns_none_for_unknown_key(tmp_path):
    assert ccr.recover(tmp_path, "a" * 64) is None


def test_recover_never_touches_the_filesystem_for_a_non_hex_key(tmp_path):
    assert ccr.recover(tmp_path, "../../etc/passwd") is None


def test_marker_names_the_key_and_what_was_dropped():
    m = ccr.marker("abc123", kept_chars=100, dropped_chars=900)
    assert "abc123" in m
    assert "900" in m
    assert "memo_retrieve" in m


def test_stash_returns_empty_key_when_the_cache_is_unwritable(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise OSError("read-only filesystem")

    monkeypatch.setattr("memo.store.crush_cache.CrushCache.cache", boom)
    assert ccr.stash(tmp_path, "content") == ""
