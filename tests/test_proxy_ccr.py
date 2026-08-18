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
    assert "Full original" in m


def test_marker_flags_a_nested_crush_reference_instead_of_claiming_full_original():
    """Fix round 1 (task 11): when `stashed` (the content actually stored
    under `key`) already carries an earlier crush's `<<memo-crush:...>>`
    reference, `key` recovers an intermediate, not the true original --
    the wording must say so instead of the (false) "Full original" claim."""
    m = ccr.marker(
        "abc123",
        kept_chars=100,
        dropped_chars=900,
        stashed='[{"id": 1}, {"_compressed": "5 rows offloaded -- ask memo '
        'retrieve <<memo-crush:deadbeef>>"}]',
    )
    assert "abc123" in m
    assert "memo_retrieve" in m
    assert "Full original" not in m
    assert "memo-crush" in m


def test_stash_returns_empty_key_when_the_cache_is_unwritable(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise OSError("read-only filesystem")

    monkeypatch.setattr("memo.store.crush_cache.CrushCache.cache", boom)
    assert ccr.stash(tmp_path, "content") == ""


def test_stash_returns_empty_key_for_content_that_cannot_be_encoded(tmp_path):
    # A lone surrogate is reachable: json.loads('"\\ud800"') produces one.
    assert ccr.stash(tmp_path, "hello \ud800 world") == ""
