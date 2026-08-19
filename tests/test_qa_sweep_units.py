"""Unit coverage for the small helpers the 4.13.0 QA sweep introduced.

Their behavioural tests live next to the surfaces they fix (delete rollback,
budget payloads, gc guards, …); these pin the helpers directly, including the
branches that only run on a machine with tantivy or MLX installed and would
otherwise never be exercised in CI.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.resource_hygiene


def test_tantivy_dirty_marker_survives_the_process(tmp_path) -> None:
    """The in-memory unhealthy flag dies with the process; the marker must not."""
    from memo.store import VecStore

    store = VecStore(tmp_path / "vec.db", dims=4)
    try:
        marker = store._tantivy_dirty_marker()
        assert marker.name == ".dirty"
        assert not marker.exists()

        store._mark_tantivy_unhealthy()

        assert store._tantivy_healthy is False
        assert marker.exists(), "a failed tantivy write must leave an on-disk marker"
    finally:
        store.close()


def test_mark_tantivy_unhealthy_tolerates_an_unwritable_dir(tmp_path, monkeypatch) -> None:
    """Marking must never raise into the write path it is reporting on."""
    from memo.store import VecStore

    store = VecStore(tmp_path / "vec.db", dims=4)
    try:

        def _boom(*_a, **_k):
            raise OSError("read-only filesystem")

        monkeypatch.setattr(Path, "mkdir", _boom)
        store._mark_tantivy_unhealthy()
        assert store._tantivy_healthy is False
    finally:
        store.close()


def test_mark_partial_flags_only_a_failed_reindex() -> None:
    from memo.server_core_records import _mark_partial

    assert _mark_partial({"checked": 3, "errors": 0}) == {"checked": 3, "errors": 0}
    assert _mark_partial({"checked": 3, "errors": 2})["partial"] == 1


def test_print_reindex_errors_is_silent_on_a_clean_run(capsys) -> None:
    from memo.cli_memory import _print_reindex_errors

    _print_reindex_errors({"checked": 5, "errors": 0})
    assert capsys.readouterr().out == ""

    _print_reindex_errors({"checked": 5, "errors": 3})
    assert "3 file(s) failed to index" in capsys.readouterr().out


def test_warn_reindex_errors_only_warns_when_something_failed(caplog) -> None:
    import logging

    from memo.memory.maintain_ops import _warn_reindex_errors

    with caplog.at_level(logging.WARNING, logger="memo.memory.record"):
        _warn_reindex_errors(0, 12)
    assert caplog.text == ""

    with caplog.at_level(logging.WARNING, logger="memo.memory.record"):
        _warn_reindex_errors(4, 12)
    assert "4 of 12" in caplog.text


def test_rag_cache_clear_drops_every_entry() -> None:
    from memo.rag_cache import RagContextCache

    cache = RagContextCache(ttl_s=60.0, max_entries=4)
    cache.put("k", "v", corpus_version="v1", now=0.0)
    assert cache.get("k", corpus_version="v1", now=1.0) == "v"

    cache.clear()
    assert cache.get("k", corpus_version="v1", now=1.0) is None


def test_rag_cache_evicts_the_entry_closest_to_expiry() -> None:
    from memo.rag_cache import RagContextCache

    cache = RagContextCache(ttl_s=60.0, max_entries=2)
    cache.put("old", 1, corpus_version="v1", now=0.0)
    cache.put("new", 2, corpus_version="v1", now=10.0)
    cache.put("newest", 3, corpus_version="v1", now=20.0)

    assert cache.get("old", corpus_version="v1", now=21.0) is None
    assert cache.get("newest", corpus_version="v1", now=21.0) == 3


def test_turn_context_cache_is_inert_when_its_ttl_is_zero() -> None:
    from memo.context_cache import TurnContextCache

    cache = TurnContextCache(max_size=8, ttl_s=0)
    cache.set("k", {"a": 1})
    assert cache.get("k") is None


def test_latest_remote_tag_refuses_an_unsafe_repo_url() -> None:
    """The refusal must happen BEFORE git is spawned."""
    from memo.runtime import autoupdate

    assert autoupdate.latest_remote_tag("ext::sh -c 'touch /tmp/pwn'") is None
    assert autoupdate.latest_remote_tag("--upload-pack=touch /tmp/pwn") is None


def test_save_rejects_an_unknown_type_as_a_validation_error(mem_with_stub) -> None:
    """A caller-input error must be a ValidationError, not a bare ValueError:
    the MCP write coordinator passes the former through with its message."""
    from memo.errors import ValidationError

    with pytest.raises(ValidationError):
        mem_with_stub.save(content="cuerpo", title="T", type_="not-a-type")
