import asyncio
import re

import pytest

from memo.config import Config
from memo.memory import Memory
from memo.proxy import ccr
from memo.server import build_server


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
    assert "memo_crush_retrieve" in m
    assert "Full original" in m


@pytest.mark.parametrize("profile", ["agent", "core", "full"])
def test_marker_names_a_tool_that_is_actually_registered(tmp_path, monkeypatch, profile):
    """Defect 1: marker() tells the model to call a specific MCP tool by
    name. A name nobody registers turns "nothing is cut without a recovery
    path" into a lie -- the model gets an unknown-tool error and the
    original is gone to it. Extract the tool name straight out of marker()'s
    own rendered text (not a hardcoded expectation) and confirm the real
    server registers it on every profile a cut could reach -- the proxy has
    no idea which profile is live when it stamps the marker."""
    m = ccr.marker("abc123", kept_chars=100, dropped_chars=900)
    match = re.search(r"\b(memo_[a-z_]+)\(", m)
    assert match, f"no tool call found in marker text: {m!r}"
    tool_name = match.group(1)

    monkeypatch.setenv("MEMO_MCP_PROFILE", profile)
    monkeypatch.delenv("MEMO_MCP_SLIM", raising=False)
    monkeypatch.setattr(
        "memo.embedder.MLXEmbedder.embed",
        lambda self, inputs: [[1.0, 0.0, 0.0, 0.0] for _ in inputs],
    )
    cfg = Config(
        data_dir=tmp_path / "data",
        state_dir=tmp_path / "state",
        vault_path=tmp_path / "vault",
        embedder_dims=4,
    )
    mem = Memory(cfg)
    try:
        server = build_server(memory=mem)
        tool = asyncio.run(server.get_tool(tool_name))
        assert tool is not None, (
            f"{tool_name!r} (from marker text) is not registered on the {profile!r} profile"
        )
    finally:
        mem.close()


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
    assert "memo_crush_retrieve" in m
    assert "Full original" not in m
    assert "memo-crush" in m


def test_marker_flags_its_own_nested_reference_instead_of_claiming_full_original():
    """Critical 3 (fix round 2): when `stashed` already carries a PRIOR
    `marker()` call's own rendered `[memo: N chars elided, M kept. ...]`
    text (not JsonCrush's `<<memo-crush:` shape), `key` STILL only recovers
    an intermediate -- e.g. StructMap's signature map for a large file,
    which ToolResults' own 4000-char threshold still catches and cuts a
    second time. The wording must say so, same as the JsonCrush case."""
    already_marked = (
        "def big():\n    ...\n"
        "\n[memo: 3000 chars elided, 500 kept. Full original: "
        'memo_crush_retrieve(hash_marker="' + "a" * 64 + '")]'
    )
    m = ccr.marker("def456", kept_chars=100, dropped_chars=400, stashed=already_marked)
    assert "def456" in m
    assert "memo_crush_retrieve" in m
    assert "Full original" not in m


def test_marker_does_not_false_positive_on_its_own_template_source():
    """This module's own SOURCE CODE (e.g. read/compressed like any other
    file) contains the same literal `memo_crush_retrieve(hash_marker="`
    fragment as an f-string template with `{key}` placeholders, never actual
    digits after "chars elided"/"kept." -- the nested-reference check must
    not mistake that template text for a REAL prior marker, or it would
    falsely claim "not the full original" about content that actually IS the
    full original."""
    template_source = (
        'def marker(key, *, kept_chars, dropped_chars, stashed=""):\n'
        "    return f'[memo: {dropped_chars} chars elided, {kept_chars} kept. "
        'Full original: memo_crush_retrieve(hash_marker="{key}")]\'\n'
    )
    m = ccr.marker("xyz789", kept_chars=10, dropped_chars=5, stashed=template_source)
    assert "Full original" in m


def test_stash_returns_empty_key_when_the_cache_is_unwritable(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise OSError("read-only filesystem")

    monkeypatch.setattr("memo.store.crush_cache.CrushCache.cache", boom)
    assert ccr.stash(tmp_path, "content") == ""


def test_stash_returns_empty_key_for_content_that_cannot_be_encoded(tmp_path):
    # A lone surrogate is reachable: json.loads('"\\ud800"') produces one.
    assert ccr.stash(tmp_path, "hello \ud800 world") == ""
