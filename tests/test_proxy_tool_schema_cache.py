"""`memo.proxy.tool_schema_cache` — the disk-backed cache that lets
`memo_tool_docs` hydrate a schema for a tool memo does not own.

`ToolSchemas` (`memo.proxy.transforms.toolschemas`) sees every tool
definition on the wire before it prunes any of them; this module is where
a pruned tool's schema is remembered so `memo_tool_docs` (a separate
process — the MCP server, not the proxy) can serve it back by name later.
Both sides only ever touch this file through the functions here.
"""

from __future__ import annotations

import json

from memo.proxy.tool_schema_cache import cache_path, lookup, remember


def test_lookup_on_a_cold_start_returns_none(tmp_path):
    assert lookup(tmp_path, "mcp__octocode__localSearchCode") is None


def test_remember_then_lookup_round_trips_a_schema(tmp_path):
    remember(
        tmp_path,
        [
            {
                "name": "mcp__octocode__localSearchCode",
                "description": "Search code locally.",
                "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}},
            }
        ],
    )
    entry = lookup(tmp_path, "mcp__octocode__localSearchCode")
    assert entry is not None
    assert entry["description"] == "Search code locally."
    assert entry["input_schema"]["properties"]["query"]["type"] == "string"


def test_remember_skips_entries_without_a_usable_name(tmp_path):
    remember(tmp_path, [{"description": "no name"}, {"name": 5}, "not-a-dict"])
    assert not cache_path(tmp_path).exists()


def test_remember_merges_across_calls_rather_than_overwriting(tmp_path):
    remember(tmp_path, [{"name": "toolA", "description": "a", "input_schema": {}}])
    remember(tmp_path, [{"name": "toolB", "description": "b", "input_schema": {}}])
    assert lookup(tmp_path, "toolA") is not None
    assert lookup(tmp_path, "toolB") is not None


def test_remember_updates_an_existing_entry_with_the_latest_schema(tmp_path):
    remember(tmp_path, [{"name": "toolA", "description": "old", "input_schema": {}}])
    remember(tmp_path, [{"name": "toolA", "description": "new", "input_schema": {}}])
    entry = lookup(tmp_path, "toolA")
    assert entry is not None
    assert entry["description"] == "new"


def test_remember_never_raises_on_an_unwritable_state_dir(tmp_path):
    blocker = tmp_path / "proxy"
    blocker.write_text("not a directory", encoding="utf-8")
    remember(tmp_path, [{"name": "toolA", "description": "a", "input_schema": {}}])  # no raise


def test_lookup_survives_corrupt_json(tmp_path):
    path = cache_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text("{not json", encoding="utf-8")
    assert lookup(tmp_path, "toolA") is None


def test_remember_with_no_valid_entries_is_a_noop(tmp_path):
    remember(tmp_path, [])
    assert not cache_path(tmp_path).exists()


def test_cache_file_is_valid_json_with_the_expected_schema_tag(tmp_path):
    remember(tmp_path, [{"name": "toolA", "description": "a", "input_schema": {}}])
    data = json.loads(cache_path(tmp_path).read_text(encoding="utf-8"))
    assert data["schema"] == "memo.proxy.tool_schema_cache.v1"
    assert "toolA" in data["tools"]
