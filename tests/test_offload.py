"""memo.offload — content-addressed payload store + deterministic synopsis."""

from __future__ import annotations


def test_synopsize_json_object():
    from memo.offload import synopsize

    kind, syn = synopsize('{"alpha": 1, "beta": [1, 2], "gamma": "x"}')
    assert kind == "json"
    assert "3 keys" in syn and "alpha" in syn


def test_synopsize_csv():
    from memo.offload import synopsize

    rows = "name,age,city\n" + "\n".join(f"p{i},3{i},bsas" for i in range(5))
    kind, syn = synopsize(rows)
    assert kind == "csv"
    assert "5 rows" in syn and "name,age,city" in syn


def test_synopsize_code():
    from memo.offload import synopsize

    code = "import os\nimport sys\n\ndef main():\n    pass\n\nclass App:\n    pass\n"
    kind, syn = synopsize(code)
    assert kind == "code"
    assert "def main" in syn


def test_synopsize_text_fallback_is_bounded():
    from memo.offload import synopsize

    kind, syn = synopsize("just prose. " * 500)
    assert kind == "text"
    assert len(syn) <= 600


def test_offload_saves_reference_tier_and_dedups(mock_memory):
    from memo.offload import offload

    payload = '{"result": "big tool output", "items": [1, 2, 3]}'
    out1 = offload(mock_memory, payload)
    assert out1["deduplicated"] is False
    rec = mock_memory.get(out1["id"])
    assert rec.type == "reference"  # excluded from auto-recall by tier
    assert "offload" in rec.tags
    out2 = offload(mock_memory, payload)
    assert out2["deduplicated"] is True
    assert out2["id"] == out1["id"]


def test_offload_rejects_empty_and_oversize(mock_memory):
    from memo.offload import offload

    assert offload(mock_memory, "  ")["error"] == "empty"
    big = "x" * (mock_memory.cfg.max_content_chars + 1)
    assert offload(mock_memory, big)["error"] == "too_large"


def test_memo_offload_tool_registered_and_roundtrips(mock_memory):
    import asyncio

    from memo.server import build_server

    server = build_server(memory=mock_memory)
    tool = asyncio.run(server.get_tool("memo_offload"))
    assert tool is not None, "memo_offload must be registered on the core surface"
    out = tool.fn(content='{"a": 1}')
    assert out["kind"] == "json"
    got = asyncio.run(server.get_tool("memo_get")).fn(id=out["id"])
    # Task 7's offload() prefixes a self-describing `# {label}` heading so the
    # reference-tier noise gate accepts the payload; the raw payload round-trips
    # verbatim as the body tail (drill-down retrievability is the contract).
    assert got["body"].endswith('{"a": 1}')
