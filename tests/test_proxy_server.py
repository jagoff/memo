import json
import logging
import re
from pathlib import Path

import pytest

from memo.proxy.plan import Context
from memo.proxy.server import forward_headers, rewrite_body


class _Clear:
    name = "clear"
    zone = "live"

    def enabled(self):
        return True

    def apply(self, zones, ctx):
        zones.live_messages.clear()
        return 7


class _Boom:
    name = "boom"
    zone = "live"

    def enabled(self):
        return True

    def apply(self, zones, ctx):
        raise RuntimeError("nope")


class _PartialMutateThenBoom:
    """Mutates a live-zone dict (which aliases the caller's message dict) and
    then raises — the exact shape that would leak into the wire if
    `rewrite_body` ever re-serialized `payload` after a failed plan instead of
    returning the pristine original bytes."""

    name = "partial"
    zone = "live"

    def enabled(self):
        return True

    def apply(self, zones, ctx):
        zones.live_messages[0]["content"] = "MUTATED"
        raise RuntimeError("boom after mutating")


def _ctx(tmp_path):
    return Context(state_dir=tmp_path, session_key="s", project=None)


# ---------------------------------------------------------------------------
# forward_headers
# ---------------------------------------------------------------------------


def test_anthropic_beta_is_forwarded_verbatim():
    out = forward_headers({"anthropic-beta": "oauth-2025-04-20,foo", "host": "x"})
    assert out["anthropic-beta"] == "oauth-2025-04-20,foo"


def test_auth_headers_are_forwarded():
    out = forward_headers({"authorization": "Bearer tok", "x-api-key": "k"})
    assert out["authorization"] == "Bearer tok"
    assert out["x-api-key"] == "k"


def test_hop_by_hop_headers_are_dropped():
    out = forward_headers(
        {
            "host": "127.0.0.1:8768",
            "content-length": "12",
            "connection": "keep-alive",
            "x-api-key": "k",
        }
    )
    assert "host" not in out
    assert "content-length" not in out
    assert "connection" not in out


def test_no_header_value_reaches_the_log(caplog):
    with caplog.at_level("DEBUG"):
        forward_headers({"x-api-key": "SUPERSECRET", "authorization": "Bearer SECRET2"})
    assert "SUPERSECRET" not in caplog.text
    assert "SECRET2" not in caplog.text


# ---------------------------------------------------------------------------
# rewrite_body
# ---------------------------------------------------------------------------


def test_rewrite_applies_a_transform(tmp_path):
    raw = json.dumps({"messages": [{"role": "user", "content": "hi"}]}).encode()
    out, plan = rewrite_body(raw, _ctx(tmp_path), [_Clear()])
    assert json.loads(out)["messages"] == []
    assert plan.applied == ["clear"]


def test_a_failing_transform_leaves_the_body_untouched(tmp_path):
    raw = json.dumps({"messages": [{"role": "user", "content": "hi"}]}).encode()
    out, plan = rewrite_body(raw, _ctx(tmp_path), [_Boom()])
    assert json.loads(out)["messages"] == [{"role": "user", "content": "hi"}]
    assert plan.applied == []


def test_a_non_json_body_is_forwarded_byte_identical(tmp_path):
    raw = b"not json at all"
    out, plan = rewrite_body(raw, _ctx(tmp_path), [_Clear()])
    assert out == raw
    assert plan.applied == []


def test_a_transform_that_mutates_then_fails_still_forwards_pristine_bytes(tmp_path):
    """Deviation 4: zones alias the caller's message dicts, so a failing
    transform can mutate them in place before raising. `rewrite_body` must
    forward the ORIGINAL bytes on a failed plan, not a re-serialization of the
    now-mutated payload — otherwise the aliasing risk noted in the design
    ruling would leak a partial edit onto the wire."""
    raw = json.dumps({"messages": [{"role": "user", "content": "hi"}]}).encode()
    out, plan = rewrite_body(raw, _ctx(tmp_path), [_PartialMutateThenBoom()])
    assert out == raw
    assert json.loads(out)["messages"][0]["content"] == "hi"
    assert plan.applied == []


# ---------------------------------------------------------------------------
# Runtime prefix-stability check (design Section 2: a mismatch within one
# session is "a test failure and a logged runtime warning"). Fix round 1 on
# Task 9 found this half only existed as a test, never wired into the actual
# request path — so a real instability would have failed silently in prod.
#
# Fix round 2 corrected WHAT is compared: the check tracks
# `stable_head_fingerprint` (system + tools only), not the stricter
# `prefix_fingerprint` (which also hashes `frozen_messages`). Provider
# caching matches the longest cached prefix, so a growing conversation is
# cache-friendly and must NOT trip the warning — only a REWRITE of the
# already-cached system/tools is a real economic problem. See
# `test_prefix_drift_warning_is_silent_when_only_history_grows` (must stay
# silent) and `test_prefix_drift_warning_fires_when_tools_change_mid_session`
# (must still fire) below.
# ---------------------------------------------------------------------------


def test_prefix_drift_within_a_session_logs_a_warning(tmp_path, caplog):
    """A cached-prefix transform that reshuffles the tools list mid-session
    (the exact Critical bug this fix round addresses) must be caught by a
    logged warning on the request path, not silently swallowed."""
    ctx = Context(state_dir=tmp_path, session_key="drift-sess", project=None)
    raw1 = json.dumps(
        {"tools": [{"name": "a"}], "messages": [{"role": "user", "content": "hi"}]}
    ).encode()
    raw2 = json.dumps(
        {"tools": [{"name": "b"}], "messages": [{"role": "user", "content": "hi"}]}
    ).encode()

    with caplog.at_level(logging.WARNING, logger="memo.proxy.server"):
        rewrite_body(raw1, ctx, transforms=[])
        rewrite_body(raw2, ctx, transforms=[])

    assert any("prefix" in r.message.lower() for r in caplog.records)


def test_stable_prefix_across_turns_does_not_warn(tmp_path, caplog):
    """The complementary negative case: an unchanging prefix across repeated
    requests in the same session must NOT spam a warning."""
    ctx = Context(state_dir=tmp_path, session_key="stable-sess", project=None)
    raw = json.dumps(
        {"tools": [{"name": "a"}], "messages": [{"role": "user", "content": "hi"}]}
    ).encode()

    with caplog.at_level(logging.WARNING, logger="memo.proxy.server"):
        rewrite_body(raw, ctx, transforms=[])
        rewrite_body(raw, ctx, transforms=[])
        rewrite_body(raw, ctx, transforms=[])

    assert not any("prefix" in r.message.lower() for r in caplog.records)


def test_prefix_drift_warning_is_silent_when_only_history_grows(tmp_path, caplog):
    """Fix round 2: provider prompt caching matches the LONGEST CACHED
    PREFIX, so APPENDING new turns is cache-friendly — earlier cached bytes
    stay valid. What actually breaks a cache hit is REWRITING content that
    was already cached (system / tools). A growing conversation changes
    `frozen_messages` turn over turn — that's normal, expected, and must NOT
    trip the warning, or a warning that always fires on ordinary traffic
    trains everyone to ignore it (exactly how the round-1 Critical stayed
    invisible). `system`/`tools` are unchanged across these two calls; only
    the message history grows (3 messages -> 5), so `frozen_messages` grows
    too (live_turns=2 default: cut 1 -> cut 3) — this must stay silent."""
    ctx = Context(state_dir=tmp_path, session_key="growth-sess", project=None)
    tools = [{"name": "a"}]
    raw1 = json.dumps(
        {
            "tools": tools,
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
                {"role": "user", "content": "how are you"},
            ],
        }
    ).encode()
    raw2 = json.dumps(
        {
            "tools": tools,
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
                {"role": "user", "content": "how are you"},
                {"role": "assistant", "content": "good, thanks"},
                {"role": "user", "content": "great, tell me more"},
            ],
        }
    ).encode()

    with caplog.at_level(logging.WARNING, logger="memo.proxy.server"):
        rewrite_body(raw1, ctx, transforms=[])
        rewrite_body(raw2, ctx, transforms=[])

    assert not any("prefix" in r.message.lower() for r in caplog.records)


def test_prefix_drift_warning_fires_when_tools_change_mid_session(tmp_path, caplog):
    """The complementary positive case, restated for the split fingerprint:
    a real change to the cached head (tools, here) — not just history growth
    — must still fire the warning. Same message count both turns, so
    `frozen_messages` is identical (empty, under live_turns=2); only `tools`
    differs."""
    ctx = Context(state_dir=tmp_path, session_key="tools-change-sess", project=None)
    raw1 = json.dumps(
        {"tools": [{"name": "a"}], "messages": [{"role": "user", "content": "hi"}]}
    ).encode()
    raw2 = json.dumps(
        {"tools": [{"name": "a"}, {"name": "b"}], "messages": [{"role": "user", "content": "hi"}]}
    ).encode()

    with caplog.at_level(logging.WARNING, logger="memo.proxy.server"):
        rewrite_body(raw1, ctx, transforms=[])
        rewrite_body(raw2, ctx, transforms=[])

    assert any("prefix" in r.message.lower() for r in caplog.records)


def test_prefix_drift_check_is_fail_open(tmp_path, monkeypatch):
    """A failure inside the drift check itself (fingerprinting blows up) must
    not raise into rewrite_body's caller — it's a diagnostic side effect, not
    part of the request's success path."""
    import memo.proxy.server as server_mod

    def _boom(*_args, **_kwargs):
        raise RuntimeError("fingerprint exploded")

    monkeypatch.setattr(server_mod, "stable_head_fingerprint", _boom)
    ctx = Context(state_dir=tmp_path, session_key="boom-sess", project=None)
    raw = json.dumps({"messages": [{"role": "user", "content": "hi"}]}).encode()

    out, plan = rewrite_body(raw, ctx, [_Clear()])

    assert json.loads(out)["messages"] == []
    assert plan.applied == ["clear"]


# ---------------------------------------------------------------------------
# REGISTRY does not exist; the default comes from build_registry()
# ---------------------------------------------------------------------------


def test_plan_module_exposes_no_registry_constant():
    import memo.proxy.plan as plan_mod

    assert not hasattr(plan_mod, "REGISTRY")


def test_build_registry_includes_toolschemas():
    """Task 9 registers the first real transform, Task 10 the second, Task 11
    the third; the registry is no longer empty (see the superseded test this
    replaced, `..._returns_an_empty_list_today`). Task 12 adds StructMap and
    Delta, positioned BEFORE JsonCrush/ToolResults -- see the ordering
    rationale in `registry.build_registry`'s docstring. Task 13 adds Pixel
    LAST of all -- it is the most generic, least-proven transform in the
    plan and must only ever see what nothing text-based upstream could
    shrink further (same docstring)."""
    from memo.proxy.registry import build_registry
    from memo.proxy.transforms.delta import Delta
    from memo.proxy.transforms.jsoncrush import JsonCrush
    from memo.proxy.transforms.pixel import Pixel
    from memo.proxy.transforms.structmap import StructMap
    from memo.proxy.transforms.toolresults import ToolResults
    from memo.proxy.transforms.toolschemas import ToolSchemas

    registry = build_registry()
    assert len(registry) == 6
    assert isinstance(registry[0], ToolSchemas)
    assert isinstance(registry[1], StructMap)
    assert isinstance(registry[2], Delta)
    assert isinstance(registry[3], JsonCrush)
    assert isinstance(registry[4], ToolResults)
    assert isinstance(registry[5], Pixel)


def test_rewrite_body_with_no_explicit_transforms_runs_the_real_registry(tmp_path):
    """No `tools` key and no tool_result blocks on this payload means none of
    the six registered transforms has anything to rewrite, so the body is
    unchanged — but the default path (transforms=None) now runs the real
    registry's transforms, not an empty one, so all six still report
    applied."""
    raw = json.dumps({"messages": [{"role": "user", "content": "hi"}]}).encode()
    out, plan = rewrite_body(raw, _ctx(tmp_path))
    assert out == raw
    assert plan.applied == [
        "toolschemas",
        "structmap",
        "delta",
        "jsoncrush",
        "toolresults",
        "pixel",
    ]


def test_rewrite_body_default_transforms_come_from_build_registry(tmp_path, monkeypatch):
    """Proves the wiring, not just the empty-list coincidence: patch
    build_registry() to return a real transform and confirm rewrite_body's
    default path (transforms=None) picks it up."""
    import memo.proxy.server as server_mod

    monkeypatch.setattr(server_mod, "build_registry", lambda: [_Clear()])
    raw = json.dumps({"messages": [{"role": "user", "content": "hi"}]}).encode()
    out, plan = rewrite_body(raw, _ctx(tmp_path))
    assert json.loads(out)["messages"] == []
    assert plan.applied == ["clear"]


def test_marker_never_claims_full_original_when_a_crush_reference_is_nested(tmp_path, monkeypatch):
    """Fix round 1 regression (task 11): JsonCrush runs before ToolResults in
    the real registry, so for a large JSON tool_result -- the common case
    this whole task exists for -- ToolResults' own recovery marker wraps
    text JsonCrush ALREADY crushed. `ccr.stash` in that path stores the
    crushed intermediate, not the true original, so a marker literally
    saying "Full original: memo_retrieve(key=...)" is false: the key
    recovers text that itself still carries JsonCrush's own
    `<<memo-crush:HASH>>` reference one hop further out.

    Isolated from the developer's real Markdown config per the process note
    on this fix round -- MEMO_CONFIG_DIR is pinned explicitly even though
    conftest.py already defaults it, since this test exercises the exact
    config_md/tuned_overlay read path the capture-flag scoping in
    JsonCrush consults.

    Pixel (task 13) is disabled here: it runs last in the real registry and,
    after its fix-round-1 correction, DOES legitimately fire on a dense
    single-line JSON-crushed block like this test's fixture -- correctly
    converting it to an image block whose own marker also detects the
    nested crush reference (the same `ccr.marker` nested-detection this
    test checks, one hop further out). That is desired behavior, but it is
    Pixel's behavior, not the JsonCrush-before-ToolResults interaction this
    test exists to isolate, so scoping it out keeps this test's assertions
    about `final_text` (a plain string) meaningful regardless of Pixel's own
    profitability threshold.
    """
    monkeypatch.setenv("MEMO_CONFIG_DIR", str(tmp_path / "config-home"))
    monkeypatch.delenv("MEMO_CRUSHER_ENABLED", raising=False)
    monkeypatch.setenv("MEMO_PROXY_PIXEL", "0")

    big = json.dumps([{"id": i, "text": "row " * 20} for i in range(400)])
    raw = json.dumps(
        {"messages": [{"role": "user", "content": [{"type": "tool_result", "content": big}]}]}
    ).encode()

    ctx = _ctx(tmp_path)
    out, plan = rewrite_body(raw, ctx)

    assert "jsoncrush" in plan.applied
    assert "toolresults" in plan.applied

    final_text = json.loads(out)["messages"][0]["content"][0]["content"]

    match = re.search(r'memo_retrieve\(key="([0-9a-f]+)"\)', final_text)
    assert match, f"expected a ToolResults recovery marker in: {final_text!r}"

    from memo.proxy import ccr

    recovered = ccr.recover(ctx.state_dir, match.group(1))
    assert recovered is not None
    assert "<<memo-crush:" in recovered, (
        "setup check: the content ToolResults stashed should still carry "
        "JsonCrush's nested reference, or this test isn't reproducing the bug"
    )
    assert "Full original" not in final_text, (
        "the marker must not claim to hold the full original when what its "
        "key actually recovers is itself only an intermediate with a "
        "further memo-crush reference nested inside it"
    )


def test_marker_never_claims_full_original_when_a_structmap_reference_is_nested(
    tmp_path, monkeypatch
):
    """Task 12, fix round 2, Critical 3: the exact same false-claim bug as
    the JsonCrush case above, one hop earlier in the pipeline. StructMap
    runs before ToolResults, and its own signature map for a large real
    file can still clear ToolResults' 4000-char fallback threshold --
    measured against this repo's own `src/memo/memory/search_ops.py`
    (80KB raw source -> 8855-char signature map, still 2x over the
    threshold). ToolResults then cuts THAT signature map a second time and
    stashes it -- so `key` recovers StructMap's intermediate, which itself
    still carries StructMap's own `[memo: N chars elided, M kept. ...]`
    marker, not the true original file. A marker saying "Full original" in
    that case is false for the identical reason the JsonCrush case above
    is false -- just via `ccr.marker`'s OWN rendered shape instead of
    JsonCrush's `<<memo-crush:` shape, so the SAME nested-reference check
    must catch it too.

    Config isolation for the same reason as the JsonCrush test above: this
    runs the real registry end to end. Pixel (task 13) is disabled for the
    same reason as the JsonCrush test above -- this test isolates the
    StructMap-before-ToolResults interaction, not Pixel's own profitability
    threshold, which could in principle also fire on a sufficiently dense
    signature map and turn `final_text` into a non-string content list.
    """
    monkeypatch.setenv("MEMO_CONFIG_DIR", str(tmp_path / "config-home"))
    monkeypatch.setenv("MEMO_PROXY_PIXEL", "0")

    repo_root = Path(__file__).resolve().parent.parent
    big_python_source = (repo_root / "src/memo/memory/search_ops.py").read_text(encoding="utf-8")
    raw = json.dumps(
        {
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "r1",
                            "name": "Read",
                            "input": {"file_path": "src/memo/memory/search_ops.py"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "r1",
                            "content": big_python_source,
                        }
                    ],
                },
            ]
        }
    ).encode()

    ctx = _ctx(tmp_path)
    out, plan = rewrite_body(raw, ctx)

    assert "structmap" in plan.applied
    assert "toolresults" in plan.applied

    final_text = json.loads(out)["messages"][1]["content"][0]["content"]

    # ToolResults' generic_fallback keeps a HEAD and a TAIL of whatever it
    # was handed -- and StructMap's own marker (appended to the very END of
    # its output) lands inside that kept tail, so the final text legitimately
    # carries TWO markers back to back: StructMap's own (inner, correctly
    # still "Full original" -- its own stash really is the pristine source)
    # and ToolResults' own (outer, appended last -- the one that describes
    # what is DIRECTLY retrievable from the text as it stands right now).
    # `re.search`/`.finditer()`'s first match would grab the INNER marker by
    # accident (both share the same `memo_retrieve(key="...")` shape) and
    # silently validate the wrong one -- the outer marker is always the LAST
    # one appended, so anchor on that specifically.
    matches = list(re.finditer(r'memo_retrieve\(key="([0-9a-f]+)"\)', final_text))
    assert matches, f"expected at least one recovery marker in: {final_text!r}"
    outer_key = matches[-1].group(1)

    from memo.proxy import ccr

    recovered = ccr.recover(ctx.state_dir, outer_key)
    assert recovered is not None
    assert re.search(r"\[memo: \d+ chars elided, \d+ kept\. ", recovered), (
        "setup check: the content ToolResults stashed should still carry "
        "StructMap's own nested marker, or this test isn't reproducing the bug"
    )

    outer_marker_text = final_text[final_text.rfind("[memo:") :]
    assert "Full original" not in outer_marker_text, (
        "the OUTER marker must not claim to hold the full original when what "
        "its own key actually recovers is itself only StructMap's "
        "intermediate signature map, with its own recovery marker nested "
        f"inside it: {outer_marker_text!r}"
    )
    assert "Not the full original" in outer_marker_text


# ---------------------------------------------------------------------------
# Streaming must never buffer
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_streams_reach_the_client_incrementally():
    """The byte watchdog aborts after 180s of silence; buffering would trip it."""
    from memo.proxy.server import _relay_chunks

    async def source():
        for chunk in (b"event: a\n", b"data: 1\n\n", b"event: b\n"):
            yield chunk

    seen = [c async for c in _relay_chunks(source())]
    assert seen == [b"event: a\n", b"data: 1\n\n", b"event: b\n"]


# ---------------------------------------------------------------------------
# sniff_usage
# ---------------------------------------------------------------------------


def test_usage_is_sniffed_out_of_a_streaming_response():
    from memo.proxy.server import sniff_usage

    captured: dict[str, int] = {}
    sniff_usage(
        b'event: message_start\ndata: {"message":{"usage":{"input_tokens":100}}}\n\n',
        captured,
    )
    sniff_usage(b'event: message_delta\ndata: {"usage":{"output_tokens":42}}\n\n', captured)
    assert captured["input_tokens"] == 100
    assert captured["output_tokens"] == 42


def test_sniffing_a_malformed_chunk_does_not_raise():
    from memo.proxy.server import sniff_usage

    captured: dict[str, int] = {}
    sniff_usage(b'data: {"usage": not json\n', captured)
    assert captured == {}


def test_sniffing_a_top_level_json_array_does_not_raise():
    """`event.get("usage")` would raise AttributeError on a list — the exact
    'outside the try' shape called out across tasks 3-6. `["usage"]` parses as
    valid JSON, is a list (not a dict), and contains the `"usage"` substring
    that gates the scan."""
    from memo.proxy.server import sniff_usage

    captured: dict[str, int] = {}
    sniff_usage(b'data: ["usage"]\n', captured)
    assert captured == {}


# ---------------------------------------------------------------------------
# record_tool_usage — testable without a running server
# ---------------------------------------------------------------------------


def _payload_with_tool_use(*names):
    content = [
        {"type": "tool_use", "id": f"t{i}", "name": n, "input": {}} for i, n in enumerate(names)
    ]
    return json.dumps(
        {
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": content},
            ]
        }
    ).encode()


def test_record_tool_usage_extracts_names_from_tool_use_blocks(tmp_path):
    from memo.proxy.server import record_tool_usage, tool_usage_path

    record_tool_usage(tmp_path, "sess-1", _payload_with_tool_use("memo_search", "memo_save"))
    data = json.loads(tool_usage_path(tmp_path).read_text())
    assert data["schema"] == "memo.proxy.tool_usage.v1"
    assert sorted(data["sessions"]["sess-1"]["tools"]) == ["memo_save", "memo_search"]
    assert isinstance(data["sessions"]["sess-1"]["ts"], float)


def test_record_tool_usage_merges_across_calls_in_the_same_session(tmp_path):
    from memo.proxy.server import record_tool_usage, tool_usage_path

    record_tool_usage(tmp_path, "sess-1", _payload_with_tool_use("memo_search"))
    record_tool_usage(tmp_path, "sess-1", _payload_with_tool_use("memo_save"))
    data = json.loads(tool_usage_path(tmp_path).read_text())
    assert sorted(data["sessions"]["sess-1"]["tools"]) == ["memo_save", "memo_search"]


def test_record_tool_usage_keeps_sessions_separate(tmp_path):
    from memo.proxy.server import record_tool_usage, tool_usage_path

    record_tool_usage(tmp_path, "sess-1", _payload_with_tool_use("memo_search"))
    record_tool_usage(tmp_path, "sess-2", _payload_with_tool_use("memo_save"))
    data = json.loads(tool_usage_path(tmp_path).read_text())
    assert data["sessions"]["sess-1"]["tools"] == ["memo_search"]
    assert data["sessions"]["sess-2"]["tools"] == ["memo_save"]


def test_record_tool_usage_ignores_a_payload_without_tool_use(tmp_path):
    from memo.proxy.server import record_tool_usage, tool_usage_path

    raw = json.dumps({"messages": [{"role": "user", "content": "hi"}]}).encode()
    record_tool_usage(tmp_path, "sess-1", raw)
    assert not tool_usage_path(tmp_path).exists()


def test_record_tool_usage_never_raises_on_malformed_json(tmp_path):
    from memo.proxy.server import record_tool_usage, tool_usage_path

    record_tool_usage(tmp_path, "sess-1", b"not json at all")
    assert not tool_usage_path(tmp_path).exists()


def test_record_tool_usage_never_raises_when_state_dir_is_unwritable(tmp_path):
    from memo.proxy.server import record_tool_usage

    # `proxy` must be a directory for tool_usage_path's parent.mkdir() to
    # succeed; put a plain FILE there instead so mkdir(exist_ok=True) raises
    # FileExistsError internally — the function must swallow it.
    blocker = tmp_path / "proxy"
    blocker.write_text("not a directory")
    record_tool_usage(tmp_path, "sess-1", _payload_with_tool_use("memo_search"))
    # No exception means the contract held; the blocker file is untouched.
    assert blocker.read_text() == "not a directory"


def test_record_tool_usage_never_raises_on_non_dict_payload(tmp_path):
    from memo.proxy.server import record_tool_usage, tool_usage_path

    record_tool_usage(tmp_path, "sess-1", b"[1, 2, 3]")
    assert not tool_usage_path(tmp_path).exists()


# ---------------------------------------------------------------------------
# build_app — full round trip over a double ASGI transport (no real network)
# ---------------------------------------------------------------------------


def _make_upstream_app(captured: dict):
    from starlette.applications import Starlette
    from starlette.responses import StreamingResponse
    from starlette.routing import Route

    async def upstream_endpoint(request):
        captured["path"] = request.url.path
        captured["query"] = request.url.query
        captured["headers"] = dict(request.headers)
        captured["body"] = await request.body()

        async def gen():
            yield b'event: message_start\ndata: {"message":{"usage":{"input_tokens":5}}}\n\n'
            yield b'event: message_delta\ndata: {"usage":{"output_tokens":2}}\n\n'

        return StreamingResponse(gen(), media_type="text/event-stream")

    return Starlette(routes=[Route("/v1/messages", upstream_endpoint, methods=["POST"])])


@pytest.fixture
def proxy_env(tmp_path, monkeypatch):
    """Isolated Config.from_env() resolution for the request handler, which
    (per the brief) constructs a fresh Config per request."""
    monkeypatch.setenv("MEMO_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("MEMO_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MEMO_PROXY_HOLDOUT_FRAC", "0")
    return tmp_path / "state"


async def _round_trip(monkeypatch, *, path_and_query: str, body: bytes, headers: dict):
    """POST `body` through a real build_app() instance to a fake upstream ASGI
    app, both wired via httpx.ASGITransport — no real network call anywhere.
    Returns (response, captured-by-upstream dict)."""
    import httpx

    from memo.proxy import server

    captured: dict = {}
    upstream_app = _make_upstream_app(captured)
    real_async_client = httpx.AsyncClient

    class _PatchedAsyncClient(real_async_client):  # type: ignore[misc]
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = httpx.ASGITransport(app=upstream_app)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", _PatchedAsyncClient)

    app = server.build_app(upstream="https://api.anthropic.com")
    proxy_transport = httpx.ASGITransport(app=app)
    async with real_async_client(transport=proxy_transport, base_url="http://proxy") as client:
        resp = await client.post(path_and_query, content=body, headers=headers)
    return resp, captured


@pytest.mark.asyncio
async def test_query_string_survives_the_round_trip(proxy_env, monkeypatch):
    """Deviation 1: `?beta=true` must reach the upstream request. Nothing else
    in this suite would catch its loss — the FastAPI route decorator matches
    regardless of query string, so a hardcoded '/v1/messages' target would
    pass every OTHER test here and still silently drop the flag."""
    body = json.dumps({"messages": []}).encode()
    resp, captured = await _round_trip(
        monkeypatch,
        path_and_query="/v1/messages?beta=true",
        body=body,
        headers={"x-api-key": "k", "content-type": "application/json"},
    )
    assert resp.status_code == 200
    assert captured["path"] == "/v1/messages"
    assert captured["query"] == "beta=true"


@pytest.mark.asyncio
async def test_no_query_string_forwards_a_bare_path(proxy_env, monkeypatch):
    body = json.dumps({"messages": []}).encode()
    resp, captured = await _round_trip(
        monkeypatch,
        path_and_query="/v1/messages",
        body=body,
        headers={"x-api-key": "k"},
    )
    assert resp.status_code == 200
    assert captured["path"] == "/v1/messages"
    assert captured["query"] == ""


@pytest.mark.asyncio
async def test_credential_headers_reach_the_upstream_request(proxy_env, monkeypatch):
    body = json.dumps({"messages": []}).encode()
    resp, captured = await _round_trip(
        monkeypatch,
        path_and_query="/v1/messages?beta=true",
        body=body,
        headers={"x-api-key": "SUPERSECRET", "anthropic-beta": "oauth-2025-04-20,foo"},
    )
    assert resp.status_code == 200
    assert captured["headers"]["x-api-key"] == "SUPERSECRET"
    assert captured["headers"]["anthropic-beta"] == "oauth-2025-04-20,foo"


@pytest.mark.asyncio
async def test_usage_counters_are_recorded_after_a_streamed_response(proxy_env, monkeypatch):
    from memo.proxy import meter

    body = json.dumps({"messages": [{"role": "user", "content": "hi"}]}).encode()
    resp, _ = await _round_trip(
        monkeypatch,
        path_and_query="/v1/messages?beta=true",
        body=body,
        headers={"x-api-key": "k"},
    )
    assert resp.status_code == 200
    # Consume the stream fully so the generator's `finally` (which appends the
    # measurement row) has actually run.
    await resp.aread()
    summary = meter.summarize(proxy_env)
    assert summary["n_treated"] + summary["n_holdout"] == 1


@pytest.mark.asyncio
async def test_saved_by_reaches_the_persisted_ledger_row(proxy_env, monkeypatch):
    """Round-2 regression: server.py built the meter.Record from `plan.applied`
    + `plan.est_saved_tokens` but never `plan.saved_by`, so every real
    request landed with an empty `saved_by` and `memo tokens --by-transform`
    could never attribute savings honestly in production, no matter what
    meter.py's aggregation does with hand-fed test rows."""
    from memo.proxy import meter

    proxy_env.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("MEMO_CRUSHER_ENABLED", "1")
    big_array = json.dumps([{"id": i, "text": "row " * 20} for i in range(200)])
    body = json.dumps(
        {"messages": [{"role": "user", "content": [{"type": "tool_result", "content": big_array}]}]}
    ).encode()
    resp, _ = await _round_trip(
        monkeypatch,
        path_and_query="/v1/messages",
        body=body,
        headers={"x-api-key": "k"},
    )
    assert resp.status_code == 200
    await resp.aread()

    line = meter.ledger_path(proxy_env).read_text(encoding="utf-8").strip().splitlines()[-1]
    row = json.loads(line)
    assert row["saved_by"], "the persisted row must carry the real per-transform split"
    assert sum(row["saved_by"].values()) == row["est_saved_tokens"]
    # Not every applied transform earned credit — proves this isn't a flat
    # split re-derived at read time.
    assert set(row["saved_by"]) < set(row["transforms"])


@pytest.mark.asyncio
async def test_tool_usage_is_recorded_during_a_real_request(proxy_env, monkeypatch):
    from memo.proxy.server import tool_usage_path

    body = json.dumps(
        {
            "messages": [
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": "t1", "name": "memo_search", "input": {}}
                    ],
                },
            ]
        }
    ).encode()
    resp, _ = await _round_trip(
        monkeypatch,
        path_and_query="/v1/messages?beta=true",
        body=body,
        headers={"x-api-key": "k", "x-claude-code-session-id": "sess-real"},
    )
    assert resp.status_code == 200
    data = json.loads(tool_usage_path(proxy_env).read_text())
    assert data["sessions"]["sess-real"]["tools"] == ["memo_search"]


@pytest.mark.asyncio
async def test_session_key_comes_from_the_header_and_freezes_across_real_requests(
    proxy_env, monkeypatch
):
    """End-to-end regression for the Critical bug found in fix round 1:
    `Context.session_key` used to be `_request_key(raw)` (a hash of the body,
    different on every single turn), which defeated ANY per-session freeze
    regardless of caching logic — `record_tool_usage` writes BEFORE
    `rewrite_body` runs, keyed on the real `x-claude-code-session-id` header,
    so the moment a previously-pruned tool got called, the next turn's
    keep-set computation (if re-run fresh, or even if frozen under the WRONG
    key) would see it and the tools list would drift.

    This drives two real requests through the actual ASGI app (not a mock)
    with the SAME session header, where the SECOND request's own body
    reports a `memo_graph` tool_use — exactly the discover-then-hydrate flow
    `memo_tool_docs` exists to enable. The wire payload's tool list must be
    byte-for-byte the same set on both turns.
    """
    monkeypatch.setenv("MEMO_PROXY_ENABLED", "1")
    monkeypatch.setenv("MEMO_PROXY_TOOL_SCHEMAS", "1")
    monkeypatch.setenv("MEMO_PROXY_TOOL_WINDOW_SESSIONS", "20")

    tools = [
        {"name": name, "description": f"description of {name} " * 5, "input_schema": {}}
        for name in ("memo_search", "memo_graph", "memo_rename")
    ]
    headers = {"x-api-key": "k", "x-claude-code-session-id": "real-sess-1"}

    # `_round_trip` patches httpx.AsyncClient itself; calling it twice with
    # ONE shared monkeypatch stacks the patch (the second _PatchedAsyncClient
    # subclasses the first, and the first's __init__ overwrites kwargs
    # AFTER the second's, silently routing turn 2's request to turn 1's fake
    # upstream). Each call gets its own MonkeyPatch scope, undone right
    # after, so turn 2 starts from the true, unpatched httpx.AsyncClient.
    mp1 = pytest.MonkeyPatch()
    try:
        # Turn 1: no usage history yet — only the always-keep tools survive.
        body1 = json.dumps(
            {"tools": tools, "messages": [{"role": "user", "content": "hi"}]}
        ).encode()
        resp1, captured1 = await _round_trip(
            mp1, path_and_query="/v1/messages?beta=true", body=body1, headers=headers
        )
    finally:
        mp1.undo()
    assert resp1.status_code == 200
    names1 = {t["name"] for t in json.loads(captured1["body"])["tools"]}
    assert "memo_graph" not in names1

    mp2 = pytest.MonkeyPatch()
    try:
        # Turn 2, same session: memo_graph gets called mid-session.
        # record_tool_usage writes it to tool_usage.json BEFORE this same
        # request's rewrite_body runs.
        body2 = json.dumps(
            {
                "tools": tools,
                "messages": [
                    {"role": "user", "content": "hi"},
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "tool_use", "id": "t1", "name": "memo_graph", "input": {}}
                        ],
                    },
                ],
            }
        ).encode()
        resp2, captured2 = await _round_trip(
            mp2, path_and_query="/v1/messages?beta=true", body=body2, headers=headers
        )
    finally:
        mp2.undo()
    assert resp2.status_code == 200
    names2 = {t["name"] for t in json.loads(captured2["body"])["tools"]}

    # Frozen: turn 2's wire tool list must match turn 1's, not the
    # now-updated tool_usage.json this same request just wrote.
    assert names2 == names1
