# tests/test_proxy_server.py
import json

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


def _ctx(tmp_path):
    return Context(state_dir=tmp_path, session_key="s", project=None)


def test_anthropic_beta_is_forwarded_verbatim():
    out = forward_headers({"anthropic-beta": "oauth-2025-04-20,foo", "host": "x"})
    assert out["anthropic-beta"] == "oauth-2025-04-20,foo"


def test_auth_headers_are_forwarded():
    out = forward_headers({"authorization": "Bearer tok", "x-api-key": "k"})
    assert out["authorization"] == "Bearer tok"
    assert out["x-api-key"] == "k"


def test_hop_by_hop_headers_are_dropped():
    out = forward_headers({"host": "127.0.0.1:8768", "content-length": "12",
                           "connection": "keep-alive", "x-api-key": "k"})
    assert "host" not in out
    assert "content-length" not in out
    assert "connection" not in out


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


def test_no_header_value_reaches_the_log(tmp_path, caplog):
    with caplog.at_level("DEBUG"):
        forward_headers({"x-api-key": "SUPERSECRET", "authorization": "Bearer SECRET2"})
    assert "SUPERSECRET" not in caplog.text
    assert "SECRET2" not in caplog.text


@pytest.mark.asyncio
async def test_streams_reach_the_client_incrementally(tmp_path):
    """The byte watchdog aborts after 180s of silence; buffering would trip it."""
    from memo.proxy.server import _relay_chunks

    async def source():
        for chunk in (b"event: a\n", b"data: 1\n\n", b"event: b\n"):
            yield chunk

    seen = [c async for c in _relay_chunks(source())]
    assert seen == [b"event: a\n", b"data: 1\n\n", b"event: b\n"]


def test_usage_is_sniffed_out_of_a_streaming_response():
    from memo.proxy.server import sniff_usage

    captured: dict[str, int] = {}
    sniff_usage(
        b'event: message_start\ndata: {"message":{"usage":{"input_tokens":100}}}\n\n',
        captured,
    )
    sniff_usage(
        b'event: message_delta\ndata: {"usage":{"output_tokens":42}}\n\n', captured
    )
    assert captured["input_tokens"] == 100
    assert captured["output_tokens"] == 42


def test_sniffing_a_malformed_chunk_does_not_raise():
    from memo.proxy.server import sniff_usage

    captured: dict[str, int] = {}
    sniff_usage(b'data: {"usage": not json\n', captured)
    assert captured == {}
