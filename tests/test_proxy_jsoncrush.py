import json

from memo.proxy.plan import Context
from memo.proxy.transforms.jsoncrush import JsonCrush
from memo.proxy.zones import Zones


def _zones(text: str) -> Zones:
    return Zones(
        live_messages=[
            {
                "role": "user",
                "content": [{"type": "tool_result", "content": text}],
            }
        ]
    )


def test_a_large_json_array_is_crushed(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMO_PROXY_JSONCRUSH", "1")
    monkeypatch.setenv("MEMO_CRUSHER_ENABLED", "1")
    big = json.dumps([{"id": i, "text": "row " * 20} for i in range(200)])
    zones = _zones(big)
    saved = JsonCrush().apply(zones, Context(state_dir=tmp_path, session_key="s"))
    assert saved > 0
    assert len(zones.live_messages[0]["content"][0]["content"]) < len(big)


def test_non_json_content_is_left_alone(tmp_path):
    zones = _zones("just some prose, definitely not json")
    saved = JsonCrush().apply(zones, Context(state_dir=tmp_path, session_key="s"))
    assert saved == 0
    assert zones.live_messages[0]["content"][0]["content"] == "just some prose, definitely not json"


def test_a_small_array_is_not_worth_crushing(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMO_CRUSHER_ENABLED", "1")
    small = json.dumps([{"id": 1}, {"id": 2}])
    zones = _zones(small)
    assert JsonCrush().apply(zones, Context(state_dir=tmp_path, session_key="s")) == 0


def test_disabled_via_proxy_flag_is_a_no_op(tmp_path, monkeypatch):
    """MEMO_PROXY_JSONCRUSH gates the transform independently of the
    capture-plane MEMO_CRUSHER_ENABLED flag."""
    monkeypatch.setenv("MEMO_CRUSHER_ENABLED", "1")
    monkeypatch.setenv("MEMO_PROXY_JSONCRUSH", "0")
    big = json.dumps([{"id": i, "text": "row " * 20} for i in range(200)])
    zones = _zones(big)
    saved = JsonCrush().apply(zones, Context(state_dir=tmp_path, session_key="s"))
    assert saved == 0
    assert zones.live_messages[0]["content"][0]["content"] == big


def test_crusher_runs_without_the_capture_flag_explicitly_set(tmp_path, monkeypatch):
    """The whole point of task 11: MEMO_CRUSHER_ENABLED defaults OFF (it
    gates memo's own ingest) but the proxy must still be able to run the
    crusher when MEMO_PROXY_JSONCRUSH is on, without the caller having to
    set the capture-plane flag itself."""
    monkeypatch.setenv("MEMO_PROXY_JSONCRUSH", "1")
    monkeypatch.delenv("MEMO_CRUSHER_ENABLED", raising=False)
    big = json.dumps([{"id": i, "text": "row " * 20} for i in range(200)])
    zones = _zones(big)
    saved = JsonCrush().apply(zones, Context(state_dir=tmp_path, session_key="s"))
    assert saved > 0
    assert len(zones.live_messages[0]["content"][0]["content"]) < len(big)


def test_capture_flag_env_is_restored_after_the_call(tmp_path, monkeypatch):
    """The process-env override this transform uses to run the crusher
    without flipping the persistent capture-plane default must not leak
    past the single call — see the module docstring in jsoncrush.py."""
    import os

    monkeypatch.delenv("MEMO_CRUSHER_ENABLED", raising=False)
    big = json.dumps([{"id": i, "text": "row " * 20} for i in range(200)])
    zones = _zones(big)
    JsonCrush().apply(zones, Context(state_dir=tmp_path, session_key="s"))
    assert "MEMO_CRUSHER_ENABLED" not in os.environ


def test_an_explicit_capture_flag_off_is_never_overridden(tmp_path, monkeypatch):
    """If the operator explicitly turned the capture-plane crusher OFF via
    the env var, the proxy must respect that -- not silently force it on
    just because MEMO_PROXY_JSONCRUSH is on."""
    monkeypatch.setenv("MEMO_CRUSHER_ENABLED", "0")
    big = json.dumps([{"id": i, "text": "row " * 20} for i in range(200)])
    zones = _zones(big)
    saved = JsonCrush().apply(zones, Context(state_dir=tmp_path, session_key="s"))
    assert saved == 0
    assert zones.live_messages[0]["content"][0]["content"] == big


# --- MEMO_PROXY_CONTENT_SCOPE: whole-history (default) vs tail-only ----------


def test_apply_crushes_a_frozen_block_under_the_default_whole_history_scope(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMO_PROXY_JSONCRUSH", "1")
    monkeypatch.delenv("MEMO_PROXY_CONTENT_SCOPE", raising=False)
    monkeypatch.delenv("MEMO_CRUSHER_ENABLED", raising=False)
    big = json.dumps([{"id": i, "text": "row " * 20} for i in range(200)])
    zones = _zones(big)
    zones.frozen_messages, zones.live_messages = zones.live_messages, []
    saved = JsonCrush().apply(zones, Context(state_dir=tmp_path, session_key="s"))
    assert saved > 0
    assert len(zones.frozen_messages[0]["content"][0]["content"]) < len(big)


def test_apply_leaves_a_frozen_block_untouched_under_tail_only_scope(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMO_PROXY_CONTENT_SCOPE", "tail")
    monkeypatch.delenv("MEMO_CRUSHER_ENABLED", raising=False)
    big = json.dumps([{"id": i, "text": "row " * 20} for i in range(200)])
    zones = _zones(big)
    zones.frozen_messages, zones.live_messages = zones.live_messages, []
    saved = JsonCrush().apply(zones, Context(state_dir=tmp_path, session_key="s"))
    assert saved == 0
    assert zones.frozen_messages[0]["content"][0]["content"] == big
