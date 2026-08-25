import pytest

from memo.proxy.zones import (
    Zones,
    prefix_fingerprint,
    scan_scope,
    split,
    stable_head_fingerprint,
    whole_history_scope,
)


def _payload(n_messages: int) -> dict:
    return {
        "model": "claude-opus-5",
        "system": [{"type": "text", "text": "you are helpful"}],
        "tools": [{"name": "memo_search", "input_schema": {"type": "object"}}],
        "messages": [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"}
            for i in range(n_messages)
        ],
    }


def test_split_puts_the_last_turns_in_the_live_zone():
    z = split(_payload(10), live_turns=2)
    assert len(z.live_messages) == 2
    assert len(z.frozen_messages) == 8
    assert z.live_messages[-1]["content"] == "m9"


def test_short_conversation_is_all_live():
    z = split(_payload(1), live_turns=2)
    assert z.frozen_messages == []
    assert len(z.live_messages) == 1


def test_missing_system_and_tools_are_empty_not_none():
    z = split({"messages": []})
    assert z.system == []
    assert z.tools == []


def test_prefix_fingerprint_ignores_the_live_zone():
    a = split(_payload(10), live_turns=2)
    b = split(_payload(10), live_turns=2)
    b.live_messages[-1]["content"] = "totally different"
    assert prefix_fingerprint(a) == prefix_fingerprint(b)


def test_prefix_fingerprint_changes_when_tools_change():
    a = split(_payload(10), live_turns=2)
    b = split(_payload(10), live_turns=2)
    b.tools.append({"name": "memo_get", "input_schema": {"type": "object"}})
    assert prefix_fingerprint(a) != prefix_fingerprint(b)


def test_stable_head_fingerprint_ignores_frozen_messages_too():
    """Unlike prefix_fingerprint, growth of frozen_messages (a normal,
    cache-friendly consequence of the conversation getting longer) must not
    change this fingerprint — only system/tools do."""
    a = split(_payload(10), live_turns=2)
    b = split(_payload(20), live_turns=2)  # far more history -> different frozen_messages
    assert a.frozen_messages != b.frozen_messages
    assert stable_head_fingerprint(a) == stable_head_fingerprint(b)


def test_stable_head_fingerprint_changes_when_tools_change():
    a = split(_payload(10), live_turns=2)
    b = split(_payload(10), live_turns=2)
    b.tools.append({"name": "memo_get", "input_schema": {"type": "object"}})
    assert stable_head_fingerprint(a) != stable_head_fingerprint(b)


def test_stable_head_fingerprint_changes_when_system_changes():
    a = split(_payload(10), live_turns=2)
    b = split(_payload(10), live_turns=2)
    b.system.append({"type": "text", "text": "extra instruction"})
    assert stable_head_fingerprint(a) != stable_head_fingerprint(b)


def test_zones_reassemble_into_an_equivalent_payload():
    original = _payload(6)
    z = split(original, live_turns=2)
    assert z.to_payload(original)["messages"] == original["messages"]


@pytest.mark.parametrize("bad", [None, [], "x", 42, 3.5, True])
def test_split_never_raises_on_a_malformed_payload(bad):
    z = split(bad)
    assert z.system == []
    assert z.tools == []
    assert z.frozen_messages == []
    assert z.live_messages == []


# ---------------------------------------------------------------------------
# whole_history_scope / scan_scope
# ---------------------------------------------------------------------------


def test_whole_history_scope_defaults_to_true(monkeypatch):
    monkeypatch.delenv("MEMO_PROXY_CONTENT_SCOPE", raising=False)
    assert whole_history_scope() is True


def test_whole_history_scope_is_false_when_flag_is_tail(monkeypatch):
    monkeypatch.setenv("MEMO_PROXY_CONTENT_SCOPE", "tail")
    assert whole_history_scope() is False


def test_whole_history_scope_never_raises(monkeypatch):
    monkeypatch.setattr(
        "memo.flags.flag_str",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    assert whole_history_scope() is True


def test_scan_scope_returns_only_live_messages_under_tail(monkeypatch):
    monkeypatch.setenv("MEMO_PROXY_CONTENT_SCOPE", "tail")
    zones = Zones(
        frozen_messages=[{"role": "user", "content": "old"}],
        live_messages=[{"role": "user", "content": "new"}],
    )
    assert scan_scope(zones) == [{"role": "user", "content": "new"}]


def test_scan_scope_returns_frozen_then_live_under_all(monkeypatch):
    monkeypatch.delenv("MEMO_PROXY_CONTENT_SCOPE", raising=False)
    frozen = [{"role": "user", "content": "old"}]
    live = [{"role": "user", "content": "new"}]
    zones = Zones(frozen_messages=frozen, live_messages=live)
    assert scan_scope(zones) == [*frozen, *live]
