from memo.proxy.zones import prefix_fingerprint, split


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


def test_zones_reassemble_into_an_equivalent_payload():
    original = _payload(6)
    z = split(original, live_turns=2)
    assert z.to_payload(original)["messages"] == original["messages"]
