from memo.flags import REGISTRY, flag_bool


def test_confidence_gate_defaults_off(monkeypatch):
    monkeypatch.delenv("MEMO_RECALL_CONFIDENCE_GATE", raising=False)
    assert flag_bool("MEMO_RECALL_CONFIDENCE_GATE") is False


def test_confidence_gate_is_registered():
    names = {s.name for s in REGISTRY.values()}
    assert "MEMO_RECALL_CONFIDENCE_GATE" in names
