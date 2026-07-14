"""save_gate: per-type preset -> GatePolicy."""

from __future__ import annotations

from memo import save_gate


def test_default_is_balanced_warn(monkeypatch):
    monkeypatch.delenv("MEMO_SAVE_GATE_PRESETS", raising=False)
    p = save_gate.resolve_gate("decision")
    assert p.dedup_mode == "warn"  # == current behavior (no-op)


def test_strict_preset_refuses(monkeypatch):
    monkeypatch.setenv("MEMO_SAVE_GATE_PRESETS", '{"decision": "strict", "bug": "strict"}')
    assert save_gate.resolve_gate("decision").dedup_mode == "refuse"
    assert save_gate.resolve_gate("bug").dedup_mode == "refuse"


def test_bare_preset_name_applies_to_all_types(monkeypatch):
    # A bare preset name (not JSON) is the natural usage — it must apply globally
    # instead of silently no-op'ing (json.loads("strict") used to fail → default).
    monkeypatch.setenv("MEMO_SAVE_GATE_PRESETS", "strict")
    assert save_gate.resolve_gate("decision").dedup_mode == "refuse"
    assert save_gate.resolve_gate("note").dedup_mode == "refuse"
    monkeypatch.setenv("MEMO_SAVE_GATE_PRESETS", "balanced")
    assert save_gate.resolve_gate("anything").dedup_mode == "warn"


def test_unlisted_type_falls_back_to_balanced(monkeypatch):
    monkeypatch.setenv("MEMO_SAVE_GATE_PRESETS", '{"decision": "strict"}')
    assert save_gate.resolve_gate("note").dedup_mode == "warn"


def test_permissive_disables_dedup(monkeypatch):
    monkeypatch.setenv("MEMO_SAVE_GATE_PRESETS", '{"note": "permissive"}')
    assert save_gate.resolve_gate("note").dedup_mode == "off"


def test_malformed_json_falls_back_to_balanced(monkeypatch):
    monkeypatch.setenv("MEMO_SAVE_GATE_PRESETS", "{not json")
    assert save_gate.resolve_gate("decision").dedup_mode == "warn"


def test_unknown_preset_name_falls_back_to_balanced(monkeypatch):
    monkeypatch.setenv("MEMO_SAVE_GATE_PRESETS", '{"decision": "bogus"}')
    assert save_gate.resolve_gate("decision").dedup_mode == "warn"
