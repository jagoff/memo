from memo import cli_token_gate as g
from memo import token_meter as tm


def _seed(state, sid, n_turns, tool, answer, injected_chars, grounded):
    led = tm._read_ledger(state)
    led.setdefault("sessions", {})[sid] = {
        "ts": "2026-07-04T00:00:00+00:00",
        "n_turns": n_turns,
        "answer_tok": answer,
        "tool_tok": tool,
        "injected_chars": injected_chars,
        "grounded": grounded,
    }
    tm._write_ledger(state, led)


def test_gate_passes_when_cost_per_grounded_drops(tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    _seed(state, "S1", 3, 100, 60, injected_chars=800, grounded=2)  # 200 tok / 2 grounded = 100
    ok, _ = g.check_gate(state, update_baseline=True)  # seed baseline
    assert ok
    _seed(state, "S1", 3, 100, 60, injected_chars=400, grounded=2)  # cost per grounded halves
    ok, info = g.check_gate(state, update_baseline=False)
    assert ok
    assert info["cost_per_grounded"] <= info["baseline"]["cost_per_grounded"]


def test_gate_fails_when_grounding_drops(tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    _seed(state, "S1", 3, 100, 60, injected_chars=800, grounded=4)
    g.check_gate(state, update_baseline=True)
    _seed(state, "S1", 3, 100, 60, injected_chars=800, grounded=1)  # grounded collapsed
    ok, _ = g.check_gate(state, update_baseline=False)
    assert not ok


def test_cost_per_grounded_is_inf_when_grounded_is_zero(tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    _seed(state, "S1", 3, 100, 60, injected_chars=800, grounded=0)
    m = g.gate_metrics(state)
    assert m["cost_per_grounded"] == float("inf")
