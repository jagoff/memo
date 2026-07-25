"""Parity tests: 3 recall levers must fire on the warm daemon path (_recall_logic).

Each test drives _recall_logic directly (daemon's production call site) and
verifies that the lever is active when its flag is set and a no-op when unset.
Helper unit tests already cover the helpers themselves (collapse_near_dups,
suppress_score, session_budget_scale). These tests assert the wiring.
"""

from __future__ import annotations

import pytest

from memo.config import Config
from memo.memory import Memory
from memo.recall_logic import _recall_logic

# ---------------------------------------------------------------------------
# Shared stub embedder fixture (mirrors test_daemon_session_cited.py)
# ---------------------------------------------------------------------------


@pytest.fixture
def mem(tmp_cfg: Config, monkeypatch) -> Memory:
    cfg = Config(
        data_dir=tmp_cfg.data_dir,
        vault_path=tmp_cfg.vault_path,
        state_dir=tmp_cfg.state_dir,
        embedder_dims=64,
    )

    def _stub_embed(self, inputs):
        out = []
        for s in inputs:
            h = sum(ord(c) for c in s) % 64
            v = [0.0] * 64
            v[h] = 1.0
            out.append(v)
        return out

    monkeypatch.setattr("memo.embedder.MLXEmbedder.embed", _stub_embed)
    monkeypatch.setattr(
        "memo.embedder.MLXEmbedder.embed_query",
        lambda self, q: _stub_embed(self, [q])[0],
    )
    m = Memory(cfg)
    yield m
    m.close()


def _base_env(monkeypatch) -> None:
    """Permissive search settings so stub embedder always surfaces results."""
    monkeypatch.setenv("MEMO_RECALL_MIN_SIM", "0.0")
    monkeypatch.setenv("MEMO_RECALL_SKIP_BELOW", "0")
    monkeypatch.setenv("MEMO_RECALL_MIN_BODY_CHARS", "0")


# ---------------------------------------------------------------------------
# Lever 1: precision-gate
# ---------------------------------------------------------------------------


def test_precision_gate_suppresses_when_flag_on(mem: Memory, monkeypatch):
    """With MEMO_RECALL_PRECISION_GATE=1 and a suppressive band, daemon returns '{}'."""
    _base_env(monkeypatch)
    monkeypatch.setenv("MEMO_RECALL_PRECISION_GATE", "1")

    mem.save(content="fix del flock", title="Flock fix", type_="fact")

    # Monkeypatch the gate to always suppress (simulates a learned zero-grounding band).
    monkeypatch.setattr(
        "memo.token_meter.load_precision_bands", lambda _: {"0.00": {"suppress": True}}
    )
    monkeypatch.setattr("memo.token_meter.suppress_score", lambda score, bands: True)

    out, _ = _recall_logic("fix del flock", None, mem, mem.cfg)
    assert out == "{}", "precision-gate must return empty injection when top score is suppressed"


def test_precision_gate_passes_when_flag_off(mem: Memory, monkeypatch):
    """With MEMO_RECALL_PRECISION_GATE unset, the gate is bypassed even if bands would suppress."""
    _base_env(monkeypatch)
    monkeypatch.delenv("MEMO_RECALL_PRECISION_GATE", raising=False)

    mem.save(content="fix del flock", title="Flock fix", type_="fact")

    # Gate would suppress, but flag is off → should still inject.
    monkeypatch.setattr("memo.token_meter.suppress_score", lambda score, bands: True)
    monkeypatch.setattr(
        "memo.token_meter.load_precision_bands", lambda _: {"0.00": {"suppress": True}}
    )

    out, _ = _recall_logic("fix del flock", None, mem, mem.cfg)
    assert out != "{}", "precision-gate must be bypassed when flag is unset"


# ---------------------------------------------------------------------------
# Lever 2: intra-dedup
# ---------------------------------------------------------------------------


def test_intra_dedup_collapses_near_dups_when_flag_on(mem: Memory, monkeypatch):
    """With MEMO_RECALL_INTRA_DEDUP=1, collapse_near_dups is called; collapsed count is captured."""
    _base_env(monkeypatch)
    monkeypatch.setenv("MEMO_RECALL_INTRA_DEDUP", "1")
    monkeypatch.setenv("MEMO_RECALL_INTRA_DEDUP_THRESHOLD", "0.5")
    monkeypatch.setenv("MEMO_RECALL_TOP_K", "5")

    # Two near-identical memories — same body, same title.
    mem.save(
        content="el cutover memflow a mac-work fue ok",
        title="Deploy cutover mac-work",
        type_="fact",
    )
    mem.save(
        content="el cutover memflow a mac work fue ok",
        title="Deploy cutover en mac-work",
        type_="fact",
    )

    # Capture collapse_near_dups calls to verify it was invoked.
    calls: list[tuple] = []
    original_collapse = __import__(
        "memo.recall_logic", fromlist=["collapse_near_dups"]
    ).collapse_near_dups

    def _spy(relevant, *, threshold):
        result = original_collapse(relevant, threshold=threshold)
        calls.append((len(relevant), len(result)))
        return result

    monkeypatch.setattr("memo.recall_logic.collapse_near_dups", _spy)

    out, _ = _recall_logic("deploy cutover mac-work", None, mem, mem.cfg)
    assert out != "{}", "expected at least one recall hit — check stub embedder / saved memories"
    assert calls, (
        "collapse_near_dups must be called when MEMO_RECALL_INTRA_DEDUP=1 and len(relevant)>1"
    )


def test_intra_dedup_skipped_when_flag_off(mem: Memory, monkeypatch):
    """With both dedup levers off, collapse_near_dups is never called. Both the
    post-top-K MEMO_RECALL_INTRA_DEDUP and the pre-top-K MEMO_RECALL_DEDUP_COLLAPSE
    (default ON since v3.0.0) call the shared collapse_near_dups, so isolating the
    INTRA_DEDUP off-path requires disabling the collapse lever too."""
    _base_env(monkeypatch)
    monkeypatch.setenv("MEMO_RECALL_INTRA_DEDUP", "0")
    monkeypatch.setenv("MEMO_RECALL_DEDUP_COLLAPSE", "0")

    mem.save(
        content="el cutover memflow a mac-work fue ok",
        title="Deploy cutover mac-work",
        type_="fact",
    )
    mem.save(
        content="el cutover memflow a mac work fue ok",
        title="Deploy cutover en mac-work",
        type_="fact",
    )

    calls: list = []

    def _spy(relevant, *, threshold):
        calls.append(True)
        return relevant

    monkeypatch.setattr("memo.recall_logic.collapse_near_dups", _spy)

    _recall_logic("deploy cutover mac-work", None, mem, mem.cfg)
    assert not calls, "collapse_near_dups must NOT be called when MEMO_RECALL_INTRA_DEDUP=0"


# ---------------------------------------------------------------------------
# Lever 3: session-budget decay
# ---------------------------------------------------------------------------


def test_session_budget_decay_triggers_when_over_budget(mem: Memory, monkeypatch):
    """With MEMO_RECALL_SESSION_TOKEN_BUDGET=10, simulate cumulative > budget.

    Captures the token_budget passed to render_recall_context and verifies
    it was halved relative to the base MEMO_RECALL_TOKEN_BUDGET.
    """
    _base_env(monkeypatch)
    monkeypatch.setenv("MEMO_RECALL_SESSION_TOKEN_BUDGET", "10")  # very low session budget
    monkeypatch.setenv("MEMO_RECALL_TOKEN_BUDGET", "600")  # base budget
    monkeypatch.setenv("MEMO_RECALL_ADAPTIVE_BUDGET", "0")
    monkeypatch.setenv("MEMO_RECALL_FORMAT", "full")

    mem.save(content="fix del flock fue ok", title="Flock fix", type_="fact")

    # Simulate cumulative > session budget (e.g. 5000 tokens already used).
    monkeypatch.setattr(
        "memo.dashboard.read_context_cost_log",
        lambda state_dir, **kw: [
            {"kind": "recall", "session_id": "sess-budget-test", "chars": 20000}
        ],
    )

    captured: list[int] = []
    import memo.recall_logic as _rl

    original_render = _rl.render_recall_context

    def _spy(relevant, nudge, *, turn, body_chars, token_budget, **kwargs):
        captured.append(token_budget)
        return original_render(
            relevant,
            nudge,
            turn=turn,
            body_chars=body_chars,
            token_budget=token_budget,
            **kwargs,
        )

    monkeypatch.setattr("memo.recall_logic.render_recall_context", _spy)

    out, _ = _recall_logic(
        "flock fix",
        None,
        mem,
        mem.cfg,
        session_id="sess-budget-test",
    )

    assert out != "{}", "expected at least one recall hit — check stub embedder / saved memories"
    assert captured, "render_recall_context must have been called"
    assert captured[0] < 600, (
        f"session budget decay must reduce token_budget below 600 (got {captured[0]})"
    )


def test_session_budget_no_decay_when_flag_off(mem: Memory, monkeypatch):
    """With MEMO_RECALL_SESSION_TOKEN_BUDGET unset, token_budget stays at base."""
    _base_env(monkeypatch)
    monkeypatch.delenv("MEMO_RECALL_SESSION_TOKEN_BUDGET", raising=False)
    monkeypatch.setenv("MEMO_RECALL_TOKEN_BUDGET", "600")
    monkeypatch.setenv("MEMO_RECALL_ADAPTIVE_BUDGET", "0")
    monkeypatch.setenv("MEMO_RECALL_FORMAT", "full")

    mem.save(content="fix del flock fue ok", title="Flock fix", type_="fact")

    # Even if cumulative is huge, without the flag it must be ignored.
    monkeypatch.setattr(
        "memo.dashboard.read_context_cost_log",
        lambda state_dir, **kw: [
            {"kind": "recall", "session_id": "sess-budget-off", "chars": 20000}
        ],
    )

    captured: list[int] = []
    import memo.recall_logic as _rl

    original_render = _rl.render_recall_context

    def _spy(relevant, nudge, *, turn, body_chars, token_budget, **kwargs):
        captured.append(token_budget)
        return original_render(
            relevant,
            nudge,
            turn=turn,
            body_chars=body_chars,
            token_budget=token_budget,
            **kwargs,
        )

    monkeypatch.setattr("memo.recall_logic.render_recall_context", _spy)

    out, _ = _recall_logic(
        "flock fix",
        None,
        mem,
        mem.cfg,
        session_id="sess-budget-off",
    )

    assert out != "{}", "expected at least one recall hit — check stub embedder / saved memories"
    assert captured, "render_recall_context must have been called"
    assert captured[0] == 600, (
        f"token_budget must stay at 600 when session budget flag is unset (got {captured[0]})"
    )
