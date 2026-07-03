"""Cited-weighted outcome scoring (MEMO_OUTCOME_CITED_WEIGHT).

A memory the answer explicitly CITED (grounding ``method == "cited"``) is
stronger evidence of usefulness than one that merely overlapped lexically or
by embedding. ``compute_utilities`` counts a cited turn as ``cited_weight``
grounded observations (default 2.0); 1.0 restores exact unweighted parity.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from memo import dashboard, outcome


def _recall(tmp: Path, sid: str, turn: int, prompt: str, mem_id: str) -> None:
    dashboard.append_recall_log(
        tmp,
        prompt=prompt,
        via="subprocess",
        session_id=sid,
        turn=turn,
        client="claude-code",
        hits=[{"id": mem_id, "score": 0.8, "title": "t"}],
    )


def _grounded(tmp: Path, sid: str, turn: int, mem_id: str, *, method: str = "lexical") -> None:
    dashboard.append_grounding_log(
        tmp,
        session_id=sid,
        turn=turn,
        recall_id=mem_id,
        used_score=1.0 if method == "cited" else 0.9,
        method=method,
    )


def _corpus(tmp: Path) -> None:
    """memaaaa1: grounded once via lexical overlap; membbbb1: grounded once via
    explicit citation; memcccc1: surfaced, never grounded (keeps prior_mean < 1
    so the weighted numerator is visible through the Bayesian smoothing)."""
    _recall(tmp, "s", 1, "deploy pipeline decision", "memaaaa1")
    _grounded(tmp, "s", 1, "memaaaa1", method="lexical")
    _recall(tmp, "s", 2, "embedder dims decision", "membbbb1")
    _grounded(tmp, "s", 2, "membbbb1", method="cited")
    _recall(tmp, "s", 3, "noise prompt never used", "memcccc1")


# ---------------- utility weighting ----------------


def test_cited_event_outranks_uncited_grounded_by_default(tmp_path: Path) -> None:
    _corpus(tmp_path)

    # No kwarg → flag default (2.0) applies.
    u = outcome.compute_utilities(tmp_path, prior_n=3.0)
    by = u["by_prefix"]
    assert by["memaaaa1"]["cited"] == 0 and by["memaaaa1"]["grounded"] == 1
    assert by["membbbb1"]["cited"] == 1 and by["membbbb1"]["grounded"] == 1
    assert by["membbbb1"]["utility"] > by["memaaaa1"]["utility"]
    # Raw counts and prior stay unweighted.
    assert u["grounded_total"] == 2 and u["surfaced_total"] == 3
    assert u["prior_mean"] == round(2 / 3, 4)


def test_cited_contribution_is_exact_weight_multiple(tmp_path: Path) -> None:
    # memaaaa1: surfaced 3 turns, grounded (lexical) in 1.
    for t in (1, 2, 3):
        _recall(tmp_path, "s", t, f"prompt a {t}", "memaaaa1")
    _grounded(tmp_path, "s", 1, "memaaaa1", method="lexical")
    # membbbb1: surfaced 3 turns, grounded (cited) in 1.
    for t in (1, 2, 3):
        _recall(tmp_path, "s", t, f"prompt b {t}", "membbbb1")
    _grounded(tmp_path, "s", 1, "membbbb1", method="cited")

    w, pn = 3.0, 3.0
    u = outcome.compute_utilities(tmp_path, prior_n=pn, cited_weight=w)
    by, pm = u["by_prefix"], u["prior_mean"]
    # Cited turn counts as w grounded observations; uncited counts as 1.
    assert by["memaaaa1"]["utility"] == round((1 + pm * pn) / (3 + pn), 4)
    assert by["membbbb1"]["utility"] == round(min(1.0, (w + pm * pn) / (3 + pn)), 4)
    assert by["membbbb1"]["utility"] > by["memaaaa1"]["utility"]


def test_weight_one_restores_parity(tmp_path: Path) -> None:
    _corpus(tmp_path)

    u = outcome.compute_utilities(tmp_path, prior_n=3.0, cited_weight=1.0)
    by = u["by_prefix"]
    assert by["membbbb1"]["utility"] == by["memaaaa1"]["utility"]


def test_weight_one_via_env_flag(tmp_path: Path, monkeypatch) -> None:
    _corpus(tmp_path)
    monkeypatch.setenv("MEMO_OUTCOME_CITED_WEIGHT", "1.0")

    u = outcome.compute_utilities(tmp_path, prior_n=3.0)
    by = u["by_prefix"]
    assert by["membbbb1"]["utility"] == by["memaaaa1"]["utility"]


def test_utility_clamped_at_one(tmp_path: Path) -> None:
    # All-cited memory with a large weight must not push utility past 1
    # (roi = floor + utility * span would overflow the cap otherwise).
    _corpus(tmp_path)
    u = outcome.compute_utilities(tmp_path, prior_n=3.0, cited_weight=50.0)
    assert u["by_prefix"]["membbbb1"]["utility"] == 1.0


def test_flag_registered_with_default(monkeypatch) -> None:
    from memo.flags import flag_float

    monkeypatch.delenv("MEMO_OUTCOME_CITED_WEIGHT", raising=False)
    assert flag_float("MEMO_OUTCOME_CITED_WEIGHT") == 2.0
    monkeypatch.setenv("MEMO_OUTCOME_CITED_WEIGHT", "1.5")
    assert flag_float("MEMO_OUTCOME_CITED_WEIGHT") == 1.5


# ---------------- flows into roi (ranking/decay input) ----------------


class _FakeStore:
    def __init__(self, ids: list[str]) -> None:
        self._ids = ids
        self.roi: dict[str, float] = {}

    def all_ids(self) -> list[str]:
        return list(self._ids)

    def set_roi_batch(self, pairs, *, floor: float, cap: float) -> int:
        self.roi = {i: max(floor, min(cap, r)) for i, r in pairs}
        return len(pairs)


def test_reconcile_roi_carries_cited_weight(tmp_path: Path) -> None:
    _corpus(tmp_path)
    store = _FakeStore(["memaaaa1", "membbbb1", "memcccc1"])
    mem = SimpleNamespace(cfg=SimpleNamespace(state_dir=tmp_path), store=store)

    res = outcome.reconcile_roi(mem, floor=0.6, cap=1.5)  # default weight 2.0
    assert res["updated"] == 3
    assert store.roi["membbbb1"] > store.roi["memaaaa1"] > store.roi["memcccc1"]

    res = outcome.reconcile_roi(mem, floor=0.6, cap=1.5, cited_weight=1.0)
    assert store.roi["membbbb1"] == store.roi["memaaaa1"]


def test_dead_weight_unaffected_by_cited_weight(tmp_path: Path, monkeypatch) -> None:
    # Dead-weight candidates have zero grounding, so the cited weight can never
    # change what gets flagged.
    for t in range(1, 6):
        _recall(tmp_path, "s", t, f"never used prompt {t}", "mem0dead")
    _recall(tmp_path, "s", 1, "useful one", "mem0live")
    _grounded(tmp_path, "s", 1, "mem0live", method="cited")

    mem = SimpleNamespace(
        cfg=SimpleNamespace(state_dir=tmp_path), store=_FakeStore(["mem0dead", "mem0live"])
    )
    baseline = {d["id"] for d in outcome.dead_weight(mem, min_surfaced=4)}
    monkeypatch.setenv("MEMO_OUTCOME_CITED_WEIGHT", "9.0")
    assert {d["id"] for d in outcome.dead_weight(mem, min_surfaced=4)} == baseline == {"mem0dead"}


# ---------------- gate: outcome ranking off → path untouched ----------------


def test_maintain_gate_off_skips_outcome_reconcile(tmp_path: Path, monkeypatch) -> None:
    from click.testing import CliRunner

    from memo.cli import cli

    calls: list[str] = []
    monkeypatch.setattr(
        "memo.outcome.compute_utilities",
        lambda *a, **k: calls.append("compute") or {"by_prefix": {}, "prior_mean": 0.5},
    )
    monkeypatch.setattr("memo.outcome.dead_weight", lambda *a, **k: calls.append("dead") or [])
    monkeypatch.setattr(
        "memo.outcome.reconcile_roi", lambda *a, **k: calls.append("roi") or {"updated": 0}
    )

    env = {
        "MEMO_CONFIG_FILE": str(tmp_path / "memo-config.toml"),
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_DATA_DIR": str(tmp_path / "data"),
        "MEMO_STATE_DIR": str(tmp_path / "state"),
        "MEMO_OUTCOME_CITED_WEIGHT": "5.0",
    }
    # Gate off → the outcome block (the only consumer of the cited weight)
    # never runs, no matter what the weight is set to.
    result = CliRunner().invoke(
        cli,
        ["maintain", "--dry-run", "--json"],
        env={**env, "MEMO_OUTCOME_RANKING_ENABLED": "0"},
    )
    assert result.exit_code == 0, result.output
    assert calls == []

    # Gate on → the same sentinels ARE reached (the gate is what skipped them).
    result = CliRunner().invoke(
        cli,
        ["maintain", "--dry-run", "--json"],
        env={**env, "MEMO_OUTCOME_RANKING_ENABLED": "1"},
    )
    assert result.exit_code == 0, result.output
    assert "compute" in calls and "dead" in calls
