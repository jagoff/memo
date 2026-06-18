"""The Outcome Loop — utility from grounding, roi reconcile, gaps, dead weight."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from memo import dashboard, outcome


def _recall(tmp: Path, sid: str, turn: int, prompt: str, mem_id: str, *, via: str = "subprocess",
            reason: str | None = None, hit: bool = True) -> None:
    hits = [{"id": mem_id, "score": 0.8, "title": "t"}] if hit else []
    dashboard.append_recall_log(
        tmp, prompt=prompt, via=via, session_id=sid, turn=turn, client="claude-code",
        hits=hits, reason=reason,
    )


def _grounded(tmp: Path, sid: str, turn: int, mem_id: str, score: float = 0.9) -> None:
    dashboard.append_grounding_log(
        tmp, session_id=sid, turn=turn, recall_id=mem_id, used_score=score, method="lexical",
    )


# ---------------- utility ----------------

def test_compute_utilities_rewards_grounded_over_surfaced(tmp_path: Path) -> None:
    # mem00001: surfaced 2 turns, grounded both → high utility
    _recall(tmp_path, "s", 1, "decision about deploy yaml pipeline", "mem00001")
    _grounded(tmp_path, "s", 1, "mem00001")
    _recall(tmp_path, "s", 2, "deploy pipeline config again", "mem00001")
    _grounded(tmp_path, "s", 2, "mem00001")
    # mem00002: surfaced 3 turns, never grounded → low utility
    for t in (1, 2, 3):
        _recall(tmp_path, "s", t, f"prompt {t}", "mem00002")

    u = outcome.compute_utilities(tmp_path, prior_n=3.0)
    by = u["by_prefix"]
    assert by["mem00001"]["surfaced"] == 2 and by["mem00001"]["grounded"] == 2
    assert by["mem00002"]["surfaced"] == 3 and by["mem00002"]["grounded"] == 0
    assert by["mem00001"]["utility"] > by["mem00002"]["utility"]


def test_compute_utilities_uses_strong_grounding_decision(tmp_path: Path) -> None:
    _recall(tmp_path, "s", 1, "same topic but not actually used", "mem00001")
    _grounded(tmp_path, "s", 1, "mem00001", score=0.72)
    _recall(tmp_path, "s", 2, "paraphrased useful recall", "mem00002")
    dashboard.append_grounding_log(
        tmp_path,
        session_id="s",
        turn=2,
        recall_id="mem00002",
        used_score=0.5,
        specific_score=0.12,
        method="both",
    )

    u = outcome.compute_utilities(tmp_path, prior_n=0.0)
    by = u["by_prefix"]
    assert by["mem00001"]["grounded"] == 0
    assert by["mem00002"]["grounded"] == 1


# ---------------- roi reconcile ----------------

class _FakeStore:
    def __init__(self, ids: list[str]) -> None:
        self._ids = ids
        self.roi: dict[str, float] = {}

    def all_ids(self) -> list[str]:
        return list(self._ids)

    def set_roi_batch(self, pairs, *, floor: float, cap: float) -> int:
        self.roi = {i: max(floor, min(cap, r)) for i, r in pairs}
        return len(pairs)


def test_reconcile_roi_promotes_grounded_demotes_dead(tmp_path: Path) -> None:
    _recall(tmp_path, "s", 1, "deploy yaml pipeline decision", "mem00001")
    _grounded(tmp_path, "s", 1, "mem00001")
    for t in (1, 2, 3, 4):
        _recall(tmp_path, "s", t, f"noise prompt number {t}", "mem00002")  # never grounded

    store = _FakeStore(["mem00001", "mem00002"])
    mem = SimpleNamespace(cfg=SimpleNamespace(state_dir=tmp_path), store=store)
    res = outcome.reconcile_roi(mem, floor=0.6, cap=1.5)
    assert res["updated"] == 2
    assert store.roi["mem00001"] > store.roi["mem00002"]
    assert store.roi["mem00002"] >= 0.6  # demoted but floored, never zeroed


# ---------------- dead weight ----------------

def test_dead_weight_flags_surfaced_never_grounded(tmp_path: Path) -> None:
    for t in range(1, 6):  # surfaced 5×, never grounded
        _recall(tmp_path, "s", t, f"never used prompt {t}", "mem0dead")
    _recall(tmp_path, "s", 1, "useful one", "mem0live")
    _grounded(tmp_path, "s", 1, "mem0live")

    mem = SimpleNamespace(cfg=SimpleNamespace(state_dir=tmp_path), store=_FakeStore(["mem0dead", "mem0live"]))
    dead = outcome.dead_weight(mem, min_surfaced=4)
    ids = {d["id"] for d in dead}
    assert "mem0dead" in ids
    assert "mem0live" not in ids
    # min_surfaced=0 disables
    assert outcome.dead_weight(mem, min_surfaced=0) == []


def test_dead_weight_disabled_when_measurement_coverage_is_zero(tmp_path: Path) -> None:
    for t in range(1, 6):
        _recall(tmp_path, "s", t, f"never measured prompt {t}", "mem0dead")

    mem = SimpleNamespace(cfg=SimpleNamespace(state_dir=tmp_path), store=_FakeStore(["mem0dead"]))

    assert outcome.dead_weight(mem, min_surfaced=4) == []


# ---------------- gaps ----------------

def test_detect_gaps_clusters_and_excludes_slash_and_answered(tmp_path: Path) -> None:
    # gap: knowledge prompt, no-match bail
    _recall(tmp_path, "s", 1, "como configuro el sync loop de memflow exactamente",
            "x", via="bail", reason="no hits above min_sim", hit=False)
    # near-dup of the above → same cluster
    _recall(tmp_path, "s", 2, "como configuro exactamente el sync loop de memflow",
            "x", via="bail", reason="no hits above min_sim", hit=False)
    # NOT a gap: slash command bail
    _recall(tmp_path, "s", 3, "/status", "x", via="bail", reason="slash command", hit=False)
    # NOT a gap: answered + grounded
    _recall(tmp_path, "s", 4, "que decidimos sobre el embedder de memo", "mem00001")
    _grounded(tmp_path, "s", 4, "mem00001")

    gaps = outcome.detect_gaps(tmp_path, sim_threshold=0.5)
    assert len(gaps) == 1, gaps  # the two memflow prompts cluster into one
    assert gaps[0]["count"] == 2
    assert "sync loop" in gaps[0]["prompt"]


def test_detect_gaps_drops_injected_system_noise(tmp_path: Path) -> None:
    # injected hook/tool blobs land in recall.log but are not user questions
    _recall(tmp_path, "s", 1, "<task-notification>\n<task-id>abc</task-id> something long here that exceeds sixty chars easily",
            "x", via="bail", reason="no hits above min_sim", hit=False)
    _recall(tmp_path, "s", 2, "system-reminder: the following context may be relevant to your task here",
            "x", via="bail", reason="no hits above min_sim", hit=False)
    assert outcome.detect_gaps(tmp_path) == []
