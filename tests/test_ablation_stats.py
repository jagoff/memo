"""With-vs-without-recall cohort stats (MEMO_RECALL_DISABLE ablation)."""

from __future__ import annotations

from memo.dashboard import append_grounding_log, append_recall_log
from memo.dashboard_metrics import ablation_stats


def _row(sd, *, sid, turn, via, prompt) -> None:
    append_recall_log(sd, prompt=prompt, hits=[], via=via, session_id=sid, turn=turn)


def test_ablation_stats_counts_and_grounding(tmp_path) -> None:
    _row(tmp_path, sid="on1", turn=1, via="subprocess", prompt="cómo va el sync remoto?")
    _row(tmp_path, sid="on1", turn=2, via="daemon", prompt="qué decidimos del overlay?")
    _row(tmp_path, sid="off1", turn=1, via="disabled", prompt="cómo va el sync remoto?")
    append_grounding_log(tmp_path, session_id="on1", turn=1,
                         recall_id="aaaabbbb", used_score=0.9, method="lexical")
    s = ablation_stats(tmp_path)
    assert s["turns_on"] == 2 and s["turns_off"] == 1
    assert s["grounded_turns_on"] == 1
    assert s["grounded_per_turn_on"] == 0.5


def test_ablation_reask_rate_per_cohort(tmp_path) -> None:
    # off cohort re-asks the same question 2 turns later; on cohort doesn't.
    _row(tmp_path, sid="off1", turn=1, via="disabled", prompt="dónde vive el registro de flags?")
    _row(tmp_path, sid="off1", turn=3, via="disabled", prompt="dónde vive el registro de flags de memo?")
    _row(tmp_path, sid="on1", turn=1, via="subprocess", prompt="dónde vive el registro de flags?")
    _row(tmp_path, sid="on1", turn=3, via="subprocess", prompt="agregá un test para el ledger")
    s = ablation_stats(tmp_path)
    assert s["reask_rate_off"] == 0.5
    assert s["reask_rate_on"] == 0.0


def test_compute_roi_exposes_ablation(tmp_path) -> None:
    from memo.cli_roi import compute_roi

    _row(tmp_path, sid="off1", turn=1, via="disabled", prompt="cómo va el sync remoto?")
    data = compute_roi(tmp_path)
    assert data["ablation"]["turns_off"] == 1
