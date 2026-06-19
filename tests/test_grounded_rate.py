"""grounded_rate: honest denominator (only grounding-scored turns count).

Regression for the dashboard utility metric reading ~15% when the true rate was
~100%: surfaced memorias on turns the Stop hook never scored were counted as
misses. They must be excluded (reported as coverage), not counted against the
rate.
"""
from __future__ import annotations

from pathlib import Path

from memo.dashboard import append_grounding_log, append_recall_log
from memo.dashboard_metrics import grounded_rate, grounding_used, verdict


def _surface(state_dir: Path, sid: str, turn: int, ids: list[str]) -> None:
    append_recall_log(
        state_dir,
        prompt=f"q{turn}",
        hits=[{"id": i, "title": i, "score": 0.9, "snippet": i} for i in ids],
        via="daemon",
        session_id=sid,
        turn=turn,
    )


def test_unscored_turns_are_not_counted_as_misses(tmp_path: Path):
    sd = tmp_path
    # Turn 1 was scored AND used. Turn 2 surfaced but never scored (no Stop hook).
    _surface(sd, "s1", 1, ["aaaa1111", "bbbb2222"])
    _surface(sd, "s1", 2, ["cccc3333", "dddd4444"])
    append_grounding_log(sd, session_id="s1", turn=1, recall_id="aaaa1111", used_score=0.9, method="lexical")
    append_grounding_log(sd, session_id="s1", turn=1, recall_id="bbbb2222", used_score=0.8, method="lexical")

    g = grounded_rate(sd)

    # Only turn 1 (2 memorias) is measured; both used -> 100%, NOT 2/4=50%.
    assert g["grounded_rate"] == 1.0
    assert g["surfaced"] == 2
    assert g["grounded"] == 2
    assert g["surfaced_turns"] == 2
    assert g["measured_turns"] == 1
    assert g["measurement_coverage"] == 0.5
    # Turn 2's 2 surfaced memorias are unmeasured, not misses.
    assert g["unmeasured_surfaced"] == 2


def test_per_answer_rate(tmp_path: Path):
    sd = tmp_path
    _surface(sd, "s1", 1, ["aaaa1111"])        # scored + used
    _surface(sd, "s1", 2, ["bbbb2222"])        # scored, NOT used (below bar)
    append_grounding_log(sd, session_id="s1", turn=1, recall_id="aaaa1111", used_score=0.9, method="lexical")
    append_grounding_log(sd, session_id="s1", turn=2, recall_id="bbbb2222", used_score=0.2, method="lexical")

    g = grounded_rate(sd)

    assert g["answers_total"] == 2          # both turns were scored
    assert g["answers_grounded"] == 1       # only turn 1 used a memoria
    assert g["answer_rate"] == 0.5


def test_topical_overlap_below_strong_bar_not_grounded(tmp_path: Path):
    sd = tmp_path
    _surface(sd, "s1", 1, ["aaaa1111"])
    # 0.72 = typical same-topic cosine (topical overlap, not real use): must NOT
    # count, since the utility bar is USED_SCORE_STRONG=0.8.
    append_grounding_log(sd, session_id="s1", turn=1, recall_id="aaaa1111", used_score=0.72, method="both")

    g = grounded_rate(sd)
    assert g["surfaced"] == 1       # measured (turn was scored)
    assert g["grounded"] == 0       # topical overlap alone is not "used"
    assert g["grounded_rate"] == 0.0


def test_downstream_action_counts_even_with_low_score(tmp_path: Path):
    sd = tmp_path
    _surface(sd, "s1", 1, ["aaaa1111"])
    # Weak text overlap, but the turn acted on what the memoria named -> used.
    append_grounding_log(
        sd, session_id="s1", turn=1, recall_id="aaaa1111", used_score=0.3,
        method="lexical", downstream_action="opened_file", action_evidence="foo.py",
    )

    g = grounded_rate(sd)
    assert g["grounded"] == 1
    assert g["grounded_rate"] == 1.0


def test_no_measured_turns_returns_none(tmp_path: Path):
    sd = tmp_path
    _surface(sd, "s1", 1, ["aaaa1111"])  # surfaced but never scored
    g = grounded_rate(sd)
    assert g["grounded_rate"] is None
    assert g["answer_rate"] is None
    assert g["unmeasured_surfaced"] == 1


def test_specific_score_recovers_paraphrase(tmp_path: Path):
    sd = tmp_path
    _surface(sd, "s1", 1, ["aaaa1111"])
    # Modest absolute match (0.5 < 0.8 bar) but clearly above the question's
    # topical baseline -> real (paraphrased) use, must count.
    append_grounding_log(
        sd, session_id="s1", turn=1, recall_id="aaaa1111", used_score=0.5,
        method="both", specific_score=0.12,
    )
    g = grounded_rate(sd)
    assert g["grounded"] == 1
    assert g["grounded_rate"] == 1.0


def test_grounding_used_helper_is_single_source_of_truth() -> None:
    assert grounding_used({"used_score": 0.85})
    assert grounding_used({"used_score": 0.5, "specific_score": 0.1})
    assert grounding_used({"used_score": 0.2, "downstream_action": "opened_file"})
    assert not grounding_used({"used_score": 0.72})
    assert not grounding_used({"used_score": 0.5, "specific_score": 0.01})


def test_verdict_unmeasured_when_reads_exist_without_grounding(tmp_path: Path) -> None:
    for turn in range(1, 21):
        _surface(tmp_path, "s1", turn, [f"mem{turn:05d}"])

    v = verdict(tmp_path, limit=500)

    assert v["status"] == "unmeasured"
    assert "NO SE MIDE" in v["label"]
    assert v["measurement_coverage"] == 0.0


def test_verdict_weak_only_after_enough_measured_turns(tmp_path: Path) -> None:
    for turn in range(1, 21):
        _surface(tmp_path, "s1", turn, [f"mem{turn:05d}"])
        if turn <= 5:
            append_grounding_log(
                tmp_path,
                session_id="s1",
                turn=turn,
                recall_id=f"mem{turn:05d}",
                used_score=0.2,
                method="lexical",
            )

    v = verdict(tmp_path, limit=500)

    assert v["status"] == "weak"
    assert v["measured_turns"] == 5


def test_verdict_weak_with_sparse_but_nonzero_coverage(tmp_path: Path) -> None:
    for turn in range(1, 21):
        _surface(tmp_path, "s1", turn, [f"mem{turn:05d}"])
        if turn <= 3:
            append_grounding_log(
                tmp_path,
                session_id="s1",
                turn=turn,
                recall_id=f"mem{turn:05d}",
                used_score=0.2,
                method="lexical",
            )

    v = verdict(tmp_path, limit=500)

    assert v["status"] == "weak"
    assert v["measurement_coverage"] == 0.15


def test_verdict_weak_when_only_one_turn_is_measured(tmp_path: Path) -> None:
    for turn in range(1, 21):
        _surface(tmp_path, "s1", turn, [f"mem{turn:05d}"])
    append_grounding_log(
        tmp_path,
        session_id="s1",
        turn=1,
        recall_id="mem00001",
        used_score=0.2,
        method="lexical",
    )

    v = verdict(tmp_path, limit=500)

    assert v["status"] == "weak"
    assert v["measurement_coverage"] == 0.05


def test_verdict_ok_when_measured_use_crosses_threshold(tmp_path: Path) -> None:
    for turn in range(1, 21):
        _surface(tmp_path, "s1", turn, [f"mem{turn:05d}"])
        if turn <= 5:
            append_grounding_log(
                tmp_path,
                session_id="s1",
                turn=turn,
                recall_id=f"mem{turn:05d}",
                used_score=0.9 if turn == 1 else 0.2,
                method="lexical",
            )

    v = verdict(tmp_path, limit=500)

    assert v["status"] == "ok"
    assert v["grounded_rate"] == 0.2


def test_knowledge_segmentation(tmp_path: Path):
    sd = tmp_path
    # Knowledge-seeking prompt (question) that used memo.
    append_recall_log(
        sd, prompt="¿qué decidimos sobre el backend de auth?",
        hits=[{"id": "kkkk1111", "title": "x", "score": 0.9, "snippet": "x"}],
        via="daemon", session_id="s1", turn=1,
    )
    # Mechanical prompt that did NOT use memo (excluded from knowledge rate).
    append_recall_log(
        sd, prompt="arregla el typo en cli.py",
        hits=[{"id": "mmmm2222", "title": "y", "score": 0.9, "snippet": "y"}],
        via="daemon", session_id="s1", turn=2,
    )
    append_grounding_log(sd, session_id="s1", turn=1, recall_id="kkkk1111", used_score=0.9, method="both")
    append_grounding_log(sd, session_id="s1", turn=2, recall_id="mmmm2222", used_score=0.3, method="both")

    g = grounded_rate(sd)
    # Overall: 1/2 answers. Knowledge-only: 1/1 (mechanical turn excluded).
    assert g["answer_rate"] == 0.5
    assert g["answers_knowledge_total"] == 1
    assert g["answer_rate_knowledge"] == 1.0
