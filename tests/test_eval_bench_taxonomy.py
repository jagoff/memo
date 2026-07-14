"""Capability-taxonomy rollup — pure mapping + arithmetic (no MLX, no store).

Covers the Memoria-style 6-bucket auxiliary view added on top of the raw
per-category benchmark numbers: bucket mapping, abstention routing, the
weighted retrieval rollup, and the first-class abstention/hallucination metric.
"""

from __future__ import annotations

from memo import eval_bench
from memo import eval_bench_taxonomy as tax
from memo.eval_bench import QAResult

# --- bucket_for ---------------------------------------------------------------


def test_locomo_categories_map_to_buckets():
    assert tax.bucket_for("single_hop") == "single_session_grounding"
    assert tax.bucket_for("multi_hop") == "multi_session_synthesis"
    assert tax.bucket_for("temporal_reasoning") == "temporal_state_tracking"
    assert tax.bucket_for("adversarial") == "abstention_constraint"


def test_longmemeval_categories_map_to_buckets():
    # LongMemEval question_types arrive hyphenated; parser normalizes to "_".
    assert tax.bucket_for("single_session_preference") == "preference_understanding"
    assert tax.bucket_for("single-session-user") == "single_session_grounding"
    assert tax.bucket_for("knowledge_update") == "knowledge_update_conflict"
    assert tax.bucket_for("multi_session") == "multi_session_synthesis"


def test_abstention_flag_overrides_topic_category():
    # An abstention question keeps its topic category but tests the *ability*
    # to decline, so it routes to the abstention bucket regardless.
    assert tax.bucket_for("single_session_user", abstention=True) == "abstention_constraint"
    assert tax.bucket_for("temporal_reasoning", abstention=True) == "abstention_constraint"


def test_unknown_category_falls_to_other_not_grounding():
    assert tax.bucket_for("totally_made_up") == tax.OTHER_BUCKET
    assert tax.OTHER_BUCKET not in tax.CAPABILITY_BUCKETS


# --- rollup_weighted ----------------------------------------------------------


def test_rollup_is_question_count_weighted():
    # Two grounding categories with different weights → weighted mean, not
    # simple mean. single_hop: recall 0.9 (n=9); open_domain: recall 0.1 (n=1)
    # → (0.9*9 + 0.1*1)/10 = 0.82.
    by_cat = {
        "single_hop": {"recall_at_k": 0.9, "n_questions": 9},
        "open_domain": {"recall_at_k": 0.1, "n_questions": 1},
        "temporal_reasoning": {"recall_at_k": 0.5, "n_questions": 4},
    }
    rolled = tax.rollup_weighted(by_cat, ("recall_at_k",))
    assert rolled["single_session_grounding"]["recall_at_k"] == 0.82
    assert rolled["single_session_grounding"]["n_questions"] == 10
    assert rolled["temporal_state_tracking"]["recall_at_k"] == 0.5


def test_rollup_skips_zero_weight_categories():
    by_cat = {
        "single_hop": {"recall_at_k": 0.9, "n_questions": 0},  # nothing scored
    }
    assert tax.rollup_weighted(by_cat, ("recall_at_k",)) == {}


# --- capability_qa + abstention_summary (over QAResult) -----------------------


def _qa(cat: str, *, abstention: bool, correct: bool) -> QAResult:
    return QAResult(
        qa_id=f"{cat}-x", category=cat, abstention=abstention, correct=correct, answer_head=""
    )


def test_capability_qa_routes_abstention_to_its_bucket():
    results = [
        _qa("single_hop", abstention=False, correct=True),
        _qa("single_hop", abstention=False, correct=False),
        _qa("temporal_reasoning", abstention=True, correct=True),  # declined correctly
    ]
    by_bucket = eval_bench.capability_qa(results)
    assert by_bucket["single_session_grounding"]["accuracy"] == 0.5
    assert by_bucket["single_session_grounding"]["n_questions"] == 2
    # The abstention question lands in the abstention bucket, NOT temporal.
    assert by_bucket["abstention_constraint"]["accuracy"] == 1.0
    assert "temporal_state_tracking" not in by_bucket


def test_abstention_summary_rates():
    results = [
        _qa("adversarial", abstention=True, correct=True),  # declined
        _qa("adversarial", abstention=True, correct=False),  # hallucinated
        _qa("adversarial", abstention=True, correct=False),  # hallucinated
        _qa("single_hop", abstention=False, correct=True),  # ignored (not abstention)
    ]
    summ = eval_bench.abstention_summary(results)
    assert summ["n_questions"] == 3
    assert summ["correct_abstentions"] == 1
    assert summ["hallucinations"] == 2
    assert summ["abstention_accuracy"] == 0.333
    assert summ["hallucination_rate"] == 0.667


def test_abstention_summary_empty_is_zero_not_error():
    summ = eval_bench.abstention_summary([_qa("single_hop", abstention=False, correct=True)])
    assert summ["n_questions"] == 0
    assert summ["hallucination_rate"] == 0.0


def test_capability_rows_render_in_report():
    receipt = {
        "schema": eval_bench.RECEIPT_SCHEMA,
        "dataset": "locomo",
        "k": 5,
        "capability_retrieval": {
            "single_session_grounding": {"recall_at_k": 0.82, "n_questions": 10}
        },
        "capability_qa": {"abstention_constraint": {"accuracy": 0.5, "n_questions": 4}},
        "abstention": {"abstention_accuracy": 0.5, "hallucination_rate": 0.5, "n_questions": 4},
    }
    md = eval_bench.render_report([{**receipt, "_file": "r.json"}])
    assert "capability/single_session_grounding/recall_at_k" in md
    assert "capability_qa/abstention_constraint/accuracy" in md
    assert "abstention/hallucination_rate" in md
