"""Bounded Synapse feedback/eval absorption."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from memo.identity import PrincipalIdentity
from memo.memory import Memory
from memo.operational_epoch import CommitContext
from tools.memflow_absorption.synapse_data import (
    EvalFixture,
    FeedbackImport,
    SynapseDataBundle,
    SynapseDataError,
    _feedback_operation_key_state,
    apply_synapse_data,
    build_synapse_data_bundle,
    extract_synapse_eval_fixtures,
    extract_synapse_feedback,
)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


@pytest.fixture(autouse=True)
def authenticated_epoch_for_operational_receipts(mem_with_stub: Memory) -> None:
    """Exercise migration receipts under an explicit test epoch, never bypassing production guards."""
    identity = PrincipalIdentity(
        principal_id="test-synapse-data",
        actor_id="test-importer",
        kind="agent",
        device_id="test-device",
        session_id="test-session",
        source_client="pytest",
    )
    context = CommitContext(
        identity=identity,
        authority_epoch=0,
        control_oid="test-control",
        origin_device="test-device",
    )

    class _TestFence:
        def verify(self, observed: CommitContext) -> None:
            assert observed == context

    mem_with_stub.operational._context_provider = lambda: context
    mem_with_stub.operational.epoch_fence = _TestFence()


@pytest.fixture
def state_dir(tmp_path: Path) -> Path:
    root = tmp_path / "synapse-state"
    _write_jsonl(
        root / "ledger.jsonl",
        [
            {
                "action": "chat_feedback",
                "action_id": "already-seen",
                "trace_id": "trace-old",
                "metadata": {
                    "feedback_id": "already-seen",
                    "rating": "up",
                    "source_ids": ["aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"],
                },
            },
            {
                "action": "chat_feedback",
                "action_id": "new-feedback",
                "trace_id": "trace-new",
                "metadata": {
                    "feedback_id": "new-feedback",
                    "rating": "down",
                    "source_ids": ["bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"],
                    "arbitrary": {"answer": "must never escape"},
                },
            },
            {"action": "runtime_loop", "metadata": {"answer": "never import"}},
        ],
    )
    _write_jsonl(
        root / "observability" / "chat-traces.jsonl",
        [
            {
                "trace_id": "trace-old",
                "query": "old feedback query",
            },
            {
                "trace_id": "trace-new",
                "query": "  how   does  feedback work? ",
                "answer": "private answer that must not be copied",
            }
        ],
    )
    _write_jsonl(
        root / "observability" / "chat_pipeline_trace.jsonl",
        [{"trace_id": "trace-new", "query_preview": "not selected over full query"}],
    )
    (root / "eval").mkdir(parents=True)
    (root / "eval" / "corpus.json").write_text(
        json.dumps(
            [
                {
                    "id": "fixture-one",
                    "question": "where is the runbook?",
                    "expected_source_ids": ["bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"],
                    "answer": "private eval answer",
                    "needs_review": False,
                },
                {
                    "id": "not-high-signal",
                    "question": "discard me",
                    "expected_source_ids": ["cccccccccccccccccccccccccccccccc"],
                    "needs_review": True,
                },
            ]
        ),
        encoding="utf-8",
    )
    return root


def test_feedback_extraction_is_idempotent_and_does_not_copy_answers(state_dir: Path) -> None:
    bundle = extract_synapse_feedback(state_dir, seen_ids={"already-seen"})
    assert {item.feedback_id for item in bundle} == {"new-feedback"}
    assert all(item.answer == "" for item in bundle)
    assert bundle[0].query == "how does feedback work?"


def test_bundle_preserves_seen_and_duplicate_feedback_ids(state_dir: Path) -> None:
    ledger = state_dir / "ledger.jsonl"
    rows = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
    rows.append(rows[1])
    _write_jsonl(ledger, rows)

    bundle = build_synapse_data_bundle(state_dir, {"already-seen"})
    assert [item.feedback_id for item in bundle.feedback] == ["new-feedback"]
    assert bundle.skipped_feedback_ids == ("already-seen", "new-feedback")


def test_seen_ids_are_casefolded_and_invalid_values_are_ignored(state_dir: Path) -> None:
    bundle = build_synapse_data_bundle(state_dir, {"ALREADY-SEEN", "not an id!"})
    assert [item.feedback_id for item in bundle.feedback] == ["new-feedback"]
    assert bundle.skipped_feedback_ids == ("already-seen",)


def test_eval_extraction_keeps_fixture_metadata_only(state_dir: Path) -> None:
    fixture = extract_synapse_eval_fixtures(state_dir)[0]
    assert fixture.source_ids
    assert fixture.query
    assert fixture.answer == ""
    assert fixture.content_sha256


def test_apply_is_replay_safe_and_stages_eval_only(
    state_dir: Path, mem_with_stub: Memory
) -> None:
    record = mem_with_stub.save(content="known source", title="Source")
    # The fixture state uses this canonical source id; rewrite it to the real
    # test memory rather than allowing an orphan feedback signal.
    ledger = state_dir / "ledger.jsonl"
    rows = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
    rows[1]["metadata"]["source_ids"] = [record.id]
    _write_jsonl(ledger, rows)
    corpus = state_dir / "eval" / "corpus.json"
    fixture_rows = json.loads(corpus.read_text(encoding="utf-8"))
    fixture_rows[0]["expected_source_ids"] = [record.id]
    corpus.write_text(json.dumps(fixture_rows), encoding="utf-8")

    data = build_synapse_data_bundle(state_dir, {"already-seen"})
    first = apply_synapse_data(mem_with_stub, data, attempt_id="synapse-import-1")
    assert first.status == "applied"
    assert first.feedback_imported == 1
    assert first.feedback_skipped == 1
    assert first.skipped_feedback_ids == ("already-seen",)
    assert len(mem_with_stub.feedback_list(source_id=record.id)) == 1
    staging = mem_with_stub.cfg.state_dir / "operator-staging" / "synapse-eval-fixtures.json"
    staged = staging.read_text(encoding="utf-8")
    assert "private" not in staged
    assert "fixture-one" in staged

    replay = apply_synapse_data(mem_with_stub, data, attempt_id="synapse-import-1")
    assert replay.status == "reused"
    changed = SynapseDataBundle(data.feedback, data.eval_fixtures, "0" * 64)
    with pytest.raises(SynapseDataError, match="different input bundle"):
        apply_synapse_data(mem_with_stub, changed, attempt_id="synapse-import-1")


def test_invalid_fixture_leaves_no_partial_feedback_or_receipt(mem_with_stub: Memory) -> None:
    record = mem_with_stub.save(content="known source", title="Source")
    data = SynapseDataBundle(
        feedback=(FeedbackImport("feedback-one", record.id, "query", "up"),),
        eval_fixtures=(
            EvalFixture("fixture-one", "query", (record.id,), "f" * 64, answer="secret"),
        ),
        input_sha256="a" * 64,
    )
    with pytest.raises(SynapseDataError, match="non-redacted eval fixture"):
        apply_synapse_data(mem_with_stub, data, attempt_id="synapse-import-2")
    assert mem_with_stub.feedback_list(source_id=record.id) == []
    assert not [
        event
        for event in mem_with_stub.operational.ledger.validated_events()
        if event.op == "receipt.synapse-data"
    ]


def test_late_feedback_failure_rolls_back_all_imported_feedback(mem_with_stub: Memory, monkeypatch) -> None:
    first = mem_with_stub.save(content="first source", title="First")
    second = mem_with_stub.save(content="second source", title="Second")
    data = SynapseDataBundle(
        feedback=(
            FeedbackImport("feedback-one", first.id, "first query", "up"),
            FeedbackImport("feedback-two", second.id, "second query", "down"),
        ),
        eval_fixtures=(),
        input_sha256="b" * 64,
    )
    original = mem_with_stub.feedback_record
    calls = 0

    def fail_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        result = original(*args, **kwargs)
        if calls == 2:
            raise RuntimeError("late feedback failure after write")
        return result

    monkeypatch.setattr(mem_with_stub, "feedback_record", fail_second)
    with pytest.raises(RuntimeError, match="late feedback failure after write"):
        apply_synapse_data(mem_with_stub, data, attempt_id="synapse-import-3")
    assert mem_with_stub.feedback_list(source_id=first.id) == []
    assert mem_with_stub.feedback_list(source_id=second.id) == []
    assert not [
        event
        for event in mem_with_stub.operational.ledger.validated_events()
        if event.op == "receipt.synapse-data"
    ]


def test_receipt_failure_rolls_back_imported_feedback(mem_with_stub: Memory, monkeypatch) -> None:
    record = mem_with_stub.save(content="known source", title="Source")
    data = SynapseDataBundle(
        feedback=(FeedbackImport("feedback-one", record.id, "query", "up"),),
        eval_fixtures=(),
        input_sha256="c" * 64,
    )

    def fail_receipt(*args, **kwargs):
        raise RuntimeError("receipt failure")

    monkeypatch.setattr(mem_with_stub.operational, "receipt", fail_receipt)
    with pytest.raises(RuntimeError, match="receipt failure"):
        apply_synapse_data(mem_with_stub, data, attempt_id="synapse-import-4")
    assert mem_with_stub.feedback_list(source_id=record.id) == []


def test_feedback_ownership_lookup_is_direct_not_limited_to_500(mem_with_stub: Memory) -> None:
    record = mem_with_stub.save(content="known source", title="Source")
    target_id = "f" * 64
    for index in range(501):
        mem_with_stub.feedback_record(
            record.id,
            query_text=f"query {index}",
            rating="up",
            feedback_id=target_id if index == 0 else f"{index:064x}",
            extra={"synapse_operation_key": target_id},
        )
    assert _feedback_operation_key_state(mem_with_stub, target_id, target_id) == "owned"


def test_caller_skipped_ids_are_canonicalized_before_receipt(mem_with_stub: Memory) -> None:
    data = SynapseDataBundle(
        feedback=(),
        eval_fixtures=(),
        input_sha256="d" * 64,
        skipped_feedback_ids=("FEEDBACK-ONE", "feedback-one"),
    )
    receipt = apply_synapse_data(mem_with_stub, data, attempt_id="synapse-import-5")
    assert receipt.feedback_skipped == 1
    assert receipt.skipped_feedback_ids == ("feedback-one",)


def test_dynamic_skipped_feedback_id_is_casefolded_before_receipt(mem_with_stub: Memory) -> None:
    record = mem_with_stub.save(content="known source", title="Source")
    mem_with_stub.feedback_record(record.id, query_text="already exists", rating="up")
    data = SynapseDataBundle(
        feedback=(FeedbackImport("FEEDBACK-DYNAMIC", record.id, "already exists", "up"),),
        eval_fixtures=(),
        input_sha256="f" * 64,
    )
    receipt = apply_synapse_data(mem_with_stub, data, attempt_id="synapse-import-7")
    assert receipt.feedback_imported == 0
    assert receipt.feedback_skipped == 1
    assert receipt.skipped_feedback_ids == ("feedback-dynamic",)


def test_invalid_caller_skipped_id_is_rejected_before_receipt(mem_with_stub: Memory) -> None:
    data = SynapseDataBundle(
        feedback=(),
        eval_fixtures=(),
        input_sha256="e" * 64,
        skipped_feedback_ids=("invalid id!",),
    )
    with pytest.raises(SynapseDataError, match="skipped_feedback_ids contains an invalid ID"):
        apply_synapse_data(mem_with_stub, data, attempt_id="synapse-import-6")
    assert not [
        event
        for event in mem_with_stub.operational.ledger.validated_events()
        if event.op == "receipt.synapse-data"
    ]
