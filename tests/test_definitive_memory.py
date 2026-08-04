"""End-to-end contracts for outcome learning and secure federation."""

from __future__ import annotations

import hashlib
import hmac
import json
from types import SimpleNamespace

import pytest

from memo.config import Config
from memo.contracts import AnswerStatus
from memo.errors import FederationError, MemoError, NotFoundError
from memo.memory import Memory
from memo.memory.record import MemoryRecord
from memo.operation_ledger import LedgerIntegrityError, OperationLedger


def _evidence_hit(
    memory_id: str,
    *,
    title: str,
    body: str,
    score: float,
    trust_tier: str = "agent_inferred",
) -> SimpleNamespace:
    return SimpleNamespace(
        id=memory_id,
        title=title,
        body=body,
        score=score,
        extra={"trust_tier": trust_tier},
        type="note",
        valid_at=None,
        invalid_at=None,
    )


def test_evidence_pack_weak_top_hit_abstains_without_relative_normalization(
    mem_with_stub,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        mem_with_stub,
        "search",
        lambda *_args, **_kwargs: [
            _evidence_hit(
                "weak-hit",
                title="Launch notes",
                body="Launch notes for the kitchen team.",
                score=0.05,
                trust_tier="agent_verified",
            )
        ],
    )

    pack = mem_with_stub.evidence_pack(
        "What is the launch sequence for the Europa probe?",
        min_coverage=0.2,
    )

    assert pack.coverage == 0.25
    assert pack.items[0].score == pytest.approx(0.12)
    assert pack.status is AnswerStatus.INSUFFICIENT_EVIDENCE
    assert "confidence" in pack.abstention_reason


def test_evidence_pack_filters_question_stopwords_for_coverage(
    mem_with_stub,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        mem_with_stub,
        "search",
        lambda *_args, **_kwargs: [
            _evidence_hit(
                "strong-hit",
                title="Invoice retry policy",
                body="The service retries every failed invoice.",
                score=0.2,
                trust_tier="human",
            )
        ],
    )

    pack = mem_with_stub.evidence_pack(
        "What is the invoice retry policy for the service?",
    )

    assert pack.coverage == 1.0
    assert pack.items[0].score == pytest.approx(0.48)
    assert pack.confidence >= 0.4
    assert pack.status is AnswerStatus.ANSWERED


def test_evidence_pack_scopes_relation_lookup_to_selected_memories(
    mem_with_stub,
    monkeypatch,
) -> None:
    hits = [
        _evidence_hit("alpha-id", title="Alpha", body="Alpha evidence.", score=0.4),
        _evidence_hit("beta-id", title="Beta", body="Beta evidence.", score=0.4),
    ]
    monkeypatch.setattr(mem_with_stub, "search", lambda *_args, **_kwargs: hits)
    request: dict[str, object] = {}

    def _relations(**kwargs):
        request.update(kwargs)
        return [
            {
                "source_id": "alpha-id",
                "target_id": "beta-id",
                "relation": "conflicts_with",
            }
        ]

    monkeypatch.setattr(mem_with_stub.store, "list_relations", _relations)

    pack = mem_with_stub.evidence_pack("Does alpha conflict with beta?", min_coverage=0.1)

    assert request["status"] == "judged"
    assert request["memory_ids"] == ["alpha-id", "beta-id"]
    assert pack.status is AnswerStatus.CONFLICTED


def _graph_hit(hid: str, *, title: str, body: str, score: float) -> MemoryRecord:
    return MemoryRecord(
        id=hid,
        path=f"{hid}.md",
        title=title,
        type="note",
        tags=[],
        created="2026-08-04T00:00:00",
        updated="2026-08-04T00:00:00",
        body=body,
        score=score,
    )


class _RareEntityGraph:
    def memory_entities(self, memory_id):
        if memory_id in ("hit-a", "hit-b"):
            return [{"name": "invoice-retry-policy", "type": "topic", "mention_count": 1}]
        return []

    def total_indexed_memories(self):
        return 10

    def entity_doc_freqs(self, names):
        return {"invoice-retry-policy": 1.0} if "invoice-retry-policy" in names else {}


def test_evidence_pack_graph_compact_noop_when_disabled(mem_with_stub, monkeypatch) -> None:
    monkeypatch.setenv("MEMO_EVIDENCE_GRAPH_COMPACT", "0")
    hits = [
        _graph_hit("hit-a", title="Retry policy", body="Standard retry policy.", score=1.0),
        _graph_hit(
            "hit-b", title="Backoff schedule", body="Exponential backoff schedule.", score=0.9
        ),
    ]
    monkeypatch.setattr(mem_with_stub, "search", lambda *_a, **_kw: hits)
    monkeypatch.setattr(mem_with_stub, "graph", _RareEntityGraph())

    pack = mem_with_stub.evidence_pack("retry backoff policy")

    assert {item.id for item in pack.items} == {"hit-a", "hit-b"}
    assert all(not item.provenance.get("related_ids") for item in pack.items)


def test_evidence_pack_graph_compact_collapses_and_cites_absorbed_hit(
    mem_with_stub, monkeypatch
) -> None:
    monkeypatch.setenv("MEMO_EVIDENCE_GRAPH_COMPACT", "1")
    hits = [
        _graph_hit("hit-a", title="Retry policy", body="Standard retry policy.", score=1.0),
        _graph_hit(
            "hit-b", title="Backoff schedule", body="Exponential backoff schedule.", score=0.9
        ),
    ]
    monkeypatch.setattr(mem_with_stub, "search", lambda *_a, **_kw: hits)
    monkeypatch.setattr(mem_with_stub, "graph", _RareEntityGraph())

    pack = mem_with_stub.evidence_pack("retry backoff policy")

    assert len(pack.items) == 1
    assert pack.items[0].id == "hit-a"
    assert pack.items[0].provenance["related_ids"] == [("hit-b", "Backoff schedule")]


def test_evidence_pack_graph_compact_credits_absorbed_coverage(mem_with_stub, monkeypatch) -> None:
    # "backoff" appears only in hit-b's text. Without crediting hit-b's tokens
    # after it's absorbed into hit-a, coverage drops from 1.0 to 0.75 and a
    # min_coverage=0.8 gate would wrongly abstain.
    monkeypatch.setenv("MEMO_EVIDENCE_GRAPH_COMPACT", "1")
    hits = [
        _graph_hit(
            "hit-a",
            title="Retry policy",
            body="Standard retry policy for invoices.",
            score=1.0,
        ),
        _graph_hit(
            "hit-b",
            title="Backoff schedule",
            body="Exponential backoff schedule applies here.",
            score=0.9,
        ),
    ]
    monkeypatch.setattr(mem_with_stub, "search", lambda *_a, **_kw: hits)
    monkeypatch.setattr(mem_with_stub, "graph", _RareEntityGraph())

    pack = mem_with_stub.evidence_pack("retry backoff policy invoices", min_coverage=0.8)

    assert len(pack.items) == 1
    assert pack.coverage == 1.0
    assert pack.status is AnswerStatus.ANSWERED


def test_outcomes_promote_useful_memory_and_are_idempotent(mem_with_stub) -> None:
    record = mem_with_stub.save(
        content="Run the focused tests before the full suite.",
        title="Verification order",
        type_="decision",
        auto_project=False,
    )

    first = mem_with_stub.record_task_outcome(
        task_id="task-1",
        status="success",
        memory_ids=[record.id],
        idempotency_key="outcome-1",
    )
    second = mem_with_stub.record_task_outcome(
        task_id="task-2",
        status="success",
        memory_ids=[record.id],
        idempotency_key="outcome-2",
    )
    replay = mem_with_stub.record_task_outcome(
        task_id="task-2",
        status="success",
        memory_ids=[record.id],
        idempotency_key="outcome-2",
    )

    updated = mem_with_stub.get(record.id)
    assert first["updated_memory_ids"] == [record.id]
    assert second["updated_memory_ids"] == [record.id]
    assert replay["idempotent_replay"] is True
    assert updated is not None
    assert updated.extra["outcome_stats"]["total"] == 2
    assert updated.extra["outcome_stats"]["utility"] == 1.0
    assert updated.extra["priority"] == "high"
    assert updated.extra["trust_tier"] == "agent_inferred"
    assert mem_with_stub.procedure_candidates()[0]["id"] == record.id


def test_outcome_idempotency_compares_the_complete_semantic_payload(
    mem_with_stub,
) -> None:
    first = mem_with_stub.save(
        content="First cited memory.",
        title="First citation",
        auto_project=False,
    )
    second = mem_with_stub.save(
        content="Second cited memory.",
        title="Second citation",
        auto_project=False,
    )
    base = {
        "task_id": "semantic-outcome",
        "status": "success",
        "memory_ids": [first.id, second.id],
        "actor_id": "agent-alpha",
        "artifacts": ["artifact://report"],
        "environment": {"runner": "local", "attempt": 1},
        "idempotency_key": "semantic-key",
    }
    mem_with_stub.record_task_outcome(**base)

    replay = mem_with_stub.record_task_outcome(
        **{
            **base,
            "memory_ids": [f" {second.id} ", first.id, second.id],
        }
    )
    assert replay["idempotent_replay"] is True

    mutations = [
        {"task_id": "other-task"},
        {"status": "partial"},
        {"memory_ids": [first.id]},
        {"actor_id": "agent-beta"},
        {"artifacts": ["artifact://different"]},
        {"environment": {"runner": "remote", "attempt": 1}},
    ]
    for mutation in mutations:
        with pytest.raises(ValueError, match="different outcome payload"):
            mem_with_stub.record_task_outcome(**{**base, **mutation})


def test_outcome_rejects_unknown_memory(mem_with_stub) -> None:
    with pytest.raises(NotFoundError):
        mem_with_stub.record_task_outcome(
            task_id="task-missing",
            status="failure",
            memory_ids=["missing-memory"],
        )


def test_idempotent_outcome_retry_repairs_failed_projection(
    mem_with_stub,
    monkeypatch,
) -> None:
    record = mem_with_stub.save(
        content="Retry a projection from the authoritative journal.",
        title="Repair projection",
        auto_project=False,
    )
    original_update = mem_with_stub.update
    attempts = 0

    def flaky_update(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("simulated projection failure")
        return original_update(*args, **kwargs)

    monkeypatch.setattr(mem_with_stub, "update", flaky_update)
    with pytest.raises(OSError, match="projection failure"):
        mem_with_stub.record_task_outcome(
            task_id="repair-task",
            status="success",
            memory_ids=[record.id],
            idempotency_key="repair-key",
        )

    repaired = mem_with_stub.record_task_outcome(
        task_id="repair-task",
        status="success",
        memory_ids=[record.id],
        idempotency_key="repair-key",
    )
    updated = mem_with_stub.get(record.id)

    assert repaired["idempotent_replay"] is True
    assert repaired["updated_memory_ids"] == [record.id]
    assert updated is not None
    assert updated.extra["outcome_stats"]["total"] == 1
    outcome_events = [
        event
        for event in mem_with_stub.operational.ledger.validated_events()
        if event.op == "outcome.record"
    ]
    assert len(outcome_events) == 1


def test_promote_learning_creates_grounded_procedure(mem_with_stub) -> None:
    source = mem_with_stub.save(
        content="If the journal fails verification, stop before syncing.",
        title="Journal gate",
        type_="decision",
        auto_project=False,
    )
    for index in range(2):
        mem_with_stub.record_task_outcome(
            task_id=f"journal-gate-{index}",
            status="success",
            memory_ids=[source.id],
            idempotency_key=f"journal-gate-outcome-{index}",
        )

    procedure = mem_with_stub.promote_learning(
        [source.id],
        title="Verify journal before sync",
        kind="procedure",
    )

    assert procedure.type == "procedure"
    assert procedure.extra["learning"]["source_memory_ids"] == [source.id]
    assert procedure.extra["provenance"]["evidence_uris"] == [f"memo://memoria/{source.id}"]


def test_promote_learning_rejects_unqualified_agent_claim(mem_with_stub) -> None:
    source = mem_with_stub.save(
        content="A plausible but untested workflow.",
        title="Untested workflow",
        auto_project=False,
    )

    # Must be a MemoError (not a bare ValueError) so the MCP write coordinator
    # surfaces the reason instead of masking it as "write failed safely".
    with pytest.raises(MemoError, match="lacks outcome evidence"):
        mem_with_stub.promote_learning(
            [source.id],
            title="Do not promote this",
        )


def test_promote_learning_rejects_unknown_kind(mem_with_stub) -> None:
    # Validation must be a MemoError so the MCP coordinator surfaces the reason.
    with pytest.raises(MemoError, match="kind must be"):
        mem_with_stub.promote_learning(["whatever"], title="x", kind="bogus")


def test_promote_learning_rejects_empty_source_ids(mem_with_stub) -> None:
    with pytest.raises(MemoError, match="at least one source memory"):
        mem_with_stub.promote_learning([], title="x")


def test_record_task_outcome_rejects_empty_task_id(mem_with_stub) -> None:
    with pytest.raises(MemoError, match="task_id cannot be empty"):
        mem_with_stub.record_task_outcome(
            task_id="   ",
            status="success",
            memory_ids=["anything"],
        )


def test_operation_ledger_imports_complete_foreign_chain_and_rejects_forks(tmp_path) -> None:
    source = OperationLedger(tmp_path / "source", device_id="device-a")
    source.append("focus.set", subject_uri="memo://focus/repo", payload={"project": "repo"})
    source.append("focus.clear", subject_uri="memo://focus/repo", payload={"project": "repo"})
    rows = [event.to_dict() for event in source.iter_events()]
    target = OperationLedger(tmp_path / "target", device_id="device-b")

    first = target.import_events(rows)
    second = target.import_events(rows)

    assert first == {"devices": 1, "imported": 2, "unchanged": 0}
    assert second == {"devices": 1, "imported": 0, "unchanged": 2}
    assert target.verify()["ok"] is True
    forked = json.loads(json.dumps(rows))
    forked[1]["payload"]["project"] = "other"
    with pytest.raises(LedgerIntegrityError):
        target.import_events(forked)


def _target_config(tmp_cfg, tmp_path) -> Config:
    return Config(
        data_dir=tmp_path / "target-data",
        vault_path=tmp_path / "target-vault",
        state_dir=tmp_path / "target-state",
        embedder_dims=4,
        reranker_enabled=False,
    )


def _canonical_json(value) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _resign_bundle(
    bundle: dict,
    key: bytes,
    *,
    derive_bundle_id: bool = True,
) -> None:
    memories = bundle["memories"]
    operations = bundle["operations"]
    bundle["manifest"] = {
        "memory_count": len(memories),
        "operation_count": len(operations),
        "memories_sha256": hashlib.sha256(_canonical_json(memories)).hexdigest(),
        "operations_sha256": hashlib.sha256(_canonical_json(operations)).hexdigest(),
    }
    if derive_bundle_id:
        identity = dict(bundle)
        identity.pop("bundle_id", None)
        identity.pop("signature", None)
        bundle["bundle_id"] = "bundle-" + hashlib.sha256(_canonical_json(identity)).hexdigest()[:24]
    unsigned = dict(bundle)
    unsigned.pop("signature", None)
    bundle["signature"]["digest"] = hmac.new(
        key,
        _canonical_json(unsigned),
        hashlib.sha256,
    ).hexdigest()


def test_federation_enforces_acl_signature_and_idempotent_import(
    mem_with_stub,
    tmp_cfg,
    tmp_path,
) -> None:
    owner = "owner-fer"
    key = b"0123456789abcdef0123456789abcdef"
    mem_with_stub.save(
        content="never leaves this machine",
        title="Local",
        extra={"visibility": "local_only"},
        auto_project=False,
    )
    mem_with_stub.save(
        content="owner only",
        title="Owner",
        extra={"visibility": "owner", "owner_principal": owner},
        auto_project=False,
    )
    shared = mem_with_stub.save(
        content="shared with bob",
        title="Shared",
        extra={
            "visibility": "shared",
            "owner_principal": owner,
            "principals": ["bob"],
        },
        auto_project=False,
    )
    bundle_path = tmp_path / "bob.memo-federation.json"

    result = mem_with_stub.federation.export_bundle(
        bundle_path,
        principal="bob",
        owner_principal=owner,
        signing_key=key,
    )
    bundle = mem_with_stub.federation.verify_bundle(
        bundle_path,
        principal="bob",
        signing_key=key,
    )

    assert result["memories"] == 1
    assert result["operations"] == 0
    assert bundle["memories"][0]["source_id"] == shared.id
    target = Memory(_target_config(tmp_cfg, tmp_path))
    try:
        first = target.federation.import_bundle(
            bundle_path,
            principal="bob",
            signing_key=key,
        )
        count = target.store.count()
        second = target.federation.import_bundle(
            bundle_path,
            principal="bob",
            signing_key=key,
        )
        imported = target.list(limit=10)[0]
        assert first["imported"] == 1
        assert target.store.count() == count
        assert second["failed"] == 0
        assert second["imported"] == 0
        assert second["unchanged"] == 1
        assert second["idempotent_replay"] is True
        assert imported.extra["trust_tier"] == "external_untrusted"
        assert imported.extra["federation"]["source_id"] == shared.id
    finally:
        target.close()

    tampered = json.loads(bundle_path.read_text(encoding="utf-8"))
    tampered["memories"][0]["body"] = "tampered"
    bundle_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(FederationError, match="signature mismatch"):
        mem_with_stub.federation.verify_bundle(
            bundle_path,
            principal="bob",
            signing_key=key,
        )


@pytest.mark.parametrize(
    ("trust_peer", "expected_trust"),
    [
        (False, "external_untrusted"),
        (True, "tool_observed"),
    ],
)
def test_federation_treats_learning_metadata_as_non_authoritative_remote_claims(
    mem_with_stub,
    tmp_cfg,
    tmp_path,
    trust_peer,
    expected_trust,
) -> None:
    owner = "owner-fer"
    key = b"0123456789abcdef0123456789abcdef"
    source = mem_with_stub.save(
        content="A foreign workflow claim without local outcomes.",
        title="Foreign workflow",
        extra={
            "visibility": "shared",
            "owner_principal": owner,
            "principals": ["bob"],
            "outcome_stats": {
                "total": 50,
                "successes": 50,
                "failures": 0,
                "partials": 0,
                "utility": 1.0,
            },
            "learning": {"kind": "procedure", "source_memory_ids": ["foreign"]},
            "priority": "high",
        },
        auto_project=False,
    )
    bundle_path = tmp_path / f"claims-{trust_peer}.memo-federation.json"
    mem_with_stub.federation.export_bundle(
        bundle_path,
        principal="bob",
        owner_principal=owner,
        signing_key=key,
    )
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    exported_extra = bundle["memories"][0]["extra"]
    assert "outcome_stats" not in exported_extra
    assert "learning" not in exported_extra
    assert "priority" not in exported_extra

    exported_extra.update(
        {
            "outcome_stats": {
                "total": 50,
                "successes": 50,
                "failures": 0,
                "partials": 0,
                "utility": 1.0,
            },
            "learning": {"kind": "procedure", "source_memory_ids": [source.id]},
            "priority": "high",
        }
    )
    _resign_bundle(bundle, key)
    bundle_path.write_text(json.dumps(bundle), encoding="utf-8")

    target = Memory(_target_config(tmp_cfg, tmp_path))
    try:
        result = target.federation.import_bundle(
            bundle_path,
            principal="bob",
            signing_key=key,
            trust_peer=trust_peer,
        )
        imported = target.list(limit=10)[0]

        assert result["imported"] == 1
        assert imported.extra["trust_tier"] == expected_trust
        assert "outcome_stats" not in imported.extra
        assert "learning" not in imported.extra
        assert "priority" not in imported.extra
        assert imported.extra["federation"]["remote_claims"] == {
            "outcome_stats": exported_extra["outcome_stats"],
            "learning": exported_extra["learning"],
            "priority": "high",
        }
        assert target.procedure_candidates() == []
    finally:
        target.close()


def test_federation_rejects_resigned_content_with_a_reused_bundle_id(
    mem_with_stub,
    tmp_path,
) -> None:
    key = b"abcdef0123456789abcdef0123456789"
    owner = mem_with_stub.cfg.device_id
    mem_with_stub.save(
        content="Original bundle content.",
        title="Bundle identity",
        auto_project=False,
    )
    bundle_path = tmp_path / "reused-id.memo-federation.json"
    mem_with_stub.federation.export_bundle(
        bundle_path,
        principal=owner,
        signing_key=key,
    )
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    original_bundle_id = bundle["bundle_id"]
    bundle["memories"][0]["body"] = "Different content under a reused id."
    _resign_bundle(bundle, key, derive_bundle_id=False)
    assert bundle["bundle_id"] == original_bundle_id
    bundle_path.write_text(json.dumps(bundle), encoding="utf-8")

    with pytest.raises(FederationError, match="bundle id mismatch"):
        mem_with_stub.federation.verify_bundle(
            bundle_path,
            principal=owner,
            signing_key=key,
        )


def test_owner_federation_includes_complete_causal_journal(mem_with_stub, tmp_path) -> None:
    key = b"fedcba9876543210fedcba9876543210"
    owner = "owner-fer"
    mem_with_stub.operational.set_focus(project="memo", summary="Ship independence")
    bundle_path = tmp_path / "owner.memo-federation.json"

    result = mem_with_stub.federation.export_bundle(
        bundle_path,
        principal=owner,
        owner_principal=owner,
        signing_key=key,
    )
    bundle = mem_with_stub.federation.verify_bundle(
        bundle_path,
        principal=owner,
        signing_key=key,
    )

    assert result["operations"] > 0
    assert bundle["operations"][0]["sequence"] == 1
    assert bundle["manifest"]["operation_count"] == len(bundle["operations"])


def test_federation_rejects_invalid_journal_before_importing_memories(
    mem_with_stub,
    tmp_cfg,
    tmp_path,
) -> None:
    key = b"abcdef0123456789abcdef0123456789"
    owner = mem_with_stub.cfg.device_id
    mem_with_stub.save(
        content="Owner backup record",
        title="Owner backup",
        auto_project=False,
    )
    mem_with_stub.operational.set_focus(project="memo", summary="backup")
    bundle_path = tmp_path / "invalid-journal.memo-federation.json"
    mem_with_stub.federation.export_bundle(
        bundle_path,
        principal=owner,
        signing_key=key,
    )
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    bundle["operations"][0]["previous_hash"] = "forged"
    _resign_bundle(bundle, key)
    bundle_path.write_text(json.dumps(bundle), encoding="utf-8")

    target = Memory(_target_config(tmp_cfg, tmp_path))
    try:
        with pytest.raises(LedgerIntegrityError):
            target.federation.import_bundle(
                bundle_path,
                principal=owner,
                signing_key=key,
            )
        assert target.store.count() == 0
    finally:
        target.close()
