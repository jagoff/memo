"""Regression contracts admitted by the Synapse retirement parity gate."""

from __future__ import annotations

from types import SimpleNamespace

from memo.contracts import AnswerStatus


def _hit(memory_id: str, *, title: str, body: str, score: float, extra: dict[str, object]):
    return SimpleNamespace(
        id=memory_id,
        title=title,
        body=body,
        score=score,
        extra=extra,
        type="decision",
        valid_at=None,
        invalid_at=None,
    )


def test_evidence_pack_emits_memo_native_provenance_for_legacy_input(
    mem_with_stub, monkeypatch
) -> None:
    monkeypatch.setattr(
        mem_with_stub,
        "search",
        lambda *_args, **_kwargs: [
            _hit(
                "source-a",
                title="Federated query policy",
                body="Federated query policy preserves source provenance.",
                score=0.9,
                extra={"synapse_trace_id": "retired-trace", "trust_tier": "human"},
            )
        ],
    )

    pack = mem_with_stub.evidence_pack("What federated query policy preserves provenance?")

    assert pack.status is AnswerStatus.ANSWERED
    assert pack.items[0].uri == "memo://memoria/source-a"
    assert pack.items[0].provenance == {"trace_id": "retired-trace"}


def test_evidence_pack_abstains_when_the_admitted_source_set_is_empty(mem_with_stub) -> None:
    pack = mem_with_stub.evidence_pack("What unmapped retired chat feature exists?")

    assert pack.status is AnswerStatus.INSUFFICIENT_EVIDENCE
    assert pack.items == ()
    assert pack.abstention_reason == "no relevant memories found"
