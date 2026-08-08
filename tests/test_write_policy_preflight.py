"""WritePolicyEngine.preflight: the conflict-override authority guards.

Overriding a write-frozen conflict requires THREE conditions in the real
code: `allow_conflict_override=True`, a non-empty `override_reason`, AND
`resolved_actor.actor_kind == "human"`. The positive path (all three
satisfied) is exercised by `test_native_write_policy_freezes_and_audits_override`
in test_operational_memory.py. These tests pin down the two negative
branches — an agent trying to self-override, and an override with an empty
reason — which were previously asserted nowhere: deleting either guard
clause left the full test suite green.
"""

from __future__ import annotations

from memo.contracts import ActorIdentity
from memo.operational import OperationalStore
from memo.write_policy import WritePolicyEngine


def _blocked_engine(tmp_path):
    operational = OperationalStore(tmp_path / "state", device_id="device-a")
    operational.open_conflict(
        topic="billing architecture",
        summary="Two incompatible billing designs are active",
    )
    return WritePolicyEngine(operational)


def test_override_by_non_human_actor_is_refused(tmp_path):
    engine = _blocked_engine(tmp_path)

    decision = engine.preflight(
        title="Billing architecture",
        content="Use the new invoice workflow.",
        tags=None,
        extra=None,
        actor=ActorIdentity(actor_id="agent-bot", actor_kind="agent"),
        allow_conflict_override=True,
        override_reason="I approve my own override",
    )

    assert decision.allowed is False
    assert "human" in decision.reason


def test_override_with_empty_reason_is_refused(tmp_path):
    engine = _blocked_engine(tmp_path)

    decision = engine.preflight(
        title="Billing architecture",
        content="Use the new invoice workflow.",
        tags=None,
        extra=None,
        actor=ActorIdentity(actor_id="maintainer", actor_kind="human"),
        allow_conflict_override=True,
        override_reason="   ",
    )

    assert decision.allowed is False
    assert "reason" in decision.reason


def test_override_by_human_with_reason_is_allowed(tmp_path):
    # Control: all three conditions satisfied → override succeeds.
    engine = _blocked_engine(tmp_path)

    decision = engine.preflight(
        title="Billing architecture",
        content="Use the new invoice workflow.",
        tags=None,
        extra=None,
        actor=ActorIdentity(actor_id="maintainer", actor_kind="human"),
        allow_conflict_override=True,
        override_reason="Maintainer approved design B",
    )

    assert decision.allowed is True
    assert decision.override is True
