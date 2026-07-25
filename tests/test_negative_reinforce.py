"""REINFORCE slice — the ⛔ negative-recall closed loop (`outcome.reconcile_negative_recall`).

A ``failure_pattern`` anti-memory that was surfaced (its id in a next-turn
verdict's ``recall_ids``) is reinforced by that turn's outcome: a REPEAT
(negative/correction verdict — the mistake happened despite the ⛔ warning)
strengthens it (roi toward cap, demoted confidence restored); a HEED (positive
verdict) is a mild positive. Gated OFF by default.

All embeds are stubbed via the ``mock_memory`` fixture — no real MLX.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from memo import dashboard, outcome

_NEUTRAL = 1.0


def _verdict(
    sd: Path,
    *,
    sid: str,
    turn: int,
    verdict: str,
    recall_ids: list[str],
) -> None:
    dashboard.append_verdict_log(
        sd,
        session_id=sid,
        turn=turn,
        prior_turn=turn - 1,
        verdict=verdict,
        prompt="what should I do before the release",
        reaction="ok",
        recall_ids=recall_ids,
    )


def _save_failure_pattern(mem, *, title: str) -> str:
    body = (
        "Pattern: skipped the pre-push gate\n"
        "Context: cutting a release under time pressure\n"
        "Wrong: pushed straight to master\n"
        "Right: run the gate from an isolated worktree first"
    )
    rec = mem.save(content=body, title=title, type_="failure_pattern")
    return rec.id


def _roi(mem, mem_id: str) -> float:
    return mem.store.get_health_batch([mem_id]).get(mem_id, {}).get("roi_score", _NEUTRAL)


def _confidence(mem, mem_id: str) -> float:
    return mem.store.get_health_batch([mem_id]).get(mem_id, {}).get("confidence", 1.0)


# ---------------- strengthen on repeat ----------------


def test_repeat_failure_strengthens_the_pattern(mock_memory, monkeypatch) -> None:
    monkeypatch.setenv("MEMO_NEGATIVE_RECALL_REINFORCE_ENABLED", "1")
    mem = mock_memory
    fp = _save_failure_pattern(mem, title="Avoid pushing straight to master")
    # Surfaced across two turns; both next turns reported the mistake recurred.
    _verdict(mem.cfg.state_dir, sid="s1", turn=2, verdict="negative", recall_ids=[fp])
    _verdict(mem.cfg.state_dir, sid="s1", turn=4, verdict="correction", recall_ids=[fp])

    res = outcome.reconcile_negative_recall(mem)

    assert res["enabled"] is True
    assert res["strengthened"] == 1
    assert res["scored"] == 1
    # Two distinct repeat turns saturate the signal → roi driven to the cap,
    # well above the 1.0 neutral so it surfaces more forcefully.
    assert _roi(mem, fp) > _NEUTRAL
    assert _roi(mem, fp) == pytest.approx(1.5, abs=1e-6)


# ---------------- heed gives a (smaller) positive ----------------


def test_heed_gives_a_positive_smaller_than_repeat(mock_memory, monkeypatch) -> None:
    monkeypatch.setenv("MEMO_NEGATIVE_RECALL_REINFORCE_ENABLED", "1")
    mem = mock_memory
    fp_repeat = _save_failure_pattern(mem, title="Avoid force-push")
    fp_heed = _save_failure_pattern(mem, title="Avoid deleting the wrong branch")
    _verdict(mem.cfg.state_dir, sid="s1", turn=2, verdict="negative", recall_ids=[fp_repeat])
    _verdict(mem.cfg.state_dir, sid="s2", turn=2, verdict="positive", recall_ids=[fp_heed])

    res = outcome.reconcile_negative_recall(mem)

    assert res["strengthened"] == 1
    assert res["heeded"] == 1
    heed_roi = _roi(mem, fp_heed)
    repeat_roi = _roi(mem, fp_repeat)
    # A heeded warning is a SUCCESS → positive, never demoted below neutral…
    assert heed_roi > _NEUTRAL
    # …but the un-heeded repeat is reinforced harder.
    assert repeat_roi > heed_roi


# ---------------- default-off no-op ----------------


def test_disabled_is_a_no_op(mock_memory) -> None:
    mem = mock_memory  # flag unset → default off
    fp = _save_failure_pattern(mem, title="Avoid rm -rf on the data dir")
    _verdict(mem.cfg.state_dir, sid="s1", turn=2, verdict="negative", recall_ids=[fp])

    res = outcome.reconcile_negative_recall(mem)

    assert res["enabled"] is False
    assert res["strengthened"] == 0 and res["scored"] == 0
    # No roi write happened — the failure_pattern stays at the neutral default.
    assert _roi(mem, fp) == pytest.approx(_NEUTRAL, abs=1e-6)


# ---------------- unknown / non-failure_pattern ids are safe ----------------


def test_unknown_and_non_failure_pattern_ids_are_safe_noops(mock_memory, monkeypatch) -> None:
    monkeypatch.setenv("MEMO_NEGATIVE_RECALL_REINFORCE_ENABLED", "1")
    mem = mock_memory
    decision = mem.save(content="use MLX for privacy", title="Runtime", type_="decision")
    # One id that resolves to nothing, one that resolves to a non-anti-memory.
    _verdict(
        mem.cfg.state_dir,
        sid="s1",
        turn=2,
        verdict="negative",
        recall_ids=["deadbeef", decision.id],
    )

    res = outcome.reconcile_negative_recall(mem)  # must not raise

    assert res["enabled"] is True
    assert res["strengthened"] == 0
    assert res["scored"] == 0
    # A normal memory is never touched by the negative-recall reinforcer.
    assert _roi(mem, decision.id) == pytest.approx(_NEUTRAL, abs=1e-6)


# ---------------- confidence restoration ----------------


def test_repeat_restores_a_demoted_confidence(mock_memory, monkeypatch) -> None:
    monkeypatch.setenv("MEMO_NEGATIVE_RECALL_REINFORCE_ENABLED", "1")
    mem = mock_memory
    fp = _save_failure_pattern(mem, title="Avoid dropping the prod table")
    # Some prior signal (e.g. a contradiction) demoted its confidence.
    mem.store.penalize_confidence_batch([fp], delta=0.6, floor=0.1)
    assert _confidence(mem, fp) == pytest.approx(0.4, abs=1e-6)
    _verdict(mem.cfg.state_dir, sid="s1", turn=2, verdict="negative", recall_ids=[fp])

    res = outcome.reconcile_negative_recall(mem)

    assert res["confidence_restored"] == 1
    # Gradual, only-raises restore toward the cap.
    assert _confidence(mem, fp) == pytest.approx(0.6, abs=1e-6)
    # roi is boosted at the same time (confidence write leaves roi intact).
    assert _roi(mem, fp) > _NEUTRAL


# ---------------- idempotency (absolute roi write) ----------------


def test_roi_is_idempotent_across_runs(mock_memory, monkeypatch) -> None:
    monkeypatch.setenv("MEMO_NEGATIVE_RECALL_REINFORCE_ENABLED", "1")
    mem = mock_memory
    fp = _save_failure_pattern(mem, title="Avoid skipping the eval gate")
    _verdict(mem.cfg.state_dir, sid="s1", turn=2, verdict="negative", recall_ids=[fp])

    outcome.reconcile_negative_recall(mem)
    first = _roi(mem, fp)
    first_conf = _confidence(mem, fp)
    # A second run over the same (unchanged) verdict.log must not inflate roi —
    # it is an absolute write recomputed from the log, not a relative boost.
    outcome.reconcile_negative_recall(mem)
    assert _roi(mem, fp) == pytest.approx(first, abs=1e-6)
    # Confidence started at the cap (never demoted) → left untouched, stable.
    assert _confidence(mem, fp) == pytest.approx(first_conf, abs=1e-6)


# ---------------- WIRING: the dream ROI-reconcile pass calls the loop ----------


def test_roi_reconcile_pass_invokes_reinforce_when_flag_on(mock_memory, monkeypatch) -> None:
    """The reinforce loop is only closed if `_run_roi_reconcile` actually CALLS
    `reconcile_negative_recall`. With the flag on it must, and its summary must
    land in the receipt (mirror the source_feedback fold)."""
    from memo import cli_dream_passes

    monkeypatch.setenv("MEMO_NEGATIVE_RECALL_REINFORCE_ENABLED", "1")
    calls: list[object] = []

    def _spy(mem: object) -> dict[str, object]:
        calls.append(mem)
        return {"enabled": True, "strengthened": 3, "heeded": 1, "scored": 4}

    monkeypatch.setattr(cli_dream_passes, "reconcile_negative_recall", _spy)

    res = cli_dream_passes._run_roi_reconcile(mock_memory)

    assert len(calls) == 1 and calls[0] is mock_memory
    assert res["negative_reinforced"] == {
        "enabled": True,
        "strengthened": 3,
        "heeded": 1,
        "scored": 4,
    }


def test_roi_reconcile_pass_skips_reinforce_when_flag_off(mock_memory, monkeypatch) -> None:
    from memo import cli_dream_passes

    monkeypatch.delenv("MEMO_NEGATIVE_RECALL_REINFORCE_ENABLED", raising=False)
    calls: list[object] = []
    monkeypatch.setattr(
        cli_dream_passes, "reconcile_negative_recall", lambda mem: calls.append(mem)
    )

    res = cli_dream_passes._run_roi_reconcile(mock_memory)

    assert calls == []  # default-off ⇒ the loop is never invoked
    assert res["negative_reinforced"] == {}  # untouched default


def test_roi_reconcile_pass_absorbs_reinforce_error(mock_memory, monkeypatch) -> None:
    """A reinforce failure is logged, not raised — the roi-reconcile pass must
    survive it (don't swallow silently, but don't abort the whole pass)."""
    from memo import cli_dream_passes

    monkeypatch.setenv("MEMO_NEGATIVE_RECALL_REINFORCE_ENABLED", "1")

    def _boom(_mem: object) -> dict[str, object]:
        raise RuntimeError("roi write failed")

    monkeypatch.setattr(cli_dream_passes, "reconcile_negative_recall", _boom)

    res = cli_dream_passes._run_roi_reconcile(mock_memory)  # must not raise

    assert res["negative_reinforced"] == {}  # error absorbed, default intact
    assert "error" not in res  # the reinforce failure is isolated to its branch
