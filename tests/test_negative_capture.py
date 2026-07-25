"""Negative-recall CAPTURE slice — deriving & persisting failure_pattern
anti-memories from supersede/reversal and from avoid verdicts.

Uses the ``mock_memory`` fixture (a real ``Memory`` with a stubbed, MLX-free
embedder) so saves write real ``.md`` files + a real sqlite index — no MLX cold
load, safe under parallel test runs. The capture flag defaults OFF; each test
that exercises a capture opts in via ``monkeypatch.setenv``.
"""

from __future__ import annotations

import pytest

from memo import negative_capture
from memo.dashboard_logs import append_verdict_log
from memo.negative_recall import (
    FAILURE_PATTERN_TYPE,
    FP_LINKS_KEY,
    FP_SOURCE_AVOID_VERDICT,
    FP_SOURCE_KEY,
    FP_SOURCE_SUPERSEDE,
    parse_failure_pattern,
)

_FLAG = "MEMO_NEGATIVE_RECALL_CAPTURE_ENABLED"


@pytest.fixture
def enabled(monkeypatch):
    """Turn the capture feature ON for the duration of a test."""
    monkeypatch.setenv(_FLAG, "1")


def _failure_patterns(mem):
    return mem.list(type_=FAILURE_PATTERN_TYPE, limit=100)


# ── supersede / reversal ─────────────────────────────────────────────────────


def test_supersede_produces_failure_pattern_with_wrong_right_and_provenance(mem_with_stub, enabled):
    mem = mem_with_stub
    wrong = mem.save(
        content="Use a global mutable singleton for the DB connection.",
        title="Global DB singleton",
        type_="decision",
    )
    right = mem.save(
        content="Use one sqlite connection per thread (thread-local).",
        title="Thread-local DB connection",
        type_="decision",
    )

    res = negative_capture.capture_from_supersede(
        mem, superseded_id=wrong.id, superseding_id=right.id
    )

    assert res["status"] == "captured"
    captured = mem.get(res["captured_id"])
    assert captured is not None
    assert captured.type == FAILURE_PATTERN_TYPE

    parsed = parse_failure_pattern(captured.body, captured.extra)
    assert parsed is not None
    # Wrong = the superseded approach, Right = the superseding approach.
    assert "global mutable singleton" in parsed.wrong
    assert "thread-local" in parsed.right.lower()

    # Provenance is auditable + links both source records.
    assert captured.extra[FP_SOURCE_KEY] == FP_SOURCE_SUPERSEDE
    assert captured.extra["wrong_id"] == wrong.id
    assert captured.extra["right_id"] == right.id
    assert set(captured.extra[FP_LINKS_KEY]) == {wrong.id, right.id}
    assert captured.extra[negative_capture.FP_PROVENANCE_HASH_KEY]


def test_supersede_dedup_on_repeat(mem_with_stub, enabled):
    mem = mem_with_stub
    wrong = mem.save(content="Old approach body.", title="Old", type_="decision")
    right = mem.save(content="New approach body.", title="New", type_="decision")

    first = negative_capture.capture_from_supersede(
        mem, superseded_id=wrong.id, superseding_id=right.id
    )
    second = negative_capture.capture_from_supersede(
        mem, superseded_id=wrong.id, superseding_id=right.id
    )

    assert first["status"] == "captured"
    assert second["status"] == "skipped_dup"
    assert second["captured_id"] is None
    # Exactly one anti-memory persisted despite two capture calls.
    assert len(_failure_patterns(mem)) == 1


def test_supersede_dry_run_does_not_persist(mem_with_stub, enabled):
    mem = mem_with_stub
    wrong = mem.save(content="Wrong body.", title="Wrong", type_="decision")
    right = mem.save(content="Right body.", title="Right", type_="decision")

    res = negative_capture.capture_from_supersede(
        mem, superseded_id=wrong.id, superseding_id=right.id, dry_run=True
    )

    assert res["status"] == "dry_run"
    assert res["captured_id"] is None
    assert _failure_patterns(mem) == []


def test_supersede_unresolved_ids_is_safe(mem_with_stub, enabled):
    mem = mem_with_stub
    res = negative_capture.capture_from_supersede(
        mem, superseded_id="ffffffffffffffffffffffffffffffff", superseding_id="00000000"
    )
    assert res["status"] == "unresolved"
    assert res["captured_id"] is None
    assert _failure_patterns(mem) == []


# ── default-off no-op ────────────────────────────────────────────────────────


def test_supersede_disabled_by_default_is_noop(mem_with_stub, monkeypatch):
    monkeypatch.delenv(_FLAG, raising=False)
    mem = mem_with_stub
    wrong = mem.save(content="Wrong body.", title="Wrong", type_="decision")
    right = mem.save(content="Right body.", title="Right", type_="decision")

    res = negative_capture.capture_from_supersede(
        mem, superseded_id=wrong.id, superseding_id=right.id
    )

    assert res["status"] == "disabled"
    assert _failure_patterns(mem) == []


def test_avoid_verdicts_disabled_by_default_is_noop(mem_with_stub, monkeypatch):
    monkeypatch.delenv(_FLAG, raising=False)
    mem = mem_with_stub
    misled = mem.save(content="A misleading fact.", title="Misleading", type_="fact")
    append_verdict_log(
        mem.cfg.state_dir,
        session_id="s1",
        turn=2,
        prior_turn=1,
        verdict="correction",
        prompt="how do I do X?",
        reaction="no, that's wrong",
        recall_ids=[misled.id],
    )

    res = negative_capture.graduate_avoid_verdicts(mem.cfg, mem)

    assert res["status"] == "disabled"
    assert res["captured"] == []
    assert _failure_patterns(mem) == []


# ── avoid verdicts ───────────────────────────────────────────────────────────


def test_avoid_verdict_produces_capture(mem_with_stub, enabled):
    mem = mem_with_stub
    misled = mem.save(
        content="Set the recall floor with the legacy MEMO_FLOOR flag.",
        title="Recall floor flag",
        type_="fact",
    )
    append_verdict_log(
        mem.cfg.state_dir,
        session_id="s1",
        turn=5,
        prior_turn=4,
        verdict="correction",
        prompt="how do I set the recall floor?",
        reaction="no, that's wrong — use MEMO_RECALL_MIN_SIM, not the legacy flag",
        recall_ids=[misled.id],
    )

    res = negative_capture.graduate_avoid_verdicts(mem.cfg, mem)

    assert res["status"] == "ok"
    assert res["candidates"] == 1
    assert len(res["captured"]) == 1

    captured = mem.get(res["captured"][0])
    assert captured is not None
    assert captured.type == FAILURE_PATTERN_TYPE

    parsed = parse_failure_pattern(captured.body, captured.extra)
    assert parsed is not None
    assert "legacy MEMO_FLOOR flag" in parsed.wrong
    assert "MEMO_RECALL_MIN_SIM" in parsed.right

    assert captured.extra[FP_SOURCE_KEY] == FP_SOURCE_AVOID_VERDICT
    assert captured.extra["origin_id"] == misled.id
    assert captured.extra["verdict"] == "correction"
    assert captured.extra[negative_capture.FP_PROVENANCE_HASH_KEY]


def test_avoid_verdict_dedup_on_repeat(mem_with_stub, enabled):
    mem = mem_with_stub
    misled = mem.save(content="A misleading claim.", title="Claim", type_="fact")
    append_verdict_log(
        mem.cfg.state_dir,
        session_id="s1",
        turn=3,
        prior_turn=2,
        verdict="negative",
        prompt="does the cache persist?",
        reaction="no, that didn't work",
        recall_ids=[misled.id],
    )

    first = negative_capture.graduate_avoid_verdicts(mem.cfg, mem)
    second = negative_capture.graduate_avoid_verdicts(mem.cfg, mem)

    assert len(first["captured"]) == 1
    assert second["captured"] == []
    assert second["skipped_dup"] == 1
    assert len(_failure_patterns(mem)) == 1


def test_avoid_verdict_recurrence_gate(mem_with_stub, enabled):
    mem = mem_with_stub
    misled = mem.save(content="Recurring bad advice.", title="Advice", type_="note")
    append_verdict_log(
        mem.cfg.state_dir,
        session_id="s1",
        turn=2,
        prior_turn=1,
        verdict="correction",
        prompt="which flag disables the reranker?",
        reaction="no, that's not it",
        recall_ids=[misled.id],
    )

    # One distinct turn: does not meet a min_occurrences=2 gate.
    res1 = negative_capture.graduate_avoid_verdicts(mem.cfg, mem, min_occurrences=2)
    assert res1["candidates"] == 0
    assert res1["captured"] == []

    # A second, distinct turn for the same recalled id crosses the gate.
    append_verdict_log(
        mem.cfg.state_dir,
        session_id="s2",
        turn=9,
        prior_turn=8,
        verdict="negative",
        prompt="which flag disables the reranker again?",
        reaction="still wrong",
        recall_ids=[misled.id],
    )
    res2 = negative_capture.graduate_avoid_verdicts(mem.cfg, mem, min_occurrences=2)
    assert res2["candidates"] == 1
    assert len(res2["captured"]) == 1


def test_avoid_verdict_skips_failure_pattern_origin(mem_with_stub, enabled):
    """A recalled failure_pattern that itself drew a negative verdict must not
    graduate into an anti-memory-of-an-anti-memory."""
    mem = mem_with_stub
    fp = mem.save(
        content="Pattern: something\nContext: c\nWrong: w\nRight: r",
        title="Existing anti-memory",
        type_=FAILURE_PATTERN_TYPE,
    )
    append_verdict_log(
        mem.cfg.state_dir,
        session_id="s1",
        turn=2,
        prior_turn=1,
        verdict="correction",
        prompt="a prompt",
        reaction="no, wrong",
        recall_ids=[fp.id],
    )

    res = negative_capture.graduate_avoid_verdicts(mem.cfg, mem)

    # The pre-existing failure_pattern is the only one; nothing new minted.
    assert res["captured"] == []
    assert len(_failure_patterns(mem)) == 1


def test_avoid_verdict_ignores_positive_verdicts(mem_with_stub, enabled):
    mem = mem_with_stub
    helped = mem.save(content="A helpful fact.", title="Helpful", type_="fact")
    append_verdict_log(
        mem.cfg.state_dir,
        session_id="s1",
        turn=2,
        prior_turn=1,
        verdict="positive",
        prompt="how do I do X?",
        reaction="perfect, thanks!",
        recall_ids=[helped.id],
    )

    res = negative_capture.graduate_avoid_verdicts(mem.cfg, mem)

    assert res["candidates"] == 0
    assert res["captured"] == []


def test_avoid_verdict_dry_run_does_not_persist(mem_with_stub, enabled):
    mem = mem_with_stub
    misled = mem.save(content="Body.", title="T", type_="fact")
    append_verdict_log(
        mem.cfg.state_dir,
        session_id="s1",
        turn=2,
        prior_turn=1,
        verdict="negative",
        prompt="q",
        reaction="no, that didn't work",
        recall_ids=[misled.id],
    )

    res = negative_capture.graduate_avoid_verdicts(mem.cfg, mem, dry_run=True)

    assert res["captured"] == ["<dry-run>"]
    assert _failure_patterns(mem) == []


# ── dream-pass wrapper ───────────────────────────────────────────────────────


def test_run_negative_capture_pass_delegates(mem_with_stub, enabled):
    from memo.cli_dream_passes import _run_negative_capture

    mem = mem_with_stub
    misled = mem.save(content="Bad idea.", title="Bad", type_="fact")
    append_verdict_log(
        mem.cfg.state_dir,
        session_id="s1",
        turn=2,
        prior_turn=1,
        verdict="correction",
        prompt="what should I do?",
        reaction="no, do the opposite",
        recall_ids=[misled.id],
    )

    res = _run_negative_capture(mem.cfg, mem)

    assert res["status"] == "ok"
    assert len(res["captured"]) == 1


def test_run_negative_capture_pass_noop_when_disabled(mem_with_stub, monkeypatch):
    from memo.cli_dream_passes import _run_negative_capture

    monkeypatch.delenv(_FLAG, raising=False)
    mem = mem_with_stub
    res = _run_negative_capture(mem.cfg, mem)
    assert res["status"] == "disabled"


# ── WIRING: `memo maintain` mints the supersede anti-memory too ──────────────
#
# The dream `_run_contradict` pass captures on supersede, but `memo maintain`
# (the SessionStart/daily freshness path) runs its OWN contradiction-resolution
# branch. The capture must be wired there too, or a supersede that only ever
# happens via `maintain` never mints its ⛔ lesson.


def _seed_and_run_maintain(mock_memory, tmp_path, *, capture_on: bool):
    import json as _json
    from unittest.mock import patch

    from click.testing import CliRunner

    from memo.cli import cli

    old = mock_memory.save(content="El puerto del dashboard es 8080", title="Puerto viejo")
    new = mock_memory.save(content="El puerto del dashboard es 8765", title="Puerto nuevo")
    mock_memory.contradict_store.upsert_open(
        memory_id_a=old.id,
        memory_id_b=new.id,
        relationship="contradiction",
        confidence=0.95,
        rationale="ports differ",
    )
    env = {
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_DATA_DIR": str(tmp_path / "data"),
        "MEMO_STATE_DIR": str(tmp_path / "state"),
    }
    if capture_on:
        env[_FLAG] = "1"
    with patch("memo.cli_maintain._get_memory", return_value=mock_memory):
        res = CliRunner().invoke(
            cli,
            ["maintain", "--skip-consolidate", "--skip-stale", "--skip-synthesize", "--json"],
            env=env,
        )
    assert res.exit_code == 0, res.output
    receipt = _json.loads(res.output[res.output.index("{") :])
    return old, new, receipt


def test_maintain_supersede_captures_failure_pattern_when_enabled(mock_memory, tmp_path):
    old, new, receipt = _seed_and_run_maintain(mock_memory, tmp_path, capture_on=True)

    assert receipt["superseded"], receipt  # the older side was superseded
    fps = _failure_patterns(mock_memory)
    assert len(fps) == 1
    parsed = parse_failure_pattern(fps[0].body, fps[0].extra)
    assert parsed is not None and parsed.source == FP_SOURCE_SUPERSEDE
    # Provenance links both source records (Wrong = older, Right = newer).
    assert set(fps[0].extra[FP_LINKS_KEY]) == {old.id, new.id}
    assert fps[0].id in receipt.get("negative_captured", [])


def test_maintain_supersede_no_capture_when_disabled(mock_memory, tmp_path, monkeypatch):
    monkeypatch.delenv(_FLAG, raising=False)
    _old, _new, receipt = _seed_and_run_maintain(mock_memory, tmp_path, capture_on=False)

    # The supersede still happens, but no anti-memory is minted (flag OFF).
    assert receipt["superseded"], receipt
    assert _failure_patterns(mock_memory) == []


def test_maintain_supersede_surfaces_capture_error_in_receipt(mock_memory, tmp_path, monkeypatch):
    """A capture error is surfaced in the maintain receipt (never swallowed) and
    does not abort the supersede."""
    from memo import negative_capture as _nc

    monkeypatch.setattr(
        _nc,
        "capture_from_supersede",
        lambda *a, **k: {"status": "error", "error": "RuntimeError: boom", "captured_id": None},
    )

    _old, _new, receipt = _seed_and_run_maintain(mock_memory, tmp_path, capture_on=True)

    assert receipt["superseded"], receipt  # supersede completed despite the error
    assert any("negative_capture: RuntimeError: boom" in e for e in receipt["errors"])


# ── error paths are surfaced, never swallowed ────────────────────────────────


def test_supersede_capture_error_is_surfaced(mem_with_stub, enabled, monkeypatch):
    mem = mem_with_stub
    wrong = mem.save(content="Wrong body.", title="Wrong", type_="decision")
    right = mem.save(content="Right body.", title="Right", type_="decision")

    def _boom(*_a, **_k):
        raise RuntimeError("disk full")

    monkeypatch.setattr(mem, "save", _boom)

    res = negative_capture.capture_from_supersede(
        mem, superseded_id=wrong.id, superseding_id=right.id
    )

    assert res["status"] == "error"
    assert "disk full" in res["error"]


def test_avoid_verdict_persist_error_is_surfaced(mem_with_stub, enabled, monkeypatch):
    mem = mem_with_stub
    misled = mem.save(content="A body.", title="T", type_="fact")
    append_verdict_log(
        mem.cfg.state_dir,
        session_id="s1",
        turn=2,
        prior_turn=1,
        verdict="negative",
        prompt="q",
        reaction="no, that didn't work",
        recall_ids=[misled.id],
    )

    def _boom(*_a, **_k):
        raise RuntimeError("disk full")

    monkeypatch.setattr(mem, "save", _boom)

    res = negative_capture.graduate_avoid_verdicts(mem.cfg, mem)

    # Whole pass still 'ok'; the per-id save failure is surfaced in errors.
    assert res["status"] == "ok"
    assert res["captured"] == []
    assert res["errors"]
    assert "disk full" in res["errors"][0]
