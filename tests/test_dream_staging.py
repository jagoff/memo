"""Tests for dream conflict-staging (`memo.dream_staging`).

Hermetic unit tests over the deterministic surface: no MLX, no network, no
real vault — every test builds an isolated `Config` (the `tmp_cfg` fixture) and
drives a fake `Memory` whose `.save` and `.operational.active_conflicts()` are
scripted. Flags are set via `monkeypatch.setenv` (memo flag resolution reads
env first).
"""

from __future__ import annotations

from typing import Any

import pytest

from memo import dream_staging as mod
from memo.errors import StorageError, WriteRefused

STAGING_FLAG = "MEMO_DREAM_STAGING_ENABLED"
STAGING_MAX_FLAG = "MEMO_DREAM_STAGING_MAX"


# --- fakes -----------------------------------------------------------------
class FakeOperational:
    """`mem.operational` double: `active_conflicts()` returns scripted rows."""

    def __init__(self, rows: Any = ()) -> None:
        self._rows = rows  # list, or callable -> list, or callable that raises

    def active_conflicts(self) -> list[dict[str, Any]]:
        rows = self._rows() if callable(self._rows) else self._rows
        return [dict(r) for r in rows]


class FakeMemory:
    """Minimal `Memory` stand-in for `staged_save` / `resume_staged_proposals`."""

    def __init__(self, *, save: Any = None, conflict_rows: Any = ()) -> None:
        self._save = save
        self.save_calls: list[dict[str, Any]] = []
        self.operational = FakeOperational(conflict_rows)

    def save(self, **kwargs: Any) -> Any:
        self.save_calls.append(dict(kwargs))
        if self._save is None:
            return {"id": "saved", "kwargs": dict(kwargs)}
        return self._save(**kwargs)


def _refuse(conflict: dict[str, Any]):
    def _raise(**_kwargs: Any) -> Any:
        raise WriteRefused(conflict)

    return _raise


def _raise_storage(**_kwargs: Any) -> Any:
    raise StorageError("disk on fire")


def _stage_one(cfg, mem, *, kind="synthesis", title="T", source_ids=("a", "b"),
               conflict=None, extra=None):
    """Helper: stage a single proposal directly via `stage_proposal`."""
    save_kwargs: dict[str, Any] = {"content": "body", "title": title, "type_": "synthesis"}
    if extra is not None:
        save_kwargs["extra"] = extra
    return mod.stage_proposal(
        cfg,
        mem,
        kind=kind,
        save_kwargs=save_kwargs,
        source_ids=list(source_ids),
        conflict=conflict or {"conflict_id": "c1", "summary": "topic frozen"},
    )


# --- staged_save: scope / flag gating --------------------------------------
def test_staged_save_flag_off_is_passthrough_and_propagates_refused(tmp_cfg, monkeypatch):
    monkeypatch.setenv(STAGING_FLAG, "0")
    mem = FakeMemory(save=_refuse({"conflict_id": "c1", "summary": "frozen"}))

    with mod.dream_staging_scope():
        with pytest.raises(WriteRefused):
            mod.staged_save(
                mem, tmp_cfg, kind="synthesis", source_ids=["a"],
                content="body", title="T", type_="synthesis",
            )
    # Passthrough: nothing was parked.
    assert mod.list_staged(tmp_cfg) == []


def test_staged_save_outside_scope_is_passthrough_and_propagates_refused(tmp_cfg, monkeypatch):
    monkeypatch.setenv(STAGING_FLAG, "1")  # flag on, but NOT inside a dream scope
    mem = FakeMemory(save=_refuse({"conflict_id": "c1", "summary": "frozen"}))

    with pytest.raises(WriteRefused):
        mod.staged_save(
            mem, tmp_cfg, kind="synthesis", source_ids=["a"],
            content="body", title="T", type_="synthesis",
        )
    assert mod.list_staged(tmp_cfg) == []


def test_staged_save_inside_scope_parks_refused_and_returns_none(tmp_cfg, monkeypatch):
    monkeypatch.setenv(STAGING_FLAG, "1")
    mem = FakeMemory(save=_refuse({"conflict_id": "c1", "summary": "topic frozen"}))

    with mod.dream_staging_scope():
        result = mod.staged_save(
            mem, tmp_cfg, kind="synthesis", source_ids=["a", "b"],
            content="body", title="T", type_="synthesis",
        )

    assert result is None
    staged = mod.list_staged(tmp_cfg)
    assert len(staged) == 1
    p = staged[0]
    assert p.proposal_id.startswith("dream-synthesis-")
    assert p.save_kwargs == {"content": "body", "title": "T", "type_": "synthesis"}
    assert p.source_ids == ("a", "b")
    assert p.conflict_ids == ("c1",)
    assert p.conflict_summary == "topic frozen"
    assert p.attempts == 1
    assert p.state == "staged"


def test_staged_save_reraises_non_refused_error(tmp_cfg, monkeypatch):
    monkeypatch.setenv(STAGING_FLAG, "1")
    mem = FakeMemory(save=_raise_storage)

    with mod.dream_staging_scope():
        with pytest.raises(StorageError):
            mod.staged_save(
                mem, tmp_cfg, kind="synthesis", source_ids=["a"],
                content="body", title="T", type_="synthesis",
            )
    # Only WriteRefused parks; every other error re-raises and stages nothing.
    assert mod.list_staged(tmp_cfg) == []


def test_staged_save_success_returns_record_and_stages_nothing(tmp_cfg, monkeypatch):
    monkeypatch.setenv(STAGING_FLAG, "1")
    mem = FakeMemory()  # default save returns a sentinel record

    with mod.dream_staging_scope():
        result = mod.staged_save(
            mem, tmp_cfg, kind="synthesis", source_ids=["a"],
            content="body", title="T", type_="synthesis",
        )

    assert result is not None
    assert result["id"] == "saved"
    assert mod.list_staged(tmp_cfg) == []


# --- stage_proposal: idempotency + provenance ------------------------------
def test_stage_proposal_is_idempotent_by_proposal_id(tmp_cfg):
    mem = FakeMemory()
    first = _stage_one(tmp_cfg, mem, title="Same")
    second = _stage_one(tmp_cfg, mem, title="Same")

    assert first.proposal_id == second.proposal_id
    staged = mod.list_staged(tmp_cfg)
    assert len(staged) == 1  # no duplicate
    assert staged[0].attempts == 2  # re-stage bumped attempts


def test_stage_proposal_distinct_titles_produce_distinct_ids(tmp_cfg):
    mem = FakeMemory()
    p1 = _stage_one(tmp_cfg, mem, title="Title one")
    p2 = _stage_one(tmp_cfg, mem, title="Title two")

    assert p1.proposal_id != p2.proposal_id
    assert len(mod.list_staged(tmp_cfg)) == 2


def test_stage_proposal_provenance_hash_from_extra_drives_id(tmp_cfg):
    mem = FakeMemory()
    # Same provenance hash but different titles => same deterministic id.
    p1 = _stage_one(tmp_cfg, mem, title="One", extra={"synthesis_sources_hash": "abc123def456"})
    p2 = _stage_one(tmp_cfg, mem, title="Two", extra={"synthesis_sources_hash": "abc123def456"})

    assert p1.proposal_id == "dream-synthesis-abc123def456"
    assert p2.proposal_id == p1.proposal_id
    assert len(mod.list_staged(tmp_cfg)) == 1


# --- conflict enrichment ----------------------------------------------------
def test_stage_proposal_enriches_conflict_from_active_rows(tmp_cfg):
    rows = [
        {
            "id": "c1",
            "summary": "enriched summary",
            "evidence_uris": ["memo://evidence/1"],
            "metadata": {"memory_ids": ["m1", "m2"]},
        }
    ]
    mem = FakeMemory(conflict_rows=rows)
    p = _stage_one(tmp_cfg, mem, conflict={"conflict_id": "c1", "summary": "raw"})

    assert p.conflict_ids == ("c1",)
    assert p.conflict_summary == "enriched summary"
    assert p.evidence_uris == ("memo://evidence/1",)


def test_stage_proposal_evidence_falls_back_to_member_memory_ids(tmp_cfg):
    rows = [{"id": "c1", "summary": "s", "metadata": {"memory_ids": ["m1", "m2"]}}]
    mem = FakeMemory(conflict_rows=rows)
    p = _stage_one(tmp_cfg, mem, conflict={"conflict_id": "c1", "summary": "raw"})

    assert p.evidence_uris == ("memo://memoria/m1", "memo://memoria/m2")


def test_stage_proposal_degrades_when_active_conflicts_raises(tmp_cfg):
    def _boom():
        raise RuntimeError("conflicts read failed")

    mem = FakeMemory(conflict_rows=_boom)
    p = _stage_one(tmp_cfg, mem, conflict={"conflict_id": "c1", "summary": "raw summary"})

    # Best-effort enrichment: degrades to the WriteRefused's own summary.
    assert p.conflict_ids == ("c1",)
    assert p.conflict_summary == "raw summary"
    assert p.evidence_uris == ()


# --- cap enforcement --------------------------------------------------------
def test_enforce_cap_drops_oldest_staged():
    proposals = [
        mod.StagedProposal(f"p{i}", "synthesis", {}, (), (), "", (), "t")
        for i in range(4)
    ]
    kept = mod._enforce_cap(proposals, 2)

    kept_ids = [p.proposal_id for p in kept]
    assert kept_ids == ["p2", "p3"]  # two oldest (p0, p1) dropped


def test_enforce_cap_ignores_non_staged_when_counting():
    proposals = [
        mod.StagedProposal("a", "k", {}, (), (), "", (), "t", state="applied"),
        mod.StagedProposal("b", "k", {}, (), (), "", (), "t", state="staged"),
        mod.StagedProposal("c", "k", {}, (), (), "", (), "t", state="staged"),
    ]
    kept = mod._enforce_cap(proposals, 1)
    # Only staged entries count toward the cap; the applied one stays.
    kept_ids = sorted(p.proposal_id for p in kept)
    assert "a" in kept_ids
    assert "c" in kept_ids
    assert "b" not in kept_ids  # oldest staged dropped


def test_stage_proposal_enforces_cap_via_flag(tmp_cfg, monkeypatch):
    monkeypatch.setenv(STAGING_MAX_FLAG, "2")
    mem = FakeMemory()
    _stage_one(tmp_cfg, mem, title="one")
    _stage_one(tmp_cfg, mem, title="two")
    _stage_one(tmp_cfg, mem, title="three")

    staged = mod.list_staged(tmp_cfg)
    assert len(staged) == 2
    titles = {p.save_kwargs["title"] for p in staged}
    assert titles == {"two", "three"}  # oldest ("one") dropped


# --- resume ----------------------------------------------------------------
def test_resume_applies_unblocked_proposal(tmp_cfg):
    stager = FakeMemory()
    _stage_one(tmp_cfg, stager, title="unblock me")

    # Later run: conflict resolved (no active conflicts), save now succeeds.
    resumer = FakeMemory(conflict_rows=[])
    result = mod.resume_staged_proposals(tmp_cfg, resumer)

    assert len(result["applied"]) == 1
    assert result["still_blocked"] == 0
    assert result["total_open"] == 0
    assert mod.list_staged(tmp_cfg) == []
    assert len(resumer.save_calls) == 1


def test_resume_keeps_still_blocked_proposal(tmp_cfg):
    stager = FakeMemory()
    _stage_one(tmp_cfg, stager, title="blocked", conflict={"conflict_id": "c1", "summary": "s"})

    # The blocking conflict is still active.
    resumer = FakeMemory(conflict_rows=[{"id": "c1", "summary": "s"}])
    result = mod.resume_staged_proposals(tmp_cfg, resumer)

    assert result["applied"] == []
    assert result["still_blocked"] == 1
    assert result["total_open"] == 1
    assert len(resumer.save_calls) == 0  # blocked proposal is not re-saved
    assert len(mod.list_staged(tmp_cfg)) == 1


def test_resume_restages_when_a_new_conflict_blocks(tmp_cfg):
    stager = FakeMemory()
    _stage_one(tmp_cfg, stager, title="restage", conflict={"conflict_id": "c1", "summary": "old"})

    # Original conflict gone (empty active list) but save hits a *new* blocker.
    resumer = FakeMemory(
        save=_refuse({"conflict_id": "c2", "summary": "new block"}),
        conflict_rows=[],
    )
    result = mod.resume_staged_proposals(tmp_cfg, resumer)

    assert result["applied"] == []
    assert result["still_blocked"] == 1
    assert result["total_open"] == 1
    staged = mod.list_staged(tmp_cfg)
    assert len(staged) == 1
    assert staged[0].conflict_ids == ("c2",)  # re-pointed at the new blocker
    assert staged[0].attempts == 2  # bumped on re-stage


def test_resume_records_error_and_keeps_on_memoerror(tmp_cfg):
    stager = FakeMemory()
    _stage_one(tmp_cfg, stager, title="erring", conflict={"conflict_id": "c1", "summary": "s"})

    resumer = FakeMemory(save=_raise_storage, conflict_rows=[])
    result = mod.resume_staged_proposals(tmp_cfg, resumer)

    assert result["applied"] == []
    assert len(result["errors"]) == 1
    assert "StorageError" in result["errors"][0]
    assert result["total_open"] == 1  # kept for a later retry
    assert len(mod.list_staged(tmp_cfg)) == 1


def test_resume_runs_outside_scope_and_restores_contextvar(tmp_cfg, monkeypatch):
    monkeypatch.setenv(STAGING_FLAG, "1")
    stager = FakeMemory()
    _stage_one(tmp_cfg, stager, title="unblock", conflict={"conflict_id": "c1", "summary": "s"})

    resumer = FakeMemory(conflict_rows=[])
    with mod.dream_staging_scope():
        assert mod._STAGING_SCOPE.get() is True
        result = mod.resume_staged_proposals(tmp_cfg, resumer)
        # Replay resets the scope internally, then restores it on exit.
        assert mod._STAGING_SCOPE.get() is True

    assert len(result["applied"]) == 1
    assert mod.list_staged(tmp_cfg) == []


# --- list / drop / resolve_command -----------------------------------------
def test_list_staged_on_fresh_cfg_is_empty(tmp_cfg):
    assert mod.list_staged(tmp_cfg) == []


def test_drop_staged_removes_by_id_and_reports(tmp_cfg):
    mem = FakeMemory()
    p = _stage_one(tmp_cfg, mem, title="dropme")

    assert mod.drop_staged(tmp_cfg, p.proposal_id) is True
    assert mod.list_staged(tmp_cfg) == []
    # Dropping again reports no removal.
    assert mod.drop_staged(tmp_cfg, p.proposal_id) is False


def test_resolve_command_uses_first_conflict_id():
    p = mod.StagedProposal(
        "dream-x-abc", "x", {}, (), ("conf-42",), "", (), "2026-01-01T00:00:00Z"
    )
    cmd = mod.resolve_command(p)
    assert "conf-42" in cmd
    assert cmd.startswith("memo operational conflict resolve conf-42")


def test_resolve_command_placeholder_without_conflict_ids():
    p = mod.StagedProposal("dream-x-abc", "x", {}, (), (), "", (), "2026-01-01T00:00:00Z")
    cmd = mod.resolve_command(p)
    assert "<conflict-id>" in cmd


# --- serialization round-trip ----------------------------------------------
def test_staged_proposal_round_trip_is_stable():
    p = mod.StagedProposal(
        proposal_id="dream-synthesis-deadbeef",
        kind="synthesis",
        save_kwargs={"content": "body", "title": "T"},
        source_ids=("a", "b"),
        conflict_ids=("c1",),
        conflict_summary="frozen",
        evidence_uris=("memo://x",),
        staged_at="2026-01-01T00:00:00Z",
        state="staged",
        attempts=3,
        metadata={"k": "v"},
    )
    restored = mod.StagedProposal.from_dict(p.to_dict())
    assert restored == p
