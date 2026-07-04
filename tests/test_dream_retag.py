"""dream_retag — cross-project grounding counts + retag decisions (pure core)."""

from __future__ import annotations

import pytest

from memo import dream_retag as dr
from memo.config import Config
from memo.dashboard_logs import append_grounding_log
from memo.memory import Memory


def _row(rid: str, project: str | None, score: float = 0.9) -> dict:
    return {"recall_id": rid, "project": project, "used_score": score, "method": "embed"}


def test_cross_project_counts_groups_thresholds_and_ignores_no_project():
    rows = [
        _row("aaaaaaaa", "project:synapse"),
        _row("aaaaaaaa", "project:memflow"),
        _row("aaaaaaaa", "project:synapse"),  # duplicate project → one entry
        _row("aaaaaaaa", "project:dotfiles", score=0.2),  # below threshold → dropped
        _row("bbbbbbbb", None),  # no project context → not evidence
        _row("", "project:synapse"),  # empty id → dropped
    ]
    counts = dr.cross_project_counts(rows, threshold=0.6)
    assert counts == {"aaaaaaaa": {"project:synapse", "project:memflow"}}


def test_retag_decision_requires_min_other_projects():
    counts = {
        "aaaaaaaa": {"project:synapse", "project:memflow"},  # 2 others → promote
        "bbbbbbbb": {"project:synapse"},  # 1 other → keep
        "cccccccc": {"project:memo", "project:synapse"},  # own + 1 other → keep
    }
    records = {
        "aaaaaaaa": {"id": "a" * 32, "tags": ["project:memo", "til"], "type": "fact"},
        "bbbbbbbb": {"id": "b" * 32, "tags": ["project:memo"], "type": "fact"},
        "cccccccc": {"id": "c" * 32, "tags": ["project:memo"], "type": "fact"},
    }
    decisions = dr.retag_decisions(counts, get_record=records.get, min_other_projects=2)
    assert [d["id"] for d in decisions] == ["a" * 32]
    assert decisions[0]["drop_tags"] == ["project:memo"]
    assert decisions[0]["new_tags"] == ["til"]
    assert decisions[0]["evidence_projects"] == ["project:memflow", "project:synapse"]


def test_retag_skips_already_global_reference_tier_and_missing():
    counts = {
        "dddddddd": {"project:x", "project:y"},
        "eeeeeeee": {"project:x", "project:y"},
        "ffffffff": {"project:x", "project:y"},
    }
    records = {
        "dddddddd": {"id": "d" * 32, "tags": ["til"], "type": "fact"},  # already global
        "eeeeeeee": {"id": "e" * 32, "tags": ["project:memo"], "type": "reference"},
        # "ffffffff" unresolvable → skipped
    }
    assert dr.retag_decisions(counts, get_record=records.get, min_other_projects=2) == []


@pytest.fixture
def mem_stub(tmp_cfg: Config, monkeypatch) -> Memory:
    cfg = Config(
        data_dir=tmp_cfg.data_dir,
        vault_path=tmp_cfg.vault_path,
        state_dir=tmp_cfg.state_dir,
        embedder_dims=4,
    )
    monkeypatch.setattr(
        "memo.embedder.MLXEmbedder.embed",
        lambda self, inputs: [[1.0, 0.0, 0.0, 0.0] for _ in inputs],
    )
    monkeypatch.setenv("MEMO_EMBEDDER_DIMS", "4")
    m = Memory(cfg)
    yield m
    m.close()


def test_run_retag_global_promotes_and_never_reembeds(mem_stub: Memory, monkeypatch):
    rec = mem_stub.save(
        content="lesson that turned out to be general",
        title="General lesson",
        tags=["project:memo", "til"],
        auto_project=False,
    )
    keep = mem_stub.save(
        content="memo-only detail", title="Local", tags=["project:memo"], auto_project=False
    )
    state_dir = mem_stub.cfg.state_dir
    for i, proj in enumerate(["project:synapse", "project:memflow"]):
        append_grounding_log(
            state_dir,
            session_id=f"s{i}",
            turn=1,
            recall_id=rec.id,
            used_score=0.9,
            method="embed",
            project=proj,
        )
    append_grounding_log(
        state_dir,
        session_id="s9",
        turn=1,
        recall_id=keep.id,
        used_score=0.9,
        method="embed",
        project="project:synapse",  # only ONE other project → keep
    )

    # A pure retag must never touch the embedder — poison it AFTER setup.
    def _boom(self, inputs):
        raise AssertionError("re-embed on pure retag path")

    monkeypatch.setattr("memo.embedder.MLXEmbedder.embed", _boom)

    res = dr.run_retag_global(mem_stub.cfg, mem_stub, min_other_projects=2, dry_run=False)

    assert [r["id"] for r in res["retagged"]] == [rec.id]
    assert res["retagged"][0]["status"] == "retagged"
    promoted = mem_stub.get(rec.id)
    assert not any(t.startswith("project:") for t in promoted.tags)
    assert "til" in promoted.tags
    assert "project:memo" in mem_stub.get(keep.id).tags  # untouched


def test_run_retag_global_dry_run_writes_nothing(mem_stub: Memory):
    rec = mem_stub.save(
        content="general lesson", title="L", tags=["project:memo"], auto_project=False
    )
    for i, proj in enumerate(["project:synapse", "project:memflow"]):
        append_grounding_log(
            mem_stub.cfg.state_dir,
            session_id=f"s{i}",
            turn=1,
            recall_id=rec.id,
            used_score=0.9,
            method="embed",
            project=proj,
        )

    res = dr.run_retag_global(mem_stub.cfg, mem_stub, min_other_projects=2, dry_run=True)

    assert res["retagged"][0]["status"] == "would_retag"
    assert "project:memo" in mem_stub.get(rec.id).tags
