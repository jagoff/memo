"""dream_retag — cross-project grounding counts + retag decisions (pure core)."""

from __future__ import annotations

from memo import dream_retag as dr


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
