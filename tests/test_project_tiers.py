from dataclasses import dataclass, field

from memo.recall_logic import _apply_project_tiers


@dataclass(frozen=True)
class _Hit:
    id: str
    title: str = ""
    body: str = ""
    type: str = "note"
    score: float | None = None
    tags: tuple[str, ...] = field(default_factory=tuple)


def test_current_project_outranks_other_at_equal_similarity():
    hits = [
        _Hit("other", score=0.80, tags=("project:synapse",)),
        _Hit("cur", score=0.80, tags=("project:memo",)),
    ]
    out = _apply_project_tiers(hits, "project:memo", 0.25, 0.10)
    assert out[0].id == "cur"  # 0.80+0.25 beats 0.80


def test_global_preference_stays_afloat_over_other_project():
    hits = [
        _Hit("other", score=0.70, type="note", tags=("project:synapse",)),
        _Hit("pref", score=0.65, type="preference", tags=()),
    ]
    out = _apply_project_tiers(hits, "project:memo", 0.25, 0.10)
    assert out[0].id == "pref"  # 0.65+0.10=0.75 beats 0.70 (other gets +0)


def test_much_more_similar_other_still_wins_soft():
    hits = [
        _Hit("cur", score=0.60, tags=("project:memo",)),
        _Hit("other", score=0.95, tags=("project:synapse",)),
    ]
    out = _apply_project_tiers(hits, "project:memo", 0.25, 0.10)
    assert out[0].id == "other"  # 0.95 beats 0.60+0.25=0.85 (soft, not a hard filter)


def test_preference_with_current_project_tag_uses_global_tier():
    # Precedence: preference/feedback -> tier-2 even with the current project tag.
    hits = [_Hit("p", score=0.50, type="preference", tags=("project:memo",))]
    out = _apply_project_tiers(hits, "project:memo", 0.25, 0.10)
    assert out[0].score == 0.60  # +0.10 (global), NOT +0.25 (project)


def test_none_score_hits_pass_through_untouched():
    hits = [_Hit("n", score=None, tags=("project:memo",))]
    out = _apply_project_tiers(hits, "project:memo", 0.25, 0.10)
    assert out[0].score is None


# -- Task 4 wiring -----------------------------------------------------------


def test_recall_server_reexports_apply_project_tiers():
    from memo.recall_server import _apply_project_tiers

    assert callable(_apply_project_tiers)


def test_project_boost_default_is_025():
    from memo.flags import flag_float

    # default comes from the registry when unset
    assert flag_float("MEMO_RECALL_PROJECT_BOOST") == 0.25


def test_global_boost_default_is_010():
    from memo.flags import flag_float

    assert flag_float("MEMO_RECALL_GLOBAL_BOOST") == 0.10
