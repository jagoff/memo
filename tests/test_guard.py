from types import SimpleNamespace

from memo.guard import guard_candidates, has_reversal_signal


def _hit(id_, type_, score):
    return SimpleNamespace(id=id_, type=type_, score=score, title="t", body="b")


def test_reversal_signal_detects_english_and_spanish():
    assert has_reversal_signal("let's use X instead of Y")
    assert has_reversal_signal("actually, switch to redis")
    assert has_reversal_signal("cambiémoslo, en vez de eso usemos foo")
    assert not has_reversal_signal("how do I add a new endpoint?")


def test_guard_candidates_filters_type_score_and_signal():
    hits = [
        _hit("a", "decision", 0.9),   # decision + high score → candidate
        _hit("b", "note", 0.95),      # wrong type → excluded
        _hit("c", "preference", 0.4), # below threshold → excluded
        _hit("d", "preference", 0.8), # preference + high score → candidate
    ]
    out = guard_candidates("switch to X instead", hits, sim_threshold=0.6)
    assert [h.id for h in out] == ["a", "d"]  # score-desc, only decision/preference over threshold


def test_guard_candidates_empty_when_no_reversal_signal():
    hits = [_hit("a", "decision", 0.9)]
    assert guard_candidates("how does recall work?", hits, sim_threshold=0.6) == []
