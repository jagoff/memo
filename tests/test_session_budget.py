from memo.cli_recall_hook import session_budget_scale


def test_no_decay_below_session_budget():
    # cumulative under budget → base unchanged
    assert session_budget_scale(cumulative=300, session_budget=2000, base_budget=600) == 600


def test_decays_but_never_below_floor():
    # over budget → halve, floored at 150
    assert session_budget_scale(cumulative=2500, session_budget=2000, base_budget=600) == 300
    assert session_budget_scale(cumulative=9999, session_budget=2000, base_budget=200) == 150


def test_disabled_when_session_budget_zero():
    assert session_budget_scale(cumulative=9999, session_budget=0, base_budget=600) == 600
