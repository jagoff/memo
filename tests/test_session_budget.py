from memo.cli_recall_hook import session_budget_scale


def test_full_budget_before_anything_is_spent():
    """Zero spend is the only case that still returns `base_budget` intact.

    The curve was a step function -- flat at base until `cumulative` reached
    `session_budget`, then halved. It is now a smooth linear decay from 1.0 to
    0.5 as spend goes from 0 to 2x`session_budget` (see
    `session_budget_scale`'s docstring: "Replaces the old step-function
    halving"). The cliff is gone, and so is the flat region: a session now
    starts giving up recall budget from its FIRST turn, not once it has spent
    its allowance. That is a deliberate product change, not a rounding detail.
    """
    assert session_budget_scale(cumulative=0, session_budget=2000, base_budget=600) == 600


def test_decay_is_proportional_to_spend():
    # 300/4000 of the way through the ramp -> 600 * (1 - 0.5*0.075) = 577
    assert session_budget_scale(cumulative=300, session_budget=2000, base_budget=600) == 577
    # 2500/4000 -> 600 * (1 - 0.5*0.625) = 412; the old step function said 300
    assert session_budget_scale(cumulative=2500, session_budget=2000, base_budget=600) == 412


def test_never_below_half_base_nor_the_floor():
    """The two guarantees the new curve keeps."""
    # Ratio saturates at 1.0, so the scale bottoms out at 0.5 of base...
    assert session_budget_scale(cumulative=10**9, session_budget=2000, base_budget=600) == 300
    # ...and the absolute floor still wins for a small base.
    assert session_budget_scale(cumulative=9999, session_budget=2000, base_budget=200) == 150


def test_disabled_when_session_budget_zero():
    assert session_budget_scale(cumulative=9999, session_budget=0, base_budget=600) == 600
