from __future__ import annotations

from memo.recall_logic import _dedup_tokens, detect_topic_shift, dynamic_stream_token_budget


def test_detect_topic_shift() -> None:
    toks1 = _dedup_tokens("how to configure postgresql database parameters for production deployment")
    toks2 = _dedup_tokens("how to configure postgresql database parameters and connection pool settings")
    toks3 = _dedup_tokens("frontend react components css styling theme design system UI elements")

    # Similar prompts (sharing many key words) -> no shift at sensitivity 0.65
    assert not detect_topic_shift(toks1, toks2, sensitivity=0.65)

    # Completely different topics -> shift detected
    assert detect_topic_shift(toks1, toks3, sensitivity=0.35)


def test_dynamic_stream_token_budget() -> None:
    base_budget = 600
    prompt_a = "how to configure postgresql database parameters for production deployment"
    prompt_b = "frontend react components css styling theme design system UI elements"

    # Turn 1 -> base budget
    b1 = dynamic_stream_token_budget(base_budget, prompt_a, turn=1)
    assert b1 == base_budget

    # Turn 2 with topic shift -> budget expanded
    b2 = dynamic_stream_token_budget(base_budget, prompt_b, turn=2, prev_prompt=prompt_a)
    assert b2 > base_budget

    # Turn 8 without topic shift -> budget decayed
    prompt_a_sub = "how to configure postgresql database parameters for production deployment with connection pool"
    b8 = dynamic_stream_token_budget(base_budget, prompt_a_sub, turn=8, prev_prompt=prompt_a)
    assert b8 < base_budget
