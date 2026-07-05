"""Next-turn verdict classifier — heuristic ES+EN reaction classification."""

from __future__ import annotations

import pytest

from memo.verdict import classify_reaction


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # corrections — ES + EN
        ("no, eso está mal — el daemon usa el puerto 8765", "correction"),
        ("that's wrong, the flag lives in flags_search.py", "correction"),
        ("te equivocaste, era el otro archivo", "correction"),
        ("not what I asked — I meant the sync tier", "correction"),
        ("no es así, el overlay pisa el default no el env", "correction"),
        # negative outcomes
        ("sigue fallando con el mismo error", "negative"),
        ("still failing after that change", "negative"),
        ("no funciona, tira ImportError", "negative"),
        ("doesn't work — same traceback", "negative"),
        # acceptance
        ("perfecto, gracias!", "positive"),
        ("that worked, thanks", "positive"),
        ("genial, ya funciona", "positive"),
        ("works now. next: add the docs", "positive"),
        # neutral — no verdict
        ("ahora agregá tests para el módulo de sync", None),
        ("cómo funciona el recall de memo?", None),
        ("explain how it works under the hood", None),
        ("", None),
    ],
)
def test_classify_reaction(text: str, expected: str | None) -> None:
    assert classify_reaction(text) == expected


def test_negation_head_beats_thanks() -> None:
    # "no, gracias — eso no era lo que pedí" is a rejection, not a thank-you.
    assert classify_reaction("no, gracias — eso no era lo que pedí") == "correction"


def test_signal_must_be_in_head() -> None:
    # A long new request that happens to end in "gracias" is NOT a reaction.
    filler = "quiero que refactorices este módulo largo y agregues validación. " * 5
    assert classify_reaction(filler + "gracias") is None
