"""grounding_judge: source->claim entailment scoring via a passed-in chat."""

from __future__ import annotations

from memo import grounding_judge


class _FakeChat:
    def __init__(self, content: str) -> None:
        self._content = content
        self.calls = 0

    def chat(self, *, model, messages, options=None):
        self.calls += 1
        return {"message": {"content": self._content}}


def test_parse_score_reads_leading_integer():
    assert grounding_judge.parse_score("85") == 0.85
    assert grounding_judge.parse_score("Score: 40/100") == 0.40
    assert grounding_judge.parse_score("100") == 1.0
    assert grounding_judge.parse_score("0") == 0.0


def test_parse_score_none_when_no_number():
    assert grounding_judge.parse_score("no idea") is None
    assert grounding_judge.parse_score("") is None


def test_parse_score_clamps_out_of_range():
    assert grounding_judge.parse_score("150") == 1.0


def test_score_grounding_high_when_entailed():
    chat = _FakeChat("90")
    score = grounding_judge.score_grounding(
        chat, "m", source="I switched the port to 8765.", claim="The port is 8765."
    )
    assert score == 0.90
    assert chat.calls == 1


def test_score_grounding_none_on_chat_error():
    class _BoomChat:
        def chat(self, **k):
            raise RuntimeError("model down")

    assert grounding_judge.score_grounding(_BoomChat(), "m", source="s", claim="c") is None
