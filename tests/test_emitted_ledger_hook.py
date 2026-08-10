from dataclasses import dataclass

from memo import recall_logic as rl


@dataclass
class _Hit:
    id: str
    title: str
    body: str
    score: float
    tags: tuple[str, ...] = ()


def _hits():
    return [
        _Hit("aaaaaaaa", "First", "x" * 900, 0.80),
        _Hit("bbbbbbbb", "Second", "short body", 0.70),
    ]


def test_sink_records_what_the_full_renderer_emitted():
    sink: list[tuple[str, str]] = []
    out = rl.render_by_format(
        "full", _hits(), [], turn=1, body_chars=400, token_budget=0, emitted_sink=sink
    )
    assert [i for i, _ in sink] == ["aaaaaaaa", "bbbbbbbb"]
    recorded = dict(sink)
    # truncated to the cap, not the stored 900 chars
    assert len(recorded["aaaaaaaa"]) <= 420
    assert recorded["aaaaaaaa"] in out or recorded["aaaaaaaa"].rstrip("…") in out
    assert recorded["bbbbbbbb"] == "short body"


def test_sink_records_empty_body_for_the_compact_renderer():
    sink: list[tuple[str, str]] = []
    rl.render_by_format(
        "compact", _hits(), [], turn=1, body_chars=400, token_budget=0, emitted_sink=sink
    )
    assert [i for i, _ in sink] == ["aaaaaaaa", "bbbbbbbb"]
    assert all(body == "" for _, body in sink)


def test_sink_omits_hits_whose_body_was_dropped_by_the_char_budget():
    sink: list[tuple[str, str]] = []
    rl.render_by_format(
        "full", _hits(), [], turn=1, body_chars=400, token_budget=20, emitted_sink=sink
    )
    for _id, body in sink:
        assert body == "" or len(body) < 400


def test_sink_is_optional_and_default_none_changes_nothing():
    with_sink = rl.render_by_format(
        "full", _hits(), [], turn=1, body_chars=400, token_budget=0, emitted_sink=[]
    )
    without = rl.render_by_format("full", _hits(), [], turn=1, body_chars=400, token_budget=0)
    assert with_sink == without
