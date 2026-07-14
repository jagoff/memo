from types import SimpleNamespace

from memo import ask_gaps as ag
from memo import interject as ij


def test_calibrated_band_vocabulary_is_high_med_low():
    from memo.confidence_calibration import _BANDS

    assert set(_BANDS) == {"high", "med", "low"}  # not "medium"
    assert "medium" not in _BANDS


def test_interject_gate_uses_high_not_medium():
    # band_of returning "medium" (a typo) must NOT fire — only exact "high" does
    hit = SimpleNamespace(id="a" * 32, title="T", type="decision", score=0.9)
    fired = ij.interject_candidates(
        "switch instead",
        [hit],
        sim_threshold=0.6,
        band_of=lambda h: "medium",
        disputed_ids={"a" * 32},
    )
    assert fired == []  # "medium" != "high"
    fired2 = ij.interject_candidates(
        "switch instead",
        [hit],
        sim_threshold=0.6,
        band_of=lambda h: "high",
        disputed_ids={"a" * 32},
    )
    assert [c.id for c in fired2] == ["a" * 32]


def test_contradiction_status_strings_are_valid():
    from memo.contradict import VALID_STATUSES

    # evaluate_and_render queries exactly these two statuses
    assert "open" in VALID_STATUSES
    assert "competing" in VALID_STATUSES


def test_interject_shadow_record_keys_roundtrip(tmp_path):
    ij.log_shadow(tmp_path, ij.shadow_record("p", ["a" * 32], rendered=True))
    row = ij.read_shadow(tmp_path)[0]
    assert set(row) == {"ts", "prompt", "ids", "rendered"}


def test_ask_shadow_record_keys_roundtrip(tmp_path):
    ag.log_shadow(tmp_path, ag.shadow_record({"prompt": "q", "count": 2}, rendered=False))
    row = ag.read_shadow(tmp_path)[0]
    assert set(row) == {"ts", "prompt", "count", "rendered"}


def test_interject_header_distinct_from_guard():
    from memo.guard import guard_banner

    hit = SimpleNamespace(id="a" * 32, title="Use vec", type="decision", score=0.9)
    gb = guard_banner("switch instead", [hit], sim_threshold=0.6)
    assert gb is not None
    assert ij.INTERJECT_HEADER not in gb  # distinct headers
    assert not gb.startswith(ij.INTERJECT_HEADER)
