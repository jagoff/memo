from types import SimpleNamespace

from memo import interject as ij


def _hit(id_, title, type_, score):
    return SimpleNamespace(id=id_, title=title, type=type_, score=score)


def test_interject_fires_only_on_high_band_and_disputed():
    hits = [
        _hit("a" * 32, "Use vec mode", "decision", 0.9),  # high + disputed  -> fires
        _hit("b" * 32, "Prefer tabs", "preference", 0.9),  # high but NOT disputed -> no
        _hit("c" * 32, "Old note", "decision", 0.55),  # disputed but LOW band -> no
    ]
    band = {"a" * 32: "high", "b" * 32: "high", "c" * 32: "low"}
    disputed = {"a" * 32, "c" * 32}
    cands = ij.interject_candidates(
        "let's switch to hybrid instead",
        hits,
        sim_threshold=0.6,
        band_of=lambda h: band[h.id],
        disputed_ids=disputed,
    )
    assert [c.id for c in cands] == ["a" * 32]


def test_interject_empty_without_reversal_signal():
    # guard_candidates returns [] with no reversal signal -> interject empty too
    hits = [_hit("a" * 32, "Use vec mode", "decision", 0.9)]
    cands = ij.interject_candidates(
        "how does recall work",
        hits,
        sim_threshold=0.6,
        band_of=lambda h: "high",
        disputed_ids={"a" * 32},
    )
    assert cands == []


def test_interject_banner_names_decision_and_dispute():
    hits = [_hit("a" * 32, "Use vec mode", "decision", 0.9)]
    banner = ij.interject_banner(
        "switch to hybrid instead",
        hits,
        sim_threshold=0.6,
        band_of=lambda h: "high",
        disputed_ids={"a" * 32},
    )
    assert banner is not None
    assert banner.startswith(ij.INTERJECT_HEADER)
    assert "a" * 32 in banner or ("a" * 8) in banner
    assert "Use vec mode" in banner


def test_interject_banner_none_when_not_disputed():
    hits = [_hit("a" * 32, "Use vec mode", "decision", 0.9)]
    banner = ij.interject_banner(
        "switch instead",
        hits,
        sim_threshold=0.6,
        band_of=lambda h: "high",
        disputed_ids=set(),
    )
    assert banner is None


def test_shadow_record_shape():
    rec = ij.shadow_record("switch instead", ["a" * 32], rendered=False)
    assert rec["prompt"] == "switch instead"
    assert rec["ids"] == ["a" * 32]
    assert rec["rendered"] is False
    assert "ts" in rec
