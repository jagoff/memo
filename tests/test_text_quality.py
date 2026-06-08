"""Universal text-quality gate — pure, FP-free garbage detection."""

from __future__ import annotations

from memo import text_quality as tq


def test_garbage_ratio_clean_text_is_zero() -> None:
    # Legit notes — including code, identifiers, CamelCase, digits — must be ~0.
    for s in [
        "Las lambda se deployan en dev-PublicCloudInfrastructure",
        "Professional Scrum with User Experience — Metrics 4/6",
        "func extract_text(path: Path) -> tuple[str, float]:",
        "Guitar — Adam Jones, cabs, tool, adam-jones-2024",
    ]:
        assert tq.garbage_ratio(s) == 0.0, s


def test_garbage_ratio_flags_replacement_chars() -> None:
    # The original mojibake carried U+FFFD replacement chars.
    garbled = "U�t*� St•tqSllf.Vkyl��"
    assert tq.garbage_ratio(garbled) > 0.05


def test_garbage_ratio_flags_control_chars() -> None:
    assert tq.garbage_ratio("abc\x00\x01\x02\x03def") > 0.2


def test_garbage_ratio_empty() -> None:
    assert tq.garbage_ratio("") == 0.0


def test_text_health_confidence_neutral_for_clean(monkeypatch) -> None:
    monkeypatch.delenv("MEMO_TEXT_QUALITY", raising=False)
    monkeypatch.delenv("MEMO_TEXT_QUALITY_THRESHOLD", raising=False)
    assert tq.text_health_confidence("a perfectly clean legible note") is None


def test_text_health_confidence_downweights_garbled(monkeypatch) -> None:
    monkeypatch.delenv("MEMO_TEXT_QUALITY", raising=False)
    monkeypatch.delenv("MEMO_TEXT_QUALITY_THRESHOLD", raising=False)
    garbled = "����� normal words here"
    c = tq.text_health_confidence(garbled)
    assert c is not None and 0.1 <= c < 1.0


def test_text_health_confidence_floored(monkeypatch) -> None:
    monkeypatch.delenv("MEMO_TEXT_QUALITY", raising=False)
    monkeypatch.setenv("MEMO_TEXT_QUALITY_THRESHOLD", "0.01")
    assert tq.text_health_confidence("�" * 100) == 0.1


def test_text_quality_disabled(monkeypatch) -> None:
    monkeypatch.setenv("MEMO_TEXT_QUALITY", "0")
    assert tq.text_health_confidence("�" * 100) is None


def test_threshold_zero_disables(monkeypatch) -> None:
    monkeypatch.delenv("MEMO_TEXT_QUALITY", raising=False)
    monkeypatch.setenv("MEMO_TEXT_QUALITY_THRESHOLD", "0")
    assert tq.text_health_confidence("�" * 100) is None
