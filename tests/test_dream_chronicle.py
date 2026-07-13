"""Tests for the nightly chronicle dream pass."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path


class _Cfg:
    """Minimal cfg fake — same shape test_dream_profile.py uses."""

    def __init__(self, tmp_path):
        self.memory_dir = tmp_path / "memories"
        self.state_dir = tmp_path / "state"
        self.helper_model = "stub-model"


def _mk_cfg(tmp_path):
    cfg = _Cfg(tmp_path)
    cfg.state_dir.mkdir(parents=True, exist_ok=True)
    return cfg


def test_chronicle_flags_registered_default_off():
    from memo.flags import REGISTRY

    for name in ("MEMO_DREAM_CHRONICLE_ENABLED", "MEMO_CHRONICLE_WEEKLY"):
        assert name in REGISTRY
        assert REGISTRY[name].default is False


def test_chronicle_path_lives_in_underscore_bucket(tmp_path):
    from memo import dream_chronicle as dc

    cfg = _mk_cfg(tmp_path)
    p = dc.chronicle_path(cfg, "2026-07-13")
    assert p == Path(cfg.memory_dir) / "_chronicle" / "2026-07-13.md"


def test_default_day_is_previous_day_before_6am():
    from memo import dream_chronicle as dc

    # dream corre 03:00 — la crónica es del día que acaba de terminar
    assert dc.default_day(datetime(2026, 7, 14, 3, 0)) == "2026-07-13"
    assert dc.default_day(datetime(2026, 7, 14, 15, 0)) == "2026-07-14"


def test_filter_cited_drops_uncited_and_fabricated_bullets():
    from memo import dream_chronicle as dc

    text = (
        "## Trabajo\n"
        "- fixed the sync race [aaaaaaaa]\n"
        "- invented claim with no citation\n"
        "- claim citing unknown id [ffffffff]\n"
        "- two real ids [aaaaaaaa] [bbbbbbbb]\n"
    )
    out, ratio = dc.filter_cited(text, {"aaaaaaaa", "bbbbbbbb"})
    assert "sync race" in out
    assert "two real ids" in out
    assert "invented claim" not in out
    assert "unknown id" not in out
    assert "## Trabajo" in out  # headings siempre pasan
    assert ratio == 0.5  # 2 de 4 bullets sobrevivieron


def test_filter_cited_no_bullets_is_ratio_one():
    from memo import dream_chronicle as dc

    out, ratio = dc.filter_cited("just prose\n", {"aaaaaaaa"})
    assert out == "just prose\n" or out == "just prose"
    assert ratio == 1.0
