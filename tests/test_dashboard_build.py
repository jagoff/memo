"""Tests for the dashboard data builder + live-refresh split (web/build.py)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

_WEB = Path(__file__).resolve().parents[1] / "web"
if str(_WEB) not in sys.path:
    sys.path.insert(0, str(_WEB))

import build  # noqa: E402

from memo.config import Config  # noqa: E402


def test_poll_mode_skips_projection(tmp_cfg: Config):
    """The live-refresh path must omit the expensive 3-D projection."""
    data = build.collect_data(tmp_cfg, include_projection=False)
    assert data["projection"] is None
    assert data["type_counts"] == {}


def test_poll_mode_keeps_cheap_metrics(tmp_cfg: Config):
    data = build.collect_data(tmp_cfg, include_projection=False)
    for key in ("pillars", "recall_util", "usefulness", "verdict", "bail_breakdown"):
        assert key in data, f"missing {key}"
    assert "consumers" in data["usefulness"]
    assert "silent" in data["usefulness"]


def test_full_mode_includes_projection(tmp_cfg: Config):
    data = build.collect_data(tmp_cfg, include_projection=True)
    assert isinstance(data["projection"], dict)
    assert "method" in data["projection"]


def test_collect_data_is_json_serializable(tmp_cfg: Config):
    data = build.collect_data(tmp_cfg, include_projection=False)
    json.dumps(data, ensure_ascii=False, default=str)  # must not raise
