"""Tests for the dashboard data builder + live-refresh split (web/build.py)."""
from __future__ import annotations

import json
import sys
from datetime import UTC, datetime, timedelta
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


def test_poll_mode_includes_gerencial_block(tmp_cfg: Config):
    """The management rollup (funnel + value KPIs + trend) must be on the cheap
    poll path — it is the centerpiece of the dashboard, not a full-build extra."""
    data = build.collect_data(tmp_cfg, include_projection=False)
    g = data["gerencial"]
    assert [s["key"] for s in g["funnel"]] == ["preguntas", "activado", "encontro"]
    for key in (
        "consults",
        "coverage_rate",
        "hit_rate",
        "grounded_rate",
        "referenced_rate",
        "used_rate",
        "measurement_coverage",
        "trend",
    ):
        assert key in g, f"missing gerencial.{key}"
    assert len(g["trend"]) == 14
    assert all({"date", "consultas", "activado"} <= d.keys() for d in g["trend"])
    assert g["consults"] == 0
    assert g["grounded_rate"] is None
    assert g["referenced_rate"] is None
    # detailed token-savings: daily series + composition, KPI consistent with panel
    td = g["token_detail"]
    assert len(td["daily"]) == 14
    assert all({"date", "grounded", "tokens"} <= d.keys() for d in td["daily"])
    assert td["total"] == td["grounded_tokens"] + td["reask_tokens"]
    assert g["tokens_saved"] == td["total"]


def test_gerencial_tokens_saved_kpi_is_daily_not_accumulated(tmp_cfg: Config, monkeypatch):
    monkeypatch.setenv("MEMO_ROI_TOKENS_PER_GROUNDED", "100")
    state_dir = tmp_cfg.state_dir
    state_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.now(UTC)
    yesterday = today - timedelta(days=1)
    rows = [
        {
            "ts": yesterday.isoformat(timespec="seconds"),
            "session_id": "s-old",
            "turn": 1,
            "recall_id": "old00001",
            "used_score": 0.9,
            "method": "lexical",
        },
        {
            "ts": today.isoformat(timespec="seconds"),
            "session_id": "s-new",
            "turn": 1,
            "recall_id": "new00001",
            "used_score": 0.9,
            "method": "lexical",
        },
    ]
    (state_dir / "grounding.log").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n",
        encoding="utf-8",
    )

    data = build.collect_data(tmp_cfg, include_projection=False)

    g = data["gerencial"]
    td = g["token_detail"]
    assert g["tokens_saved_today"] == 100
    assert g["tokens_saved_today_human"] == "100"
    assert td["today_tokens"] == 100
    assert td["total"] == 200
    assert g["tokens_saved"] == 200


def test_full_mode_includes_projection(tmp_cfg: Config):
    data = build.collect_data(tmp_cfg, include_projection=True)
    assert isinstance(data["projection"], dict)
    assert "method" in data["projection"]


def test_collect_data_is_json_serializable(tmp_cfg: Config):
    data = build.collect_data(tmp_cfg, include_projection=False)
    json.dumps(data, ensure_ascii=False, default=str)  # must not raise
