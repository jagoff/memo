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
    assert not tmp_cfg.db_path.exists()


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
    assert [s["key"] for s in g["funnel"]] == ["preguntas", "muestra", "activadas"]
    for key in (
        "consults",
        "consults_total",
        "consults_sampled",
        "coverage_rate",
        "activation_rate_total",
        "activation_rate_sampled",
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
    assert g["consults_total"] == 0
    assert g["consults_sampled"] == 0
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


def test_gerencial_historic_survives_grounding_log_rotation(tmp_cfg: Config, monkeypatch):
    """The historic tokens-saved headline must come from the durable ledger, so
    it does not shrink when old grounded rows scroll out of the capped
    grounding.log. Seed a ledger with more history than the log holds."""
    monkeypatch.setenv("MEMO_ROI_TOKENS_PER_GROUNDED", "100")
    from memo import token_ledger

    state_dir = tmp_cfg.state_dir
    state_dir.mkdir(parents=True, exist_ok=True)
    # Durable ledger remembers 50 grounded across days no longer in the log.
    token_ledger.write_ledger(
        state_dir,
        {
            "schema": token_ledger.LEDGER_SCHEMA,
            "days": {"2026-05-01": {"grounded": 30}, "2026-05-02": {"grounded": 20}},
        },
    )
    # grounding.log only still holds 1 recent grounded row.
    now = datetime.now(UTC).isoformat(timespec="seconds")
    (state_dir / "grounding.log").write_text(
        json.dumps(
            {"ts": now, "session_id": "s", "turn": 1, "recall_id": "r0000001",
             "used_score": 0.9, "method": "lexical"}
        )
        + "\n",
        encoding="utf-8",
    )

    g = build.collect_data(tmp_cfg, include_projection=False)["gerencial"]

    # Historic = durable 50 + the fresh 1 (rolled up), not just the 1 in the log.
    assert g["token_detail"]["grounded"] == 51
    assert g["tokens_saved"] == 51 * 100


def test_gerencial_reports_context_cost_and_net_tokens(tmp_cfg: Config, monkeypatch):
    monkeypatch.setenv("MEMO_ROI_TOKENS_PER_GROUNDED", "100")
    state_dir = tmp_cfg.state_dir
    state_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(UTC).isoformat(timespec="seconds")
    (state_dir / "grounding.log").write_text(
        json.dumps(
            {
                "ts": now,
                "session_id": "s-net",
                "turn": 1,
                "recall_id": "net00001",
                "used_score": 0.9,
                "method": "lexical",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (state_dir / "context_cost.log").write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {"ts": now, "kind": "recall", "chars": 200, "tokens_est": 50},
                {"ts": now, "kind": "briefing", "chars": 400, "tokens_est": 100},
            )
        )
        + "\n",
        encoding="utf-8",
    )

    g = build.collect_data(tmp_cfg, include_projection=False)["gerencial"]

    # Gross components are reported as-is for transparency...
    assert g["tokens_saved_today"] == 100
    assert g["context_tokens_today"] == 150
    assert g["token_detail"]["context_costs"] == {"briefing": 100, "recall": 50}
    # ...but "ahorro neto" floors at 0: savings are measurement-gated (only
    # grounded recalls) while context cost counts every injection, so a raw
    # net of 100 - 150 = -50 is a coverage artifact, not a real loss. A day
    # that saved nothing nets 0, never negative.
    assert g["tokens_net_today"] == 0
    assert g["tokens_net"] == 0
    assert g["token_detail"]["today_net"] == 0
    assert g["token_detail"]["net"] == 0


def test_gerencial_consults_uses_daily_trend_total(tmp_cfg: Config):
    state_dir = tmp_cfg.state_dir
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "daily_trend.json").write_text(
        json.dumps(
            {
                "2026-06-19": {"consultas": 1126, "activado": 41},
                "2026-06-20": {"consultas": 745, "activado": 24},
            }
        ),
        encoding="utf-8",
    )

    g = build.collect_data(tmp_cfg, include_projection=False)["gerencial"]

    assert g["consults"] == 1871
    assert g["consults_total"] == 1871
    assert g["consults_sampled"] == 0
    assert g["activations_total"] == 65
    assert g["coverage_rate"] == 0.035
    assert g["activation_rate_total"] == 0.035
    assert g["activation_rate_sampled"] is None


def test_verdict_exposes_total_and_sampled_consults(tmp_cfg: Config):
    state_dir = tmp_cfg.state_dir
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "daily_trend.json").write_text(
        json.dumps(
            {
                "2026-06-19": {"consultas": 1126, "activado": 41},
                "2026-06-20": {"consultas": 745, "activado": 24},
            }
        ),
        encoding="utf-8",
    )

    verdict = build.collect_data(tmp_cfg, include_projection=False)["verdict"]

    assert verdict["consults"] == 0
    assert verdict["consults_sampled"] == 0
    assert verdict["consults_total"] == 1871
    assert verdict["activations_total"] == 65


def test_full_mode_includes_projection(tmp_cfg: Config):
    data = build.collect_data(tmp_cfg, include_projection=True)
    assert isinstance(data["projection"], dict)
    assert "method" in data["projection"]


def test_collect_data_is_json_serializable(tmp_cfg: Config):
    data = build.collect_data(tmp_cfg, include_projection=False)
    json.dumps(data, ensure_ascii=False, default=str)  # must not raise


def test_consult_trend_today_survives_trimmed_recall_log(tmp_path: Path):
    """daily_trend.json is the synchronous, complete accumulator; recall.log is
    size-capped and trims to its last lines. On a busy day the trimmed log holds
    only a fraction of today's rows, so the live count under-reports today. The
    trend must take the per-field max so a trimmed recall.log never shrinks
    today's bar below the persisted accumulator."""
    today = datetime.now(UTC).date().isoformat()
    (tmp_path / "daily_trend.json").write_text(
        json.dumps({today: {"consultas": 796, "activado": 31}}), encoding="utf-8"
    )
    # recall.log retains only 3 of today's rows (the rest trimmed away).
    kept = [json.dumps({"ts": f"{today}T1{i}:00:00+00:00", "via": "daemon"}) for i in range(3)]
    (tmp_path / "recall.log").write_text("\n".join(kept) + "\n", encoding="utf-8")

    trend = build._consult_trend(tmp_path)
    bucket = next(d for d in trend if d["date"] == today)
    assert bucket["consultas"] == 796  # persisted total, not the 3 surviving rows
    assert bucket["activado"] == 31
