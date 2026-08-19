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
    # grounded-usage breakdown: daily series of raw counts, no fabricated
    # "tokens saved" figure (see CHANGELOG: MEMO_ROI_TOKENS_PER_* retired).
    td = g["token_detail"]
    assert len(td["daily"]) == 14
    assert all({"date", "grounded", "context_tokens"} <= d.keys() for d in td["daily"])
    assert "tokens_saved" not in g
    assert "tokens" not in td["daily"][0]


def test_gerencial_grounded_kpi_reflects_new_events_across_days(tmp_cfg: Config):
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
    assert td["grounded"] == 2
    today_key = today.date().isoformat()
    assert next(d["grounded"] for d in td["daily"] if d["date"] == today_key) == 1


def test_gerencial_historic_survives_grounding_log_rotation(tmp_cfg: Config):
    """The historic grounded headline must come from the durable ledger, so it
    does not shrink when old grounded rows scroll out of the capped
    grounding.log. Seed a ledger with more history than the log holds."""
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
            {
                "ts": now,
                "session_id": "s",
                "turn": 1,
                "recall_id": "r0000001",
                "used_score": 0.9,
                "method": "lexical",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    g = build.collect_data(tmp_cfg, include_projection=False)["gerencial"]

    # Historic = durable 50 + the fresh 1 (rolled up), not just the 1 in the log.
    assert g["token_detail"]["grounded"] == 51


def test_gerencial_reports_real_context_cost_not_a_fabricated_estimate(tmp_cfg: Config):
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

    # Context cost is real (from context_cost.log) — reported as-is.
    assert g["context_tokens_today"] == 150
    assert g["token_detail"]["context_costs"] == {"briefing": 100, "recall": 50}
    # The old "tokens saved" / "net" KPIs (grounded * hardcoded constant, minus
    # context cost) are gone — no fabricated savings figure to net against.
    assert "tokens_saved_today" not in g
    assert "tokens_net_today" not in g
    assert "tokens_net" not in g
    assert "today_net" not in g["token_detail"]
    assert "net" not in g["token_detail"]


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


_MEMVEC_DOCTOR_DB = [
    {
        "label": "memvec",
        "exists": True,
        "records": 5,
        "vec_dims": 1024,
        "integrity_check": "ok",
        "path": "/x/memvec.db",
        "size_bytes": 4096,
        "latest_memory_update": "2026-07-21",
    }
]
_SQLITE_VEC_OK = {"label": "sqlite_vec", "ok": True, "error": ""}


# -- finding #5: the body_hash drift scan is a full-build-only step ---------


def test_poll_mode_skips_body_hash_drift_scan(tmp_cfg: Config, monkeypatch):
    """The cheap poll must not re-scan the whole corpus (rglob + double
    frontmatter parse) every interval — body_hash drift is full-build only."""
    import memo.web_build as wb

    calls: list[object] = []
    monkeypatch.setattr(
        wb,
        "_body_hash_drift",
        lambda cfg: (
            calls.append(cfg) or {"checked": 0, "drifted": 0, "missing_file": 0, "untracked_md": 0}
        ),
    )
    build.collect_data(tmp_cfg, include_projection=False)
    assert calls == []


def test_full_build_runs_body_hash_drift_scan(tmp_cfg: Config, monkeypatch):
    import memo.web_build as wb

    calls: list[object] = []
    monkeypatch.setattr(
        wb,
        "_body_hash_drift",
        lambda cfg: (
            calls.append(cfg) or {"checked": 0, "drifted": 0, "missing_file": 0, "untracked_md": 0}
        ),
    )
    build.collect_data(tmp_cfg, include_projection=True)
    assert len(calls) == 1


def test_pillar_vector_db_marks_drift_unchecked_when_none():
    import memo.web_build as wb

    doctor = {"db": _MEMVEC_DOCTOR_DB, "imports": [_SQLITE_VEC_OK]}
    pillar = wb._pillar_vector_db(doctor, None)
    assert pillar["status"] == "green"
    assert any("not checked" in d for d in pillar["detail"])


def test_pillar_vector_db_flags_drift_when_present():
    import memo.web_build as wb

    doctor = {"db": _MEMVEC_DOCTOR_DB, "imports": [_SQLITE_VEC_OK]}
    drift = {"checked": 5, "drifted": 2, "missing_file": 0, "untracked_md": 1}
    pillar = wb._pillar_vector_db(doctor, drift)
    assert pillar["status"] == "yellow"


# -- finding #3: backend-aware embedder probe (no false-RED on CPU) ---------


def test_imports_probe_is_backend_aware_for_st(tmp_cfg: Config, monkeypatch):
    import memo.web_build as wb

    monkeypatch.setattr(wb, "resolve_backend", lambda cfg: "st")
    monkeypatch.setattr(wb, "_probe_sentence_transformers", lambda: None)
    labels = {p["label"] for p in wb._imports_probe(tmp_cfg)}
    assert "sentence_transformers" in labels
    assert "mlx" not in labels  # the CPU install is not probed for MLX


def test_imports_probe_surfaces_real_error(tmp_cfg: Config, monkeypatch):
    import memo.web_build as wb

    monkeypatch.setattr(wb, "resolve_backend", lambda cfg: "mlx")

    def boom() -> None:
        raise ImportError("No module named 'mlx'")

    monkeypatch.setattr(wb, "_probe_mlx", boom)
    mlx = next(p for p in wb._imports_probe(tmp_cfg) if p["label"] == "mlx")
    assert mlx["ok"] is False
    assert "No module named 'mlx'" in mlx["error"]  # not swallowed as "probe unavailable"


def _healthy_profile() -> dict:
    return {
        "ok": True,
        "status": "ok",
        "profile": "cpu",
        "active": {
            "embedder_model": "BAAI/bge-small",
            "embedder_dims": 384,
            "llm_model": None,
            "helper_model": None,
            "reranker_model": None,
        },
        "models": [{"cached": True, "role": "embedder"}],
    }


def test_pillar_embedder_green_on_healthy_cpu_backend():
    import memo.web_build as wb

    doctor = {
        "profile": _healthy_profile(),
        "imports": [
            _SQLITE_VEC_OK,
            {"label": "sentence_transformers", "ok": True, "error": ""},
        ],
    }
    pillar = wb._pillar_embedder(doctor)
    assert pillar["status"] == "green"  # healthy CPU install is NOT false-RED
    assert pillar["label"] == "Embedder (CPU)"


def test_pillar_embedder_red_surfaces_real_import_error():
    import memo.web_build as wb

    doctor = {
        "profile": _healthy_profile(),
        "imports": [
            _SQLITE_VEC_OK,
            {"label": "mlx", "ok": False, "error": "No module named 'mlx'"},
        ],
    }
    pillar = wb._pillar_embedder(doctor)
    assert pillar["status"] == "red"
    assert "No module named 'mlx'" in pillar["summary"]  # real error, not swallowed


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


def test_html_template_has_no_fabricated_token_savings_section():
    """Round-2 fix: the 'Ahorro de tokens' panel + its KPI tile rendered the
    retired MEMO_ROI_TOKENS_PER_* estimate as a literal fabricated 0 behind
    `|| 0` JS fallbacks — deleted, not gated, so no code path can regress into
    printing a fake savings number again."""
    from memo import web_build

    html = web_build._render_html({"generated_at": "x", "memo_version": "x"})
    assert "Ahorro de tokens" not in html
    assert "Ahorro neto de tokens hoy" not in html
    assert "tok-total" not in html
    assert "Estimación con supuestos explícitos" not in html
