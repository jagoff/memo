from datetime import UTC, datetime

from memo import token_ledger
from memo.eval_baseline import SNAPSHOT_SCHEMA, build_baseline_snapshot


def test_snapshot_shape_and_param_version_base(tmp_path):
    snap = build_baseline_snapshot(
        tmp_path, {"precision_at_k": 0.2, "noise_at_k": 0.0}, now=datetime(2026, 7, 1, tzinfo=UTC)
    )
    assert snap["schema"] == SNAPSHOT_SCHEMA
    assert snap["ts"] == "2026-07-01T00:00:00+00:00"
    assert snap["params_version"] == "base"
    assert snap["offline"] == {"precision_at_k": 0.2, "noise_at_k": 0.0}
    assert snap["online"]["window_7d"] == {"grounded": 0, "tokens": 0}


def test_snapshot_online_reads_durable_ledger(tmp_path):
    today = datetime.now().astimezone().date().isoformat()
    token_ledger.write_ledger(
        tmp_path, {"schema": token_ledger.LEDGER_SCHEMA, "days": {today: {"grounded": 3}}}
    )
    snap = build_baseline_snapshot(tmp_path, {"precision_at_k": 0.0, "noise_at_k": 0.0})
    assert snap["online"]["window_7d"]["grounded"] == 3
    assert snap["online"]["window_7d"]["tokens"] > 0  # 3 * tokens-per-grounded
