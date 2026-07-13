"""Tests for the `memo onboard` Day-0 wizard."""
from __future__ import annotations


def _env(tmp_path):
    return {
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_DATA_DIR": str(tmp_path / "data"),
        "MEMO_STATE_DIR": str(tmp_path / "state"),
        "MEMO_VAULT_PATH": str(tmp_path / "vault"),
        "MEMO_EMBEDDER_VIA_DAEMON": "0",
        "MEMO_SKIP_MODEL_VERSION_CHECK": "1",
    }


def test_onboard_backfill_days_flag_registered():
    from memo.flags import REGISTRY

    spec = REGISTRY["MEMO_ONBOARD_BACKFILL_DAYS"]
    assert spec.default == 90
