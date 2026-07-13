from pathlib import Path

from memo import confidence_calibration as cc


def test_predicted_band_thresholds():
    assert cc.predicted_band(0.9) == "high"
    assert cc.predicted_band(0.6) == "med"
    assert cc.predicted_band(0.2) == "low"


class _Store:
    def __init__(self, conf: dict[str, float]):
        self._conf = conf

    def all_ids(self):
        return list(self._conf.keys())

    def get_health_batch(self, ids):
        return {i: {"confidence": self._conf[i], "roi_score": 1.0}
                for i in ids if i in self._conf}


class _Mem:
    def __init__(self, conf):
        self.store = _Store(conf)


def _write_grounding(sd: Path, rows):
    from memo.dashboard_logs import append_grounding_log
    for r in rows:
        append_grounding_log(sd, session_id=r["sid"], turn=r["turn"],
                             recall_id=r["rid"], used_score=r["used"], method="test")


def test_build_calibration_bins_by_confidence_and_use(tmp_path: Path):
    # two high-confidence memories grounded; two low-confidence never grounded.
    conf = {"aaaaaaaa11": 0.9, "bbbbbbbb22": 0.9, "cccccccc33": 0.2, "dddddddd44": 0.2}
    _write_grounding(tmp_path, [
        {"sid": "s", "turn": 1, "rid": "aaaaaaaa11", "used": 0.95},
        {"sid": "s", "turn": 2, "rid": "bbbbbbbb22", "used": 0.90},
        {"sid": "s", "turn": 3, "rid": "cccccccc33", "used": 0.05},
        {"sid": "s", "turn": 4, "rid": "dddddddd44", "used": 0.02},
    ])
    doc = cc.build_calibration(tmp_path, _Mem(conf), min_bin=1)
    assert doc["bins"]["high"]["observed"] == 1.0
    assert doc["bins"]["low"]["observed"] == 0.0
    # monotonic map keeps high >= low; identity is fine when already monotonic.
    assert doc["map"]["high"] in {"high", "med"}
    assert doc["map"]["low"] == "low"


def test_save_load_roundtrip_and_recalibrate(tmp_path: Path):
    doc = {"bins": {}, "map": {"high": "med", "low": "low"}}
    cc.save_calibration(tmp_path, doc)
    assert cc.load_calibration(tmp_path)["map"]["high"] == "med"
    # render-time lookup demotes a "high" score-band to the calibrated "med".
    assert cc.recalibrated_band(tmp_path, "high") == "med"
    # unknown band / no entry -> identity.
    assert cc.recalibrated_band(tmp_path, "med") == "med"


def test_recalibrate_identity_when_no_map(tmp_path: Path):
    assert cc.recalibrated_band(tmp_path, "high") == "high"
