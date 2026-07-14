import logging
import sqlite3
from pathlib import Path

from memo import confidence_calibration as cc

# Ordinal rank for a band NAME (not the observed rate) — used to assert the
# map itself is non-decreasing in the canonical low -> med -> high order.
# Plain string comparison ("high" <= "low") is NOT semantically meaningful
# here (alphabetical order != band order), so tests compare via this rank.
_RANK = {"low": 0, "med": 1, "high": 2}


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
        return {i: {"confidence": self._conf[i], "roi_score": 1.0} for i in ids if i in self._conf}


class _Mem:
    def __init__(self, conf):
        self.store = _Store(conf)


def _write_grounding(sd: Path, rows):
    from memo.dashboard_logs import append_grounding_log

    for r in rows:
        append_grounding_log(
            sd,
            session_id=r["sid"],
            turn=r["turn"],
            recall_id=r["rid"],
            used_score=r["used"],
            method="test",
        )


def test_build_calibration_bins_by_confidence_and_use(tmp_path: Path):
    # two high-confidence memories grounded; two low-confidence never grounded.
    conf = {"aaaaaaaa11": 0.9, "bbbbbbbb22": 0.9, "cccccccc33": 0.2, "dddddddd44": 0.2}
    _write_grounding(
        tmp_path,
        [
            {"sid": "s", "turn": 1, "rid": "aaaaaaaa11", "used": 0.95},
            {"sid": "s", "turn": 2, "rid": "bbbbbbbb22", "used": 0.90},
            {"sid": "s", "turn": 3, "rid": "cccccccc33", "used": 0.05},
            {"sid": "s", "turn": 4, "rid": "dddddddd44", "used": 0.02},
        ],
    )
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


def test_map_is_monotonic_for_middle_spike():
    # Adversarial middle spike: med grounds far better than both its
    # neighbors. The pairwise-if implementation this regresses against
    # produced low=0.3, med=0.9, high=0.3 (high demoted straight to low),
    # i.e. med > high -- NON-monotonic. The map's band-NAME values, read in
    # canonical low->med->high order, must never regress in rank.
    bins = {
        "low": {"predicted": 0.25, "observed": 0.3, "n": 10},
        "med": {"predicted": 0.6, "observed": 0.9, "n": 10},
        "high": {"predicted": 0.9, "observed": 0.2, "n": 10},
    }
    m = cc._monotonic_map(bins, min_bin=1)
    ranks = [_RANK[m["low"]], _RANK[m["med"]], _RANK[m["high"]]]
    assert ranks[0] <= ranks[1] <= ranks[2]


def test_map_is_monotonic_for_decreasing_sequence():
    # Fully decreasing raw observed rates across all three bands.
    bins = {
        "low": {"predicted": 0.25, "observed": 0.9, "n": 10},
        "med": {"predicted": 0.6, "observed": 0.5, "n": 10},
        "high": {"predicted": 0.9, "observed": 0.1, "n": 10},
    }
    m = cc._monotonic_map(bins, min_bin=1)
    ranks = [_RANK[m["low"]], _RANK[m["med"]], _RANK[m["high"]]]
    assert ranks[0] <= ranks[1] <= ranks[2]


def test_sparse_band_falls_back_without_breaking_monotonicity():
    # 'low' is under-observed (n=1 < min_bin=5) with a spuriously high raw
    # rate that -- if trusted -- would outrank both neighbors. The min-bin
    # guard must be applied BEFORE the monotone pass (not per-pair after),
    # so a sparse band can never be the source of a violation.
    bins = {
        "low": {"predicted": 0.25, "observed": 0.99, "n": 1},
        "med": {"predicted": 0.6, "observed": 0.5, "n": 10},
        "high": {"predicted": 0.9, "observed": 0.9, "n": 10},
    }
    m = cc._monotonic_map(bins, min_bin=5)
    ranks = [_RANK[m["low"]], _RANK[m["med"]], _RANK[m["high"]]]
    assert ranks[0] <= ranks[1] <= ranks[2]


def test_join_failure_is_not_silently_identity(caplog, tmp_path: Path):
    # A genuinely broken store.all_ids() must be observable (logged), not
    # silently swallowed into a plausible-looking identity map.
    class _BrokenStore:
        def all_ids(self):
            raise AttributeError("boom: no such column")

        def get_health_batch(self, ids):
            return {}

    class _BrokenMem:
        def __init__(self):
            self.store = _BrokenStore()

    with caplog.at_level(logging.WARNING, logger="memo.confidence_calibration"):
        doc = cc.build_calibration(tmp_path, _BrokenMem(), min_bin=1)

    assert any("all_ids" in r.getMessage() for r in caplog.records)
    # still returns a safe identity map (no rows resolve), but the failure
    # itself is now visible in the logs rather than silently masked.
    assert doc["map"] == {"low": "low", "med": "med", "high": "high"}


def test_join_failure_sqlite_error_is_logged_not_raised(caplog, tmp_path: Path):
    # A genuine store.all_ids() sqlite3.OperationalError (e.g. "db locked") must
    # be caught and logged, not propagated unhandled.
    class _SqliteFailStore:
        def all_ids(self):
            raise sqlite3.OperationalError("db locked")

        def get_health_batch(self, ids):
            return {}

    class _SqliteFailMem:
        def __init__(self):
            self.store = _SqliteFailStore()

    with caplog.at_level(logging.WARNING, logger="memo.confidence_calibration"):
        doc = cc.build_calibration(tmp_path, _SqliteFailMem(), min_bin=1)

    assert any("all_ids" in r.getMessage() for r in caplog.records)
    # still returns a safe identity map (no rows resolve), but the failure
    # itself is now visible in the logs rather than silently masked.
    assert doc["map"] == {"low": "low", "med": "med", "high": "high"}
