from memo import dream_tune_online as dto


def test_cohort_fraction_basic():
    rows = [
        {"params_version": "v1", "used_score": 0.9},   # grounded (>=0.6)
        {"params_version": "v1", "used_score": 0.1},   # not grounded
        {"params_version": "v1", "used_score": 0.7},   # grounded
        {"params_version": "v2", "used_score": 0.9},   # other cohort
        {"params_version": "v1"},                       # no used_score → ignored
    ]
    frac, n = dto.cohort_fraction(rows, "v1")
    assert n == 3
    assert frac == 2 / 3


def test_cohort_fraction_empty():
    assert dto.cohort_fraction([], "v1") == (0.0, 0)
    assert dto.cohort_fraction([{"params_version": "x", "used_score": 0.9}], "v1") == (0.0, 0)


def test_pending_roundtrip_and_clear(tmp_path):
    assert dto.read_pending(tmp_path) is None
    dto.write_pending(tmp_path, {"version_after": "v2", "online_before": 0.5})
    assert dto.read_pending(tmp_path)["version_after"] == "v2"
    dto.clear_pending(tmp_path)
    assert dto.read_pending(tmp_path) is None
    dto.clear_pending(tmp_path)  # idempotent, no raise


def test_ledger_append_and_read_limit(tmp_path):
    assert dto.read_ledger(tmp_path) == []
    for i in range(3):
        dto.append_ledger(tmp_path, {"verdict": "confirmed", "i": i})
    entries = dto.read_ledger(tmp_path, limit=2)
    assert [e["i"] for e in entries] == [1, 2]


def test_online_fraction_reads_grounding_log(tmp_path):
    from memo.dashboard_logs import append_grounding_log
    from memo.tuned_overlay import params_version, write_overlay

    write_overlay(tmp_path, {"MEMO_RECALL_MIN_SIM": 0.7}, {"set_by": "test"})
    v = params_version(tmp_path)
    append_grounding_log(tmp_path, session_id="s", turn=1, recall_id="a1", used_score=0.9, method="lexical")
    append_grounding_log(tmp_path, session_id="s", turn=2, recall_id="a2", used_score=0.1, method="lexical")
    frac, n = dto.online_fraction(tmp_path, v)
    assert n == 2
    assert frac == 0.5


def _pending(**over):
    base = {
        "version_before": "v1",
        "version_after": "v2",
        "floor_before": 0.5,
        "floor_after": 0.6,
        "offline_before": {"precision_at_k": 0.2, "noise_at_k": 0.0},
        "offline_after": {"precision_at_k": 0.3, "noise_at_k": 0.0},
        "online_before": 0.5,
    }
    base.update(over)
    return base


def test_resolve_none(tmp_path):
    assert dto.resolve_pending(tmp_path, min_cohort=20, eps=0.02) == {"status": "none"}


def test_resolve_waiting_keeps_pending(tmp_path, monkeypatch):
    dto.write_pending(tmp_path, _pending())
    monkeypatch.setattr(dto, "online_fraction", lambda sd, v, **k: (0.9, 5))
    r = dto.resolve_pending(tmp_path, min_cohort=20, eps=0.02)
    assert r["status"] == "waiting"
    assert r["n_after"] == 5
    assert dto.read_pending(tmp_path) is not None          # kept
    assert dto.read_ledger(tmp_path) == []                 # nothing recorded


def test_resolve_confirmed(tmp_path, monkeypatch):
    dto.write_pending(tmp_path, _pending(online_before=0.5))
    monkeypatch.setattr(dto, "online_fraction", lambda sd, v, **k: (0.55, 40))
    r = dto.resolve_pending(tmp_path, min_cohort=20, eps=0.02)
    assert r["status"] == "confirmed"
    assert r["realized_delta"] == 0.05
    assert dto.read_pending(tmp_path) is None               # cleared
    led = dto.read_ledger(tmp_path)
    assert len(led) == 1 and led[0]["verdict"] == "confirmed"


def test_resolve_reverted_carries_offline_before(tmp_path, monkeypatch):
    dto.write_pending(tmp_path, _pending(online_before=0.5))
    monkeypatch.setattr(dto, "online_fraction", lambda sd, v, **k: (0.40, 40))  # -0.10 < -eps
    r = dto.resolve_pending(tmp_path, min_cohort=20, eps=0.02)
    assert r["status"] == "reverted"
    assert r["offline_before"] == {"precision_at_k": 0.2, "noise_at_k": 0.0}
    assert dto.read_pending(tmp_path) is None
    assert dto.read_ledger(tmp_path)[0]["verdict"] == "reverted"


def test_resolve_expired_on_version_drift(tmp_path, monkeypatch):
    dto.write_pending(tmp_path, _pending(version_after="v2"))
    monkeypatch.setattr(dto, "online_fraction", lambda sd, v, **k: (0.9, 3))  # cohort < min
    r = dto.resolve_pending(tmp_path, min_cohort=20, eps=0.02, live_version="vDRIFT")
    assert r["status"] == "expired"
    assert r["reason"] == "version_drift"
    assert dto.read_pending(tmp_path) is None                    # cleared
    assert dto.read_ledger(tmp_path)[0]["verdict"] == "expired"


def test_resolve_waiting_when_live_matches_version(tmp_path, monkeypatch):
    dto.write_pending(tmp_path, _pending(version_after="v2"))
    monkeypatch.setattr(dto, "online_fraction", lambda sd, v, **k: (0.9, 3))
    r = dto.resolve_pending(tmp_path, min_cohort=20, eps=0.02, live_version="v2")
    assert r["status"] == "waiting"
    assert dto.read_pending(tmp_path) is not None                # kept (still live)
    assert dto.read_ledger(tmp_path) == []
