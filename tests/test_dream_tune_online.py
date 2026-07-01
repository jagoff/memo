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
