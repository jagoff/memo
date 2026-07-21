import json
import threading

from memo.dashboard_logs import (
    _write_jsonl_entry,
    append_grounding_log,
    read_grounding_log,
    recall_log_path,
)


def test_grounding_row_stamped_base_without_overlay(tmp_path):
    append_grounding_log(
        tmp_path, session_id="s1", turn=1, recall_id="abc1234567", used_score=0.9, method="lexical"
    )
    rows = read_grounding_log(tmp_path)
    assert len(rows) == 1
    assert rows[0]["params_version"] == "base"


def test_grounding_row_stamped_with_active_overlay(tmp_path):
    from memo.tuned_overlay import params_version, write_overlay

    write_overlay(tmp_path, {"MEMO_RECALL_MIN_SIM": 0.7}, {"set_by": "test"})
    append_grounding_log(
        tmp_path, session_id="s1", turn=2, recall_id="deadbeef99", used_score=0.9, method="embed"
    )
    rows = read_grounding_log(tmp_path)
    assert len(rows) == 1
    expected = params_version(tmp_path)
    assert expected != "base"
    assert rows[0]["params_version"] == expected


def test_grounding_row_carries_project_when_passed(tmp_path):
    append_grounding_log(
        tmp_path,
        session_id="s1",
        turn=1,
        recall_id="abc1234567",
        used_score=0.9,
        method="lexical",
        project="project:memo",
    )
    rows = read_grounding_log(tmp_path)
    assert len(rows) == 1
    assert rows[0]["project"] == "project:memo"


def test_grounding_row_omits_project_when_none(tmp_path):
    append_grounding_log(
        tmp_path, session_id="s1", turn=1, recall_id="abc1234567", used_score=0.9, method="lexical"
    )
    rows = read_grounding_log(tmp_path)
    assert len(rows) == 1
    assert "project" not in rows[0]


def test_concurrent_appends_are_not_lost(tmp_path):
    """Shared-data_dir invariant: concurrent sessions append to one JSONL. With
    the flock serializing append+trim, no write is dropped and no line is
    interleaved/corrupted. (cap high → no trim, so every append must survive.)"""
    path = recall_log_path(tmp_path)
    n_threads, per_thread = 8, 50

    def worker(tid):
        for i in range(per_thread):
            _write_jsonl_entry(path, {"tid": tid, "i": i}, cap=100_000, size_limit=100_000_000)

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == n_threads * per_thread  # no lost writes
    seen = {(json.loads(ln)["tid"], json.loads(ln)["i"]) for ln in lines}  # each parses
    assert len(seen) == n_threads * per_thread  # every (tid, i) exactly once


def test_concurrent_trim_keeps_valid_jsonl(tmp_path):
    """With a tiny size cap forcing constant trims, the read→truncate→rewrite
    window is under the same lock as the append, so no concurrent appender ever
    leaves a half-written line and the file stays bounded near `cap`."""
    path = recall_log_path(tmp_path)
    cap = 20

    def worker(tid):
        for i in range(60):
            _write_jsonl_entry(
                path,
                {"tid": tid, "i": i, "pad": "x" * 200},
                cap=cap,
                size_limit=2_000,  # small → trim fires on nearly every append
            )

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) <= cap  # trim keeps it bounded
    for ln in lines:
        json.loads(ln)  # no corruption from a truncate landing mid-append
