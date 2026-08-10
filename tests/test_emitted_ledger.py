from pathlib import Path

import pytest

from memo import emitted_ledger as el


def _entry(mid: str, text: str, ref: str = "memo-r/aaaaaa", t: int = 1000) -> el.Entry:
    return el.Entry(id=mid, h=el.emitted_hash(text), n=len(text), ref=ref, t=t, src="mcp")


def test_append_then_read_roundtrip(tmp_path: Path):
    el.append(tmp_path, "sess1", [_entry("mem_a", "hello"), _entry("mem_b", "world")])
    got = el.read(tmp_path, "sess1")
    assert set(got) == {"mem_a", "mem_b"}
    assert got["mem_a"].n == 5
    assert got["mem_a"].h == el.emitted_hash("hello")


def test_read_missing_file_is_empty(tmp_path: Path):
    assert el.read(tmp_path, "nope") == {}


def test_last_entry_per_id_wins(tmp_path: Path):
    el.append(tmp_path, "s", [_entry("mem_a", "short", t=1)])
    el.append(tmp_path, "s", [_entry("mem_a", "much longer body", t=2)])
    got = el.read(tmp_path, "s")
    assert got["mem_a"].n == len("much longer body")
    assert got["mem_a"].t == 2


def test_torn_final_line_is_skipped(tmp_path: Path):
    el.append(tmp_path, "s", [_entry("mem_a", "hello")])
    path = el.ledger_path(tmp_path, "s")
    with path.open("a", encoding="utf-8") as fh:
        fh.write('{"id":"mem_b","h":"dead')  # no newline, truncated JSON
    got = el.read(tmp_path, "s")
    assert set(got) == {"mem_a"}


def test_cap_is_fifo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MEMO_EMIT_LEDGER_MAX", "3")
    for i in range(5):
        el.append(tmp_path, "s", [_entry(f"mem_{i}", "x", t=i)])
    got = el.read(tmp_path, "s")
    assert set(got) == {"mem_2", "mem_3", "mem_4"}


def test_reset_removes_only_that_session(tmp_path: Path):
    el.append(tmp_path, "s1", [_entry("mem_a", "a")])
    el.append(tmp_path, "s2", [_entry("mem_b", "b")])
    assert el.reset(tmp_path, "s1") is True
    assert el.read(tmp_path, "s1") == {}
    assert set(el.read(tmp_path, "s2")) == {"mem_b"}


def test_reset_is_idempotent(tmp_path: Path):
    assert el.reset(tmp_path, "never-existed") is False
    el.append(tmp_path, "s", [_entry("mem_a", "a")])
    assert el.reset(tmp_path, "s") is True
    assert el.reset(tmp_path, "s") is False


def test_append_is_fail_open_on_unwritable_dir(tmp_path: Path):
    target = tmp_path / "ro"
    target.mkdir()
    target.chmod(0o500)
    try:
        el.append(target, "s", [_entry("mem_a", "a")])  # must not raise
        assert el.read(target, "s") == {}
    finally:
        target.chmod(0o700)


def test_session_id_is_sanitised(tmp_path: Path):
    el.append(tmp_path, "../escape/../../etc", [_entry("mem_a", "a")])
    written = list((tmp_path / "emitted").glob("*.jsonl"))
    assert len(written) == 1
    assert ".." not in written[0].name and "/" not in written[0].name


def test_prune_removes_only_old_files(tmp_path: Path):
    import os
    import time

    el.append(tmp_path, "old", [_entry("mem_a", "a")])
    el.append(tmp_path, "new", [_entry("mem_b", "b")])
    old_path = el.ledger_path(tmp_path, "old")
    stale = time.time() - 60 * 60 * 72
    os.utime(old_path, (stale, stale))
    assert el.prune(tmp_path, max_age_s=60 * 60 * 48) == 1
    assert not old_path.exists()
    assert el.ledger_path(tmp_path, "new").exists()


def test_mint_ref_is_stable_and_order_insensitive():
    a = el.mint_ref(["mem_b", "mem_a"], 1000)
    b = el.mint_ref(["mem_a", "mem_b"], 1000)
    assert a == b
    assert a.startswith("memo-r/") and len(a) == len("memo-r/") + 6
    assert el.mint_ref(["mem_a"], 1000, prefix="memo-h").startswith("memo-h/")
