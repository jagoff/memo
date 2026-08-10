import json
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
    """read()'s own lines[-cap:] slice caps what comes back even when the
    on-disk file hasn't been rewritten by _trim yet (5 appends, cap=3, well
    under the amortised cap*2=6 rewrite threshold)."""
    monkeypatch.setenv("MEMO_EMITTED_LEDGER_MAX", "3")
    for i in range(5):
        el.append(tmp_path, "s", [_entry(f"mem_{i}", "x", t=i)])
    got = el.read(tmp_path, "s")
    assert set(got) == {"mem_2", "mem_3", "mem_4"}


def test_trim_rewrites_file_on_disk_past_double_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Distinct from test_cap_is_fifo: this proves _trim itself fires and
    rewrites the file, not just that read() slices what it gets back. cap=3,
    so the amortised threshold (cap*2=6) is crossed by the 7th append; asserts
    directly on the bytes on disk, bypassing read()'s own capping."""
    monkeypatch.setenv("MEMO_EMITTED_LEDGER_MAX", "3")
    for i in range(10):
        el.append(tmp_path, "s", [_entry(f"mem_{i}", "x", t=i)])
    path = el.ledger_path(tmp_path, "s")
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) < 10  # unbounded growth would mean _trim never ran
    ids = [json.loads(line)["id"] for line in lines]
    assert ids == [f"mem_{i}" for i in range(4, 10)]


def test_trim_temp_filename_is_process_specific(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Pins the carried Task 1 review fix: _trim's temp file must be named with
    our own pid, not a fixed `path.name + ".tmp"`. A fixed name would let two
    processes racing into _trim interleave writes into one shared temp path
    before either reached os.replace. Spies on os.replace (as _trim calls it)
    to capture the temp path actually used, then delegates to the real
    implementation so the trim still completes normally."""
    import os

    monkeypatch.setenv("MEMO_EMITTED_LEDGER_MAX", "3")
    captured_tmp_names: list[str] = []
    real_replace = os.replace

    def spy_replace(src: object, dst: object) -> None:
        captured_tmp_names.append(Path(str(src)).name)
        real_replace(src, dst)

    monkeypatch.setattr(el.os, "replace", spy_replace)
    for i in range(10):
        el.append(tmp_path, "s", [_entry(f"mem_{i}", "x", t=i)])

    assert captured_tmp_names, "expected _trim to fire and call os.replace"
    for name in captured_tmp_names:
        assert name.endswith(f".{os.getpid()}.tmp")


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


def test_emitted_hash_is_eight_hex_chars():
    h = el.emitted_hash("hello world")
    assert len(h) == 8
    assert all(c in "0123456789abcdef" for c in h)


def test_emitted_hash_differs_for_different_text():
    assert el.emitted_hash("hello") != el.emitted_hash("world")


def test_entry_for_text_derives_h_and_n_from_the_same_string():
    entry = el.Entry.for_text("mem_a", "hello world", "memo-r/aaaaaa", 1000, "mcp")
    assert entry.h == el.emitted_hash("hello world")
    assert entry.n == len("hello world")


def test_read_survives_corrupt_config_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """_cap() resolves MEMO_EMITTED_LEDGER_MAX from memo's Markdown config on
    disk (memo.config_md). A corrupt config file (non-UTF-8 bytes) must not
    turn read() into a raise — the recall hook calls this inside a 5s budget
    and must stay fail-open regardless of what else is wrong in memo's config.

    Reproduction requires an existing, readable ledger file: read()'s own
    file-read is already guarded, so the bug only surfaces once execution
    reaches the unguarded `cap = _cap()` call after that first try succeeds.
    """
    el.append(tmp_path, "s", [_entry("mem_a", "hello")])

    config_home = tmp_path / "config-home"
    (config_home / "config").mkdir(parents=True)
    (config_home / "config" / "advanced-config.md").write_bytes(
        b"\xff\xfe not valid utf-8 \x80\x81"
    )
    monkeypatch.setenv("MEMO_CONFIG_DIR", str(config_home))

    got = el.read(tmp_path, "s")  # must not raise UnicodeDecodeError
    assert set(got) == {"mem_a"}
