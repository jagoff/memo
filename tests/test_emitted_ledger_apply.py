import pytest

from memo import emitted_ledger as el
from memo import server_common as sc
from memo import server_session_patterns as ssp


class _Cfg:
    def __init__(self, state_dir):
        self.state_dir = state_dir


class _Mem:
    def __init__(self, state_dir):
        self.cfg = _Cfg(state_dir)


def _hits():
    return [
        {"id": "mem_a", "title": "A", "body": "body a"},
        {"id": "mem_b", "title": "B", "body": "body b"},
    ]


@pytest.fixture
def mem(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMO_EMITTED_LEDGER", "1")
    monkeypatch.setenv("MEMO_SESSION_ID", "sess-apply")
    return _Mem(tmp_path)


def test_flag_off_is_a_passthrough(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMO_EMITTED_LEDGER", "0")
    monkeypatch.setenv("MEMO_SESSION_ID", "sess-off")
    hits = _hits()
    out, extra = sc.apply_ledger(_Mem(tmp_path), "memo_search", hits)
    assert out == hits and extra == {}
    assert el.read(tmp_path, "sess-off") == {}


def test_tool_not_in_allowlist_is_a_passthrough(mem, tmp_path):
    hits = _hits()
    out, extra = sc.apply_ledger(mem, "memo_get", hits)
    assert out == hits and extra == {}
    assert el.read(tmp_path, "sess-apply") == {}


def test_first_call_emits_all_and_records(mem, tmp_path):
    out, extra = sc.apply_ledger(mem, "memo_search", _hits())
    assert [h["id"] for h in out] == ["mem_a", "mem_b"]
    assert extra == {}  # nothing suppressed on a cold ledger
    assert set(el.read(tmp_path, "sess-apply")) == {"mem_a", "mem_b"}


def test_second_identical_call_digests(mem):
    sc.apply_ledger(mem, "memo_search", _hits())
    out, extra = sc.apply_ledger(mem, "memo_search", _hits())
    assert out == []
    assert [e["id"] for e in extra["already_in_context"]] == ["mem_a", "mem_b"]
    assert extra["already_in_context"][0]["title"] == "A"
    assert extra["already_in_context"][0]["ref"].startswith("memo-r/")
    assert "memo_get" in extra["hint"]


def test_partial_overlap_across_tools(mem):
    sc.apply_ledger(mem, "memo_search", _hits())
    later = [*_hits(), {"id": "mem_c", "title": "C", "body": "body c"}]
    out, extra = sc.apply_ledger(mem, "memo_ask", later)
    assert [h["id"] for h in out] == ["mem_c"]
    assert [e["id"] for e in extra["already_in_context"]] == ["mem_a", "mem_b"]


def test_changed_body_is_reemitted(mem):
    sc.apply_ledger(mem, "memo_search", _hits())
    edited = [{"id": "mem_a", "title": "A", "body": "body a, edited"}]
    out, extra = sc.apply_ledger(mem, "memo_search", edited)
    assert [h["id"] for h in out] == ["mem_a"]
    assert extra == {}


def test_unwritable_state_dir_degrades_to_passthrough(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMO_EMITTED_LEDGER", "1")
    monkeypatch.setenv("MEMO_SESSION_ID", "sess-ro")
    ro = tmp_path / "ro"
    ro.mkdir()
    ro.chmod(0o500)
    try:
        hits = _hits()
        out, extra = sc.apply_ledger(_Mem(ro), "memo_search", hits)
        assert out == hits and extra == {}
    finally:
        ro.chmod(0o700)


def test_partition_raising_degrades_to_passthrough(mem, monkeypatch):
    """Fault-injection on the exception boundary itself: partition() is a pure
    function that "should never" raise, but apply_ledger must not trust that —
    a bug here must cost tokens (full bodies), never break the caller's tool
    call. Patches the emitted_ledger module attribute directly since
    apply_ledger's `from memo import emitted_ledger as el` re-binds to the
    same module object at call time."""

    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(el, "partition", boom)
    hits = _hits()
    out, extra = sc.apply_ledger(mem, "memo_search", hits)
    assert out == hits and extra == {}


def test_session_id_lookup_raising_degrades_to_passthrough(mem, monkeypatch):
    """Fault-injection on session-id resolution: `_effective_session_id`
    currently never raises in practice (it falls back to a generated id), but
    apply_ledger's contract is "no session id resolvable -> passthrough", so
    this must hold even if that assumption ever breaks."""

    def boom() -> str:
        raise RuntimeError("boom")

    monkeypatch.setattr(ssp, "_effective_session_id", boom)
    hits = _hits()
    out, extra = sc.apply_ledger(mem, "memo_search", hits)
    assert out == hits and extra == {}
