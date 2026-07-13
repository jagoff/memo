from pathlib import Path

from memo import interject as ij


def test_should_render_respects_budget(tmp_path: Path):
    sd, sid = tmp_path, "sess-1"
    assert ij.should_render(sd, sid, max_per_session=1) is True
    ij.note_rendered(sd, sid)
    assert ij.should_render(sd, sid, max_per_session=1) is False  # budget spent


def test_silence_suppresses_render(tmp_path: Path):
    sd, sid = tmp_path, "sess-2"
    assert ij.should_render(sd, sid, max_per_session=5) is True
    ij.silence(sd, sid)
    assert ij.should_render(sd, sid, max_per_session=5) is False


def test_zero_budget_never_renders(tmp_path: Path):
    assert ij.should_render(tmp_path, "sess-3", max_per_session=0) is False


def test_marker_none_session_degrades(tmp_path: Path):
    # a None session_id must not crash; caller passes "_no_session"
    assert ij.should_render(tmp_path, "_no_session", max_per_session=1) is True


def test_shadow_log_roundtrip(tmp_path: Path):
    sd = tmp_path
    ij.log_shadow(sd, ij.shadow_record("switch instead", ["a" * 32], rendered=False))
    ij.log_shadow(sd, ij.shadow_record("revert that", ["b" * 32], rendered=True))
    rows = ij.read_shadow(sd)
    assert len(rows) == 2
    # newest_first
    assert rows[0]["prompt"] == "revert that"
    assert rows[0]["rendered"] is True
    assert rows[1]["rendered"] is False
