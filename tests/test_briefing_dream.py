"""dream_digest_lines — '☾ Last night' one-shot briefing section (F3)."""

import json
import time
from pathlib import Path

from memo.briefing import dream_digest_lines


def _write_receipt(state_dir: Path, *, ts: float, **overrides) -> None:
    receipt = {
        "ts": ts,
        "superseded": [{"id": "a"}, {"id": "b"}, {"id": "c"}],
        "merged": [{"id": "d"}, {"id": "e"}],
        "archived_stale": [],
        "synthesized": [{"id": "f"}],
        "signal_gathered": {"files_processed": 4, "memories_saved": 2, "skipped_dup": 0},
        "tuner": {"status": "noop"},
        "errors": [],
    }
    receipt.update(overrides)
    d = state_dir / "dream"
    d.mkdir(parents=True, exist_ok=True)
    (d / "last.json").write_text(json.dumps(receipt), encoding="utf-8")


def test_fresh_receipt_renders_counts(tmp_path: Path) -> None:
    _write_receipt(tmp_path, ts=time.time() - 3600)
    lines = dream_digest_lines(tmp_path)
    joined = "\n".join(lines)
    assert "☾" in joined
    assert "3 contradictions superseded" in joined
    assert "2 duplicates merged" in joined
    assert "1 synthesis" in joined
    assert "2 memories mined" in joined
    assert "tuner: noop" in joined


def test_second_call_is_empty_one_shot(tmp_path: Path) -> None:
    _write_receipt(tmp_path, ts=time.time() - 3600)
    assert dream_digest_lines(tmp_path)  # first show
    assert dream_digest_lines(tmp_path) == []  # stamped — shown once


def test_new_receipt_shows_again(tmp_path: Path) -> None:
    _write_receipt(tmp_path, ts=time.time() - 7200)
    assert dream_digest_lines(tmp_path)
    _write_receipt(tmp_path, ts=time.time() - 60)  # a NEWER run
    assert dream_digest_lines(tmp_path)


def test_old_receipt_is_skipped(tmp_path: Path) -> None:
    _write_receipt(tmp_path, ts=time.time() - 48 * 3600)
    assert dream_digest_lines(tmp_path) == []


def test_errors_are_surfaced(tmp_path: Path) -> None:
    _write_receipt(tmp_path, ts=time.time() - 3600, errors=["tuner: boom", "gc: boom"])
    joined = "\n".join(dream_digest_lines(tmp_path))
    assert "2 errors" in joined


def test_missing_or_corrupt_receipt_is_empty(tmp_path: Path) -> None:
    assert dream_digest_lines(tmp_path) == []
    d = tmp_path / "dream"
    d.mkdir(parents=True)
    (d / "last.json").write_text("{corrupt", encoding="utf-8")
    assert dream_digest_lines(tmp_path) == []


def test_clean_run_says_so(tmp_path: Path) -> None:
    _write_receipt(
        tmp_path,
        ts=time.time() - 3600,
        superseded=[],
        merged=[],
        synthesized=[],
        signal_gathered={"memories_saved": 0},
        tuner={},
    )
    joined = "\n".join(dream_digest_lines(tmp_path))
    assert "ran clean" in joined
