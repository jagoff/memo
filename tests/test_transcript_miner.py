"""Transcript miner — JSONL parsing, mtime filter, resumable state."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from memo.transcript_miner import find_transcripts, iter_exchanges


def _write_jsonl(path: Path, entries: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")


def test_iter_exchanges_pairs_user_then_assistant(tmp_path: Path) -> None:
    f = tmp_path / "session.jsonl"
    _write_jsonl(
        f,
        [
            {"type": "user", "message": {"content": "fix the bug"}},
            {
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "found it in auth.py"}]},
            },
            {"type": "user", "message": {"content": "now run tests"}},
            {
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "all 42 passing"}]},
            },
        ],
    )
    pairs = list(iter_exchanges(f))
    assert len(pairs) == 2
    assert pairs[0] == ("fix the bug", "found it in auth.py")
    assert pairs[1] == ("now run tests", "all 42 passing")


def test_iter_exchanges_concatenates_multiple_assistant_msgs(tmp_path: Path) -> None:
    f = tmp_path / "multi.jsonl"
    _write_jsonl(
        f,
        [
            {"type": "user", "message": {"content": "prompt"}},
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "part 1"}]}},
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "part 2"}]}},
        ],
    )
    pairs = list(iter_exchanges(f))
    assert len(pairs) == 1
    assert pairs[0][1] == "part 1\n\npart 2"


def test_iter_exchanges_skips_tool_blocks(tmp_path: Path) -> None:
    f = tmp_path / "tools.jsonl"
    _write_jsonl(
        f,
        [
            {"type": "user", "message": {"content": "do it"}},
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {"type": "text", "text": "running the tool"},
                        {"type": "tool_use", "name": "Bash", "input": {"command": "ls"}},
                        {"type": "text", "text": "result analyzed"},
                    ]
                },
            },
        ],
    )
    pairs = list(iter_exchanges(f))
    assert len(pairs) == 1
    assert "tool_use" not in pairs[0][1]
    assert "running the tool" in pairs[0][1]
    assert "result analyzed" in pairs[0][1]


def test_iter_exchanges_orphan_assistant_dropped(tmp_path: Path) -> None:
    f = tmp_path / "orphan.jsonl"
    _write_jsonl(
        f,
        [
            {"type": "assistant", "message": {"content": "no user before me"}},
        ],
    )
    assert list(iter_exchanges(f)) == []


def test_iter_exchanges_missing_file() -> None:
    assert list(iter_exchanges(Path("/no/such/transcript.jsonl"))) == []


def test_find_transcripts_recursive(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "b" / "c").mkdir(parents=True)
    (tmp_path / "a" / "s1.jsonl").write_text("{}")
    (tmp_path / "b" / "c" / "s2.jsonl").write_text("{}")
    (tmp_path / "noise.txt").write_text("x")
    files = find_transcripts(tmp_path)
    assert len(files) == 2
    assert all(p.suffix == ".jsonl" for p in files)


def test_find_transcripts_since_days(tmp_path: Path) -> None:
    new = tmp_path / "new.jsonl"
    old = tmp_path / "old.jsonl"
    new.write_text("{}")
    old.write_text("{}")
    # Backdate the "old" file by 100 days.
    cutoff = time.time() - (100 * 86400)
    os.utime(old, (cutoff, cutoff))
    files = find_transcripts(tmp_path, since_days=30)
    assert new in files
    assert old not in files


def test_find_transcripts_missing_root_returns_empty(tmp_path: Path) -> None:
    assert find_transcripts(tmp_path / "nope") == []
