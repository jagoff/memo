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


def test_mine_transcripts_stamps_provenance(tmp_path: Path, monkeypatch) -> None:
    """mine_transcripts stamps session_id/transcript_path/turn_hash in extra."""
    from memo.transcript_miner import mine_transcripts

    # Pin env so a shell with MEMO_VAULT_PATH/MEMO_MEMORIES_IN_VAULT exported
    # can never route Memory(cfg) into the real vault.
    monkeypatch.setenv("MEMO_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MEMO_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("MEMO_VAULT_PATH", str(tmp_path / "vault"))
    monkeypatch.setenv("MEMO_EMBEDDER_DIMS", "4")
    monkeypatch.setattr(
        "memo.embedder.MLXEmbedder.embed",
        lambda self, inputs: [[1.0, 0.0, 0.0, 0.0] for _ in inputs],
    )

    monkeypatch.setattr(
        "memo.transcript_miner.extract_insights",
        lambda *a, **k: [
            {
                "title": "Pin transformers below 5.13",
                "type": "decision",
                "body": (
                    "We decided to pin transformers<5.13 because 5.13 breaks the "
                    "mlx-lm import on Apple Silicon and every fresh macOS install."
                ),
                "tags": ["memo", "deps"],
            }
        ],
    )

    transcript = tmp_path / "my-session-id.jsonl"
    assistant = (
        "We decided to pin transformers below 5.13 because the newer release "
        "breaks the mlx-lm import chain on Apple Silicon. " + "detail " * 40
    )
    _write_jsonl(
        transcript,
        [
            {"type": "user", "message": {"content": "pin it?"}},
            {"type": "assistant", "message": {"content": [{"type": "text", "text": assistant}]}},
        ],
    )

    result = mine_transcripts(root=tmp_path)
    assert result["status"] == "ok"
    assert len(result["saved"]) == 1

    from memo.config import Config
    from memo.memory import Memory

    cfg = Config.from_env()
    mem = Memory(cfg)
    try:
        rec = mem.get(result["saved"][0])
        assert rec.extra["transcript_path"].endswith(transcript.name)
        assert rec.extra["session_id"] == transcript.stem
        assert isinstance(rec.extra["turn_hash"], str) and rec.extra["turn_hash"]
    finally:
        mem.close()


def test_mine_exchange_stream_extracts_dedups_saves(mock_memory, monkeypatch):
    import memo.transcript_miner as tm

    monkeypatch.setattr(tm, "_passes_prefilter", lambda text: True)
    monkeypatch.setattr(
        tm,
        "extract_insights",
        lambda chat, model, u, a: [
            {"title": "Use uv for envs", "type": "decision", "body": a, "tags": ["tooling"]}
        ],
    )
    monkeypatch.setattr(tm, "is_near_duplicate", lambda mem, cand: False)

    exchanges = iter(
        [
            ("how do we manage envs?", "decision: use uv everywhere"),
            ("repeat", "decision: use uv everywhere"),  # identical assistant turn → hash dedup
        ]
    )
    hashes: set[str] = set()
    result = tm.mine_exchange_stream(
        mock_memory, object(), mock_memory.cfg, exchanges, turn_hashes=hashes
    )

    assert result["candidates"] == 1
    assert len(result["saved"]) == 1
    assert result["skipped_dup"] == 0
    saved = mock_memory.get(result["saved"][0])
    assert saved.title == "Use uv for envs"


def test_mine_exchange_stream_dry_run_saves_nothing(mock_memory, monkeypatch):
    import memo.transcript_miner as tm

    monkeypatch.setattr(tm, "_passes_prefilter", lambda text: True)
    monkeypatch.setattr(
        tm,
        "extract_insights",
        lambda chat, model, u, a: [{"title": "T", "type": "note", "body": "B", "tags": []}],
    )
    monkeypatch.setattr(tm, "is_near_duplicate", lambda mem, cand: False)

    result = tm.mine_exchange_stream(
        mock_memory, object(), mock_memory.cfg,
        iter([("u", "a")]), turn_hashes=set(), dry_run=True,
    )

    assert result["saved"] == ["<dry-run>"]
    assert mock_memory.list(limit=10) == []


def test_state_helpers_accept_custom_filename(tmp_path):
    from memo.transcript_miner import _load_state, _save_state

    _save_state(tmp_path, {"/a/b.jsonl": {"lines_processed": 7}}, name="import-history.json")
    assert (tmp_path / "import-history.json").is_file()
    assert not (tmp_path / "mine-history.json").exists()
    assert _load_state(tmp_path, name="import-history.json")["/a/b.jsonl"]["lines_processed"] == 7
