"""Daemon-path session marking — the root cause of 0 method=cited rows.

_recall_logic (the daemon/production path) must populate the session's
recalled_ids exactly like the subprocess path does: cited-grounding validates
``[id8]`` cites against that map, and session dedup filters re-injections.
Before the fix only cli_recall_hook marked, so warm-daemon machines never
produced a single cited grounding row.
"""

from __future__ import annotations

import json

import pytest

from memo import session as session_mod
from memo.config import Config
from memo.grounding import score_turn
from memo.memory import Memory
from memo.recall_logic import _recall_logic


@pytest.fixture
def mem(tmp_cfg: Config, monkeypatch) -> Memory:
    cfg = Config(
        data_dir=tmp_cfg.data_dir,
        vault_path=tmp_cfg.vault_path,
        state_dir=tmp_cfg.state_dir,
        embedder_dims=64,
    )

    def _stub_embed(self, inputs):
        out = []
        for s in inputs:
            h = sum(ord(c) for c in s) % 64
            v = [0.0] * 64
            v[h] = 1.0
            out.append(v)
        return out

    monkeypatch.setattr("memo.embedder.MLXEmbedder.embed", _stub_embed)
    monkeypatch.setattr(
        "memo.embedder.MLXEmbedder.embed_query", lambda self, q: _stub_embed(self, [q])[0]
    )
    m = Memory(cfg)
    yield m
    m.close()


def _recall(mem: Memory, prompt: str, session_id: str, turn: int) -> str:
    # Mirror the full daemon flow: the client stamps the turn correlation
    # (cli_recall_hook:146) and recall_socket invokes the returned log fn.
    session_mod.stamp_recall_turn(mem.cfg.state_dir, session_id, turn)
    out, log_fn = _recall_logic(
        prompt,
        None,
        mem,
        mem.cfg,
        session_id=session_id,
        turn=turn,
    )
    if log_fn is not None:
        log_fn()
    return out


def test_daemon_path_marks_recalled_ids(mem: Memory, monkeypatch):
    monkeypatch.setenv("MEMO_RECALL_MIN_SIM", "0.0")
    monkeypatch.setenv("MEMO_RECALL_SKIP_BELOW", "0")
    monkeypatch.setenv("MEMO_RECALL_MIN_BODY_CHARS", "0")
    rec = mem.save(
        content="el fix del flock fue usar lock sidecar en el state file",
        title="Fix flock",
        type_="fact",
    )
    out = _recall(mem, "fix del flock en state file", "sess-daemon-1", 1)
    assert out != "{}"
    recalled = session_mod.get_recalled_ids(mem.cfg.state_dir, "sess-daemon-1")
    assert rec.id in recalled, "daemon path must persist recalled ids for cited-grounding"
    assert recalled[rec.id] == 1


def test_daemon_path_dedups_across_turns(mem: Memory, monkeypatch):
    monkeypatch.setenv("MEMO_RECALL_MIN_SIM", "0.0")
    monkeypatch.setenv("MEMO_RECALL_SKIP_BELOW", "0")
    monkeypatch.setenv("MEMO_RECALL_MIN_BODY_CHARS", "0")
    mem.save(
        content="el fix del flock fue usar lock sidecar en el state file",
        title="Fix flock",
        type_="fact",
    )
    first = _recall(mem, "fix del flock en state file", "sess-daemon-2", 1)
    assert first != "{}"
    second = _recall(mem, "fix del flock en state file", "sess-daemon-2", 2)
    assert second == "{}", "same hits must dedup on the second turn (subprocess parity)"


def test_cited_grounding_fires_after_daemon_recall(mem: Memory, monkeypatch, tmp_path):
    """End-to-end: daemon recall marks the id → a citing answer produces a
    method=cited grounding row. This chain was dead before the fix."""
    monkeypatch.setenv("MEMO_RECALL_MIN_SIM", "0.0")
    monkeypatch.setenv("MEMO_RECALL_SKIP_BELOW", "0")
    monkeypatch.setenv("MEMO_RECALL_MIN_BODY_CHARS", "0")
    rec = mem.save(
        content="el fix del flock fue usar lock sidecar en el state file",
        title="Fix flock",
        type_="fact",
    )
    sid = "sess-daemon-3"
    assert _recall(mem, "fix del flock en state file", sid, 1) != "{}"

    transcript = tmp_path / "t.jsonl"
    answer = f"El fix fue el lock sidecar, per your memory [{rec.id[:8]}]."
    transcript.write_text(
        json.dumps(
            {
                "type": "assistant",
                "message": {"role": "assistant", "content": [{"type": "text", "text": answer}]},
            }
        )
        + "\n"
    )
    res = score_turn(
        mem.cfg.state_dir,
        {"session_id": sid, "transcript_path": str(transcript), "cwd": str(tmp_path)},
    )
    assert res is not None and res.get("scored", 0) >= 1
    rows = [json.loads(ln) for ln in (mem.cfg.state_dir / "grounding.log").read_text().splitlines()]
    cited = [r for r in rows if r.get("method") == "cited"]
    assert cited, "a validated [id8] cite must produce a method=cited row"
    assert cited[0]["recall_id"] == rec.id[:8]
