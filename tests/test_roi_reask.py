"""P1 re-ask avoidance, P2 downstream action, P3 memo roi — no real MLX."""
from __future__ import annotations

import json
from pathlib import Path

from memo import dashboard, grounding
from memo.cli_roi import compute_roi


def _recall(tmp: Path, sid: str, turn: int, prompt: str, mem_id: str, snippet: str = "x") -> None:
    dashboard.append_recall_log(
        tmp, prompt=prompt, via="subprocess", session_id=sid, turn=turn, client="claude-code",
        hits=[{"id": mem_id, "score": 0.8, "title": "t", "snippet": snippet}],
    )


def _grounded(tmp: Path, sid: str, turn: int, mem_id: str, score: float = 0.9, **kw) -> None:
    dashboard.append_grounding_log(
        tmp, session_id=sid, turn=turn, recall_id=mem_id, used_score=score,
        method="lexical", client="claude-code", **kw,
    )


# ---------------- P1 re-ask avoidance ----------------

def test_reask_avoided_when_no_followup(tmp_path: Path) -> None:
    _recall(tmp_path, "s", 1, "how to configure the deploy pipeline yaml", "mem00001")
    _grounded(tmp_path, "s", 1, "mem00001")
    # later turn asks something unrelated → not a re-ask
    _recall(tmp_path, "s", 2, "what is the capital of france today", "mem00002")
    stats = dashboard.reask_stats(tmp_path)
    assert stats["considered"] == 1
    assert stats["reask"] == 0
    assert stats["reask_avoided"] == 1


def test_reask_detected_on_near_duplicate_followup(tmp_path: Path) -> None:
    _recall(tmp_path, "s", 1, "how to configure the deploy pipeline yaml settings", "mem00001")
    _grounded(tmp_path, "s", 1, "mem00001")
    # near-duplicate follow-up within window → re-ask (memo didn't save it)
    _recall(tmp_path, "s", 2, "configure the deploy pipeline yaml settings how", "mem00003")
    stats = dashboard.reask_stats(tmp_path)
    assert stats["considered"] == 1
    assert stats["reask"] == 1
    assert stats["reask_avoided"] == 0


def test_reask_only_counts_grounded(tmp_path: Path) -> None:
    # recall not grounded → not considered for re-ask avoidance
    _recall(tmp_path, "s", 1, "some prompt about things here", "mem00001")
    dashboard.append_grounding_log(
        tmp_path, session_id="s", turn=1, recall_id="mem00001",
        used_score=0.1, method="embed",  # below threshold → not grounded
    )
    stats = dashboard.reask_stats(tmp_path)
    assert stats["considered"] == 0
    assert stats["reask_rate"] is None


# ---------------- P2 downstream action ----------------

def test_collect_tool_targets_and_match(tmp_path: Path) -> None:
    tp = tmp_path / "t.jsonl"
    tp.write_text("\n".join(json.dumps(x) for x in [
        {"type": "assistant", "message": {"role": "assistant", "content": [
            {"type": "tool_use", "name": "Read", "input": {"file_path": "/repo/src/deploy.py"}},
            {"type": "tool_use", "name": "Bash", "input": {"command": "make deploy production"}},
        ]}},
    ]) + "\n", encoding="utf-8")
    targets = grounding.collect_recent_tool_targets(tp)
    assert {"action": "opened_file", "target": "/repo/src/deploy.py"} in targets
    # snippet that mentions deploy.py → opened_file action
    a = grounding._action_for_snippet("see src/deploy.py for the rollout", targets)
    assert a and a["downstream_action"] == "opened_file"
    # snippet mentioning the command tokens → ran_command
    a2 = grounding._action_for_snippet("run make deploy production to ship", targets)
    assert a2 and a2["downstream_action"] == "ran_command"
    # unrelated snippet → no action
    assert grounding._action_for_snippet("totally unrelated prose", targets) is None


# ---------------- P3 memo roi ----------------

def test_compute_roi_time_saved_and_table(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MEMO_ROI_SECS_PER_GROUNDED", "30")
    monkeypatch.setenv("MEMO_ROI_SECS_PER_REASK", "120")
    _recall(tmp_path, "s", 1, "configure the deploy pipeline yaml settings now", "mem00001")
    _grounded(tmp_path, "s", 1, "mem00001", downstream_action="opened_file", action_evidence="/x.py")
    # no near-dup follow-up → 1 reask_avoided ; 1 grounded
    data = compute_roi(tmp_path)
    assert data["grounded"] == 1
    assert data["reask"]["reask_avoided"] == 1
    # time saved = 1*30 + 1*120 = 150s
    assert data["time_saved_seconds"] == 150
    assert data["grounded_rate"] == 1.0
    cc = next(c for c in data["by_consumer"] if c["consumer"] == "claude-code")
    assert cc["grounded_rate"] == 1.0
    assert data["actions_by_client"]["claude-code"]["actions"] == 1


def test_compute_roi_empty(tmp_path: Path) -> None:
    data = compute_roi(tmp_path)
    assert data["by_consumer"] == []
    assert data["time_saved_seconds"] == 0
