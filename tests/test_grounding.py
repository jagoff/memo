"""P0 utility-measurement: recall→answer correlation + grounding detector.

No real MLX — the embedder is monkeypatched to canned vectors.
"""
from __future__ import annotations

import json
from pathlib import Path

from memo import dashboard, grounding, session

# ---------------- data layer: correlation + grounded_rate ----------------

def test_append_recall_log_carries_correlation(tmp_path: Path) -> None:
    dashboard.append_recall_log(
        tmp_path,
        prompt="how do we deploy",
        hits=[{"id": "abc12345", "score": 0.8, "title": "deploy", "snippet": "run make deploy"}],
        via="subprocess",
        session_id="sess-1",
        turn=3,
        client="claude-code",
    )
    rows = dashboard.read_recall_log(tmp_path, limit=5)
    assert len(rows) == 1
    r = rows[0]
    assert r["session_id"] == "sess-1"
    assert r["turn"] == 3
    assert r["client"] == "claude-code"
    assert r["hits"][0]["snippet"] == "run make deploy"
    assert r["hits"][0]["id"] == "abc12345"


def test_grounded_rate_joins_and_excludes_old_rows(tmp_path: Path) -> None:
    # Two correlatable surfacings + one legacy row (no session_id → excluded).
    dashboard.append_recall_log(
        tmp_path, prompt="q1", via="subprocess", session_id="s", turn=1,
        hits=[{"id": "mem00001", "score": 0.9, "title": "t", "snippet": "x"},
              {"id": "mem00002", "score": 0.7, "title": "t", "snippet": "y"}],
    )
    dashboard.append_recall_log(  # legacy row, no session_id
        tmp_path, prompt="q0", via="subprocess",
        hits=[{"id": "mem99999", "score": 0.6, "title": "t"}],
    )
    # Only mem00001 was grounded in the answer.
    dashboard.append_grounding_log(
        tmp_path, session_id="s", turn=1, recall_id="mem00001",
        used_score=0.81, method="lexical",
    )
    dashboard.append_grounding_log(
        tmp_path, session_id="s", turn=1, recall_id="mem00002",
        used_score=0.10, method="embed",
    )
    gr = dashboard.grounded_rate(tmp_path)
    # denominator = 2 correlatable surfacings (legacy mem99999 excluded)
    assert gr["surfaced"] == 2
    assert gr["grounded"] == 1
    assert gr["grounded_rate"] == 0.5


def test_recall_health_and_breakdown_expose_grounded(tmp_path: Path) -> None:
    dashboard.append_recall_log(
        tmp_path, prompt="q", via="subprocess", session_id="s", turn=1,
        client="claude-code", latency_ms=120,
        hits=[{"id": "mem00001", "score": 0.9, "title": "t", "snippet": "x"}],
    )
    dashboard.append_grounding_log(
        tmp_path, session_id="s", turn=1, recall_id="mem00001",
        used_score=0.9, method="lexical", client="claude-code",
    )
    health = dashboard.recall_health(tmp_path)
    assert health["grounded_rate"] == 1.0
    assert health["grounded"] == 1
    bd = dashboard.consult_breakdown(tmp_path)
    cc = next(c for c in bd["consumers"] if c["consumer"] == "claude-code")
    assert cc["grounded_rate"] == 1.0


# ---------------- correlation stamp ----------------

def test_session_stamp_and_next_turn(tmp_path: Path) -> None:
    sid = "sess-xyz"
    t1 = session.next_turn(tmp_path, sid)
    assert t1 == 1
    session.stamp_recall_turn(tmp_path, sid, t1)
    snap = session.get_session(tmp_path, sid)
    assert snap is not None
    assert snap["last_recall_turn"] == 1


# ---------------- grounding.score_turn (stubbed embedder) ----------------

def _write_transcript(tmp_path: Path, assistant_text: str) -> Path:
    tp = tmp_path / "transcript.jsonl"
    lines = [
        {"type": "user", "message": {"role": "user", "content": "the question"}},
        {"type": "assistant", "message": {"role": "assistant",
                                          "content": [{"type": "text", "text": assistant_text}]}},
    ]
    tp.write_text("\n".join(json.dumps(x) for x in lines) + "\n", encoding="utf-8")
    return tp


def _setup_turn(tmp_path: Path, snippet: str, assistant_text: str) -> dict:
    sid = "gsess"
    turn = session.next_turn(tmp_path, sid)  # 1
    session.stamp_recall_turn(tmp_path, sid, turn)
    dashboard.append_recall_log(
        tmp_path, prompt="prompt", via="subprocess", session_id=sid, turn=turn,
        client="claude-code",
        hits=[{"id": "memaaaa1", "score": 0.8, "title": "t", "snippet": snippet}],
    )
    tp = _write_transcript(tmp_path, assistant_text)
    return {"session_id": sid, "transcript_path": str(tp)}


def test_grounding_lexical_marks_used_without_embedding(tmp_path: Path, monkeypatch) -> None:
    # Answer quotes the snippet's salient tokens → lexical containment high.
    def _boom(*a, **k):  # embedding must NOT be called on the high-lexical path
        raise AssertionError("embedder should not be called when lexical is high")
    monkeypatch.setattr("memo.embedder_client.embed", _boom)
    payload = _setup_turn(
        tmp_path,
        snippet="kubernetes deployment rollout strategy",
        assistant_text="Use the kubernetes deployment rollout strategy described earlier.",
    )
    summary = grounding.score_turn(tmp_path, payload)
    assert summary and summary["scored"] == 1
    g = dashboard.read_grounding_log(tmp_path)
    assert len(g) == 1
    assert g[0]["recall_id"] == "memaaaa1"
    assert g[0]["used_score"] >= 0.6
    assert g[0]["method"] == "lexical"


def test_grounding_embed_catches_paraphrase(tmp_path: Path, monkeypatch) -> None:
    # No lexical overlap → embedding pass; canned high-cosine vectors → grounded.
    monkeypatch.setattr(
        "memo.embedder_client.embed",
        # batch = [answer, question, snippet]; answer & snippet aligned, question off-axis
        lambda texts, state_dir=None: [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]],
    )
    payload = _setup_turn(
        tmp_path,
        snippet="zzz qqq www",  # disjoint tokens
        assistant_text="completely different words here entirely",
    )
    summary = grounding.score_turn(tmp_path, payload)
    assert summary and summary["scored"] == 1
    g = dashboard.read_grounding_log(tmp_path)
    assert g[0]["used_score"] >= 0.9  # cosine of aligned vectors == 1.0
    assert g[0]["method"] in ("embed", "both")


def test_grounding_unrelated_not_used(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "memo.embedder_client.embed",
        # batch = [answer, question, snippet]; snippet orthogonal to answer → cosine 0
        lambda texts, state_dir=None: [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
    )
    payload = _setup_turn(
        tmp_path,
        snippet="zzz qqq www",
        assistant_text="completely different words here entirely",
    )
    grounding.score_turn(tmp_path, payload)
    g = dashboard.read_grounding_log(tmp_path)
    assert g[0]["used_score"] < 0.5  # not grounded


def test_grounding_no_stamp_is_noop(tmp_path: Path) -> None:
    # No last_recall_turn stamped → nothing to correlate.
    tp = _write_transcript(tmp_path, "some answer")
    out = grounding.score_turn(tmp_path, {"session_id": "nope", "transcript_path": str(tp)})
    assert out == {"session_id": "nope", "scored": 0, "bailed": "missing_last_recall_turn"}
    assert dashboard.read_grounding_log(tmp_path) == []


def test_grounding_missing_payload_fields_noop(tmp_path: Path) -> None:
    assert grounding.score_turn(tmp_path, {}) == {"scored": 0, "bailed": "missing_session_id"}
    assert grounding.score_turn(tmp_path, {"session_id": "x"}) == {
        "session_id": "x",
        "scored": 0,
        "bailed": "missing_transcript_path",
    }


def test_grounding_noop_writes_diagnostic(tmp_path: Path) -> None:
    out = grounding.score_turn(tmp_path, {"session_id": "x"})
    assert out and out["bailed"] == "missing_transcript_path"
    rows = dashboard.read_grounding_diag_log(tmp_path)
    assert rows[0]["reason"] == "missing_transcript_path"
    assert rows[0]["session_id"] == "x"
