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
        tmp_path,
        prompt="q1",
        via="subprocess",
        session_id="s",
        turn=1,
        hits=[
            {"id": "mem00001", "score": 0.9, "title": "t", "snippet": "x"},
            {"id": "mem00002", "score": 0.7, "title": "t", "snippet": "y"},
        ],
    )
    dashboard.append_recall_log(  # legacy row, no session_id
        tmp_path,
        prompt="q0",
        via="subprocess",
        hits=[{"id": "mem99999", "score": 0.6, "title": "t"}],
    )
    # Only mem00001 was grounded in the answer.
    dashboard.append_grounding_log(
        tmp_path,
        session_id="s",
        turn=1,
        recall_id="mem00001",
        used_score=0.81,
        method="lexical",
    )
    dashboard.append_grounding_log(
        tmp_path,
        session_id="s",
        turn=1,
        recall_id="mem00002",
        used_score=0.10,
        method="embed",
    )
    gr = dashboard.grounded_rate(tmp_path)
    # denominator = 2 correlatable surfacings (legacy mem99999 excluded)
    assert gr["surfaced"] == 2
    assert gr["grounded"] == 1
    assert gr["grounded_rate"] == 0.5


def test_recall_health_and_breakdown_expose_grounded(tmp_path: Path) -> None:
    dashboard.append_recall_log(
        tmp_path,
        prompt="q",
        via="subprocess",
        session_id="s",
        turn=1,
        client="claude-code",
        latency_ms=120,
        hits=[{"id": "mem00001", "score": 0.9, "title": "t", "snippet": "x"}],
    )
    dashboard.append_grounding_log(
        tmp_path,
        session_id="s",
        turn=1,
        recall_id="mem00001",
        used_score=0.9,
        method="lexical",
        client="claude-code",
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
        {
            "type": "assistant",
            "message": {"role": "assistant", "content": [{"type": "text", "text": assistant_text}]},
        },
    ]
    tp.write_text("\n".join(json.dumps(x) for x in lines) + "\n", encoding="utf-8")
    return tp


def _setup_turn(tmp_path: Path, snippet: str, assistant_text: str) -> dict:
    sid = "gsess"
    turn = session.next_turn(tmp_path, sid)  # 1
    session.stamp_recall_turn(tmp_path, sid, turn)
    dashboard.append_recall_log(
        tmp_path,
        prompt="prompt",
        via="subprocess",
        session_id=sid,
        turn=turn,
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


def test_grounding_uses_hook_log_when_recall_log_is_flooded(tmp_path: Path, monkeypatch) -> None:
    # The Stop hook writes a durable copy into recall_hook.log. If recall.log is
    # flooded, grounding must still find the turn there instead of dropping to
    # "no_recalled_hits".
    def _boom(*_args, **_kwargs) -> None:
        raise AssertionError("embedder should not run when lexical is high")

    monkeypatch.setattr("memo.embedder_client.embed", _boom)
    payload = _setup_turn(
        tmp_path,
        snippet="kubernetes deployment rollout strategy",
        assistant_text="Use the kubernetes deployment rollout strategy described earlier.",
    )
    for i in range(401):
        dashboard.append_recall_log(
            tmp_path,
            prompt=f"noise {i}",
            hits=[{"id": f"noise{i:05d}", "score": 0.1, "title": "noise"}],
            via="subprocess",
        )

    summary = grounding.score_turn(tmp_path, payload)
    assert summary and summary["scored"] == 1
    g = dashboard.read_grounding_log(tmp_path)
    assert len(g) == 1
    assert g[0]["recall_id"] == "memaaaa1"


# ---------------- project context in grounding rows (Fase 2 writer side) ----


def test_score_turn_stamps_derived_project_tag(tmp_path: Path, monkeypatch) -> None:
    # cwd in the Stop-hook payload points inside a git repo → the grounding
    # rows carry the derived `project:<slug>` tag.
    monkeypatch.delenv("MEMO_PROJECT_TAG", raising=False)

    def _boom(*a, **k):  # lexical is high → embedder must not run
        raise AssertionError("embedder should not be called when lexical is high")

    monkeypatch.setattr("memo.embedder_client.embed", _boom)
    repo = tmp_path / "myproj"
    (repo / ".git").mkdir(parents=True)
    payload = _setup_turn(
        tmp_path,
        snippet="kubernetes deployment rollout strategy",
        assistant_text="Use the kubernetes deployment rollout strategy described earlier.",
    )
    payload["cwd"] = str(repo)
    summary = grounding.score_turn(tmp_path, payload)
    assert summary and summary["scored"] == 1
    g = dashboard.read_grounding_log(tmp_path)
    assert len(g) == 1
    assert g[0]["project"] == "project:myproj"


def test_score_turn_omits_project_when_underivable(tmp_path: Path, monkeypatch) -> None:
    # cwd with no .git anywhere up the chain → no project field on the row.
    monkeypatch.delenv("MEMO_PROJECT_TAG", raising=False)
    monkeypatch.setattr("memo.embedder_client.embed", lambda texts, state_dir=None: [])
    plain = tmp_path / "plain"
    plain.mkdir()
    payload = _setup_turn(
        tmp_path,
        snippet="kubernetes deployment rollout strategy",
        assistant_text="Use the kubernetes deployment rollout strategy described earlier.",
    )
    payload["cwd"] = str(plain)
    summary = grounding.score_turn(tmp_path, payload)
    assert summary and summary["scored"] == 1
    g = dashboard.read_grounding_log(tmp_path)
    assert len(g) == 1
    assert "project" not in g[0]


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


# ── cited-id parsing (F1c: visible attribution → grounding signal) ──────────


def test_cited_ids_extracts_hex_prefixes() -> None:
    from memo.grounding import cited_ids

    answer = "Per your memory [a1b2c3d4] the sync tier is local; also [f6e5d4] applies."
    assert cited_ids(answer) == {"a1b2c3d4", "f6e5d4"}


def test_cited_ids_uppercase_normalized() -> None:
    from memo.grounding import cited_ids

    result = cited_ids("[A1B2C3D4] other text")
    assert result == {"a1b2c3d4"}


def test_cited_ids_five_chars_ignored() -> None:
    from memo.grounding import cited_ids

    assert cited_ids("[a1b2c]") == set()  # 5 hex chars → below minimum of 6


def test_cited_ids_ignores_non_hex_and_wrong_length() -> None:
    from memo.grounding import cited_ids

    assert cited_ids("[zzzz] [a1b2] [a1b2c3d4e5f6g7h8] plain text") == set()


def test_cited_ids_empty_answer() -> None:
    from memo.grounding import cited_ids

    assert cited_ids("") == set()


def test_match_cited_requires_session_membership() -> None:
    from memo.grounding import match_cited

    session_ids = ["a1b2c3d4e5f60789", "0123456789abcdef"]
    # a1b2c3d4 was recalled this session → matches. deadbeef was not → dropped.
    assert match_cited({"a1b2c3d4", "deadbeef"}, session_ids) == {"a1b2c3d4e5f60789"}


def test_match_cited_empty_inputs() -> None:
    from memo.grounding import match_cited

    assert match_cited(set(), ["a1b2c3d4e5f60789"]) == set()
    assert match_cited({"a1b2c3d4"}, []) == set()


# ── score_turn cited-id integration (8-char vs full-id mismatch fix) ─────────


def _write_transcript_citing(tmp_path: Path, prefix: str) -> Path:
    """Transcript whose last assistant message cites a memory by its 8-char prefix."""
    tp = tmp_path / "transcript.jsonl"
    lines = [
        {"type": "user", "message": {"role": "user", "content": "what do you know?"}},
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": f"See memory [{prefix}] for the confirmed decision."}
                ],
            },
        },
    ]
    tp.write_text("\n".join(json.dumps(x) for x in lines) + "\n", encoding="utf-8")
    return tp


def test_cited_upgrade_no_duplicate_when_recall_log_has_8char_ids(
    tmp_path: Path, monkeypatch
) -> None:
    """In-turn cited upgrade: recall_hook.log stores ids truncated to 8 chars,
    but the session recalled-ids map holds full ids. The cited memory must be
    upgraded to method='cited'/used_score=1.0 with NO duplicate standalone row.

    Pre-fix behaviour (bug): rid (8-char) not in cited_full (full ids) so the
    upgrade branch is dead code → the write loop emits a lexical/embed row, then
    the standalone loop emits a second cited row for the same memory (2 rows
    total, neither correctly upgraded in the write loop).
    Post-fix: exactly 1 row with method='cited' and used_score=1.0.
    """
    # Embed returns nothing → forces lexical-only (no cosine inflation)
    monkeypatch.setattr("memo.embedder_client.embed", lambda texts, state_dir=None: [])

    full_id = "a1b2c3d4e5f60789"
    short_prefix = full_id[:8]  # "a1b2c3d4" — what recall_hook.log stores

    sid = "cited-integ-1"
    turn = session.next_turn(tmp_path, sid)  # 1
    session.stamp_recall_turn(tmp_path, sid, turn)

    # append_recall_log truncates hit ids to 8 chars in recall_hook.log
    dashboard.append_recall_log(
        tmp_path,
        prompt="what do you know?",
        via="subprocess",
        session_id=sid,
        turn=turn,
        client="claude-code",
        hits=[{"id": full_id, "score": 0.85, "title": "strategy", "snippet": "zqz qqq www"}],
    )
    # Session map holds the FULL id (as mark_ids_recalled writes it)
    session.mark_ids_recalled(tmp_path, sid, {full_id: turn})

    tp = _write_transcript_citing(tmp_path, short_prefix)
    summary = grounding.score_turn(tmp_path, {"session_id": sid, "transcript_path": str(tp)})

    assert summary is not None and not summary.get("bailed"), f"score_turn bailed: {summary}"

    g = dashboard.read_grounding_log(tmp_path)
    # grounding.log also truncates recall_id to 8 chars; so the short prefix is
    # what we find in both the in-turn and the standalone rows
    matching = [r for r in g if r.get("recall_id") == short_prefix]
    assert len(matching) == 1, (
        f"Expected exactly 1 grounding row for {short_prefix!r}, got {len(matching)}: {matching}"
    )
    assert matching[0]["method"] == "cited", (
        f"Expected method='cited', got {matching[0]['method']!r}"
    )
    assert matching[0]["used_score"] == 1.0, (
        f"Expected used_score=1.0, got {matching[0]['used_score']}"
    )


def test_cited_no_current_hits_uses_earlier_turn_memory(tmp_path: Path, monkeypatch) -> None:
    """Earlier-turn cited path: current turn has NO recall hits at all, but the
    answer cites a memory from a prior turn → exactly one cited grounding row,
    no bail on 'no_recalled_hits'.
    """
    monkeypatch.setattr("memo.embedder_client.embed", lambda texts, state_dir=None: [])

    earlier_full_id = "eee1fff2ggg3hh44"
    earlier_prefix = earlier_full_id[:8]  # "eee1fff2"

    sid = "cited-no-current-hits"
    # Mark the earlier memory as recalled in a prior turn (turn 0, before this session)
    session.mark_ids_recalled(tmp_path, sid, {earlier_full_id: 0})

    # Stamp a current turn — no recall_hook.log entries for this turn
    turn = session.next_turn(tmp_path, sid)  # 1
    session.stamp_recall_turn(tmp_path, sid, turn)

    # Answer cites the earlier memory's 8-char prefix
    tp = _write_transcript_citing(tmp_path, earlier_prefix)
    summary = grounding.score_turn(tmp_path, {"session_id": sid, "transcript_path": str(tp)})

    assert summary is not None and not summary.get("bailed"), f"score_turn bailed: {summary}"

    g = dashboard.read_grounding_log(tmp_path)
    cited_rows = [r for r in g if r.get("method") == "cited"]
    assert len(cited_rows) == 1, (
        f"Expected 1 cited row for early-turn memory, got {len(cited_rows)}: {cited_rows}"
    )
    assert cited_rows[0]["recall_id"] == earlier_prefix, (
        f"Expected recall_id={earlier_prefix!r}, got {cited_rows[0]['recall_id']!r}"
    )
    assert cited_rows[0]["used_score"] == 1.0


def test_cited_standalone_for_earlier_turn_memory(tmp_path: Path, monkeypatch) -> None:
    """Earlier-turn cited standalone: a memory present in the session recalled-ids
    map (recalled in a previous turn) but NOT among the current turn's recall hits
    should produce a standalone cited row when the answer cites its prefix.
    """
    monkeypatch.setattr("memo.embedder_client.embed", lambda texts, state_dir=None: [])

    earlier_full_id = "aaa1bbb2cccc3333"
    earlier_prefix = earlier_full_id[:8]  # "aaa1bbb2"
    current_full_id = "memcurrent0000bb"  # not cited in the answer

    sid = "cited-integ-2"
    # Earlier memory: add to session map before the current turn
    session.mark_ids_recalled(tmp_path, sid, {earlier_full_id: 0})

    # Current turn: different memory in the recall hits
    turn = session.next_turn(tmp_path, sid)  # 1
    session.stamp_recall_turn(tmp_path, sid, turn)
    dashboard.append_recall_log(
        tmp_path,
        prompt="what do you know?",
        via="subprocess",
        session_id=sid,
        turn=turn,
        client="claude-code",
        hits=[
            {
                "id": current_full_id,
                "score": 0.7,
                "title": "current",
                "snippet": "current memory content",
            }
        ],
    )
    session.mark_ids_recalled(tmp_path, sid, {current_full_id: turn})

    # Answer cites the EARLIER memory, not the current-turn one
    tp = _write_transcript_citing(tmp_path, earlier_prefix)
    summary = grounding.score_turn(tmp_path, {"session_id": sid, "transcript_path": str(tp)})

    assert summary is not None and not summary.get("bailed"), f"score_turn bailed: {summary}"

    g = dashboard.read_grounding_log(tmp_path)
    # Should have a standalone cited row for the earlier-turn memory
    cited_rows = [r for r in g if r.get("method") == "cited"]
    assert len(cited_rows) == 1, (
        f"Expected 1 standalone cited row, got {len(cited_rows)}: {cited_rows}"
    )
    assert cited_rows[0]["recall_id"] == earlier_prefix, (
        f"Expected recall_id={earlier_prefix!r}, got {cited_rows[0]['recall_id']!r}"
    )
    assert cited_rows[0]["used_score"] == 1.0
