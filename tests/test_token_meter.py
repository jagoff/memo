import json as _json

from click.testing import CliRunner

from memo import token_meter as tm


def _assistant(mid: str, out: int, *, tool: bool = False) -> dict:
    content = [{"type": "text", "text": "x"}]
    if tool:
        content.append({"type": "tool_use", "name": "Read", "input": {}})
    return {"type": "assistant", "message": {"role": "assistant", "id": mid,
            "usage": {"output_tokens": out}, "content": content}}


def _human(text: str = "hola que onda") -> dict:
    return {"type": "user", "message": {"role": "user", "content": text}}


def _tool_result() -> dict:
    return {"type": "user", "message": {"role": "user",
            "content": [{"type": "tool_result", "content": "ok"}]}}


def test_iter_prompt_turns_splits_on_human_prompts_and_dedups_message_id():
    rows = [
        _human(),
        _assistant("m1", 100, tool=True),   # tool step
        _assistant("m1", 100, tool=True),   # SAME id repeated → count once
        _tool_result(),
        _assistant("m2", 40),               # final answer of turn 0
        _human(),
        _assistant("m3", 70),               # single answer of turn 1
    ]
    turns = tm.iter_prompt_turns(rows)
    assert [t.index for t in turns] == [0, 1]
    # turn 0: answer = last assistant (m2=40); tool_tok = m1 (100, counted once)
    assert turns[0].answer_tok == 40
    assert turns[0].tool_tok == 100
    assert turns[0].n_tool_steps == 1
    # turn 1: single assistant → it is the answer, no tool spend
    assert turns[1].answer_tok == 70
    assert turns[1].tool_tok == 0


def test_iter_prompt_turns_skips_sidechain_rows():
    rows = [
        _human(),
        {"type": "assistant", "isSidechain": True,
         "message": {"role": "assistant", "id": "s1", "usage": {"output_tokens": 999},
                     "content": [{"type": "text", "text": "sub"}]}},
        _assistant("m1", 50),
    ]
    turns = tm.iter_prompt_turns(rows)
    assert len(turns) == 1
    assert turns[0].answer_tok == 50
    assert turns[0].tool_tok == 0  # sidechain 999 ignored


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(_json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def _transcript(tmp_path, sid, turns):
    """turns = list of (tool_out, answer_out); build a minimal JSONL transcript."""
    rows = []
    for i, (tool_out, ans_out) in enumerate(turns):
        rows.append({"type": "user", "sessionId": sid,
                     "message": {"role": "user", "content": f"prompt {i}"}})
        if tool_out:
            rows.append({"type": "assistant", "sessionId": sid,
                         "message": {"role": "assistant", "id": f"{sid}-t{i}",
                                     "usage": {"output_tokens": tool_out},
                                     "content": [{"type": "tool_use", "name": "Read", "input": {}}]}})
        rows.append({"type": "assistant", "sessionId": sid,
                     "message": {"role": "assistant", "id": f"{sid}-a{i}",
                                 "usage": {"output_tokens": ans_out},
                                 "content": [{"type": "text", "text": "answer"}]}})
    p = tmp_path / f"{sid}.jsonl"
    p.write_text("\n".join(_json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return p


def test_roll_joins_injection_and_grounding(tmp_path):
    from memo.dashboard_logs import context_cost_log_path, grounding_log_path

    state = tmp_path / "state"
    state.mkdir()
    # session S1: 2 turns, tool loop 200 in turn 0
    tp = _transcript(tmp_path, "S1", [(200, 40), (0, 30)])
    _write_jsonl(context_cost_log_path(state),
                 [{"kind": "recall", "session_id": "S1", "turn": 1, "chars": 400}])
    _write_jsonl(grounding_log_path(state),
                 [{"session_id": "S1", "turn": 1, "recall_id": "abc", "used_score": 0.9}])

    tm.roll(state, "S1", tp)
    s = tm.summarize(state)
    assert s["sessions"] == 1
    assert s["tool_tok"] == 200
    assert s["answer_tok"] == 70
    assert s["injected_tokens"] == 100        # 400 chars / 4
    assert s["grounded"] == 1


def test_meter_flag_declared_and_default_on(monkeypatch):
    monkeypatch.delenv("MEMO_TOKEN_METER_ENABLED", raising=False)
    from memo.flags import flag_bool, REGISTRY

    assert "MEMO_TOKEN_METER_ENABLED" in REGISTRY
    assert flag_bool("MEMO_TOKEN_METER_ENABLED") is True  # opt-out default-on


def test_summarize_proxy_grounded_vs_ungrounded(tmp_path):
    from memo.dashboard_logs import context_cost_log_path, grounding_log_path

    state = tmp_path / "state"
    state.mkdir()
    # grounded session: low tool spend; ungrounded-but-injected: high tool spend
    tg = _transcript(tmp_path, "G", [(50, 20)])          # 1 turn, tool 50
    tu = _transcript(tmp_path, "U", [(500, 20)])         # 1 turn, tool 500
    _write_jsonl(context_cost_log_path(state), [
        {"kind": "recall", "session_id": "G", "turn": 1, "chars": 100},
        {"kind": "recall", "session_id": "U", "turn": 1, "chars": 100},
    ])
    _write_jsonl(grounding_log_path(state),
                 [{"session_id": "G", "turn": 1, "recall_id": "x", "used_score": 0.9}])
    tm.roll(state, "G", tg)
    tm.roll(state, "U", tu)
    s = tm.summarize(state)
    p = s["proxy"]
    assert p["grounded_tool_tok_per_turn"] == 50.0
    assert p["ungrounded_tool_tok_per_turn"] == 500.0
    assert p["delta"] == 450.0  # ungrounded − grounded (positive ⇒ memo correlates with less tool spend)


# ---------------------------------------------------------------------------
# Task-4: additive measured panel in `memo tokens`
# ---------------------------------------------------------------------------


def _cli_env(tmp_path):
    return {"MEMO_NONINTERACTIVE": "1",
            "MEMO_DATA_DIR": str(tmp_path / "data"),
            "MEMO_STATE_DIR": str(tmp_path / "state")}


def test_tokens_cmd_shows_measured_block(tmp_path):
    from memo.cli import cli

    state = tmp_path / "state"
    state.mkdir(parents=True)
    tp = _transcript(tmp_path, "S1", [(120, 40)])
    tm.roll(state, "S1", tp)  # populate the measured ledger
    runner = CliRunner()
    res = runner.invoke(cli, ["tokens"], env=_cli_env(tmp_path))
    assert res.exit_code == 0
    assert "medido" in res.output.lower() or "measured" in res.output.lower()


def test_tokens_json_preserves_ledger_schema(tmp_path):
    from memo.cli import cli

    runner = CliRunner()
    res = runner.invoke(cli, ["tokens", "--json"], env=_cli_env(tmp_path))
    assert res.exit_code == 0
    payload = _json.loads(res.output)
    # frozen token_ledger keys must all remain present
    for key in ("today", "month", "historic", "daily", "monthly", "growth", "tpg", "ledger_path"):
        assert key in payload
    assert "measured" in payload   # additive key must actually be written
