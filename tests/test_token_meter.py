import json as _json

from click.testing import CliRunner

from memo import token_meter as tm


def _assistant(
    mid: str,
    out: int,
    *,
    tool: bool = False,
    model: str | None = None,
    input_tok: int | None = None,
    cache_read: int | None = None,
    cache_creation: int | None = None,
) -> dict:
    content = [{"type": "text", "text": "x"}]
    if tool:
        content.append({"type": "tool_use", "name": "Read", "input": {}})
    usage = {"output_tokens": out}
    if cache_read is not None:
        usage["cache_read_input_tokens"] = cache_read
    if cache_creation is not None:
        usage["cache_creation_input_tokens"] = cache_creation
    if input_tok is not None:
        usage["input_tokens"] = input_tok
    message: dict = {
        "role": "assistant",
        "id": mid,
        "usage": usage,
        "content": content,
    }
    if model:
        message["model"] = model
    return {
        "type": "assistant",
        "message": message,
    }


def _human(text: str = "hola que onda") -> dict:
    return {"type": "user", "message": {"role": "user", "content": text}}


def _tool_result() -> dict:
    return {
        "type": "user",
        "message": {"role": "user", "content": [{"type": "tool_result", "content": "ok"}]},
    }


def test_iter_prompt_turns_splits_on_human_prompts_and_dedups_message_id():
    rows = [
        _human(),
        _assistant("m1", 100, tool=True),  # tool step
        _assistant("m1", 100, tool=True),  # SAME id repeated → count once
        _tool_result(),
        _assistant("m2", 40),  # final answer of turn 0
        _human(),
        _assistant("m3", 70),  # single answer of turn 1
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
        {
            "type": "assistant",
            "isSidechain": True,
            "message": {
                "role": "assistant",
                "id": "s1",
                "usage": {"output_tokens": 999},
                "content": [{"type": "text", "text": "sub"}],
            },
        },
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
        rows.append(
            {
                "type": "user",
                "sessionId": sid,
                "message": {"role": "user", "content": f"prompt {i}"},
            }
        )
        if tool_out:
            rows.append(
                {
                    "type": "assistant",
                    "sessionId": sid,
                    "message": {
                        "role": "assistant",
                        "id": f"{sid}-t{i}",
                        "usage": {"output_tokens": tool_out},
                        "content": [{"type": "tool_use", "name": "Read", "input": {}}],
                    },
                }
            )
        rows.append(
            {
                "type": "assistant",
                "sessionId": sid,
                "message": {
                    "role": "assistant",
                    "id": f"{sid}-a{i}",
                    "usage": {"output_tokens": ans_out},
                    "content": [{"type": "text", "text": "answer"}],
                },
            }
        )
    p = tmp_path / f"{sid}.jsonl"
    p.write_text("\n".join(_json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return p


def test_roll_joins_injection_and_grounding(tmp_path):
    from memo.dashboard_logs import context_cost_log_path, grounding_log_path

    state = tmp_path / "state"
    state.mkdir()
    # session S1: 2 turns, tool loop 200 in turn 0
    tp = _transcript(tmp_path, "S1", [(200, 40), (0, 30)])
    _write_jsonl(
        context_cost_log_path(state),
        [{"kind": "recall", "session_id": "S1", "turn": 1, "chars": 400}],
    )
    _write_jsonl(
        grounding_log_path(state),
        [{"session_id": "S1", "turn": 1, "recall_id": "abc", "used_score": 0.9}],
    )

    tm.roll(state, "S1", tp)
    s = tm.summarize(state)
    assert s["sessions"] == 1
    assert s["tool_tok"] == 200
    assert s["answer_tok"] == 70
    assert s["injected_tokens"] == 100  # 400 chars / 4
    assert s["grounded"] == 1


def test_meter_flag_declared_and_default_on(monkeypatch):
    monkeypatch.delenv("MEMO_TOKEN_METER_ENABLED", raising=False)
    from memo.flags import REGISTRY, flag_bool

    assert "MEMO_TOKEN_METER_ENABLED" in REGISTRY
    assert flag_bool("MEMO_TOKEN_METER_ENABLED") is True  # opt-out default-on


def test_summarize_proxy_grounded_vs_ungrounded(tmp_path):
    from memo.dashboard_logs import context_cost_log_path, grounding_log_path

    state = tmp_path / "state"
    state.mkdir()
    # grounded session: low tool spend; ungrounded-but-injected: high tool spend
    tg = _transcript(tmp_path, "G", [(50, 20)])  # 1 turn, tool 50
    tu = _transcript(tmp_path, "U", [(500, 20)])  # 1 turn, tool 500
    _write_jsonl(
        context_cost_log_path(state),
        [
            {"kind": "recall", "session_id": "G", "turn": 1, "chars": 100},
            {"kind": "recall", "session_id": "U", "turn": 1, "chars": 100},
        ],
    )
    _write_jsonl(
        grounding_log_path(state),
        [{"session_id": "G", "turn": 1, "recall_id": "x", "used_score": 0.9}],
    )
    tm.roll(state, "G", tg)
    tm.roll(state, "U", tu)
    s = tm.summarize(state)
    p = s["proxy"]
    assert p["grounded_tool_tok_per_turn"] == 50.0
    assert p["ungrounded_tool_tok_per_turn"] == 500.0
    assert (
        p["delta"] == 450.0
    )  # ungrounded − grounded (positive ⇒ memo correlates with less tool spend)


def test_summarize_net_subtracts_the_injected_context_it_paid_for(tmp_path):
    """`delta` counts only the tool-loop side of the trade. The injected recall
    block is a real cost on every turn memo fires, so the headline number has to
    be net of it — on the live corpus `delta` was negative and the estimated
    savings panel beside it still read 1.79M.

    Here: 2 turns total, 100 injected chars each = 200 chars = 50 tokens over 2
    turns = 25 tok/turn of context bought. Net = 450 − 25 = 425.
    """
    from memo.dashboard_logs import context_cost_log_path, grounding_log_path

    state = tmp_path / "state"
    state.mkdir()
    tg = _transcript(tmp_path, "G", [(50, 20)])
    tu = _transcript(tmp_path, "U", [(500, 20)])
    _write_jsonl(
        context_cost_log_path(state),
        [
            {"kind": "recall", "session_id": "G", "turn": 1, "chars": 100},
            {"kind": "recall", "session_id": "U", "turn": 1, "chars": 100},
        ],
    )
    _write_jsonl(
        grounding_log_path(state),
        [{"session_id": "G", "turn": 1, "recall_id": "x", "used_score": 0.9}],
    )
    tm.roll(state, "G", tg)
    tm.roll(state, "U", tu)

    p = tm.summarize(state)["proxy"]

    assert p["injected_tok_per_turn"] == 25.0
    assert p["net_tok_per_turn"] == 425.0
    # n is part of the claim: a delta over two sessions is not evidence.
    assert p["grounded_turns"] == 1
    assert p["ungrounded_turns"] == 1


def test_summarize_net_is_none_without_both_cohorts(tmp_path):
    """No ungrounded sessions to compare against ⇒ no net claim, rather than a
    net computed against a missing baseline."""
    from memo.dashboard_logs import context_cost_log_path, grounding_log_path

    state = tmp_path / "state"
    state.mkdir()
    tg = _transcript(tmp_path, "G", [(50, 20)])
    _write_jsonl(
        context_cost_log_path(state),
        [{"kind": "recall", "session_id": "G", "turn": 1, "chars": 100}],
    )
    _write_jsonl(
        grounding_log_path(state),
        [{"session_id": "G", "turn": 1, "recall_id": "x", "used_score": 0.9}],
    )
    tm.roll(state, "G", tg)

    p = tm.summarize(state)["proxy"]

    assert p["delta"] is None
    assert p["net_tok_per_turn"] is None


# ---------------------------------------------------------------------------
# Task-4: additive measured panel in `memo tokens`
# ---------------------------------------------------------------------------


def _cli_env(tmp_path):
    return {
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_DATA_DIR": str(tmp_path / "data"),
        "MEMO_STATE_DIR": str(tmp_path / "state"),
    }


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
    assert "measured" in payload  # additive key must actually be written


# ---------------------------------------------------------------------------
# ccusage-style prompt-side accounting: 4-field usage + per-model breakdown
# ---------------------------------------------------------------------------


def test_session_usage_aggregates_input_cache_and_models(tmp_path):
    """input_tokens is a per-call footprint (take the MAX); the cache splits
    are billed per-call volumes (SUM); models tally output spend by name.
    ``<synthetic>`` rows (internal generations) stay out of the prompt-side
    aggregation — nothing user-facing bills them."""
    rows = [
        _human(),
        _assistant(
            "m1",
            30,
            tool=True,
            model="claude-opus-5",
            input_tok=500,
            cache_read=300,
            cache_creation=200,
        ),
        _tool_result(),
        _assistant(
            "m2",
            10,
            model="claude-opus-5",
            input_tok=900,
            cache_read=700,
            cache_creation=0,
        ),
        _human(),
        _assistant("m3", 5, model="claude-sonnet-5", input_tok=400, cache_read=400),
        # synthetic rows must not pollute footprint/cache/model accounting
        _assistant("m4", 999, model="<synthetic>", input_tok=1),
    ]
    p = tmp_path / "s.jsonl"
    _write_jsonl(p, rows)
    su = tm.session_usage(p)
    assert su is not None
    assert su.output_tok == 1044  # answer/tool turns stay inclusive (v1 behavior)
    assert su.input_tok == 900  # MAX, not sum (footprint of the last call)
    assert su.cache_read_tok == 1400  # 300 + 700 + 400
    assert su.cache_creation_tok == 200
    assert su.models == {"claude-opus-5": 40, "claude-sonnet-5": 5}


def test_prompt_side_dedups_streaming_rows(tmp_path):
    """The JSONL repeats one message across streaming rows; the prompt-side
    pass must count each unique message once (mirroring iter_prompt_turns)."""
    rows = [
        _human(),
        _assistant("m1", 20, model="claude-opus-5", input_tok=900, cache_read=500),
        _assistant("m1", 20, model="claude-opus-5", input_tok=900, cache_read=500),
        _assistant("m1", 20, model="claude-opus-5", input_tok=900, cache_read=500),
        _assistant(
            "m2", 10, tool=True, model="claude-sonnet-5", input_tok=800, cache_read=700
        ),
        _assistant(
            "m2", 10, tool=True, model="claude-sonnet-5", input_tok=800, cache_read=700
        ),
    ]
    p = tmp_path / "dup.jsonl"
    _write_jsonl(p, rows)
    su = tm.session_usage(p)
    assert su is not None
    assert su.input_tok == 900  # max over unique messages
    assert su.cache_read_tok == 1200  # 500 + 700, NOT 500*3 + 700*2
    assert su.models == {"claude-opus-5": 20, "claude-sonnet-5": 10}


def test_prompt_side_ignores_degenerate_input_tokens(tmp_path):
    """Some Claude Code builds stamp input_tokens=1..2 on every call. The
    footprint must read as unknown (0) then, while the cache splits (real
    billed volumes) and the model tally still land."""
    rows = [
        _human(),
        _assistant("m1", 20, model="claude-opus-5", input_tok=2, cache_read=21420),
        _assistant(
            "m2", 10, tool=True, model="claude-sonnet-5", input_tok=2, cache_read=98078
        ),
    ]
    p = tmp_path / "deg.jsonl"
    _write_jsonl(p, rows)
    su = tm.session_usage(p)
    assert su is not None
    assert su.input_tok == 0
    assert su.cache_read_tok == 119498
    assert su.models == {"claude-opus-5": 20, "claude-sonnet-5": 10}


def test_roll_persists_prompt_side_and_summarize_exposes_it(tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    rows = [
        _human(),
        _assistant(
            "m1", 20, model="claude-opus-5", input_tok=600, cache_read=500, cache_creation=60
        ),
    ]
    tp = tmp_path / "S2.jsonl"
    _write_jsonl(tp, rows)
    tm.roll(state, "S2", tp)
    s = tm.summarize(state)
    assert s["input_tok"] == 600
    assert s["cache_read_tok"] == 500
    assert s["cache_creation_tok"] == 60
    assert s["models"] == {"claude-opus-5": 20}


def test_summarize_older_ledger_rows_stay_readable(tmp_path):
    """v1 ledger rows (no prompt-side keys) must not break summarize."""
    import json

    state = tmp_path / "state"
    state.mkdir()
    ledger = {
        "schema": "memo.token_meter.sessions.v1",
        "sessions": {
            "OLD": {
                "ts": "2026-07-01T00:00:00+00:00",
                "n_turns": 3,
                "answer_tok": 60,
                "tool_tok": 40,
                "injected_chars": 800,
                "grounded": 1,
            }
        },
    }
    (state / "token_meter.json").write_text(json.dumps(ledger), encoding="utf-8")
    s = tm.summarize(state)
    assert s["sessions"] == 1
    assert s["answer_tok"] == 60
    assert s["input_tok"] == 0
    assert s["models"] == {}


def test_tokens_cmd_shows_prompt_side_line(tmp_path):
    from memo.cli import cli

    state = tmp_path / "state"
    state.mkdir(parents=True)
    rows = [
        _human(),
        _assistant(
            "m1", 20, model="claude-opus-5", input_tok=600, cache_read=500, cache_creation=60
        ),
    ]
    tp = tmp_path / "S3.jsonl"
    _write_jsonl(tp, rows)
    tm.roll(state, "S3", tp)
    runner = CliRunner()
    res = runner.invoke(cli, ["tokens"], env=_cli_env(tmp_path))
    assert res.exit_code == 0
    assert "input footprint" in res.output
    assert "cache-read" in res.output
    assert "by model" in res.output
    assert "claude-opus-5" in res.output
