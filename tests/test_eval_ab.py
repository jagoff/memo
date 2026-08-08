"""Tests for `memo eval ab` — blind-judge A/B (recall context ON vs OFF).

All unit tests stub the chat and search callables — no MLX forward pass ever
runs here (the real run is `memo eval ab`, owner-invoked). CliRunner tests pin
MEMO_DATA_DIR / MEMO_STATE_DIR / MEMO_NONINTERACTIVE per conftest discipline.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from click.testing import CliRunner

from memo import eval_ab
from memo.cli_eval import eval_group
from memo.eval_recall import LabelSet, Prompt

# --- Helpers ------------------------------------------------------------------


@dataclass
class _Hit:
    id: str
    title: str
    body: str
    score: float = 0.9


def _judge_json(a: dict[str, float], b: dict[str, float]) -> str:
    return json.dumps({"a": a, "b": b})


_GOOD = {"correctness": 0.9, "groundedness": 0.9, "specificity": 0.9}
_BAD = {"correctness": 0.2, "groundedness": 0.1, "specificity": 0.2}


class _ScriptedChat:
    """Chat stub: answers responder calls, scores judge calls; records calls.

    Both responder conditions share the same system prompt (symmetric by
    design), so the stub tells them apart by the user turn: the ON turn opens
    with the neutral context label, the OFF turn is the bare question.
    `on_wins=True` makes the judge favor whichever side is the ON answer —
    it detects the ON answer by the marker text the responder stub embeds.
    """

    def __init__(self, on_wins: bool = True) -> None:
        self.on_wins = on_wins
        self.calls: list[tuple[str, str]] = []

    def __call__(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        if system == eval_ab.JUDGE_SYSTEM:
            # judge call: figure out which side (A/B) carries the ON answer
            a_part = user.split("Answer A:", 1)[1].split("Answer B:", 1)[0]
            on_is_a = "ON-ANSWER" in a_part
            good, bad = (_GOOD, _BAD) if self.on_wins else (_BAD, _GOOD)
            return _judge_json(good, bad) if on_is_a else _judge_json(bad, good)
        if user.startswith(eval_ab.CONTEXT_LABEL):
            return "ON-ANSWER grounded detail"
        return "OFF-ANSWER generic"


# Bodies are >= 40 chars so the hits also survive the recall pipeline's
# min_body_chars gate when routed through recall_search_fn (CLI tests).
_BODY_ONE = "lambdas deploy via terraform apply from the infra repo pipeline"
_BODY_TWO = "body two carries enough padding characters to pass the min-body gate"


def _search_stub(_query: str) -> list[_Hit]:
    return [
        _Hit(id="aaaaaaaa1111", title="Deploy runbook", body=_BODY_ONE),
        _Hit(id="bbbbbbbb2222", title="Other note", body=_BODY_TWO),
    ]


# --- answerable_prompts -------------------------------------------------------


def test_answerable_prompts_filters_unanswerable():
    labels = LabelSet(
        prompts=[
            Prompt("has expect ids", expect_ids=["aaaaaaaa1111"]),
            Prompt("relevant no ids", relevant=True),
            Prompt("not answerable", relevant=False),
        ]
    )
    texts = [p.text for p in eval_ab.answerable_prompts(labels)]
    assert texts == ["has expect ids", "relevant no ids"]


# --- build_context ------------------------------------------------------------


def test_build_context_has_title_and_body_but_never_ids():
    ctx = eval_ab.build_context(_search_stub(""))
    assert "Deploy runbook" in ctx
    assert "lambdas deploy via terraform" in ctx
    assert "aaaaaaaa1111" not in ctx  # ids must never leak toward the judge
    assert "bbbbbbbb2222" not in ctx


def test_build_context_truncates_bodies():
    hits = [_Hit(id="cc", title="T", body="x" * 5000)]
    ctx = eval_ab.build_context(hits, per_hit_chars=100)
    assert len(ctx) < 300


# --- symmetric responder prompts (judge blindness, F1a) -----------------------


def test_responder_system_is_shared_and_forbids_source_mentions():
    # ONE system prompt for both conditions — no per-condition framing at all.
    chat = _ScriptedChat()
    eval_ab.run_ab([Prompt("q?", relevant=True)], search=_search_stub, chat=chat, k=2, seed=42)
    responder_systems = {s for s, _ in chat.calls if s != eval_ab.JUDGE_SYSTEM}
    assert responder_systems == {eval_ab.RESPONDER_SYSTEM}
    assert "Never mention what sources, context, notes, or memory" in eval_ab.RESPONDER_SYSTEM
    # the old asymmetric framing must be gone from the shared prompt
    for tell in ("MEMORY CONTEXT", "saved memory"):
        assert tell not in eval_ab.RESPONDER_SYSTEM


def test_responder_user_on_uses_neutral_label_not_memory_words():
    turn = eval_ab.responder_user_on("q?", "1. Note\n   body")
    assert turn.startswith(eval_ab.CONTEXT_LABEL)
    assert "MEMORY CONTEXT" not in turn
    assert "memor" not in turn.lower()  # no memory/memories vocabulary at all


def test_responder_user_on_recall_miss_is_identical_to_off_turn():
    # A miss must not inject "(no memories recalled)" — the ON turn collapses
    # to the bare question, byte-identical to the OFF condition.
    assert eval_ab.responder_user_on("la pregunta", "") == "la pregunta"


# --- leak scrub (judge blindness, F1b) ----------------------------------------


def test_detect_leak_flags_telltale_phrases_case_insensitive():
    for phrase in eval_ab.LEAK_PHRASES:
        assert eval_ab.detect_leak(f"Well, according to the {phrase.upper()}, use X.") is True
    assert eval_ab.detect_leak("Deploys go through terraform in the infra repo.") is False
    assert eval_ab.detect_leak("") is False


def test_run_ab_leaked_answer_forces_tie_and_never_reaches_judge():
    class _LeakyChat(_ScriptedChat):
        def __call__(self, system: str, user: str) -> str:
            out = super().__call__(system, user)
            if out.startswith("ON-ANSWER"):
                return "Based on your saved memory, deploys use terraform."
            return out

    chat = _LeakyChat()
    (r,) = eval_ab.run_ab(
        [Prompt("donde se deploya", relevant=True)], search=_search_stub, chat=chat, k=2, seed=42
    )
    assert r.leaked is True
    assert r.winner == "tie"
    assert set(r.scores_on.values()) == {0.0} and set(r.scores_off.values()) == {0.0}
    assert r.judge_raw == ""
    # the de-anonymizing pair must never be shown to the judge
    assert not any(s == eval_ab.JUDGE_SYSTEM for s, _ in chat.calls)

    s = eval_ab.summarize([r])
    assert s["leaked_pairs"] == 1
    assert s["ties"] == 1 and s["wins_on"] == 0 and s["losses_on"] == 0


def test_run_ab_leak_in_off_answer_also_forces_tie():
    class _LeakyOffChat(_ScriptedChat):
        def __call__(self, system: str, user: str) -> str:
            out = super().__call__(system, user)
            if out.startswith("OFF-ANSWER"):
                return "I have no memories about that, sorry."
            return out

    chat = _LeakyOffChat()
    (r,) = eval_ab.run_ab(
        [Prompt("pregunta", relevant=True)], search=_search_stub, chat=chat, k=2, seed=42
    )
    assert r.leaked is True and r.winner == "tie"
    assert not any(s == eval_ab.JUDGE_SYSTEM for s, _ in chat.calls)


# --- deterministic order ------------------------------------------------------


def test_on_goes_first_is_deterministic_for_fixed_seed():
    first = [eval_ab.on_goes_first(42, i, f"prompt {i}") for i in range(50)]
    again = [eval_ab.on_goes_first(42, i, f"prompt {i}") for i in range(50)]
    assert first == again
    # and the seed actually mixes: both orders appear across 50 pairs
    assert True in first and False in first


# --- judge parsing ------------------------------------------------------------


def test_parse_judge_scores_roundtrip_and_clamp():
    raw = _judge_json(
        {"correctness": 1.7, "groundedness": -0.5, "specificity": 0.5},
        {"correctness": 0.4, "groundedness": 0.4, "specificity": "bogus"},
    )
    a, b, err = eval_ab.parse_judge_scores(raw)
    assert err is False
    assert a == {"correctness": 1.0, "groundedness": 0.0, "specificity": 0.5}
    assert b["specificity"] == 0.0  # non-numeric clamps to 0


def test_parse_judge_scores_tolerates_surrounding_prose():
    raw = "Sure! Here is my grading:\n" + _judge_json(_GOOD, _BAD) + "\nHope that helps."
    a, b, err = eval_ab.parse_judge_scores(raw)
    assert err is False
    assert a == _GOOD and b == _BAD


def test_parse_judge_scores_failure_zeroes_both_sides():
    for raw in ("", "no json here", '{"a": 1}', '{"a": {}, "nope": {}}'):
        a, b, err = eval_ab.parse_judge_scores(raw)
        assert err is True
        assert set(a.values()) == {0.0} and set(b.values()) == {0.0}


# --- winner decision ----------------------------------------------------------


def test_decide_winner_respects_tie_band():
    assert eval_ab.decide_winner(0.80, 0.78, tie_band=0.05) == "tie"
    assert eval_ab.decide_winner(0.90, 0.50, tie_band=0.05) == "on"
    assert eval_ab.decide_winner(0.50, 0.90, tie_band=0.05) == "off"


# --- run_ab -------------------------------------------------------------------


def test_run_ab_on_condition_gets_context_and_off_does_not():
    chat = _ScriptedChat()
    prompts = [Prompt("donde se deploya la lambda", relevant=True)]
    results = eval_ab.run_ab(prompts, search=_search_stub, chat=chat, k=2, seed=42)

    responder_calls = [u for s, u in chat.calls if s == eval_ab.RESPONDER_SYSTEM]
    on_calls = [u for u in responder_calls if u.startswith(eval_ab.CONTEXT_LABEL)]
    off_calls = [u for u in responder_calls if not u.startswith(eval_ab.CONTEXT_LABEL)]
    assert len(on_calls) == 1 and "Deploy runbook" in on_calls[0]
    assert len(off_calls) == 1 and off_calls[0] == "donde se deploya la lambda"

    (r,) = results
    assert r.n_context_hits == 2
    assert r.context_tokens_on > 0
    assert r.context_tokens_off == 0


def test_run_ab_judge_is_blind_no_ids_no_context_no_condition_names():
    chat = _ScriptedChat()
    prompts = [Prompt("q1", relevant=True, expect_ids=["aaaaaaaa1111"])]
    eval_ab.run_ab(prompts, search=_search_stub, chat=chat, k=2, seed=42)

    judge_calls = [(s, u) for s, u in chat.calls if s == eval_ab.JUDGE_SYSTEM]
    assert len(judge_calls) == 1
    _, user = judge_calls[0]
    assert "aaaaaaaa1111" not in user  # no memory/label ids
    assert eval_ab.CONTEXT_LABEL not in user  # no recall context block
    for leak in ("memo", "recall", "condition", "with memory", "without memory"):
        assert leak not in user.lower()
    for phrase in eval_ab.LEAK_PHRASES:  # judge input clean of every tell-tale
        assert phrase not in user.lower()


def test_run_ab_scores_map_back_through_randomized_order():
    # Across several prompts both orders occur; the ON side must always get
    # the good scores regardless of A/B position when the judge favors ON.
    chat = _ScriptedChat(on_wins=True)
    prompts = [Prompt(f"question number {i}", relevant=True) for i in range(8)]
    results = eval_ab.run_ab(prompts, search=_search_stub, chat=chat, k=2, seed=7)

    orders = {r.order for r in results}
    assert orders == {"on_first", "off_first"}  # both positions exercised
    for r in results:
        assert r.scores_on == _GOOD
        assert r.scores_off == _BAD
        assert r.winner == "on"


def test_run_ab_off_can_win():
    chat = _ScriptedChat(on_wins=False)
    prompts = [Prompt("pregunta", relevant=True)]
    (r,) = eval_ab.run_ab(prompts, search=_search_stub, chat=chat, k=2, seed=42)
    assert r.winner == "off"
    assert r.mean_off > r.mean_on


def test_run_ab_judge_parse_error_forces_tie():
    class _BrokenJudgeChat(_ScriptedChat):
        def __call__(self, system: str, user: str) -> str:
            if system == eval_ab.JUDGE_SYSTEM:
                self.calls.append((system, user))
                return "I refuse to emit JSON"
            return super().__call__(system, user)

    chat = _BrokenJudgeChat()
    prompts = [Prompt("pregunta", relevant=True)]
    (r,) = eval_ab.run_ab(prompts, search=_search_stub, chat=chat, k=2, seed=42)
    assert r.judge_parse_error is True
    assert r.winner == "tie"


# --- summarize ----------------------------------------------------------------


def test_summarize_counts_and_token_totals():
    chat = _ScriptedChat(on_wins=True)
    prompts = [Prompt(f"q {i}", relevant=True) for i in range(3)]
    results = eval_ab.run_ab(prompts, search=_search_stub, chat=chat, k=2, seed=42)
    s = eval_ab.summarize(results)
    assert s["prompts"] == 3
    assert s["wins_on"] == 3 and s["ties"] == 0 and s["losses_on"] == 0
    assert s["win_rate_on"] == 1.0
    assert s["mean_delta"] > 0
    assert set(s["sub_deltas"]) == set(eval_ab.SUBSCORES)
    assert s["context_tokens_on"] == sum(r.context_tokens_on for r in results) > 0
    assert s["context_tokens_off"] == 0
    assert s["judge_parse_errors"] == 0
    assert s["leaked_pairs"] == 0


def test_summarize_empty_is_all_zero():
    s = eval_ab.summarize([])
    assert s["prompts"] == 0
    assert s["win_rate_on"] == 0.0
    assert s["mean_delta"] == 0.0
    assert s["leaked_pairs"] == 0


# --- recall_search_fn (recall-faithful ON retrieval, F2) ----------------------


class _RecordingMemory:
    """Memory stub for recall_search_fn: records search kwargs, returns hits."""

    def __init__(self, hits: list[_Hit]) -> None:
        self.hits = hits
        self.calls: list[tuple[str, dict]] = []

    def search(self, query: str, **kwargs) -> list[_Hit]:
        self.calls.append((query, dict(kwargs)))
        return list(self.hits)


def _pin_recall_flags(monkeypatch) -> None:
    """Pin the flags recall_search_fn resolves so the test is deterministic
    regardless of the developer's shell environment."""
    monkeypatch.setenv("MEMO_RECALL_MODE", "vec")
    monkeypatch.setenv("MEMO_RECALL_MIN_SIM", "0.5")
    monkeypatch.setenv("MEMO_RECALL_MIN_BODY_CHARS", "40")
    monkeypatch.setenv("MEMO_RECALL_RECENCY_BAND_DAYS", "0")
    monkeypatch.setenv("MEMO_RECALL_EXCLUDE_REFERENCE", "1")
    monkeypatch.setenv("MEMO_RECALL_EXCLUDE_UNCERTAIN", "1")
    monkeypatch.setenv("MEMO_RECALL_SKIP_BELOW", "0.45")
    monkeypatch.setenv("MEMO_RECALL_GAP_THRESHOLD", "0.10")
    monkeypatch.setenv("MEMO_RECALL_UNMATCHED_TERM_GATE", "0")
    monkeypatch.setenv("MEMO_RECALL_DEDUP_COLLAPSE", "1")
    monkeypatch.setenv("MEMO_RECALL_INTRA_DEDUP_THRESHOLD", "0.8")
    monkeypatch.setenv("MEMO_RECALL_MMR_LAMBDA", "0")
    monkeypatch.setenv("MEMO_RECALL_SYNTHESIS_BOOST", "0")
    monkeypatch.setenv("MEMO_RECALL_ALTITUDE", "0")


def test_recall_search_fn_builds_pool_like_the_recall_pipeline(monkeypatch):
    _pin_recall_flags(monkeypatch)
    mem = _RecordingMemory(_search_stub(""))
    hits = eval_ab.recall_search_fn(mem, k=2)("donde se deploya la lambda")

    assert [h.id for h in hits] == ["aaaaaaaa1111", "bbbbbbbb2222"]
    ((query, kwargs),) = mem.calls
    assert query == "donde se deploya la lambda"
    # the recall-faithful pool, not a raw top-k search: over-fetch + the
    # hook's exclusions + reranker off (rank_hits IS the ranking, as in
    # eval_recall's gate)
    assert kwargs["limit"] == 8  # k * 4
    assert kwargs["disable_reranker"] is True
    assert kwargs["mode"] == "vec"
    assert kwargs["exclude_types"] == {"reference"}
    assert kwargs["exclude_tags"] == {"_uncertain"}


def test_recall_search_fn_applies_rank_hits_gates(monkeypatch):
    _pin_recall_flags(monkeypatch)
    long_body = "a body comfortably longer than the forty-char minimum gate"
    mem = _RecordingMemory(
        [
            _Hit(id="keep1", title="Top hit", body=long_body, score=0.9),
            _Hit(id="short", title="Short body", body="tiny", score=0.95),
            _Hit(id="lowsim", title="Below floor", body=long_body + " again", score=0.2),
            _Hit(
                id="keep2",
                title="Second hit",
                body="another distinct body long enough to pass the gate",
                score=0.85,
            ),
            _Hit(
                id="keep3",
                title="Third hit",
                body="completely different content about database migrations and index tuning",
                score=0.8,
            ),
        ]
    )
    hits = eval_ab.recall_search_fn(mem, k=2)("query")
    # min_sim drops lowsim, min_body drops short, top-k caps at 2 — the same
    # gates rank_hits applies for the hook and the eval_recall gate.
    assert [h.id for h in hits] == ["keep1", "keep2"]


def _access_snapshot(mem) -> tuple[int, int, str | None]:
    row = mem.store._conn.execute(
        "SELECT COUNT(*), COALESCE(SUM(access_count), 0), MAX(last_accessed) FROM access"
    ).fetchone()
    return tuple(row)


def test_recall_search_fn_does_not_inflate_access_count(mock_memory, monkeypatch):
    """`memo eval ab`'s ON-condition retrieval reproduces the recall pipeline
    against the live corpus; without `_track_usage=False` every search hit
    writes an access-log row (search_ops.py's `_stage_record_usage`),
    inflating access_count on whatever memory the eval surfaces — the same
    signal `memo usefulness` / `dead_weight()` read to decide what's noise.
    """
    _pin_recall_flags(monkeypatch)
    mock_memory.save(
        content="the alpha rollout decision was made after deliberation about the gates",
        title="alpha rollout decision",
        auto_project=False,
    )

    before = _access_snapshot(mock_memory)
    eval_ab.recall_search_fn(mock_memory, k=3)("alpha rollout decision")
    assert _access_snapshot(mock_memory) == before

    # Contrast: a real (non-eval) search against the same corpus DOES bump
    # it — proving the assertion above isn't vacuously true because search
    # never actually ran against a real candidate.
    mock_memory.search("alpha rollout decision", mode="vec")
    assert _access_snapshot(mock_memory) != before


# --- CLI ----------------------------------------------------------------------


def _write_labels(tmp_path) -> str:
    path = tmp_path / "labels.json"
    path.write_text(
        json.dumps(
            {
                "schema": "memo.eval_recall.labels.v1",
                "prompts": [
                    {"text": "donde se deploya la lambda", "relevant": True},
                    {"text": "pregunta sin respuesta", "relevant": False},
                ],
            }
        ),
        encoding="utf-8",
    )
    return str(path)


class _StubMLXChat:
    """Stands in for memo.llm.MLXChat inside the CLI command."""

    def chat(self, model: str, messages: list[dict[str, str]], options: dict | None = None):
        system = messages[0]["content"]
        if system == eval_ab.JUDGE_SYSTEM:
            return {"message": {"content": _judge_json(_GOOD, _BAD)}}
        return {"message": {"content": f"answer via {system[:20]}"}}


class _StubMemory:
    def __init__(self) -> None:
        self.search_calls: list[tuple[str, dict]] = []

    def search(self, query: str, **kwargs):
        self.search_calls.append((query, dict(kwargs)))
        return _search_stub(query)


def test_cli_eval_ab_runs_and_writes_detail(tmp_path, monkeypatch):
    state = tmp_path / "state"
    monkeypatch.setenv("MEMO_STATE_DIR", str(state))
    monkeypatch.setenv("MEMO_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MEMO_NONINTERACTIVE", "1")
    _pin_recall_flags(monkeypatch)
    stub_mem = _StubMemory()
    monkeypatch.setattr("memo.cli_eval._get_memory", lambda cfg: stub_mem)
    monkeypatch.setattr("memo.llm.MLXChat", _StubMLXChat)

    res = CliRunner().invoke(eval_group, ["ab", "--labels", _write_labels(tmp_path), "--k", "2"])
    assert res.exit_code == 0, res.output
    assert "1 win / 0 tie / 0 loss over 1 prompts" in res.output  # unanswerable one skipped

    # the ON condition went through the recall-faithful pipeline, not raw search
    ((_, kwargs),) = stub_mem.search_calls
    assert kwargs["limit"] == 8  # k * 4 over-fetched pool
    assert kwargs["disable_reranker"] is True
    assert kwargs["exclude_types"] == {"reference"}

    detail_files = list((state / "eval").glob("ab_*.json"))
    assert len(detail_files) == 1
    detail = json.loads(detail_files[0].read_text(encoding="utf-8"))
    assert detail["schema"] == eval_ab.AB_SCHEMA
    assert detail["prompts_version"] == eval_ab.PROMPTS_VERSION
    assert detail["summary"]["wins_on"] == 1
    assert detail["summary"]["leaked_pairs"] == 0
    assert len(detail["pairs"]) == 1
    assert detail["pairs"][0]["winner"] == "on"
    assert detail["pairs"][0]["n_context_hits"] == 2  # both stub hits survive ranking
    assert detail["pairs"][0]["judge_raw"]  # raw judge output kept for audit


def test_cli_eval_ab_json_output(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMO_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("MEMO_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MEMO_NONINTERACTIVE", "1")
    _pin_recall_flags(monkeypatch)
    monkeypatch.setattr("memo.cli_eval._get_memory", lambda cfg: _StubMemory())
    monkeypatch.setattr("memo.llm.MLXChat", _StubMLXChat)

    res = CliRunner().invoke(
        eval_group, ["ab", "--labels", _write_labels(tmp_path), "--k", "2", "--json"]
    )
    assert res.exit_code == 0, res.output
    payload = json.loads(res.output)
    assert payload["summary"]["prompts"] == 1
    assert payload["seed"] == 42
    assert payload["detail_path"].endswith(".json")


def test_cli_eval_ab_errors_when_no_answerable_prompts(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMO_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("MEMO_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MEMO_NONINTERACTIVE", "1")
    path = tmp_path / "labels.json"
    path.write_text(
        json.dumps(
            {
                "schema": "memo.eval_recall.labels.v1",
                "prompts": [{"text": "nada answerable", "relevant": False}],
            }
        ),
        encoding="utf-8",
    )
    res = CliRunner().invoke(eval_group, ["ab", "--labels", str(path)])
    assert res.exit_code != 0
    assert "no answerable prompts" in res.output
