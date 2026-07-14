"""Next-turn verdict classifier — heuristic ES+EN reaction classification."""

from __future__ import annotations

import pytest

from memo.dashboard import append_recall_log, append_verdict_log, read_verdict_log
from memo.verdict import classify_reaction, score_next_turn


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # corrections — ES + EN
        ("no, eso está mal — el daemon usa el puerto 8765", "correction"),
        ("that's wrong, the flag lives in flags_search.py", "correction"),
        ("te equivocaste, era el otro archivo", "correction"),
        ("not what I asked — I meant the sync tier", "correction"),
        ("no es así, el overlay pisa el default no el env", "correction"),
        # negative outcomes
        ("sigue fallando con el mismo error", "negative"),
        ("still failing after that change", "negative"),
        ("no funciona, tira ImportError", "negative"),
        ("doesn't work — same traceback", "negative"),
        # acceptance
        ("perfecto, gracias!", "positive"),
        ("that worked, thanks", "positive"),
        ("genial, ya funciona", "positive"),
        ("works now. next: add the docs", "positive"),
        # neutral — no verdict
        ("ahora agregá tests para el módulo de sync", None),
        ("cómo funciona el recall de memo?", None),
        ("explain how it works under the hood", None),
        ("", None),
    ],
)
def test_classify_reaction(text: str, expected: str | None) -> None:
    assert classify_reaction(text) == expected


def test_negation_head_beats_thanks() -> None:
    # "no, gracias — eso no era lo que pedí" is a rejection, not a thank-you.
    assert classify_reaction("no, gracias — eso no era lo que pedí") == "correction"


def test_signal_must_be_in_head() -> None:
    # A long new request that happens to end in "gracias" is NOT a reaction.
    filler = "quiero que refactorices este módulo largo y agregues validación. " * 5
    assert classify_reaction(filler + "gracias") is None


# ---------------------------------------------------------------------------
# Task 2: score_next_turn + verdict.log
# ---------------------------------------------------------------------------


def _hit(hid: str, title: str = "arch note") -> dict:
    return {"id": hid, "score": 0.9, "title": title, "snippet": "body text"}


def test_score_next_turn_attributes_prior_recall(tmp_path) -> None:
    append_recall_log(
        tmp_path,
        prompt="cómo configuro el sync remoto?",
        hits=[_hit("aaaabbbb11112222")],
        via="subprocess",
        session_id="s1",
        turn=3,
    )
    append_recall_log(
        tmp_path,
        prompt="no funciona, tira el mismo error",
        hits=[],
        via="subprocess",
        session_id="s1",
        turn=4,
    )
    rec = score_next_turn(tmp_path, {"session_id": "s1"})
    assert rec is not None
    assert rec["verdict"] == "negative"
    assert rec["turn"] == 4 and rec["prior_turn"] == 3
    assert rec["recall_ids"] == ["aaaabbbb"]  # append_recall_log stores 8-char ids
    assert rec["prompt"] == "cómo configuro el sync remoto?"


def test_score_next_turn_none_without_verdict(tmp_path) -> None:
    append_recall_log(
        tmp_path,
        prompt="cómo configuro el sync remoto?",
        hits=[_hit("aaaabbbb11112222")],
        via="subprocess",
        session_id="s1",
        turn=3,
    )
    append_recall_log(
        tmp_path,
        prompt="ahora agregá tests para el módulo",
        hits=[],
        via="subprocess",
        session_id="s1",
        turn=4,
    )
    assert score_next_turn(tmp_path, {"session_id": "s1"}) is None


def test_score_next_turn_respects_turn_gap(tmp_path) -> None:
    append_recall_log(
        tmp_path,
        prompt="cómo configuro el sync remoto?",
        hits=[_hit("aaaabbbb11112222")],
        via="subprocess",
        session_id="s1",
        turn=1,
    )
    append_recall_log(tmp_path, prompt="dale", hits=[], via="subprocess", session_id="s1", turn=2)
    append_recall_log(
        tmp_path,
        prompt="otra cosa distinta acá",
        hits=[],
        via="subprocess",
        session_id="s1",
        turn=3,
    )
    append_recall_log(
        tmp_path,
        prompt="sigue fallando con el mismo error",
        hits=[],
        via="subprocess",
        session_id="s1",
        turn=4,
    )
    # prior recall is 3 turns back (> _MAX_TURN_GAP=2) — no attribution
    assert score_next_turn(tmp_path, {"session_id": "s1"}) is None


def test_score_next_turn_dedups_repeat_stop_events(tmp_path) -> None:
    append_recall_log(
        tmp_path,
        prompt="cómo configuro el sync remoto?",
        hits=[_hit("aaaabbbb11112222")],
        via="subprocess",
        session_id="s1",
        turn=3,
    )
    append_recall_log(
        tmp_path,
        prompt="no funciona, tira el mismo error",
        hits=[],
        via="subprocess",
        session_id="s1",
        turn=4,
    )
    assert score_next_turn(tmp_path, {"session_id": "s1"}) is not None
    assert score_next_turn(tmp_path, {"session_id": "s1"}) is None  # already written


class _StubMem:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def feedback_record(self, source_id, *, query_text, rating, only_if_absent=False, extra=None):
        self.calls.append(
            (source_id, query_text, rating, only_if_absent, (extra or {}).get("origin"))
        )
        return {"feedback_id": "f1"}


def test_record_verdicts_writes_implicit_feedback(tmp_cfg) -> None:
    from memo.verdict import record_verdicts

    sd = tmp_cfg.state_dir
    append_recall_log(
        sd,
        prompt="cómo configuro el sync remoto?",
        hits=[_hit("aaaabbbb11112222")],
        via="subprocess",
        session_id="s1",
        turn=3,
    )
    append_recall_log(
        sd,
        prompt="no funciona, tira el mismo error",
        hits=[],
        via="subprocess",
        session_id="s1",
        turn=4,
    )
    mem = _StubMem()
    rec = record_verdicts(tmp_cfg, {"session_id": "s1"}, memory=mem)
    assert rec is not None and rec["verdict"] == "negative"
    assert mem.calls == [
        ("aaaabbbb", "cómo configuro el sync remoto?", "ignore", True, "next_turn_verdict"),
    ]


def test_record_verdicts_positive_maps_to_click(tmp_cfg) -> None:
    from memo.verdict import record_verdicts

    sd = tmp_cfg.state_dir
    append_recall_log(
        sd,
        prompt="cómo configuro el sync remoto?",
        hits=[_hit("ccccdddd11112222")],
        via="subprocess",
        session_id="s2",
        turn=1,
    )
    append_recall_log(
        sd,
        prompt="perfecto, gracias — quedó andando",
        hits=[],
        via="subprocess",
        session_id="s2",
        turn=2,
    )
    mem = _StubMem()
    rec = record_verdicts(tmp_cfg, {"session_id": "s2"}, memory=mem)
    assert rec["verdict"] == "positive"
    assert mem.calls[0][2] == "click"


def test_capture_stop_calls_verdicts_when_enabled(tmp_cfg, monkeypatch) -> None:
    import json as _json

    from click.testing import CliRunner

    from memo.cli import cli

    called: dict = {}
    monkeypatch.setattr("memo.capture.run_capture", lambda *a, **k: {"saved_titles": []})
    monkeypatch.setattr("memo.grounding.score_turn", lambda *a, **k: None)
    monkeypatch.setattr("memo.token_ledger.roll_up", lambda *a, **k: {})
    monkeypatch.setattr(
        "memo.verdict.record_verdicts",
        lambda cfg, payload, **k: called.setdefault("payload", payload),
    )
    transcript = tmp_cfg.state_dir / "t.jsonl"
    transcript.write_text("", encoding="utf-8")
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["capture-stop"],
        input=_json.dumps({"transcript_path": str(transcript), "session_id": "s1"}),
        env={
            "MEMO_NONINTERACTIVE": "1",
            "MEMO_VERDICT_ENABLED": "1",
            "MEMO_DATA_DIR": str(tmp_cfg.data_dir),
            "MEMO_STATE_DIR": str(tmp_cfg.state_dir),
        },
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    assert called["payload"]["session_id"] == "s1"


def test_capture_stop_skips_verdicts_when_flag_off(tmp_cfg, monkeypatch) -> None:
    import json as _json

    from click.testing import CliRunner

    from memo.cli import cli

    called: dict = {}
    monkeypatch.setattr("memo.capture.run_capture", lambda *a, **k: {"saved_titles": []})
    monkeypatch.setattr("memo.grounding.score_turn", lambda *a, **k: None)
    monkeypatch.setattr("memo.token_ledger.roll_up", lambda *a, **k: {})
    monkeypatch.setattr(
        "memo.verdict.record_verdicts",
        lambda cfg, payload, **k: called.setdefault("payload", payload),
    )
    transcript = tmp_cfg.state_dir / "t.jsonl"
    transcript.write_text("", encoding="utf-8")
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["capture-stop"],
        input=_json.dumps({"transcript_path": str(transcript), "session_id": "s1"}),
        env={
            "MEMO_NONINTERACTIVE": "1",  # MEMO_VERDICT_ENABLED intentionally unset
            "MEMO_DATA_DIR": str(tmp_cfg.data_dir),
            "MEMO_STATE_DIR": str(tmp_cfg.state_dir),
        },
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    assert called == {}, "flag off must not call record_verdicts"


def test_verdict_log_roundtrip(tmp_path) -> None:
    append_verdict_log(
        tmp_path,
        session_id="s1",
        turn=4,
        prior_turn=3,
        verdict="negative",
        prompt="q",
        reaction="no funciona",
        recall_ids=["aaaabbbb11112222"],
    )
    rows = read_verdict_log(tmp_path)
    assert len(rows) == 1
    assert rows[0]["verdict"] == "negative"
    assert rows[0]["recall_ids"] == ["aaaabbbb"]
