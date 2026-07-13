from pathlib import Path

from memo.graduation import ledger


def test_record_then_history_roundtrip(tmp_path: Path):
    sd = tmp_path
    ledger.record(sd, "MEMO_X", {"verdict": "confirmed", "realized_delta": 0.02})
    ledger.record(sd, "MEMO_X", {"verdict": "reverted", "realized_delta": -0.01})
    hist = ledger.history(sd, "MEMO_X")
    assert [h["verdict"] for h in hist] == ["confirmed", "reverted"]  # oldest->newest


def test_streak_counts_trailing_confirmed_wins(tmp_path: Path):
    sd = tmp_path
    for _ in range(3):
        ledger.record(sd, "MEMO_X", {"verdict": "confirmed", "realized_delta": 0.01})
    assert ledger.streak(sd, "MEMO_X") == 3
    ledger.record(sd, "MEMO_X", {"verdict": "reverted", "realized_delta": -0.02})
    assert ledger.streak(sd, "MEMO_X") == 0  # a loss breaks the streak


def test_history_is_per_flag_isolated(tmp_path: Path):
    ledger.record(tmp_path, "MEMO_A", {"verdict": "confirmed", "realized_delta": 0.0})
    assert ledger.history(tmp_path, "MEMO_B") == []


def test_corrupt_lines_are_skipped(tmp_path: Path):
    p = tmp_path / "graduation" / "MEMO_X.jsonl"
    p.parent.mkdir(parents=True)
    p.write_text('{"verdict": "confirmed", "realized_delta": 0.0}\nnot-json\n', encoding="utf-8")
    assert len(ledger.history(tmp_path, "MEMO_X")) == 1
