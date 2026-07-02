from memo import dream_tune_online as dto


def test_graduation_streak_counts_trailing_confirmed():
    entries = [
        {"verdict": "reverted", "realized_delta": -0.1},
        {"verdict": "confirmed", "realized_delta": 0.03},
        {"verdict": "confirmed", "realized_delta": 0.0},
    ]
    assert dto.graduation_streak(entries) == 2


def test_graduation_streak_breaks_on_expired_or_negative():
    assert dto.graduation_streak([{"verdict": "expired"}]) == 0
    assert (
        dto.graduation_streak(
            [
                {"verdict": "confirmed", "realized_delta": 0.1},
                {"verdict": "confirmed", "realized_delta": -0.01},
            ]
        )
        == 0
    )  # newest has negative delta → streak 0


def test_graduation_status_reads_ledger(tmp_path):
    for _ in range(3):
        dto.append_ledger(tmp_path, {"verdict": "confirmed", "realized_delta": 0.02})
    st = dto.graduation_status(tmp_path, k=2)
    assert st == {"streak": 3, "k": 2, "graduated": True}
    st2 = dto.graduation_status(tmp_path, k=5)
    assert st2["graduated"] is False
