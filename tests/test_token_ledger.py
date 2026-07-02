"""Tests for the durable token-savings ledger (`memo.token_ledger`).

The ledger folds grounded recall events from the (capped, rotating)
grounding.log into a small per-day file BEFORE they rotate out, so the
all-time historic total is durable and monotonic — it never shrinks when
grounding.log evicts old rows, and it grows as memo accumulates grounded
(actually-used) memories.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from memo import dashboard, token_ledger


def _g(ts: str, score: float = 0.9, client: str = "claude-code") -> dict:
    """Build a grounding.log row for a given UTC timestamp."""
    return {
        "ts": ts,
        "session_id": "s",
        "turn": 1,
        "recall_id": "abcd1234",
        "used_score": score,
        "method": "lexical",
        "client": client,
        "answer_len": 800,
    }


# --- grounded_by_day -------------------------------------------------------


def test_grounded_by_day_counts_only_grounded_rows() -> None:
    rows = [
        _g("2026-06-10T12:00:00+00:00", score=0.9),  # grounded
        _g("2026-06-10T13:00:00+00:00", score=0.61),  # grounded (>= threshold)
        _g("2026-06-10T14:00:00+00:00", score=0.2),  # NOT grounded
        _g("2026-06-11T12:00:00+00:00", score=0.95),  # grounded, next day
    ]
    by_day = token_ledger.grounded_by_day(rows, to_day=lambda ts: ts[:10])
    assert by_day == {"2026-06-10": 2, "2026-06-11": 1}


def test_grounded_by_day_ignores_unparseable_ts() -> None:
    rows = [_g("not-a-date", score=0.9), _g("2026-06-10T12:00:00+00:00", score=0.9)]
    by_day = token_ledger.grounded_by_day(
        rows, to_day=lambda ts: None if ts == "not-a-date" else ts[:10]
    )
    assert by_day == {"2026-06-10": 1}


# --- roll_up (durability + monotonicity) -----------------------------------


def _historic_grounded(ledger: dict) -> int:
    return sum(d["grounded"] for d in ledger["days"].values())


def test_roll_up_persists_grounded_counts(tmp_path: Path) -> None:
    for ts in ("2026-06-10T12:00:00+00:00", "2026-06-10T13:00:00+00:00"):
        dashboard.append_grounding_log(
            tmp_path,
            session_id="s",
            turn=1,
            recall_id="a" * 8,
            used_score=0.9,
            method="lexical",
            answer_len=800,
        )
        _ = ts  # ts not used by append helper; rows share its own now-stamp
    ledger = token_ledger.roll_up(tmp_path)
    # Two grounded rows on whatever local day "now" falls — historic == 2.
    assert _historic_grounded(ledger) == 2


def test_roll_up_is_monotonic_when_log_rotates(tmp_path: Path) -> None:
    """Old days evicted from grounding.log must NOT shrink the durable total."""
    log = dashboard.grounding_log_path(tmp_path)
    # Seed three distinct days, 2 grounded each, by writing the log directly.
    import json

    rows = []
    for day in ("2026-06-01", "2026-06-05", "2026-06-09"):
        rows += [_g(f"{day}T12:00:00+00:00"), _g(f"{day}T13:00:00+00:00")]
    log.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    first = token_ledger.roll_up(tmp_path)
    assert _historic_grounded(first) == 6

    # grounding.log rotates: only the newest day's rows survive.
    log.write_text(
        "\n".join(json.dumps(_g(f"2026-06-09T{h}:00:00+00:00")) for h in (12, 13)) + "\n",
        encoding="utf-8",
    )
    second = token_ledger.roll_up(tmp_path)
    # Historic preserved across rotation (old days kept from the durable ledger).
    assert _historic_grounded(second) == 6


def test_roll_up_accumulates_new_days(tmp_path: Path) -> None:
    import json

    log = dashboard.grounding_log_path(tmp_path)
    log.write_text(json.dumps(_g("2026-06-01T12:00:00+00:00")) + "\n", encoding="utf-8")
    token_ledger.roll_up(tmp_path)

    # A new day's grounded rows arrive in the log.
    with log.open("a", encoding="utf-8") as f:
        f.write(json.dumps(_g("2026-06-02T12:00:00+00:00")) + "\n")
        f.write(json.dumps(_g("2026-06-02T13:00:00+00:00")) + "\n")
    ledger = token_ledger.roll_up(tmp_path)
    assert _historic_grounded(ledger) == 3


def test_roll_up_does_not_regress_higher_stored_count(tmp_path: Path) -> None:
    """A day already stored with a HIGHER grounded count must survive a roll_up
    that observes fewer rows for it (monotonic max-merge contract)."""
    import json

    token_ledger.write_ledger(
        tmp_path,
        {"schema": token_ledger.LEDGER_SCHEMA, "days": {"2026-06-09": {"grounded": 5}}},
    )
    log = dashboard.grounding_log_path(tmp_path)
    log.write_text(
        "\n".join(json.dumps(_g(f"2026-06-09T{h}:00:00+00:00")) for h in (12, 13)) + "\n",
        encoding="utf-8",
    )
    ledger = token_ledger.roll_up(tmp_path)
    assert ledger["days"]["2026-06-09"]["grounded"] == 5


# --- concurrency (flock serialization + atomic unique-tmp writes) -----------


def test_roll_up_serializes_under_flock(tmp_path: Path) -> None:
    """A concurrent roll_up must wait behind the holder's flock instead of
    interleaving its read-merge-write (last-writer-wins lost-update)."""
    import fcntl
    import json
    import threading

    log = dashboard.grounding_log_path(tmp_path)
    log.write_text(json.dumps(_g("2026-06-09T12:00:00+00:00")) + "\n", encoding="utf-8")

    done = threading.Event()
    result: dict = {}

    def run() -> None:
        result["ledger"] = token_ledger.roll_up(tmp_path)
        done.set()

    with token_ledger._ledger_lock_path(tmp_path).open("a+", encoding="utf-8") as holder:
        fcntl.flock(holder.fileno(), fcntl.LOCK_EX)
        t = threading.Thread(target=run, daemon=True)
        t.start()
        assert not done.wait(timeout=0.3)  # blocked behind the holder's lock
        fcntl.flock(holder.fileno(), fcntl.LOCK_UN)
    assert done.wait(timeout=2.0)  # proceeds once the lock is released
    t.join(timeout=2.0)
    assert _historic_grounded(result["ledger"]) == 1


def test_write_ledger_uses_unique_tmp_names(tmp_path: Path, monkeypatch) -> None:
    """Each writer gets its own tmp file — a fixed tmp name would let one
    concurrent writer publish another's half-written tmp via os.replace."""
    import os

    seen: list[str] = []
    real_replace = os.replace

    def recording_replace(src, dst):
        seen.append(str(src))
        real_replace(src, dst)

    monkeypatch.setattr(token_ledger.os, "replace", recording_replace)
    _seed_ledger(tmp_path, {"2026-06-01": 1})
    _seed_ledger(tmp_path, {"2026-06-02": 2})
    assert len(seen) == 2
    assert seen[0] != seen[1]


def test_write_ledger_failed_replace_cleans_tmp_and_keeps_ledger(
    tmp_path: Path, monkeypatch
) -> None:
    import pytest

    _seed_ledger(tmp_path, {"2026-06-01": 2})

    def boom(src, dst):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(token_ledger.os, "replace", boom)
    with pytest.raises(OSError):
        _seed_ledger(tmp_path, {"2026-06-02": 9})
    monkeypatch.undo()
    # No leftover tmp lingers to clobber a concurrent writer...
    assert list(tmp_path.glob("*.tmp")) == []
    # ...and the previously-persisted ledger is intact.
    assert token_ledger.read_ledger(tmp_path)["days"] == {"2026-06-01": {"grounded": 2}}


# --- summarize (day / month / historic + chart series) ---------------------


def _seed_ledger(tmp_path: Path, days: dict[str, int]) -> None:
    token_ledger.write_ledger(
        tmp_path,
        {
            "schema": token_ledger.LEDGER_SCHEMA,
            "days": {d: {"grounded": n} for d, n in days.items()},
        },
    )


def test_summarize_buckets_today_month_historic(tmp_path: Path) -> None:
    _seed_ledger(
        tmp_path,
        {
            "2026-06-30": 4,  # today
            "2026-06-15": 6,  # this month, earlier
            "2026-05-20": 10,  # last month
            "2026-04-02": 3,  # older
        },
    )
    s = token_ledger.summarize(tmp_path, today=date(2026, 6, 30))
    tpg = s["tpg"]
    assert s["today"]["grounded"] == 4
    assert s["today"]["tokens"] == 4 * tpg
    assert s["month"]["grounded"] == 10  # 4 + 6
    assert s["month"]["tokens"] == 10 * tpg
    assert s["historic"]["grounded"] == 23  # 4 + 6 + 10 + 3
    assert s["historic"]["tokens"] == 23 * tpg


def test_summarize_growth_compares_months(tmp_path: Path) -> None:
    _seed_ledger(tmp_path, {"2026-06-10": 20, "2026-05-10": 10})
    s = token_ledger.summarize(tmp_path, today=date(2026, 6, 30))
    g = s["growth"]
    assert g["this_month_tokens"] == 20 * s["tpg"]
    assert g["prev_month_tokens"] == 10 * s["tpg"]
    assert g["up"] is True
    assert g["pct"] == 100.0  # doubled


def test_summarize_daily_series_is_continuous_and_ends_today(tmp_path: Path) -> None:
    _seed_ledger(tmp_path, {"2026-06-28": 5, "2026-06-30": 2})
    s = token_ledger.summarize(tmp_path, today=date(2026, 6, 30), days_back=7)
    daily = s["daily"]
    assert len(daily) == 7
    assert daily[-1]["date"] == "2026-06-30"
    assert daily[-1]["grounded"] == 2
    # A gap day with no grounded events is filled with zero (continuous chart).
    gap = {d["date"]: d["grounded"] for d in daily}
    assert gap["2026-06-29"] == 0
    assert gap["2026-06-28"] == 5


def test_summarize_empty_ledger(tmp_path: Path) -> None:
    s = token_ledger.summarize(tmp_path, today=date(2026, 6, 30))
    assert s["today"]["tokens"] == 0
    assert s["historic"]["tokens"] == 0
    assert s["growth"]["up"] is None
    assert all(d["grounded"] == 0 for d in s["daily"])


# --- CLI wiring (`memo tokens`) --------------------------------------------


def _cli_env(tmp_path: Path) -> dict:
    return {
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_DATA_DIR": str(tmp_path / "d"),
        "MEMO_STATE_DIR": str(tmp_path / "s"),
    }


def _seed_grounding(state_dir: Path, n: int) -> None:
    import json

    state_dir.mkdir(parents=True, exist_ok=True)
    log = dashboard.grounding_log_path(state_dir)
    log.write_text(
        "\n".join(json.dumps(_g(f"2026-06-09T{10 + i}:00:00+00:00")) for i in range(n)) + "\n",
        encoding="utf-8",
    )


def test_tokens_cmd_json_rolls_up_and_reports(tmp_path: Path) -> None:
    from click.testing import CliRunner

    from memo.cli import cli

    state = tmp_path / "s"
    _seed_grounding(state, 3)
    r = CliRunner().invoke(cli, ["tokens", "--json"], env=_cli_env(tmp_path))
    assert r.exit_code == 0, r.output
    import json

    s = json.loads(r.output)
    assert s["historic"]["grounded"] == 3
    assert s["historic"]["tokens"] == 3 * s["tpg"]
    # Durable ledger was written by the roll_up inside the command.
    assert token_ledger.ledger_path(state).is_file()


def test_tokens_cmd_empty_is_graceful(tmp_path: Path) -> None:
    from click.testing import CliRunner

    from memo.cli import cli

    r = CliRunner().invoke(cli, ["tokens"], env=_cli_env(tmp_path))
    assert r.exit_code == 0, r.output
    assert "Todavía no hay" in r.output


def test_tokens_cmd_renders_numbers_and_bars(tmp_path: Path) -> None:
    from click.testing import CliRunner

    from memo.cli import cli

    _seed_grounding(tmp_path / "s", 5)
    r = CliRunner().invoke(cli, ["tokens"], env=_cli_env(tmp_path))
    assert r.exit_code == 0, r.output
    assert "HISTÓRICO" in r.output
    assert "tokens ahorrados" in r.output
