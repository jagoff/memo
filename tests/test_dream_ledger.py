"""Tests for the append-only dream learning ledger.

``memo.dream_ledger`` is pure JSONL I/O over ``state_dir/dream/`` — MLX-free,
flag-free, and best-effort (never raises into the pipeline). Every test here is
hermetic: it only touches ``tmp_path``. The module reads no ``MEMO_*`` flags and
needs no ``Config``, so no env/flag setup is required.
"""

from __future__ import annotations

import json
import string
from pathlib import Path

from memo import dream_ledger as mod


def _is_hex_id(value: object) -> bool:
    return isinstance(value, str) and len(value) == 32 and all(c in string.hexdigits for c in value)


# --- record_action -------------------------------------------------------


def test_record_action_returns_hex_id_and_writes_entry(tmp_path: Path) -> None:
    action_id = mod.record_action(tmp_path, action="supersede", pass_name="contradict")

    assert _is_hex_id(action_id)
    entries = mod._read_all(tmp_path)
    assert len(entries) == 1
    entry = entries[0]
    assert entry["entry_id"] == action_id
    assert entry["kind"] == "action"
    assert entry["action"] == "supersede"
    assert entry["pass_name"] == "contradict"
    assert entry["dry_run"] is False


def test_record_action_strips_action_and_defaults_collections(tmp_path: Path) -> None:
    action_id = mod.record_action(tmp_path, action="  merge  ")
    entry = mod.get_action(tmp_path, action_id)  # type: ignore[arg-type]

    assert entry is not None
    assert entry["action"] == "merge"
    assert entry["candidate_ids"] == []
    assert entry["affected_ids"] == []
    assert entry["evidence"] == {}


def test_record_action_none_on_dry_run_and_writes_nothing(tmp_path: Path) -> None:
    result = mod.record_action(tmp_path, action="archive_stale", dry_run=True)

    assert result is None
    assert mod._read_all(tmp_path) == []


def test_record_action_none_on_empty_action(tmp_path: Path) -> None:
    assert mod.record_action(tmp_path, action="") is None
    assert mod.record_action(tmp_path, action="   ") is None
    assert mod._read_all(tmp_path) == []


def test_record_action_none_on_non_string_action(tmp_path: Path) -> None:
    assert mod.record_action(tmp_path, action=123) is None  # type: ignore[arg-type]
    assert mod.record_action(tmp_path, action=None) is None  # type: ignore[arg-type]


def test_record_action_coerces_confidence(tmp_path: Path) -> None:
    aid_ok = mod.record_action(tmp_path, action="tune", confidence=0.9)
    aid_bool = mod.record_action(tmp_path, action="tune", confidence=True)  # type: ignore[arg-type]

    a_ok = mod.get_action(tmp_path, aid_ok)  # type: ignore[arg-type]
    a_bool = mod.get_action(tmp_path, aid_bool)  # type: ignore[arg-type]
    assert a_ok is not None and a_ok["confidence"] == 0.9
    # bool is an int subclass but is not a valid confidence -> None
    assert a_bool is not None and a_bool["confidence"] is None


# --- best-effort append (never raises) -----------------------------------


def test_record_action_best_effort_none_on_unserializable_payload(tmp_path: Path) -> None:
    # A non-JSON-serializable source_signal makes json.dumps raise TypeError,
    # which _append must swallow -> record_action returns None, no exception.
    result = mod.record_action(tmp_path, action="synthesize", source_signal=object())

    assert result is None
    assert mod._read_all(tmp_path) == []


def test_append_returns_false_and_never_raises_on_io_error(tmp_path: Path) -> None:
    # Collide the ledger file path with a directory so open("a") raises
    # IsADirectoryError (an OSError). _append must catch it and return False.
    path = mod.ledger_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.mkdir()

    assert mod._append(tmp_path, {"kind": "action"}) is False


# --- resolve_action ------------------------------------------------------


def test_resolve_action_chains_outcome_to_known_action(tmp_path: Path) -> None:
    action_id = mod.record_action(tmp_path, action="tune")
    outcome_id = mod.resolve_action(
        tmp_path, action_id, outcome="reinforced", verdict="keep", delta=0.05
    )  # type: ignore[arg-type]

    assert _is_hex_id(outcome_id)
    entries = mod._read_all(tmp_path)
    assert len(entries) == 2
    outcome_entry = entries[1]
    assert outcome_entry["kind"] == "outcome"
    assert outcome_entry["action_id"] == action_id
    assert outcome_entry["outcome"] == "reinforced"
    assert outcome_entry["verdict"] == "keep"
    assert outcome_entry["delta"] == 0.05


def test_resolve_action_none_on_unknown_action_id(tmp_path: Path) -> None:
    # No matching action in the ledger -> nothing written.
    result = mod.resolve_action(tmp_path, "deadbeef" * 4, outcome="reverted")

    assert result is None
    assert mod._read_all(tmp_path) == []


def test_resolve_action_none_on_invalid_input(tmp_path: Path) -> None:
    action_id = mod.record_action(tmp_path, action="prune")

    assert mod.resolve_action(tmp_path, "", outcome="ok") is None
    assert mod.resolve_action(tmp_path, action_id, outcome="") is None  # type: ignore[arg-type]
    assert mod.resolve_action(tmp_path, action_id, outcome="   ") is None  # type: ignore[arg-type]
    # only the action was ever written
    assert len(mod._read_all(tmp_path)) == 1


# --- open_actions --------------------------------------------------------


def test_open_actions_excludes_resolved(tmp_path: Path) -> None:
    a1 = mod.record_action(tmp_path, action="supersede")
    a2 = mod.record_action(tmp_path, action="merge")
    mod.resolve_action(tmp_path, a1, outcome="confirmed")  # type: ignore[arg-type]

    open_ids = [e["entry_id"] for e in mod.open_actions(tmp_path)]
    assert open_ids == [a2]  # oldest-first, only the unresolved action


def test_open_actions_empty_when_all_resolved(tmp_path: Path) -> None:
    a1 = mod.record_action(tmp_path, action="evict")
    mod.resolve_action(tmp_path, a1, outcome="neutral")  # type: ignore[arg-type]

    assert mod.open_actions(tmp_path) == []


# --- get_action ----------------------------------------------------------


def test_get_action_folds_latest_outcome(tmp_path: Path) -> None:
    action_id = mod.record_action(tmp_path, action="graduate")
    mod.resolve_action(tmp_path, action_id, outcome="neutral")  # type: ignore[arg-type]
    mod.resolve_action(tmp_path, action_id, outcome="rollback_candidate")  # type: ignore[arg-type]

    folded = mod.get_action(tmp_path, action_id)  # type: ignore[arg-type]
    assert folded is not None
    assert folded["action"] == "graduate"
    assert folded["outcome"] is not None
    # latest outcome in file order wins
    assert folded["outcome"]["outcome"] == "rollback_candidate"


def test_get_action_open_action_has_none_outcome(tmp_path: Path) -> None:
    action_id = mod.record_action(tmp_path, action="compress")

    folded = mod.get_action(tmp_path, action_id)  # type: ignore[arg-type]
    assert folded is not None
    assert folded["outcome"] is None


def test_get_action_unknown_returns_none(tmp_path: Path) -> None:
    mod.record_action(tmp_path, action="prune")
    assert mod.get_action(tmp_path, "cafe" * 8) is None


# --- summarize -----------------------------------------------------------


def test_summarize_counts_open_and_rollback_candidates(tmp_path: Path) -> None:
    a1 = mod.record_action(tmp_path, action="supersede")
    a2 = mod.record_action(tmp_path, action="supersede")
    mod.record_action(tmp_path, action="merge")  # a3, left open
    mod.record_action(tmp_path, action="archive_stale")  # a4, left open

    mod.resolve_action(tmp_path, a1, outcome="reinforced")  # type: ignore[arg-type]
    mod.resolve_action(tmp_path, a2, outcome="rollback_candidate")  # type: ignore[arg-type]

    s = mod.summarize(tmp_path)
    assert s["actions"] == 4
    assert s["outcomes"] == 2
    assert s["open"] == 2
    assert s["rollback_candidates"] == 1
    assert s["by_action"] == {"supersede": 2, "merge": 1, "archive_stale": 1}


def test_summarize_rollback_detected_via_verdict(tmp_path: Path) -> None:
    a1 = mod.record_action(tmp_path, action="tune")
    # "rollback" lives in the verdict, not the outcome, and must still count.
    mod.resolve_action(tmp_path, a1, outcome="reverted", verdict="rollback")  # type: ignore[arg-type]

    assert mod.summarize(tmp_path)["rollback_candidates"] == 1


def test_summarize_empty_ledger(tmp_path: Path) -> None:
    s = mod.summarize(tmp_path)
    assert s == {
        "actions": 0,
        "outcomes": 0,
        "open": 0,
        "rollback_candidates": 0,
        "by_action": {},
    }


# --- _read_all (robust parsing) ------------------------------------------


def test_read_all_missing_file_returns_empty(tmp_path: Path) -> None:
    assert mod._read_all(tmp_path) == []


def test_read_all_skips_blank_and_corrupt_and_non_dict_lines(tmp_path: Path) -> None:
    path = mod.ledger_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                json.dumps({"entry_id": "a", "kind": "action", "action": "x"}),
                "",  # blank
                "   ",  # whitespace only
                "not valid json at all",  # corrupt
                json.dumps([1, 2, 3]),  # valid JSON, non-dict
                json.dumps("a bare string"),  # valid JSON, non-dict
                "123",  # valid JSON int, non-dict
                json.dumps({"entry_id": "b", "kind": "outcome", "action_id": "a"}),
            ]
        ),
        encoding="utf-8",
    )

    entries = mod._read_all(tmp_path)
    assert len(entries) == 2
    assert [e["entry_id"] for e in entries] == ["a", "b"]
    assert all(isinstance(e, dict) for e in entries)


# --- read_ledger / recent_actions (thin wrappers) ------------------------


def test_read_ledger_returns_last_n_oldest_first(tmp_path: Path) -> None:
    ids = [mod.record_action(tmp_path, action=f"a{i}") for i in range(5)]

    last_two = mod.read_ledger(tmp_path, limit=2)
    assert [e["entry_id"] for e in last_two] == ids[-2:]


def test_recent_actions_newest_first_with_filters(tmp_path: Path) -> None:
    mod.record_action(tmp_path, action="merge", pass_name="consolidate")
    a2 = mod.record_action(tmp_path, action="supersede", pass_name="contradict")
    a3 = mod.record_action(tmp_path, action="supersede", pass_name="contradict")

    by_action = mod.recent_actions(tmp_path, action="supersede")
    assert [e["entry_id"] for e in by_action] == [a3, a2]  # newest-first

    by_phase = mod.recent_actions(tmp_path, phase="consolidate")
    assert [e["action"] for e in by_phase] == ["merge"]


# --- record_from_receipt (Fase 3 receipt mapping) ------------------------


def _receipt_with_mutations() -> dict:
    return {
        "contradict": {
            "superseded": [{"pair_id": 11, "older": "aaaa1111"}],
            "evolved": [22],
        },
        "consolidate_dups": {
            "merged": [{"merged_id": "mmmm0000", "archived_ids": ["bbbb2222", "cccc3333"]}]
        },
        "stale": {"archived": [{"id": "dddd4444", "days": 400}]},
    }


def test_record_from_receipt_maps_each_mutation_to_an_action(tmp_path: Path) -> None:
    counts = mod.record_from_receipt(tmp_path, _receipt_with_mutations())
    assert counts == {"supersede": 1, "evolve": 1, "merge": 1, "archive_stale": 1}

    actions = [e for e in mod.read_ledger(tmp_path, limit=100) if e.get("kind") == "action"]
    by_action = {a["action"] for a in actions}
    assert by_action == {"supersede", "evolve", "merge", "archive_stale"}

    supersede = next(a for a in actions if a["action"] == "supersede")
    assert supersede["affected_ids"] == ["aaaa1111"]
    assert supersede["reversal"] == {"type": "inactive_md", "handle": "inactive/aaaa1111.md"}
    assert supersede["confidence"] == 0.9

    merge = next(a for a in actions if a["action"] == "merge")
    assert merge["affected_ids"] == ["bbbb2222", "cccc3333"]


def test_record_from_receipt_dry_run_writes_nothing(tmp_path: Path) -> None:
    counts = mod.record_from_receipt(tmp_path, _receipt_with_mutations(), dry_run=True)
    assert counts == {}
    assert mod.read_ledger(tmp_path, limit=100) == []


def test_record_from_receipt_skips_empty_and_idless_fragments(tmp_path: Path) -> None:
    receipt = {"contradict": {"superseded": [{"pair_id": 1, "older": ""}]}, "stale": {}}
    assert mod.record_from_receipt(tmp_path, receipt) == {}


# --- resolve_open_actions (close-the-loop) -------------------------------


def test_resolve_open_actions_reinforces_still_archived(tmp_path: Path) -> None:
    from datetime import UTC, datetime, timedelta

    old = datetime(2020, 1, 1, tzinfo=UTC)
    mod.record_action(tmp_path, action="archive_stale", affected_ids=["gone9999"], now=old)
    # Nothing came back to life -> reinforced.
    out = mod.resolve_open_actions(tmp_path, lambda _mid: False, now=old + timedelta(days=1))
    assert out["reinforced"] == 1
    assert out["rollback_candidate"] == 0
    assert mod.open_actions(tmp_path) == []  # now resolved


def test_resolve_open_actions_flags_resurrected_as_rollback(tmp_path: Path) -> None:
    from datetime import UTC, datetime, timedelta

    old = datetime(2020, 1, 1, tzinfo=UTC)
    aid = mod.record_action(tmp_path, action="supersede", affected_ids=["back0001"], now=old)
    out = mod.resolve_open_actions(
        tmp_path, lambda mid: mid == "back0001", now=old + timedelta(days=1)
    )
    assert out["rollback_candidate"] == 1
    folded = mod.get_action(tmp_path, str(aid))
    assert folded["outcome"]["outcome"] == "rollback_candidate"
    assert folded["outcome"]["verdict"] == "reopened"


def test_resolve_open_actions_skips_actions_younger_than_min_age(tmp_path: Path) -> None:
    from datetime import UTC, datetime

    now = datetime(2020, 1, 1, 12, 0, tzinfo=UTC)
    mod.record_action(tmp_path, action="archive_stale", affected_ids=["x"], now=now)
    # Same night: too young to judge.
    out = mod.resolve_open_actions(tmp_path, lambda _m: False, now=now, min_age_hours=20.0)
    assert out == {"reinforced": 0, "rollback_candidate": 0, "skipped": 1}
    assert len(mod.open_actions(tmp_path)) == 1
