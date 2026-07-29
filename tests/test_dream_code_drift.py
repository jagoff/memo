"""Dream code-drift pass — re-verify memories with code_refs against codegraph.

Covers:
- the flag spec (bool, default OFF) + its graduation gate in dream_flags.GATES;
- flag off → the pass is a complete no-op (not even the DB guard runs);
- live ref → no drift; dead ref → outdated candidate via feedback_flag
  (reversible archive, dry-run only reports);
- partially drifted memories are reported, never archived;
- missing or stale (>24h) codegraph.db aborts the pass without marking
  anything (a stale index would read as mass false drift);
- refs minted against ANOTHER repo (codegraph://<repo_id>/…) are unverifiable
  against this DB — never dead (other indexed repos are not drift);
- `memo dream run` wiring: flag on invokes the pass and lands the result in
  the receipt; flag off leaves the {"status": "disabled"} default untouched.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

from memo.cli_dream import dream_cmd
from memo.cli_dream_passes import _run_code_drift
from memo.code_traceability import codegraph_repo_id
from memo.dream_flags import CODE_DRIFT_FLAG, GATES
from memo.flags import REGISTRY

# --- synthetic codegraph.db (NEVER the real index — see test_codegraph_loader) --

_NODES = [
    # (id, kind, name, qualified_name, file_path, start_line, end_line)
    ("function:save", "function", "save", "memo.store.save", "src/memo/store.py", 10, 42),
    ("file:src/memo/store.py", "file", "store.py", None, "src/memo/store.py", None, None),
]


def _seed_graph(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE nodes (
            id TEXT PRIMARY KEY, kind TEXT, name TEXT, qualified_name TEXT,
            file_path TEXT, start_line INTEGER, end_line INTEGER
        );
        CREATE TABLE edges (source TEXT, target TEXT, kind TEXT);
        """
    )
    conn.executemany("INSERT INTO nodes VALUES (?, ?, ?, ?, ?, ?, ?)", _NODES)
    conn.commit()
    conn.close()


@pytest.fixture
def graph_db(tmp_path: Path) -> Path:
    db = tmp_path / ".codegraph" / "codegraph.db"
    db.parent.mkdir(parents=True)
    _seed_graph(db)
    return db


def _ref(
    file_path: str,
    label: str = "",
    qualified: str = "",
    kind: str = "function",
    repo_id: str = "",
) -> dict:
    # repo_id defaults to "" (no repo claim → judged on file_path/symbol);
    # a non-empty repo_id must match the DB's repo or the ref is unverifiable.
    return {
        "uri": f"codegraph://{repo_id or 'testrepo'}/{label or file_path}",
        "repo_id": repo_id,
        "stable_symbol_id": label or file_path,
        "kind": kind,
        "label": label,
        "qualified_name": qualified,
        "file_path": file_path,
        "relation": "modified",
        "confidence": 0.95,
    }


LIVE_REF = _ref("src/memo/store.py", label="save", qualified="memo.store.save")
DEAD_SYMBOL_REF = _ref("src/memo/store.py", label="old_save", qualified="memo.store.old_save")
DEAD_FILE_REF = _ref("src/memo/gone.py", label="gone", qualified="memo.gone.gone")


def _save_with_refs(mock_memory, refs: list[dict], content: str):
    return mock_memory.save(content=content, type_="fact", extra={"code_refs": refs})


# --- flag spec + graduation gate (the CI contract for dark flags) ---------------


def test_flag_is_registered_dark_and_gated():
    spec = REGISTRY[CODE_DRIFT_FLAG]
    assert spec.kind == "bool"
    assert spec.default is False
    assert not spec.opt_out
    gate = GATES[CODE_DRIFT_FLAG]
    assert gate.kind == "manual"
    assert gate.reason.strip()


# --- flag off → the pass does not run --------------------------------------------


def test_flag_off_pass_is_a_noop(mock_memory, tmp_path, monkeypatch):
    monkeypatch.delenv(CODE_DRIFT_FLAG, raising=False)
    rec = _save_with_refs(mock_memory, [DEAD_FILE_REF], "notes about a deleted module")

    # db_path points nowhere: disabled must short-circuit BEFORE the DB guard.
    res = _run_code_drift(mock_memory, db_path=tmp_path / "missing.db")

    assert res == {"status": "disabled"}
    assert mock_memory.get(rec.id) is not None


# --- live vs dead refs ------------------------------------------------------------


def test_live_ref_is_not_drift(mock_memory, graph_db, monkeypatch):
    monkeypatch.setenv(CODE_DRIFT_FLAG, "1")
    rec = _save_with_refs(mock_memory, [LIVE_REF], "save() writes md first, then indexes")

    res = _run_code_drift(mock_memory, db_path=graph_db)

    assert res["status"] == "ok"
    assert res["scanned"] == 1
    assert res["outdated"] == []
    assert res["partial"] == []
    assert mock_memory.get(rec.id) is not None


def test_file_kind_ref_checks_file_path_only(mock_memory, graph_db, monkeypatch):
    monkeypatch.setenv(CODE_DRIFT_FLAG, "1")
    # label deliberately mismatches the node name: file refs carry no symbol,
    # so existence is judged on file_path alone.
    ref = _ref("src/memo/store.py", label="renamed.py", kind="file")
    rec = _save_with_refs(mock_memory, [ref], "notes about the store module")

    res = _run_code_drift(mock_memory, db_path=graph_db)

    assert res["outdated"] == []
    assert mock_memory.get(rec.id) is not None


def test_all_refs_dead_marks_memory_outdated(mock_memory, graph_db, monkeypatch):
    monkeypatch.setenv(CODE_DRIFT_FLAG, "1")
    rec = _save_with_refs(
        mock_memory, [DEAD_SYMBOL_REF, DEAD_FILE_REF], "old_save() handles retries"
    )

    res = _run_code_drift(mock_memory, db_path=graph_db)

    assert res["status"] == "ok"
    assert [e["id"] for e in res["outdated"]] == [rec.id]
    assert res["outdated"][0]["refs_dead"] == 2
    assert res["outdated"][0]["refs_total"] == 2
    # Reversible archive via feedback_flag — the record leaves the active index.
    assert mock_memory.get(rec.id) is None


def test_dry_run_reports_candidate_without_archiving(mock_memory, graph_db, monkeypatch):
    monkeypatch.setenv(CODE_DRIFT_FLAG, "1")
    rec = _save_with_refs(mock_memory, [DEAD_FILE_REF], "notes about a deleted module")

    res = _run_code_drift(mock_memory, db_path=graph_db, dry_run=True)

    assert [e["id"] for e in res["outdated"]] == [rec.id]
    assert mock_memory.get(rec.id) is not None


def test_partial_drift_is_reported_not_archived(mock_memory, graph_db, monkeypatch):
    monkeypatch.setenv(CODE_DRIFT_FLAG, "1")
    rec = _save_with_refs(mock_memory, [LIVE_REF, DEAD_FILE_REF], "save() + a deleted helper")

    res = _run_code_drift(mock_memory, db_path=graph_db)

    assert res["outdated"] == []
    assert [e["id"] for e in res["partial"]] == [rec.id]
    assert res["partial"][0]["refs_dead"] == 1
    assert res["partial"][0]["refs_total"] == 2
    assert mock_memory.get(rec.id) is not None


def test_unverifiable_refs_never_count_as_drift(mock_memory, graph_db, monkeypatch):
    monkeypatch.setenv(CODE_DRIFT_FLAG, "1")
    # No file_path → nothing to verify; the memory must not be scanned or flagged.
    rec = _save_with_refs(mock_memory, [_ref("", label="mystery")], "ref without a file path")

    res = _run_code_drift(mock_memory, db_path=graph_db)

    assert res["scanned"] == 0
    assert res["outdated"] == []
    assert mock_memory.get(rec.id) is not None


# --- repo scoping: refs from another repo are unverifiable, never dead ------------


def test_foreign_repo_ref_is_unverifiable_never_dead(mock_memory, graph_db, monkeypatch):
    monkeypatch.setenv(CODE_DRIFT_FLAG, "1")
    # The file_path does not exist in THIS DB — but the ref was minted against
    # another indexed repo, so it must be unverifiable (None), never dead.
    ref = _ref(
        "src/synapse/router.py",
        label="route",
        qualified="synapse.router.route",
        repo_id="feedfacefeedface",
    )
    rec = _save_with_refs(mock_memory, [ref], "routing decision in another repo")

    res = _run_code_drift(mock_memory, db_path=graph_db)

    assert res["status"] == "ok"
    assert res["scanned"] == 0
    assert res["outdated"] == []
    assert res["partial"] == []
    assert mock_memory.get(rec.id) is not None


def test_dead_local_ref_plus_foreign_ref_is_partial_not_archived(
    mock_memory, graph_db, monkeypatch
):
    monkeypatch.setenv(CODE_DRIFT_FLAG, "1")
    # The local ref is dead, but the foreign ref may be fully vigente in ITS
    # repo — it was never checked against any DB. Archiving on the verifiable
    # subset alone would archive on partial evidence: report as partial only.
    foreign = _ref(
        "src/synapse/router.py",
        label="route",
        qualified="synapse.router.route",
        repo_id="feedfacefeedface",
    )
    rec = _save_with_refs(mock_memory, [DEAD_FILE_REF, foreign], "dead here, alive elsewhere")

    res = _run_code_drift(mock_memory, db_path=graph_db)

    assert res["outdated"] == []
    assert [e["id"] for e in res["partial"]] == [rec.id]
    assert mock_memory.get(rec.id) is not None


def test_matching_repo_ref_is_still_verified(mock_memory, graph_db, monkeypatch):
    monkeypatch.setenv(CODE_DRIFT_FLAG, "1")
    # graph_db lives at <repo_root>/.codegraph/codegraph.db — a ref carrying
    # THAT repo's id is verifiable, so a dead path still counts as drift.
    ref = _ref(
        "src/memo/gone.py",
        label="gone",
        qualified="memo.gone.gone",
        repo_id=codegraph_repo_id(graph_db.parent.parent),
    )
    rec = _save_with_refs(mock_memory, [ref], "notes about a deleted module")

    res = _run_code_drift(mock_memory, db_path=graph_db, dry_run=True)

    assert res["scanned"] == 1
    assert [e["id"] for e in res["outdated"]] == [rec.id]


# --- nightly hub gaps: computed here, read by the briefing from the receipt -------


def test_hub_gaps_land_in_receipt_for_the_briefing(mock_memory, graph_db, monkeypatch):
    monkeypatch.setenv(CODE_DRIFT_FLAG, "1")
    monkeypatch.delenv("MEMO_GAPS_CODE_HUBS", raising=False)
    conn = sqlite3.connect(graph_db)
    conn.execute(
        "INSERT INTO nodes VALUES "
        "('function:caller', 'function', 'caller', 'memo.cli.caller', 'src/memo/cli.py', 1, 5)"
    )
    conn.execute("INSERT INTO edges VALUES ('function:caller', 'function:save', 'calls')")
    conn.commit()
    conn.close()

    res = _run_code_drift(mock_memory, db_path=graph_db)

    assert res["hub_gaps"] == ["hub sin memoria: save (1 callers) — src/memo/store.py"]


def test_hub_gaps_absent_from_receipt_when_flag_off(mock_memory, graph_db, monkeypatch):
    monkeypatch.setenv(CODE_DRIFT_FLAG, "1")
    monkeypatch.setenv("MEMO_GAPS_CODE_HUBS", "0")

    res = _run_code_drift(mock_memory, db_path=graph_db)

    assert "hub_gaps" not in res


# --- guard: missing or stale index aborts without marking anything ----------------


def test_missing_db_aborts_without_marking(mock_memory, tmp_path, monkeypatch):
    monkeypatch.setenv(CODE_DRIFT_FLAG, "1")
    rec = _save_with_refs(mock_memory, [DEAD_FILE_REF], "notes about a deleted module")

    res = _run_code_drift(mock_memory, db_path=tmp_path / "missing.db")

    assert res["status"] == "aborted"
    assert res["reason"] == "codegraph_db_missing"
    assert mock_memory.get(rec.id) is not None


def test_default_db_resolution_honors_override(mock_memory, graph_db, tmp_path, monkeypatch):
    """db_path=None resolves like the recall render (_resolve_db):
    MEMO_CODEGRAPH_DB rescues a dead module default — the pipx install and the
    nightly daemon whose cwd is $HOME, where discovery finds nothing."""
    from memo import codegraph_loader

    monkeypatch.setenv(CODE_DRIFT_FLAG, "1")
    monkeypatch.setenv("MEMO_CODEGRAPH_DISCOVERY", "0")
    monkeypatch.setattr(codegraph_loader, "CODEGRAPH_DB", tmp_path / "missing.db")
    monkeypatch.setenv("MEMO_CODEGRAPH_DB", str(graph_db))
    _save_with_refs(mock_memory, [LIVE_REF], "memory citing a live symbol")

    res = _run_code_drift(mock_memory)

    assert res["status"] == "ok"
    assert res["scanned"] == 1
    assert res["outdated"] == []


def test_stale_db_aborts_without_marking(mock_memory, graph_db, monkeypatch):
    monkeypatch.setenv(CODE_DRIFT_FLAG, "1")
    rec = _save_with_refs(mock_memory, [DEAD_FILE_REF], "notes about a deleted module")
    old = time.time() - 25 * 3600
    os.utime(graph_db, (old, old))

    res = _run_code_drift(mock_memory, db_path=graph_db)

    assert res["status"] == "aborted"
    assert res["reason"] == "codegraph_db_stale"
    assert res["age_hours"] > 24
    assert mock_memory.get(rec.id) is not None


# --- `memo dream run` wiring: flag on invokes the pass, flag off does not ---------

_RUN_SKIPS = [
    "--skip-maintain",
    "--skip-orientation",
    "--skip-signal-gather",
    "--skip-entities",
    "--skip-decay",
    "--skip-prune-floor",
    "--skip-evict",
    "--skip-compress",
    "--skip-prewarm",
    "--skip-presynthesis",
]


def _dream_run_receipt(tmp_cfg, mock_memory, monkeypatch) -> dict:
    monkeypatch.setenv("MEMO_DATA_DIR", str(tmp_cfg.data_dir))
    monkeypatch.setenv("MEMO_STATE_DIR", str(tmp_cfg.state_dir))
    monkeypatch.setenv("MEMO_NONINTERACTIVE", "1")
    monkeypatch.setenv("MEMO_GRAPH_PROJECTION_ENABLED", "0")
    monkeypatch.setenv("MEMO_OUTCOME_RANKING_ENABLED", "0")
    monkeypatch.setenv("MEMO_DREAM_EVAL_ENABLED", "0")
    monkeypatch.setenv("MEMO_DYNAMIC_MANDATE_SYNC_ENABLED", "0")
    monkeypatch.setattr("memo.cli_dream._get_memory", lambda _cfg: mock_memory)

    result = CliRunner().invoke(dream_cmd, ["run", "--dry-run", "--json", *_RUN_SKIPS])

    assert result.exit_code == 0, result.output
    return json.loads(result.output[result.output.index("{") :])


def test_dream_run_invokes_code_drift_when_flag_on(tmp_cfg, mock_memory, monkeypatch):
    monkeypatch.setenv(CODE_DRIFT_FLAG, "1")
    run_drift = MagicMock(
        return_value={
            "status": "ok",
            "scanned": 2,
            "outdated": [],
            "partial": [],
            "error": "RuntimeError: boom",
        }
    )
    monkeypatch.setattr("memo.cli_dream._run_code_drift", run_drift)

    receipt = _dream_run_receipt(tmp_cfg, mock_memory, monkeypatch)

    run_drift.assert_called_once_with(mock_memory, dry_run=True)
    assert receipt["code_drift"]["status"] == "ok"
    assert receipt["code_drift"]["scanned"] == 2
    assert "code_drift: RuntimeError: boom" in receipt["errors"]


def test_dream_run_skips_code_drift_when_flag_off(tmp_cfg, mock_memory, monkeypatch):
    monkeypatch.delenv(CODE_DRIFT_FLAG, raising=False)
    run_drift = MagicMock()
    monkeypatch.setattr("memo.cli_dream._run_code_drift", run_drift)

    receipt = _dream_run_receipt(tmp_cfg, mock_memory, monkeypatch)

    run_drift.assert_not_called()
    assert receipt["code_drift"] == {"status": "disabled"}
