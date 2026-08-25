"""`memo maintain` — corpus-freshness orchestrator.

Unit-covers the older-side picker and the daily `--if-due` guard, plus a
dry-run on an empty isolated corpus (no MLX needed — nothing to embed).
Includes tests for the proactive synthesis pass (MEMO_MAINT_SYNTHESIZE=1).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from memo.cli import cli
from memo.cli_maintain import (
    _older_id,
    _read_synthesis_last_run,
    _synthesis_state_path,
    _write_synthesis_last_run,
)
from memo.config import Config


def _env(tmp_path: Path) -> dict[str, str]:
    return {
        "MEMO_CONFIG_FILE": str(tmp_path / "memo-config.toml"),
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_DATA_DIR": str(tmp_path / "data"),
        "MEMO_STATE_DIR": str(tmp_path / "state"),
    }


def test_older_id_picks_earlier_updated():
    recs = {
        "a": SimpleNamespace(updated="2026-01-01T00:00:00+00:00"),
        "b": SimpleNamespace(updated="2026-05-01T00:00:00+00:00"),
    }
    mem = SimpleNamespace(get=lambda i: recs.get(i))
    assert _older_id(mem, "a", "b") == ("a", "b")
    assert _older_id(mem, "b", "a") == ("a", "b")  # order-independent


def test_older_id_falls_back_to_pair_order_when_missing():
    mem = SimpleNamespace(get=lambda i: None)
    assert _older_id(mem, "x", "y") == ("x", "y")


def test_if_due_is_noop_when_recently_run(tmp_path: Path):
    state = tmp_path / "state"
    (state / "maintain").mkdir(parents=True)
    (state / "maintain" / ".last_run_ts").write_text(str(time.time()), encoding="utf-8")

    result = CliRunner().invoke(cli, ["maintain", "--if-due"], env=_env(tmp_path))
    assert result.exit_code == 0, result.output
    # not due → no work, no output
    assert result.output.strip() == ""


def test_if_due_disabled_by_env(tmp_path: Path):
    result = CliRunner().invoke(
        cli, ["maintain", "--if-due"], env={**_env(tmp_path), "MEMO_MAINTAIN_DISABLE": "1"}
    )
    assert result.exit_code == 0, result.output
    assert result.output.strip() == ""


def test_if_due_spawns_maintain_routed_through_daemon(tmp_path: Path):
    """The detached daily maintain routes its embeds through the warm recall
    daemon (MEMO_EMBEDDER_VIA_DAEMON=1) so its GPU forward passes serialize in
    the daemon's queue instead of grabbing the cross-process flock independently
    and starving live recall (recall_lock_bail on embed_query)."""
    # conftest pins MEMO_EMBEDDER_VIA_DAEMON=0 for isolation; unset it here so the
    # spawn exercises its production default (unset → "1").
    env = {**_env(tmp_path), "MEMO_EMBEDDER_VIA_DAEMON": None}
    with patch("subprocess.Popen") as popen:
        result = CliRunner().invoke(cli, ["maintain", "--if-due"], env=env)
    assert result.exit_code == 0, result.output
    popen.assert_called_once()
    args, kwargs = popen.call_args
    assert args[0] == ["memo", "maintain", "--max-pairs", "50", "--max-scan-seconds", "300"]
    assert kwargs["start_new_session"] is True
    assert kwargs["env"]["MEMO_NONINTERACTIVE"] == "1"
    assert kwargs["env"]["MEMO_EMBEDDER_VIA_DAEMON"] == "1"


def test_if_due_respects_explicit_via_daemon_override(tmp_path: Path):
    """An explicit MEMO_EMBEDDER_VIA_DAEMON in the environment is not clobbered."""
    with patch("subprocess.Popen") as popen:
        result = CliRunner().invoke(
            cli,
            ["maintain", "--if-due"],
            env={**_env(tmp_path), "MEMO_EMBEDDER_VIA_DAEMON": "0"},
        )
    assert result.exit_code == 0, result.output
    _, kwargs = popen.call_args
    assert kwargs["env"]["MEMO_EMBEDDER_VIA_DAEMON"] == "0"


def test_dry_run_on_empty_corpus_is_safe_noop(tmp_path: Path):
    result = CliRunner().invoke(cli, ["maintain", "--dry-run", "--json"], env=_env(tmp_path))
    assert result.exit_code == 0, result.output
    import json

    receipt = json.loads(result.output)
    assert receipt["dry_run"] is True
    assert receipt["superseded"] == []
    assert receipt["merged"] == []
    assert receipt["archived_stale"] == []


def test_maintain_evicts_expired_crush_cache_entries(tmp_path: Path):
    """An expired crush-cache original is unlinked by the maintain run and the
    receipt reports the count. retrieve() only skips at read-time; maintain is
    what reclaims the disk."""
    import json as _json
    from datetime import UTC, datetime, timedelta

    state = tmp_path / "state"
    cache_dir = state / "crush_cache"
    cache_dir.mkdir(parents=True)

    expired = cache_dir / ("a" * 16 + ".json")
    old_ts = (datetime.now(UTC) - timedelta(days=60)).isoformat()
    expired.write_text(_json.dumps({"ts": old_ts, "content": "[]"}), encoding="utf-8")

    fresh = cache_dir / ("b" * 16 + ".json")
    fresh_ts = datetime.now(UTC).isoformat()
    fresh.write_text(_json.dumps({"ts": fresh_ts, "content": "[]"}), encoding="utf-8")

    result = CliRunner().invoke(cli, ["maintain", "--json"], env=_env(tmp_path))
    assert result.exit_code == 0, result.output

    receipt = json.loads(result.output)
    assert receipt["crush_cache_evicted"] == 1
    assert not expired.exists()
    assert fresh.exists()


def test_maintain_dry_run_does_not_evict_crush_cache(tmp_path: Path):
    """A --dry-run maintain must not unlink expired crush-cache entries."""
    import json as _json
    from datetime import UTC, datetime, timedelta

    cache_dir = tmp_path / "state" / "crush_cache"
    cache_dir.mkdir(parents=True)
    expired = cache_dir / ("c" * 16 + ".json")
    old_ts = (datetime.now(UTC) - timedelta(days=60)).isoformat()
    expired.write_text(_json.dumps({"ts": old_ts, "content": "[]"}), encoding="utf-8")

    result = CliRunner().invoke(cli, ["maintain", "--dry-run", "--json"], env=_env(tmp_path))
    assert result.exit_code == 0, result.output
    assert expired.exists()


# -- Memory.lint() (was untested) ------------------------------------------


def test_lint_empty_corpus_returns_empty_categories(mock_memory):
    out = mock_memory.lint()
    assert out == {
        "legacy_extra": [],
        "few_tags": [],
        "body_skinny": [],
        "untitled": [],
    }


def test_lint_flags_few_tags_and_skinny_body(mock_memory):
    rec = mock_memory.save(content="short", title="Tiny", tags=["only-one"])
    out = mock_memory.lint()
    assert rec.id in {e["id"] for e in out["few_tags"]}
    assert rec.id in {e["id"] for e in out["body_skinny"]}


def test_lint_flags_legacy_extra_fields(mock_memory):
    rec = mock_memory.save(
        content="x" * 200,
        title="Has legacy fields",
        tags=["a", "b", "c"],
        extra={"usage_count": 5, "agent_id": "old"},
    )
    legacy = {e["id"]: e for e in mock_memory.lint()["legacy_extra"]}
    assert rec.id in legacy
    assert "usage_count" in legacy[rec.id]["reason"]
    assert "agent_id" in legacy[rec.id]["reason"]


def test_lint_clean_memoria_has_no_issues(mock_memory):
    rec = mock_memory.save(
        content="x" * 200,
        title="Well formed memoria",
        tags=["project:memo", "domain:test", "technique:unit"],
    )
    out = mock_memory.lint()
    flagged = {e["id"] for cat in out.values() for e in cat}
    assert rec.id not in flagged


def _lint_tool(memory):
    """Register the core-records tools against a stub server and hand back memo_lint."""
    from memo.server_core_records import register

    tools: dict = {}

    class _Srv:
        def tool(self, *a, **k):
            def wrap(fn):
                tools[fn.__name__] = fn
                return fn

            return wrap

    register(_Srv(), memory)
    return tools["memo_lint"]


def test_memo_lint_tool_trims_categories_and_reports_true_counts(mock_memory):
    """The whole lint report is far past a client's response budget (725k chars
    on an 11k-memory corpus, few_tags alone 3,962 entries), so the tool returns
    the first `limit` per category and the real totals under `counts`."""
    for i in range(5):
        mock_memory.save(content="short", title=f"Tiny {i}", tags=["only-one"])

    out = _lint_tool(mock_memory)(limit=2)

    assert len(out["few_tags"]) == 2
    assert len(out["body_skinny"]) == 2
    assert out["counts"]["few_tags"] == 5
    assert out["counts"]["body_skinny"] == 5
    assert out["counts"]["legacy_extra"] == 0
    assert out["limit"] == 2


def test_memo_lint_tool_keeps_categories_whole_under_the_limit(mock_memory):
    rec = mock_memory.save(content="short", title="Tiny", tags=["only-one"])

    out = _lint_tool(mock_memory)()

    assert [e["id"] for e in out["few_tags"]] == [rec.id]
    assert out["counts"]["few_tags"] == 1


# -- Proactive synthesis (MEMO_MAINT_SYNTHESIZE) --------------------------------


def test_synthesis_state_helpers_round_trip(tmp_path: Path):
    """_write / _read synthesis_state.json persist and return the ISO timestamp."""
    state = tmp_path / "state"
    state.mkdir()
    cfg = Config(data_dir=tmp_path / "data", state_dir=state, reranker_enabled=False)

    assert _read_synthesis_last_run(cfg) is None  # no file yet

    ts = "2026-06-13T12:00:00+00:00"
    _write_synthesis_last_run(cfg, ts)

    assert _synthesis_state_path(cfg).is_file()
    stored = json.loads(_synthesis_state_path(cfg).read_text(encoding="utf-8"))
    assert stored["last_run"] == ts
    assert _read_synthesis_last_run(cfg) == ts


def test_maint_synthesize_flag_off_no_state_file(tmp_path: Path):
    """When MEMO_MAINT_SYNTHESIZE is not set, synthesis_state.json is never created."""
    env = _env(tmp_path)
    # Do NOT set MEMO_MAINT_SYNTHESIZE
    result = CliRunner().invoke(cli, ["maintain", "--json"], env=env)
    assert result.exit_code == 0, result.output
    state_file = tmp_path / "state" / "synthesis_state.json"
    assert not state_file.exists()


def test_maint_synthesize_flag_on_creates_state_file(tmp_path: Path):
    """With MEMO_MAINT_SYNTHESIZE=1, maintain creates synthesis_state.json and
    includes synthesis_count in the JSON receipt.

    The corpus is empty (no memorias to cluster), so synthesis_count=0,
    but the timestamp file must still be written after a real (non-dry-run) run.
    """
    env = {
        **_env(tmp_path),
        "MEMO_MAINT_SYNTHESIZE": "1",
        "MEMO_SYNTHESIS_ENABLED": "0",  # disable step-4 synthesis to avoid double-run
        "MEMO_SKIP_MODEL_VERSION_CHECK": "1",
    }
    result = CliRunner().invoke(cli, ["maintain", "--json"], env=env)
    assert result.exit_code == 0, result.output

    receipt = json.loads(result.output)
    assert "synthesis_count" in receipt
    assert receipt["synthesis_count"] == 0  # empty corpus → no clusters

    state_file = tmp_path / "state" / "synthesis_state.json"
    assert state_file.is_file(), "synthesis_state.json must be written after the maintain run"
    data = json.loads(state_file.read_text(encoding="utf-8"))
    assert "last_run" in data
    assert data["last_run"]  # non-empty ISO timestamp


def test_maint_synthesize_dry_run_does_not_write_state(tmp_path: Path):
    """A --dry-run maintain with MEMO_MAINT_SYNTHESIZE=1 must NOT persist the state file."""
    env = {
        **_env(tmp_path),
        "MEMO_MAINT_SYNTHESIZE": "1",
        "MEMO_SYNTHESIS_ENABLED": "0",
        "MEMO_SKIP_MODEL_VERSION_CHECK": "1",
    }
    result = CliRunner().invoke(cli, ["maintain", "--dry-run", "--json"], env=env)
    assert result.exit_code == 0, result.output

    state_file = tmp_path / "state" / "synthesis_state.json"
    assert not state_file.exists(), "dry-run must not write synthesis_state.json"


def test_maint_synthesize_non_fatal_on_error(tmp_path: Path):
    """If synthesize_cross_cluster raises, maintain finishes its remaining passes.

    "Non-fatal" means the run is not aborted and the receipt is complete — not
    that the failure is invisible. Since the P1 audit the exit code reports it.
    """
    from memo.cli_maintain import _synthesis_state_path

    env = {
        **_env(tmp_path),
        "MEMO_MAINT_SYNTHESIZE": "1",
        "MEMO_SYNTHESIS_ENABLED": "0",
        "MEMO_SKIP_MODEL_VERSION_CHECK": "1",
    }

    # Patch Memory.synthesize_cross_cluster to raise.
    with patch("memo.memory.Memory.synthesize_cross_cluster", side_effect=RuntimeError("boom")):
        result = CliRunner().invoke(cli, ["maintain", "--json"], env=env)

    # The pass failed, so maintain reports it in the exit code (P1 audit).
    assert result.exit_code == 1, result.output
    receipt = json.loads(result.output)
    # Error should be recorded but not fatal.
    assert any("maint_synthesize" in e for e in receipt.get("errors", []))
    # State file should NOT be written when synthesis crashed.
    assert not _synthesis_state_path(
        Config(
            data_dir=tmp_path / "data",
            state_dir=tmp_path / "state",
            reranker_enabled=False,
        )
    ).exists()


def test_maintain_writes_timestamped_run_receipt(tmp_cfg):
    import json as _json

    from click.testing import CliRunner

    from memo.cli_maintain import maintain_cmd

    runner = CliRunner()
    res = runner.invoke(
        maintain_cmd,
        ["--skip-contradict", "--skip-consolidate", "--skip-stale", "--skip-synthesize", "--json"],
        env={
            "MEMO_NONINTERACTIVE": "1",
            "MEMO_DATA_DIR": str(tmp_cfg.data_dir),
            "MEMO_STATE_DIR": str(tmp_cfg.state_dir),
            "MEMO_OUTCOME_RANKING_ENABLED": "0",
        },
    )
    assert res.exit_code == 0, res.output
    runs = list((tmp_cfg.state_dir / "maintain" / "runs").glob("*.json"))
    assert len(runs) == 1
    receipt = _json.loads(runs[0].read_text(encoding="utf-8"))
    assert receipt["run"] == runs[0].stem
    # last.json still written (daily guard + undo default read it)
    assert (tmp_cfg.state_dir / "maintain" / "last.json").is_file()


def test_undo_targets_and_restore_from_inactive(mock_memory):
    from memo.cli_maintain import _restore_archived, _undo_targets

    a = mock_memory.save(content="stale one", title="Stale")
    qc = mock_memory.save(content="compact me", title="Compact")
    f = mock_memory.save(content="dead weight", title="Dead")
    assert mock_memory.lifecycle.archive_memory(a.id) is True
    assert mock_memory.lifecycle.archive_memory(qc.id, superseded_by="canonical") is True
    assert mock_memory.get(a.id) is None
    assert mock_memory.get(qc.id) is None
    mock_memory.forget(f.id, reason="test")

    receipt = {
        "superseded": [],
        "merged": [],
        "archived_stale": [{"id": a.id, "days": 400}],
        "quality_compacted": [{"proposal_id": "quality-compact-demo", "archived_ids": [qc.id]}],
        "dead_archived": [f.id],
        "forgotten": [],
    }
    archived, forgotten, invalidated = _undo_targets(receipt)
    assert archived == [a.id, qc.id] and forgotten == [f.id] and invalidated == []

    restored, missing = _restore_archived(mock_memory, archived, dry_run=False)
    assert set(restored) == {a.id, qc.id} and missing == []
    mock_memory.reindex()
    assert mock_memory.get(a.id) is not None
    assert mock_memory.get(qc.id) is not None
    assert mock_memory.unforget(f.id) is not None


def test_maintain_undo_reopens_invalidated_loser(mem_with_stub):
    """`memo maintain undo` must reverse a contradiction-supersede (Bug B).

    Supersede now closes the loser's interval in place with action='invalidate'
    (not archive), so undo has to REOPEN it — clear invalid_at + drop
    superseded_by in both the index and the markdown — restoring it to default
    recall, mirroring how archive-undo restores a moved file.
    """
    from memo.cli_maintain import _reopen_invalidated, _undo_targets

    loser = mem_with_stub.save(
        content="prod db is postgres",
        title="A",
        type_="fact",
        valid_at="2020-06-01T00:00:00",
    )
    winner = mem_with_stub.save(
        content="prod db is mysql",
        title="B",
        type_="fact",
        valid_at="2020-07-01T00:00:00",
    )

    ok = mem_with_stub.lifecycle.invalidate_in_place(
        loser_id=loser.id, winner_id=winner.id, invalid_at=winner.valid_at
    )
    assert ok is True

    # Precondition: the closed interval hides the loser from default recall and
    # stamps supersede provenance in the index + markdown.
    closed = mem_with_stub.get(loser.id)
    assert closed.invalid_at == "2020-07-01T00:00:00"
    assert closed.extra.get("superseded_by") == winner.id
    before = {r.id for r in mem_with_stub.search("prod db", mode="bm25", limit=10)}
    assert loser.id not in before

    receipt = {
        "superseded": [
            {"pair_id": 1, "older": loser.id, "action": "invalidate", "confidence": 0.95}
        ],
        "merged": [],
        "archived_stale": [],
        "dead_archived": [],
        "forgotten": [],
    }
    archived, forgotten, invalidated = _undo_targets(receipt)
    assert archived == [] and forgotten == [] and invalidated == [loser.id]

    reopened, missing = _reopen_invalidated(mem_with_stub, invalidated, dry_run=False)
    assert reopened == [loser.id] and missing == []

    # Index reopened: interval cleared, provenance gone → back in default recall.
    rec = mem_with_stub.get(loser.id)
    assert rec.invalid_at is None
    assert "superseded_by" not in (rec.extra or {})
    after = {r.id for r in mem_with_stub.search("prod db", mode="bm25", limit=10)}
    assert loser.id in after

    # Markdown mirrored so a reindex --rebuild keeps the interval open.
    md_text = (mem_with_stub.cfg.memory_dir / rec.path).read_text(encoding="utf-8")
    assert "invalid_at:" not in md_text
    assert "superseded_by" not in md_text


def test_quality_compact_rollback_ids_include_attempted_ids():
    from memo.cli_maintain import _quality_compact_rollback_ids

    attempted_id = ("abc123" * 5 + "ab")[:32]
    archived_id = ("def456" * 5 + "de")[:32]
    receipt = {
        "quality_compacted": [
            {
                "proposal_id": "quality-compact-demo",
                "archived_ids": [],
                "attempted_ids": [attempted_id],
            },
            {"proposal_id": "quality-compact-demo-2", "archived_ids": [archived_id]},
        ]
    }

    rollback_ids = _quality_compact_rollback_ids(receipt)
    assert rollback_ids == [attempted_id, archived_id]


def test_maintain_undo_cli_dry_run_reads_receipt(tmp_cfg):
    import json as _json

    from click.testing import CliRunner

    from memo.cli_maintain import maintain_cmd

    d = tmp_cfg.state_dir / "maintain"
    d.mkdir(parents=True, exist_ok=True)
    (d / "last.json").write_text(
        _json.dumps(
            {
                "ts": 1.0,
                "superseded": [{"older": "a" * 32, "action": "archive"}],
                "merged": [],
                "archived_stale": [],
                "dead_archived": [],
                "forgotten": [],
            }
        ),
        encoding="utf-8",
    )
    runner = CliRunner()
    res = runner.invoke(
        maintain_cmd,
        ["undo", "--dry-run", "--json"],
        env={
            "MEMO_NONINTERACTIVE": "1",
            "MEMO_DATA_DIR": str(tmp_cfg.data_dir),
            "MEMO_STATE_DIR": str(tmp_cfg.state_dir),
        },
    )
    assert res.exit_code == 0, res.output
    out = _json.loads(res.output)
    assert out["dry_run"] is True
    assert out["missing"] == ["a" * 32]  # nothing in inactive/ to restore


def test_maintain_undo_rejects_receipt_path_traversal(tmp_cfg):
    from click.testing import CliRunner

    from memo.cli_maintain import maintain_cmd

    (tmp_cfg.state_dir / "maintain" / "runs").mkdir(parents=True)
    victim = tmp_cfg.state_dir / "victim.json"
    victim.write_text(
        json.dumps(
            {
                "superseded": [],
                "merged": [],
                "archived_stale": [],
                "dead_archived": [],
                "forgotten": [],
            }
        ),
        encoding="utf-8",
    )

    res = CliRunner().invoke(
        maintain_cmd,
        ["undo", "--run", "../../victim", "--dry-run", "--json"],
        env={
            "MEMO_NONINTERACTIVE": "1",
            "MEMO_DATA_DIR": str(tmp_cfg.data_dir),
            "MEMO_STATE_DIR": str(tmp_cfg.state_dir),
        },
    )

    assert res.exit_code != 0
    assert "receipt" in res.output.lower()
    assert victim.is_file()


# -- receipt honesty: only actual successes are recorded ---------------------


def test_stale_archive_failure_is_not_recorded_as_archived(tmp_path: Path):
    """archive_memory() returning False (e.g. memory deleted concurrently)
    must not land in receipt['archived_stale'] — `memo maintain undo` would
    chase ids that were never moved."""
    from unittest.mock import MagicMock

    mem = MagicMock()
    mem.lifecycle.enforce_forget_ttl.return_value = []
    mem.temporal.detect_stale_memories.return_value = [
        {"id": "aaaa1111", "days_since_update": 400},
        {"id": "bbbb2222", "days_since_update": 500},
    ]
    mem.lifecycle.archive_memory.side_effect = [True, False]

    env = {**_env(tmp_path), "MEMO_OUTCOME_RANKING_ENABLED": "0"}
    with patch("memo.cli_maintain._get_memory", return_value=mem):
        result = CliRunner().invoke(
            cli,
            ["maintain", "--skip-contradict", "--skip-consolidate", "--skip-synthesize", "--json"],
            env=env,
        )

    # The pass failed, so maintain reports it in the exit code (P1 audit).
    assert result.exit_code == 1, result.output
    receipt = json.loads(result.output)
    assert [e["id"] for e in receipt["archived_stale"]] == ["aaaa1111"]
    assert any("stale: archive failed for bbbb2222" in e for e in receipt["errors"])


def test_supersede_invalidate_failure_is_not_recorded_as_superseded(tmp_path: Path):
    """When the invalidate/delete of the dominated side fails, the pair stays
    open and must NOT be listed in receipt['superseded']."""
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from memo.belief import ARCHIVE

    mem = MagicMock()
    mem.lifecycle.enforce_forget_ttl.return_value = []
    mem.temporal.detect_stale_memories.return_value = []
    mem.contradict_store.list_open.return_value = [
        SimpleNamespace(
            pair_id="p1",
            memory_id_a="aaaa1111",
            memory_id_b="bbbb2222",
            relationship="contradicts",
            confidence=0.95,
        )
    ]
    updated = {
        "aaaa1111": "2026-01-01T00:00:00+00:00",
        "bbbb2222": "2026-01-02T00:00:00+00:00",
    }
    # The winner (bbbb2222) must expose a valid_at so the call site can compute
    # the loser's close-date before invoking invalidate_in_place.
    mem.get.side_effect = lambda i: SimpleNamespace(
        updated=updated[i], valid_at=updated[i], created=updated[i]
    )
    mem.lifecycle.invalidate_in_place.return_value = False

    decision = SimpleNamespace(
        action=ARCHIVE,
        dominated_id="aaaa1111",
        dominant_id="bbbb2222",
        reason="test",
        support_dominated=0,
    )

    env = {**_env(tmp_path), "MEMO_OUTCOME_RANKING_ENABLED": "0", "MEMO_CROSSREF_INDEX": "0"}
    with (
        patch("memo.cli_maintain._get_memory", return_value=mem),
        patch("memo.cli_maintain.supersede_decision", return_value=decision),
    ):
        result = CliRunner().invoke(
            cli,
            ["maintain", "--skip-consolidate", "--skip-stale", "--skip-synthesize", "--json"],
            env=env,
        )

    # The pass failed, so maintain reports it in the exit code (P1 audit).
    assert result.exit_code == 1, result.output
    receipt = json.loads(result.output)
    assert receipt["superseded"] == []
    assert any("supersede: invalidate failed for aaaa1111" in e for e in receipt["errors"])
    mem.contradict_store.resolve.assert_not_called()


def test_supersede_hard_delete_calls_delete(tmp_path: Path):
    """`--hard-delete` takes the destructive branch: the dominated side is
    deleted (not invalidated) and recorded with action='delete'."""
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from memo.belief import ARCHIVE

    mem = MagicMock()
    mem.lifecycle.enforce_forget_ttl.return_value = []
    mem.temporal.detect_stale_memories.return_value = []
    mem.contradict_store.list_open.return_value = [
        SimpleNamespace(
            pair_id="p1",
            memory_id_a="aaaa1111",
            memory_id_b="bbbb2222",
            relationship="contradicts",
            confidence=0.95,
        )
    ]
    updated = {
        "aaaa1111": "2026-01-01T00:00:00+00:00",
        "bbbb2222": "2026-01-02T00:00:00+00:00",
    }
    mem.get.side_effect = lambda i: SimpleNamespace(
        updated=updated[i], valid_at=updated[i], created=updated[i]
    )
    mem.delete.return_value = True

    decision = SimpleNamespace(
        action=ARCHIVE,
        dominated_id="aaaa1111",
        dominant_id="bbbb2222",
        reason="test",
        support_dominated=0,
    )

    env = {**_env(tmp_path), "MEMO_OUTCOME_RANKING_ENABLED": "0", "MEMO_CROSSREF_INDEX": "0"}
    with (
        patch("memo.cli_maintain._get_memory", return_value=mem),
        patch("memo.cli_maintain.supersede_decision", return_value=decision),
    ):
        result = CliRunner().invoke(
            cli,
            [
                "maintain",
                "--hard-delete",
                "--skip-consolidate",
                "--skip-stale",
                "--skip-synthesize",
                "--json",
            ],
            env=env,
        )

    assert result.exit_code == 0, result.output
    receipt = json.loads(result.output)
    mem.delete.assert_called_once_with("aaaa1111")
    mem.lifecycle.invalidate_in_place.assert_not_called()
    assert receipt["superseded"] and receipt["superseded"][0]["action"] == "delete"


def test_undo_targets_skips_superseded_without_older():
    """A superseded entry that isn't a dict, or carries no `older`, is skipped —
    never dereferenced."""
    from memo.cli_maintain import _undo_targets

    receipt = {
        "superseded": [
            "not-a-dict",
            {"action": "archive"},  # no `older`
            {"older": "", "action": "invalidate"},  # falsy `older`
        ],
        "merged": [],
        "archived_stale": [],
        "dead_archived": [],
        "forgotten": [],
    }
    archived, forgotten, invalidated = _undo_targets(receipt)
    assert archived == [] and forgotten == [] and invalidated == []


def test_reopen_invalidated_skips_missing_record(mock_memory):
    """A loser whose record is gone is reported missing, never reopened."""
    from memo.cli_maintain import _reopen_invalidated

    reopened, missing = _reopen_invalidated(mock_memory, ["deadbeef" * 4], dry_run=False)
    assert reopened == [] and missing == ["deadbeef" * 4]


def test_reopen_invalidated_frontmatter_failure_is_non_fatal(mem_with_stub, monkeypatch):
    """A failed markdown reopen still reopens the index row and is swallowed —
    the loser returns to recall even if the disk mirror can't be written."""
    import frontmatter

    from memo.cli_maintain import _reopen_invalidated

    loser = mem_with_stub.save(
        content="prod db is postgres", title="A", type_="fact", valid_at="2020-06-01T00:00:00"
    )
    winner = mem_with_stub.save(
        content="prod db is mysql", title="B", type_="fact", valid_at="2020-07-01T00:00:00"
    )
    mem_with_stub.lifecycle.invalidate_in_place(
        loser_id=loser.id, winner_id=winner.id, invalid_at=winner.valid_at
    )

    def boom(*a, **k):
        raise RuntimeError("yaml dump failed")

    monkeypatch.setattr(frontmatter, "dumps", boom)

    reopened, missing = _reopen_invalidated(mem_with_stub, [loser.id], dry_run=False)
    assert reopened == [loser.id] and missing == []
    # Index reopened despite the markdown mirror failing.
    assert mem_with_stub.get(loser.id).invalid_at is None


def test_maintain_exits_nonzero_when_the_receipt_carries_errors(tmp_path):
    """A failed pass must not hide under a success banner and exit 0."""
    import unittest.mock

    env = {
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_DATA_DIR": str(tmp_path / "data"),
        "MEMO_STATE_DIR": str(tmp_path / "state"),
    }
    runner = CliRunner()

    # enforce_forget_ttl is maintain's first pass; its except clause is the
    # shortest path to a populated receipt["errors"].
    with unittest.mock.patch(
        "memo.lifecycle.LifecycleManager.enforce_forget_ttl",
        side_effect=RuntimeError("boom"),
    ):
        result = runner.invoke(cli, ["maintain"], env=env)

    assert result.exit_code != 0, result.output
    assert "forget: RuntimeError: boom" in result.output


def test_maintain_dry_run_reports_errors_without_failing(tmp_path):
    """A preview surfaces the error but stays exit 0 — it changed nothing."""
    import unittest.mock

    env = {
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_DATA_DIR": str(tmp_path / "data"),
        "MEMO_STATE_DIR": str(tmp_path / "state"),
    }
    runner = CliRunner()

    with unittest.mock.patch(
        "memo.lifecycle.LifecycleManager.enforce_forget_ttl",
        side_effect=RuntimeError("boom"),
    ):
        result = runner.invoke(cli, ["maintain", "--dry-run"], env=env)

    assert result.exit_code == 0, result.output
    assert "forget: RuntimeError: boom" in result.output


def test_a_vanished_source_file_is_skipped_not_failed():
    """A pair whose source .md is gone must not fail the whole run.

    Vault-ingested rows carry a path relative to the Obsidian vault root
    (`notes/...`, `work/...`), which `_resolve_existing` cannot resolve
    because `Config.from_env()` leaves `vault_path` unset — so the file is
    genuinely unreachable and the pair raises FileNotFoundError. That is an
    expected, recurring, non-actionable condition affecting thousands of
    rows, not a failure of the pass: the run still supersedes, merges and
    synthesizes everything else.

    Folding it into `errors` made `_fail_on_pass_errors` exit 1, so the
    nightly `contradict-resolve` reported FAILED every night with the work
    actually done. An exit code that is always red teaches the operator to
    ignore it, which is worse than no exit code.
    """
    from memo.cli_maintain import _record_pair_failure

    receipt = {"errors": [], "skipped": []}

    _record_pair_failure(receipt, 42, FileNotFoundError(2, "No such file", "notes/gone.md"))
    assert receipt["errors"] == []
    assert len(receipt["skipped"]) == 1
    assert "42" in receipt["skipped"][0]

    _record_pair_failure(receipt, 43, RuntimeError("boom"))
    assert len(receipt["errors"]) == 1
    assert "RuntimeError" in receipt["errors"][0]
    assert len(receipt["skipped"]) == 1


def test_only_real_errors_fail_the_run():
    """The exit code must mean something: skipped pairs are reported but do
    not turn the run red; a genuine error still does."""
    from memo.cli_maintain import _fail_on_pass_errors

    _fail_on_pass_errors({"errors": [], "skipped": ["pair 42: source file gone"]}, dry_run=False)

    with pytest.raises(SystemExit):
        _fail_on_pass_errors({"errors": ["contradict: RuntimeError: boom"]}, dry_run=False)
