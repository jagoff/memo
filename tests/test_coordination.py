"""Tests for cross-agent coordination — live collision scan + LLM directives.

Covers the spec's test plan (docs/SPECS/2026-07-31-cross-agent-coordination-design.md):
store dedup/delivery/status transitions, candidate detection fixtures that
reproduce the three 2026-07-31 collisions, a mocked-LLM judge (JSON / garbage /
timeout — the real model is NEVER loaded here), hook delivery idempotence,
the watcher trigger thread, and CLI smoke.
"""

from __future__ import annotations

import inspect
import json
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

from memo import coordination
from memo.coordination import (
    Collision,
    CoordinationStore,
    ScanResult,
    build_activity,
    collision_id,
    coordination_db_path,
    deliver_pending_block,
    detect_candidates,
    gather_activities,
    maybe_start_scan_thread,
    render_directives_block,
    scan_collisions,
)

if TYPE_CHECKING:
    from memo.config import Config

NOW = datetime(2026, 7, 31, 12, 0, 0, tzinfo=UTC)
LATER = NOW + timedelta(hours=1)


# ── helpers ──────────────────────────────────────────────────────────────────


def _collision(
    session_a: str = "sess-a",
    session_b: str = "sess-b",
    resource: str = "README.md",
    **overrides: Any,
) -> Collision:
    fields: dict[str, Any] = {
        "id": collision_id(session_a, session_b, resource),
        "session_a": session_a,
        "session_b": session_b,
        "resource": resource,
        "kind": "file",
        "severity": "warn",
        "rationale": "both touch it",
        "directive_a": "hold README edits until sess-b lands",
        "directive_b": "continue, you own the README change",
        "status": "open",
        "created_at": NOW.isoformat(),
        "delivered_a": None,
        "delivered_b": None,
    }
    fields.update(overrides)
    return Collision(**fields)


def _row(
    session_id: str,
    title: str,
    *,
    files_modified: tuple[str, ...] = (),
    files_read: tuple[str, ...] = (),
    tags: tuple[str, ...] = (),
) -> dict[str, Any]:
    return {
        "id": f"mem-{abs(hash((session_id, title))):x}",
        "title": title,
        "type": "capture",
        "tags": list(tags),
        "updated": NOW.isoformat(),
        "extra": {
            "provenance": {"session_id": session_id},
            "files_read": list(files_read),
            "files_modified": list(files_modified),
        },
    }


def _write_session(
    state_dir: Path,
    session_id: str,
    *,
    updated: datetime,
    project: str = "memo",
) -> None:
    d = state_dir / "sessions"
    d.mkdir(parents=True, exist_ok=True)
    payload = {
        "session_id": session_id,
        "project": project,
        "cwd": "/tmp/repo",
        "updated": updated.isoformat(timespec="seconds"),
    }
    (d / f"{session_id}.json").write_text(json.dumps(payload), encoding="utf-8")


class _FakeStore:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def list_recent(self, limit: int = 20, **_kw: Any) -> list[dict[str, Any]]:
        return self.rows[:limit]


class _FakeMemory:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.store = _FakeStore(rows)


class _JsonChat:
    """Chat double returning a fixed payload — the 4B model is never loaded."""

    def __init__(self, payload: str) -> None:
        self.payload = payload
        self.calls = 0

    def chat(self, **_kw: Any) -> dict[str, Any]:
        self.calls += 1
        return {"message": {"content": self.payload}}


class _BoomChat:
    """Chat double that must never be reached."""

    def chat(self, **_kw: Any) -> dict[str, Any]:
        raise AssertionError("LLM judge must not run for deduped candidates")


_COLLISION_JSON = json.dumps(
    {
        "collision": True,
        "severity": "warn",
        "rationale": "both agents are editing the same README banner",
        "directive_a": "pause your README edit; sess-b owns the banner fix",
        "directive_b": "land your README fix; sess-a will rebase on top",
    }
)


def _activity(
    session_id: str,
    titles: tuple[str, ...],
    *,
    files: tuple[str, ...] = (),
    project: str = "memo",
    focus: str = "",
) -> coordination.SessionActivity:
    rows = [
        _row(session_id, title, files_modified=files if i == 0 else ())
        for i, title in enumerate(titles)
    ]
    return build_activity(session_id, project, rows, focus)


# ── store: dedup, delivery stamping, status transitions ──────────────────────


def test_store_insert_then_active_dedup_keeps_original(tmp_path: Path) -> None:
    with CoordinationStore(tmp_path / "coordination.db") as store:
        assert store.upsert(_collision()) is True

        assert store.upsert(_collision(rationale="second sighting")) is False

        rows = store.list_collisions()
        assert len(rows) == 1
        assert rows[0].rationale == "both touch it"
        assert rows[0].status == "open"


def test_store_resolved_row_is_rejudged_and_refreshed(tmp_path: Path) -> None:
    with CoordinationStore(tmp_path / "coordination.db") as store:
        store.upsert(_collision())
        cid = collision_id("sess-a", "sess-b", "README.md")
        assert store.resolve(cid) is True

        assert store.upsert(_collision(rationale="fresh", created_at=LATER.isoformat())) is True

        row = store.list_collisions(statuses=("open",))[0]
        assert row.rationale == "fresh"
        assert row.created_at == LATER.isoformat()
        assert row.delivered_a is None and row.delivered_b is None


def test_store_delivery_stamps_each_side_then_status_delivered(tmp_path: Path) -> None:
    with CoordinationStore(tmp_path / "coordination.db") as store:
        store.upsert(_collision())
        cid = collision_id("sess-a", "sess-b", "README.md")

        pending_a = store.pending_directives("sess-a")
        assert len(pending_a) == 1
        assert pending_a[0].directive == "hold README edits until sess-b lands"
        assert pending_a[0].other_session == "sess-b"

        store.mark_delivered(cid, "sess-a", NOW.isoformat())
        assert store.pending_directives("sess-a") == []
        assert store.list_collisions()[0].status == "open"  # b not delivered yet

        pending_b = store.pending_directives("sess-b")
        assert pending_b[0].directive == "continue, you own the README change"
        store.mark_delivered(cid, "sess-b", NOW.isoformat())

        row = store.list_collisions(statuses=("delivered",))[0]
        assert row.status == "delivered"
        assert row.delivered_a and row.delivered_b
        assert store.pending_directives("sess-b") == []


def test_store_resolve_unknown_id_and_stale_expiry(tmp_path: Path) -> None:
    with CoordinationStore(tmp_path / "coordination.db") as store:
        assert store.resolve("deadbeefdeadbeef") is False

        old = _collision(created_at=(NOW - timedelta(hours=7)).isoformat())
        store.upsert(old)
        expired = store.expire_stale((NOW - timedelta(hours=6)).isoformat())

        assert expired == 1
        assert store.active_ids() == set()
        assert store.list_collisions(statuses=("stale",))[0].id == old.id


# ── candidate detection: the three 2026-07-31 collisions ─────────────────────


def test_detects_file_overlap_readme_banner() -> None:
    a = _activity("sess-a", ("fix README banner drift",), files=("repos/memo/README.md",))
    b = _activity("sess-b", ("realign the README banner",), files=("repos/memo/README.md",))

    candidates = detect_candidates((a, b))

    assert any(c.kind == "file" and c.resource.endswith("README.md") for c in candidates)


def test_detects_daemon_label_overlap() -> None:
    a = _activity("sess-a", ("realign uv-tool runtime for com.memo.chat",))
    b = _activity("sess-b", ("debug com.memo.chat crashloop under KeepAlive",))

    candidates = detect_candidates((a, b))

    assert any(c.kind == "daemon" and c.resource == "com.memo.chat" for c in candidates)


def test_detects_branch_overlap_master_merge_queue() -> None:
    a = _activity("sess-a", ("merge PR to master branch now",))
    b = _activity("sess-b", ("serialized dependabot queue on master",))

    candidates = detect_candidates((a, b))

    assert any(c.kind == "branch" and c.resource == "master" for c in candidates)


def test_detects_slash_branch_names() -> None:
    a = _activity("sess-a", ("push feat/cross-agent-coordination for review",))
    b = _activity("sess-b", ("rebase feat/cross-agent-coordination onto upstream",))

    candidates = detect_candidates((a, b))

    assert any(
        c.kind == "branch" and c.resource == "feat/cross-agent-coordination" for c in candidates
    )


def test_detects_topic_jaccard_overlap() -> None:
    a = _activity("sess-a", ("memo recall hook token budget tuning",))
    b = _activity("sess-b", ("tuning recall hook token budget",))

    candidates = detect_candidates((a, b))

    assert any(c.kind == "topic" for c in candidates)


def test_no_candidates_below_topic_threshold_or_single_session() -> None:
    a = _activity("sess-a", ("postgres schema migration invoices",))
    b = _activity("sess-b", ("frontend hero animation polish",))

    assert detect_candidates((a, b)) == ()
    assert detect_candidates((a,)) == ()
    assert detect_candidates(()) == ()


# ── gathering: sessions + capture memories + focus (fail-open) ───────────────


def test_gather_activities_filters_by_active_window(tmp_cfg: Config) -> None:
    _write_session(tmp_cfg.state_dir, "sess-live", updated=NOW - timedelta(minutes=5))
    _write_session(tmp_cfg.state_dir, "sess-dead", updated=NOW - timedelta(hours=10))
    mem = _FakeMemory([_row("sess-live", "fix README", files_modified=("README.md",))])

    activities = gather_activities(mem, tmp_cfg, now=NOW)

    assert [a.session_id for a in activities] == ["sess-live"]
    assert any(f.endswith("README.md") for f in activities[0].files)


def test_gather_activities_joins_operational_focus(tmp_cfg: Config) -> None:
    from memo.operational import OperationalStore

    _write_session(tmp_cfg.state_dir, "sess-live", updated=NOW, project="memo")
    OperationalStore(
        tmp_cfg.state_dir,
        device_id=tmp_cfg.device_id,
        context_provider=tmp_cfg.operational_context_provider,
        epoch_fence=tmp_cfg.operational_epoch_fence,
    ).set_focus(project="memo", summary="fix the README banner")

    activities = gather_activities(_FakeMemory([]), tmp_cfg, now=NOW)

    assert activities[0].focus == "fix the README banner"


def test_gather_activities_fail_open_on_empty_state(tmp_cfg: Config) -> None:
    class _BrokenStore:
        def list_recent(self, **_kw: Any) -> list[dict[str, Any]]:
            raise OSError("sidecar unavailable")

    mem = MagicMock()
    mem.store = _BrokenStore()

    assert gather_activities(mem, tmp_cfg, now=NOW) == ()

    _write_session(tmp_cfg.state_dir, "sess-live", updated=NOW)
    activities = gather_activities(mem, tmp_cfg, now=NOW)
    assert len(activities) == 1
    assert activities[0].files == frozenset()


# ── scan: judge with mocked chat (JSON / garbage / timeout / dedup) ──────────


def _seed_two_readme_sessions(cfg: Config) -> _FakeMemory:
    _write_session(cfg.state_dir, "sess-a", updated=NOW)
    _write_session(cfg.state_dir, "sess-b", updated=NOW)
    return _FakeMemory(
        [
            _row("sess-a", "fix README banner", files_modified=("repos/memo/README.md",)),
            _row("sess-b", "align README banner", files_modified=("repos/memo/README.md",)),
        ]
    )


def test_scan_persists_confirmed_collision(tmp_cfg: Config) -> None:
    mem = _seed_two_readme_sessions(tmp_cfg)
    chat = _JsonChat(_COLLISION_JSON)

    result = scan_collisions(mem, tmp_cfg, now=NOW, chat=chat)

    assert result.sessions == 2
    assert result.judged == 1
    assert result.collisions == 1
    with CoordinationStore(coordination_db_path(tmp_cfg)) as store:
        rows = store.list_collisions(statuses=("open",))
        assert len(rows) == 1
        assert rows[0].directive_a.startswith("pause your README edit")
        assert rows[0].severity == "warn"
        assert {rows[0].session_a, rows[0].session_b} == {"sess-a", "sess-b"}


def test_scan_skips_garbage_judge_output(tmp_cfg: Config) -> None:
    mem = _seed_two_readme_sessions(tmp_cfg)

    result = scan_collisions(mem, tmp_cfg, now=NOW, chat=_JsonChat("not json at all"))

    assert result.judged == 1
    assert result.collisions == 0
    with CoordinationStore(coordination_db_path(tmp_cfg)) as store:
        assert store.list_collisions() == []


def test_scan_skips_non_collision_and_missing_directives(tmp_cfg: Config) -> None:
    mem = _seed_two_readme_sessions(tmp_cfg)

    no_collision = json.dumps({"collision": False})
    assert scan_collisions(mem, tmp_cfg, now=NOW, chat=_JsonChat(no_collision)).collisions == 0

    missing = json.dumps({"collision": True, "directive_a": "only one side"})
    assert scan_collisions(mem, tmp_cfg, now=NOW, chat=_JsonChat(missing)).collisions == 0


def test_scan_normalizes_unknown_severity_to_warn(tmp_cfg: Config) -> None:
    mem = _seed_two_readme_sessions(tmp_cfg)
    payload = json.dumps(
        {
            "collision": True,
            "severity": "catastrophic",
            "directive_a": "a directive",
            "directive_b": "b directive",
        }
    )

    result = scan_collisions(mem, tmp_cfg, now=NOW, chat=_JsonChat(payload))

    assert result.collisions == 1
    with CoordinationStore(coordination_db_path(tmp_cfg)) as store:
        assert store.list_collisions()[0].severity == "warn"


def test_scan_timeout_fails_open(tmp_cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    mem = _seed_two_readme_sessions(tmp_cfg)
    monkeypatch.setattr(coordination, "chat_with_timeout", lambda *_a, **_k: None)

    result = scan_collisions(mem, tmp_cfg, now=NOW, chat=_JsonChat(_COLLISION_JSON))

    assert result.judged == 1
    assert result.collisions == 0


def test_scan_does_not_rejudge_open_collisions(tmp_cfg: Config) -> None:
    mem = _seed_two_readme_sessions(tmp_cfg)
    first = scan_collisions(mem, tmp_cfg, now=NOW, chat=_JsonChat(_COLLISION_JSON))
    assert first.collisions == 1

    second = scan_collisions(mem, tmp_cfg, now=NOW, chat=_BoomChat())

    assert second.judged == 0
    assert second.skipped_active >= 1


def test_scan_fail_open_when_nothing_is_active(tmp_cfg: Config) -> None:
    result = scan_collisions(_FakeMemory([]), tmp_cfg, now=NOW, chat=_BoomChat())

    assert result == ScanResult(sessions=0, candidates=0, judged=0, collisions=0, skipped_active=0)


# ── delivery: recall-hook block (pure sqlite read, injected once) ────────────


def test_deliver_pending_block_injects_once_per_side(tmp_cfg: Config) -> None:
    with CoordinationStore(coordination_db_path(tmp_cfg)) as store:
        store.upsert(_collision())

    block = deliver_pending_block(tmp_cfg, "sess-a", now=NOW)
    assert "<memo-coordination>" in block
    assert "</memo-coordination>" in block
    assert "hold README edits until sess-b lands" in block
    assert "README.md" in block

    assert deliver_pending_block(tmp_cfg, "sess-a", now=NOW) == ""

    block_b = deliver_pending_block(tmp_cfg, "sess-b", now=NOW)
    assert "continue, you own the README change" in block_b
    with CoordinationStore(coordination_db_path(tmp_cfg)) as store:
        assert store.list_collisions(statuses=("delivered",))[0].status == "delivered"


def test_deliver_pending_block_without_session_or_rows(tmp_cfg: Config) -> None:
    assert deliver_pending_block(tmp_cfg, None, now=NOW) == ""
    assert deliver_pending_block(tmp_cfg, "", now=NOW) == ""
    assert deliver_pending_block(tmp_cfg, "sess-unknown", now=NOW) == ""


def test_deliver_pending_block_respects_disable_flag(
    tmp_cfg: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    with CoordinationStore(coordination_db_path(tmp_cfg)) as store:
        store.upsert(_collision())
    monkeypatch.setenv("MEMO_COORD_ENABLED", "0")

    assert deliver_pending_block(tmp_cfg, "sess-a", now=NOW) == ""


def test_deliver_pending_block_survives_unexpected_store_errors(
    tmp_cfg: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The recall hook rides on this — any store error must return ''."""

    def _boom(*_a: Any, **_k: Any) -> Any:
        raise AttributeError("corrupted sidecar row")

    monkeypatch.setattr(coordination, "CoordinationStore", _boom)

    assert deliver_pending_block(tmp_cfg, "sess-a", now=NOW) == ""


def test_hook_path_uses_short_store_timeout(tmp_path: Path) -> None:
    assert coordination._HOOK_DB_TIMEOUT_S == 0.05
    with CoordinationStore(tmp_path / "coordination.db", timeout_s=0.05) as store:
        busy_ms = store._conn.execute("PRAGMA busy_timeout").fetchone()[0]
        assert busy_ms == 50
    with CoordinationStore(tmp_path / "coordination.db") as store:
        assert store._conn.execute("PRAGMA busy_timeout").fetchone()[0] == 10000


def test_judge_prompt_is_capped_trimming_longer_summary_first() -> None:
    long_side, short_side = coordination._capped_summaries("a" * 5000, "b" * 1000, 3600)
    assert len(long_side) + len(short_side) <= 3600
    assert short_side == "b" * 1000  # the longer summary is truncated first

    big = _activity("sess-a", tuple(f"title {i} " + "x" * 200 for i in range(20)))
    small = _activity("sess-b", ("tiny title",))
    candidate = coordination.CollisionCandidate("sess-a", "sess-b", "README.md", "file")
    prompt = coordination._judge_prompt(candidate, big, small)
    assert len(prompt) <= coordination._MAX_JUDGE_PROMPT_CHARS
    assert "tiny title" in prompt


def test_render_directives_block_lists_each_directive(tmp_path: Path) -> None:
    with CoordinationStore(tmp_path / "coordination.db") as store:
        store.upsert(_collision())
        store.upsert(_collision(resource="com.memo.chat", kind="daemon", severity="block"))
        directives = store.pending_directives("sess-a")

    block = render_directives_block(directives)

    assert block.count("- [") == 2
    assert "[block]" in block and "[warn]" in block
    assert "com.memo.chat" in block


# ── trigger: watcher interval thread, gated by MEMO_COORD_ENABLED ────────────


def test_maybe_start_scan_thread_disabled_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MEMO_COORD_ENABLED", "0")

    assert maybe_start_scan_thread(threading.Event()) is None


def test_scan_thread_ticks_and_stops(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEMO_COORD_ENABLED", "1")
    calls: list[int] = []
    monkeypatch.setattr(coordination, "_run_one_scan", lambda: calls.append(1))
    stop = threading.Event()

    thread = maybe_start_scan_thread(stop, interval_s=0.02)

    assert thread is not None and thread.daemon
    deadline = time.monotonic() + 2.0
    while len(calls) < 2 and time.monotonic() < deadline:
        time.sleep(0.01)
    stop.set()
    thread.join(timeout=2.0)
    assert not thread.is_alive()
    assert len(calls) >= 2


def test_run_watcher_wires_coordination_trigger() -> None:
    from memo.watcher import run_watcher

    assert "maybe_start_scan_thread" in inspect.getsource(run_watcher)


def test_recall_hook_wires_coordination_delivery(tmp_cfg: Config) -> None:
    from memo.cli_recall_hook import _append_coordination_block, recall_hook

    assert "_append_coordination_block" in inspect.getsource(recall_hook.callback)

    with CoordinationStore(coordination_db_path(tmp_cfg)) as store:
        store.upsert(_collision())
    out = _append_coordination_block(tmp_cfg, "sess-a", "RECALL CONTEXT")
    assert out.startswith("RECALL CONTEXT\n\n<memo-coordination>")
    assert _append_coordination_block(tmp_cfg, None, "RECALL CONTEXT") == "RECALL CONTEXT"


# ── config knobs ─────────────────────────────────────────────────────────────


def test_coord_flags_registered_with_defaults(tmp_path: Path) -> None:
    from memo import flags

    env = {
        "MEMO_CONFIG_DIR": str(tmp_path / "no-md-config"),
        "MEMO_STATE_DIR": str(tmp_path / "no-overlay-state"),
    }

    assert flags.flag_bool("MEMO_COORD_ENABLED", env=env) is True
    assert flags.flag_int("MEMO_COORD_SCAN_INTERVAL", env=env) == 300
    assert flags.flag_int("MEMO_COORD_ACTIVE_WINDOW", env=env) == 21600


# ── CLI smoke: scan / status / resolve ───────────────────────────────────────


def _cli_env(cfg: Config) -> dict[str, str]:
    return {
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_DATA_DIR": str(cfg.data_dir),
        "MEMO_STATE_DIR": str(cfg.state_dir),
        "MEMO_VAULT_PATH": str(cfg.vault_path),
        "MEMO_SKIP_MODEL_VERSION_CHECK": "1",
    }


def test_cli_coordinate_status_and_resolve(tmp_cfg: Config) -> None:
    from memo.cli import cli

    with CoordinationStore(coordination_db_path(tmp_cfg)) as store:
        store.upsert(_collision())
    cid = collision_id("sess-a", "sess-b", "README.md")
    runner = CliRunner()

    status = runner.invoke(cli, ["coordinate", "status", "--json"], env=_cli_env(tmp_cfg))
    assert status.exit_code == 0, status.output
    assert cid in status.output

    resolve = runner.invoke(cli, ["coordinate", "resolve", cid], env=_cli_env(tmp_cfg))
    assert resolve.exit_code == 0, resolve.output
    with CoordinationStore(coordination_db_path(tmp_cfg)) as store:
        assert store.list_collisions(statuses=("resolved",))[0].id == cid


def test_cli_coordinate_scan_smoke(tmp_cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    import memo.cli_coordinate as cli_coordinate
    from memo.cli import cli

    monkeypatch.setattr(cli_coordinate, "_get_memory", lambda _cfg: MagicMock())
    monkeypatch.setattr(
        cli_coordinate,
        "scan_collisions",
        lambda _mem, _cfg: ScanResult(
            sessions=2, candidates=1, judged=1, collisions=1, skipped_active=0
        ),
    )

    result = CliRunner().invoke(cli, ["coordinate", "scan", "--json"], env=_cli_env(tmp_cfg))

    assert result.exit_code == 0, result.output
    assert '"collisions": 1' in result.output
