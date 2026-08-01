from __future__ import annotations

import inspect
import json
import os
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import MagicMock

import frontmatter
import pytest
from click.testing import CliRunner

from memo.atomic_io import atomic_write_text
from memo.cli import cli
from memo.contracts import ActorIdentity, AnswerStatus, normalize_provenance
from memo.errors import WriteRefused
from memo.independence_migration import migrate_independence
from memo.memory import Memory
from memo.operation_ledger import LedgerIntegrityError, OperationLedger
from memo.operational import OperationalStore
from memo.server_core_records import register as register_record_tools
from memo.server_operational import register as register_operational_tools
from tests.operational_authority import build_test_operational_authority


class _ToolServer:
    def __init__(self) -> None:
        self.tools: dict[str, object] = {}

    def tool(self, **_kwargs):
        def decorator(func):
            self.tools[func.__name__] = func
            return func

        return decorator


def _legacy_store(tmp_path, *, device_id="device-a"):
    authority = build_test_operational_authority(
        tmp_path / "test-operational-authority",
        device_id=device_id,
    )
    return OperationalStore(
        tmp_path,
        device_id=device_id,
        context_provider=authority.context_provider,
        epoch_fence=authority.fence,
    )


def test_operational_state_rebuilds_from_hash_chained_journal(tmp_path):
    store = _legacy_store(tmp_path)
    focus = store.set_focus(
        project="memo",
        summary="Ship native continuity",
        actor=ActorIdentity(actor_id="codex", actor_kind="agent"),
    )
    handoff = store.create_handoff(
        project="memo",
        summary="Run release checks",
        from_actor="codex",
        to_actor="maintainer",
    )
    store.add_attention(project="memo", summary="Review migration", severity="high")

    store.snapshot_path.unlink()
    rebuilt = store.state(project="memo")

    assert rebuilt["focus"]["memo"]["id"] == focus.id
    assert rebuilt["handoffs"][handoff.id]["consumed_at"] == ""
    assert store.ledger.verify()["ok"] is True


def test_operational_snapshot_rebuilds_when_journal_advances(tmp_path):
    store = _legacy_store(tmp_path)
    store.set_focus(project="memo", summary="before")
    conflict_id = "anomaly-fixed"
    store.record_anomaly(
        {
            "anomaly_id": conflict_id,
            "kind": "semantic_contradiction",
            "state": "detected",
            "summary": "incompatible facts",
            "memory_id_a": "one",
            "memory_id_b": "two",
            "relationship": "contradicts",
            "evidence_uris": ["memo://memoria/one", "memo://memoria/two"],
            "created_at": "2026-07-23T00:00:00+00:00",
        },
    )

    state = store.state()

    assert state["conflicts"][conflict_id]["freeze_write"] is True
    assert state["journal_heads"]["device-a"]


def test_operational_commit_rebuilds_when_later_same_device_event_wins_race(
    tmp_path,
    monkeypatch,
):
    store = _legacy_store(tmp_path)
    store.state()
    append = store._append_authorized_event

    def append_then_race(op, **kwargs):
        event = append(op, **kwargs)
        append(
            "focus.clear",
            subject_uri="memo://focus/memo",
            payload={
                "project": "memo",
                "cleared_at": "2026-07-23T00:00:00+00:00",
            },
        )
        return event

    monkeypatch.setattr(store, "_append_authorized_event", append_then_race)

    store.set_focus(project="memo", summary="superseded before snapshot commit")
    state = store.state()

    assert "memo" not in state["focus"]
    assert state["journal_heads"] == store.ledger.head_hashes()
    assert len(store.ledger.validated_events()) == 2


def test_operational_ledger_view_rejects_unauthorized_append(tmp_path):
    store = _legacy_store(tmp_path)

    with pytest.raises(PermissionError, match=r"(?i)store authorization"):
        store.ledger.append("focus.set", subject_uri="memo://focus/memo")


def test_operational_signal_cli_remembers_lists_and_fences_epochs(tmp_path):
    env = {
        "MEMO_DATA_DIR": str(tmp_path / "data"),
        "MEMO_STATE_DIR": str(tmp_path / "state"),
        "MEMO_VAULT_PATH": str(tmp_path / "vault"),
        "MEMO_CONFIG_FILE": str(tmp_path / "memo-config.toml"),
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_EMBEDDER_VIA_DAEMON": "0",
    }
    runner = CliRunner()

    remembered = runner.invoke(
        cli,
        [
            "operational",
            "signal",
            "remember",
            "--marker",
            "watcher:repo:memo",
            "--epoch",
            "3",
            "--fence",
            "leader-a",
            "--payload-json",
            '{"commits": 2}',
            "--actor-id",
            "synapse-watcher",
        ],
        env=env,
    )
    assert remembered.exit_code == 0, remembered.output
    record = json.loads(remembered.output)
    assert record["marker"] == "watcher:repo:memo"
    assert record["epoch"] == 3
    assert record["fence"] == "leader-a"
    assert record["payload"] == {"commits": 2}

    stale = runner.invoke(
        cli,
        [
            "operational",
            "signal",
            "remember",
            "--marker",
            "watcher:repo:memo",
            "--epoch",
            "2",
        ],
        env=env,
    )
    assert stale.exit_code != 0
    assert "stale signal epoch" in stale.output

    listed = runner.invoke(
        cli,
        [
            "operational",
            "signal",
            "list",
            "--marker",
            "watcher:repo:memo",
            "--min-epoch",
            "3",
            "--limit",
            "1",
        ],
        env=env,
    )
    assert listed.exit_code == 0, listed.output
    envelope = json.loads(listed.output)
    assert envelope["count"] == 1
    assert envelope["signals"] == [record]


def test_operational_signal_cli_rejects_non_object_payload(tmp_path):
    result = CliRunner().invoke(
        cli,
        [
            "operational",
            "signal",
            "remember",
            "--marker",
            "watcher:repo:memo",
            "--payload-json",
            "[]",
        ],
        env={
            "MEMO_DATA_DIR": str(tmp_path / "data"),
            "MEMO_STATE_DIR": str(tmp_path / "state"),
            "MEMO_VAULT_PATH": str(tmp_path / "vault"),
            "MEMO_CONFIG_FILE": str(tmp_path / "memo-config.toml"),
            "MEMO_NONINTERACTIVE": "1",
            "MEMO_EMBEDDER_VIA_DAEMON": "0",
        },
    )

    assert result.exit_code != 0
    assert "must decode to a JSON object" in result.output


def test_mcp_operational_signal_tools_preserve_json_contract(tmp_path):
    store = _legacy_store(tmp_path)
    memory = SimpleNamespace(
        operational=store,
        cfg=SimpleNamespace(device_id="device-a"),
    )
    server = _ToolServer()
    register_operational_tools(server, memory)

    remembered = server.tools["memo_signal_remember"](
        "watcher:repo:memo",
        epoch=4,
        fence="leader-b",
        payload={"commits": 3},
        actor_id="synapse-watcher",
    )
    assert remembered == {
        "marker": "watcher:repo:memo",
        "epoch": 4,
        "fence": "leader-b",
        "payload": {"commits": 3},
        "created_at": remembered["created_at"],
    }

    listed = server.tools["memo_signal_list"](
        marker="watcher:repo:memo",
        min_epoch=4,
        limit=1,
    )
    assert listed == {"signals": [remembered], "count": 1}


def test_operation_ledger_repairs_stale_head_and_rejects_unsafe_timestamps(tmp_path):
    ledger = OperationLedger(tmp_path, device_id="device-a")
    first = ledger.append("focus.set", subject_uri="memo://focus/memo")
    ledger._head_path.write_text(
        json.dumps({"sequence": 0, "event_hash": ""}),
        encoding="utf-8",
    )

    second = ledger.append("focus.clear", subject_uri="memo://focus/memo")

    assert second.sequence == 2
    assert second.previous_hash == first.event_hash
    assert ledger.verify()["ok"] is True
    with pytest.raises(LedgerIntegrityError, match="timestamp"):
        ledger.append(
            "focus.set",
            subject_uri="memo://focus/escape",
            ts="../../escape",
        )
    assert not (tmp_path / "escape.jsonl").exists()


def test_operation_ledger_fsyncs_authoritative_event_only(tmp_path, monkeypatch):
    ledger = OperationLedger(tmp_path, device_id="device-a")
    fsync_calls: list[int] = []

    monkeypatch.setattr(os, "fsync", lambda descriptor: fsync_calls.append(descriptor))

    ledger.append("focus.set", subject_uri="memo://focus/memo")

    assert len(fsync_calls) == 1
    assert ledger.verify()["ok"] is True


def test_atomic_write_text_is_durable_by_default_and_can_write_rebuildable_cache(
    tmp_path,
    monkeypatch,
):
    fsync_calls: list[int] = []
    monkeypatch.setattr(os, "fsync", lambda descriptor: fsync_calls.append(descriptor))

    durable_path = tmp_path / "durable.json"
    cache_path = tmp_path / "cache.json"
    atomic_write_text(durable_path, "durable")
    atomic_write_text(cache_path, "cache", durable=False)

    assert len(fsync_calls) == 1
    assert durable_path.read_text(encoding="utf-8") == "durable"
    assert cache_path.read_text(encoding="utf-8") == "cache"


def test_operation_ledger_detects_missing_or_malformed_segments(tmp_path):
    missing = OperationLedger(tmp_path / "missing", device_id="device-a")
    event = missing.append("focus.set", subject_uri="memo://focus/memo")
    missing._segment_path(event.ts).unlink()
    assert missing.verify()["ok"] is False

    malformed = OperationLedger(tmp_path / "malformed", device_id="device-b")
    event = malformed.append("focus.set", subject_uri="memo://focus/memo")
    with malformed._segment_path(event.ts).open("a", encoding="utf-8") as handle:
        handle.write("{bad-json}\n")
    assert malformed.verify()["ok"] is False


def test_operation_ledger_rejects_symlinked_authority_roots(tmp_path):
    target = tmp_path / "outside"
    target.mkdir()

    event_ledger = OperationLedger(tmp_path / "events", device_id="device-a")
    event_ledger.root.mkdir(parents=True)
    (event_ledger.root / "events").symlink_to(target, target_is_directory=True)
    with pytest.raises(LedgerIntegrityError, match="event root"):
        event_ledger.validated_events()

    head_ledger = OperationLedger(tmp_path / "heads", device_id="device-a")
    head_ledger.root.mkdir(parents=True)
    (head_ledger.root / "heads").symlink_to(target, target_is_directory=True)
    with pytest.raises(LedgerIntegrityError, match="head root"):
        head_ledger.validated_events()


@pytest.mark.parametrize(
    "linked_component",
    ("journal", "events", "heads", "device", "segment"),
)
def test_operation_ledger_append_rejects_symlinked_path_ancestors(
    tmp_path,
    linked_component,
):
    target = tmp_path / "outside"
    target.mkdir()
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    ledger = OperationLedger(state_dir, device_id="device-a")

    if linked_component == "journal":
        ledger.root.symlink_to(target, target_is_directory=True)
    else:
        ledger.root.mkdir()
        if linked_component in {"events", "heads"}:
            (ledger.root / linked_component).symlink_to(target, target_is_directory=True)
        elif linked_component == "device":
            (ledger.root / "events").mkdir()
            (ledger.root / "events" / "device-a").symlink_to(
                target,
                target_is_directory=True,
            )
        else:
            (ledger.root / "events" / "device-a").mkdir(parents=True)
            ledger._segment_path("2026-07-23T00:00:00+00:00").symlink_to(
                target,
                target_is_directory=True,
            )

    with pytest.raises(LedgerIntegrityError, match="symlink"):
        ledger.append(
            "focus.set",
            subject_uri="memo://focus/memo",
            ts="2026-07-23T00:00:00+00:00",
        )

    assert list(target.iterdir()) == []


def test_operation_ledger_allows_external_symlink_ancestor(tmp_path):
    real_parent = tmp_path / "private" / "var"
    real_parent.mkdir(parents=True)
    alias = tmp_path / "var"
    alias.symlink_to(real_parent, target_is_directory=True)
    state_dir = alias / "memo-state"
    ledger = OperationLedger(state_dir, device_id="device-a")

    event = ledger.append(
        "focus.set",
        subject_uri="memo://focus/memo",
        ts="2026-07-23T00:00:00+00:00",
    )

    assert ledger.verify()["ok"] is True
    assert (real_parent / "memo-state" / "journal" / "events" / "device-a").is_dir()
    assert ledger._segment_path(event.ts).is_file()


@pytest.mark.parametrize("row", ("[]", '{"actor": []}'))
def test_operation_ledger_verify_reports_non_object_event_shapes(tmp_path, row):
    ledger = OperationLedger(tmp_path, device_id="device-a")
    event = ledger.append("focus.set", subject_uri="memo://focus/memo")
    ledger._segment_path(event.ts).write_text(f"{row}\n", encoding="utf-8")

    result = ledger.verify()

    assert result["ok"] is False
    assert any(
        "malformed journal row" in error for error in result["devices"]["device-a"]["errors"]
    )


def test_native_write_policy_freezes_and_audits_override(tmp_cfg):
    memory = Memory(tmp_cfg)
    try:
        conflict = memory.operational.open_conflict(
            topic="billing architecture",
            summary="Two incompatible billing designs are active",
        )
        with pytest.raises(WriteRefused) as exc:
            memory.save(
                title="Billing architecture",
                content="Use the new invoice workflow.",
                defer_embed=True,
                auto_project=False,
            )
        assert exc.value.conflict["conflict_id"] == conflict.id

        record = memory.save(
            title="Billing architecture",
            content="Use the reviewed invoice workflow.",
            defer_embed=True,
            auto_project=False,
            allow_conflict_override=True,
            override_reason="Maintainer approved design B",
            actor=ActorIdentity(actor_id="maintainer", actor_kind="human"),
        )
        assert record.extra["write_policy"]["override"] is True
        assert record.extra["write_policy"]["reason"].startswith("human override:")
        assert memory.operational.ledger.verify()["ok"] is True
    finally:
        memory.close()


def test_save_normalizes_legacy_provenance_without_external_runtime(tmp_cfg):
    memory = Memory(tmp_cfg)
    try:
        record = memory.save(
            title="Native provenance",
            content="Memo owns its trace contract.",
            extra={
                "synapse_trace_id": "legacy-trace",
                "synapse_agent_id": "old-agent",
            },
            defer_embed=True,
            auto_project=False,
        )
        assert "synapse_trace_id" not in record.extra
        assert record.extra["provenance"]["trace_id"] == "legacy-trace"
        assert record.extra["provenance"]["actor_id"] == "old-agent"
    finally:
        memory.close()


def test_nested_legacy_provenance_is_normalized_and_cannot_elevate_trust(tmp_cfg):
    normalized = normalize_provenance(
        {
            "provenance": {
                "synapse_trace_id": "nested-trace",
                "synapse_agent_id": "nested-agent",
            }
        }
    )
    assert normalized == {
        "trace_id": "nested-trace",
        "actor_id": "nested-agent",
    }

    memory = Memory(tmp_cfg)
    try:
        owned = memory.save(
            title="Bound owner",
            content="Owner identity comes from the local authority.",
            extra={"owner_principal": "attacker"},
            defer_embed=True,
            auto_project=False,
        )
        assert owned.extra["owner_principal"] == tmp_cfg.device_id
        with pytest.raises(ValueError, match="actor ceiling"):
            memory.save(
                title="Forged authority",
                content="An agent cannot self-label as human.",
                extra={"trust_tier": "human"},
                defer_embed=True,
                auto_project=False,
            )
    finally:
        memory.close()


def test_evidence_pack_answers_or_abstains_explicitly(mock_memory):
    mock_memory.save(
        title="Invoice retry policy",
        content="Invoice retries use exponential backoff capped at five attempts.",
        type_="decision",
        defer_embed=True,
        auto_project=False,
        extra={"trust_tier": "human"},
        actor=ActorIdentity(actor_id="maintainer", actor_kind="human"),
    )
    answered = mock_memory.evidence_pack(
        "What invoice retry policy uses exponential backoff?",
        min_coverage=0.1,
    )
    assert answered.status is AnswerStatus.ANSWERED
    assert answered.items[0].uri.startswith("memo://memoria/")
    assert answered.claims[0]["evidence_uris"]

    abstained = mock_memory.evidence_pack(
        "What is the launch code for the lunar submarine?",
        min_coverage=0.8,
    )
    assert abstained.status is AnswerStatus.INSUFFICIENT_EVIDENCE
    assert abstained.abstention_reason


def test_independence_migration_rewrites_provenance_and_is_idempotent(
    tmp_cfg,
    monkeypatch,
):
    path = tmp_cfg.memory_dir / "legacy.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        frontmatter.dumps(
            frontmatter.Post(
                "Migrated body",
                id="a" * 32,
                title="Legacy integration",
                type="fact",
                tags=[],
                created="2026-01-01T00:00:00+00:00",
                updated="2026-01-01T00:00:00+00:00",
                extra={
                    "synapse_trace_id": "trace-old",
                    "synapse_agent_id": "agent-old",
                },
            )
        ),
        encoding="utf-8",
    )
    memory = Memory(tmp_cfg)
    monkeypatch.setattr(memory, "reindex", lambda: {})
    try:
        dry = migrate_independence(memory)
        assert dry["markdown"]["migrated"] == 1
        applied = migrate_independence(memory, write=True)
        assert applied["markdown"]["migrated"] == 1
        migrated = frontmatter.loads(path.read_text(encoding="utf-8"))
        assert migrated["extra"]["provenance"]["trace_id"] == "trace-old"
        assert "synapse_trace_id" not in migrated["extra"]
        again = migrate_independence(memory, write=True)
        assert again["markdown"]["migrated"] == 0
    finally:
        memory.close()


def test_independence_migration_handles_nested_provenance_and_prevalidates_ledger(
    tmp_cfg,
    monkeypatch,
    tmp_path,
):
    path = tmp_cfg.memory_dir / "nested.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        frontmatter.dumps(
            frontmatter.Post(
                "Body",
                id="b" * 32,
                title="Nested legacy",
                type="fact",
                tags=[],
                created="2026-01-01T00:00:00+00:00",
                updated="2026-01-01T00:00:00+00:00",
                extra={"provenance": {"synapse_trace_id": "nested-trace"}},
            )
        ),
        encoding="utf-8",
    )
    legacy = tmp_path / "legacy.jsonl"
    legacy.write_text('{"op":"one"}\n{bad-json}\n', encoding="utf-8")
    config = tmp_path / ".env"
    config.write_text('MEMO_CACHE_BACKEND="memflow"\n', encoding="utf-8")
    memory = Memory(tmp_cfg)
    monkeypatch.setattr(memory, "reindex", lambda: {})
    try:
        failed = migrate_independence(
            memory,
            write=True,
            legacy_ledger=legacy,
            config_paths=[config],
        )
        assert failed["legacy_ledger"]["errors"] == 1
        assert memory.operational.ledger.validated_events() == []
        migrated = frontmatter.loads(path.read_text(encoding="utf-8"))
        assert migrated["extra"]["provenance"]["trace_id"] == "nested-trace"
        assert "synapse_trace_id" not in migrated["extra"]["provenance"]
        assert config.read_text(encoding="utf-8") == 'MEMO_CACHE_BACKEND="vault"\n'
    finally:
        memory.close()


def test_independence_migration_legacy_ledger_retry_is_idempotent(tmp_cfg, tmp_path):
    legacy = tmp_path / "legacy.jsonl"
    legacy.write_text(
        '{"op":"focus.set","subject_uri":"memo://focus/memo"}\n'
        '{"op":"focus.clear","subject_uri":"memo://focus/memo"}\n',
        encoding="utf-8",
    )
    memory = Memory(tmp_cfg)
    try:
        first = migrate_independence(memory, write=True, legacy_ledger=legacy)
        assert first["legacy_ledger"]["imported"] == 2
        assert len(memory.operational.ledger.validated_events()) == 2

        stamp = tmp_cfg.state_dir / "independence-migration.json"
        stamp.unlink()
        retry = migrate_independence(memory, write=True, legacy_ledger=legacy)

        assert retry["legacy_ledger"]["imported"] == 0
        assert retry["legacy_ledger"]["skipped"] == 2
        assert len(memory.operational.ledger.validated_events()) == 2
        assert stamp.is_file()
    finally:
        memory.close()


def test_independence_migration_append_imports_only_new_rows(tmp_cfg, tmp_path):
    legacy = tmp_path / "append-only.jsonl"
    legacy.write_text(
        '{"op":"focus.set","subject_uri":"memo://focus/a"}\n'
        '{"op":"focus.set","subject_uri":"memo://focus/b"}\n',
        encoding="utf-8",
    )
    memory = Memory(tmp_cfg)
    try:
        first = migrate_independence(
            memory,
            write=True,
            legacy_ledger=legacy,
            config_paths=[],
        )
        assert first["legacy_ledger"]["imported"] == 2
        original_ids = [event.event_id for event in memory.operational.ledger.validated_events()]

        with legacy.open("a", encoding="utf-8") as handle:
            handle.write('{"op":"focus.set","subject_uri":"memo://focus/c"}\n')
        appended = migrate_independence(
            memory,
            write=True,
            legacy_ledger=legacy,
            config_paths=[],
        )

        events = memory.operational.ledger.validated_events()
        assert appended["legacy_ledger"] == {
            "checked": 3,
            "imported": 1,
            "errors": 0,
            "skipped": 2,
        }
        assert [event.event_id for event in events[:2]] == original_ids
        assert len(events) == 3
        assert len({event.event_id for event in events}) == 3
    finally:
        memory.close()


def test_independence_migration_identical_rows_use_stable_occurrences(tmp_cfg, tmp_path):
    legacy = tmp_path / "duplicates.jsonl"
    legacy.write_text(
        '{"op":"focus.set","subject_uri":"memo://focus/same"}\n'
        '{"subject_uri":"memo://focus/same","op":"focus.set"}\n',
        encoding="utf-8",
    )
    memory = Memory(tmp_cfg)
    try:
        first = migrate_independence(
            memory,
            write=True,
            legacy_ledger=legacy,
            config_paths=[],
        )
        assert first["legacy_ledger"]["imported"] == 2
        original_ids = {event.event_id for event in memory.operational.ledger.validated_events()}
        assert len(original_ids) == 2

        with legacy.open("a", encoding="utf-8") as handle:
            handle.write('{"op":"focus.set","subject_uri":"memo://focus/same"}\n')
        appended = migrate_independence(
            memory,
            write=True,
            legacy_ledger=legacy,
            config_paths=[],
        )

        events = memory.operational.ledger.validated_events()
        assert appended["legacy_ledger"]["imported"] == 1
        assert appended["legacy_ledger"]["skipped"] == 2
        assert len(events) == 3
        assert original_ids < {event.event_id for event in events}
        assert [event.payload["legacy_occurrence"] for event in events] == [1, 2, 3]
    finally:
        memory.close()


def test_conditional_operational_transitions_are_serialized(tmp_path):
    store = _legacy_store(tmp_path)
    handoff = store.create_handoff(
        project="memo",
        summary="consume once",
        from_actor="codex",
    )

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: store.consume_handoff(handoff.id), range(2)))

    assert sorted(results) == [False, True]
    consumes = [event for event in store.ledger.validated_events() if event.op == "handoff.consume"]
    assert len(consumes) == 1


def test_legacy_store_does_not_create_or_write_dormant_v2_generation(tmp_path):
    store = _legacy_store(tmp_path)

    store.set_focus(project="memo", summary="legacy remains authoritative")

    assert store.state()["focus"]["memo"]["summary"] == "legacy remains authoritative"
    assert not (tmp_path / "operational-v2").exists()


def test_mcp_surfaces_do_not_expose_human_override_or_owner_impersonation():
    memory = MagicMock()
    memory.cfg = SimpleNamespace(device_id="device-owner")
    record_server = _ToolServer()
    operational_server = _ToolServer()
    register_record_tools(record_server, memory)
    register_operational_tools(operational_server, memory)

    save_signature = inspect.signature(record_server.tools["memo_save"])
    assert "enforce_write_policy" not in save_signature.parameters
    assert "allow_conflict_override" not in save_signature.parameters
    assert "override_reason" not in save_signature.parameters
    memory.save.return_value.to_dict.return_value = {"id": "saved"}
    saved = record_server.tools["memo_save"](
        "agent content",
        extra={
            "trust_tier": "human",
            "visibility": "shared",
            "principals": ["attacker"],
            "owner_principal": "attacker",
        },
    )
    assert saved == {"id": "saved"}
    assert memory.save.call_args.kwargs["extra"] == {}

    resolve = operational_server.tools["memo_conflict_resolve"]
    assert resolve("conflict-one")["requires_human_cli"] is True
    memory.operational.resolve_conflict.assert_not_called()

    preview = operational_server.tools["memo_federation_preview"]
    denied = preview("device-owner")
    assert denied["error"] == "owner_preview_requires_local_cli"
    memory.federation.preview.assert_not_called()

    outcome_signature = inspect.signature(operational_server.tools["memo_outcome_record"])
    assert outcome_signature.parameters["idempotency_key"].default is inspect.Parameter.empty


def test_update_and_delete_enforce_native_policy_and_trust_ceiling(
    mem_with_stub,
    monkeypatch,
):
    memory = mem_with_stub
    record = memory.save(
        title="Protected design",
        content="Design A is active.",
        defer_embed=True,
        auto_project=False,
    )
    forged = dict(record.extra)
    forged["trust_tier"] = "human"
    with pytest.raises(ValueError, match="actor ceiling"):
        memory.update(record.id, extra=forged)

    conflict = memory.operational.open_conflict(
        topic="protected design",
        summary="Design A and B disagree",
    )
    with pytest.raises(WriteRefused):
        memory.update(record.id, content="Design B is active.")
    with pytest.raises(WriteRefused):
        memory.delete(record.id)

    memory.operational.resolve_conflict(
        conflict.id,
        resolution="maintainer selected B",
        actor=ActorIdentity(actor_id="maintainer", actor_kind="human"),
    )
    monkeypatch.setattr(
        memory.operational,
        "receipt",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("journal unavailable")),
    )
    updated = memory.update(record.id, content="Design B is active.")
    assert updated is not None
    assert updated.body == "Design B is active."
