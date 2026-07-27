"""End-to-end trust invariants adopted from the Engram architecture review."""

from __future__ import annotations

import json
import multiprocessing
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from memo.config import Config
from memo.errors import IdentityConflictError
from memo.memory import Memory


def _process_exact_save(
    data_dir: str,
    vault_path: str,
    state_dir: str,
    embedder_dims: int,
    barrier: Any,
    queue: Any,
) -> None:
    """Spawn-safe worker exercising the real cross-process authority lock."""
    memory = Memory(
        Config(
            data_dir=Path(data_dir),
            vault_path=Path(vault_path),
            state_dir=Path(state_dir),
            embedder_dims=embedder_dims,
            reranker_enabled=False,
        )
    )
    try:
        barrier.wait(timeout=15)
        record = memory.save(
            content="one canonical concurrent fact",
            title="Concurrent exact fact",
            auto_project=False,
            defer_embed=True,
        )
        queue.put((record.id, record.action, record.index_pending))
    except Exception as exc:  # pragma: no cover - surfaced in parent assertion
        queue.put(("ERROR", type(exc).__name__, str(exc)))
    finally:
        memory.close()


def _file_snapshot(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()
    }


def test_update_never_runs_model_work_under_authority_lock(
    mem_with_stub: Memory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = mem_with_stub.save(
        content="original update body",
        title="Update lock probe",
        auto_project=False,
    )
    real_embed = mem_with_stub._embed_cached
    observed_depths: list[int] = []

    def _observe_embed(text: str, *, ctx: str) -> list[float]:
        observed_depths.append(mem_with_stub._data_lock_depth)
        return real_embed(text, ctx=ctx)

    monkeypatch.setattr(mem_with_stub, "_embed_cached", _observe_embed)

    updated = mem_with_stub.update(record.id, content="changed update body")

    assert updated is not None
    assert observed_depths
    assert observed_depths == [0] * len(observed_depths)


def test_topic_revision_never_runs_model_work_under_authority_lock(
    mem_with_stub: Memory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mem_with_stub.save(
        content="original topic body",
        title="Topic lock probe",
        topic_key="topic-lock-probe",
        auto_project=False,
    )
    real_embed = mem_with_stub._embed_cached
    observed_depths: list[int] = []

    def _observe_embed(text: str, *, ctx: str) -> list[float]:
        observed_depths.append(mem_with_stub._data_lock_depth)
        return real_embed(text, ctx=ctx)

    monkeypatch.setattr(mem_with_stub, "_embed_cached", _observe_embed)

    revised = mem_with_stub.save(
        content="revised topic body",
        title="Topic lock probe",
        topic_key="topic-lock-probe",
        auto_project=False,
    )

    assert revised.action == "revised"
    assert observed_depths
    assert observed_depths == [0] * len(observed_depths)


def test_direct_save_redacts_before_any_persistence(
    mem_with_stub: Memory,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    token = "sk-" + "TrustBoundaryCanary1234567890"
    private_canary = "PRIVATE_CANARY_MUST_NOT_PERSIST"
    receipts: list[tuple[str, dict[str, Any]]] = []

    monkeypatch.setattr(
        mem_with_stub.operational,
        "receipt",
        lambda operation, **kwargs: receipts.append((operation, kwargs)),
    )
    record = mem_with_stub.save(
        content=f"public {token} <private>{private_canary}</private> tail",
        title=f"credential {token}",
        tags=[f"tag-{token}"],
        topic_key=f"topic-{token}",
        normalized_hash=f"legacy-{token}",
        extra={f"key-{token}": {"nested": [token, f"<private>{private_canary}</private>"]}},
        auto_project=False,
    )

    assert "_redacted" in record.tags
    forbidden = (token.encode(), private_canary.encode())
    roots = (mem_with_stub.cfg.data_dir, mem_with_stub.cfg.state_dir)
    persisted = [path.read_bytes() for root in roots for path in root.rglob("*") if path.is_file()]
    assert all(secret not in payload for secret in forbidden for payload in persisted)
    emitted = json.dumps(receipts, default=str) + caplog.text
    assert token not in emitted
    assert private_canary not in emitted


def test_same_topic_key_is_isolated_by_namespace(mem_with_stub: Memory) -> None:
    records = [
        mem_with_stub.save(
            content="project alpha fact",
            title="Shared topic",
            topic_key="Shared  Topic",
            tags=["project:alpha"],
        ),
        mem_with_stub.save(
            content="project beta fact",
            title="Shared topic",
            topic_key="shared topic",
            tags=["project:beta"],
        ),
        mem_with_stub.save(
            content="global fact",
            title="Shared topic",
            topic_key="shared topic",
            auto_project=False,
        ),
        mem_with_stub.save(
            content="unscoped fact",
            title="Shared topic",
            topic_key="shared topic",
            auto_project=True,
        ),
    ]

    assert len({record.id for record in records}) == 4
    assert records[2].path.split("/", 1)[0] != "_unscoped"
    assert records[3].path.split("/", 1)[0] == "_unscoped"
    assert mem_with_stub.store.count() == 4


def test_multiple_project_tags_fail_without_mutation(mem_with_stub: Memory) -> None:
    before_disk = _file_snapshot(mem_with_stub.cfg.data_dir)
    before_count = mem_with_stub.store.count()

    with pytest.raises(IdentityConflictError) as raised:
        mem_with_stub.save(
            content="ambiguous project ownership",
            title="Ambiguous",
            tags=["project:alpha", "project:beta"],
        )

    assert raised.value.kind == "ambiguous_namespace"
    assert mem_with_stub.store.count() == before_count
    assert _file_snapshot(mem_with_stub.cfg.data_dir) == before_disk


def test_exact_duplicate_corroborates_one_record_without_embedding(
    mem_with_stub: Memory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = mem_with_stub.save(
        content="same durable observation",
        title="Exact identity",
        auto_project=False,
    )
    calls = 0

    def _unexpected_embed(_query: str) -> list[float]:
        nonlocal calls
        calls += 1
        return [1.0, 0.0, 0.0, 0.0]

    monkeypatch.setattr(mem_with_stub.embedder, "embed_query", _unexpected_embed)
    history_before = mem_with_stub.history.list_recent(limit=100, record_id=first.id)
    second = mem_with_stub.save(
        content="same durable observation\n",
        title="  EXACT   identity ",
        auto_project=False,
    )

    assert second.id == first.id
    assert first.action == "created"
    assert second.action == "corroborated"
    assert calls == 0
    assert len(list(mem_with_stub.cfg.memory_dir.rglob("*.md"))) == 1
    assert mem_with_stub.store.get_support_batch([first.id]) == {first.id: 1}
    assert mem_with_stub.history.list_recent(limit=100, record_id=first.id) == history_before


@pytest.mark.parametrize(
    ("second_kwargs",),
    [
        ({"tags": ["project:other"]},),
        ({"type_": "preference"},),
        ({"title": "Different title"},),
        ({"content": "different body"},),
    ],
)
def test_exact_identity_changes_create_distinct_records(
    mem_with_stub: Memory, second_kwargs: dict[str, Any]
) -> None:
    base: dict[str, Any] = {
        "content": "base body",
        "title": "Base title",
        "type_": "note",
        "tags": ["project:base"],
    }
    first = mem_with_stub.save(**base)
    second = mem_with_stub.save(**(base | second_kwargs))
    assert second.id != first.id
    assert second.action == "created"


def test_topic_precedence_rules(mem_with_stub: Memory) -> None:
    first = mem_with_stub.save(
        content="version one",
        title="Versioned topic",
        topic_key="release-state",
        auto_project=False,
    )
    revised = mem_with_stub.save(
        content="version two",
        title="Versioned topic",
        topic_key="release-state",
        auto_project=False,
    )
    assert revised.id == first.id
    assert revised.path == first.path
    assert revised.created == first.created
    assert revised.action == "revised"
    assert revised.body == "version two"
    versions = mem_with_stub.versioning.get_version_history(first.id, limit=10)
    assert len(versions) == 1
    assert versions[0].body == "version one"

    unkeyed = mem_with_stub.save(
        content="attach topic later",
        title="Attachable",
        auto_project=False,
    )
    attached = mem_with_stub.save(
        content="attach topic later",
        title="Attachable",
        topic_key="first-key",
        auto_project=False,
    )
    assert attached.id == unkeyed.id
    assert attached.action == "revised"
    assert mem_with_stub.store.get_identity_keys(unkeyed.id)["topic_key"] == "first-key"

    corroborated = mem_with_stub.save(
        content="attach topic later",
        title="Attachable",
        auto_project=False,
    )
    assert corroborated.id == unkeyed.id
    assert corroborated.action == "corroborated"
    assert mem_with_stub.store.get_identity_keys(unkeyed.id)["topic_key"] == "first-key"


def test_exact_identity_with_different_explicit_topic_is_conflict(mem_with_stub: Memory) -> None:
    first = mem_with_stub.save(
        content="one immutable fact",
        title="Explicit identity",
        topic_key="topic-one",
        auto_project=False,
    )
    before = _file_snapshot(mem_with_stub.cfg.data_dir)

    with pytest.raises(IdentityConflictError) as raised:
        mem_with_stub.save(
            content="one immutable fact",
            title="Explicit identity",
            topic_key="topic-two",
            auto_project=False,
        )

    assert raised.value.kind == "exact_identity_topic_mismatch"
    assert mem_with_stub.store.count() == 1
    assert mem_with_stub.get(first.id) is not None
    assert _file_snapshot(mem_with_stub.cfg.data_dir) == before


def test_update_recomputes_identity_and_rejects_occupied_topic_without_mutation(
    mem_with_stub: Memory,
) -> None:
    alpha = mem_with_stub.save(
        content="alpha body",
        title="Alpha",
        topic_key="occupied",
        tags=["project:alpha"],
    )
    beta = mem_with_stub.save(
        content="beta body",
        title="Beta",
        topic_key="occupied",
        tags=["project:beta"],
    )
    beta_path = mem_with_stub.cfg.memory_dir / beta.path
    before_markdown = beta_path.read_bytes()
    before_row = mem_with_stub.store.get(beta.id)
    before_history = mem_with_stub.history.list_recent(limit=100, record_id=beta.id)
    before_versions = mem_with_stub.versioning.get_version_history(beta.id, limit=100)
    before_support = mem_with_stub.store.get_support_batch([alpha.id, beta.id])

    with pytest.raises(IdentityConflictError) as raised:
        mem_with_stub.update(beta.id, tags=["project:alpha"])

    assert raised.value.kind == "update_topic_identity_conflict"
    assert beta_path.read_bytes() == before_markdown
    assert mem_with_stub.store.get(beta.id) == before_row
    assert mem_with_stub.history.list_recent(limit=100, record_id=beta.id) == before_history
    assert mem_with_stub.versioning.get_version_history(beta.id, limit=100) == before_versions
    assert mem_with_stub.store.get_support_batch([alpha.id, beta.id]) == before_support

    updated = mem_with_stub.update(
        beta.id,
        title="Beta revised",
        type_="preference",
        content="beta body revised",
    )
    assert updated is not None
    identity = mem_with_stub.store.get_identity_keys(beta.id)
    assert identity["normalized_title"] == "beta revised"
    assert (
        identity["normalized_content_hash"]
        != mem_with_stub.store.get_identity_keys(alpha.id)["normalized_content_hash"]
    )


def test_corroboration_transaction_failure_rolls_back_all_signals(
    mem_with_stub: Memory,
) -> None:
    first = mem_with_stub.save(
        content="atomic corroboration",
        title="Atomic evidence",
        auto_project=False,
    )
    path = mem_with_stub.cfg.memory_dir / first.path
    before_markdown = path.read_bytes()
    before_row = mem_with_stub.store.get(first.id)
    before_support = mem_with_stub.store.get_support_batch([first.id])
    before_history = mem_with_stub.history.list_recent(limit=100, record_id=first.id)
    with mem_with_stub.store._tx() as connection:
        connection.execute(
            "CREATE TRIGGER fail_identity_support BEFORE UPDATE OF support_count "
            "ON memory_health BEGIN SELECT RAISE(ABORT, 'injected support failure'); END"
        )

    with pytest.raises(sqlite3.IntegrityError, match="injected support failure"):
        mem_with_stub.save(
            content="atomic corroboration",
            title="Atomic evidence",
            auto_project=False,
        )

    assert path.read_bytes() == before_markdown
    assert mem_with_stub.store.get(first.id) == before_row
    assert mem_with_stub.store.get_support_batch([first.id]) == before_support
    assert mem_with_stub.history.list_recent(limit=100, record_id=first.id) == before_history


def test_topic_revision_index_failure_rolls_back_markdown_history_and_version(
    mem_with_stub: Memory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = mem_with_stub.save(
        content="revision before",
        title="Revision rollback",
        topic_key="rollback-topic",
        auto_project=False,
    )
    path = mem_with_stub.cfg.memory_dir / first.path
    before_markdown = path.read_bytes()
    before_row = mem_with_stub.store.get(first.id)
    before_history = mem_with_stub.history.list_recent(limit=100, record_id=first.id)
    before_versions = mem_with_stub.versioning.get_version_history(first.id, limit=100)
    monkeypatch.setattr(
        mem_with_stub.store,
        "upsert",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("injected revision upsert")),
    )

    with pytest.raises(RuntimeError, match="injected revision upsert"):
        mem_with_stub.save(
            content="revision after",
            title="Revision rollback",
            topic_key="rollback-topic",
            auto_project=False,
        )

    assert path.read_bytes() == before_markdown
    assert mem_with_stub.store.get(first.id) == before_row
    assert mem_with_stub.store.get_fts_body(first.id) == "revision before"
    assert mem_with_stub.history.list_recent(limit=100, record_id=first.id) == before_history
    assert mem_with_stub.versioning.get_version_history(first.id, limit=100) == before_versions


def test_receipt_failure_does_not_turn_committed_save_into_failure(
    mem_with_stub: Memory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        mem_with_stub.operational,
        "receipt",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("receipt unavailable")),
    )

    record = mem_with_stub.save(
        content="receipt-independent canonical body",
        title="Receipt failure",
        auto_project=False,
    )

    assert record.action == "created"
    assert mem_with_stub.get(record.id) is not None
    assert (mem_with_stub.cfg.memory_dir / record.path).is_file()


def test_threaded_exact_saves_create_one_file_and_n_minus_one_support(tmp_cfg: Config) -> None:
    memories = [Memory(tmp_cfg) for _ in range(6)]

    def _save(memory: Memory):
        return memory.save(
            content="threaded exact observation",
            title="Thread exact",
            auto_project=False,
            defer_embed=True,
        )

    try:
        with ThreadPoolExecutor(max_workers=len(memories)) as pool:
            records = list(pool.map(_save, memories))
        canonical = records[0].id
        assert {record.id for record in records} == {canonical}
        assert [record.action for record in records].count("created") == 1
        assert [record.action for record in records].count("corroborated") == 5
        assert memories[0].store.get_support_batch([canonical]) == {canonical: 5}
        assert len(list(tmp_cfg.memory_dir.rglob("*.md"))) == 1
    finally:
        for memory in memories:
            memory.close()


def test_process_exact_saves_create_one_file_and_n_minus_one_support(
    tmp_cfg: Config, monkeypatch
) -> None:
    # Initialize schema before workers race only the persistence decision.
    Memory(tmp_cfg).close()
    # ``pytest``'s console entry point can omit the repository root from
    # ``sys.path``. Spawned children must be able to re-import this test module.
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    ctx = multiprocessing.get_context("spawn")
    worker_count = 3
    barrier = ctx.Barrier(worker_count)
    queue = ctx.Queue()
    processes = [
        ctx.Process(
            target=_process_exact_save,
            args=(
                str(tmp_cfg.data_dir),
                str(tmp_cfg.vault_path),
                str(tmp_cfg.state_dir),
                tmp_cfg.embedder_dims,
                barrier,
                queue,
            ),
        )
        for _ in range(worker_count)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=30)
        assert not process.is_alive()
        assert process.exitcode == 0
    results = [queue.get(timeout=5) for _ in range(worker_count)]
    assert all(result[0] != "ERROR" for result in results), results

    memory = Memory(tmp_cfg)
    try:
        canonical = results[0][0]
        assert {result[0] for result in results} == {canonical}
        assert [result[1] for result in results].count("created") == 1
        assert [result[1] for result in results].count("corroborated") == worker_count - 1
        assert memory.store.get_support_batch([canonical]) == {canonical: worker_count - 1}
        assert len(list(tmp_cfg.memory_dir.rglob("*.md"))) == 1
    finally:
        memory.close()
