from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from memo.durable_outbox import canonical_save_request_hash
from memo.errors import IdentityConflictError, ValidationError


def _save_kwargs(content: str = "Verify the journal before syncing.") -> dict[str, object]:
    return {
        "content": content,
        "title": "Verify journal",
        "type_": "procedure",
        "tags": ["procedural", "outcome-backed"],
        "extra": {
            "provenance": {
                "actor_id": "memo",
                "route_reason": "procedural_promotion",
                "evidence_uris": ["memo://memoria/source-1"],
            }
        },
        "auto_project": False,
        "enforce_write_policy": False,
    }


def _markdown_files(memory_dir: Path) -> list[Path]:
    return sorted(path for path in memory_dir.rglob("*.md") if path.is_file())


def test_same_operation_and_request_return_the_same_memory(mem_with_stub) -> None:
    kwargs = _save_kwargs()
    request_hash = canonical_save_request_hash(kwargs)

    first = mem_with_stub.save_operation(
        operation_key="promotion/" + "a" * 64,
        request_hash=request_hash,
        save_kwargs=kwargs,
    )
    replay = mem_with_stub.save_operation(
        operation_key="promotion/" + "a" * 64,
        request_hash=request_hash,
        save_kwargs=kwargs,
    )

    assert replay.id == first.id
    assert replay.extra["_memo_operation"] == {
        "operation_key": "promotion/" + "a" * 64,
        "request_hash": request_hash,
    }
    assert len(_markdown_files(mem_with_stub.cfg.memory_dir)) == 1


def test_same_operation_with_changed_request_raises_identity_conflict(
    mem_with_stub,
) -> None:
    original = _save_kwargs()
    changed = _save_kwargs("Different body")
    operation_key = "promotion/" + "b" * 64
    mem_with_stub.save_operation(
        operation_key=operation_key,
        request_hash=canonical_save_request_hash(original),
        save_kwargs=original,
    )

    with pytest.raises(IdentityConflictError) as raised:
        mem_with_stub.save_operation(
            operation_key=operation_key,
            request_hash=canonical_save_request_hash(changed),
            save_kwargs=changed,
        )

    assert raised.value.kind == "durable_operation"
    assert len(_markdown_files(mem_with_stub.cfg.memory_dir)) == 1


def test_request_hash_must_match_complete_save_kwargs(mem_with_stub) -> None:
    with pytest.raises(ValidationError, match="request_hash"):
        mem_with_stub.save_operation(
            operation_key="promotion/" + "c" * 64,
            request_hash="0" * 64,
            save_kwargs=_save_kwargs(),
        )

    assert _markdown_files(mem_with_stub.cfg.memory_dir) == []


def test_caller_cannot_supply_reserved_operation_frontmatter(mem_with_stub) -> None:
    kwargs = _save_kwargs()
    kwargs["extra"] = {
        "_memo_operation": {
            "operation_key": "promotion/" + "9" * 64,
            "request_hash": "8" * 64,
        }
    }

    with pytest.raises(ValidationError, match="reserved"):
        mem_with_stub.save_operation(
            operation_key="promotion/" + "7" * 64,
            request_hash=canonical_save_request_hash(kwargs),
            save_kwargs=kwargs,
        )

    assert _markdown_files(mem_with_stub.cfg.memory_dir) == []


def test_markdown_without_sqlite_row_is_recovered_not_duplicated(mem_with_stub) -> None:
    kwargs = _save_kwargs()
    operation_key = "promotion/" + "d" * 64
    request_hash = canonical_save_request_hash(kwargs)
    first = mem_with_stub.save_operation(
        operation_key=operation_key,
        request_hash=request_hash,
        save_kwargs=kwargs,
    )
    assert mem_with_stub.store.delete(first.id) is True
    assert mem_with_stub.get(first.id) is None

    recovered = mem_with_stub.save_operation(
        operation_key=operation_key,
        request_hash=request_hash,
        save_kwargs=kwargs,
    )

    assert recovered.id == first.id
    assert mem_with_stub.get(first.id) is not None
    assert len(_markdown_files(mem_with_stub.cfg.memory_dir)) == 1


def test_embed_pending_operation_replay_keeps_one_markdown(mem_with_stub) -> None:
    kwargs = {**_save_kwargs(), "defer_embed": True}
    operation_key = "promotion/" + "e" * 64
    request_hash = canonical_save_request_hash(kwargs)

    first = mem_with_stub.save_operation(
        operation_key=operation_key,
        request_hash=request_hash,
        save_kwargs=kwargs,
    )
    replay = mem_with_stub.save_operation(
        operation_key=operation_key,
        request_hash=request_hash,
        save_kwargs=kwargs,
    )

    assert replay.id == first.id
    assert replay.index_pending is True
    assert len(_markdown_files(mem_with_stub.cfg.memory_dir)) == 1


def test_multiple_markdown_claims_for_operation_fail_closed(mem_with_stub) -> None:
    kwargs = _save_kwargs()
    operation_key = "promotion/" + "f" * 64
    request_hash = canonical_save_request_hash(kwargs)
    mem_with_stub.save_operation(
        operation_key=operation_key,
        request_hash=request_hash,
        save_kwargs=kwargs,
    )
    original = _markdown_files(mem_with_stub.cfg.memory_dir)[0]
    duplicate = original.with_name("duplicate-operation-claim.md")
    shutil.copy2(original, duplicate)

    with pytest.raises(IdentityConflictError) as raised:
        mem_with_stub.find_by_operation_key(operation_key, request_hash)

    assert raised.value.kind == "durable_operation"


@pytest.mark.parametrize("operation_key", ("", "plain-key", "promotion/not-a-hash"))
def test_operation_key_must_use_the_promotion_sha256_namespace(
    mem_with_stub,
    operation_key: str,
) -> None:
    kwargs = _save_kwargs()

    with pytest.raises(ValidationError, match="operation_key"):
        mem_with_stub.save_operation(
            operation_key=operation_key,
            request_hash=canonical_save_request_hash(kwargs),
            save_kwargs=kwargs,
        )
