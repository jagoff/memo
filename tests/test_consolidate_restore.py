"""Undoing a consolidation merge.

Consolidation archives the memories it absorbs. Until now that was a
one-way door: `_archive_memory` stripped the frontmatter `id`, so the
archived copy could not be moved back into the live tree — `reindex`
skips anything without a canonical id. `maintain`'s compaction has had
the symmetric `_restore_archived` all along; consolidation had nothing.

These tests pin the round trip: archive keeps enough provenance to
reverse itself, and `restore_archived` reverses it.
"""

from __future__ import annotations

import frontmatter
import pytest
from click.testing import CliRunner

from memo.cli import cli
from memo.consolidation import AdvancedConsolidator, MergeProposal


@pytest.fixture
def consolidator(mock_memory):
    return AdvancedConsolidator(mock_memory)


def _archived_post(consolidator, memory_id: str) -> frontmatter.Post:
    path = consolidator._archival_dir / f"{memory_id}.md"
    return frontmatter.loads(path.read_text(encoding="utf-8"))


def _save(mem, title: str, content: str, **kw):
    return mem.save(content=content, title=title, **kw)


# ── what the archive has to preserve ──────────────────────────────────────


def test_archived_copy_keeps_its_id(consolidator, mock_memory):
    """Without the id the archived file can never be reindexed back."""
    rec = _save(mock_memory, "Kept id", "body")
    keeper = _save(mock_memory, "Keeper", "other")

    assert consolidator._archive_memory(rec.id, keeper.id)

    assert _archived_post(consolidator, rec.id).get("id") == rec.id


def test_archived_copy_records_the_path_it_came_from(consolidator, mock_memory):
    """Restoring in place is only possible if the origin path survives."""
    rec = _save(mock_memory, "Origin path", "body", tags=["project:demo"])
    keeper = _save(mock_memory, "Keeper", "other", tags=["project:demo"])
    original_path = rec.path

    consolidator._archive_memory(rec.id, keeper.id)

    assert _archived_post(consolidator, rec.id).get("archived_from") == original_path


# ── the round trip ────────────────────────────────────────────────────────


def test_restore_brings_an_archived_memory_back_into_the_index(consolidator, mock_memory):
    rec = _save(mock_memory, "Comes back", "the body that must survive")
    keeper = _save(mock_memory, "Keeper", "other")
    consolidator._archive_memory(rec.id, keeper.id)
    assert mock_memory.get(rec.id) is None

    result = consolidator.restore_archived([rec.id])

    assert result.restored_ids == [rec.id]
    revived = mock_memory.get(rec.id)
    assert revived is not None
    assert revived.body.strip() == "the body that must survive"
    assert not (consolidator._archival_dir / f"{rec.id}.md").exists()


def test_restore_puts_the_file_back_where_it_was(consolidator, mock_memory):
    rec = _save(mock_memory, "Same slot", "body", tags=["project:demo"])
    keeper = _save(mock_memory, "Keeper", "other", tags=["project:demo"])
    original_path = rec.path
    consolidator._archive_memory(rec.id, keeper.id)

    consolidator.restore_archived([rec.id])

    assert mock_memory.get(rec.id).path == original_path


def test_restore_removes_the_archival_markers(consolidator, mock_memory):
    rec = _save(mock_memory, "Clean frontmatter", "body")
    keeper = _save(mock_memory, "Keeper", "other")
    consolidator._archive_memory(rec.id, keeper.id)

    consolidator.restore_archived([rec.id])

    restored = frontmatter.loads(
        (mock_memory.cfg.memory_dir / mock_memory.get(rec.id).path).read_text(encoding="utf-8")
    )
    assert "archived_for" not in restored.metadata
    assert "archived_at" not in restored.metadata
    assert "archived_from" not in restored.metadata


def test_restore_recovers_a_legacy_archive_that_lost_its_id(consolidator, mock_memory):
    """Files archived by the old code have no `id` — the filename holds it."""
    rec = _save(mock_memory, "Legacy archive", "body")
    keeper = _save(mock_memory, "Keeper", "other")
    consolidator._archive_memory(rec.id, keeper.id)
    archived_path = consolidator._archival_dir / f"{rec.id}.md"
    post = frontmatter.loads(archived_path.read_text(encoding="utf-8"))
    post.metadata.pop("id")
    post.metadata.pop("archived_from", None)
    archived_path.write_text(frontmatter.dumps(post), encoding="utf-8")

    result = consolidator.restore_archived([rec.id])

    assert result.restored_ids == [rec.id]
    assert mock_memory.get(rec.id) is not None


def test_restore_accepts_a_short_id_prefix(consolidator, mock_memory):
    """`consolidate list-archived` prints 8-char prefixes; they must work."""
    rec = _save(mock_memory, "Prefixed", "body")
    keeper = _save(mock_memory, "Keeper", "other")
    consolidator._archive_memory(rec.id, keeper.id)

    result = consolidator.restore_archived([rec.id[:8]])

    assert result.restored_ids == [rec.id]


def test_restore_reports_ids_it_could_not_find(consolidator):
    result = consolidator.restore_archived(["deadbeefdeadbeefdeadbeefdeadbeef"])

    assert result.restored_ids == []
    assert result.missing_ids == ["deadbeefdeadbeefdeadbeefdeadbeef"]


def test_one_unreadable_archive_does_not_sink_the_batch(consolidator, mock_memory):
    good = _save(mock_memory, "Readable", "body")
    broken = _save(mock_memory, "Corrupt", "body")
    keeper = _save(mock_memory, "Keeper", "other")
    consolidator._archive_memory(good.id, keeper.id)
    consolidator._archive_memory(broken.id, keeper.id)
    (consolidator._archival_dir / f"{broken.id}.md").write_text(
        "---\ntitle: [unterminated\n---\nbody\n", encoding="utf-8"
    )

    result = consolidator.restore_archived([good.id, broken.id])

    assert result.restored_ids == [good.id]
    assert result.missing_ids == [broken.id]
    assert mock_memory.get(good.id) is not None


def test_restore_does_not_overwrite_a_live_file_holding_the_old_path(consolidator, mock_memory):
    """A later save may have taken the slug back; the survivor wins."""
    rec = _save(mock_memory, "Contested slug", "original body")
    keeper = _save(mock_memory, "Keeper", "other")
    original_path = rec.path
    consolidator._archive_memory(rec.id, keeper.id)
    squatter = mock_memory.cfg.memory_dir / original_path
    squatter.parent.mkdir(parents=True, exist_ok=True)
    squatter.write_text("squatter", encoding="utf-8")

    result = consolidator.restore_archived([rec.id])

    assert result.restored_ids == [rec.id]
    assert squatter.read_text(encoding="utf-8") == "squatter"
    assert mock_memory.get(rec.id).path != original_path


def test_restore_gives_up_a_topic_slot_a_newer_record_took(consolidator, mock_memory):
    """`(namespace, topic_key)` is UNIQUE among live rows.

    Archiving soft-deletes the row, which frees its topic reservation, so a
    later save can claim it. Restoring the frontmatter verbatim would make
    reindex's un-delete violate that index — and reindex swallows the
    IntegrityError as a per-file warning, leaving the `.md` in the live tree
    permanently unindexed while the restore reports success.
    """
    rec = _save(mock_memory, "Deploy process", "the original", topic_key="deploy-process")
    keeper = _save(mock_memory, "Keeper", "other")
    consolidator._archive_memory(rec.id, keeper.id)
    newer = _save(mock_memory, "Deploy process v2", "the newer one", topic_key="deploy-process")

    result = consolidator.restore_archived([rec.id])

    assert result.restored_ids == [rec.id]
    assert result.unindexed_ids == []
    revived = mock_memory.get(rec.id)
    assert revived is not None, "restore reported success but the record is not indexed"
    assert revived.body.strip() == "the original"
    assert mock_memory.get(newer.id) is not None


def test_restore_reports_a_memory_the_index_refused(consolidator, mock_memory, monkeypatch):
    """A restore the index would not adopt must never be reported as done."""
    rec = _save(mock_memory, "Never indexed", "body")
    keeper = _save(mock_memory, "Keeper", "other")
    consolidator._archive_memory(rec.id, keeper.id)
    monkeypatch.setattr(mock_memory, "reindex", lambda **kw: {"added": 0})

    result = consolidator.restore_archived([rec.id])

    assert result.restored_ids == []
    assert result.unindexed_ids == [rec.id]
    assert "reindex --rebuild" in result.summary


def test_restore_dry_run_changes_nothing(consolidator, mock_memory):
    rec = _save(mock_memory, "Untouched", "body")
    keeper = _save(mock_memory, "Keeper", "other")
    consolidator._archive_memory(rec.id, keeper.id)

    result = consolidator.restore_archived([rec.id], dry_run=True)

    assert result.restored_ids == [rec.id]
    assert (consolidator._archival_dir / f"{rec.id}.md").exists()
    assert mock_memory.get(rec.id) is None


# ── undoing a whole merge ─────────────────────────────────────────────────


def _merge(consolidator, mock_memory, members, *, strategy="synthesis"):
    archived = [m.id for m in members[1:]] if strategy == "keep_latest" else [m.id for m in members]
    proposal = MergeProposal(
        cluster_id=1,
        memory_ids=[m.id for m in members],
        merged_title="Merged record",
        merged_body="merged body",
        merge_strategy=strategy,
        rationale="test",
        archived_ids=archived,
    )
    return consolidator.apply_merge(proposal)


def test_a_merged_record_records_the_ids_it_absorbed(consolidator, mock_memory):
    """Provenance is what tells an undo the record was created BY the merge."""
    a = _save(mock_memory, "Member A", "body a")
    b = _save(mock_memory, "Member B", "body b")

    outcome = _merge(consolidator, mock_memory, [a, b])

    merged = mock_memory.get(outcome.merged_id)
    assert sorted(merged.extra["consolidated_from"]) == sorted([a.id, b.id])


def test_restore_for_a_merge_brings_back_every_member(consolidator, mock_memory):
    a = _save(mock_memory, "Member A", "body a")
    b = _save(mock_memory, "Member B", "body b")
    outcome = _merge(consolidator, mock_memory, [a, b])

    result = consolidator.restore_archived(for_merged=outcome.merged_id)

    assert sorted(result.restored_ids) == sorted([a.id, b.id])
    assert mock_memory.get(a.id) is not None
    assert mock_memory.get(b.id) is not None


def test_dropping_the_merged_record_undoes_the_whole_merge(consolidator, mock_memory):
    a = _save(mock_memory, "Member A", "body a")
    b = _save(mock_memory, "Member B", "body b")
    outcome = _merge(consolidator, mock_memory, [a, b])

    result = consolidator.restore_archived(for_merged=outcome.merged_id, drop_merged=True)

    assert result.dropped_merged_id == outcome.merged_id
    assert mock_memory.get(outcome.merged_id) is None
    assert mock_memory.get(a.id) is not None


def test_drop_merged_refuses_a_record_the_merge_did_not_create(consolidator, mock_memory):
    """`keep_latest` keeps a pre-existing member — deleting it destroys data."""
    a = _save(mock_memory, "Older", "body a")
    b = _save(mock_memory, "Newer", "body b")
    outcome = _merge(consolidator, mock_memory, [a, b], strategy="keep_latest")
    survivor = outcome.merged_id

    result = consolidator.restore_archived(for_merged=survivor, drop_merged=True)

    assert result.dropped_merged_id is None
    assert mock_memory.get(survivor) is not None
    assert "not created by a merge" in result.summary


# ── the archived copy must stay out of every recovery path ────────────────


def test_topic_reservation_recovery_ignores_archived_files(consolidator, mock_memory):
    """Keeping the id must not let a disk-only recovery resurrect an archive."""
    rec = _save(
        mock_memory,
        "Reserved topic",
        "body",
        tags=["project:demo"],
        topic_key="the-reserved-topic",
    )
    keeper = _save(mock_memory, "Keeper", "other", tags=["project:demo"])
    consolidator._archive_memory(rec.id, keeper.id)

    recovered = mock_memory._recover_topic_reservation_locked(
        namespace="project:demo", topic_key="the-reserved-topic"
    )

    assert recovered == []


# ── CLI ───────────────────────────────────────────────────────────────────


def test_cli_consolidate_restore_reports_what_it_restored(consolidator, mock_memory, monkeypatch):
    rec = _save(mock_memory, "Via the CLI", "body")
    keeper = _save(mock_memory, "Keeper", "other")
    consolidator._archive_memory(rec.id, keeper.id)
    monkeypatch.setattr("memo.cli_consolidate._get_memory", lambda cfg: mock_memory)
    monkeypatch.setattr(
        "memo.cli_consolidate.Config.from_env", staticmethod(lambda: mock_memory.cfg)
    )

    result = CliRunner().invoke(
        cli,
        ["consolidate", "restore", rec.id],
        env={"MEMO_NONINTERACTIVE": "1"},
    )

    assert result.exit_code == 0, result.output
    assert rec.id[:8] in result.output
    assert mock_memory.get(rec.id) is not None


def test_cli_consolidate_restore_refuses_ids_together_with_a_merge(mock_memory, monkeypatch):
    """`--for` ignores positional ids; refuse rather than silently drop them."""
    monkeypatch.setattr("memo.cli_consolidate._get_memory", lambda cfg: mock_memory)
    monkeypatch.setattr(
        "memo.cli_consolidate.Config.from_env", staticmethod(lambda: mock_memory.cfg)
    )

    result = CliRunner().invoke(
        cli,
        ["consolidate", "restore", "deadbeefdeadbeefdeadbeefdeadbeef", "--for", "cafebabe"],
        env={"MEMO_NONINTERACTIVE": "1"},
    )

    assert result.exit_code != 0
    assert "--for" in result.output


def test_cli_consolidate_restore_needs_ids_or_a_merge(mock_memory, monkeypatch):
    monkeypatch.setattr("memo.cli_consolidate._get_memory", lambda cfg: mock_memory)
    monkeypatch.setattr(
        "memo.cli_consolidate.Config.from_env", staticmethod(lambda: mock_memory.cfg)
    )

    result = CliRunner().invoke(cli, ["consolidate", "restore"], env={"MEMO_NONINTERACTIVE": "1"})

    assert result.exit_code != 0
    assert "--for" in result.output
