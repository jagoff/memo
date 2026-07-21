"""Tests for sync & backup module."""

import fcntl
import hashlib
import json
import shutil
import sqlite3
import stat
import tarfile
import threading
from contextlib import closing
from pathlib import Path

import pytest

from memo.config import Config
from memo.history import HistoryStore
from memo.memory import Memory
from memo.sync import (
    BackupManager,
    BackupMetadata,
    SyncDiff,
    SyncManager,
)


def _build_remote_memory(root) -> Memory:
    """An isolated `Memory` (own data/state dir → distinct device id) with a
    deterministic stub embedder, standing in for another machine."""
    data = root / "data"
    state = root / "state"
    data.mkdir(parents=True)
    state.mkdir(parents=True)
    cfg = Config(data_dir=data, state_dir=state, reranker_enabled=False)
    mem = Memory(cfg)

    def _fake_embedding(text: str) -> list[float]:
        digest = hashlib.sha256((text or "").encode("utf-8")).digest()
        values = [((digest[i % len(digest)] / 255.0) * 2.0) - 1.0 for i in range(cfg.embedder_dims)]
        norm = sum(v * v for v in values) ** 0.5
        return [v / norm for v in values]

    mem.embedder.embed = lambda inputs: [_fake_embedding(t) for t in inputs]
    mem.embedder.embed_query = lambda q: _fake_embedding(q)
    return mem


def _copy_md(remote: Memory, local: Memory, rel_path: str) -> None:
    """Copy a memoria's .md from remote → local memory dir, preserving subdirs
    (stands in for the file-sync layer, e.g. iCloud)."""
    src = remote.cfg.memory_dir / rel_path
    dst = local.cfg.memory_dir / rel_path
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, dst)


def test_sync_replay_save_preserves_record_id_no_duplicate(sync_manager, mock_memory, tmp_path):
    """Replaying a remote `save` must index the existing `{id}.md` under its
    own id — not mint a fresh uuid (which forks a duplicate memoria). B1."""
    remote = _build_remote_memory(tmp_path / "remote")
    try:
        rec = remote.save(content="A durable fact about the trinity.", title="Fact", tags=["t"])
        # Simulate the .md having already synced to the local machine (iCloud).
        _copy_md(remote, mock_memory, rec.path)

        diff = sync_manager.sync_from_remote(remote.cfg.history_db)

        assert diff.applied == 1
        assert mock_memory.get(rec.id) is not None, "synced record must keep its id"
        rows = mock_memory.store.list_recent(limit=100)
        assert len(rows) == 1, "must not create a duplicate under a new uuid"
        assert rows[0]["id"] == rec.id
    finally:
        remote.close()


def test_sync_cursor_not_wedged_by_missing_then_present_file(sync_manager, mock_memory, tmp_path):
    """A remote save whose file hasn't synced yet must defer (cursor stays put,
    order preserved) and apply on a later pass — never permanently skipped. B2."""
    remote = _build_remote_memory(tmp_path / "remote")
    try:
        rec = remote.save(content="Deferred fact.", title="Deferred", tags=["t"])

        # File not present locally yet → first pass applies nothing, advances nothing.
        diff1 = sync_manager.sync_from_remote(remote.cfg.history_db)
        assert diff1.applied == 0
        assert mock_memory.get(rec.id) is None

        # File arrives; a second pass must still see and apply the event.
        _copy_md(remote, mock_memory, rec.path)
        diff2 = sync_manager.sync_from_remote(remote.cfg.history_db)
        assert diff2.applied == 1
        assert mock_memory.get(rec.id) is not None
    finally:
        remote.close()


def test_sync_processes_large_backlog_from_oldest_event_forward(
    sync_manager, mock_memory, tmp_path
):
    remote_path = tmp_path / "remote-history.db"
    remote = HistoryStore(remote_path, device_id="remote-device")
    try:
        for index in range(1001):
            remote.log_delete(
                ts=f"2026-01-01T00:00:{index:04d}Z",
                record_id=f"{index:032x}",
                title=f"Remote {index}",
                type_="note",
            )
    finally:
        remote.close()

    first = sync_manager.sync_from_remote(remote_path)

    assert first.applied == 1000
    assert mock_memory.history.get_sync_state("remote-device") == 1000

    second = sync_manager.sync_from_remote(remote_path)

    assert second.applied == 1
    assert mock_memory.history.get_sync_state("remote-device") == 1001


@pytest.fixture
def backup_manager(tmp_cfg):
    """Fixture providing BackupManager instance."""
    return BackupManager(
        memory_dir=tmp_cfg.memory_dir,
        db_dir=tmp_cfg.state_dir,
        backup_dir=tmp_cfg.state_dir / "backups",
    )


@pytest.fixture
def sync_manager(mock_memory):
    """Fixture providing SyncManager instance."""
    return SyncManager(mock_memory)


def test_backup_manager_init(backup_manager):
    """Test BackupManager initialization."""
    assert backup_manager.memory_dir.is_dir()
    assert backup_manager.backup_dir.is_dir()


def test_backup_manager_lock_file_is_not_truncated(backup_manager):
    lock_file = backup_manager.backup_dir / ".backup.lock"
    lock_file.write_text("existing lock contents", encoding="utf-8")

    backup_manager.create_backup(compress=False, name="keeps-lock")

    assert lock_file.read_text(encoding="utf-8") == "existing lock contents"


def test_backup_manager_rejects_symlink_lock_without_touching_target(
    backup_manager, tmp_path: Path
):
    target = tmp_path / "lock-target.txt"
    target.write_text("must survive", encoding="utf-8")
    (backup_manager.backup_dir / ".backup.lock").symlink_to(target)

    with pytest.raises(RuntimeError, match="backup lock"):
        backup_manager.create_backup(compress=False, name="unsafe-lock")

    assert target.read_text(encoding="utf-8") == "must survive"


def test_backup_manager_create_backup(backup_manager, mock_memory):
    """Test creating a backup."""
    # Create a test memoria
    mock_memory.save(
        content="Test content",
        title="Test",
        tags=["test"],
    )

    metadata = backup_manager.create_backup(compress=False)

    assert metadata.memory_count > 0
    assert metadata.checksum
    assert metadata.original_size > 0


def test_backup_manager_create_compressed_backup(backup_manager, mock_memory):
    """Test creating a compressed backup."""
    mock_memory.save(
        content="Test content",
        title="Test",
        tags=["test"],
    )

    metadata = backup_manager.create_backup(compress=True)

    assert metadata.compressed_size > 0
    assert metadata.compressed_size < metadata.original_size


def test_backup_manager_compressed_archive_is_private(backup_manager, mock_memory):
    mock_memory.save(content="private backup content", title="Private")

    backup_manager.create_backup(compress=True, name="private")

    archive = backup_manager.backup_dir / "private.tar.gz"
    assert stat.S_IMODE(archive.stat().st_mode) == 0o600


def test_backup_manager_compressed_archive_is_published_atomically(backup_manager, monkeypatch):
    source = backup_manager.memory_dir / "atomic.md"
    source.write_text("atomic backup body", encoding="utf-8")
    archive = backup_manager.backup_dir / "atomic.tar.gz"
    adding = threading.Event()
    resume = threading.Event()
    errors: list[BaseException] = []
    original_add = tarfile.TarFile.add

    def paused_add(self, *args, **kwargs):
        adding.set()
        assert resume.wait(5)
        return original_add(self, *args, **kwargs)

    monkeypatch.setattr(tarfile.TarFile, "add", paused_add)

    def create() -> None:
        try:
            backup_manager.create_backup(compress=True, name="atomic")
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=create)
    worker.start()
    try:
        assert adding.wait(5)
        assert not archive.exists()
    finally:
        resume.set()
        worker.join(5)

    assert not worker.is_alive()
    assert errors == []
    assert archive.is_file()


def test_backup_manager_compressed_archive_never_clobbers_racing_destination(
    backup_manager,
    monkeypatch,
):
    """A non-cooperating writer cannot be overwritten at publication time."""
    source = backup_manager.memory_dir / "no-clobber.md"
    source.write_text("backup body", encoding="utf-8")
    archive = backup_manager.backup_dir / "no-clobber.tar.gz"
    sentinel = b"pre-existing backup"
    original_exists = Path.exists
    archive_checks = 0

    def create_destination_after_check(path: Path) -> bool:
        nonlocal archive_checks
        existed = original_exists(path)
        if path == archive:
            archive_checks += 1
            if archive_checks == 2 and not existed:
                archive.write_bytes(sentinel)
        return existed

    monkeypatch.setattr(Path, "exists", create_destination_after_check)

    with pytest.raises(FileExistsError):
        backup_manager.create_backup(compress=True, name="no-clobber")

    assert archive.read_bytes() == sentinel


def test_backup_manager_list_waits_for_backup_creation(backup_manager, monkeypatch):
    source = backup_manager.memory_dir / "listed.md"
    source.write_text("listed backup body", encoding="utf-8")
    populated = threading.Event()
    resume = threading.Event()
    listed = threading.Event()
    results: list[BackupMetadata] = []
    errors: list[BaseException] = []
    original_populate = backup_manager._populate_backup
    original_acquire = backup_manager._acquire_lock
    shared_lock_attempted = threading.Event()

    def paused_populate(*args, **kwargs):
        result = original_populate(*args, **kwargs)
        populated.set()
        assert resume.wait(5)
        return result

    def observed_acquire(fh, *, exclusive=True):
        if not exclusive:
            shared_lock_attempted.set()
        return original_acquire(fh, exclusive=exclusive)

    monkeypatch.setattr(backup_manager, "_populate_backup", paused_populate)
    monkeypatch.setattr(backup_manager, "_acquire_lock", observed_acquire)

    def create() -> None:
        try:
            backup_manager.create_backup(compress=True, name="listed")
        except BaseException as exc:
            errors.append(exc)

    def list_created() -> None:
        try:
            results.extend(backup_manager.list_backups())
        except BaseException as exc:
            errors.append(exc)
        finally:
            listed.set()

    creator = threading.Thread(target=create)
    reader = threading.Thread(target=list_created)
    creator.start()
    try:
        assert populated.wait(5)
        reader.start()
        assert shared_lock_attempted.wait(5)
        assert not listed.wait(0.1)
    finally:
        resume.set()
        creator.join(5)
        reader.join(5)

    assert not creator.is_alive()
    assert not reader.is_alive()
    assert errors == []
    assert [backup.name for backup in results] == ["listed"]


def test_backup_manager_restore_waits_for_backup_creation(backup_manager, monkeypatch):
    source = backup_manager.memory_dir / "restored.md"
    source.write_text("restored backup body", encoding="utf-8")
    adding = threading.Event()
    resume = threading.Event()
    restored = threading.Event()
    results: list[bool] = []
    errors: list[BaseException] = []
    original_add = tarfile.TarFile.add
    original_acquire = backup_manager._acquire_lock
    shared_lock_attempted = threading.Event()

    def paused_add(self, *args, **kwargs):
        adding.set()
        assert resume.wait(5)
        return original_add(self, *args, **kwargs)

    def observed_acquire(fh, *, exclusive=True):
        if not exclusive:
            shared_lock_attempted.set()
        return original_acquire(fh, exclusive=exclusive)

    monkeypatch.setattr(tarfile.TarFile, "add", paused_add)
    monkeypatch.setattr(backup_manager, "_acquire_lock", observed_acquire)

    def create() -> None:
        try:
            backup_manager.create_backup(compress=True, name="restored")
        except BaseException as exc:
            errors.append(exc)

    def restore() -> None:
        try:
            results.append(backup_manager.restore_backup("restored", restore_dbs=False))
        except BaseException as exc:
            errors.append(exc)
        finally:
            restored.set()

    creator = threading.Thread(target=create)
    reader = threading.Thread(target=restore)
    creator.start()
    try:
        assert adding.wait(5)
        source.unlink()
        reader.start()
        assert shared_lock_attempted.wait(5)
        assert not restored.wait(0.1)
    finally:
        resume.set()
        creator.join(5)
        reader.join(5)

    assert not creator.is_alive()
    assert not reader.is_alive()
    assert errors == []
    assert results == [True]
    assert source.read_text(encoding="utf-8") == "restored backup body"


def test_backup_manager_read_operations_use_shared_lock(backup_manager, monkeypatch):
    operations: list[int] = []
    monkeypatch.setattr(fcntl, "flock", lambda _descriptor, operation: operations.append(operation))

    assert backup_manager.list_backups() == []
    assert operations == [fcntl.LOCK_SH, fcntl.LOCK_UN]

    operations.clear()
    assert not backup_manager.restore_backup("missing")
    assert operations == [fcntl.LOCK_SH, fcntl.LOCK_UN]


def test_backup_manager_list_ignores_hidden_scratch_directories(backup_manager):
    scratch = backup_manager.backup_dir / ".backup-stale"
    scratch.mkdir()
    metadata = BackupMetadata(
        timestamp="2026-07-20T00:00:00+00:00",
        memory_count=1,
        checksum="checksum",
        compressed_size=1,
        original_size=1,
        name="should-not-surface",
    )
    (scratch / "metadata.json").write_text(
        json.dumps(metadata.__dict__),
        encoding="utf-8",
    )

    assert backup_manager.list_backups() == []


def test_backup_manager_restores_valid_compressed_archive(backup_manager):
    original = backup_manager.memory_dir / "compressed.md"
    original.write_text("compressed backup body", encoding="utf-8")
    backup_manager.create_backup(compress=True, name="compressed")
    original.unlink()

    assert backup_manager.restore_backup("compressed", restore_dbs=False)
    assert original.read_text(encoding="utf-8") == "compressed backup body"


def test_backup_manager_normalizes_archive_suffix_as_logical_name(backup_manager):
    original = backup_manager.memory_dir / "logical-name.md"
    original.write_text("logical backup body", encoding="utf-8")

    metadata = backup_manager.create_backup(compress=True, name="logical.tar.gz")

    assert metadata.name == "logical"
    assert (backup_manager.backup_dir / "logical.tar.gz").is_file()
    assert not (backup_manager.backup_dir / "logical.tar.gz.tar.gz").exists()

    original.unlink()
    assert backup_manager.restore_backup("logical.tar.gz", restore_dbs=False)
    assert original.read_text(encoding="utf-8") == "logical backup body"


def test_backup_manager_rejects_tar_compression_bomb_before_extracting(backup_manager):
    archive = backup_manager.backup_dir / "bomb.tar.gz"
    payload = backup_manager.backup_dir / "payload.md"
    payload.write_bytes(b"0" * (8 * 1024 * 1024))
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(payload, arcname="bomb/memories/payload.md")

    with pytest.raises(ValueError, match="compression ratio"):
        backup_manager.restore_backup("bomb", restore_dbs=False)

    assert not (backup_manager.memory_dir / "payload.md").exists()


def test_backup_manager_rejects_tar_links_before_extracting(backup_manager):
    archive = backup_manager.backup_dir / "links.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        link = tarfile.TarInfo("links/memories/linked.md")
        link.type = tarfile.SYMTYPE
        link.linkname = "/etc/passwd"
        tar.addfile(link)

    with pytest.raises(ValueError, match="Unsupported archive member"):
        backup_manager.restore_backup("links", restore_dbs=False)

    assert not (backup_manager.memory_dir / "linked.md").exists()


def test_backup_manager_skips_markdown_symlink_outside_root(backup_manager, tmp_path: Path):
    outside = tmp_path / "outside-secret.md"
    outside.write_text("NAMED_BACKUP_EXFILTRATION_MARKER", encoding="utf-8")
    (backup_manager.memory_dir / "linked.md").symlink_to(outside)

    metadata = backup_manager.create_backup(compress=False, name="no-symlink")

    backed_up = backup_manager.backup_dir / "no-symlink" / "memories" / "linked.md"
    assert metadata.memory_count == 0
    assert not backed_up.exists()


def test_backup_excludes_legacy_secret_markdown(backup_manager, mock_memory):
    mock_memory.save(content="safe memory", title="Safe")
    marker = mock_memory.cfg.memory_dir / "secrets" / "2026" / "07" / "key.md"
    marker.parent.mkdir(parents=True)
    marker.write_text("---\nid: sec_old\ntype: secret\n---\n[ENCRYPTED]\n", encoding="utf-8")

    metadata = backup_manager.create_backup(compress=False, name="without-secrets")

    assert metadata.memory_count == 1
    assert not (backup_manager.backup_dir / "without-secrets" / "memories" / "secrets").exists()


def test_backup_manager_snapshots_wal_sanitizes_secrets_and_restores(
    backup_manager,
):
    marker = b"MEMO_SECRET_NAMED_BACKUP_MARKER_91ac" * 16
    source = backup_manager.db_dir / "wal-consistent.db"
    writer = sqlite3.connect(source)
    assert writer.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
    writer.execute("PRAGMA wal_autocheckpoint=0")
    writer.executescript(
        "CREATE TABLE visible (value TEXT NOT NULL);"
        "CREATE TABLE secret_store (encrypted_blob BLOB NOT NULL);"
    )
    writer.execute("INSERT INTO secret_store VALUES (?)", (marker,))
    writer.commit()
    writer.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    writer.execute("INSERT INTO visible VALUES ('committed-in-wal')")
    writer.commit()
    assert source.with_name(f"{source.name}-wal").stat().st_size > 0

    try:
        backup_manager.create_backup(compress=False, name="wal-safe")
    finally:
        writer.close()

    snapshot = backup_manager.backup_dir / "wal-safe" / "db" / source.name
    with closing(sqlite3.connect(snapshot)) as connection:
        assert connection.execute("PRAGMA integrity_check").fetchone() == ("ok",)
        assert connection.execute("SELECT COUNT(*) FROM secret_store").fetchone() == (0,)
        assert connection.execute("SELECT value FROM visible").fetchall() == [("committed-in-wal",)]
    assert marker not in snapshot.read_bytes()

    for suffix in ("", "-wal", "-shm"):
        source.with_name(f"{source.name}{suffix}").unlink(missing_ok=True)

    assert backup_manager.restore_backup(
        "wal-safe",
        restore_memories=False,
        restore_dbs=True,
    )
    with closing(sqlite3.connect(source)) as connection:
        assert connection.execute("PRAGMA integrity_check").fetchone() == ("ok",)
        assert connection.execute("SELECT value FROM visible").fetchall() == [("committed-in-wal",)]
        assert connection.execute("SELECT COUNT(*) FROM secret_store").fetchone() == (0,)
    assert marker not in source.read_bytes()


def test_backup_manager_restores_sqlite_without_splitting_live_connections(
    backup_manager, mock_memory, tmp_cfg
):
    before = mock_memory.save(content="included in backup", title="Before")
    backup_manager.create_backup(compress=False, name="live-connections")
    after = mock_memory.save(content="must disappear after restore", title="After")

    observer = Memory(tmp_cfg)
    database = tmp_cfg.db_path
    inode_before = database.stat().st_ino
    try:
        assert mock_memory.store.get(after.id) is not None
        assert observer.store.get(after.id) is not None

        assert backup_manager.restore_backup(
            "live-connections",
            restore_memories=False,
            restore_dbs=True,
        )

        assert database.stat().st_ino == inode_before
        assert mock_memory.store.get(before.id) is not None
        assert observer.store.get(before.id) is not None
        assert mock_memory.store.get(after.id) is None
        assert observer.store.get(after.id) is None
    finally:
        observer.close()


def test_backup_manager_rolls_back_sqlite_logically_after_late_failure(backup_manager, monkeypatch):
    first = backup_manager.db_dir / "a-first.db"
    second = backup_manager.db_dir / "b-second.db"
    for database in (first, second):
        with closing(sqlite3.connect(database)) as connection:
            connection.execute("CREATE TABLE state (value TEXT NOT NULL)")
            connection.execute("INSERT INTO state VALUES ('backup value')")
            connection.commit()

    backup_manager.create_backup(compress=False, name="logical-rollback")
    for database in (first, second):
        with closing(sqlite3.connect(database)) as connection:
            connection.execute("UPDATE state SET value = 'live value'")
            connection.commit()

    live_connection = sqlite3.connect(first)
    inode_before = first.stat().st_ino
    real_restore = backup_manager._restore_sqlite_in_place
    failed = False

    def fail_second(source: Path, destination: Path) -> None:
        nonlocal failed
        if destination.name == second.name and not failed:
            failed = True
            raise OSError("injected restore failure")
        real_restore(source, destination)

    monkeypatch.setattr(backup_manager, "_restore_sqlite_in_place", fail_second)
    try:
        with pytest.raises(OSError, match="injected restore failure"):
            backup_manager.restore_backup(
                "logical-rollback",
                restore_memories=False,
                restore_dbs=True,
            )

        assert first.stat().st_ino == inode_before
        assert live_connection.execute("SELECT value FROM state").fetchone() == ("live value",)
        with closing(sqlite3.connect(second)) as connection:
            assert connection.execute("SELECT value FROM state").fetchone() == ("live value",)
    finally:
        live_connection.close()


@pytest.mark.parametrize(
    "name",
    ["../escaped", "nested/backup", r"nested\backup", ".", "..", ""],
)
def test_backup_manager_rejects_unsafe_backup_names(backup_manager, name):
    """User-controlled backup names must remain a single safe path component."""
    with pytest.raises(ValueError, match="backup name"):
        backup_manager.create_backup(compress=False, name=name)


def test_backup_manager_rejects_absolute_backup_name_without_deleting_it(
    backup_manager, tmp_path: Path
):
    """An absolute existing directory must never become the backup scratch directory."""
    victim = tmp_path / "victim"
    victim.mkdir()
    marker = victim / "keep.txt"
    marker.write_text("keep", encoding="utf-8")

    with pytest.raises(ValueError, match="backup name"):
        backup_manager.create_backup(compress=True, name=str(victim))

    assert marker.read_text(encoding="utf-8") == "keep"


def test_backup_manager_refuses_to_overwrite_existing_backup_directory(backup_manager):
    """A safe-name collision must leave the existing directory untouched."""
    existing = backup_manager.backup_dir / "existing"
    existing.mkdir()
    marker = existing / "keep.txt"
    marker.write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        backup_manager.create_backup(compress=True, name="existing")

    assert marker.read_text(encoding="utf-8") == "keep"


def test_backup_manager_restore_rejects_paths_outside_backup_dir(backup_manager, tmp_path: Path):
    """Restore names must not address an arbitrary directory outside backup_dir."""
    external = tmp_path / "external"
    memories = external / "memories"
    memories.mkdir(parents=True)
    (memories / "injected.md").write_text("injected", encoding="utf-8")

    with pytest.raises(ValueError, match="backup name"):
        backup_manager.restore_backup(str(external))

    assert not (backup_manager.memory_dir / "injected.md").exists()


def test_backup_manager_restore_rejects_symlinked_destination(backup_manager, tmp_path: Path):
    project = backup_manager.memory_dir / "project"
    project.mkdir()
    (project / "memory.md").write_text("backup body", encoding="utf-8")
    backup_manager.create_backup(compress=False, name="safe-source")
    shutil.rmtree(project)
    outside = tmp_path / "outside"
    outside.mkdir()
    project.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="symlinked restore destination"):
        backup_manager.restore_backup("safe-source", restore_dbs=False)

    assert not (outside / "memory.md").exists()


def test_backup_manager_restore_verifies_checksum_before_writing(backup_manager):
    original = backup_manager.memory_dir / "original.md"
    original.write_text("original", encoding="utf-8")
    backup_manager.create_backup(compress=False, name="checksum")
    original.unlink()
    archived = backup_manager.backup_dir / "checksum" / "memories" / "original.md"
    archived.write_text("tampered", encoding="utf-8")

    with pytest.raises(ValueError, match="checksum"):
        backup_manager.restore_backup("checksum", restore_dbs=False)

    assert not original.exists()


def test_backup_manager_restore_checks_staged_bytes_after_planning(backup_manager, monkeypatch):
    original = backup_manager.memory_dir / "staged.md"
    original.write_text("validated", encoding="utf-8")
    backup_manager.create_backup(compress=False, name="staged-checksum")
    original.unlink()

    real_safe_regular_files = backup_manager._safe_regular_files
    mutated = False

    def mutate_during_restore(root: Path, pattern: str) -> list[Path]:
        nonlocal mutated
        if root.name == "memories" and not mutated:
            (root / "staged.md").write_text("changed after initial check", encoding="utf-8")
            mutated = True
        return real_safe_regular_files(root, pattern)

    monkeypatch.setattr(backup_manager, "_safe_regular_files", mutate_during_restore)

    with pytest.raises(ValueError, match="checksum"):
        backup_manager.restore_backup("staged-checksum", restore_dbs=False)

    assert mutated
    assert not original.exists()


def test_backup_manager_restore_uses_staged_snapshot(backup_manager, monkeypatch):
    original = backup_manager.memory_dir / "snapshot.md"
    original.write_text("staged value", encoding="utf-8")
    backup_manager.create_backup(compress=False, name="staged-snapshot")
    original.unlink()

    backup_path = backup_manager.backup_dir / "staged-snapshot"
    archived = backup_path / "memories" / "snapshot.md"
    real_copytree = shutil.copytree
    staged = False

    def copy_then_mutate_source(source, destination, *args, **kwargs):
        nonlocal staged
        result = real_copytree(source, destination, *args, **kwargs)
        if Path(source) == backup_path:
            archived.write_text("changed after staging", encoding="utf-8")
            staged = True
        return result

    monkeypatch.setattr(shutil, "copytree", copy_then_mutate_source)

    assert backup_manager.restore_backup("staged-snapshot", restore_dbs=False)
    assert staged
    assert original.read_text(encoding="utf-8") == "staged value"


def test_backup_manager_list_backups(backup_manager):
    """Test listing backups."""
    # Create a backup
    backup_manager.create_backup(compress=True, name="test_backup")

    backups = backup_manager.list_backups()

    assert len(backups) >= 1
    assert all(isinstance(b, BackupMetadata) for b in backups)


def test_sync_manager_init(sync_manager, mock_memory):
    """SyncManager holds the local Memory it replays events into."""
    assert sync_manager.mem is mock_memory


def test_sync_from_remote_missing_file_reports_error(sync_manager, tmp_path):
    """A missing remote history.db is an error, not a crash."""
    diff = sync_manager.sync_from_remote(tmp_path / "does-not-exist.db")
    assert isinstance(diff, SyncDiff)
    assert diff.applied == 0
    assert diff.errors == 1


def test_sync_from_remote_loopback_is_noop(sync_manager, mock_memory):
    """Syncing from this device's own history is a no-op (same device id)."""
    # Force the history.db into existence with at least one event.
    mock_memory.save(content="seed", title="Seed", tags=["x"])
    diff = sync_manager.sync_from_remote(mock_memory.cfg.history_db)
    assert diff == SyncDiff(0, 0, 0)


def test_sync_diff_dataclass():
    """SyncDiff carries applied/conflicts/errors counters."""
    diff = SyncDiff(applied=3, conflicts=1, errors=2)
    assert diff.applied == 3
    assert diff.conflicts == 1
    assert diff.errors == 2


def test_backup_metadata_dataclass():
    """Test BackupMetadata dataclass structure."""
    metadata = BackupMetadata(
        timestamp="2026-01-01T00:00:00Z",
        memory_count=10,
        checksum="abc123",
        compressed_size=1000,
        original_size=5000,
    )
    assert metadata.memory_count == 10
    assert metadata.checksum == "abc123"
    assert metadata.compressed_size == 1000


def test_backup_manager_compute_checksum(backup_manager, tmp_path):
    """Test checksum computation."""
    # Create a test file
    test_file = tmp_path / "test.txt"
    test_file.write_text("test content", encoding="utf-8")

    checksum = backup_manager._compute_checksum(tmp_path)

    assert checksum
    assert len(checksum) == 64  # SHA256 hex length
