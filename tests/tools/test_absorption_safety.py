from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tools.memflow_absorption.safety import (
    ATTEMPT_SENTINEL,
    SafetyError,
    assert_safe_attempt_root,
    initialize_attempt_root,
    resolve_under_attempt,
)

MANIFEST_SHA256 = "a" * 64


@pytest.mark.parametrize("bad", [Path("/"), Path.home()])
def test_attempt_root_rejects_broad_targets(bad: Path) -> None:
    with pytest.raises(SafetyError):
        assert_safe_attempt_root(bad, "attempt-123")


def test_attempt_root_requires_exact_cutover_suffix_and_rejects_repo(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()

    with pytest.raises(SafetyError, match="repository"):
        assert_safe_attempt_root(repo, "repo")
    with pytest.raises(SafetyError, match="memo/cutover"):
        assert_safe_attempt_root(tmp_path / "attempt-123", "attempt-123")
    with pytest.raises(SafetyError, match="unresolved"):
        assert_safe_attempt_root(Path("$STATE/memo/cutover/attempt-123"), "attempt-123")


def test_attempt_root_rejects_symlink_component_and_wrong_sentinel(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)
    candidate = linked / "memo" / "cutover" / "attempt-123"

    with pytest.raises(SafetyError, match="symlink"):
        assert_safe_attempt_root(candidate, "attempt-123")

    root = tmp_path / "state" / "memo" / "cutover" / "attempt-123"
    root.mkdir(parents=True)
    (root / ATTEMPT_SENTINEL).write_text(
        json.dumps({"schema": "memo.cutover_attempt.v1", "attempt_id": "other"}),
        encoding="utf-8",
    )
    with pytest.raises(SafetyError, match="sentinel"):
        assert_safe_attempt_root(
            root,
            "attempt-123",
            require_sentinel=True,
            manifest_sha256=MANIFEST_SHA256,
        )


def test_initialize_is_explicit_and_resolution_stays_below_attempt(tmp_path: Path) -> None:
    root = tmp_path / "state" / "memo" / "cutover" / "attempt-123"

    initialized = initialize_attempt_root(root, "attempt-123", MANIFEST_SHA256)

    assert initialized == root
    assert (root / ATTEMPT_SENTINEL).is_file()
    assert json.loads((root / ATTEMPT_SENTINEL).read_text(encoding="utf-8"))[
        "manifest_sha256"
    ] == MANIFEST_SHA256
    assert resolve_under_attempt(
        root,
        "snapshots/memo.json",
        "attempt-123",
        MANIFEST_SHA256,
    ) == (
        root / "snapshots" / "memo.json"
    )
    with pytest.raises(SafetyError, match="manifest"):
        resolve_under_attempt(root, "snapshots/memo.json", "attempt-123", "b" * 64)
    with pytest.raises(SafetyError, match="escapes"):
        resolve_under_attempt(root, "../outside", "attempt-123", MANIFEST_SHA256)
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "linked").symlink_to(outside, target_is_directory=True)
    with pytest.raises(SafetyError, match="symlink"):
        resolve_under_attempt(root, "linked/value.json", "attempt-123", MANIFEST_SHA256)


def test_operator_snapshot_command_is_dry_run_by_default(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_text('{"safe":true}', encoding="utf-8")
    root = tmp_path / "state" / "memo" / "cutover" / "attempt-123"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.memflow_absorption",
            "snapshot",
            "--attempt-root",
            str(root),
            "--attempt-id",
            "attempt-123",
            "--source",
            str(source),
            "--target-name",
            "source.json",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    payload = json.loads(completed.stdout)

    assert payload["dry_run"] is True
    assert payload["would_write"].endswith("/snapshots/source.json")
    assert not root.exists()


def test_operator_snapshot_dry_run_rejects_target_traversal(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_text('{"safe":true}', encoding="utf-8")
    root = tmp_path / "state" / "memo" / "cutover" / "attempt-123"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.memflow_absorption",
            "snapshot",
            "--attempt-root",
            str(root),
            "--attempt-id",
            "attempt-123",
            "--source",
            str(source),
            "--target-name",
            "../outside.json",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode != 0
    assert "safe filename" in completed.stderr
    assert not root.exists()


def test_operator_snapshot_apply_requires_preexisting_manifest_bound_authority(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.json"
    source.write_text('{"safe":true}', encoding="utf-8")
    root = tmp_path / "state" / "memo" / "cutover" / "attempt-123"
    command = [
        sys.executable,
        "-m",
        "tools.memflow_absorption",
        "snapshot",
        "--attempt-root",
        str(root),
        "--attempt-id",
        "attempt-123",
        "--source",
        str(source),
        "--target-name",
        "source.json",
        "--apply",
    ]

    missing = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert missing.returncode != 0
    assert "manifest SHA-256" in missing.stderr
    assert not root.exists()

    initialize_attempt_root(root, "attempt-123", MANIFEST_SHA256)
    mismatch = subprocess.run(
        [*command, "--manifest-sha256", "b" * 64],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert mismatch.returncode != 0
    assert "manifest" in mismatch.stderr
    assert not (root / "snapshots").exists()

    applied = subprocess.run(
        [*command, "--manifest-sha256", MANIFEST_SHA256],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert json.loads(applied.stdout)["dry_run"] is False
    assert (root / "snapshots" / "source.json").is_file()
