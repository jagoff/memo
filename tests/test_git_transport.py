from __future__ import annotations

import hashlib
import os
import subprocess
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from memo.errors import OperationalError, OperationalErrorCode
from memo.git_transport import GitTransport, TransportHead
from memo.operational_event import ChainAnchor, OriginBundle, canonical_json_bytes


def _git(root: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ("git", *arguments),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        env={
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
            "GIT_AUTHOR_NAME": "transport test",
            "GIT_AUTHOR_EMAIL": "transport@example.invalid",
            "GIT_COMMITTER_NAME": "transport test",
            "GIT_COMMITTER_EMAIL": "transport@example.invalid",
        },
    )


def _head(**changes: object) -> TransportHead:
    value = TransportHead(
        schema="memo.operational_transport_head.v1",
        origin_device="device-a",
        ledger_epoch=0,
        sequence=1,
        event_hash="e" * 64,
        anchor_hash="a" * 64,
        checkpoint_id="checkpoint-a",
        roster_version=1,
        key_id="device-key-a",
        signature="signed-head",
    )
    return replace(value, **changes)


def _empty_bundle() -> OriginBundle:
    checkpoint = b"{}"
    anchor = ChainAnchor(
        schema="memo.operational_anchor.v1",
        anchor_id="anchor-a",
        origin_device="device-a",
        ledger_epoch=0,
        reducer_version=1,
        kind="empty",
        base_sequence=0,
        base_event_hash="",
        final_sequence=0,
        final_event_hash="",
        previous_anchor_hash="",
        source_manifest_sha256="",
        state_sha256=hashlib.sha256(checkpoint).hexdigest(),
        checkpoint_id="checkpoint-a",
        checkpoint_sha256=hashlib.sha256(checkpoint).hexdigest(),
        checkpoint_size=len(checkpoint),
        created_at="2026-08-01T00:00:00Z",
        anchor_hash="a" * 64,
        roster_version=1,
        signer_role="origin",
        attested_origin="device-a",
        key_id="device-key-a",
        signature="signed-anchor",
    )
    return OriginBundle(
        anchor=anchor,
        checkpoint=checkpoint,
        events=(),
        head_sequence=0,
        head_hash="",
    )


def test_existing_git_repository_requires_explicit_memo_transport_marker(
    tmp_path: Path,
) -> None:
    root = tmp_path / "generic-repository"
    root.mkdir()
    _git(root, "init", "--quiet")
    before = (root / ".git" / "config").read_bytes()

    with pytest.raises(OperationalError, match="explicitly initialized") as raised:
        GitTransport(root)

    assert raised.value.code is OperationalErrorCode.INVALID_EVENT
    assert (root / ".git" / "config").read_bytes() == before
    assert not (root / ".memo-operational-transport").exists()


def test_transport_marker_binds_remote_and_rejects_rebinding(tmp_path: Path) -> None:
    root = tmp_path / "transport"
    remote_a = tmp_path / "remote-a.git"
    remote_b = tmp_path / "remote-b.git"
    remote_a.mkdir()
    remote_b.mkdir()
    _git(remote_a, "init", "--quiet", "--bare")
    _git(remote_b, "init", "--quiet", "--bare")
    GitTransport(root, remote=remote_a)

    with pytest.raises(OperationalError, match="remote binding changed"):
        GitTransport(root, remote=remote_b)


def test_symlinked_marker_is_rejected_without_touching_target(tmp_path: Path) -> None:
    root = tmp_path / "transport"
    root.mkdir()
    _git(root, "init", "--quiet")
    outside = tmp_path / "outside-marker"
    outside.write_bytes(b"outside must remain unchanged")
    (root / ".memo-operational-transport").symlink_to(outside)

    with pytest.raises(OperationalError, match=r"unsafe|symlink|regular"):
        GitTransport(root)

    assert outside.read_bytes() == b"outside must remain unchanged"


def test_read_head_never_follows_symlinked_heads_directory(tmp_path: Path) -> None:
    root = tmp_path / "transport"
    transport = GitTransport(root)
    outside = tmp_path / "outside-heads"
    outside.mkdir()
    head_path = outside / "device-a.json"
    head_path.write_bytes(canonical_json_bytes(asdict(_head())))
    (root / "heads").symlink_to(outside, target_is_directory=True)

    with pytest.raises(OperationalError, match=r"unsafe|symlink"):
        transport.read_head("device-a")

    assert head_path.read_bytes() == canonical_json_bytes(asdict(_head()))


def test_read_head_rejects_hard_link_to_file_outside_transport(tmp_path: Path) -> None:
    root = tmp_path / "transport"
    transport = GitTransport(root)
    outside = tmp_path / "outside-head.json"
    encoded = canonical_json_bytes(asdict(_head()))
    outside.write_bytes(encoded)
    (root / "heads").mkdir()
    os.link(outside, root / "heads" / "device-a.json")

    with pytest.raises(OperationalError, match="repository entry"):
        transport.read_head("device-a")

    assert outside.read_bytes() == encoded


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("ledger_epoch", True),
        ("ledger_epoch", -1),
        ("ledger_epoch", 1 << 63),
        ("sequence", False),
        ("sequence", -1),
        ("sequence", 1 << 63),
        ("event_hash", "../outside"),
        ("anchor_hash", "../../outside"),
        ("checkpoint_id", "../outside"),
        ("roster_version", True),
        ("roster_version", 0),
        ("roster_version", 1 << 63),
        ("key_id", "../outside"),
        ("signature", ""),
    ],
)
def test_read_head_validates_types_ranges_and_path_material_before_use(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    root = tmp_path / "transport"
    transport = GitTransport(root)
    heads = root / "heads"
    heads.mkdir()
    malicious = asdict(_head())
    malicious[field] = value
    (heads / "device-a.json").write_bytes(canonical_json_bytes(malicious))

    with pytest.raises(OperationalError) as raised:
        transport.read_head("device-a")

    assert raised.value.code is OperationalErrorCode.INVALID_EVENT


def test_supplied_head_is_revalidated_before_artifact_path_construction(
    tmp_path: Path,
) -> None:
    transport = GitTransport(tmp_path / "transport")
    outside = tmp_path / "outside.json"
    outside.write_bytes(b"not transport data")

    with pytest.raises(OperationalError, match="anchor hash"):
        transport.read_anchor(_head(anchor_hash="../../../../outside"))

    assert outside.read_bytes() == b"not transport data"


@pytest.mark.parametrize("sequence", [False, 0, -1, 2, 1 << 63])
def test_segment_sequence_must_be_an_integer_inside_the_signed_head(
    tmp_path: Path,
    sequence: object,
) -> None:
    transport = GitTransport(tmp_path / "transport")

    with pytest.raises(OperationalError, match="segment sequence"):
        transport.read_segment(_head(sequence=1), sequence)  # type: ignore[arg-type]


def test_publish_never_writes_through_symlinked_artifact_directory(tmp_path: Path) -> None:
    root = tmp_path / "transport"
    transport = GitTransport(root)
    outside = tmp_path / "outside-artifacts"
    outside.mkdir()
    (root / "anchors").symlink_to(outside, target_is_directory=True)

    with pytest.raises(OperationalError, match=r"unsafe|symlink"):
        transport.publish(_empty_bundle())

    assert list(outside.iterdir()) == []
    assert not (root / "checkpoints").exists()
    assert not (root / "heads").exists()


def test_hostile_remote_symlink_tree_is_rejected_before_checkout(tmp_path: Path) -> None:
    remote = tmp_path / "remote.git"
    remote.mkdir()
    _git(remote, "init", "--quiet", "--bare")
    attacker = tmp_path / "attacker"
    attacker.mkdir()
    _git(attacker, "init", "--quiet")
    target = tmp_path / "outside-remote-target"
    target.mkdir()
    (attacker / "heads").symlink_to(target, target_is_directory=True)
    _git(attacker, "add", "--", "heads")
    _git(attacker, "commit", "--quiet", "-m", "hostile transport tree")
    _git(
        attacker,
        "push",
        "--quiet",
        str(remote),
        "HEAD:refs/heads/memo-operational",
    )
    local = tmp_path / "local"

    with pytest.raises(OperationalError, match="remote tree entry"):
        GitTransport(local, remote=remote)

    assert not (local / "heads").exists()
    assert list(target.iterdir()) == []


def test_hostile_git_config_is_rejected_before_git_operation(tmp_path: Path) -> None:
    root = tmp_path / "transport"
    transport = GitTransport(root)
    outside = tmp_path / "outside-config"
    outside.write_text("[core]\n\trepositoryformatversion = 0\n", encoding="utf-8")
    with (root / ".git" / "config").open("a", encoding="utf-8") as stream:
        stream.write(f'\n[include]\n\tpath = "{outside}"\n')

    with pytest.raises(OperationalError, match="config section"):
        transport.publish(_empty_bundle())

    assert outside.read_text(encoding="utf-8") == "[core]\n\trepositoryformatversion = 0\n"


def test_normal_transport_reopens_and_publishes(tmp_path: Path) -> None:
    root = tmp_path / "transport"
    transport = GitTransport(root)

    result = transport.publish(_empty_bundle())
    reopened = GitTransport(root)

    assert len(result.git_oid) in {40, 64}
    assert reopened.read_head("device-a") == _head(
        sequence=0,
        event_hash="",
        signature="signed-anchor",
    )
    assert reopened.read_anchor(reopened.read_head("device-a")) == canonical_json_bytes(  # type: ignore[arg-type]
        _empty_bundle().anchor
    )
