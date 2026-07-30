from __future__ import annotations

import json
from pathlib import Path

import pytest

from memo.artifact_store import ArtifactIntegrityError, ContentAddressedArtifactStore


def test_content_addressed_json_round_trip_and_export(tmp_path: Path) -> None:
    store = ContentAddressedArtifactStore(tmp_path / "cas")
    ref = store.put_json("signals", {"z": 1, "a": ["x"]}, metadata={"repo": "r"})
    same = store.put_json("signals", {"z": 1, "a": ["x"]}, metadata={"repo": "r"})

    assert same.digest == ref.digest
    assert same.created_at == ref.created_at
    assert store.load_json(ref) == {"z": 1, "a": ["x"]}
    assert store.verify(ref)["ok"] is True

    exported = store.export(ref, tmp_path / "shared")
    assert Path(exported["artifact"]).is_file()
    manifest = json.loads(Path(exported["manifest"]).read_text(encoding="utf-8"))
    assert manifest["digest"] == ref.digest

    imported = store.import_file(
        "imported",
        Path(exported["artifact"]),
        expected_digest=ref.digest,
        media_type="application/json",
    )
    assert imported.digest == ref.digest
    assert store.load_json(imported) == {"z": 1, "a": ["x"]}


def test_artifact_read_rejects_tampering(tmp_path: Path) -> None:
    store = ContentAddressedArtifactStore(tmp_path / "cas")
    ref = store.put_bytes("binary", b"trusted")
    Path(ref.path).write_bytes(b"tampered")

    with pytest.raises(ArtifactIntegrityError, match="mismatch"):
        store.load_bytes(ref)
    assert store.verify(ref)["ok"] is False


def test_content_addressed_store_rejects_media_type_conflict(tmp_path: Path) -> None:
    store = ContentAddressedArtifactStore(tmp_path / "cas")
    store.put_bytes("binary", b"trusted", media_type="application/octet-stream")

    with pytest.raises(ArtifactIntegrityError, match="already stored"):
        store.put_bytes("binary", b"trusted", media_type="text/plain")
