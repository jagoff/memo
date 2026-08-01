"""Verified content-addressed artifacts used by repo intelligence providers.

Artifacts are immutable, self-describing files.  The digest is calculated
over the exact payload bytes, every read verifies both size and digest, and
publishing uses ``os.replace`` so readers never observe a partial file.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from memo.util import utc_now_iso

_NAMESPACE_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")


class ArtifactIntegrityError(RuntimeError):
    """Raised when an artifact no longer matches its recorded identity."""


@dataclass(frozen=True)
class ArtifactRef:
    schema: str
    namespace: str
    digest: str
    size_bytes: int
    media_type: str
    path: str
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _canonical_json(payload: Any) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")


class ContentAddressedArtifactStore:
    """Immutable local CAS with portable sidecar manifests."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def put_json(
        self,
        namespace: str,
        payload: Any,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactRef:
        envelope = {
            "schema": "memo.artifact.payload.v1",
            "metadata": dict(metadata or {}),
            "payload": payload,
        }
        return self.put_bytes(
            namespace,
            _canonical_json(envelope),
            media_type="application/json",
        )

    def put_bytes(
        self,
        namespace: str,
        data: bytes,
        *,
        media_type: str = "application/octet-stream",
    ) -> ArtifactRef:
        namespace = self._validate_namespace(namespace)
        digest = hashlib.sha256(data).hexdigest()
        created_at = utc_now_iso()
        artifact_path = self._artifact_path(namespace, digest)
        manifest_path = self._manifest_path(namespace, digest)
        artifact_path.parent.mkdir(parents=True, exist_ok=True)

        if artifact_path.exists():
            self._verify_path(artifact_path, digest=digest, size_bytes=len(data))
        else:
            self._atomic_write(artifact_path, data)

        if manifest_path.exists():
            persisted = self._load_manifest(manifest_path)
            if (
                persisted.namespace != namespace
                or persisted.digest != digest
                or persisted.size_bytes != len(data)
                or Path(persisted.path) != artifact_path
            ):
                raise ArtifactIntegrityError(
                    f"artifact manifest {manifest_path} does not match its content identity"
                )
            if persisted.media_type != media_type:
                raise ArtifactIntegrityError(
                    f"artifact {digest[:12]} is already stored as "
                    f"{persisted.media_type}, not {media_type}"
                )
            return persisted

        ref = ArtifactRef(
            schema="memo.artifact.ref.v1",
            namespace=namespace,
            digest=digest,
            size_bytes=len(data),
            media_type=media_type,
            path=str(artifact_path),
            created_at=created_at,
        )
        manifest = {
            **ref.to_dict(),
            "artifact_file": artifact_path.name,
        }
        manifest_bytes = json.dumps(
            manifest,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        ).encode("utf-8")
        self._atomic_write(manifest_path, manifest_bytes)
        return ref

    def load_bytes(self, ref: ArtifactRef | dict[str, Any]) -> bytes:
        parsed = self._coerce_ref(ref)
        path = Path(parsed.path)
        data = path.read_bytes()
        self._verify_bytes(data, digest=parsed.digest, size_bytes=parsed.size_bytes)
        return data

    def load_json(self, ref: ArtifactRef | dict[str, Any]) -> Any:
        parsed = self._coerce_ref(ref)
        if parsed.media_type != "application/json":
            raise ArtifactIntegrityError(
                f"artifact {parsed.digest[:12]} is {parsed.media_type}, not JSON"
            )
        try:
            envelope = json.loads(self.load_bytes(parsed).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ArtifactIntegrityError(
                f"artifact {parsed.digest[:12]} contains invalid JSON"
            ) from exc
        if not isinstance(envelope, dict) or envelope.get("schema") != "memo.artifact.payload.v1":
            raise ArtifactIntegrityError(
                f"artifact {parsed.digest[:12]} has an unsupported payload schema"
            )
        return envelope.get("payload")

    def verify(self, ref: ArtifactRef | dict[str, Any]) -> dict[str, Any]:
        parsed = self._coerce_ref(ref)
        try:
            self.load_bytes(parsed)
        except (OSError, ArtifactIntegrityError) as exc:
            return {
                "ok": False,
                "digest": parsed.digest,
                "path": parsed.path,
                "error": f"{type(exc).__name__}: {exc}",
            }
        return {
            "ok": True,
            "digest": parsed.digest,
            "path": parsed.path,
            "size_bytes": parsed.size_bytes,
            "media_type": parsed.media_type,
        }

    def export(
        self,
        ref: ArtifactRef | dict[str, Any],
        destination: Path,
    ) -> dict[str, str]:
        """Copy a verified artifact and its portable manifest to a directory."""
        parsed = self._coerce_ref(ref)
        self.load_bytes(parsed)
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        source = Path(parsed.path)
        artifact_out = destination / source.name
        manifest_out = destination / f"{parsed.digest}.manifest.json"
        shutil.copyfile(source, artifact_out)
        self._atomic_write(
            manifest_out,
            json.dumps(parsed.to_dict(), ensure_ascii=False, sort_keys=True, indent=2).encode(
                "utf-8"
            ),
        )
        return {"artifact": str(artifact_out), "manifest": str(manifest_out)}

    def import_file(
        self,
        namespace: str,
        artifact: Path,
        *,
        expected_digest: str | None = None,
        media_type: str = "application/octet-stream",
    ) -> ArtifactRef:
        data = Path(artifact).read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if expected_digest is not None and digest != expected_digest:
            raise ArtifactIntegrityError(
                f"import digest mismatch: expected {expected_digest}, got {digest}"
            )
        return self.put_bytes(namespace, data, media_type=media_type)

    def _artifact_path(self, namespace: str, digest: str) -> Path:
        return self.root / namespace / digest[:2] / f"{digest}.artifact"

    def _manifest_path(self, namespace: str, digest: str) -> Path:
        return self.root / namespace / digest[:2] / f"{digest}.manifest.json"

    @staticmethod
    def _validate_namespace(namespace: str) -> str:
        value = str(namespace or "").strip().lower()
        if not _NAMESPACE_RE.fullmatch(value):
            raise ValueError(f"invalid artifact namespace: {namespace!r}")
        return value

    @staticmethod
    def _coerce_ref(ref: ArtifactRef | dict[str, Any]) -> ArtifactRef:
        if isinstance(ref, ArtifactRef):
            return ref
        try:
            return ArtifactRef(
                schema=str(ref["schema"]),
                namespace=str(ref["namespace"]),
                digest=str(ref["digest"]),
                size_bytes=int(ref["size_bytes"]),
                media_type=str(ref["media_type"]),
                path=str(ref["path"]),
                created_at=str(ref["created_at"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ArtifactIntegrityError("invalid artifact reference") from exc

    @staticmethod
    def _load_manifest(path: Path) -> ArtifactRef:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ArtifactIntegrityError(f"invalid artifact manifest: {path}") from exc
        if not isinstance(payload, dict):
            raise ArtifactIntegrityError(f"invalid artifact manifest: {path}")
        return ContentAddressedArtifactStore._coerce_ref(payload)

    @staticmethod
    def _verify_path(path: Path, *, digest: str, size_bytes: int) -> None:
        ContentAddressedArtifactStore._verify_bytes(
            path.read_bytes(),
            digest=digest,
            size_bytes=size_bytes,
        )

    @staticmethod
    def _verify_bytes(data: bytes, *, digest: str, size_bytes: int) -> None:
        if len(data) != size_bytes:
            raise ArtifactIntegrityError(
                f"artifact size mismatch: expected {size_bytes}, got {len(data)}"
            )
        actual = hashlib.sha256(data).hexdigest()
        if actual != digest:
            raise ArtifactIntegrityError(
                f"artifact digest mismatch: expected {digest}, got {actual}"
            )

    @staticmethod
    def _atomic_write(path: Path, data: bytes) -> None:
        temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            with temp.open("xb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp, path)
        finally:
            temp.unlink(missing_ok=True)


__all__ = [
    "ArtifactIntegrityError",
    "ArtifactRef",
    "ContentAddressedArtifactStore",
]
