"""Anchored, signed operational ledger v2.

The JSONL segments are authoritative.  Per-origin heads are durable advisory
caches and may only be repaired from a completely verified segment chain.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import tempfile
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, fields, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from memo.atomic_io import atomic_write_text, authority_write_lock
from memo.contracts import MemoEvent
from memo.errors import OperationalError, OperationalErrorCode
from memo.identity import PrincipalIdentity
from memo.operation_ledger_v1 import LegacyOperationLedger
from memo.operational_epoch import CommitContext, EpochFence
from memo.operational_event import (
    EMPTY_REDUCER_STATE_BYTES,
    ChainAnchor,
    LedgerImportReport,
    OperationalCommand,
    OperationalEventV2,
    OriginBundle,
    OriginPosition,
    SourceProof,
    VerificationReport,
    canonical_anchor_hash,
    canonical_event_hash,
    canonical_json_bytes,
    canonical_signed_bytes,
    validate_anchor,
    validate_event,
    validate_migration_origin,
)
from memo.operational_event_types import validate_event_payload
from memo.operational_key_store import AuthorityPinStore, PublicKeyRecord
from memo.operational_roster import VerificationRoster
from memo.operational_signing import (
    OperationalSigner,
    OperationalVerifier,
    SignatureEnvelope,
)

_SAFE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_ANCHOR_FIELDS = frozenset(field.name for field in fields(ChainAnchor))
_EVENT_FIELDS = frozenset(field.name for field in fields(OperationalEventV2))
_IDENTITY_FIELDS = frozenset(field.name for field in fields(PrincipalIdentity))
_SOURCE_PROOF_FIELDS = frozenset(field.name for field in fields(SourceProof))
_HEAD_FIELDS = frozenset({"schema", "origin_device", "sequence", "event_hash", "anchor_hash"})
_HEAD_SCHEMA = "memo.operational_head.v1"
_QUARANTINE_SCHEMA = "memo.operational_quarantine.v1"


def _failure(
    code: OperationalErrorCode,
    message: str,
    *,
    details: dict[str, Any] | None = None,
) -> OperationalError:
    return OperationalError(code, message, retryable=False, details=details)


def _require_string(value: object, field: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        raise _failure(
            OperationalErrorCode.INVALID_EVENT,
            f"{field} must be a {'string' if allow_empty else 'non-empty string'}",
        )
    return value


def _require_int(value: object, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise _failure(
            OperationalErrorCode.INVALID_EVENT,
            f"{field} must be an integer >= {minimum}",
        )
    return value


def _parse_timestamp(value: str, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, ValueError) as exc:
        raise _failure(
            OperationalErrorCode.INVALID_EVENT,
            f"{field} must be an ISO-8601 timestamp",
        ) from exc
    if parsed.tzinfo is None:
        raise _failure(
            OperationalErrorCode.INVALID_EVENT,
            f"{field} must include a timezone",
        )
    return parsed.astimezone(UTC)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class OperationLedgerV2:
    """Local anchored ledger plus verified federation bundle admission."""

    def __init__(
        self,
        operational_root: Path,
        *,
        device_id: str,
        clock: Callable[[], str | datetime] | None = None,
        signer: OperationalSigner | None = None,
        verifier: OperationalVerifier | None = None,
        roster: VerificationRoster | None = None,
        roster_root: Path | None = None,
        pin_store: AuthorityPinStore | None = None,
        epoch_fence: EpochFence | None = None,
        reducer_version: int = 1,
    ) -> None:
        if not isinstance(device_id, str) or not _SAFE_ID_RE.fullmatch(device_id):
            raise _failure(
                OperationalErrorCode.INVALID_EVENT,
                f"unsafe operational device id: {device_id!r}",
            )
        if (
            isinstance(reducer_version, bool)
            or not isinstance(reducer_version, int)
            or reducer_version < 1
        ):
            raise ValueError("reducer_version must be a positive integer")
        self.operational_root = Path(operational_root).absolute()
        self.root = self.operational_root / "journal"
        self.device_id = device_id
        self.clock = clock or (lambda: datetime.now(UTC))
        self.signer = signer
        self.verifier = verifier
        self.roster = roster
        self.roster_root = (
            Path(roster_root).absolute()
            if roster_root is not None
            else (Path(epoch_fence.root).absolute() if epoch_fence is not None else None)
        )
        self.pin_store = pin_store
        self.epoch_fence = epoch_fence
        self.reducer_version = reducer_version
        self._assert_safe_path(self.root)

    @property
    def anchors_dir(self) -> Path:
        return self.root / "anchors"

    @property
    def checkpoints_dir(self) -> Path:
        return self.root / "checkpoints"

    @property
    def events_dir(self) -> Path:
        return self.root / "events"

    @property
    def heads_dir(self) -> Path:
        return self.root / "heads"

    @property
    def quarantine_dir(self) -> Path:
        return self.root / "quarantine"

    def _now(self) -> str:
        value = self.clock()
        if isinstance(value, datetime):
            parsed = value
        elif isinstance(value, str):
            parsed = _parse_timestamp(value, "clock")
        else:
            raise _failure(
                OperationalErrorCode.INVALID_EVENT,
                "clock must return a datetime or ISO-8601 string",
            )
        if parsed.tzinfo is None:
            raise _failure(
                OperationalErrorCode.INVALID_EVENT,
                "clock datetime must include a timezone",
            )
        return parsed.astimezone(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")

    @staticmethod
    def _validate_safe_id(value: str, field: str) -> str:
        if not isinstance(value, str) or not _SAFE_ID_RE.fullmatch(value):
            raise _failure(
                OperationalErrorCode.INVALID_EVENT,
                f"unsafe {field}: {value!r}",
            )
        return value

    def _assert_safe_path(self, path: Path) -> None:
        root = self.root.absolute()
        target = Path(path).absolute()
        try:
            relative = target.relative_to(root)
        except ValueError as exc:
            raise _failure(
                OperationalErrorCode.STORAGE_UNAVAILABLE,
                f"operational path escapes journal root: {target}",
            ) from exc
        candidates = [root]
        current = root
        for part in relative.parts:
            current /= part
            candidates.append(current)
        if self.operational_root.is_symlink():
            raise _failure(
                OperationalErrorCode.STORAGE_UNAVAILABLE,
                f"unsafe operational symlink: {self.operational_root}",
            )
        for candidate in candidates:
            if candidate.is_symlink():
                raise _failure(
                    OperationalErrorCode.STORAGE_UNAVAILABLE,
                    f"unsafe operational symlink: {candidate}",
                )

    def _ensure_directory(self, path: Path) -> None:
        self._assert_safe_path(path)
        if not self.operational_root.exists():
            try:
                self.operational_root.mkdir(mode=0o700, parents=True)
                _fsync_directory(self.operational_root.parent)
            except FileExistsError:
                pass
            except OSError as exc:
                raise _failure(
                    OperationalErrorCode.STORAGE_UNAVAILABLE,
                    f"cannot create operational root: {self.operational_root}",
                ) from exc
        if not self.operational_root.is_dir() or self.operational_root.is_symlink():
            raise _failure(
                OperationalErrorCode.STORAGE_UNAVAILABLE,
                f"unsafe operational root: {self.operational_root}",
            )
        missing: list[Path] = []
        cursor = path
        while cursor != self.operational_root and not cursor.exists():
            missing.append(cursor)
            cursor = cursor.parent
        if cursor.is_symlink():
            raise _failure(
                OperationalErrorCode.STORAGE_UNAVAILABLE,
                f"unsafe operational symlink: {cursor}",
            )
        for directory in reversed(missing):
            self._assert_safe_path(directory)
            try:
                directory.mkdir(mode=0o700)
                _fsync_directory(directory.parent)
            except FileExistsError:
                if not directory.is_dir() or directory.is_symlink():
                    raise _failure(
                        OperationalErrorCode.STORAGE_UNAVAILABLE,
                        f"unsafe operational directory: {directory}",
                    ) from None
            except OSError as exc:
                raise _failure(
                    OperationalErrorCode.STORAGE_UNAVAILABLE,
                    f"cannot create operational directory: {directory}",
                ) from exc
        if path.exists() and (not path.is_dir() or path.is_symlink()):
            raise _failure(
                OperationalErrorCode.STORAGE_UNAVAILABLE,
                f"unsafe operational directory: {path}",
            )

    def _atomic_write_json(self, path: Path, value: object) -> None:
        self._assert_safe_path(path)
        self._ensure_directory(path.parent)
        try:
            atomic_write_text(path, canonical_json_bytes(value).decode("utf-8"))
            _fsync_directory(path.parent)
        except (OSError, ValueError) as exc:
            raise _failure(
                OperationalErrorCode.STORAGE_UNAVAILABLE,
                f"cannot atomically write operational record: {path}",
            ) from exc

    def _atomic_write_bytes(self, path: Path, value: bytes) -> None:
        self._assert_safe_path(path)
        self._ensure_directory(path.parent)
        if path.is_symlink():
            raise _failure(
                OperationalErrorCode.STORAGE_UNAVAILABLE,
                f"unsafe operational symlink: {path}",
            )
        descriptor = -1
        temporary: Path | None = None
        try:
            descriptor, name = tempfile.mkstemp(
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
            )
            temporary = Path(name)
            with os.fdopen(descriptor, "wb") as handle:
                descriptor = -1
                handle.write(value)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary, 0o600)
            os.replace(temporary, path)
            _fsync_directory(path.parent)
        except OSError as exc:
            raise _failure(
                OperationalErrorCode.STORAGE_UNAVAILABLE,
                f"cannot atomically write operational bytes: {path}",
            ) from exc
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def _latest_roster(self) -> VerificationRoster:
        if self.verifier is None:
            raise _failure(
                OperationalErrorCode.SIGNATURE_INVALID,
                "operational verification authority is unavailable",
            )
        if self.roster_root is not None:
            try:
                return VerificationRoster.load(
                    self.roster_root,
                    pin_store=self.pin_store,
                )
            except OperationalError:
                raise
            except Exception as exc:
                raise _failure(
                    OperationalErrorCode.SIGNATURE_INVALID,
                    "operational roster authority is unavailable",
                ) from exc
        if self.roster is None:
            raise _failure(
                OperationalErrorCode.SIGNATURE_INVALID,
                "operational roster authority is unavailable",
            )
        return self.roster

    def _roster_for(self, version: int) -> VerificationRoster:
        if self.verifier is None:
            raise _failure(
                OperationalErrorCode.SIGNATURE_INVALID,
                "operational verification authority is unavailable",
            )
        if self.roster_root is not None:
            try:
                return VerificationRoster.load_version(
                    self.roster_root,
                    version=version,
                    pin_store=self.pin_store,
                )
            except OperationalError:
                raise
            except Exception as exc:
                raise _failure(
                    OperationalErrorCode.SIGNATURE_INVALID,
                    f"verification roster version is unavailable: {version}",
                ) from exc
        if self.roster is None or self.roster.version != version:
            raise _failure(
                OperationalErrorCode.SIGNATURE_INVALID,
                f"verification roster version is unavailable: {version}",
            )
        return self.roster

    def _signing_authority(self) -> tuple[OperationalSigner, VerificationRoster]:
        roster = self._latest_roster()
        if self.signer is None:
            raise _failure(
                OperationalErrorCode.SIGNATURE_INVALID,
                "operational signing authority is unavailable",
            )
        if self.signer.roster_version != roster.version:
            raise _failure(
                OperationalErrorCode.SIGNATURE_INVALID,
                "operational signer roster is stale",
            )
        return self.signer, roster

    def _verification_authority(
        self, version: int
    ) -> tuple[OperationalVerifier, VerificationRoster]:
        roster = self._roster_for(version)
        if self.verifier is None:
            raise _failure(
                OperationalErrorCode.SIGNATURE_INVALID,
                "operational verification authority is unavailable",
            )
        return self.verifier, roster

    @staticmethod
    def _active_key(
        roster: VerificationRoster,
        *,
        device_id: str,
        role: str,
        activation_sequence: int,
        exclusive_role: bool = False,
    ) -> PublicKeyRecord:
        matches = [
            key
            for key in roster.keys
            if key.device_id == device_id
            and role in key.roles
            and (not exclusive_role or key.roles == (role,))
            and key.enrollment_sequence <= activation_sequence
            and (key.revocation_sequence is None or activation_sequence < key.revocation_sequence)
        ]
        if len(matches) != 1:
            raise _failure(
                OperationalErrorCode.SIGNATURE_INVALID,
                f"origin has no unique active {role} key: {device_id}",
            )
        return matches[0]

    def _anchor_path(self, origin: str) -> Path:
        self._validate_safe_id(origin, "origin device")
        path = self.anchors_dir / f"{origin}.json"
        self._assert_safe_path(path)
        return path

    def _checkpoint_path(self, anchor: ChainAnchor) -> Path:
        self._validate_safe_id(anchor.origin_device, "origin device")
        self._validate_safe_id(anchor.anchor_id, "anchor id")
        path = self.checkpoints_dir / anchor.origin_device / f"{anchor.anchor_id}.json"
        self._assert_safe_path(path)
        return path

    def _head_path(self, origin: str) -> Path:
        self._validate_safe_id(origin, "origin device")
        path = self.heads_dir / f"{origin}.json"
        self._assert_safe_path(path)
        return path

    def _segment_path(self, event: OperationalEventV2) -> Path:
        self._validate_safe_id(event.origin_device, "origin device")
        day = _parse_timestamp(event.created_at, "created_at").date().isoformat()
        path = self.events_dir / event.origin_device / f"{day}.jsonl"
        self._assert_safe_path(path)
        return path

    def _decode_anchor_bytes(self, encoded: bytes, description: str) -> ChainAnchor:
        try:
            value = json.loads(encoded.decode("utf-8"))
            if (
                not isinstance(value, dict)
                or set(value) != _ANCHOR_FIELDS
                or canonical_json_bytes(value) != encoded
            ):
                raise ValueError
            return ChainAnchor(**value)
        except (
            TypeError,
            ValueError,
            KeyError,
            UnicodeDecodeError,
            json.JSONDecodeError,
        ) as exc:
            raise _failure(
                OperationalErrorCode.ANCHOR_CONFLICT,
                f"invalid canonical operational anchor: {description}",
            ) from exc

    def _decode_identity(self, value: object) -> PrincipalIdentity:
        if not isinstance(value, dict) or set(value) != _IDENTITY_FIELDS:
            raise _failure(
                OperationalErrorCode.INVALID_EVENT,
                "operational event identity fields are invalid",
            )
        try:
            return PrincipalIdentity(**value)
        except TypeError as exc:
            raise _failure(
                OperationalErrorCode.INVALID_EVENT,
                "operational event identity is invalid",
            ) from exc

    def _decode_source_proof(self, value: object) -> SourceProof | None:
        if value is None:
            return None
        if not isinstance(value, dict) or set(value) != _SOURCE_PROOF_FIELDS:
            raise _failure(
                OperationalErrorCode.INVALID_EVENT,
                "operational source proof fields are invalid",
            )
        actor = value.get("source_actor")
        if not isinstance(actor, dict):
            raise _failure(
                OperationalErrorCode.INVALID_EVENT,
                "operational source proof actor is invalid",
            )
        try:
            return SourceProof(**value)
        except TypeError as exc:
            raise _failure(
                OperationalErrorCode.INVALID_EVENT,
                "operational source proof is invalid",
            ) from exc

    def _decode_event_bytes(self, encoded: bytes, description: str) -> OperationalEventV2:
        try:
            value = json.loads(encoded.decode("utf-8"))
            if (
                not isinstance(value, dict)
                or set(value) != _EVENT_FIELDS
                or canonical_json_bytes(value) != encoded
            ):
                raise ValueError
            actor = self._decode_identity(value["actor"])
            source_proof = self._decode_source_proof(value["source_proof"])
            caused_by = value["caused_by"]
            payload = value["payload"]
            if not isinstance(caused_by, list) or not all(
                isinstance(item, str) for item in caused_by
            ):
                raise ValueError
            if not isinstance(payload, dict):
                raise ValueError
            body = dict(value)
            body["actor"] = actor
            body["source_proof"] = source_proof
            body["caused_by"] = tuple(caused_by)
            body["payload"] = payload
            return OperationalEventV2(**body)
        except OperationalError:
            raise
        except (
            TypeError,
            ValueError,
            KeyError,
            UnicodeDecodeError,
            json.JSONDecodeError,
        ) as exc:
            raise _failure(
                OperationalErrorCode.INVALID_EVENT,
                f"invalid canonical operational event: {description}",
            ) from exc

    def _read_bytes(self, path: Path, description: str) -> bytes:
        self._assert_safe_path(path)
        try:
            return path.read_bytes()
        except FileNotFoundError:
            raise _failure(
                OperationalErrorCode.NOT_FOUND,
                f"{description} is missing: {path}",
            ) from None
        except OSError as exc:
            raise _failure(
                OperationalErrorCode.STORAGE_UNAVAILABLE,
                f"cannot read {description}: {path}",
            ) from exc

    def _validate_checkpoint(self, anchor: ChainAnchor, checkpoint: bytes) -> None:
        if not isinstance(checkpoint, bytes):
            raise _failure(
                OperationalErrorCode.ANCHOR_CONFLICT,
                "anchor checkpoint must be raw bytes",
            )
        try:
            parsed = json.loads(checkpoint.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise _failure(
                OperationalErrorCode.ANCHOR_CONFLICT,
                "anchor checkpoint must be canonical JSON reducer-state bytes",
            ) from exc
        if canonical_json_bytes(parsed) != checkpoint:
            raise _failure(
                OperationalErrorCode.ANCHOR_CONFLICT,
                "anchor checkpoint must be canonical JSON reducer-state bytes",
            )
        if anchor.reducer_version != self.reducer_version:
            raise _failure(
                OperationalErrorCode.ANCHOR_CONFLICT,
                "anchor reducer version is unsupported",
            )
        digest = hashlib.sha256(checkpoint).hexdigest()
        if anchor.state_sha256 != digest:
            raise _failure(
                OperationalErrorCode.ANCHOR_CONFLICT,
                "anchor checkpoint state digest mismatch",
            )

    @staticmethod
    def _validate_anchor_shape(anchor: ChainAnchor) -> None:
        if not isinstance(anchor, ChainAnchor):
            raise _failure(
                OperationalErrorCode.ANCHOR_CONFLICT,
                "operational anchor is required",
            )
        for field in (
            "schema",
            "anchor_id",
            "origin_device",
            "kind",
            "base_event_hash",
            "final_event_hash",
            "previous_anchor_hash",
            "source_manifest_sha256",
            "state_sha256",
            "checkpoint_id",
            "checkpoint_sha256",
            "created_at",
            "anchor_hash",
            "signer_role",
            "attested_origin",
            "key_id",
            "signature",
        ):
            _require_string(getattr(anchor, field), f"anchor {field}", allow_empty=True)
        for field, minimum in (
            ("ledger_epoch", 0),
            ("reducer_version", 1),
            ("base_sequence", 0),
            ("final_sequence", 0),
            ("checkpoint_size", 0),
            ("roster_version", 1),
        ):
            _require_int(getattr(anchor, field), f"anchor {field}", minimum=minimum)

    @staticmethod
    def _validate_identity_shape(identity: PrincipalIdentity) -> None:
        if not isinstance(identity, PrincipalIdentity):
            raise _failure(
                OperationalErrorCode.INVALID_EVENT,
                "principal identity is required",
            )
        for field in (
            "principal_id",
            "actor_id",
            "kind",
            "device_id",
            "session_id",
            "source_client",
            "signature",
            "key_id",
        ):
            _require_string(
                getattr(identity, field),
                f"principal {field}",
                allow_empty=True,
            )

    @classmethod
    def _validate_source_proof_shape(cls, proof: SourceProof) -> None:
        if not isinstance(proof, SourceProof):
            raise _failure(
                OperationalErrorCode.INVALID_EVENT,
                "operational source proof is required",
            )
        for field in (
            "source_system",
            "source_event_id",
            "source_schema",
            "source_origin",
            "source_previous_hash",
            "source_event_hash",
            "source_content_hash",
            "source_subject_uri",
        ):
            _require_string(
                getattr(proof, field),
                f"source proof {field}",
                allow_empty=True,
            )
        _require_int(
            proof.source_sequence,
            "source proof source_sequence",
            minimum=0,
        )
        if not isinstance(proof.source_actor, Mapping) or not all(
            isinstance(key, str) for key in proof.source_actor
        ):
            raise _failure(
                OperationalErrorCode.INVALID_EVENT,
                "source proof actor must be a string-keyed mapping",
            )

    @classmethod
    def _validate_event_shape(cls, event: OperationalEventV2) -> None:
        if not isinstance(event, OperationalEventV2):
            raise _failure(
                OperationalErrorCode.INVALID_EVENT,
                "operational event is required",
            )
        for field in (
            "schema",
            "event_id",
            "event_type",
            "project",
            "workspace",
            "origin_device",
            "logical_clock",
            "control_oid",
            "created_at",
            "visibility",
            "idempotency_key",
            "subject_uri",
            "trace_id",
            "content_hash",
            "previous_hash",
            "event_hash",
            "key_id",
            "signature",
        ):
            _require_string(
                getattr(event, field),
                f"event {field}",
                allow_empty=True,
            )
        for field, minimum in (
            ("schema_version", 1),
            ("origin_sequence", 1),
            ("authority_epoch", 0),
            ("roster_version", 1),
        ):
            _require_int(getattr(event, field), f"event {field}", minimum=minimum)
        if event.target_id is not None and not isinstance(event.target_id, str):
            raise _failure(
                OperationalErrorCode.INVALID_EVENT,
                "event target_id must be a string or null",
            )
        if event.expires_at is not None and not isinstance(event.expires_at, str):
            raise _failure(
                OperationalErrorCode.INVALID_EVENT,
                "event expires_at must be a string or null",
            )
        if not isinstance(event.caused_by, tuple) or not all(
            isinstance(item, str) for item in event.caused_by
        ):
            raise _failure(
                OperationalErrorCode.INVALID_EVENT,
                "event caused_by must be a tuple of strings",
            )
        if not isinstance(event.payload, Mapping) or not all(
            isinstance(key, str) for key in event.payload
        ):
            raise _failure(
                OperationalErrorCode.INVALID_EVENT,
                "event payload must be a string-keyed mapping",
            )
        cls._validate_identity_shape(event.actor)
        if event.source_proof is not None:
            cls._validate_source_proof_shape(event.source_proof)

    def _validate_anchor_authority(self, anchor: ChainAnchor, checkpoint: bytes) -> None:
        self._validate_anchor_shape(anchor)
        self._validate_safe_id(anchor.origin_device, "origin device")
        self._validate_safe_id(anchor.anchor_id, "anchor id")
        self._validate_safe_id(anchor.checkpoint_id, "checkpoint id")
        _parse_timestamp(anchor.created_at, "anchor created_at")
        self._validate_checkpoint(anchor, checkpoint)
        verifier, roster = self._verification_authority(anchor.roster_version)
        if anchor.origin_device not in roster.peers:
            raise _failure(
                OperationalErrorCode.SIGNATURE_INVALID,
                f"anchor origin is not enrolled: {anchor.origin_device}",
            )
        validate_anchor(
            anchor,
            checkpoint=checkpoint,
            roster=roster,
            verifier=verifier,
        )

    def _read_anchor_with_checkpoint(self, origin: str) -> tuple[ChainAnchor, bytes]:
        path = self._anchor_path(origin)
        anchor = self._decode_anchor_bytes(
            self._read_bytes(path, "operational anchor"),
            str(path),
        )
        if anchor.origin_device != origin:
            raise _failure(
                OperationalErrorCode.ANCHOR_CONFLICT,
                f"anchor origin/path mismatch: {path}",
            )
        checkpoint = self._read_bytes(
            self._checkpoint_path(anchor),
            "operational checkpoint",
        )
        self._validate_anchor_authority(anchor, checkpoint)
        return anchor, checkpoint

    def anchor(self, origin_device: str | None = None) -> ChainAnchor:
        origin = origin_device or self.device_id
        with authority_write_lock(self.root):
            return self._read_anchor_with_checkpoint(origin)[0]

    def _make_empty_anchor(self) -> tuple[ChainAnchor, bytes]:
        signer, roster = self._signing_authority()
        checkpoint = EMPTY_REDUCER_STATE_BYTES
        key = self._active_key(
            roster,
            device_id=self.device_id,
            role="origin",
            activation_sequence=roster.version,
        )
        created_at = self._now()
        digest = hashlib.sha256(checkpoint).hexdigest()
        anchor_id = f"empty-{hashlib.sha256(f'{self.device_id}:{created_at}'.encode()).hexdigest()}"
        unsigned = ChainAnchor(
            schema="memo.operational_anchor.v1",
            anchor_id=anchor_id,
            origin_device=self.device_id,
            ledger_epoch=0,
            reducer_version=self.reducer_version,
            kind="empty",
            base_sequence=0,
            base_event_hash="",
            final_sequence=0,
            final_event_hash="",
            previous_anchor_hash="",
            source_manifest_sha256="",
            state_sha256=digest,
            checkpoint_id=f"checkpoint-{anchor_id}",
            checkpoint_sha256=digest,
            checkpoint_size=len(checkpoint),
            created_at=created_at,
            anchor_hash="",
            roster_version=roster.version,
            signer_role="origin",
            attested_origin=self.device_id,
            key_id=key.key_id,
            signature="",
        )
        hashed = replace(unsigned, anchor_hash=canonical_anchor_hash(unsigned))
        envelope = signer.sign(
            domain="memo.operational.anchor.v1",
            payload=canonical_signed_bytes(hashed),
            key_id=key.key_id,
        )
        anchor = replace(hashed, signature=envelope.signature)
        self._validate_anchor_authority(anchor, checkpoint)
        return anchor, checkpoint

    def _write_head(
        self,
        *,
        origin: str,
        sequence: int,
        event_hash: str,
        anchor_hash: str,
    ) -> None:
        self._atomic_write_json(
            self._head_path(origin),
            {
                "schema": _HEAD_SCHEMA,
                "origin_device": origin,
                "sequence": sequence,
                "event_hash": event_hash,
                "anchor_hash": anchor_hash,
            },
        )

    def _persist_anchor(self, anchor: ChainAnchor, checkpoint: bytes) -> None:
        self._validate_anchor_authority(anchor, checkpoint)
        self._atomic_write_bytes(self._checkpoint_path(anchor), checkpoint)
        self._atomic_write_json(self._anchor_path(anchor.origin_device), asdict(anchor))
        self._write_head(
            origin=anchor.origin_device,
            sequence=anchor.base_sequence,
            event_hash=anchor.base_event_hash,
            anchor_hash=anchor.anchor_hash,
        )

    def _ensure_anchor_locked(
        self,
        anchor: ChainAnchor | None = None,
        *,
        checkpoint: bytes | None = None,
    ) -> ChainAnchor:
        if anchor is not None:
            self._validate_anchor_shape(anchor)
        origin = anchor.origin_device if anchor is not None else self.device_id
        path = self._anchor_path(origin)
        if anchor is None and path.exists():
            return self._read_anchor_with_checkpoint(origin)[0]
        if anchor is None:
            anchor, checkpoint = self._make_empty_anchor()
        if checkpoint is None:
            if anchor.kind in {"empty", "memo_v1"}:
                checkpoint = EMPTY_REDUCER_STATE_BYTES
            else:
                raise _failure(
                    OperationalErrorCode.ANCHOR_CONFLICT,
                    "compaction anchor requires matching checkpoint bytes",
                )
        self._validate_anchor_authority(anchor, checkpoint)
        if path.exists():
            current, current_checkpoint = self._read_anchor_with_checkpoint(origin)
            if current.anchor_hash == anchor.anchor_hash:
                if current_checkpoint != checkpoint:
                    raise _failure(
                        OperationalErrorCode.ANCHOR_CONFLICT,
                        "identical anchor has different checkpoint bytes",
                    )
                return current
            position = self._load_position(origin, current, repair=False)
            if (
                anchor.kind != "compaction"
                or anchor.previous_anchor_hash != current.anchor_hash
                or anchor.ledger_epoch < current.ledger_epoch
                or anchor.base_sequence != position.sequence
                or anchor.base_event_hash != position.event_hash
            ):
                raise _failure(
                    OperationalErrorCode.ANCHOR_CONFLICT,
                    f"anchor regression or conflict for origin {origin}",
                )
        self._persist_anchor(anchor, checkpoint)
        return anchor

    def ensure_anchor(
        self,
        anchor: ChainAnchor | None = None,
        *,
        checkpoint: bytes | None = None,
    ) -> ChainAnchor:
        self._assert_safe_path(self.root)
        with authority_write_lock(self.root):
            self._assert_safe_path(self.root)
            return self._ensure_anchor_locked(anchor, checkpoint=checkpoint)

    @staticmethod
    def _legacy_snapshot(
        legacy_ledger: LegacyOperationLedger,
    ) -> tuple[dict[str, object], list[MemoEvent], dict[str, str]]:
        """Freeze verified v1 authority into a content-addressed file manifest."""
        with authority_write_lock(legacy_ledger.root):
            report = legacy_ledger.verify()
            if not bool(report.get("ok")):
                raise _failure(
                    OperationalErrorCode.ANCHOR_CONFLICT,
                    "legacy ledger verification failed",
                    details={"report": report},
                )
            events = legacy_ledger.validated_events()
            heads = legacy_ledger.head_hashes()
            paths = {
                legacy_ledger.root
                / "events"
                / event.device_id
                / f"{_parse_timestamp(event.ts, 'legacy event timestamp').date().isoformat()}.jsonl"
                for event in events
            }
            paths.update(legacy_ledger.root / "heads" / f"{origin}.json" for origin in heads)
            files: list[dict[str, object]] = []
            root = legacy_ledger.root.absolute()
            for path in sorted(paths, key=lambda item: item.as_posix()):
                target = path.absolute()
                try:
                    relative = target.relative_to(root)
                except ValueError as exc:
                    raise _failure(
                        OperationalErrorCode.ANCHOR_CONFLICT,
                        f"legacy manifest path escapes journal root: {target}",
                    ) from exc
                current = root
                for part in relative.parts:
                    current /= part
                    if current.is_symlink():
                        raise _failure(
                            OperationalErrorCode.ANCHOR_CONFLICT,
                            f"legacy manifest contains a symlink: {current}",
                        )
                try:
                    encoded = target.read_bytes()
                except OSError as exc:
                    raise _failure(
                        OperationalErrorCode.ANCHOR_CONFLICT,
                        f"cannot retain verified legacy bytes: {target}",
                    ) from exc
                files.append(
                    {
                        "path": relative.as_posix(),
                        "size": len(encoded),
                        "sha256": hashlib.sha256(encoded).hexdigest(),
                    }
                )
            manifest: dict[str, object] = {
                "schema": "memo.operational_v1_manifest.v1",
                "files": files,
                "heads": heads,
            }
            return manifest, events, heads

    @classmethod
    def _legacy_manifest(cls, legacy_ledger: LegacyOperationLedger) -> dict[str, object]:
        return cls._legacy_snapshot(legacy_ledger)[0]

    @classmethod
    def legacy_manifest_sha256(cls, legacy_ledger: LegacyOperationLedger) -> str:
        """Hash verified v1 authority bytes without mutating the frozen reader."""
        return hashlib.sha256(canonical_json_bytes(cls._legacy_manifest(legacy_ledger))).hexdigest()

    def ensure_anchor_from_v1(
        self,
        legacy_ledger: LegacyOperationLedger,
        *,
        source_head_hash: str,
        migration_attestor: OperationalSigner,
        attestor_key_id: str,
        checkpoint: bytes = EMPTY_REDUCER_STATE_BYTES,
        ledger_epoch: int = 0,
    ) -> ChainAnchor:
        """Build and persist a memo-v1 genesis anchor from verified source bytes."""
        if not isinstance(legacy_ledger, LegacyOperationLedger):
            raise TypeError("legacy_ledger must be a LegacyOperationLedger")
        if not _SHA256_RE.fullmatch(source_head_hash):
            raise _failure(
                OperationalErrorCode.ANCHOR_CONFLICT,
                "legacy source head hash is invalid",
            )
        manifest, events, heads = self._legacy_snapshot(legacy_ledger)
        actual_head = heads.get(legacy_ledger.device_id)
        if actual_head != source_head_hash:
            raise _failure(
                OperationalErrorCode.ANCHOR_CONFLICT,
                "legacy source head does not match verified head",
            )
        source_events = [event for event in events if event.device_id == legacy_ledger.device_id]
        if not source_events or source_events[-1].event_hash != source_head_hash:
            raise _failure(
                OperationalErrorCode.ANCHOR_CONFLICT,
                "legacy source head event is missing",
            )
        source = source_events[-1]
        roster = self._latest_roster()
        if migration_attestor.roster_version != roster.version:
            raise _failure(
                OperationalErrorCode.SIGNATURE_INVALID,
                "migration attestor roster is stale",
            )
        key = roster.key(attestor_key_id)
        activation = max(source.sequence, roster.version)
        expected = self._active_key(
            roster,
            device_id=key.device_id,
            role="migration_attestor",
            activation_sequence=activation,
            exclusive_role=True,
        )
        if expected.key_id != attestor_key_id:
            raise _failure(
                OperationalErrorCode.SIGNATURE_INVALID,
                "migration attestor key is not the active exclusive key",
            )
        source_manifest_sha256 = hashlib.sha256(canonical_json_bytes(manifest)).hexdigest()
        checkpoint_digest = hashlib.sha256(checkpoint).hexdigest()
        anchor_id = (
            "memo-v1-"
            + hashlib.sha256(
                canonical_json_bytes(
                    {
                        "origin": legacy_ledger.device_id,
                        "head": source_head_hash,
                        "manifest": source_manifest_sha256,
                    }
                )
            ).hexdigest()
        )
        unsigned = ChainAnchor(
            schema="memo.operational_anchor.v1",
            anchor_id=anchor_id,
            origin_device=legacy_ledger.device_id,
            ledger_epoch=ledger_epoch,
            reducer_version=self.reducer_version,
            kind="memo_v1",
            base_sequence=source.sequence,
            base_event_hash=source.event_hash,
            final_sequence=source.sequence,
            final_event_hash=source.event_hash,
            previous_anchor_hash="",
            source_manifest_sha256=source_manifest_sha256,
            state_sha256=checkpoint_digest,
            checkpoint_id=f"checkpoint-{anchor_id}",
            checkpoint_sha256=checkpoint_digest,
            checkpoint_size=len(checkpoint),
            created_at=self._now(),
            anchor_hash="",
            roster_version=roster.version,
            signer_role="migration_attestor",
            attested_origin=legacy_ledger.device_id,
            key_id=attestor_key_id,
            signature="",
        )
        hashed = replace(unsigned, anchor_hash=canonical_anchor_hash(unsigned))
        envelope = migration_attestor.sign(
            domain="memo.operational.anchor.v1",
            payload=canonical_signed_bytes(hashed),
            key_id=attestor_key_id,
        )
        return self.ensure_anchor(
            replace(hashed, signature=envelope.signature),
            checkpoint=checkpoint,
        )

    def _decode_head(self, origin: str) -> tuple[int, str, str] | None:
        path = self._head_path(origin)
        self._assert_safe_path(path)
        try:
            encoded = path.read_bytes()
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise _failure(
                OperationalErrorCode.STORAGE_UNAVAILABLE,
                f"cannot read operational head: {path}",
            ) from exc
        try:
            value = json.loads(encoded.decode("utf-8"))
            if (
                not isinstance(value, dict)
                or set(value) != _HEAD_FIELDS
                or canonical_json_bytes(value) != encoded
                or value["schema"] != _HEAD_SCHEMA
                or value["origin_device"] != origin
            ):
                raise ValueError
            sequence = _require_int(value["sequence"], "head sequence")
            event_hash = _require_string(value["event_hash"], "head event hash", allow_empty=True)
            anchor_hash = _require_string(value["anchor_hash"], "head anchor hash")
            return sequence, event_hash, anchor_hash
        except OperationalError:
            raise
        except (
            KeyError,
            TypeError,
            ValueError,
            UnicodeDecodeError,
            json.JSONDecodeError,
        ) as exc:
            raise _failure(
                OperationalErrorCode.ANCHOR_CONFLICT,
                f"invalid canonical operational head: {path}",
            ) from exc

    def _event_segment_paths(self, origin: str) -> list[Path]:
        self._validate_safe_id(origin, "origin device")
        event_dir = self.events_dir / origin
        self._assert_safe_path(event_dir)
        if not event_dir.exists():
            return []
        if not event_dir.is_dir() or event_dir.is_symlink():
            raise _failure(
                OperationalErrorCode.STORAGE_UNAVAILABLE,
                f"unsafe operational event directory: {event_dir}",
            )
        paths: list[Path] = []
        try:
            entries = sorted(event_dir.iterdir(), key=lambda item: item.name)
        except OSError as exc:
            raise _failure(
                OperationalErrorCode.STORAGE_UNAVAILABLE,
                f"cannot list operational event directory: {event_dir}",
            ) from exc
        for path in entries:
            self._assert_safe_path(path)
            if path.is_symlink():
                raise _failure(
                    OperationalErrorCode.STORAGE_UNAVAILABLE,
                    f"unsafe operational event symlink: {path}",
                )
            if not path.is_file() or path.suffix != ".jsonl":
                raise _failure(
                    OperationalErrorCode.INVALID_EVENT,
                    f"unexpected operational event path: {path}",
                )
            try:
                datetime.strptime(path.stem, "%Y-%m-%d")
            except ValueError as exc:
                raise _failure(
                    OperationalErrorCode.INVALID_EVENT,
                    f"invalid operational segment day: {path}",
                ) from exc
            paths.append(path)
        return paths

    def _verify_event_signature(self, event: OperationalEventV2) -> None:
        verifier, roster = self._verification_authority(event.roster_version)
        if event.origin_device not in roster.peers:
            raise _failure(
                OperationalErrorCode.SIGNATURE_INVALID,
                f"event origin is not enrolled: {event.origin_device}",
            )
        verifier.verify(
            domain="memo.operational.event.v2",
            payload=canonical_signed_bytes(event),
            envelope=SignatureEnvelope(
                algorithm="ed25519",
                key_id=event.key_id,
                roster_version=event.roster_version,
                signature=event.signature,
            ),
            roster=roster,
        )

    def _validate_source_proof(self, proof: SourceProof | None, anchor: ChainAnchor) -> None:
        if proof is None:
            return
        self._validate_source_proof_shape(proof)
        if anchor.kind != "memo_v1":
            raise _failure(
                OperationalErrorCode.INVALID_EVENT,
                "source proof requires a memo_v1 anchor",
            )
        if (
            proof.source_system != "memo_v1"
            or proof.source_origin != anchor.origin_device
            or proof.source_sequence < 1
            or proof.source_sequence > anchor.base_sequence
            or not proof.source_event_id
            or not proof.source_schema
            or not _SHA256_RE.fullmatch(proof.source_event_hash)
        ):
            raise _failure(
                OperationalErrorCode.INVALID_EVENT,
                "source proof is not linked to the sealed memo_v1 source",
            )
        if (
            proof.source_sequence == anchor.base_sequence
            and proof.source_event_hash != anchor.base_event_hash
        ):
            raise _failure(
                OperationalErrorCode.INVALID_EVENT,
                "source proof does not match the sealed v1 head",
            )

    def _read_events_chain(self, origin: str, anchor: ChainAnchor) -> list[OperationalEventV2]:
        rows: list[OperationalEventV2] = []
        expected_sequence = anchor.base_sequence + 1
        expected_previous = anchor.base_event_hash
        previous_time: datetime | None = None
        previous_seen_sequence = 0
        for path in self._event_segment_paths(origin):
            try:
                segment_bytes = path.read_bytes()
            except OSError as exc:
                raise _failure(
                    OperationalErrorCode.STORAGE_UNAVAILABLE,
                    f"cannot read operational segment: {path}",
                ) from exc
            if not segment_bytes or not segment_bytes.endswith(b"\n"):
                raise _failure(
                    OperationalErrorCode.INVALID_EVENT,
                    f"operational segment has an incomplete final row: {path}",
                )
            lines = segment_bytes.split(b"\n")[:-1]
            for line_number, encoded in enumerate(lines, start=1):
                if not encoded:
                    raise _failure(
                        OperationalErrorCode.INVALID_EVENT,
                        f"blank operational event row: {path}:{line_number}",
                    )
                event = self._decode_event_bytes(
                    encoded,
                    f"{path}:{line_number}",
                )
                self._validate_event_shape(event)
                if event.origin_device != origin:
                    raise _failure(
                        OperationalErrorCode.INVALID_EVENT,
                        f"event origin/path mismatch: {path}:{line_number}",
                    )
                event_time = _parse_timestamp(event.created_at, "created_at")
                if event_time.date().isoformat() != path.stem:
                    raise _failure(
                        OperationalErrorCode.INVALID_EVENT,
                        f"event timestamp/path mismatch: {path}:{line_number}",
                    )
                if event.origin_sequence <= previous_seen_sequence:
                    raise _failure(
                        OperationalErrorCode.SEQUENCE_GAP,
                        f"origin {origin} event sequences are not strictly increasing",
                    )
                previous_seen_sequence = event.origin_sequence
                validate_event(event)
                self._verify_event_signature(event)
                if event.origin_sequence <= anchor.base_sequence:
                    if (
                        event.origin_sequence == anchor.base_sequence
                        and event.event_hash != anchor.base_event_hash
                    ):
                        raise _failure(
                            OperationalErrorCode.ANCHOR_CONFLICT,
                            f"retained event does not match compacted anchor for {origin}",
                        )
                    continue
                if event.origin_sequence != expected_sequence:
                    raise _failure(
                        OperationalErrorCode.SEQUENCE_GAP,
                        (
                            f"origin {origin} sequence {event.origin_sequence} "
                            f"!= expected {expected_sequence}"
                        ),
                    )
                if event.previous_hash != expected_previous:
                    raise _failure(
                        OperationalErrorCode.ANCHOR_CONFLICT,
                        f"previous hash mismatch at origin {origin} sequence {expected_sequence}",
                    )
                if previous_time is not None and event_time < previous_time:
                    raise _failure(
                        OperationalErrorCode.INVALID_EVENT,
                        f"event timestamps decrease for origin {origin}",
                    )
                self._validate_source_proof(event.source_proof, anchor)
                rows.append(event)
                expected_sequence += 1
                expected_previous = event.event_hash
                previous_time = event_time
        return rows

    def _load_position(
        self,
        origin: str,
        anchor: ChainAnchor,
        *,
        repair: bool,
    ) -> OriginPosition:
        events = self._read_events_chain(origin, anchor)
        tail_sequence = events[-1].origin_sequence if events else anchor.base_sequence
        tail_hash = events[-1].event_hash if events else anchor.base_event_hash
        cached = self._decode_head(origin)
        expected = (tail_sequence, tail_hash, anchor.anchor_hash)
        if cached == expected:
            return OriginPosition(
                origin_device=origin,
                sequence=tail_sequence,
                event_hash=tail_hash,
                anchor_hash=anchor.anchor_hash,
            )
        positions = {
            anchor.base_sequence: anchor.base_event_hash,
            **{event.origin_sequence: event.event_hash for event in events},
        }
        repairable = cached is None or (
            cached[2] == anchor.anchor_hash
            and cached[0] < tail_sequence
            and positions.get(cached[0]) == cached[1]
        )
        if repair and repairable:
            self._write_head(
                origin=origin,
                sequence=tail_sequence,
                event_hash=tail_hash,
                anchor_hash=anchor.anchor_hash,
            )
        else:
            raise _failure(
                OperationalErrorCode.ANCHOR_CONFLICT,
                f"operational head fork or mismatch for origin {origin}",
            )
        return OriginPosition(
            origin_device=origin,
            sequence=tail_sequence,
            event_hash=tail_hash,
            anchor_hash=anchor.anchor_hash,
        )

    def position(self, origin_device: str | None = None) -> OriginPosition:
        origin = origin_device or self.device_id
        with authority_write_lock(self.root):
            anchor = self._read_anchor_with_checkpoint(origin)[0]
            return self._load_position(origin, anchor, repair=True)

    def _discover_origins(self) -> tuple[str, ...]:
        origins: set[str] = set()
        for directory, suffix, nested in (
            (self.anchors_dir, ".json", False),
            (self.heads_dir, ".json", False),
            (self.events_dir, "", True),
        ):
            self._assert_safe_path(directory)
            if not directory.exists():
                continue
            if not directory.is_dir() or directory.is_symlink():
                raise _failure(
                    OperationalErrorCode.STORAGE_UNAVAILABLE,
                    f"unsafe operational authority directory: {directory}",
                )
            for path in directory.iterdir():
                self._assert_safe_path(path)
                if path.is_symlink():
                    raise _failure(
                        OperationalErrorCode.STORAGE_UNAVAILABLE,
                        f"unsafe operational authority symlink: {path}",
                    )
                if nested:
                    if not path.is_dir():
                        raise _failure(
                            OperationalErrorCode.INVALID_EVENT,
                            f"unexpected operational event origin path: {path}",
                        )
                    origin = path.name
                else:
                    if not path.is_file() or path.suffix != suffix:
                        raise _failure(
                            OperationalErrorCode.INVALID_EVENT,
                            f"unexpected operational authority path: {path}",
                        )
                    origin = path.stem
                self._validate_safe_id(origin, "origin device")
                origins.add(origin)
        return tuple(sorted(origins))

    def positions(self) -> tuple[OriginPosition, ...]:
        with authority_write_lock(self.root):
            positions: list[OriginPosition] = []
            for origin in self._discover_origins():
                anchor = self._read_anchor_with_checkpoint(origin)[0]
                positions.append(self._load_position(origin, anchor, repair=True))
            return tuple(positions)

    def iter_events(
        self,
        *,
        origin_device: str | None = None,
        limit: int | None = None,
    ) -> list[OperationalEventV2]:
        if limit is not None and (
            isinstance(limit, bool) or not isinstance(limit, int) or limit < 0
        ):
            raise ValueError("limit must be a non-negative integer")
        with authority_write_lock(self.root):
            origins = (
                (self._validate_safe_id(origin_device, "origin device"),)
                if origin_device is not None
                else self._discover_origins()
            )
            rows: list[OperationalEventV2] = []
            for origin in origins:
                anchor = self._read_anchor_with_checkpoint(origin)[0]
                events = self._read_events_chain(origin, anchor)
                self._load_position(origin, anchor, repair=False)
                rows.extend(events)
            rows.sort(
                key=lambda event: (
                    _parse_timestamp(event.created_at, "created_at"),
                    event.origin_device,
                    event.origin_sequence,
                )
            )
            if limit == 0:
                return []
            return rows[-limit:] if limit is not None else rows

    def validated_events(self) -> list[OperationalEventV2]:
        """Compatibility spelling for consumers that require a full verification."""
        return self.iter_events()

    def _append_event_fsync(self, event: OperationalEventV2) -> None:
        path = self._segment_path(event)
        self._ensure_directory(path.parent)
        self._assert_safe_path(path)
        flags = os.O_CREAT | os.O_APPEND | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0)
        created = not path.exists()
        try:
            descriptor = os.open(path, flags, 0o600)
            with os.fdopen(descriptor, "ab") as handle:
                descriptor = -1
                handle.write(canonical_json_bytes(asdict(event)) + b"\n")
                handle.flush()
                os.fsync(handle.fileno())
            if created:
                _fsync_directory(path.parent)
        except OSError as exc:
            raise _failure(
                OperationalErrorCode.STORAGE_UNAVAILABLE,
                f"cannot append operational event: {path}",
            ) from exc

    def _write_head_atomic(self, event: OperationalEventV2) -> None:
        anchor = self._read_anchor_with_checkpoint(event.origin_device)[0]
        self._write_head(
            origin=event.origin_device,
            sequence=event.origin_sequence,
            event_hash=event.event_hash,
            anchor_hash=anchor.anchor_hash,
        )

    def _validate_migration_context(
        self,
        context: CommitContext,
        anchor: ChainAnchor,
        command: OperationalCommand,
        *,
        at_time: str,
    ) -> None:
        migration = context.migration_origin
        if migration is None:
            return
        if migration.migration_device_id != anchor.origin_device:
            raise _failure(
                OperationalErrorCode.INVALID_EVENT,
                "migration origin does not match anchored origin",
            )
        if (
            anchor.kind != "memo_v1"
            or migration.source_manifest_sha256 != anchor.source_manifest_sha256
        ):
            raise _failure(
                OperationalErrorCode.ANCHOR_CONFLICT,
                "migration origin is not bound to the sealed v1 manifest",
            )
        verifier, roster = self._verification_authority(migration.roster_version)
        validate_migration_origin(
            migration,
            roster=roster,
            verifier=verifier,
            at_time=at_time,
        )
        if command.source_proof is None:
            raise _failure(
                OperationalErrorCode.INVALID_EVENT,
                "migration event requires a source proof",
            )
        self._validate_source_proof(command.source_proof, anchor)

    def _build_event(
        self,
        command: OperationalCommand,
        *,
        context: CommitContext,
        origin: str,
        sequence: int,
        previous_hash: str,
        created_at: str,
        roster: VerificationRoster,
        key: PublicKeyRecord,
    ) -> OperationalEventV2:
        content_hash = hashlib.sha256(canonical_json_bytes(asdict(command))).hexdigest()
        unsigned = OperationalEventV2(
            schema="memo.operational_event.v2",
            schema_version=2,
            event_id=uuid.uuid4().hex,
            event_type=command.event_type,
            actor=command.actor,
            target_id=command.target_id,
            project=command.project,
            workspace=command.workspace,
            origin_device=origin,
            origin_sequence=sequence,
            logical_clock=f"{context.authority_epoch}:{sequence}",
            authority_epoch=context.authority_epoch,
            control_oid=context.control_oid,
            created_at=created_at,
            expires_at=command.expires_at,
            visibility=command.visibility,
            idempotency_key=command.idempotency_key,
            caused_by=tuple(command.caused_by),
            subject_uri=command.subject_uri,
            trace_id=command.trace_id,
            payload=dict(command.payload),
            content_hash=content_hash,
            previous_hash=previous_hash,
            event_hash="",
            source_proof=command.source_proof,
            roster_version=roster.version,
            key_id=key.key_id,
            signature="",
        )
        return replace(unsigned, event_hash=canonical_event_hash(unsigned))

    def append(
        self,
        command: OperationalCommand,
        *,
        context: CommitContext,
    ) -> OperationalEventV2:
        if not isinstance(command, OperationalCommand):
            raise _failure(
                OperationalErrorCode.INVALID_EVENT,
                "operational command is required",
            )
        if self.epoch_fence is None:
            raise _failure(
                OperationalErrorCode.INVALID_EVENT,
                "operational epoch authority is unavailable",
            )
        if not isinstance(context, CommitContext):
            raise _failure(
                OperationalErrorCode.INVALID_EVENT,
                "commit context authority is required",
            )
        validate_event_payload(command.event_type, command.payload)
        self._assert_safe_path(self.root)
        with authority_write_lock(self.root), self.epoch_fence.verified(context):
            self._assert_safe_path(self.root)
            if command.actor != context.identity:
                raise _failure(
                    OperationalErrorCode.INVALID_EVENT,
                    "operational command actor differs from authenticated context",
                )
            signer, roster = self._signing_authority()
            origin = (
                context.migration_origin.migration_device_id
                if context.migration_origin is not None
                else self.device_id
            )
            self._validate_safe_id(origin, "origin device")
            if context.migration_origin is not None and not self._anchor_path(origin).exists():
                raise _failure(
                    OperationalErrorCode.ANCHOR_CONFLICT,
                    "migration origin requires a pre-existing memo_v1 anchor",
                )
            anchor = (
                self._read_anchor_with_checkpoint(origin)[0]
                if context.migration_origin is not None
                else self._ensure_anchor_locked()
            )
            position = self._load_position(origin, anchor, repair=True)
            created_at = self._now()
            self._validate_migration_context(
                context,
                anchor,
                command,
                at_time=created_at,
            )
            activation_sequence = position.sequence + 1
            key = self._active_key(
                roster,
                device_id=origin,
                role="origin",
                activation_sequence=activation_sequence,
            )
            event = self._build_event(
                command,
                context=context,
                origin=origin,
                sequence=activation_sequence,
                previous_hash=position.event_hash,
                created_at=created_at,
                roster=roster,
                key=key,
            )
            self._validate_event_shape(event)
            envelope = signer.sign(
                domain="memo.operational.event.v2",
                payload=canonical_signed_bytes(event),
                key_id=key.key_id,
            )
            event = replace(event, signature=envelope.signature)
            validate_event(event)
            self._verify_event_signature(event)
            self._validate_source_proof(event.source_proof, anchor)
            self._append_event_fsync(event)
            self._write_head_atomic(event)
            return event

    def verify(self) -> VerificationReport:
        checked_origins: list[str] = []
        checked_events = 0
        positions: list[OriginPosition] = []
        errors: list[str] = []
        try:
            with authority_write_lock(self.root):
                for origin in self._discover_origins():
                    checked_origins.append(origin)
                    try:
                        anchor = self._read_anchor_with_checkpoint(origin)[0]
                        events = self._read_events_chain(origin, anchor)
                        position = self._load_position(origin, anchor, repair=False)
                    except Exception as exc:
                        errors.append(f"{origin}: {exc}")
                        continue
                    checked_events += len(events)
                    positions.append(position)
        except Exception as exc:
            errors.append(str(exc))
        state_sha256 = hashlib.sha256(
            canonical_json_bytes([asdict(position) for position in positions])
        ).hexdigest()
        return VerificationReport(
            ok=not errors,
            checked_origins=tuple(checked_origins),
            checked_events=checked_events,
            state_sha256=state_sha256,
            errors=tuple(errors),
        )

    def _validate_bundle(self, bundle: OriginBundle) -> None:
        if not isinstance(bundle, OriginBundle):
            raise _failure(
                OperationalErrorCode.INVALID_EVENT,
                "origin bundle is required",
            )
        anchor = bundle.anchor
        self._validate_anchor_authority(anchor, bundle.checkpoint)
        _require_int(bundle.head_sequence, "bundle head sequence", minimum=0)
        _require_string(bundle.head_hash, "bundle head hash", allow_empty=True)
        expected_sequence = anchor.base_sequence + 1
        expected_previous = anchor.base_event_hash
        previous_time: datetime | None = None
        for event in bundle.events:
            self._validate_event_shape(event)
            if event.origin_device != anchor.origin_device:
                raise _failure(
                    OperationalErrorCode.INVALID_EVENT,
                    "bundle event origin differs from anchor",
                )
            if event.origin_sequence != expected_sequence:
                raise _failure(
                    OperationalErrorCode.SEQUENCE_GAP,
                    (
                        f"bundle origin {anchor.origin_device} sequence "
                        f"{event.origin_sequence} != expected {expected_sequence}"
                    ),
                )
            if event.previous_hash != expected_previous:
                raise _failure(
                    OperationalErrorCode.ANCHOR_CONFLICT,
                    f"bundle previous hash mismatch at sequence {expected_sequence}",
                )
            event_time = _parse_timestamp(event.created_at, "created_at")
            if previous_time is not None and event_time < previous_time:
                raise _failure(
                    OperationalErrorCode.INVALID_EVENT,
                    "bundle event timestamps decrease",
                )
            validate_event(event)
            self._verify_event_signature(event)
            self._validate_source_proof(event.source_proof, anchor)
            expected_sequence += 1
            expected_previous = event.event_hash
            previous_time = event_time
        expected_head_sequence = (
            bundle.events[-1].origin_sequence if bundle.events else anchor.base_sequence
        )
        expected_head_hash = (
            bundle.events[-1].event_hash if bundle.events else anchor.base_event_hash
        )
        if bundle.head_sequence != expected_head_sequence or bundle.head_hash != expected_head_hash:
            raise _failure(
                OperationalErrorCode.ANCHOR_CONFLICT,
                "bundle final head does not match its verified chain",
            )

    def _validate_bundle_against_existing(
        self, bundle: OriginBundle
    ) -> tuple[list[OperationalEventV2], int]:
        origin = bundle.anchor.origin_device
        anchor_path = self._anchor_path(origin)
        if not anchor_path.exists():
            return list(bundle.events), 0
        current, _ = self._read_anchor_with_checkpoint(origin)
        if current.anchor_hash != bundle.anchor.anchor_hash:
            if (
                bundle.anchor.kind != "compaction"
                or bundle.anchor.previous_anchor_hash != current.anchor_hash
                or bundle.anchor.base_sequence < current.base_sequence
            ):
                raise _failure(
                    OperationalErrorCode.ANCHOR_CONFLICT,
                    f"anchor regression or fork for origin {origin}",
                )
            position = self._load_position(origin, current, repair=False)
            if (
                bundle.anchor.base_sequence != position.sequence
                or bundle.anchor.base_event_hash != position.event_hash
            ):
                raise _failure(
                    OperationalErrorCode.ANCHOR_CONFLICT,
                    f"compaction anchor does not continue origin {origin}",
                )
            return list(bundle.events), 0
        existing = self._read_events_chain(origin, current)
        by_sequence = {event.origin_sequence: event for event in existing}
        replayed = 0
        new_events: list[OperationalEventV2] = []
        for event in bundle.events:
            present = by_sequence.get(event.origin_sequence)
            if present is None:
                new_events.append(event)
            elif present.event_hash != event.event_hash:
                raise _failure(
                    OperationalErrorCode.ANCHOR_CONFLICT,
                    (f"origin {origin} fork at same-position sequence {event.origin_sequence}"),
                )
            else:
                replayed += 1
        if new_events:
            position = self._load_position(origin, current, repair=False)
            first = new_events[0]
            if (
                first.origin_sequence != position.sequence + 1
                or first.previous_hash != position.event_hash
            ):
                raise _failure(
                    OperationalErrorCode.SEQUENCE_GAP,
                    f"bundle does not extend current origin sequence: {origin}",
                )
        return new_events, replayed

    def _validate_import_bundles_locked(
        self, bundles: Sequence[OriginBundle]
    ) -> list[tuple[OriginBundle, list[OperationalEventV2], int]]:
        seen: set[str] = set()
        validated: list[tuple[OriginBundle, list[OperationalEventV2], int]] = []
        for bundle in bundles:
            if not isinstance(bundle, OriginBundle):
                raise _failure(
                    OperationalErrorCode.INVALID_EVENT,
                    "origin bundle is required",
                )
            self._validate_bundle(bundle)
            origin = bundle.anchor.origin_device
            self._validate_safe_id(origin, "origin device")
            if origin in seen:
                raise _failure(
                    OperationalErrorCode.ANCHOR_CONFLICT,
                    f"duplicate origin bundle: {origin}",
                )
            seen.add(origin)
            new_events, replayed = self._validate_bundle_against_existing(bundle)
            validated.append((bundle, new_events, replayed))
        return validated

    def validate_import_bundles(self, bundles: Iterable[OriginBundle]) -> tuple[OriginBundle, ...]:
        materialized = tuple(bundles)
        with authority_write_lock(self.root):
            self._validate_import_bundles_locked(materialized)
        return materialized

    @staticmethod
    def _bundle_wire(bundle: OriginBundle) -> dict[str, object]:
        return {
            "anchor": asdict(bundle.anchor),
            "checkpoint": base64.urlsafe_b64encode(bundle.checkpoint).rstrip(b"=").decode("ascii"),
            "events": [asdict(event) for event in bundle.events],
            "head_sequence": bundle.head_sequence,
            "head_hash": bundle.head_hash,
        }

    def quarantine(self, bundle: OriginBundle, *, reason: str) -> Path:
        encoded_bundle = canonical_json_bytes(self._bundle_wire(bundle))
        digest = hashlib.sha256(encoded_bundle).hexdigest()
        self._ensure_directory(self.quarantine_dir)
        existing = sorted(self.quarantine_dir.glob(f"*-{digest}.json"))
        for path in existing:
            self._assert_safe_path(path)
            if path.is_symlink():
                raise _failure(
                    OperationalErrorCode.STORAGE_UNAVAILABLE,
                    f"unsafe quarantine symlink: {path}",
                )
            return path
        timestamp = _parse_timestamp(self._now(), "clock").strftime("%Y%m%dT%H%M%S%fZ")
        path = self.quarantine_dir / f"{timestamp}-{digest}.json"
        self._atomic_write_json(
            path,
            {
                "schema": _QUARANTINE_SCHEMA,
                "sha256": digest,
                "reason": reason,
                "bundle": self._bundle_wire(bundle),
            },
        )
        return path

    def _snapshot_paths(
        self,
        validated: Sequence[tuple[OriginBundle, list[OperationalEventV2], int]],
    ) -> dict[Path, bytes | None]:
        paths: set[Path] = set()
        for bundle, new_events, _ in validated:
            paths.update(
                {
                    self._anchor_path(bundle.anchor.origin_device),
                    self._checkpoint_path(bundle.anchor),
                    self._head_path(bundle.anchor.origin_device),
                }
            )
            paths.update(self._segment_path(event) for event in new_events)
        snapshots: dict[Path, bytes | None] = {}
        for path in paths:
            self._assert_safe_path(path)
            try:
                snapshots[path] = path.read_bytes()
            except FileNotFoundError:
                snapshots[path] = None
            except OSError as exc:
                raise _failure(
                    OperationalErrorCode.STORAGE_UNAVAILABLE,
                    f"cannot snapshot import target: {path}",
                ) from exc
        return snapshots

    def _restore_snapshots(self, snapshots: Mapping[Path, bytes | None]) -> None:
        failures: list[str] = []
        cleanup_candidates: set[Path] = set()
        for path, encoded in snapshots.items():
            try:
                if encoded is None:
                    path.unlink(missing_ok=True)
                    cleanup_candidates.add(path.parent)
                    if path.parent.exists():
                        _fsync_directory(path.parent)
                else:
                    self._atomic_write_bytes(path, encoded)
            except (OSError, OperationalError) as exc:
                failures.append(f"{path}: {exc}")
        for directory in sorted(
            cleanup_candidates,
            key=lambda candidate: len(candidate.parts),
            reverse=True,
        ):
            current = directory
            while current != self.root and current.is_relative_to(self.root):
                try:
                    current.rmdir()
                    _fsync_directory(current.parent)
                except FileNotFoundError:
                    pass
                except OSError:
                    break
                current = current.parent
        if failures:
            raise _failure(
                OperationalErrorCode.STORAGE_UNAVAILABLE,
                "operational import rollback failed",
                details={"errors": failures},
            )

    def import_bundles(self, bundles: Iterable[OriginBundle]) -> LedgerImportReport:
        materialized = tuple(bundles)
        quarantined: list[str] = []
        try:
            with authority_write_lock(self.root):
                validated = self._validate_import_bundles_locked(materialized)
                snapshots = self._snapshot_paths(validated)
                inserted = 0
                replayed = 0
                try:
                    for bundle, new_events, replay_count in validated:
                        origin = bundle.anchor.origin_device
                        anchor_path = self._anchor_path(origin)
                        if not anchor_path.exists():
                            self._persist_anchor(bundle.anchor, bundle.checkpoint)
                        else:
                            current = self._read_anchor_with_checkpoint(origin)[0]
                            if current.anchor_hash != bundle.anchor.anchor_hash:
                                self._persist_anchor(
                                    bundle.anchor,
                                    bundle.checkpoint,
                                )
                        for event in new_events:
                            self._append_event_fsync(event)
                        if new_events:
                            self._write_head(
                                origin=origin,
                                sequence=bundle.head_sequence,
                                event_hash=bundle.head_hash,
                                anchor_hash=bundle.anchor.anchor_hash,
                            )
                        inserted += len(new_events)
                        replayed += replay_count
                except BaseException:
                    self._restore_snapshots(snapshots)
                    raise
                final_positions: list[OriginPosition] = []
                for bundle, _, _ in validated:
                    anchor = self._read_anchor_with_checkpoint(bundle.anchor.origin_device)[0]
                    final_positions.append(
                        self._load_position(
                            bundle.anchor.origin_device,
                            anchor,
                            repair=False,
                        )
                    )
        except OperationalError as exc:
            for bundle in materialized:
                try:
                    quarantine_path = self.quarantine(bundle, reason=str(exc))
                except (OperationalError, TypeError, ValueError):
                    quarantine_path = None
                if quarantine_path is not None:
                    quarantined.append(str(quarantine_path))
            raise
        manifest_sha256 = hashlib.sha256(
            canonical_json_bytes(
                sorted(
                    (self._bundle_wire(bundle) for bundle in materialized),
                    key=lambda item: cast(dict[str, Any], item["anchor"])["origin_device"],
                )
            )
        ).hexdigest()
        return LedgerImportReport(
            manifest_sha256=manifest_sha256,
            origins_seen=tuple(sorted(bundle.anchor.origin_device for bundle in materialized)),
            events_inserted=inserted,
            events_replayed=replayed,
            quarantined=tuple(quarantined),
            final_positions=tuple(
                sorted(final_positions, key=lambda position: position.origin_device)
            ),
        )

    def export_bundles(self, *, origins: Iterable[str] | None = None) -> tuple[OriginBundle, ...]:
        requested = (
            tuple(self._validate_safe_id(origin, "origin device") for origin in origins)
            if origins is not None
            else None
        )
        with authority_write_lock(self.root):
            selected = requested if requested is not None else self._discover_origins()
            bundles: list[OriginBundle] = []
            for origin in selected:
                anchor, checkpoint = self._read_anchor_with_checkpoint(origin)
                events = tuple(self._read_events_chain(origin, anchor))
                position = self._load_position(origin, anchor, repair=False)
                bundles.append(
                    OriginBundle(
                        anchor=anchor,
                        checkpoint=checkpoint,
                        events=events,
                        head_sequence=position.sequence,
                        head_hash=position.event_hash,
                    )
                )
            return tuple(bundles)


__all__ = ["OperationLedgerV2"]
