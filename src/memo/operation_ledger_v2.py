"""Anchored, signed operational ledger v2.

The JSONL segments are authoritative.  Per-origin heads are durable advisory
caches and may only be repaired from a completely verified segment chain.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import os
import re
import stat
import uuid
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, fields, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from memo.atomic_io import (
    SecureDirectory,
    authority_admission_lock,
    authority_write_lock,
    open_secure_directory,
)
from memo.contracts import MemoEvent
from memo.errors import OperationalError, OperationalErrorCode
from memo.identity import PrincipalIdentity
from memo.operation_ledger_v1 import LegacyOperationLedger
from memo.operational_epoch import CommitContext, EpochFence
from memo.operational_event import (
    EMPTY_REDUCER_STATE_BYTES,
    ChainAnchor,
    LedgerImportReport,
    LedgerRecoveryReport,
    MigrationOrigin,
    OperationalCommand,
    OperationalEventV2,
    OriginBundle,
    OriginPosition,
    SourceProof,
    SourceProofAuthentication,
    StateCheckpoint,
    VerificationReport,
    canonical_anchor_hash,
    canonical_event_hash,
    canonical_json_bytes,
    canonical_signed_bytes,
    operational_wire_dict,
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
_EVENT_LEGACY_FIELDS = _EVENT_FIELDS - {"migration_origin", "migration_origin_sha256"}
_IDENTITY_FIELDS = frozenset(field.name for field in fields(PrincipalIdentity))
_SOURCE_PROOF_FIELDS = frozenset(field.name for field in fields(SourceProof))
_SOURCE_PROOF_LEGACY_FIELDS = _SOURCE_PROOF_FIELDS - {"authentication"}
_SOURCE_AUTH_FIELDS = frozenset(field.name for field in fields(SourceProofAuthentication))
_MIGRATION_ORIGIN_FIELDS = frozenset(field.name for field in fields(MigrationOrigin))
_CHECKPOINT_FIELDS = frozenset(field.name for field in fields(StateCheckpoint))
_HEAD_FIELDS = frozenset({"schema", "origin_device", "sequence", "event_hash", "anchor_hash"})
_HEAD_SCHEMA = "memo.operational_head.v1"
_QUARANTINE_SCHEMA = "memo.operational_quarantine.v1"
_RECOVERY_SCHEMA = "memo.operational_recovery.v1"
_TRANSACTION_SCHEMA = "memo.operational_transaction.v1"
_TRANSACTION_MARKER_SCHEMA = "memo.operational_transaction_marker.v1"
_TRANSACTION_RECEIPT_SCHEMA = "memo.operational_transaction_receipt.v1"
_MAX_TRANSACTION_RECEIPTS = 256
_TRANSACTION_FIELDS = frozenset(
    {
        "schema",
        "transaction_id",
        "transaction_sha256",
        "request_sha256",
        "kind",
        "origins",
        "before_positions",
        "after_positions",
        "targets",
        "prepared_at",
    }
)
_TRANSACTION_TARGET_FIELDS = frozenset(
    {
        "relative_target",
        "mode",
        "before_sha256",
        "after_sha256",
        "size",
        "stage_blob",
    }
)
_TRANSACTION_MARKER_FIELDS = frozenset(
    {"schema", "transaction_id", "manifest_sha256", "phase", "recorded_at"}
)
_TRANSACTION_RECEIPT_FIELDS = frozenset(
    {
        "schema",
        "transaction_id",
        "transaction_sha256",
        "request_sha256",
        "kind",
        "origins",
        "after_positions",
        "applied_marker_sha256",
        "finalized_at",
    }
)


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


def _decode_base64url(value: object, field: str) -> bytes:
    if not isinstance(value, str):
        raise _failure(
            OperationalErrorCode.ANCHOR_CONFLICT,
            f"{field} must be canonical base64url",
        )
    try:
        decoded = base64.b64decode(
            value + ("=" * (-len(value) % 4)),
            altchars=b"-_",
            validate=True,
        )
    except (ValueError, binascii.Error) as exc:
        raise _failure(
            OperationalErrorCode.ANCHOR_CONFLICT,
            f"{field} must be canonical base64url",
        ) from exc
    canonical = base64.urlsafe_b64encode(decoded).rstrip(b"=").decode("ascii")
    if canonical != value:
        raise _failure(
            OperationalErrorCode.ANCHOR_CONFLICT,
            f"{field} must be canonical base64url",
        )
    return decoded


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
        explicit_roster_root = Path(roster_root).absolute() if roster_root is not None else None
        epoch_root = Path(epoch_fence.root).absolute() if epoch_fence is not None else None
        if explicit_roster_root is not None and epoch_root is not None:
            try:
                same_authority = os.path.samefile(explicit_roster_root, epoch_root)
            except FileNotFoundError:
                same_authority = explicit_roster_root == epoch_root
            if not same_authority:
                raise ValueError("roster and epoch fence must share one authority root")
        self.roster_root = explicit_roster_root if explicit_roster_root is not None else epoch_root
        self.pin_store = pin_store
        self.epoch_fence = epoch_fence
        self.reducer_version = reducer_version
        self._secure_directory: ContextVar[SecureDirectory | None] = ContextVar(
            f"memo_ledger_secure_directory_{id(self)}",
            default=None,
        )
        self._transaction_staging = False
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

    @property
    def transactions_dir(self) -> Path:
        return self.root / "transactions"

    @property
    def recovery_dir(self) -> Path:
        return self.root / "recovery"

    @property
    def transaction_receipts_dir(self) -> Path:
        return self.recovery_dir / "transactions"

    @property
    def anchor_history_dir(self) -> Path:
        return self.root / "anchor-history"

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
            target.relative_to(root)
        except ValueError as exc:
            raise _failure(
                OperationalErrorCode.STORAGE_UNAVAILABLE,
                f"operational path escapes journal root: {target}",
            ) from exc

    def _relative(self, path: Path) -> Path:
        self._assert_safe_path(path)
        relative = Path(path).absolute().relative_to(self.root.absolute())
        if not relative.parts:
            raise _failure(
                OperationalErrorCode.STORAGE_UNAVAILABLE,
                "journal root is not a file target",
            )
        return relative

    @contextmanager
    def _secure_io(self, *, create: bool) -> Iterator[SecureDirectory]:
        retained = self._secure_directory.get()
        if retained is not None:
            yield retained
            return
        try:
            directory = open_secure_directory(self.root, create=create)
        except (OperationalError, FileNotFoundError):
            raise
        except (OSError, ValueError) as exc:
            raise _failure(
                OperationalErrorCode.STORAGE_UNAVAILABLE,
                f"unsafe operational journal path or storage failure: {self.root}",
            ) from exc
        with directory:
            token = self._secure_directory.set(directory)
            try:
                yield directory
            finally:
                self._secure_directory.reset(token)

    def _exists(self, path: Path) -> bool:
        try:
            with self._secure_io(create=False) as directory:
                return directory.exists(self._relative(path))
        except FileNotFoundError:
            return False
        except OperationalError:
            raise
        except (OSError, ValueError) as exc:
            raise _failure(
                OperationalErrorCode.STORAGE_UNAVAILABLE,
                f"cannot inspect operational path: {path}",
            ) from exc

    def _list_names(self, path: Path) -> tuple[str, ...]:
        try:
            with self._secure_io(create=False) as directory:
                return directory.list_names(self._relative(path))
        except FileNotFoundError:
            return ()
        except OperationalError:
            raise
        except (OSError, ValueError) as exc:
            raise _failure(
                OperationalErrorCode.STORAGE_UNAVAILABLE,
                f"cannot list operational directory: {path}",
            ) from exc

    def _stat(self, path: Path) -> os.stat_result:
        try:
            with self._secure_io(create=False) as directory:
                return directory.stat(self._relative(path))
        except OperationalError:
            raise
        except (OSError, ValueError) as exc:
            raise _failure(
                OperationalErrorCode.STORAGE_UNAVAILABLE,
                f"cannot inspect operational path: {path}",
            ) from exc

    @contextmanager
    def _locked_journal(self, *, create: bool = True) -> Iterator[SecureDirectory]:
        journal_lock = authority_write_lock(self.root)
        try:
            journal_lock.__enter__()
        except (OSError, ValueError) as exc:
            raise _failure(
                OperationalErrorCode.STORAGE_UNAVAILABLE,
                f"unsafe operational journal path or storage failure: {self.root}",
            ) from exc
        try:
            with self._secure_io(create=create) as directory:
                yield directory
        finally:
            journal_lock.__exit__(None, None, None)

    @contextmanager
    def _authority_operation(
        self,
        *,
        context: CommitContext | None = None,
        recover: bool = True,
    ) -> Iterator[None]:
        if self.roster_root is None:
            raise _failure(
                OperationalErrorCode.SIGNATURE_INVALID,
                "pinned operational roster authority is unavailable",
            )
        if context is not None and self.epoch_fence is None:
            raise _failure(
                OperationalErrorCode.INVALID_EVENT,
                "operational epoch authority is unavailable",
            )
        admission = authority_admission_lock(self.roster_root)
        try:
            admission.__enter__()
        except (OSError, ValueError) as exc:
            raise _failure(
                OperationalErrorCode.STORAGE_UNAVAILABLE,
                "cannot retain operational authority and journal snapshot",
            ) from exc
        try:
            self._latest_roster()
            with self._locked_journal():
                if context is None:
                    if recover:
                        self._recover_locked()
                    yield
                else:
                    assert self.epoch_fence is not None
                    with self.epoch_fence.verified(context):
                        if recover:
                            self._recover_locked()
                        yield
        finally:
            admission.__exit__(None, None, None)

    def _ensure_directory(self, path: Path) -> None:
        if Path(path).absolute() == self.root.absolute():
            with self._secure_io(create=True):
                return
        try:
            with self._secure_io(create=True) as directory:
                directory.ensure_directory(self._relative(path))
        except OperationalError:
            raise
        except (OSError, ValueError) as exc:
            raise _failure(
                OperationalErrorCode.STORAGE_UNAVAILABLE,
                f"cannot create operational directory: {path}",
            ) from exc

    def _atomic_write_json(self, path: Path, value: object) -> None:
        self._atomic_write_bytes(path, canonical_json_bytes(value))

    def _atomic_write_bytes(self, path: Path, value: bytes) -> None:
        try:
            with self._secure_io(create=True) as directory:
                directory.atomic_write_bytes(self._relative(path), bytes(value))
        except OperationalError:
            raise
        except (OSError, ValueError) as exc:
            raise _failure(
                OperationalErrorCode.STORAGE_UNAVAILABLE,
                f"cannot atomically write operational bytes: {path}",
            ) from exc

    def _create_bytes_exclusive(self, path: Path, value: bytes) -> None:
        try:
            with self._secure_io(create=True) as directory:
                directory.create_bytes_exclusive(
                    self._relative(path),
                    bytes(value),
                )
        except OperationalError:
            raise
        except (OSError, ValueError) as exc:
            raise _failure(
                OperationalErrorCode.STORAGE_UNAVAILABLE,
                f"cannot create durable operational marker: {path}",
            ) from exc

    def _optional_bytes(self, path: Path) -> bytes | None:
        try:
            return self._read_bytes(path, "operational transaction target")
        except OperationalError as exc:
            if exc.code == OperationalErrorCode.NOT_FOUND:
                return None
            raise

    def _latest_roster(self) -> VerificationRoster:
        if self.verifier is None:
            raise _failure(
                OperationalErrorCode.SIGNATURE_INVALID,
                "operational verification authority is unavailable",
            )
        if self.roster_root is None:
            raise _failure(
                OperationalErrorCode.SIGNATURE_INVALID,
                "pinned operational roster authority is unavailable",
            )
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

    def _roster_for(self, version: int) -> VerificationRoster:
        if self.verifier is None:
            raise _failure(
                OperationalErrorCode.SIGNATURE_INVALID,
                "operational verification authority is unavailable",
            )
        if self.roster_root is None:
            raise _failure(
                OperationalErrorCode.SIGNATURE_INVALID,
                "pinned operational roster authority is unavailable",
            )
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

    def _anchor_history_path(self, origin: str, anchor_hash: str) -> Path:
        self._validate_safe_id(origin, "origin device")
        if not isinstance(anchor_hash, str) or not _SHA256_RE.fullmatch(anchor_hash):
            raise _failure(
                OperationalErrorCode.ANCHOR_CONFLICT,
                f"invalid predecessor anchor history digest: {anchor_hash!r}",
            )
        path = self.anchor_history_dir / origin / f"{anchor_hash}.json"
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
        if not isinstance(value, dict) or set(value) not in {
            _SOURCE_PROOF_LEGACY_FIELDS,
            _SOURCE_PROOF_FIELDS,
        }:
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
            body = dict(value)
            authentication = body.get("authentication")
            if authentication is not None:
                if (
                    not isinstance(authentication, dict)
                    or set(authentication) != _SOURCE_AUTH_FIELDS
                ):
                    raise TypeError
                path = authentication.get("merkle_path")
                if not isinstance(path, list) or not all(isinstance(item, str) for item in path):
                    raise TypeError
                body["authentication"] = SourceProofAuthentication(
                    **{
                        **authentication,
                        "merkle_path": tuple(path),
                    }
                )
            return SourceProof(**body)
        except TypeError as exc:
            raise _failure(
                OperationalErrorCode.INVALID_EVENT,
                "operational source proof is invalid",
            ) from exc

    def _decode_migration_origin(self, value: object) -> MigrationOrigin | None:
        if value is None:
            return None
        if not isinstance(value, dict) or set(value) != _MIGRATION_ORIGIN_FIELDS:
            raise _failure(
                OperationalErrorCode.INVALID_EVENT,
                "operational migration origin fields are invalid",
            )
        try:
            return MigrationOrigin(**value)
        except TypeError as exc:
            raise _failure(
                OperationalErrorCode.INVALID_EVENT,
                "operational migration origin is invalid",
            ) from exc

    def _decode_event_bytes(self, encoded: bytes, description: str) -> OperationalEventV2:
        try:
            value = json.loads(encoded.decode("utf-8"))
            if not isinstance(value, dict) or set(value) not in {
                _EVENT_LEGACY_FIELDS,
                _EVENT_FIELDS,
            }:
                raise ValueError
            actor = self._decode_identity(value["actor"])
            source_proof = self._decode_source_proof(value["source_proof"])
            migration_origin = self._decode_migration_origin(value.get("migration_origin"))
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
            body["migration_origin"] = migration_origin
            body.setdefault("migration_origin_sha256", "")
            body["caused_by"] = tuple(caused_by)
            body["payload"] = payload
            event = OperationalEventV2(**body)
            if canonical_json_bytes(event) != encoded:
                raise ValueError
            return event
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
        try:
            with self._secure_io(create=False) as directory:
                return directory.read_bytes(self._relative(path))
        except FileNotFoundError:
            raise _failure(
                OperationalErrorCode.NOT_FOUND,
                f"{description} is missing: {path}",
            ) from None
        except OperationalError:
            raise
        except (OSError, ValueError) as exc:
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
        if anchor.reducer_version != self.reducer_version:
            raise _failure(
                OperationalErrorCode.ANCHOR_CONFLICT,
                "anchor reducer version is unsupported",
            )
        if anchor.kind in {"empty", "memo_v1"}:
            if checkpoint != EMPTY_REDUCER_STATE_BYTES:
                raise _failure(
                    OperationalErrorCode.ANCHOR_CONFLICT,
                    f"{anchor.kind} anchor requires the canonical empty checkpoint",
                )
            if anchor.state_sha256 != hashlib.sha256(checkpoint).hexdigest():
                raise _failure(
                    OperationalErrorCode.ANCHOR_CONFLICT,
                    "anchor checkpoint state digest mismatch",
                )
            return
        if anchor.kind != "compaction":
            raise _failure(
                OperationalErrorCode.ANCHOR_CONFLICT,
                "unsupported operational checkpoint kind",
            )
        try:
            parsed = json.loads(checkpoint.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise _failure(
                OperationalErrorCode.ANCHOR_CONFLICT,
                "compaction checkpoint must be canonical StateCheckpoint bytes",
            ) from exc
        if (
            not isinstance(parsed, dict)
            or set(parsed) != _CHECKPOINT_FIELDS
            or canonical_json_bytes(parsed) != checkpoint
        ):
            raise _failure(
                OperationalErrorCode.ANCHOR_CONFLICT,
                "compaction checkpoint must be a canonical StateCheckpoint",
            )
        state_bytes = _decode_base64url(parsed["state_bytes"], "checkpoint state_bytes")
        try:
            checkpoint_value = StateCheckpoint(
                **{
                    **parsed,
                    "state_bytes": state_bytes,
                }
            )
        except TypeError as exc:
            raise _failure(
                OperationalErrorCode.ANCHOR_CONFLICT,
                "compaction checkpoint fields are invalid",
            ) from exc
        if canonical_json_bytes(checkpoint_value) != checkpoint:
            raise _failure(
                OperationalErrorCode.ANCHOR_CONFLICT,
                "compaction checkpoint encoding is not canonical",
            )
        _parse_timestamp(checkpoint_value.created_at, "checkpoint created_at")
        if (
            checkpoint_value.schema != "memo.operational_checkpoint.v1"
            or checkpoint_value.checkpoint_id != anchor.checkpoint_id
            or checkpoint_value.origin_device != anchor.origin_device
            or checkpoint_value.reducer_version != anchor.reducer_version
            or checkpoint_value.through_sequence != anchor.base_sequence
            or checkpoint_value.through_event_hash != anchor.base_event_hash
        ):
            raise _failure(
                OperationalErrorCode.ANCHOR_CONFLICT,
                "compaction checkpoint identity, reducer, or through-position mismatch",
            )
        state_digest = hashlib.sha256(state_bytes).hexdigest()
        if checkpoint_value.state_sha256 != state_digest or anchor.state_sha256 != state_digest:
            raise _failure(
                OperationalErrorCode.ANCHOR_CONFLICT,
                "compaction checkpoint inner state digest mismatch",
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
        authentication = proof.authentication
        if authentication is not None:
            if not isinstance(authentication, SourceProofAuthentication):
                raise _failure(
                    OperationalErrorCode.INVALID_EVENT,
                    "source proof authentication is invalid",
                )
            _require_string(
                authentication.schema,
                "source proof authentication schema",
            )
            _require_string(
                authentication.source_manifest_sha256,
                "source proof authentication manifest",
            )
            _require_int(
                authentication.leaf_index,
                "source proof authentication leaf_index",
            )
            _require_int(
                authentication.leaf_count,
                "source proof authentication leaf_count",
                minimum=1,
            )
            if not isinstance(authentication.merkle_path, tuple) or not all(
                isinstance(item, str) for item in authentication.merkle_path
            ):
                raise _failure(
                    OperationalErrorCode.INVALID_EVENT,
                    "source proof authentication merkle_path is invalid",
                )

    @staticmethod
    def _validate_migration_origin_shape(origin: MigrationOrigin) -> None:
        if not isinstance(origin, MigrationOrigin):
            raise _failure(
                OperationalErrorCode.INVALID_EVENT,
                "operational migration origin is required",
            )
        for field in (
            "schema",
            "attempt_id",
            "migration_device_id",
            "source_manifest_sha256",
            "capability_manifest_sha256",
            "attestor_device_id",
            "attestor_key_id",
            "issued_at",
            "expires_at",
            "signature",
            "source_proof_root_sha256",
        ):
            _require_string(
                getattr(origin, field),
                f"migration origin {field}",
                allow_empty=True,
            )
        _require_int(
            origin.roster_version,
            "migration origin roster_version",
            minimum=1,
        )
        _require_int(
            origin.source_proof_count,
            "migration origin source_proof_count",
            minimum=1,
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
        if event.migration_origin is not None:
            cls._validate_migration_origin_shape(event.migration_origin)
        _require_string(
            event.migration_origin_sha256,
            "event migration_origin_sha256",
            allow_empty=True,
        )

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

    def _read_anchor_history(self, origin: str, anchor_hash: str) -> ChainAnchor:
        path = self._anchor_history_path(origin, anchor_hash)
        try:
            encoded = self._read_bytes(path, "operational predecessor anchor history")
        except OperationalError as exc:
            if exc.code == OperationalErrorCode.NOT_FOUND:
                raise _failure(
                    OperationalErrorCode.ANCHOR_CONFLICT,
                    f"predecessor anchor history is unavailable: {origin}/{anchor_hash}",
                ) from exc
            raise
        anchor = self._decode_anchor_bytes(encoded, str(path))
        if anchor.origin_device != origin or anchor.anchor_hash != anchor_hash:
            raise _failure(
                OperationalErrorCode.ANCHOR_CONFLICT,
                f"predecessor anchor history identity mismatch: {origin}/{anchor_hash}",
            )
        checkpoint = self._read_bytes(
            self._checkpoint_path(anchor),
            "historical operational checkpoint",
        )
        self._validate_anchor_authority(anchor, checkpoint)
        return anchor

    def anchor(self, origin_device: str | None = None) -> ChainAnchor:
        origin = origin_device or self.device_id
        with self._authority_operation(recover=False):
            # Complete committed multi-origin publications, but leave torn-tail
            # repair to recover()/position()/event observations.
            self._recover_transactions_locked()
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

    def _persist_anchor_history(self, anchor: ChainAnchor) -> None:
        history_path = self._anchor_history_path(
            anchor.origin_device,
            anchor.anchor_hash,
        )
        history_bytes = canonical_json_bytes(anchor)
        existing_history = self._optional_bytes(history_path)
        if existing_history is None:
            self._create_bytes_exclusive(history_path, history_bytes)
        elif existing_history != history_bytes:
            raise _failure(
                OperationalErrorCode.ANCHOR_CONFLICT,
                f"immutable anchor history changed: {anchor.anchor_hash}",
            )

    def _persist_anchor(self, anchor: ChainAnchor, checkpoint: bytes) -> None:
        self._validate_anchor_authority(anchor, checkpoint)
        self._atomic_write_bytes(self._checkpoint_path(anchor), checkpoint)
        self._persist_anchor_history(anchor)
        self._atomic_write_json(self._anchor_path(anchor.origin_device), asdict(anchor))
        self._write_head(
            origin=anchor.origin_device,
            sequence=anchor.base_sequence,
            event_hash=anchor.base_event_hash,
            anchor_hash=anchor.anchor_hash,
        )

    def _assert_current_anchor_authority(
        self,
        anchor: ChainAnchor,
        roster: VerificationRoster,
    ) -> None:
        if anchor.roster_version != roster.version:
            raise _failure(
                OperationalErrorCode.SIGNATURE_INVALID,
                "new anchor must use the latest pinned roster",
            )
        key = roster.key(anchor.key_id)
        role = anchor.signer_role
        expected = self._active_key(
            roster,
            device_id=(anchor.origin_device if role == "origin" else key.device_id),
            role=role,
            activation_sequence=roster.version,
            exclusive_role=role == "migration_attestor",
        )
        if expected.key_id != anchor.key_id:
            raise _failure(
                OperationalErrorCode.SIGNATURE_INVALID,
                "new anchor signer is not the current active authority",
            )

    @staticmethod
    def _validate_anchor_transition(
        current: ChainAnchor,
        successor: ChainAnchor,
        position: OriginPosition,
    ) -> None:
        if (
            successor.kind != "compaction"
            or successor.origin_device != current.origin_device
            or successor.previous_anchor_hash != current.anchor_hash
            or successor.ledger_epoch < current.ledger_epoch
            or successor.reducer_version != current.reducer_version
            or successor.base_sequence != position.sequence
            or successor.base_event_hash != position.event_hash
        ):
            raise _failure(
                OperationalErrorCode.ANCHOR_CONFLICT,
                f"anchor epoch regression or transition conflict for {current.origin_device}",
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
        if anchor is None and self._exists(path):
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
        latest = self._latest_roster()
        current: ChainAnchor | None = None
        position: OriginPosition | None = None
        if self._exists(path):
            current, current_checkpoint = self._read_anchor_with_checkpoint(origin)
            if current.anchor_hash == anchor.anchor_hash:
                if current_checkpoint != checkpoint:
                    raise _failure(
                        OperationalErrorCode.ANCHOR_CONFLICT,
                        "identical anchor has different checkpoint bytes",
                    )
                return current
            position = self._load_position(origin, current, repair=False)
            self._validate_anchor_transition(current, anchor, position)
        self._assert_current_anchor_authority(anchor, latest)
        if current is not None and position is not None and anchor.kind == "compaction":
            current_history_path = self._anchor_history_path(origin, current.anchor_hash)
            current_history_bytes = canonical_json_bytes(current)
            existing_history = self._optional_bytes(current_history_path)
            if existing_history is None:
                self._persist_anchor_history(current)
            elif existing_history != current_history_bytes:
                raise _failure(
                    OperationalErrorCode.ANCHOR_CONFLICT,
                    f"immutable predecessor anchor history changed: {current.anchor_hash}",
                )
            after = OriginPosition(
                origin_device=origin,
                sequence=anchor.base_sequence,
                event_hash=anchor.base_event_hash,
                anchor_hash=anchor.anchor_hash,
            )
            request_sha256 = hashlib.sha256(
                canonical_json_bytes(
                    {
                        "anchor": operational_wire_dict(anchor),
                        "checkpoint": base64.urlsafe_b64encode(checkpoint)
                        .rstrip(b"=")
                        .decode("ascii"),
                    }
                )
            ).hexdigest()
            self._run_transaction(
                kind="compaction",
                request_sha256=request_sha256,
                origins=(origin,),
                before_positions=(self._position_wire(position, origin=origin),),
                after_positions=(self._position_wire(after, origin=origin),),
                target_bytes=(
                    (self._checkpoint_path(anchor), checkpoint),
                    (current_history_path, current_history_bytes),
                    (
                        self._anchor_history_path(origin, anchor.anchor_hash),
                        canonical_json_bytes(anchor),
                    ),
                    (path, canonical_json_bytes(anchor)),
                    (
                        self._head_path(origin),
                        canonical_json_bytes(
                            {
                                "schema": _HEAD_SCHEMA,
                                "origin_device": origin,
                                "sequence": anchor.base_sequence,
                                "event_hash": anchor.base_event_hash,
                                "anchor_hash": anchor.anchor_hash,
                            }
                        ),
                    ),
                ),
            )
            return anchor
        self._persist_anchor(anchor, checkpoint)
        return anchor

    def ensure_anchor(
        self,
        anchor: ChainAnchor | None = None,
        *,
        checkpoint: bytes | None = None,
    ) -> ChainAnchor:
        with self._authority_operation():
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
        expected = self._active_key(
            roster,
            device_id=key.device_id,
            role="migration_attestor",
            activation_sequence=roster.version,
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
        try:
            encoded = self._read_bytes(path, "operational head")
        except OperationalError as exc:
            if exc.code == OperationalErrorCode.NOT_FOUND:
                return None
            raise
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
        paths: list[Path] = []
        for name in self._list_names(event_dir):
            path = event_dir / name
            observed = self._stat(path)
            if not stat.S_ISREG(observed.st_mode) or path.suffix != ".jsonl":
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

    def _validate_source_proof(
        self,
        event: OperationalEventV2,
        anchor: ChainAnchor,
    ) -> None:
        proof = event.source_proof
        if proof is None:
            return
        self._validate_source_proof_shape(proof)
        migration = event.migration_origin
        if migration is None:
            raise _failure(
                OperationalErrorCode.INVALID_EVENT,
                "source proof requires authenticated migration authority",
            )
        self._validate_migration_origin_shape(migration)
        verifier, roster = self._verification_authority(migration.roster_version)
        validate_migration_origin(
            migration,
            roster=roster,
            verifier=verifier,
            at_time=event.created_at,
        )
        if anchor.kind == "empty":
            return
        if anchor.kind != "memo_v1" or (
            migration.source_manifest_sha256 != anchor.source_manifest_sha256
        ):
            raise _failure(
                OperationalErrorCode.ANCHOR_CONFLICT,
                "migration origin is not bound to the sealed v1 manifest",
            )
        source_anchor = anchor
        if proof.source_origin != anchor.origin_device:
            if not self._exists(self._anchor_path(proof.source_origin)):
                raise _failure(
                    OperationalErrorCode.INVALID_EVENT,
                    "source proof origin has no sealed memo_v1 anchor",
                )
            source_anchor = self._read_anchor_with_checkpoint(proof.source_origin)[0]
            if (
                source_anchor.kind != "memo_v1"
                or source_anchor.source_manifest_sha256
                != anchor.source_manifest_sha256
            ):
                raise _failure(
                    OperationalErrorCode.ANCHOR_CONFLICT,
                    "source proof origin is not bound to the same sealed v1 manifest",
                )
        if (
            proof.source_system != "memo_v1"
            or proof.source_sequence < 1
            or proof.source_sequence > source_anchor.base_sequence
            or not proof.source_event_id
            or not proof.source_schema
            or not _SHA256_RE.fullmatch(proof.source_event_hash)
        ):
            raise _failure(
                OperationalErrorCode.INVALID_EVENT,
                "source proof is not linked to the sealed memo_v1 source",
            )
        if (
            proof.source_sequence == source_anchor.base_sequence
            and proof.source_event_hash != source_anchor.base_event_hash
        ):
            raise _failure(
                OperationalErrorCode.INVALID_EVENT,
                "source proof does not match the sealed v1 head",
            )

    def _scan_events_chain(
        self,
        origin: str,
        anchor: ChainAnchor,
        *,
        allow_torn_tail: bool,
    ) -> tuple[list[OperationalEventV2], tuple[Path, int] | None]:
        rows: list[OperationalEventV2] = []
        expected_sequence = anchor.base_sequence + 1
        expected_previous = anchor.base_event_hash
        previous_time: datetime | None = None
        previous_seen_sequence = 0
        paths = self._event_segment_paths(origin)
        torn: tuple[Path, int] | None = None
        for path_index, path in enumerate(paths):
            segment_bytes = self._read_bytes(path, "operational segment")
            if not segment_bytes or not segment_bytes.endswith(b"\n"):
                if not allow_torn_tail or path_index != len(paths) - 1:
                    raise _failure(
                        OperationalErrorCode.INVALID_EVENT,
                        f"operational segment has an incomplete final row: {path}",
                    )
                prefix_size = segment_bytes.rfind(b"\n") + 1
                torn = (path, prefix_size)
                segment_bytes = segment_bytes[:prefix_size]
            lines = segment_bytes.split(b"\n")[:-1] if segment_bytes else []
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
                self._validate_source_proof(event, anchor)
                rows.append(event)
                expected_sequence += 1
                expected_previous = event.event_hash
                previous_time = event_time
        return rows, torn

    def _read_events_chain(self, origin: str, anchor: ChainAnchor) -> list[OperationalEventV2]:
        return self._scan_events_chain(
            origin,
            anchor,
            allow_torn_tail=False,
        )[0]

    def _write_recovery_record(
        self,
        *,
        kind: str,
        origins: Sequence[str],
        details: Mapping[str, object],
    ) -> None:
        recovery_id = uuid.uuid4().hex
        self._atomic_write_json(
            self.recovery_dir / f"{recovery_id}.json",
            {
                "schema": _RECOVERY_SCHEMA,
                "recovery_id": recovery_id,
                "kind": kind,
                "origins": tuple(origins),
                "recorded_at": self._now(),
                "details": details,
            },
        )

    def _transaction_failpoint(self, _label: str) -> None:
        """Process-loss hook used only by adversarial transaction tests."""

    def _decode_transaction_manifest(
        self,
        encoded: bytes,
        *,
        transaction_id: str,
    ) -> dict[str, Any]:
        try:
            value = json.loads(encoded.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise _failure(
                OperationalErrorCode.STORAGE_UNAVAILABLE,
                f"transaction manifest is invalid: {transaction_id}",
            ) from exc
        if (
            not isinstance(value, dict)
            or set(value) != _TRANSACTION_FIELDS
            or canonical_json_bytes(value) != encoded
            or value.get("schema") != _TRANSACTION_SCHEMA
            or value.get("transaction_id") != transaction_id
        ):
            raise _failure(
                OperationalErrorCode.STORAGE_UNAVAILABLE,
                f"transaction manifest is not canonical: {transaction_id}",
            )
        transaction_sha256 = value.get("transaction_sha256")
        request_sha256 = value.get("request_sha256")
        if (
            not isinstance(transaction_sha256, str)
            or not _SHA256_RE.fullmatch(transaction_sha256)
            or not isinstance(request_sha256, str)
            or not _SHA256_RE.fullmatch(request_sha256)
        ):
            raise _failure(
                OperationalErrorCode.STORAGE_UNAVAILABLE,
                f"transaction hashes are invalid: {transaction_id}",
            )
        unhashed = dict(value)
        unhashed["transaction_sha256"] = ""
        if hashlib.sha256(canonical_json_bytes(unhashed)).hexdigest() != transaction_sha256:
            raise _failure(
                OperationalErrorCode.STORAGE_UNAVAILABLE,
                f"transaction manifest hash mismatch: {transaction_id}",
            )
        origins = value.get("origins")
        targets = value.get("targets")
        if (
            not isinstance(origins, list)
            or not all(
                isinstance(origin, str) and _SAFE_ID_RE.fullmatch(origin) for origin in origins
            )
            or origins != sorted(set(origins))
            or not isinstance(targets, list)
        ):
            raise _failure(
                OperationalErrorCode.STORAGE_UNAVAILABLE,
                f"transaction origins or targets are invalid: {transaction_id}",
            )
        seen_targets: set[str] = set()
        heads_started = False
        for ordinal, target in enumerate(targets):
            if not isinstance(target, dict) or set(target) != _TRANSACTION_TARGET_FIELDS:
                raise _failure(
                    OperationalErrorCode.STORAGE_UNAVAILABLE,
                    f"transaction target is invalid: {transaction_id}/{ordinal}",
                )
            relative = target.get("relative_target")
            stage_blob = target.get("stage_blob")
            before = target.get("before_sha256")
            after = target.get("after_sha256")
            size = target.get("size")
            mode = target.get("mode")
            if (
                not isinstance(relative, str)
                or not relative
                or Path(relative).is_absolute()
                or any(part in {"", ".", ".."} for part in Path(relative).parts)
                or relative in seen_targets
                or stage_blob != f"stage/{ordinal:06d}.bin"
                or mode not in {"create", "replace"}
                or (
                    before is not None
                    and (not isinstance(before, str) or not _SHA256_RE.fullmatch(before))
                )
                or (mode == "create" and before is not None)
                or (mode == "replace" and before is None)
                or not isinstance(after, str)
                or not _SHA256_RE.fullmatch(after)
                or isinstance(size, bool)
                or not isinstance(size, int)
                or size < 0
            ):
                raise _failure(
                    OperationalErrorCode.STORAGE_UNAVAILABLE,
                    f"transaction target metadata is invalid: {transaction_id}/{ordinal}",
                )
            self._relative(self.root / relative)
            seen_targets.add(relative)
            is_head = relative.startswith("heads/")
            if is_head:
                heads_started = True
            elif heads_started:
                raise _failure(
                    OperationalErrorCode.STORAGE_UNAVAILABLE,
                    f"transaction heads are not published last: {transaction_id}",
                )
        return cast(dict[str, Any], value)

    def _validate_transaction_marker(
        self,
        path: Path,
        *,
        transaction_id: str,
        manifest_sha256: str,
        expected_phase: str,
    ) -> dict[str, Any]:
        encoded = self._read_bytes(path, "operational transaction marker")
        try:
            value = json.loads(encoded.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise _failure(
                OperationalErrorCode.STORAGE_UNAVAILABLE,
                f"transaction marker is invalid: {path}",
            ) from exc
        if (
            not isinstance(value, dict)
            or set(value) != _TRANSACTION_MARKER_FIELDS
            or canonical_json_bytes(value) != encoded
            or value["schema"] != _TRANSACTION_MARKER_SCHEMA
            or value["transaction_id"] != transaction_id
            or value["manifest_sha256"] != manifest_sha256
            or value["phase"] != expected_phase
            or not isinstance(value["recorded_at"], str)
        ):
            raise _failure(
                OperationalErrorCode.STORAGE_UNAVAILABLE,
                f"transaction marker phase or manifest does not match: {path}",
            )
        try:
            _parse_timestamp(value["recorded_at"], "transaction marker recorded_at")
        except OperationalError as exc:
            raise _failure(
                OperationalErrorCode.STORAGE_UNAVAILABLE,
                f"transaction marker timestamp is invalid: {path}",
            ) from exc
        return cast(dict[str, Any], value)

    def _apply_transaction_manifest(
        self,
        manifest: Mapping[str, Any],
        *,
        transaction_root: Path,
        invoke_failpoints: bool,
    ) -> int:
        published = 0
        targets = cast(list[dict[str, Any]], manifest["targets"])
        for ordinal, target in enumerate(targets):
            stage_path = transaction_root / cast(str, target["stage_blob"])
            staged = self._read_bytes(stage_path, "operational transaction stage")
            after = cast(str, target["after_sha256"])
            size = cast(int, target["size"])
            if len(staged) != size or hashlib.sha256(staged).hexdigest() != after:
                raise _failure(
                    OperationalErrorCode.STORAGE_UNAVAILABLE,
                    f"transaction stage digest mismatch: {stage_path}",
                )
            target_path = self.root / cast(str, target["relative_target"])
            current = self._optional_bytes(target_path)
            current_sha = hashlib.sha256(current).hexdigest() if current is not None else None
            if current_sha != after:
                if current_sha != target["before_sha256"]:
                    raise _failure(
                        OperationalErrorCode.ANCHOR_CONFLICT,
                        f"transaction target changed outside its manifest: {target_path}",
                    )
                self._atomic_write_bytes(target_path, staged)
                published += 1
                verified = self._read_bytes(
                    target_path,
                    "published operational transaction target",
                )
                if hashlib.sha256(verified).hexdigest() != after:
                    raise _failure(
                        OperationalErrorCode.STORAGE_UNAVAILABLE,
                        f"published transaction target failed verification: {target_path}",
                    )
            if invoke_failpoints:
                self._transaction_failpoint(f"after_target:{ordinal}")
        return published

    def _marker_bytes(
        self,
        *,
        transaction_id: str,
        manifest_sha256: str,
        phase: str,
    ) -> bytes:
        if phase not in {"committed", "applied"}:
            raise ValueError(f"unsupported transaction marker phase: {phase}")
        return canonical_json_bytes(
            {
                "schema": _TRANSACTION_MARKER_SCHEMA,
                "transaction_id": transaction_id,
                "manifest_sha256": manifest_sha256,
                "phase": phase,
                "recorded_at": self._now(),
            }
        )

    def _verify_transaction_targets(
        self,
        manifest: Mapping[str, Any],
        *,
        transaction_root: Path,
    ) -> None:
        targets = cast(list[dict[str, Any]], manifest["targets"])
        for target in targets:
            stage_path = transaction_root / cast(str, target["stage_blob"])
            staged = self._read_bytes(stage_path, "operational transaction stage")
            expected_sha256 = cast(str, target["after_sha256"])
            expected_size = cast(int, target["size"])
            if (
                len(staged) != expected_size
                or hashlib.sha256(staged).hexdigest() != expected_sha256
            ):
                raise _failure(
                    OperationalErrorCode.STORAGE_UNAVAILABLE,
                    f"transaction stage digest verification failed: {stage_path}",
                )
            target_path = self.root / cast(str, target["relative_target"])
            published = self._read_bytes(
                target_path,
                "published operational transaction target",
            )
            if (
                len(published) != expected_size
                or hashlib.sha256(published).hexdigest() != expected_sha256
            ):
                raise _failure(
                    OperationalErrorCode.STORAGE_UNAVAILABLE,
                    f"transaction target digest verification failed: {target_path}",
                )

    def _transaction_receipt_bytes(
        self,
        manifest: Mapping[str, Any],
        *,
        applied_marker_bytes: bytes,
        applied_marker: Mapping[str, Any],
    ) -> bytes:
        return canonical_json_bytes(
            {
                "schema": _TRANSACTION_RECEIPT_SCHEMA,
                "transaction_id": manifest["transaction_id"],
                "transaction_sha256": manifest["transaction_sha256"],
                "request_sha256": manifest["request_sha256"],
                "kind": manifest["kind"],
                "origins": manifest["origins"],
                "after_positions": manifest["after_positions"],
                "applied_marker_sha256": hashlib.sha256(applied_marker_bytes).hexdigest(),
                "finalized_at": applied_marker["recorded_at"],
            }
        )

    def _validate_transaction_receipt(
        self,
        encoded: bytes,
        *,
        transaction_id: str,
        manifest: Mapping[str, Any] | None = None,
        applied_marker_bytes: bytes | None = None,
    ) -> dict[str, Any]:
        try:
            value = json.loads(encoded.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise _failure(
                OperationalErrorCode.STORAGE_UNAVAILABLE,
                f"transaction receipt is invalid: {transaction_id}",
            ) from exc
        if (
            not isinstance(value, dict)
            or set(value) != _TRANSACTION_RECEIPT_FIELDS
            or canonical_json_bytes(value) != encoded
            or value["schema"] != _TRANSACTION_RECEIPT_SCHEMA
            or value["transaction_id"] != transaction_id
            or not isinstance(value["transaction_sha256"], str)
            or not _SHA256_RE.fullmatch(value["transaction_sha256"])
            or not isinstance(value["request_sha256"], str)
            or not _SHA256_RE.fullmatch(value["request_sha256"])
            or not isinstance(value["applied_marker_sha256"], str)
            or not _SHA256_RE.fullmatch(value["applied_marker_sha256"])
            or not isinstance(value["finalized_at"], str)
            or not isinstance(value["kind"], str)
            or not value["kind"]
            or not isinstance(value["origins"], list)
            or not all(
                isinstance(origin, str) and _SAFE_ID_RE.fullmatch(origin)
                for origin in value["origins"]
            )
            or value["origins"] != sorted(set(value["origins"]))
            or not isinstance(value["after_positions"], list)
        ):
            raise _failure(
                OperationalErrorCode.STORAGE_UNAVAILABLE,
                f"transaction receipt is not canonical: {transaction_id}",
            )
        try:
            _parse_timestamp(value["finalized_at"], "transaction receipt finalized_at")
        except OperationalError as exc:
            raise _failure(
                OperationalErrorCode.STORAGE_UNAVAILABLE,
                f"transaction receipt timestamp is invalid: {transaction_id}",
            ) from exc
        if manifest is not None and (
            value["transaction_sha256"] != manifest["transaction_sha256"]
            or value["request_sha256"] != manifest["request_sha256"]
            or value["kind"] != manifest["kind"]
            or value["origins"] != manifest["origins"]
            or value["after_positions"] != manifest["after_positions"]
        ):
            raise _failure(
                OperationalErrorCode.STORAGE_UNAVAILABLE,
                f"transaction receipt does not match its manifest: {transaction_id}",
            )
        if (
            applied_marker_bytes is not None
            and value["applied_marker_sha256"]
            != hashlib.sha256(applied_marker_bytes).hexdigest()
        ):
            raise _failure(
                OperationalErrorCode.STORAGE_UNAVAILABLE,
                f"transaction receipt applied marker digest mismatch: {transaction_id}",
            )
        return cast(dict[str, Any], value)

    def _prune_transaction_receipts(self) -> None:
        names = self._list_names(self.transaction_receipts_dir)
        for name in names:
            self._validate_safe_id(Path(name).stem, "transaction receipt id")
            if Path(name).suffix != ".json":
                raise _failure(
                    OperationalErrorCode.STORAGE_UNAVAILABLE,
                    f"unexpected transaction receipt path: {name}",
                )
        excess = len(names) - _MAX_TRANSACTION_RECEIPTS
        if excess <= 0:
            return
        try:
            with self._secure_io(create=False) as directory:
                for name in names[:excess]:
                    directory.unlink(self._relative(self.transaction_receipts_dir / name))
        except OperationalError:
            raise
        except (OSError, ValueError) as exc:
            raise _failure(
                OperationalErrorCode.STORAGE_UNAVAILABLE,
                "cannot prune operational transaction receipts",
            ) from exc

    def _finalize_transaction(
        self,
        manifest: Mapping[str, Any],
        *,
        transaction_root: Path,
        applied_marker_bytes: bytes,
        applied_marker: Mapping[str, Any],
        invoke_failpoints: bool,
    ) -> None:
        transaction_id = cast(str, manifest["transaction_id"])
        self._verify_transaction_targets(manifest, transaction_root=transaction_root)
        receipt_path = self.transaction_receipts_dir / f"{transaction_id}.json"
        receipt_bytes = self._transaction_receipt_bytes(
            manifest,
            applied_marker_bytes=applied_marker_bytes,
            applied_marker=applied_marker,
        )
        existing = self._optional_bytes(receipt_path)
        if existing is None:
            self._create_bytes_exclusive(receipt_path, receipt_bytes)
        elif existing != receipt_bytes:
            raise _failure(
                OperationalErrorCode.STORAGE_UNAVAILABLE,
                f"transaction receipt changed across recovery: {transaction_id}",
            )
        if invoke_failpoints:
            self._transaction_failpoint("after_receipt")
        try:
            with self._secure_io(create=False) as directory:
                directory.remove_tree(self._relative(transaction_root), missing_ok=True)
        except OperationalError:
            raise
        except (OSError, ValueError) as exc:
            raise _failure(
                OperationalErrorCode.STORAGE_UNAVAILABLE,
                f"cannot retire operational transaction: {transaction_id}",
            ) from exc
        self._prune_transaction_receipts()

    def _recover_transactions_locked(self) -> LedgerRecoveryReport:
        recovered: list[str] = []
        discarded: list[str] = []
        published = 0
        for transaction_id in self._list_names(self.transactions_dir):
            self._validate_safe_id(transaction_id, "transaction id")
            transaction_root = self.transactions_dir / transaction_id
            if not stat.S_ISDIR(self._stat(transaction_root).st_mode):
                raise _failure(
                    OperationalErrorCode.STORAGE_UNAVAILABLE,
                    f"unexpected transaction path: {transaction_root}",
                )
            receipt_path = self.transaction_receipts_dir / f"{transaction_id}.json"
            receipt_bytes = self._optional_bytes(receipt_path)
            if receipt_bytes is not None:
                manifest_for_receipt: dict[str, Any] | None = None
                manifest_path = transaction_root / "manifest.json"
                manifest_bytes = self._optional_bytes(manifest_path)
                applied_for_receipt: bytes | None = None
                if manifest_bytes is not None:
                    manifest_for_receipt = self._decode_transaction_manifest(
                        manifest_bytes,
                        transaction_id=transaction_id,
                    )
                    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
                    committed_for_receipt = self._optional_bytes(
                        transaction_root / "COMMITTED.json"
                    )
                    if committed_for_receipt is not None:
                        self._validate_transaction_marker(
                            transaction_root / "COMMITTED.json",
                            transaction_id=transaction_id,
                            manifest_sha256=manifest_sha256,
                            expected_phase="committed",
                        )
                    applied_for_receipt = self._optional_bytes(
                        transaction_root / "APPLIED.json"
                    )
                    if applied_for_receipt is not None:
                        self._validate_transaction_marker(
                            transaction_root / "APPLIED.json",
                            transaction_id=transaction_id,
                            manifest_sha256=manifest_sha256,
                            expected_phase="applied",
                        )
                self._validate_transaction_receipt(
                    receipt_bytes,
                    transaction_id=transaction_id,
                    manifest=manifest_for_receipt,
                    applied_marker_bytes=applied_for_receipt,
                )
                try:
                    with self._secure_io(create=False) as directory:
                        directory.remove_tree(self._relative(transaction_root), missing_ok=True)
                except (OSError, ValueError) as exc:
                    raise _failure(
                        OperationalErrorCode.STORAGE_UNAVAILABLE,
                        f"cannot finish retiring transaction: {transaction_id}",
                    ) from exc
                continue
            committed_path = transaction_root / "COMMITTED.json"
            applied_path = transaction_root / "APPLIED.json"
            committed = self._exists(committed_path)
            applied = self._exists(applied_path)
            if not committed:
                if applied:
                    raise _failure(
                        OperationalErrorCode.STORAGE_UNAVAILABLE,
                        f"transaction applied without commit point: {transaction_id}",
                    )
                try:
                    with self._secure_io(create=False) as directory:
                        directory.remove_tree(self._relative(transaction_root))
                except (OSError, ValueError) as exc:
                    raise _failure(
                        OperationalErrorCode.STORAGE_UNAVAILABLE,
                        f"cannot discard prepared transaction: {transaction_id}",
                    ) from exc
                discarded.append(transaction_id)
                self._write_recovery_record(
                    kind="discarded_transaction",
                    origins=(),
                    details={"transaction_id": transaction_id},
                )
                continue
            manifest_path = transaction_root / "manifest.json"
            manifest_bytes = self._read_bytes(
                manifest_path,
                "operational transaction manifest",
            )
            manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
            manifest = self._decode_transaction_manifest(
                manifest_bytes,
                transaction_id=transaction_id,
            )
            self._validate_transaction_marker(
                committed_path,
                transaction_id=transaction_id,
                manifest_sha256=manifest_sha256,
                expected_phase="committed",
            )
            if applied:
                applied_marker_bytes = self._read_bytes(
                    applied_path,
                    "operational transaction applied marker",
                )
                applied_marker = self._validate_transaction_marker(
                    applied_path,
                    transaction_id=transaction_id,
                    manifest_sha256=manifest_sha256,
                    expected_phase="applied",
                )
                self._finalize_transaction(
                    manifest,
                    transaction_root=transaction_root,
                    applied_marker_bytes=applied_marker_bytes,
                    applied_marker=applied_marker,
                    invoke_failpoints=False,
                )
                continue
            published += self._apply_transaction_manifest(
                manifest,
                transaction_root=transaction_root,
                invoke_failpoints=False,
            )
            applied_marker_bytes = self._marker_bytes(
                transaction_id=transaction_id,
                manifest_sha256=manifest_sha256,
                phase="applied",
            )
            self._create_bytes_exclusive(
                applied_path,
                applied_marker_bytes,
            )
            applied_marker = self._validate_transaction_marker(
                applied_path,
                transaction_id=transaction_id,
                manifest_sha256=manifest_sha256,
                expected_phase="applied",
            )
            self._finalize_transaction(
                manifest,
                transaction_root=transaction_root,
                applied_marker_bytes=applied_marker_bytes,
                applied_marker=applied_marker,
                invoke_failpoints=False,
            )
            recovered.append(transaction_id)
            origins = cast(list[str], manifest["origins"])
            self._write_recovery_record(
                kind="recovered_transaction",
                origins=origins,
                details={"transaction_id": transaction_id},
            )
        return LedgerRecoveryReport(
            recovered_transactions=tuple(recovered),
            discarded_transactions=tuple(discarded),
            published_targets=published,
        )

    def _recover_locked(self) -> LedgerRecoveryReport:
        transaction_report = self._recover_transactions_locked()
        repaired_tails: list[str] = []
        repaired_heads: list[str] = []
        recovered_compactions: list[str] = []
        for origin in self._discover_origins():
            anchor = self._read_anchor_with_checkpoint(origin)[0]
            events, torn = self._scan_events_chain(
                origin,
                anchor,
                allow_torn_tail=True,
            )
            tail_sequence = events[-1].origin_sequence if events else anchor.base_sequence
            tail_hash = events[-1].event_hash if events else anchor.base_event_hash
            cached = self._decode_head(origin)
            positions = {
                anchor.base_sequence: anchor.base_event_hash,
                **{event.origin_sequence: event.event_hash for event in events},
            }
            current_prefix = cached is None or (
                cached[2] == anchor.anchor_hash
                and cached[0] <= tail_sequence
                and positions.get(cached[0]) == cached[1]
            )
            legacy_compaction = (
                cached is not None
                and anchor.kind == "compaction"
                and bool(anchor.previous_anchor_hash)
                and cached[2] == anchor.previous_anchor_hash
                and cached[0] == anchor.base_sequence
                and cached[1] == anchor.base_event_hash
            )
            if legacy_compaction:
                assert cached is not None
                predecessor = self._read_anchor_history(
                    origin,
                    anchor.previous_anchor_hash,
                )
                self._validate_anchor_transition(
                    predecessor,
                    anchor,
                    OriginPosition(
                        origin_device=origin,
                        sequence=cached[0],
                        event_hash=cached[1],
                        anchor_hash=cached[2],
                    ),
                )
            if not current_prefix and not legacy_compaction:
                raise _failure(
                    OperationalErrorCode.ANCHOR_CONFLICT,
                    f"operational head is advanced or forked for origin {origin}",
                )
            if torn is not None:
                path, prefix_size = torn
                try:
                    with self._secure_io(create=False) as directory:
                        directory.truncate(self._relative(path), prefix_size)
                except OperationalError:
                    raise
                except (OSError, ValueError) as exc:
                    raise _failure(
                        OperationalErrorCode.STORAGE_UNAVAILABLE,
                        f"cannot truncate torn operational segment: {path}",
                    ) from exc
                repaired_tails.append(origin)
            expected_head = (tail_sequence, tail_hash, anchor.anchor_hash)
            if cached != expected_head:
                self._write_head(
                    origin=origin,
                    sequence=tail_sequence,
                    event_hash=tail_hash,
                    anchor_hash=anchor.anchor_hash,
                )
                if legacy_compaction:
                    recovered_compactions.append(origin)
                elif torn is None:
                    repaired_heads.append(origin)
            if torn is not None or cached != expected_head:
                self._write_recovery_record(
                    kind=(
                        "legacy_compaction"
                        if legacy_compaction
                        else ("torn_tail" if torn is not None else "stale_head")
                    ),
                    origins=(origin,),
                    details={
                        "sequence": tail_sequence,
                        "event_hash": tail_hash,
                        "anchor_hash": anchor.anchor_hash,
                    },
                )
        return LedgerRecoveryReport(
            recovered_transactions=transaction_report.recovered_transactions,
            discarded_transactions=transaction_report.discarded_transactions,
            repaired_tails=tuple(sorted(repaired_tails)),
            repaired_heads=tuple(sorted(repaired_heads)),
            recovered_compactions=tuple(sorted(recovered_compactions)),
            published_targets=transaction_report.published_targets,
        )

    def recover(self) -> LedgerRecoveryReport:
        with self._authority_operation(recover=False):
            return self._recover_locked()

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
        with self._authority_operation():
            anchor = self._read_anchor_with_checkpoint(origin)[0]
            return self._load_position(origin, anchor, repair=False)

    def _discover_origins(self) -> tuple[str, ...]:
        origins: set[str] = set()
        for directory, suffix, nested in (
            (self.anchors_dir, ".json", False),
            (self.heads_dir, ".json", False),
            (self.events_dir, "", True),
        ):
            for name in self._list_names(directory):
                path = directory / name
                observed = self._stat(path)
                if nested:
                    if not stat.S_ISDIR(observed.st_mode):
                        raise _failure(
                            OperationalErrorCode.INVALID_EVENT,
                            f"unexpected operational event origin path: {path}",
                        )
                    origin = path.name
                else:
                    if not stat.S_ISREG(observed.st_mode) or path.suffix != suffix:
                        raise _failure(
                            OperationalErrorCode.INVALID_EVENT,
                            f"unexpected operational authority path: {path}",
                        )
                    origin = path.stem
                self._validate_safe_id(origin, "origin device")
                origins.add(origin)
        return tuple(sorted(origins))

    def positions(self) -> tuple[OriginPosition, ...]:
        with self._authority_operation():
            positions: list[OriginPosition] = []
            for origin in self._discover_origins():
                anchor = self._read_anchor_with_checkpoint(origin)[0]
                positions.append(self._load_position(origin, anchor, repair=False))
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
        with self._authority_operation():
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
        if self._transaction_staging:
            return
        path = self._segment_path(event)
        try:
            with self._secure_io(create=True) as directory:
                directory.append_bytes(
                    self._relative(path),
                    canonical_json_bytes(event) + b"\n",
                )
        except OperationalError:
            raise
        except (OSError, ValueError) as exc:
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
            if command.source_proof is not None:
                raise _failure(
                    OperationalErrorCode.INVALID_EVENT,
                    "source proof requires authenticated migration authority",
                )
            return
        if migration.migration_device_id != anchor.origin_device:
            raise _failure(
                OperationalErrorCode.INVALID_EVENT,
                "migration origin does not match anchored origin",
            )
        if anchor.kind == "memo_v1" and (
            migration.source_manifest_sha256 != anchor.source_manifest_sha256
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
        event_id: str | None = None,
    ) -> OperationalEventV2:
        content_hash = hashlib.sha256(canonical_json_bytes(asdict(command))).hexdigest()
        unsigned = OperationalEventV2(
            schema="memo.operational_event.v2",
            schema_version=2,
            event_id=event_id or uuid.uuid4().hex,
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
            migration_origin=context.migration_origin,
            migration_origin_sha256=(
                hashlib.sha256(canonical_json_bytes(context.migration_origin)).hexdigest()
                if context.migration_origin is not None
                else ""
            ),
        )
        return replace(unsigned, event_hash=canonical_event_hash(unsigned))

    @staticmethod
    def _validate_migration_event_id(
        command: OperationalCommand,
        context: CommitContext,
        event_id: str,
    ) -> None:
        migration = context.migration_origin
        if (
            migration is None
            or command.source_proof is None
            or command.source_proof.source_system != "memo_v1"
        ):
            raise _failure(
                OperationalErrorCode.INVALID_EVENT,
                "deterministic event IDs require a memo_v1 migration context",
            )
        expected_prefix = f"memo-v1/{migration.source_manifest_sha256}/"
        if (
            not isinstance(event_id, str)
            or event_id != command.idempotency_key
            or not event_id.startswith(expected_prefix)
            or len(event_id) <= len(expected_prefix)
        ):
            raise _failure(
                OperationalErrorCode.INVALID_EVENT,
                "deterministic migration event ID is outside its sealed namespace",
            )

    def _append(
        self,
        command: OperationalCommand,
        *,
        context: CommitContext,
        event_id: str | None,
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
        if context.identity.device_id != self.device_id:
            raise _failure(
                OperationalErrorCode.INVALID_EVENT,
                "authenticated principal origin differs from this ledger device",
            )
        if command.actor != context.identity:
            raise _failure(
                OperationalErrorCode.INVALID_EVENT,
                "operational command actor differs from authenticated context",
            )
        if event_id is not None:
            self._validate_migration_event_id(command, context, event_id)
        validate_event_payload(command.event_type, command.payload)
        with self._authority_operation(context=context):
            signer, roster = self._signing_authority()
            origin = (
                context.migration_origin.migration_device_id
                if context.migration_origin is not None
                else self.device_id
            )
            self._validate_safe_id(origin, "origin device")
            anchor_path = self._anchor_path(origin)
            if self._exists(anchor_path):
                anchor = self._read_anchor_with_checkpoint(origin)[0]
            elif origin == self.device_id:
                anchor = self._ensure_anchor_locked()
            else:
                raise _failure(
                    OperationalErrorCode.ANCHOR_CONFLICT,
                    "migration origin requires its own pre-existing anchor",
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
                activation_sequence=roster.version,
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
                event_id=event_id,
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
            self._validate_source_proof(event, anchor)
            self._append_event_fsync(event)
            self._write_head_atomic(event)
            return event

    def append(
        self,
        command: OperationalCommand,
        *,
        context: CommitContext,
    ) -> OperationalEventV2:
        return self._append(command, context=context, event_id=None)

    def append_migration_seed(
        self,
        command: OperationalCommand,
        *,
        context: CommitContext,
        event_id: str,
    ) -> OperationalEventV2:
        """Append one deterministic event only inside sealed memo-v1 migration."""
        return self._append(command, context=context, event_id=event_id)

    def verify(self) -> VerificationReport:
        checked_origins: list[str] = []
        checked_events = 0
        positions: list[OriginPosition] = []
        errors: list[str] = []
        try:
            with self._authority_operation():
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
            self._validate_source_proof(event, anchor)
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
        if not self._exists(anchor_path):
            return list(bundle.events), 0
        current, _ = self._read_anchor_with_checkpoint(origin)
        if current.anchor_hash != bundle.anchor.anchor_hash:
            position = self._load_position(origin, current, repair=False)
            self._validate_anchor_transition(current, bundle.anchor, position)
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

    def _assert_current_event_authority(
        self,
        event: OperationalEventV2,
        roster: VerificationRoster,
    ) -> None:
        if event.roster_version != roster.version:
            raise _failure(
                OperationalErrorCode.SIGNATURE_INVALID,
                "new event must use the latest pinned roster; historical bytes are replay-only",
            )
        expected = self._active_key(
            roster,
            device_id=event.origin_device,
            role="origin",
            activation_sequence=roster.version,
        )
        if expected.key_id != event.key_id:
            raise _failure(
                OperationalErrorCode.SIGNATURE_INVALID,
                "new event signer is revoked or not the current origin authority",
            )

    def _validate_import_bundles_locked(
        self, bundles: Sequence[OriginBundle]
    ) -> list[tuple[OriginBundle, list[OperationalEventV2], int]]:
        seen: set[str] = set()
        validated: list[tuple[OriginBundle, list[OperationalEventV2], int]] = []
        latest = self._latest_roster()
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
            anchor_path = self._anchor_path(origin)
            anchor_is_new = not self._exists(anchor_path)
            if not anchor_is_new:
                current = self._read_anchor_with_checkpoint(origin)[0]
                anchor_is_new = current.anchor_hash != bundle.anchor.anchor_hash
            if anchor_is_new:
                self._assert_current_anchor_authority(bundle.anchor, latest)
            for event in new_events:
                self._assert_current_event_authority(event, latest)
            validated.append((bundle, new_events, replayed))
        return validated

    def validate_import_bundles(self, bundles: Iterable[OriginBundle]) -> tuple[OriginBundle, ...]:
        materialized = tuple(bundles)
        with self._authority_operation():
            self._validate_import_bundles_locked(materialized)
        return materialized

    @staticmethod
    def _bundle_wire(bundle: OriginBundle) -> dict[str, object]:
        return {
            "anchor": operational_wire_dict(bundle.anchor),
            "checkpoint": base64.urlsafe_b64encode(bundle.checkpoint).rstrip(b"=").decode("ascii"),
            "events": [operational_wire_dict(event) for event in bundle.events],
            "head_sequence": bundle.head_sequence,
            "head_hash": bundle.head_hash,
        }

    @staticmethod
    def _position_wire(position: OriginPosition | None, *, origin: str) -> dict[str, object]:
        if position is None:
            return {
                "origin_device": origin,
                "sequence": None,
                "event_hash": None,
                "anchor_hash": None,
            }
        return cast(dict[str, object], operational_wire_dict(position))

    def _run_transaction(
        self,
        *,
        kind: str,
        request_sha256: str,
        origins: Sequence[str],
        before_positions: Sequence[Mapping[str, object]],
        after_positions: Sequence[Mapping[str, object]],
        target_bytes: Sequence[tuple[Path, bytes]],
    ) -> str | None:
        if not target_bytes:
            return None
        transaction_id = uuid.uuid4().hex
        transaction_root = self.transactions_dir / transaction_id
        non_heads = [
            (path, encoded)
            for path, encoded in target_bytes
            if not self._relative(path).as_posix().startswith("heads/")
        ]
        heads = [
            (path, encoded)
            for path, encoded in target_bytes
            if self._relative(path).as_posix().startswith("heads/")
        ]
        ordered = [*non_heads, *heads]
        targets: list[dict[str, object]] = []
        staged: list[tuple[Path, bytes]] = []
        seen: set[str] = set()
        for ordinal, (path, encoded) in enumerate(ordered):
            relative = self._relative(path).as_posix()
            if relative in seen:
                raise _failure(
                    OperationalErrorCode.STORAGE_UNAVAILABLE,
                    f"duplicate operational transaction target: {relative}",
                )
            seen.add(relative)
            before = self._optional_bytes(path)
            before_sha = hashlib.sha256(before).hexdigest() if before is not None else None
            after_sha = hashlib.sha256(encoded).hexdigest()
            stage_blob = f"stage/{ordinal:06d}.bin"
            targets.append(
                {
                    "relative_target": relative,
                    "mode": "create" if before is None else "replace",
                    "before_sha256": before_sha,
                    "after_sha256": after_sha,
                    "size": len(encoded),
                    "stage_blob": stage_blob,
                }
            )
            staged.append((transaction_root / stage_blob, encoded))
        manifest: dict[str, object] = {
            "schema": _TRANSACTION_SCHEMA,
            "transaction_id": transaction_id,
            "transaction_sha256": "",
            "request_sha256": request_sha256,
            "kind": kind,
            "origins": tuple(sorted(set(origins))),
            "before_positions": tuple(before_positions),
            "after_positions": tuple(after_positions),
            "targets": tuple(targets),
            "prepared_at": self._now(),
        }
        manifest["transaction_sha256"] = hashlib.sha256(canonical_json_bytes(manifest)).hexdigest()
        manifest_bytes = canonical_json_bytes(manifest)
        # Decode our own manifest before creating the durable prepare.
        self._decode_transaction_manifest(
            manifest_bytes,
            transaction_id=transaction_id,
        )
        for path, encoded in staged:
            self._atomic_write_bytes(path, encoded)
        self._atomic_write_bytes(transaction_root / "manifest.json", manifest_bytes)
        self._transaction_failpoint("before_commit")
        manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        self._create_bytes_exclusive(
            transaction_root / "COMMITTED.json",
            self._marker_bytes(
                transaction_id=transaction_id,
                manifest_sha256=manifest_sha256,
                phase="committed",
            ),
        )
        self._transaction_failpoint("after_commit")
        decoded = self._decode_transaction_manifest(
            manifest_bytes,
            transaction_id=transaction_id,
        )
        self._apply_transaction_manifest(
            decoded,
            transaction_root=transaction_root,
            invoke_failpoints=True,
        )
        applied_path = transaction_root / "APPLIED.json"
        applied_marker_bytes = self._marker_bytes(
            transaction_id=transaction_id,
            manifest_sha256=manifest_sha256,
            phase="applied",
        )
        self._create_bytes_exclusive(
            applied_path,
            applied_marker_bytes,
        )
        self._transaction_failpoint("after_applied")
        applied_marker = self._validate_transaction_marker(
            applied_path,
            transaction_id=transaction_id,
            manifest_sha256=manifest_sha256,
            expected_phase="applied",
        )
        self._finalize_transaction(
            decoded,
            transaction_root=transaction_root,
            applied_marker_bytes=applied_marker_bytes,
            applied_marker=applied_marker,
            invoke_failpoints=True,
        )
        return transaction_id

    def _import_transaction_targets(
        self,
        validated: Sequence[tuple[OriginBundle, list[OperationalEventV2], int]],
    ) -> tuple[
        list[tuple[Path, bytes]],
        list[dict[str, object]],
        list[dict[str, object]],
    ]:
        target_bytes: list[tuple[Path, bytes]] = []
        before_positions: list[dict[str, object]] = []
        after_positions: list[dict[str, object]] = []
        head_targets: list[tuple[Path, bytes]] = []
        self._transaction_staging = True
        try:
            for _, new_events, _ in validated:
                for event in new_events:
                    # Preserve the existing private durability hook without
                    # publishing authority bytes before COMMITTED.
                    self._append_event_fsync(event)
        finally:
            self._transaction_staging = False
        for bundle, new_events, _ in validated:
            origin = bundle.anchor.origin_device
            anchor_path = self._anchor_path(origin)
            existing_anchor: ChainAnchor | None = None
            before_position: OriginPosition | None = None
            if self._exists(anchor_path):
                existing_anchor = self._read_anchor_with_checkpoint(origin)[0]
                before_position = self._load_position(
                    origin,
                    existing_anchor,
                    repair=False,
                )
            anchor_changed = (
                existing_anchor is None or existing_anchor.anchor_hash != bundle.anchor.anchor_hash
            )
            before_positions.append(self._position_wire(before_position, origin=origin))
            after_position = OriginPosition(
                origin_device=origin,
                sequence=bundle.head_sequence,
                event_hash=bundle.head_hash,
                anchor_hash=bundle.anchor.anchor_hash,
            )
            after_positions.append(self._position_wire(after_position, origin=origin))
            if anchor_changed:
                if existing_anchor is not None:
                    existing_history_path = self._anchor_history_path(
                        origin,
                        existing_anchor.anchor_hash,
                    )
                    existing_history_bytes = canonical_json_bytes(existing_anchor)
                    if self._optional_bytes(existing_history_path) is None:
                        self._persist_anchor_history(existing_anchor)
                    target_bytes.append(
                        (existing_history_path, existing_history_bytes),
                    )
                target_bytes.extend(
                    (
                        (self._checkpoint_path(bundle.anchor), bundle.checkpoint),
                        (
                            self._anchor_history_path(
                                origin,
                                bundle.anchor.anchor_hash,
                            ),
                            canonical_json_bytes(bundle.anchor),
                        ),
                        (
                            anchor_path,
                            canonical_json_bytes(bundle.anchor),
                        ),
                    )
                )
            segments: dict[Path, bytearray] = {}
            for event in new_events:
                path = self._segment_path(event)
                if path not in segments:
                    segments[path] = bytearray(self._optional_bytes(path) or b"")
                segments[path].extend(canonical_json_bytes(event) + b"\n")
            target_bytes.extend(
                (path, bytes(encoded))
                for path, encoded in sorted(
                    segments.items(),
                    key=lambda item: item[0].as_posix(),
                )
            )
            if anchor_changed or new_events:
                head_targets.append(
                    (
                        self._head_path(origin),
                        canonical_json_bytes(
                            {
                                "schema": _HEAD_SCHEMA,
                                "origin_device": origin,
                                "sequence": bundle.head_sequence,
                                "event_hash": bundle.head_hash,
                                "anchor_hash": bundle.anchor.anchor_hash,
                            }
                        ),
                    )
                )
        target_bytes.extend(head_targets)
        return target_bytes, before_positions, after_positions

    def quarantine(self, bundle: OriginBundle, *, reason: str) -> Path:
        encoded_bundle = canonical_json_bytes(self._bundle_wire(bundle))
        digest = hashlib.sha256(encoded_bundle).hexdigest()
        with self._authority_operation():
            suffix = f"-{digest}.json"
            for name in self._list_names(self.quarantine_dir):
                if name.endswith(suffix):
                    path = self.quarantine_dir / name
                    if not stat.S_ISREG(self._stat(path).st_mode):
                        raise _failure(
                            OperationalErrorCode.STORAGE_UNAVAILABLE,
                            f"unsafe quarantine path: {path}",
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

    def import_bundles(
        self,
        bundles: Iterable[OriginBundle],
        *,
        context: CommitContext,
    ) -> LedgerImportReport:
        materialized = tuple(bundles)
        quarantined: list[str] = []
        if not isinstance(context, CommitContext):
            raise _failure(
                OperationalErrorCode.INVALID_EVENT,
                "authenticated commit context authority is required for import",
            )
        if self.epoch_fence is None:
            raise _failure(
                OperationalErrorCode.INVALID_EVENT,
                "operational epoch authority is unavailable",
            )
        manifest_sha256 = hashlib.sha256(
            canonical_json_bytes(
                sorted(
                    (self._bundle_wire(bundle) for bundle in materialized),
                    key=lambda item: cast(dict[str, Any], item["anchor"])["origin_device"],
                )
            )
        ).hexdigest()
        try:
            with self._authority_operation(context=context):
                validated = self._validate_import_bundles_locked(materialized)
                target_bytes, before_positions, after_positions = self._import_transaction_targets(
                    validated
                )
                inserted = sum(len(new_events) for _, new_events, _ in validated)
                replayed = sum(replay_count for _, _, replay_count in validated)
                self._run_transaction(
                    kind="bundle_import",
                    request_sha256=manifest_sha256,
                    origins=tuple(bundle.anchor.origin_device for bundle, _, _ in validated),
                    before_positions=before_positions,
                    after_positions=after_positions,
                    target_bytes=target_bytes,
                )
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
        with self._authority_operation():
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
