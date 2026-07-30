"""Signed, immutable verification-roster history."""

from __future__ import annotations

import base64
import hashlib
import json
import os
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from memo.atomic_io import (
    authority_admission_lock,
    authority_write_lock,
    open_secure_directory,
)
from memo.errors import SignatureError
from memo.operational_key_store import (
    AuthorityPinStore,
    DeviceKeyStore,
    KeyStoreError,
    PublicKeyRecord,
    SignatureAlgorithm,
    _is_canonical_p256_signature,
)
from memo.operational_signing import (
    OperationalSigner,
    OperationalVerifier,
    SignatureEnvelope,
)

_BOOTSTRAP_DOMAIN = "memo.operational.roster.bootstrap.v1"
_UPDATE_DOMAIN = "memo.operational.roster.v1"


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _decode(value: str) -> bytes:
    try:
        return base64.b64decode(
            value + "=" * (-len(value) % 4),
            altchars=b"-_",
            validate=True,
        )
    except (ValueError, TypeError) as exc:
        raise RosterError("invalid roster base64url value") from exc


def _fsync_directory(path: Path) -> None:
    with open_secure_directory(path) as directory:
        os.fsync(directory.descriptor)


def _atomic_authority_write(path: Path, data: bytes) -> None:
    with open_secure_directory(path.parent, create=True) as directory:
        directory.atomic_write_bytes(path.name, data)


def _create_authority_file(path: Path, data: bytes) -> None:
    with open_secure_directory(path.parent, create=True) as directory:
        directory.create_bytes_exclusive(path.name, data)


def _authority_relative(root: Path, path: Path) -> Path:
    try:
        relative = Path(path).absolute().relative_to(Path(root).absolute())
    except ValueError as exc:
        raise RosterError(f"verification roster path escapes authority root: {path}") from exc
    if not relative.parts:
        raise RosterError("verification roster file path is required")
    return relative


def _read_authority_bytes(root: Path, path: Path) -> bytes:
    with open_secure_directory(root) as directory:
        return directory.read_bytes(_authority_relative(root, path))


def _optional_authority_bytes(root: Path, path: Path) -> bytes | None:
    try:
        return _read_authority_bytes(root, path)
    except FileNotFoundError:
        return None


def _authority_json_paths(root: Path, directory_path: Path) -> list[Path]:
    try:
        with open_secure_directory(root) as directory:
            names = directory.list_names(_authority_relative(root, directory_path))
    except FileNotFoundError:
        return []
    return sorted(directory_path / name for name in names if Path(name).suffix == ".json")


class RosterError(SignatureError):
    """A verification roster is structurally or cryptographically invalid."""


@dataclass(frozen=True)
class VerificationRoster:
    version: int
    peers: tuple[str, ...]
    keys: tuple[PublicKeyRecord, ...]
    local_device_id: str
    schema: Literal["memo.operational_roster.v1"] = "memo.operational_roster.v1"
    created_at: str = ""
    previous_roster_hash: str = ""
    roster_hash: str = ""
    signature: SignatureEnvelope | None = None

    def __post_init__(self) -> None:
        if self.schema != "memo.operational_roster.v1":
            raise RosterError(f"unsupported verification roster schema: {self.schema}")
        if self.version < 1:
            raise RosterError("roster version must be positive")
        if tuple(sorted(set(self.peers))) != self.peers:
            raise RosterError("roster peers must be sorted and unique")
        if self.local_device_id not in self.peers:
            raise RosterError("local device is absent from roster peers")
        key_ids = [key.key_id for key in self.keys]
        fingerprints = [key.fingerprint for key in self.keys]
        if len(key_ids) != len(set(key_ids)):
            raise RosterError("duplicate roster key id")
        if len(fingerprints) != len(set(fingerprints)):
            raise RosterError("duplicate roster key fingerprint")
        if any(key.device_id not in self.peers for key in self.keys):
            raise RosterError("roster key belongs to an unknown peer")
        allowed_roles = {"origin", "migration_attestor"}
        for key in self.keys:
            if (
                not key.roles
                or len(key.roles) != len(set(key.roles))
                or not set(key.roles).issubset(allowed_roles)
            ):
                raise RosterError(f"invalid roles for roster key: {key.key_id}")
            public_bytes = _decode(key.public_key)
            try:
                bound_algorithm = DeviceKeyStore.algorithm_for_key_id(key.key_id)
            except KeyStoreError:
                bound_algorithm = None
            if key.algorithm == "ed25519":
                valid_public_key = bound_algorithm == key.algorithm and len(public_bytes) == 32
            elif key.algorithm == "ecdsa-p256-sha256":
                valid_public_key = (
                    bound_algorithm == key.algorithm
                    and len(public_bytes) == 65
                    and public_bytes[:1] == b"\x04"
                )
                if valid_public_key:
                    try:
                        ec.EllipticCurvePublicKey.from_encoded_point(
                            ec.SECP256R1(),
                            public_bytes,
                        )
                    except ValueError:
                        valid_public_key = False
            else:
                valid_public_key = False
            if not valid_public_key or hashlib.sha256(public_bytes).hexdigest() != key.fingerprint:
                raise RosterError(f"public key fingerprint mismatch: {key.key_id}")

    @property
    def local_key_id(self) -> str:
        keys = [
            key.key_id
            for key in self.keys
            if key.device_id == self.local_device_id
            and "origin" in key.roles
            and key.enrollment_sequence <= self.version
            and (key.revocation_sequence is None or key.revocation_sequence > self.version)
        ]
        if len(keys) != 1:
            raise RosterError("local roster must have exactly one active key")
        return keys[0]

    def key(self, key_id: str) -> PublicKeyRecord:
        matches = [key for key in self.keys if key.key_id == key_id]
        if len(matches) != 1:
            raise RosterError(f"unknown roster key: {key_id}")
        return matches[0]

    def to_dict(self, *, blank_signature: bool = False) -> dict[str, Any]:
        signature: object
        if blank_signature or self.signature is None:
            signature = ""
        else:
            signature = {
                "algorithm": self.signature.algorithm,
                "key_id": self.signature.key_id,
                "roster_version": self.signature.roster_version,
                "signature": self.signature.signature,
            }
        return {
            "schema": self.schema,
            "version": self.version,
            "peers": list(self.peers),
            "keys": [key.to_dict() for key in self.keys],
            "local_device_id": self.local_device_id,
            "created_at": self.created_at,
            "previous_roster_hash": self.previous_roster_hash,
            "roster_hash": self.roster_hash,
            "signature": signature,
        }

    def _hash(self) -> str:
        body = self.to_dict(blank_signature=True)
        body["roster_hash"] = ""
        return hashlib.sha256(_canonical(body)).hexdigest()

    def _signed_payload(self) -> bytes:
        return _canonical(self.to_dict(blank_signature=True))

    @classmethod
    def bootstrap(
        cls,
        *,
        device_id: str,
        key: PublicKeyRecord,
        root: Path,
        pin_store: AuthorityPinStore | None = None,
    ) -> VerificationRoster:
        root = Path(root)
        pin_store = _resolve_pin_store(root, pin_store)
        if key.device_id != device_id:
            raise RosterError("bootstrap key device mismatch")
        allowed_roles = {"origin", "migration_attestor"}
        if not key.roles or not set(key.roles).issubset(allowed_roles):
            raise RosterError("bootstrap key has unsupported roles")
        roster = cls(
            version=1,
            peers=(device_id,),
            keys=(key,),
            local_device_id=device_id,
            created_at="",
        )
        roster = replace(roster, roster_hash=roster._hash())
        if not key._bootstrap_signature:
            raise RosterError("bootstrap key lacks a roster authorization")
        roster = replace(
            roster,
            signature=SignatureEnvelope(
                algorithm=key.algorithm,
                key_id=key.key_id,
                roster_version=1,
                signature=key._bootstrap_signature,
            ),
        )
        _verify_roster(roster, previous=None)
        encoded = _canonical(roster.to_dict())
        history = root / "verification-rosters" / "00000001.json"
        current = root / "verification-roster.json"
        with authority_admission_lock(root):
            try:
                pin_store._stage_roster(root, encoded)
            except KeyStoreError as exc:
                raise RosterError("verification roster authority pin rejected bootstrap") from exc
            with authority_write_lock(root / "verification-rosters"):
                history_bytes = _optional_authority_bytes(root, history)
                current_bytes = _optional_authority_bytes(root, current)
                if history_bytes not in {None, encoded} or current_bytes not in {
                    None,
                    encoded,
                }:
                    raise RosterError("verification roster already exists")
                if history_bytes is None:
                    _create_authority_file(history, encoded)
                if current_bytes is None:
                    _atomic_authority_write(current, encoded)
            try:
                pin_store._finish_roster(root, encoded)
            except KeyStoreError as exc:
                raise RosterError("verification roster authority pin commit failed") from exc
        return roster

    @classmethod
    def load(
        cls,
        root: Path,
        *,
        pin_store: AuthorityPinStore | None = None,
    ) -> VerificationRoster:
        history = _load_pinned_roster_history(root, pin_store)
        return history[-1]

    @classmethod
    def load_version(
        cls,
        root: Path,
        *,
        version: int,
        pin_store: AuthorityPinStore | None = None,
    ) -> VerificationRoster:
        """Load one historical roster only through the complete pinned history."""
        if isinstance(version, bool) or not isinstance(version, int) or version < 1:
            raise RosterError("verification roster version must be positive")
        history = _load_pinned_roster_history(root, pin_store)
        if version > len(history):
            raise RosterError(f"verification roster version is missing: {version}")
        roster = history[version - 1]
        if roster.version != version:
            raise RosterError("verification roster history has a version gap")
        return roster

    def with_keys(
        self,
        *,
        version: int,
        peers: tuple[str, ...],
        keys: tuple[PublicKeyRecord, ...],
        signer: OperationalSigner | None = None,
        root: Path | None = None,
        pin_store: AuthorityPinStore | None = None,
    ) -> VerificationRoster:
        if version <= self.version:
            raise RosterError("roster version regression")
        if version != self.version + 1:
            raise RosterError("roster version must advance exactly once")
        if signer is None or root is None:
            raise RosterError("signed roster update requires signer and root")
        root = Path(root)
        pin_store = _resolve_pin_store(root, pin_store)
        with authority_admission_lock(root):
            canonical_predecessor = VerificationRoster.load(
                root,
                pin_store=pin_store,
            )
            if canonical_predecessor != self:
                raise RosterError("signed roster update requires the exact pinned predecessor")
            updated = VerificationRoster(
                version=version,
                peers=peers,
                keys=keys,
                local_device_id=self.local_device_id,
                created_at=datetime.now(UTC).isoformat(timespec="milliseconds"),
                previous_roster_hash=self.roster_hash,
            )
            updated = replace(updated, roster_hash=updated._hash())
            envelope = signer.sign(
                domain=_UPDATE_DOMAIN,
                payload=updated._signed_payload(),
                key_id=self.local_key_id,
            )
            updated = replace(updated, signature=envelope)
            _verify_roster(updated, previous=self)
            encoded = _canonical(updated.to_dict())
            history = root / "verification-rosters" / f"{version:08d}.json"
            current = root / "verification-roster.json"
            with authority_write_lock(root / "verification-rosters"):
                loaded = _load_roster_files(root)
                if loaded != self:
                    raise RosterError("verification roster changed concurrently")
                try:
                    pin_store._stage_roster(root, encoded)
                except KeyStoreError as exc:
                    raise RosterError("verification roster authority pin rejected update") from exc
                history_bytes = _optional_authority_bytes(root, history)
                if history_bytes is not None:
                    if history_bytes != encoded:
                        raise RosterError("verification roster history already exists")
                else:
                    _create_authority_file(history, encoded)
                _atomic_authority_write(current, encoded)
            try:
                pin_store._finish_roster(root, encoded)
            except KeyStoreError as exc:
                raise RosterError("verification roster authority pin commit failed") from exc
            return updated


def _resolve_pin_store(
    root: Path,
    pin_store: AuthorityPinStore | None,
) -> AuthorityPinStore:
    if pin_store is not None:
        return pin_store
    try:
        return AuthorityPinStore.for_root(root)
    except KeyStoreError as exc:
        raise RosterError("verification roster authority pin is unavailable") from exc


def _load_pinned_roster_history(
    root: Path,
    pin_store: AuthorityPinStore | None,
) -> tuple[VerificationRoster, ...]:
    root = Path(root)
    resolved_pin_store = _resolve_pin_store(root, pin_store)
    _recover_prepared_roster(root, resolved_pin_store)
    with authority_write_lock(root / "verification-rosters"):
        history = _load_roster_history(root)
        latest = history[-1]
        try:
            resolved_pin_store._verify_roster(
                root,
                version=latest.version,
                roster_hash=latest.roster_hash,
            )
        except KeyStoreError as exc:
            raise RosterError("verification roster rollback or substitution detected") from exc
    return history


def _recover_prepared_roster(root: Path, pin_store: AuthorityPinStore) -> None:
    try:
        encoded = pin_store._prepared_roster(root)
        state = pin_store._read(root)
    except KeyStoreError as exc:
        raise RosterError("verification roster authority pin is unavailable") from exc
    if encoded is None:
        return
    prepared = _decode_roster_bytes(encoded, "prepared authority record")
    if (
        state.pending_roster_version != prepared.version
        or state.pending_roster_hash != prepared.roster_hash
        or prepared.version != state.roster_version + 1
    ):
        raise RosterError("prepared verification roster does not match its authority pin")

    history_root = root / "verification-rosters"
    current_path = root / "verification-roster.json"
    target_path = history_root / f"{prepared.version:08d}.json"
    with authority_write_lock(history_root):
        paths = _authority_json_paths(root, history_root)
        expected_predecessor_names = [
            f"{version:08d}.json" for version in range(1, prepared.version)
        ]
        actual_predecessor_names = [path.name for path in paths if path.name != target_path.name]
        if actual_predecessor_names != expected_predecessor_names or any(
            path.name > target_path.name for path in paths
        ):
            raise RosterError("prepared verification roster destination is inconsistent")

        previous: VerificationRoster | None = None
        for path in paths:
            if path.name == target_path.name:
                continue
            roster = _decode_roster(path, root=root)
            _verify_roster(roster, previous=previous)
            previous = roster
        if previous is None:
            if state.roster_version != 0 or state.roster_hash:
                raise RosterError("prepared bootstrap roster predecessor is missing")
        elif previous.version != state.roster_version or previous.roster_hash != state.roster_hash:
            raise RosterError("prepared verification roster predecessor is not pinned")
        _verify_roster(prepared, previous=previous)

        try:
            target_bytes = _optional_authority_bytes(root, target_path)
            current_bytes = _optional_authority_bytes(root, current_path)
        except OSError as exc:
            raise RosterError("prepared verification roster destination is invalid") from exc
        previous_bytes = _canonical(previous.to_dict()) if previous is not None else None
        if target_bytes not in {None, encoded} or current_bytes not in {
            None,
            previous_bytes,
            encoded,
        }:
            raise RosterError("prepared verification roster destination is inconsistent")
        if previous is not None and current_bytes is None:
            raise RosterError("prepared verification roster predecessor pointer is missing")
        if target_bytes is None:
            _create_authority_file(target_path, encoded)
        if current_bytes != encoded:
            _atomic_authority_write(current_path, encoded)
    try:
        pin_store._finish_roster(root, encoded)
    except KeyStoreError as exc:
        raise RosterError("verification roster authority pin commit failed") from exc


def _load_roster_files(root: Path) -> VerificationRoster:
    return _load_roster_history(root)[-1]


def _load_roster_history(root: Path) -> tuple[VerificationRoster, ...]:
    root = Path(root)
    history_root = root / "verification-rosters"
    current_path = root / "verification-roster.json"
    paths = _authority_json_paths(root, history_root)
    if not paths:
        raise RosterError("verification roster history is missing")
    previous: VerificationRoster | None = None
    history: list[VerificationRoster] = []
    for expected, path in enumerate(paths, start=1):
        if path.name != f"{expected:08d}.json":
            raise RosterError("verification roster history has a version gap")
        roster = _decode_roster(path, root=root)
        _verify_roster(roster, previous=previous)
        previous = roster
        history.append(roster)
    assert previous is not None
    try:
        current_bytes: bytes | None = _read_authority_bytes(root, current_path)
    except FileNotFoundError:
        current_bytes = None
    except OSError as exc:
        raise RosterError("verification roster current pointer is invalid") from exc
    latest_bytes = _canonical(previous.to_dict())
    if current_bytes != latest_bytes:
        raise RosterError("verification roster current pointer mismatch")
    return tuple(history)


def _decode_roster(path: Path, *, root: Path) -> VerificationRoster:
    try:
        encoded = _read_authority_bytes(root, path)
    except OSError as exc:
        raise RosterError(f"invalid verification roster: {path}") from exc
    return _decode_roster_bytes(encoded, str(path))


def _decode_key_algorithm(item: dict[str, Any]) -> SignatureAlgorithm:
    if "algorithm" not in item:
        return "ed25519"
    if item["algorithm"] == "ecdsa-p256-sha256":
        return "ecdsa-p256-sha256"
    raise ValueError("non-canonical roster key algorithm")


def _decode_roster_bytes(encoded: bytes, description: str) -> VerificationRoster:
    try:
        body = json.loads(encoded.decode("utf-8"))
        if not isinstance(body, dict) or _canonical(body) != encoded:
            raise ValueError
        keys = tuple(
            PublicKeyRecord(
                device_id=str(item["device_id"]),
                key_id=str(item["key_id"]),
                fingerprint=str(item["fingerprint"]),
                public_key=str(item["public_key"]),
                roles=tuple(str(role) for role in item["roles"]),
                enrollment_sequence=int(item["enrollment_sequence"]),
                algorithm=_decode_key_algorithm(item),
                revocation_sequence=(
                    int(item["revocation_sequence"])
                    if item.get("revocation_sequence") is not None
                    else None
                ),
                proof_of_possession=str(item.get("proof_of_possession") or ""),
            )
            for item in body["keys"]
        )
        signature_raw = body["signature"]
        signature = SignatureEnvelope(
            algorithm=str(signature_raw["algorithm"]),  # type: ignore[arg-type]
            key_id=str(signature_raw["key_id"]),
            roster_version=int(signature_raw["roster_version"]),
            signature=str(signature_raw["signature"]),
        )
        return VerificationRoster(
            version=int(body["version"]),
            peers=tuple(str(peer) for peer in body["peers"]),
            keys=keys,
            local_device_id=str(body["local_device_id"]),
            schema=str(body["schema"]),  # type: ignore[arg-type]
            created_at=str(body.get("created_at") or ""),
            previous_roster_hash=str(body.get("previous_roster_hash") or ""),
            roster_hash=str(body.get("roster_hash") or ""),
            signature=signature,
        )
    except (KeyError, TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RosterError(f"invalid verification roster: {description}") from exc


def _verify_pop(key: PublicKeyRecord) -> None:
    if not key.proof_of_possession:
        raise RosterError(f"missing proof of possession: {key.key_id}")
    try:
        public_bytes = _decode(key.public_key)
        signature = _decode(key.proof_of_possession)
        if key.algorithm == "ed25519":
            ed_public_key = Ed25519PublicKey.from_public_bytes(public_bytes)
            ed_public_key.verify(signature, key.proof_payload())
        else:
            if not _is_canonical_p256_signature(signature):
                raise ValueError
            p256_public_key = ec.EllipticCurvePublicKey.from_encoded_point(
                ec.SECP256R1(),
                public_bytes,
            )
            p256_public_key.verify(
                signature,
                key.proof_payload(),
                ec.ECDSA(hashes.SHA256()),
            )
    except (InvalidSignature, ValueError) as exc:
        raise RosterError(f"invalid proof of possession: {key.key_id}") from exc


def _verify_roster(
    roster: VerificationRoster,
    *,
    previous: VerificationRoster | None,
) -> None:
    if roster.roster_hash != roster._hash():
        raise RosterError("verification roster hash mismatch")
    for key in roster.keys:
        _verify_pop(key)
    if roster.signature is None:
        raise RosterError("verification roster signature is missing")
    if previous is None:
        if (
            roster.version != 1
            or roster.previous_roster_hash
            or len(roster.peers) != 1
            or len(roster.keys) != 1
        ):
            raise RosterError("invalid bootstrap roster shape")
        key = roster.key(roster.signature.key_id)
        verifier_roster = roster
        domain = _BOOTSTRAP_DOMAIN
    else:
        if (
            roster.version != previous.version + 1
            or roster.previous_roster_hash != previous.roster_hash
        ):
            raise RosterError("verification roster history is discontinuous")
        key = previous.key(roster.signature.key_id)
        verifier_roster = previous
        domain = _UPDATE_DOMAIN
    if "origin" not in key.roles:
        raise RosterError("roster signer is not an origin key")
    try:
        OperationalVerifier().verify(
            domain=domain,
            payload=roster._signed_payload(),
            envelope=roster.signature,
            roster=verifier_roster,
        )
    except SignatureError as exc:
        raise RosterError("verification roster signature is invalid") from exc


def verify_bootstrap(roster: VerificationRoster, key: PublicKeyRecord) -> bool:
    try:
        if roster.keys != (key,) or roster.peers != (key.device_id,):
            return False
        _verify_roster(roster, previous=None)
    except RosterError:
        return False
    return True


__all__ = ["RosterError", "VerificationRoster", "verify_bootstrap"]
