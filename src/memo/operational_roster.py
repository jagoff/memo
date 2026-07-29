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
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from memo.atomic_io import atomic_write_text, authority_write_lock
from memo.errors import SignatureError
from memo.operational_key_store import (
    AuthorityPinStore,
    KeyStoreError,
    PublicKeyRecord,
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
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_authority_write(path: Path, data: bytes) -> None:
    atomic_write_text(path, data.decode("utf-8"))
    _fsync_directory(path.parent)


def _create_authority_file(path: Path, data: bytes) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    _fsync_directory(path.parent)


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
            if (
                len(public_bytes) != 32
                or hashlib.sha256(public_bytes).hexdigest() != key.fingerprint
            ):
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
        pin_store: AuthorityPinStore,
    ) -> VerificationRoster:
        root = Path(root)
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
                algorithm="ed25519",
                key_id=key.key_id,
                roster_version=1,
                signature=key._bootstrap_signature,
            ),
        )
        _verify_roster(roster, previous=None)
        encoded = _canonical(roster.to_dict())
        history = root / "verification-rosters" / "00000001.json"
        current = root / "verification-roster.json"
        try:
            pin_store.prepare_roster(version=1, roster_hash=roster.roster_hash)
        except KeyStoreError as exc:
            raise RosterError("verification roster authority pin rejected bootstrap") from exc
        with authority_write_lock(root / "verification-rosters"):
            history_bytes = history.read_bytes() if history.exists() else None
            current_bytes = current.read_bytes() if current.exists() else None
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
            pin_store.commit_roster(version=1, roster_hash=roster.roster_hash)
        except KeyStoreError as exc:
            raise RosterError("verification roster authority pin commit failed") from exc
        return roster

    @classmethod
    def load(cls, root: Path, *, pin_store: AuthorityPinStore) -> VerificationRoster:
        try:
            pin = pin_store.read()
        except KeyStoreError as exc:
            raise RosterError("verification roster authority pin is unavailable") from exc
        root = Path(root)
        with authority_write_lock(root / "verification-rosters"):
            previous = _load_roster_files(
                root,
                pending_version=pin.pending_roster_version,
                pending_hash=pin.pending_roster_hash,
            )
        try:
            pin_store.verify_roster(
                version=previous.version,
                roster_hash=previous.roster_hash,
            )
        except KeyStoreError as exc:
            raise RosterError("verification roster rollback or substitution detected") from exc
        return previous

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
        if pin_store is None:
            raise RosterError("signed roster update requires an authority pin store")
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
        root = Path(root)
        encoded = _canonical(updated.to_dict())
        history = root / "verification-rosters" / f"{version:08d}.json"
        current = root / "verification-roster.json"
        with authority_write_lock(root / "verification-rosters"):
            loaded = _load_roster_files(root)
            if loaded.roster_hash != self.roster_hash:
                raise RosterError("verification roster changed concurrently")
            try:
                pin_store.prepare_roster(version=version, roster_hash=updated.roster_hash)
            except KeyStoreError as exc:
                raise RosterError("verification roster authority pin rejected update") from exc
            if history.exists():
                if history.read_bytes() != encoded:
                    raise RosterError("verification roster history already exists")
            else:
                _create_authority_file(history, encoded)
            _atomic_authority_write(current, encoded)
        try:
            pin_store.commit_roster(version=version, roster_hash=updated.roster_hash)
        except KeyStoreError as exc:
            raise RosterError("verification roster authority pin commit failed") from exc
        return updated


def _load_roster_files(
    root: Path,
    *,
    pending_version: int | None = None,
    pending_hash: str = "",
) -> VerificationRoster:
    root = Path(root)
    history_root = root / "verification-rosters"
    current_path = root / "verification-roster.json"
    paths = sorted(history_root.glob("*.json"))
    if not paths:
        raise RosterError("verification roster history is missing")
    previous: VerificationRoster | None = None
    before_latest: VerificationRoster | None = None
    for expected, path in enumerate(paths, start=1):
        if path.name != f"{expected:08d}.json":
            raise RosterError("verification roster history has a version gap")
        roster = _decode_roster(path)
        _verify_roster(roster, previous=previous)
        before_latest = previous
        previous = roster
    assert previous is not None
    try:
        current_bytes: bytes | None = current_path.read_bytes()
    except FileNotFoundError:
        current_bytes = None
    except OSError as exc:
        raise RosterError("verification roster current pointer is invalid") from exc
    latest_bytes = _canonical(previous.to_dict())
    if current_bytes != latest_bytes and (
        pending_version == previous.version
        and pending_hash == previous.roster_hash
        and (
            current_bytes is None
            or (before_latest is not None and current_bytes == _canonical(before_latest.to_dict()))
        )
    ):
        _atomic_authority_write(current_path, latest_bytes)
        current_bytes = latest_bytes
    if current_bytes != latest_bytes:
        raise RosterError("verification roster current pointer mismatch")
    return previous


def _decode_roster(path: Path) -> VerificationRoster:
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
        keys = tuple(
            PublicKeyRecord(
                device_id=str(item["device_id"]),
                key_id=str(item["key_id"]),
                fingerprint=str(item["fingerprint"]),
                public_key=str(item["public_key"]),
                roles=tuple(str(role) for role in item["roles"]),
                enrollment_sequence=int(item["enrollment_sequence"]),
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
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RosterError(f"invalid verification roster: {path}") from exc


def _verify_pop(key: PublicKeyRecord) -> None:
    if not key.proof_of_possession:
        raise RosterError(f"missing proof of possession: {key.key_id}")
    try:
        public_key = Ed25519PublicKey.from_public_bytes(_decode(key.public_key))
        public_key.verify(_decode(key.proof_of_possession), key.proof_payload())
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
