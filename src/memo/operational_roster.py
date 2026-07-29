"""Immutable public-key rosters for operational verification."""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from memo.atomic_io import atomic_write_text, authority_write_lock
from memo.errors import SignatureError
from memo.operational_key_store import PublicKeyRecord


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


class RosterError(SignatureError):
    """A verification roster is structurally invalid or regressed."""


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
            try:
                public_bytes = _decode(key.public_key)
            except (ValueError, TypeError) as exc:
                raise RosterError(f"invalid public key encoding: {key.key_id}") from exc
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
            and (
                key.revocation_sequence is None
                or key.revocation_sequence > self.version
            )
        ]
        if len(keys) != 1:
            raise RosterError("local roster must have exactly one active key")
        return keys[0]

    def key(self, key_id: str) -> PublicKeyRecord:
        matches = [key for key in self.keys if key.key_id == key_id]
        if len(matches) != 1:
            raise RosterError(f"unknown roster key: {key_id}")
        return matches[0]

    def _hash(self) -> str:
        body = self.to_dict()
        body["roster_hash"] = ""
        return hashlib.sha256(_canonical(body)).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        body = asdict(self)
        body["peers"] = list(self.peers)
        body["keys"] = [asdict(key) for key in self.keys]
        return body

    @classmethod
    def bootstrap(
        cls,
        *,
        device_id: str,
        key: PublicKeyRecord,
        root: Path,
    ) -> VerificationRoster:
        if key.device_id != device_id:
            raise RosterError("bootstrap key device mismatch")
        allowed_roles = {"origin", "migration_attestor"}
        if not key.roles or not set(key.roles).issubset(allowed_roles):
            raise RosterError("bootstrap key has unsupported roles")
        stamp = datetime.now(UTC).isoformat(timespec="milliseconds")
        roster = cls(
            version=1,
            peers=(device_id,),
            keys=(key,),
            local_device_id=device_id,
            created_at=stamp,
        )
        roster = replace(roster, roster_hash=roster._hash())
        if not verify_bootstrap(roster, key):
            raise RosterError("bootstrap key proof is invalid")
        path = Path(root) / "verification-roster.json"
        with authority_write_lock(path):
            if path.exists():
                raise RosterError("verification roster already exists")
            atomic_write_text(
                path,
                json.dumps(
                    roster.to_dict(),
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                ),
            )
        return roster

    @classmethod
    def load(cls, root: Path) -> VerificationRoster:
        path = Path(root) / "verification-roster.json"
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
            roster = cls(
                version=int(body["version"]),
                peers=tuple(str(peer) for peer in body["peers"]),
                keys=keys,
                local_device_id=str(body["local_device_id"]),
                schema=str(body["schema"]),  # type: ignore[arg-type]
                created_at=str(body.get("created_at") or ""),
                previous_roster_hash=str(body.get("previous_roster_hash") or ""),
                roster_hash=str(body.get("roster_hash") or ""),
            )
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RosterError(f"invalid verification roster: {path}") from exc
        if roster.roster_hash != roster._hash():
            raise RosterError("verification roster hash mismatch")
        return roster

    def with_keys(
        self,
        *,
        version: int,
        peers: tuple[str, ...],
        keys: tuple[PublicKeyRecord, ...],
    ) -> VerificationRoster:
        if version <= self.version:
            raise RosterError("roster version regression")
        updated = VerificationRoster(
            version=version,
            peers=peers,
            keys=keys,
            local_device_id=self.local_device_id,
            created_at=datetime.now(UTC).isoformat(timespec="milliseconds"),
            previous_roster_hash=self.roster_hash,
        )
        return replace(updated, roster_hash=updated._hash())


def verify_bootstrap(roster: VerificationRoster, key: PublicKeyRecord) -> bool:
    if (
        roster.schema != "memo.operational_roster.v1"
        or roster.version != 1
        or roster.peers != (key.device_id,)
        or roster.keys != (key,)
        or roster.local_device_id != key.device_id
        or roster.roster_hash != roster._hash()
        or not key.proof_of_possession
    ):
        return False
    try:
        public_key = Ed25519PublicKey.from_public_bytes(_decode(key.public_key))
        public_key.verify(_decode(key.proof_of_possession), key.proof_payload())
    except (InvalidSignature, ValueError):
        return False
    return True


__all__ = ["RosterError", "VerificationRoster", "verify_bootstrap"]
