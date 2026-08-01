"""Native macOS Secure Enclave operations for operational P-256 keys."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

from memo.atomic_io import SecureDirectory, authority_write_lock, open_secure_directory
from memo.operational_key_store import (
    KeyStoreError,
    _normalize_p256_signature,
)

_SERVICE_RE = re.compile(
    r"com\.memo\.operational-signing"
    r"(?:\.[A-Za-z0-9][A-Za-z0-9-]{0,62}){0,4}\Z"
)
_KEY_ID_RE = re.compile(r"p256-se-[0-9a-f]{32}\Z")
_OPERATIONS = frozenset({"generate", "sign", "destroy"})
_SYSTEM_CODESIGN = Path("/usr/bin/codesign")
_MAX_HELPER_BINARY_BYTES = 16 * 1024 * 1024
_MAX_BINDING_BYTES = 4096
_SERVICE_NAMESPACE = "com.memo.operational-signing.v2"
_BINDING_SCHEMA = "memo.secure_enclave_helper_binding.v1"
_BINDING_PREFIX = b"memo.secure-enclave-binding.v1\0"
_BINDING_STATES = frozenset({"generating", "active", "destroying"})


def binding_digest(service: str, key_id: str) -> str:
    """Return the immutable descriptor used for a key/helper binding."""
    SecureEnclaveP256Backend._validate_service(service)
    SecureEnclaveP256Backend._validate_key_id(key_id)
    return hashlib.sha256(_BINDING_PREFIX + service.encode() + b"\0" + key_id.encode()).hexdigest()


def canonical_binding(*, service: str, key_id: str, helper_sha256: str, state: str) -> bytes:
    SecureEnclaveP256Backend._validate_service(service)
    SecureEnclaveP256Backend._validate_key_id(key_id)
    if not re.fullmatch(r"[0-9a-f]{64}", helper_sha256) or state not in _BINDING_STATES:
        raise KeyStoreError("Secure Enclave binding is invalid")
    return json.dumps(
        {
            "helper_sha256": helper_sha256,
            "key_id": key_id,
            "schema": _BINDING_SCHEMA,
            "service": service,
            "state": state,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()


def validate_binding(
    raw: bytes, *, service: str, key_id: str, expected_name: str | None = None
) -> dict[str, str]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise KeyStoreError("Secure Enclave binding is not canonical JSON") from None
    if (
        not isinstance(value, dict)
        or set(value) != {"helper_sha256", "key_id", "schema", "service", "state"}
        or not all(isinstance(value[k], str) for k in value)
    ):
        raise KeyStoreError("Secure Enclave binding fields are invalid")
    encoded = canonical_binding(
        service=service, key_id=key_id, helper_sha256=value["helper_sha256"], state=value["state"]
    )
    if (
        raw != encoded
        or value["schema"] != _BINDING_SCHEMA
        or (expected_name and expected_name != binding_digest(service, key_id) + ".json")
    ):
        raise KeyStoreError("Secure Enclave binding is not canonical")
    return value


class SecureEnclaveP256Backend:
    """Opaque CryptoKit helper whose wrapped key handle lives in Keychain."""

    algorithm: Literal["ecdsa-p256-sha256"] = "ecdsa-p256-sha256"

    def __init__(
        self,
        *,
        service: str,
    ) -> None:
        if sys.platform != "darwin":
            raise KeyStoreError("macOS Secure Enclave signing is unavailable")
        self._validate_service(service)
        if not (service == _SERVICE_NAMESPACE or service.startswith(_SERVICE_NAMESPACE + ".")):
            raise KeyStoreError("Secure Enclave Keychain service must use operational v2 namespace")
        self._service: str = service
        self._helper: Path
        self._helper_sha256: str
        self._helper, self._helper_sha256 = self._install_helper()
        self._bindings = self._native_tools_root() / "key-bindings-v1"
        self._verify_helper()

    @staticmethod
    def _validate_service(service: str) -> None:
        if not _SERVICE_RE.fullmatch(service):
            raise KeyStoreError("Secure Enclave Keychain service is unsafe")

    @staticmethod
    def _validate_key_id(key_id: str) -> None:
        if not _KEY_ID_RE.fullmatch(key_id):
            raise KeyStoreError("Secure Enclave key id is unsafe")

    @staticmethod
    def _safe_tool_environment() -> dict[str, str]:
        environment = {
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
            "LANG": "C",
            "LC_ALL": "C",
        }
        temporary_root = os.getenv("TMPDIR")
        if temporary_root:
            environment["TMPDIR"] = temporary_root
        return environment

    @staticmethod
    def _read_regular_snapshot(
        path: Path,
        *,
        description: str,
        maximum_bytes: int,
        allowed_owners: frozenset[int],
        required_mode: int | None = None,
        require_single_link: bool = True,
    ) -> bytes:
        descriptor = -1
        try:
            descriptor = os.open(
                path,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            )
            observed = os.fstat(descriptor)
            mode = stat.S_IMODE(observed.st_mode)
            if (
                not stat.S_ISREG(observed.st_mode)
                or observed.st_uid not in allowed_owners
                or (require_single_link and observed.st_nlink != 1)
                or (required_mode is not None and mode != required_mode)
                or (required_mode is None and mode & 0o022)
                or observed.st_size > maximum_bytes
            ):
                raise KeyStoreError(f"{description} has unsafe ownership or mode")
            chunks: list[bytes] = []
            remaining = maximum_bytes + 1
            while remaining:
                chunk = os.read(descriptor, min(1024 * 1024, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            encoded = b"".join(chunks)
            final = os.fstat(descriptor)
            try:
                current = os.lstat(path)
            except OSError:
                raise KeyStoreError(f"{description} changed while being read") from None
            if (
                len(encoded) > maximum_bytes
                or (observed.st_dev, observed.st_ino) != (final.st_dev, final.st_ino)
                or (observed.st_dev, observed.st_ino) != (current.st_dev, current.st_ino)
                or observed.st_size != len(encoded)
                or observed.st_mtime_ns != final.st_mtime_ns
                or observed.st_ctime_ns != final.st_ctime_ns
            ):
                raise KeyStoreError(f"{description} changed while being read")
            return encoded
        except KeyStoreError:
            raise
        except OSError as exc:
            raise KeyStoreError(f"{description} is unavailable") from exc
        finally:
            if descriptor >= 0:
                os.close(descriptor)

    @staticmethod
    def _verify_system_tool(path: Path, description: str) -> None:
        try:
            observed = os.lstat(path)
        except OSError as exc:
            raise KeyStoreError(f"{description} is unavailable") from exc
        if (
            not stat.S_ISREG(observed.st_mode)
            or stat.S_ISLNK(observed.st_mode)
            or observed.st_uid != 0
            or stat.S_IMODE(observed.st_mode) & 0o022
            or not observed.st_mode & stat.S_IXUSR
        ):
            raise KeyStoreError(f"{description} is unsafe")

    @classmethod
    def _verify_code_signature(cls, helper: Path) -> None:
        cls._verify_system_tool(_SYSTEM_CODESIGN, "macOS code-signing verifier")
        try:
            result = subprocess.run(
                [
                    str(_SYSTEM_CODESIGN),
                    "--verify",
                    "--strict",
                    "--verbose=0",
                    str(helper),
                ],
                check=False,
                capture_output=True,
                timeout=10,
                env=cls._safe_tool_environment(),
            )
        except (OSError, subprocess.SubprocessError):
            raise KeyStoreError("Memo Secure Enclave helper code signature is invalid") from None
        if result.returncode != 0:
            raise KeyStoreError("Memo Secure Enclave helper code signature is invalid")

    @staticmethod
    def _native_tools_root() -> Path:
        return Path.home() / "Library" / "Application Support" / "Memo" / "native-tools"

    @staticmethod
    def _assert_private_directory(descriptor: int, description: str) -> None:
        observed = os.fstat(descriptor)
        if observed.st_uid != os.getuid() or stat.S_IMODE(observed.st_mode) & 0o077:
            raise KeyStoreError(f"{description} has unsafe ownership or mode")

    @classmethod
    def _install_helper(cls) -> tuple[Path, str]:
        # Production never compiles Swift at runtime.  The platform wheel must
        # carry this immutable, ad-hoc-signed arm64 helper asset.
        packaged = Path(__file__).parent / "native" / "darwin-arm64" / "memo-secure-enclave-helper"
        if not packaged.is_file():
            raise KeyStoreError("packaged Secure Enclave helper is unavailable")
        candidate = cls._read_regular_snapshot(
            packaged,
            description="packaged Memo Secure Enclave helper",
            maximum_bytes=_MAX_HELPER_BINARY_BYTES,
            allowed_owners=frozenset({0, os.getuid()}),
            required_mode=0o500,
        )
        helper_sha256 = hashlib.sha256(candidate).hexdigest()
        root = cls._native_tools_root()
        helpers = root / "helpers-v1"
        target = helpers / helper_sha256
        with authority_write_lock(root), open_secure_directory(helpers, create=True) as directory:
            cls._assert_private_directory(directory.descriptor, "Memo native helper directory")
            if directory.exists(target.name):
                existing, observed = directory.read_bytes_snapshot(target.name)
                if (
                    existing != candidate
                    or observed.st_uid != os.getuid()
                    or observed.st_nlink != 1
                    or stat.S_IMODE(observed.st_mode) != 0o500
                ):
                    raise KeyStoreError(
                        "cached Secure Enclave helper failed content-address verification"
                    )
            else:
                directory.create_bytes_exclusive(target.name, candidate, mode=0o500)
        cls._verify_code_signature(target)
        return target, helper_sha256

    @classmethod
    def _read_helper_snapshot(cls, helper: Path) -> bytes:
        return cls._read_regular_snapshot(
            helper,
            description="Memo Secure Enclave helper",
            maximum_bytes=_MAX_HELPER_BINARY_BYTES,
            allowed_owners=frozenset({os.getuid()}),
            required_mode=0o500,
        )

    def _verify_helper(self) -> None:
        helper_bytes = self._read_helper_snapshot(self._helper)
        if hashlib.sha256(helper_bytes).hexdigest() != self._helper_sha256:
            raise KeyStoreError("Memo Secure Enclave helper failed content-address verification")
        self._verify_code_signature(self._helper)

    def _run_helper(self, operation: str, key_id: str, payload: bytes = b"") -> bytes:
        if operation not in _OPERATIONS:
            raise KeyStoreError("Secure Enclave helper operation is unsafe")
        self._validate_key_id(key_id)
        self._verify_helper()
        try:
            result = subprocess.run(
                [
                    str(self._helper),
                    operation,
                    self._service,
                    key_id,
                ],
                input=bytes(payload),
                check=False,
                capture_output=True,
                timeout=30,
                env=self._safe_tool_environment(),
            )
        except (OSError, subprocess.SubprocessError):
            raise KeyStoreError("Secure Enclave helper execution failed") from None
        if result.returncode != 0:
            message = result.stderr.decode("utf-8", errors="replace").strip()
            allowed = {
                "duplicate private key id",
                "unknown private key id",
                "Secure Enclave is unavailable",
                "Secure Enclave key generation failed",
                "Secure Enclave key recovery failed",
                "Secure Enclave signing failed",
                "Keychain write failed",
                "Keychain read failed",
                "Keychain delete failed",
                "invalid Keychain service",
                "invalid private key id",
            }
            if message not in allowed:
                message = "Secure Enclave helper operation failed"
            raise KeyStoreError(f"{message}: {key_id}")
        return bytes(result.stdout)

    def _binding_name(self, key_id: str) -> str:
        return binding_digest(self._service, key_id) + ".json"

    @contextlib.contextmanager
    def _binding_authority(self, key_id: str) -> Iterator[tuple[SecureDirectory, str]]:
        name = self._binding_name(key_id)
        lock_path = self._bindings / name
        try:
            with (
                authority_write_lock(lock_path),
                open_secure_directory(self._bindings, create=True) as directory,
            ):
                self._assert_private_directory(
                    directory.descriptor,
                    "Secure Enclave binding directory",
                )
                yield directory, name
        except KeyStoreError:
            raise
        except (OSError, ValueError):
            raise KeyStoreError("Secure Enclave binding store is unsafe") from None

    def _read_binding(
        self,
        directory: SecureDirectory,
        name: str,
        key_id: str,
        *,
        allowed_states: frozenset[str],
    ) -> dict[str, str]:
        try:
            raw, observed = directory.read_bytes_snapshot(name)
        except (OSError, ValueError):
            raise KeyStoreError("Secure Enclave key binding is unavailable or unsafe") from None
        if (
            observed.st_uid != os.getuid()
            or observed.st_nlink != 1
            or stat.S_IMODE(observed.st_mode) != 0o600
            or len(raw) > _MAX_BINDING_BYTES
        ):
            raise KeyStoreError("Secure Enclave key binding is unavailable or unsafe")
        value = validate_binding(
            raw,
            service=self._service,
            key_id=key_id,
            expected_name=name,
        )
        if value["helper_sha256"] != self._helper_sha256:
            raise KeyStoreError("Secure Enclave key is bound to a different helper")
        if value["state"] not in allowed_states:
            raise KeyStoreError(f"Secure Enclave key binding is {value['state']}; failing closed")
        return value

    def _write_binding(
        self,
        directory: SecureDirectory,
        name: str,
        key_id: str,
        state: str,
        *,
        exclusive: bool = False,
    ) -> None:
        encoded = canonical_binding(
            service=self._service,
            key_id=key_id,
            helper_sha256=self._helper_sha256,
            state=state,
        )
        try:
            if exclusive:
                directory.create_bytes_exclusive(name, encoded, mode=0o600)
            else:
                directory.atomic_write_bytes(name, encoded, mode=0o600)
        except FileExistsError:
            raise KeyStoreError(f"Secure Enclave key binding already exists: {key_id}") from None
        except (OSError, ValueError):
            raise KeyStoreError("Secure Enclave key binding update failed") from None

    @staticmethod
    def _unlink_binding(directory: SecureDirectory, name: str) -> None:
        try:
            directory.unlink(name)
        except (OSError, ValueError):
            raise KeyStoreError("Secure Enclave key binding cleanup failed") from None

    def _destroy_locked(self, directory: SecureDirectory, name: str, key_id: str) -> None:
        self._write_binding(directory, name, key_id, "destroying")
        try:
            output = self._run_helper("destroy", key_id)
        except KeyStoreError as exc:
            if not str(exc).startswith("unknown private key id:"):
                raise
        else:
            if output:
                raise KeyStoreError("Secure Enclave helper returned unexpected output")
        self._unlink_binding(directory, name)

    def generate(self, key_id: str) -> bytes:
        self._validate_key_id(key_id)
        with self._binding_authority(key_id) as (directory, name):
            if directory.exists(name):
                self._read_binding(
                    directory,
                    name,
                    key_id,
                    allowed_states=_BINDING_STATES,
                )
                raise KeyStoreError(f"Secure Enclave key binding already exists: {key_id}")
            self._write_binding(
                directory,
                name,
                key_id,
                "generating",
                exclusive=True,
            )
            public_key = self._run_helper("generate", key_id)
            if len(public_key) != 65 or public_key[:1] != b"\x04":
                with contextlib.suppress(KeyStoreError):
                    self._destroy_locked(directory, name, key_id)
                raise KeyStoreError("Secure Enclave returned an invalid P-256 public key")
            try:
                self._write_binding(directory, name, key_id, "active")
            except KeyStoreError:
                with contextlib.suppress(KeyStoreError):
                    self._destroy_locked(directory, name, key_id)
                raise
            return public_key

    def sign(self, key_id: str, payload: bytes) -> bytes:
        self._validate_key_id(key_id)
        with self._binding_authority(key_id) as (directory, name):
            self._read_binding(
                directory,
                name,
                key_id,
                allowed_states=frozenset({"active"}),
            )
            signature = self._run_helper("sign", key_id, bytes(payload))
            return _normalize_p256_signature(signature)

    def destroy(self, key_id: str) -> None:
        self._validate_key_id(key_id)
        with self._binding_authority(key_id) as (directory, name):
            self._read_binding(
                directory,
                name,
                key_id,
                allowed_states=_BINDING_STATES,
            )
            self._destroy_locked(directory, name, key_id)
