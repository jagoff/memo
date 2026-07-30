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
import tempfile
from pathlib import Path
from typing import Literal

from memo.atomic_io import authority_write_lock, open_secure_directory
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
_SYSTEM_SWIFTC = Path("/usr/bin/swiftc")
_SYSTEM_CODESIGN = Path("/usr/bin/codesign")
_MAX_HELPER_SOURCE_BYTES = 256 * 1024
_MAX_HELPER_BINARY_BYTES = 16 * 1024 * 1024
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
    return json.dumps({"helper_sha256": helper_sha256, "key_id": key_id,
                       "schema": _BINDING_SCHEMA, "service": service, "state": state},
                      sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def validate_binding(raw: bytes, *, service: str, key_id: str, expected_name: str | None = None) -> dict[str, str]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise KeyStoreError("Secure Enclave binding is not canonical JSON") from None
    if (not isinstance(value, dict) or set(value) != {"helper_sha256", "key_id", "schema", "service", "state"}
            or not all(isinstance(value[k], str) for k in value)):
        raise KeyStoreError("Secure Enclave binding fields are invalid")
    encoded = canonical_binding(service=service, key_id=key_id,
                                helper_sha256=value["helper_sha256"], state=value["state"])
    if raw != encoded or value["schema"] != _BINDING_SCHEMA or (expected_name and expected_name != binding_digest(service, key_id) + ".json"):
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
        self._service = service
        self._helper, self._helper_sha256 = self._install_helper()
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
    def _source() -> Path:
        source = Path(__file__).parent / "native" / "memo_secure_enclave_helper.swift"
        return source

    @classmethod
    def _source_snapshot(cls) -> tuple[Path, bytes]:
        source = cls._source()
        encoded = cls._read_regular_snapshot(
            source,
            description="Memo Secure Enclave helper source",
            maximum_bytes=_MAX_HELPER_SOURCE_BYTES,
            allowed_owners=frozenset({0, os.getuid()}),
            require_single_link=False,
        )
        return source, encoded

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

    @classmethod
    def _install_helper(cls) -> tuple[Path, str]:
        # Production never compiles Swift at runtime.  The platform wheel must
        # carry this immutable, ad-hoc-signed arm64 helper asset.
        packaged = Path(__file__).parent / "native" / "darwin-arm64" / "memo-secure-enclave-helper"
        if not packaged.is_file():
            raise KeyStoreError("packaged Secure Enclave helper is unavailable")
        candidate = cls._read_regular_snapshot(
            packaged, description="packaged Memo Secure Enclave helper",
            maximum_bytes=_MAX_HELPER_BINARY_BYTES,
            allowed_owners=frozenset({0, os.getuid()}), required_mode=0o500,
        )
        helper_sha256 = hashlib.sha256(candidate).hexdigest()
        root = Path.home() / "Library" / "Application Support" / "Memo" / "native-tools"
        helpers = root / "helpers-v1"
        target = helpers / helper_sha256
        with authority_write_lock(root), open_secure_directory(helpers, create=True) as directory:
            if directory.exists(target.name):
                existing, observed = directory.read_bytes_snapshot(target.name)
                if existing != candidate or observed.st_uid != os.getuid() or observed.st_nlink != 1 or stat.S_IMODE(observed.st_mode) != 0o500:
                    raise KeyStoreError("cached Secure Enclave helper failed content-address verification")
            else:
                directory.create_bytes_exclusive(target.name, candidate, mode=0o500)
        cls._verify_code_signature(target)
        return target, helper_sha256

        # Unreachable legacy source compiler retained below for provenance only.
        _, source_bytes = cls._source_snapshot()
        source_sha256 = hashlib.sha256(source_bytes).hexdigest()
        root = Path.home() / "Library" / "Application Support" / "Memo" / "native-tools"
        cls._verify_system_tool(_SYSTEM_SWIFTC, "system Swift compiler")
        with authority_write_lock(root):
            with open_secure_directory(root, create=True) as directory:
                root_state = os.fstat(directory.descriptor)
                if root_state.st_uid != os.getuid() or stat.S_IMODE(root_state.st_mode) & 0o077:
                    raise KeyStoreError("Memo native helper directory has unsafe ownership or mode")
            try:
                with tempfile.TemporaryDirectory(
                    prefix=".secure-enclave-build-",
                    dir=root,
                ) as temporary_name:
                    source_copy = (
                        Path(temporary_name)
                        / "memo_secure_enclave_helper.swift"
                    )
                    source_descriptor = os.open(
                        source_copy,
                        os.O_CREAT
                        | os.O_EXCL
                        | os.O_WRONLY
                        | getattr(os, "O_NOFOLLOW", 0),
                        0o400,
                    )
                    try:
                        view = memoryview(source_bytes)
                        written = 0
                        while written < len(view):
                            count = os.write(
                                source_descriptor,
                                view[written:],
                            )
                            if count <= 0:
                                raise OSError("short helper source write")
                            written += count
                        os.fchmod(source_descriptor, 0o400)
                        os.fsync(source_descriptor)
                    finally:
                        os.close(source_descriptor)
                    temporary = Path(temporary_name) / "memo-secure-enclave-helper"
                    result = subprocess.run(
                        [
                            str(_SYSTEM_SWIFTC),
                            "-O",
                            "-whole-module-optimization",
                            "-module-name",
                            "MemoSecureEnclaveHelper",
                            "-o",
                            str(temporary),
                            str(source_copy),
                        ],
                        check=False,
                        capture_output=True,
                        timeout=120,
                        env=cls._safe_tool_environment(),
                    )
                    if result.returncode != 0:
                        raise KeyStoreError("Secure Enclave helper compilation failed")
                    temporary.chmod(0o500)
                    with open(temporary, "rb") as handle:
                        os.fsync(handle.fileno())
                    cls._verify_code_signature(temporary)
                    candidate = cls._read_regular_snapshot(
                        temporary,
                        description="compiled Memo Secure Enclave helper",
                        maximum_bytes=_MAX_HELPER_BINARY_BYTES,
                        allowed_owners=frozenset({os.getuid()}),
                        required_mode=0o500,
                    )
                    helper_sha256 = hashlib.sha256(candidate).hexdigest()
                    target = root / (f"secure-enclave-{source_sha256}-{helper_sha256}")
                    try:
                        with open_secure_directory(root) as directory:
                            if directory.exists(target.name):
                                existing, observed = directory.read_bytes_snapshot(target.name)
                                if (
                                    observed.st_uid != os.getuid()
                                    or observed.st_nlink != 1
                                    or stat.S_IMODE(observed.st_mode) != 0o500
                                    or existing != candidate
                                ):
                                    raise KeyStoreError(
                                        "cached Secure Enclave helper failed "
                                        "content-address verification"
                                    )
                            else:
                                directory.create_bytes_exclusive(
                                    target.name,
                                    candidate,
                                    mode=0o500,
                                )
                    except KeyStoreError:
                        raise
                    except (OSError, ValueError):
                        raise KeyStoreError("cached Secure Enclave helper is unsafe") from None
                    cls._verify_code_signature(target)
                    return target, helper_sha256
            except KeyStoreError:
                raise
            except (OSError, subprocess.SubprocessError):
                raise KeyStoreError("Secure Enclave helper compilation failed") from None

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

    def _run(self, operation: str, key_id: str, payload: bytes = b"") -> bytes:
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

    def generate(self, key_id: str) -> bytes:
        public_key = self._run("generate", key_id)
        if len(public_key) != 65 or public_key[:1] != b"\x04":
            with contextlib.suppress(KeyStoreError):
                self._run("destroy", key_id)
            raise KeyStoreError("Secure Enclave returned an invalid P-256 public key")
        return public_key

    def sign(self, key_id: str, payload: bytes) -> bytes:
        signature = self._run("sign", key_id, bytes(payload))
        return _normalize_p256_signature(signature)

    def destroy(self, key_id: str) -> None:
        output = self._run("destroy", key_id)
        if output:
            raise KeyStoreError("Secure Enclave helper returned unexpected output")
