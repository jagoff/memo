"""Immutable Git-backed transport for signed operational ledger artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from memo.atomic_io import SecureDirectory, authority_write_lock, open_secure_directory
from memo.errors import OperationalError, OperationalErrorCode
from memo.operational_event import OriginBundle, canonical_json_bytes

_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_GIT_OID = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
_EPOCH = re.compile(r"(?:0|[1-9][0-9]*)\Z")
_SEQUENCE_FILE = re.compile(r"[0-9]{20}\.jsonl\Z")
_HEAD_SCHEMA = "memo.operational_transport_head.v1"
_MARKER_SCHEMA = "memo.operational_git_transport.v1"
_MARKER_PATH = Path(".memo-operational-transport")
_REMOTE_BRANCH = "memo-operational"
_MAX_INT64 = (1 << 63) - 1


def _failure(message: str) -> OperationalError:
    return OperationalError(
        OperationalErrorCode.STORAGE_UNAVAILABLE,
        message,
        retryable=True,
    )


def _safe(value: str, description: str) -> str:
    if not isinstance(value, str) or not _SAFE_ID.fullmatch(value):
        raise OperationalError(
            OperationalErrorCode.INVALID_EVENT,
            f"unsafe operational transport {description}: {value!r}",
            retryable=False,
        )
    return value


def _invalid(message: str) -> OperationalError:
    return OperationalError(
        OperationalErrorCode.INVALID_EVENT,
        message,
        retryable=False,
    )


def _integer(value: object, description: str, *, minimum: int = 0) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
        or value > _MAX_INT64
    ):
        raise _invalid(
            f"operational transport {description} must be an integer "
            f"between {minimum} and {_MAX_INT64}"
        )
    return value


def _sha256(value: object, description: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (
        not _SHA256.fullmatch(value) and not (allow_empty and value == "")
    ):
        raise _invalid(f"operational transport {description} must be a SHA-256 digest")
    return value


@dataclass(frozen=True)
class TransportHead:
    schema: str
    origin_device: str
    ledger_epoch: int
    sequence: int
    event_hash: str
    anchor_hash: str
    checkpoint_id: str
    roster_version: int
    key_id: str
    signature: str


@dataclass(frozen=True)
class TransportPublishResult:
    published_events: int
    duplicates: int
    git_oid: str


class GitTransport:
    """Publish signed artifacts into an isolated Git repository."""

    def __init__(self, root: Path, *, remote: str | Path | None = None) -> None:
        self.root = Path(os.path.abspath(os.fspath(root)))
        self.remote = self._remote_value(remote)
        self._initialize()

    @staticmethod
    def _remote_value(remote: str | Path | None) -> str | None:
        if remote is None:
            return None
        value = os.fspath(remote)
        if (
            not isinstance(value, str)
            or not value
            or value.startswith("-")
            or "\x00" in value
            or "\n" in value
            or "\r" in value
        ):
            raise _invalid("operational Git transport remote is invalid")
        return value

    def _marker_bytes(self) -> bytes:
        remote_sha256 = (
            hashlib.sha256(self.remote.encode("utf-8")).hexdigest()
            if self.remote is not None
            else None
        )
        return canonical_json_bytes(
            {
                "remote_sha256": remote_sha256,
                "schema": _MARKER_SCHEMA,
            }
        )

    def _unsafe_repository(self, detail: str, exc: BaseException | None = None) -> OperationalError:
        error = _invalid(f"unsafe operational Git transport repository: {detail}")
        if exc is not None:
            error.__cause__ = exc
        return error

    def _open_root(self, *, create: bool = False) -> SecureDirectory:
        try:
            directory = open_secure_directory(self.root, create=create)
        except (OSError, ValueError) as exc:
            raise self._unsafe_repository(str(exc), exc) from exc
        return directory

    @staticmethod
    def _require_regular(observed: os.stat_result, description: str) -> None:
        if not stat.S_ISREG(observed.st_mode) or observed.st_nlink != 1:
            raise _invalid(f"unsafe operational Git transport repository entry: {description}")

    @classmethod
    def _read_regular(cls, directory: SecureDirectory, relative: Path) -> bytes:
        encoded, observed = directory.read_bytes_snapshot(relative)
        cls._require_regular(observed, str(relative))
        return encoded

    def _validate_marker(self, directory: SecureDirectory) -> None:
        try:
            observed = directory.stat(_MARKER_PATH)
            self._require_regular(observed, str(_MARKER_PATH))
            encoded = self._read_regular(directory, _MARKER_PATH)
        except FileNotFoundError:
            raise self._unsafe_repository(
                "repository was not explicitly initialized as a Memo transport"
            ) from None
        except (OSError, ValueError, OperationalError) as exc:
            if isinstance(exc, OperationalError):
                raise
            raise self._unsafe_repository(str(exc), exc) from exc
        if encoded != self._marker_bytes():
            raise self._unsafe_repository("Memo transport marker or remote binding changed")

    def _validate_git_config(self, directory: SecureDirectory) -> None:
        try:
            encoded = self._read_regular(directory, Path(".git/config"))
            text = encoded.decode("utf-8")
        except (OSError, ValueError, UnicodeDecodeError) as exc:
            raise self._unsafe_repository(
                "Git config is not a safe regular UTF-8 file", exc
            ) from exc

        section = ""
        allowed: dict[str, frozenset[str]] = {
            "core": frozenset(
                {
                    "repositoryformatversion",
                    "filemode",
                    "bare",
                    "logallrefupdates",
                    "ignorecase",
                    "precomposeunicode",
                }
            ),
            "user": frozenset({"name", "email"}),
            "remote": frozenset({"url", "fetch"}),
            "branch": frozenset({"remote", "merge"}),
        }
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or line.startswith(("#", ";")):
                continue
            if line.startswith("[") and line.endswith("]"):
                section = line[1:-1].split(maxsplit=1)[0].lower()
                if section not in allowed:
                    raise self._unsafe_repository(
                        f"Git config section is not allowed: {section or '<empty>'}"
                    )
                continue
            if "=" not in line or not section:
                raise self._unsafe_repository("Git config contains malformed input")
            key = line.split("=", 1)[0].strip().lower()
            if key not in allowed[section]:
                raise self._unsafe_repository(f"Git config key is not allowed: {section}.{key}")

    @staticmethod
    def _valid_transport_member(relative: Path, *, directory: bool) -> bool:
        parts = relative.parts
        if not parts:
            return False
        if parts[0] == "heads":
            if directory:
                return len(parts) == 1
            return (
                len(parts) == 2
                and parts[1].endswith(".json")
                and _SAFE_ID.fullmatch(parts[1][:-5]) is not None
            )
        if parts[0] not in {"anchors", "checkpoints", "events"}:
            return False
        if directory:
            if len(parts) == 1:
                return True
            if len(parts) == 2:
                return _SAFE_ID.fullmatch(parts[1]) is not None
            if len(parts) == 3:
                if _SAFE_ID.fullmatch(parts[1]) is None or _EPOCH.fullmatch(parts[2]) is None:
                    return False
                return int(parts[2]) <= _MAX_INT64
            return False
        if len(parts) != 4:
            return False
        if _SAFE_ID.fullmatch(parts[1]) is None or _EPOCH.fullmatch(parts[2]) is None:
            return False
        if int(parts[2]) > _MAX_INT64:
            return False
        filename = parts[3]
        if parts[0] == "anchors":
            return filename.endswith(".json") and _SHA256.fullmatch(filename[:-5]) is not None
        if parts[0] == "checkpoints":
            return filename.endswith(".json") and _SAFE_ID.fullmatch(filename[:-5]) is not None
        if _SEQUENCE_FILE.fullmatch(filename) is None:
            return False
        return 1 <= int(filename[:-6]) <= _MAX_INT64

    def _validate_directory_tree(
        self,
        directory: SecureDirectory,
        relative: Path,
        *,
        git_internal: bool,
    ) -> None:
        try:
            names = directory.list_names(relative)
        except (OSError, ValueError) as exc:
            raise self._unsafe_repository(str(exc), exc) from exc
        for name in names:
            member = relative / name
            try:
                observed = directory.stat(member)
            except FileNotFoundError:
                # Git may remove its own short-lived lock files after listdir.
                # A disappearing entry cannot be followed or written through.
                continue
            except (OSError, ValueError) as exc:
                raise self._unsafe_repository(str(exc), exc) from exc
            if stat.S_ISDIR(observed.st_mode):
                if not git_internal and not self._valid_transport_member(member, directory=True):
                    raise self._unsafe_repository(f"unexpected directory: {member}")
                self._validate_directory_tree(
                    directory,
                    member,
                    git_internal=git_internal,
                )
                continue
            try:
                self._require_regular(observed, str(member))
            except OperationalError as exc:
                raise self._unsafe_repository(str(member), exc) from exc
            if git_internal:
                if member == Path(".git/objects/info/alternates"):
                    raise self._unsafe_repository("Git object alternates are not allowed")
            elif not self._valid_transport_member(member, directory=False):
                raise self._unsafe_repository(f"unexpected file: {member}")

    def _validate_repository(self, directory: SecureDirectory, *, require_git: bool) -> None:
        self._validate_marker(directory)
        try:
            root_names = directory.list_names()
        except (OSError, ValueError) as exc:
            raise self._unsafe_repository(str(exc), exc) from exc
        allowed_root = {str(_MARKER_PATH), ".git", "anchors", "checkpoints", "events", "heads"}
        unexpected = set(root_names) - allowed_root
        if unexpected:
            raise self._unsafe_repository(
                f"unexpected root entries: {', '.join(sorted(unexpected))}"
            )
        if ".git" not in root_names:
            if require_git:
                raise self._unsafe_repository("Memo transport Git metadata is missing")
            return
        try:
            git_stat = directory.stat(".git")
        except (OSError, ValueError) as exc:
            raise self._unsafe_repository(str(exc), exc) from exc
        if not stat.S_ISDIR(git_stat.st_mode):
            raise self._unsafe_repository(".git is not a real directory")
        self._validate_directory_tree(directory, Path(".git"), git_internal=True)
        self._validate_git_config(directory)
        for name in ("anchors", "checkpoints", "events", "heads"):
            if name in root_names:
                self._validate_directory_tree(directory, Path(name), git_internal=False)

    def _git(self, *arguments: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        environment = {
            **{key: value for key, value in os.environ.items() if not key.startswith("GIT_")},
            "GIT_AUTHOR_NAME": "memo operational transport",
            "GIT_AUTHOR_EMAIL": "memo@localhost",
            "GIT_COMMITTER_NAME": "memo operational transport",
            "GIT_COMMITTER_EMAIL": "memo@localhost",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_SYSTEM": os.devnull,
            "GIT_EDITOR": "true",
            "GIT_LITERAL_PATHSPECS": "1",
            "GIT_PAGER": "cat",
            "GIT_SEQUENCE_EDITOR": "true",
            "GIT_TERMINAL_PROMPT": "0",
            "LC_ALL": "C",
        }
        with self._open_root() as directory:
            root_identity = directory.identity
            self._validate_repository(
                directory,
                require_git=not (arguments and arguments[0] == "init"),
            )
            command = (
                "git",
                "--git-dir=.git",
                "--work-tree=.",
                "-c",
                "core.hooksPath=/dev/null",
                "-c",
                "core.fsmonitor=false",
                "-c",
                "core.attributesFile=/dev/null",
                "-c",
                "core.excludesFile=/dev/null",
                "-c",
                "commit.gpgSign=false",
                "-c",
                "gc.auto=0",
                "-c",
                "maintenance.auto=false",
                "-c",
                "protocol.ext.allow=never",
                *arguments,
            )
            try:
                result = subprocess.run(
                    command,
                    cwd=self.root,
                    env=environment,
                    check=False,
                    text=True,
                    capture_output=True,
                )
            except (OSError, UnicodeError) as exc:
                raise _failure(f"operational Git transport failed: {exc}") from exc
            try:
                current_root = os.stat(self.root, follow_symlinks=False)
            except OSError as exc:
                raise self._unsafe_repository(
                    "transport root changed during Git operation", exc
                ) from exc
            if (
                not stat.S_ISDIR(current_root.st_mode)
                or (current_root.st_dev, current_root.st_ino) != root_identity
            ):
                raise self._unsafe_repository("transport root changed during Git operation")
        with self._open_root() as directory:
            self._validate_repository(directory, require_git=True)
        if check and result.returncode != 0:
            raise _failure(
                f"operational Git transport failed: "
                f"{result.stderr.strip() or result.stdout.strip()}"
            )
        return result

    def _initialize(self) -> None:
        with self._open_root(create=True) as directory:
            try:
                root_names = directory.list_names()
            except (OSError, ValueError) as exc:
                raise self._unsafe_repository(str(exc), exc) from exc
            if str(_MARKER_PATH) not in root_names:
                if root_names:
                    raise self._unsafe_repository(
                        "existing directory was not explicitly initialized as a Memo transport"
                    )
                try:
                    directory.create_bytes_exclusive(_MARKER_PATH, self._marker_bytes())
                except (OSError, ValueError) as exc:
                    raise self._unsafe_repository(str(exc), exc) from exc
            self._validate_marker(directory)
            has_git = ".git" in root_names
            if has_git:
                self._validate_repository(directory, require_git=True)
            elif set(directory.list_names()) != {str(_MARKER_PATH)}:
                raise self._unsafe_repository("uninitialized transport contains unexpected data")
        if not has_git:
            self._git("init", "--quiet")
        if self.remote is not None:
            self.refresh(required=False)

    def refresh(self, *, required: bool = True) -> bool:
        """Fetch and merge the configured immutable-artifact branch.

        Each Memo peer owns its own clone.  The remote branch is the only
        cross-device rendezvous; callers never read another peer's state
        directory directly.
        """

        if self.remote is None:
            return False
        fetched = self._git(
            "fetch",
            "--quiet",
            "--",
            self.remote,
            f"refs/heads/{_REMOTE_BRANCH}",
            check=False,
        )
        if fetched.returncode != 0:
            missing = (
                "couldn't find remote ref" in fetched.stderr.lower()
                or "could not find remote branch" in fetched.stderr.lower()
            )
            if missing and not required:
                return False
            raise _failure(f"operational Git transport fetch failed: {fetched.stderr.strip()}")
        fetched_oid = self._git("rev-parse", "--verify", "FETCH_HEAD^{commit}").stdout.strip()
        if _GIT_OID.fullmatch(fetched_oid) is None:
            raise self._unsafe_repository("fetched Git commit identity is invalid")
        self._validate_commit_tree(fetched_oid)
        local_head = self._git("rev-parse", "--verify", "HEAD", check=False)
        if local_head.returncode != 0:
            self._git("checkout", "--quiet", "-B", _REMOTE_BRANCH, fetched_oid)
        else:
            self._git("merge", "--quiet", "--no-edit", fetched_oid)
        return True

    def _push(self) -> None:
        if self.remote is None:
            return
        for attempt in range(2):
            pushed = self._git(
                "push",
                "--quiet",
                "--",
                self.remote,
                f"HEAD:refs/heads/{_REMOTE_BRANCH}",
                check=False,
            )
            if pushed.returncode == 0:
                return
            if attempt == 0:
                self.refresh(required=True)
                continue
            raise _failure(f"operational Git transport push failed: {pushed.stderr.strip()}")

    def _validate_commit_tree(self, revision: str) -> None:
        listed = self._git("ls-tree", "-rz", "--full-tree", revision)
        for entry in listed.stdout.split("\x00"):
            if not entry:
                continue
            try:
                metadata, raw_path = entry.split("\t", 1)
                mode, kind, oid = metadata.split(" ", 2)
                raw_path.encode("ascii")
            except (UnicodeEncodeError, ValueError) as exc:
                raise self._unsafe_repository("remote tree entry is malformed", exc) from exc
            relative = Path(raw_path)
            if (
                mode != "100644"
                or kind != "blob"
                or _GIT_OID.fullmatch(oid) is None
                or relative.is_absolute()
                or any(part in {"", ".", ".."} for part in relative.parts)
                or not self._valid_transport_member(relative, directory=False)
            ):
                raise self._unsafe_repository(f"remote tree entry is not allowed: {raw_path!r}")

    def _write_immutable(
        self,
        directory: SecureDirectory,
        relative: Path,
        data: bytes,
    ) -> bool:
        try:
            directory.create_bytes_exclusive(relative, data, mode=0o600)
        except FileExistsError:
            try:
                existing = self._read_regular(directory, relative)
            except (OSError, ValueError) as exc:
                raise self._unsafe_repository(str(exc), exc) from exc
            if existing != data:
                raise OperationalError(
                    OperationalErrorCode.ANCHOR_CONFLICT,
                    f"immutable operational transport artifact changed: {relative}",
                    retryable=False,
                ) from None
            return False
        except (OSError, ValueError) as exc:
            raise self._unsafe_repository(str(exc), exc) from exc
        return True

    def _write_head(
        self,
        directory: SecureDirectory,
        relative: Path,
        data: bytes,
    ) -> None:
        try:
            directory.atomic_write_bytes(relative, data, mode=0o600)
        except (OSError, ValueError) as exc:
            raise self._unsafe_repository(str(exc), exc) from exc

    @staticmethod
    def _validate_head(head: object, *, expected_origin: str | None = None) -> TransportHead:
        if not isinstance(head, TransportHead):
            raise _invalid("operational transport head has an invalid type")
        if head.schema != _HEAD_SCHEMA:
            raise _invalid("operational transport head schema is unsupported")
        origin = _safe(head.origin_device, "head origin")
        if expected_origin is not None and origin != expected_origin:
            raise _invalid(f"operational transport head identity mismatch: {expected_origin}")
        _integer(head.ledger_epoch, "head ledger epoch")
        sequence = _integer(head.sequence, "head sequence")
        _sha256(head.event_hash, "head event hash", allow_empty=sequence == 0)
        if sequence == 0 and head.event_hash != "":
            raise _invalid("operational transport zero sequence requires an empty event hash")
        _sha256(head.anchor_hash, "head anchor hash")
        _safe(head.checkpoint_id, "head checkpoint")
        _integer(head.roster_version, "head roster version", minimum=1)
        _safe(head.key_id, "head key id")
        if (
            not isinstance(head.signature, str)
            or not head.signature
            or len(head.signature) > 8192
            or any(ord(character) < 0x20 for character in head.signature)
        ):
            raise _invalid("operational transport head signature is invalid")
        return head

    def _decode_head(
        self,
        directory: SecureDirectory,
        origin: str,
        *,
        required: bool,
    ) -> TransportHead | None:
        relative = Path("heads") / f"{_safe(origin, 'origin')}.json"
        try:
            encoded = self._read_regular(directory, relative)
        except FileNotFoundError:
            if not required:
                return None
            raise _failure(f"operational transport head is missing: {origin}") from None
        except (OSError, ValueError) as exc:
            raise self._unsafe_repository(str(exc), exc) from exc
        try:
            value = json.loads(encoded.decode("utf-8"))
            if not isinstance(value, dict) or canonical_json_bytes(value) != encoded:
                raise ValueError
            head = TransportHead(**value)
        except (TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise OperationalError(
                OperationalErrorCode.INVALID_EVENT,
                f"invalid operational transport head: {origin}",
                retryable=False,
            ) from exc
        return self._validate_head(head, expected_origin=origin)

    def publish(self, bundle: OriginBundle) -> TransportPublishResult:
        if not isinstance(bundle, OriginBundle):
            raise _invalid("operational transport publish requires an origin bundle")
        origin = _safe(bundle.anchor.origin_device, "origin")
        epoch = _integer(bundle.anchor.ledger_epoch, "ledger epoch")
        key_id = bundle.events[-1].key_id if bundle.events else bundle.anchor.key_id
        signature = bundle.events[-1].signature if bundle.events else bundle.anchor.signature
        head = self._validate_head(
            TransportHead(
                schema=_HEAD_SCHEMA,
                origin_device=origin,
                ledger_epoch=epoch,
                sequence=bundle.head_sequence,
                event_hash=bundle.head_hash,
                anchor_hash=bundle.anchor.anchor_hash,
                checkpoint_id=bundle.anchor.checkpoint_id,
                roster_version=bundle.anchor.roster_version,
                key_id=key_id,
                signature=signature,
            ),
            expected_origin=origin,
        )
        if not isinstance(bundle.checkpoint, bytes):
            raise _invalid("operational transport checkpoint must be bytes")
        anchor_bytes = canonical_json_bytes(bundle.anchor)
        event_artifacts: list[tuple[Path, bytes]] = []
        previous_sequence = 0
        for event in bundle.events:
            if event.origin_device != origin:
                raise _invalid("operational transport bundle mixes origins")
            sequence = _integer(event.origin_sequence, "event sequence", minimum=1)
            if sequence <= previous_sequence or sequence > head.sequence:
                raise _invalid("operational transport bundle event sequence is invalid")
            previous_sequence = sequence
            event_artifacts.append(
                (
                    Path("events") / origin / str(epoch) / f"{sequence:020d}.jsonl",
                    canonical_json_bytes(event) + b"\n",
                )
            )
        if bundle.events and previous_sequence != head.sequence:
            raise _invalid("operational transport bundle head sequence does not match events")
        published = 0
        duplicates = 0
        with authority_write_lock(self.root / _MARKER_PATH):
            self.refresh(required=False)
            with self._open_root() as directory:
                self._validate_repository(directory, require_git=True)
                try:
                    for artifact_root in ("anchors", "checkpoints", "events", "heads"):
                        directory.ensure_directory(artifact_root)
                except (OSError, ValueError) as exc:
                    raise self._unsafe_repository(str(exc), exc) from exc
                anchor_path = Path("anchors") / origin / str(epoch) / f"{head.anchor_hash}.json"
                checkpoint_path = (
                    Path("checkpoints") / origin / str(epoch) / f"{head.checkpoint_id}.json"
                )
                self._write_immutable(directory, anchor_path, anchor_bytes)
                self._write_immutable(directory, checkpoint_path, bundle.checkpoint)
                for segment, encoded in event_artifacts:
                    if self._write_immutable(directory, segment, encoded):
                        published += 1
                    else:
                        duplicates += 1
                existing = self._decode_head(directory, origin, required=False)
                if existing is not None:
                    if existing.sequence > head.sequence:
                        raise OperationalError(
                            OperationalErrorCode.ANCHOR_CONFLICT,
                            "operational transport head cannot regress",
                            retryable=False,
                        )
                    if (
                        existing.sequence == head.sequence
                        and existing.event_hash != head.event_hash
                    ):
                        raise OperationalError(
                            OperationalErrorCode.ANCHOR_CONFLICT,
                            "operational transport head fork detected",
                            retryable=False,
                        )
                self._write_head(
                    directory,
                    Path("heads") / f"{origin}.json",
                    canonical_json_bytes(asdict(head)),
                )
            self._git("add", "--", "events", "anchors", "checkpoints", "heads")
            status = self._git("diff", "--cached", "--quiet", check=False)
            if status.returncode not in {0, 1}:
                raise _failure(status.stderr.strip() or "cannot inspect operational Git index")
            if status.returncode == 1:
                self._git("commit", "--quiet", "-m", f"memo operational {origin}@{head.sequence}")
            self._push()
            oid = self._git("rev-parse", "HEAD").stdout.strip()
        return TransportPublishResult(
            published_events=published,
            duplicates=duplicates,
            git_oid=oid,
        )

    def origins(self) -> tuple[str, ...]:
        with self._open_root() as directory:
            self._validate_marker(directory)
            try:
                names = directory.list_names("heads")
            except FileNotFoundError:
                return ()
            except (OSError, ValueError) as exc:
                raise self._unsafe_repository(str(exc), exc) from exc
            origins: list[str] = []
            for name in names:
                if not name.endswith(".json") or _SAFE_ID.fullmatch(name[:-5]) is None:
                    raise self._unsafe_repository(f"invalid transport head filename: {name!r}")
                relative = Path("heads") / name
                try:
                    self._require_regular(directory.stat(relative), str(relative))
                except (OSError, ValueError, OperationalError) as exc:
                    if isinstance(exc, OperationalError):
                        raise self._unsafe_repository(str(relative), exc) from exc
                    raise self._unsafe_repository(str(exc), exc) from exc
                origins.append(name[:-5])
            return tuple(sorted(origins))

    def read_head(self, origin: str, *, required: bool = True) -> TransportHead | None:
        with self._open_root() as directory:
            self._validate_marker(directory)
            return self._decode_head(directory, _safe(origin, "origin"), required=required)

    def read_anchor(self, head: TransportHead) -> bytes:
        valid = self._validate_head(head)
        relative = (
            Path("anchors")
            / valid.origin_device
            / str(valid.ledger_epoch)
            / f"{valid.anchor_hash}.json"
        )
        return self._read_artifact(relative)

    def read_checkpoint(self, head: TransportHead) -> bytes:
        valid = self._validate_head(head)
        relative = (
            Path("checkpoints")
            / valid.origin_device
            / str(valid.ledger_epoch)
            / f"{valid.checkpoint_id}.json"
        )
        return self._read_artifact(relative)

    def segment_path(self, head: TransportHead, sequence: int) -> Path:
        valid = self._validate_head(head)
        valid_sequence = _integer(sequence, "segment sequence", minimum=1)
        if valid_sequence > valid.sequence:
            raise _invalid("operational transport segment sequence exceeds the signed head")
        return (
            self.root
            / "events"
            / valid.origin_device
            / str(valid.ledger_epoch)
            / f"{valid_sequence:020d}.jsonl"
        )

    def read_segment(self, head: TransportHead, sequence: int) -> bytes | None:
        valid = self._validate_head(head)
        valid_sequence = _integer(sequence, "segment sequence", minimum=1)
        if valid_sequence > valid.sequence:
            raise _invalid("operational transport segment sequence exceeds the signed head")
        relative = (
            Path("events")
            / valid.origin_device
            / str(valid.ledger_epoch)
            / f"{valid_sequence:020d}.jsonl"
        )
        try:
            encoded = self._read_artifact(relative)
        except FileNotFoundError:
            return None
        if not encoded.endswith(b"\n") or encoded.count(b"\n") != 1:
            raise OperationalError(
                OperationalErrorCode.INVALID_EVENT,
                f"invalid operational transport segment framing: "
                f"{valid.origin_device}/{valid_sequence}",
                retryable=False,
            )
        return encoded[:-1]

    def _read_artifact(self, relative: Path) -> bytes:
        with self._open_root() as directory:
            self._validate_marker(directory)
            try:
                return self._read_regular(directory, relative)
            except FileNotFoundError:
                raise
            except (OSError, ValueError) as exc:
                raise self._unsafe_repository(str(exc), exc) from exc


__all__ = [
    "GitTransport",
    "TransportHead",
    "TransportPublishResult",
]
