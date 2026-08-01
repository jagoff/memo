"""Immutable Git-backed transport for signed operational ledger artifacts."""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from memo.atomic_io import authority_write_lock
from memo.errors import OperationalError, OperationalErrorCode
from memo.operational_event import OriginBundle, canonical_json_bytes

_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_HEAD_SCHEMA = "memo.operational_transport_head.v1"
_REMOTE_BRANCH = "memo-operational"


def _failure(message: str) -> OperationalError:
    return OperationalError(
        OperationalErrorCode.STORAGE_UNAVAILABLE,
        message,
        retryable=True,
    )


def _safe(value: str, description: str) -> str:
    if not _SAFE_ID.fullmatch(value):
        raise OperationalError(
            OperationalErrorCode.INVALID_EVENT,
            f"unsafe operational transport {description}: {value!r}",
            retryable=False,
        )
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
        self.root = Path(root).resolve()
        self.remote = str(remote) if remote is not None else None
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._initialize()

    def _git(self, *arguments: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        environment = {
            **os.environ,
            "GIT_AUTHOR_NAME": "memo operational transport",
            "GIT_AUTHOR_EMAIL": "memo@localhost",
            "GIT_COMMITTER_NAME": "memo operational transport",
            "GIT_COMMITTER_EMAIL": "memo@localhost",
            "GIT_EDITOR": "true",
            "GIT_SEQUENCE_EDITOR": "true",
        }
        try:
            return subprocess.run(
                ("git", *arguments),
                cwd=self.root,
                env=environment,
                check=check,
                text=True,
                capture_output=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            detail = exc.stderr.strip() if isinstance(exc, subprocess.CalledProcessError) else str(exc)
            raise _failure(f"operational Git transport failed: {detail}") from exc

    def _initialize(self) -> None:
        if not (self.root / ".git").is_dir():
            self._git("init", "--quiet")
        self._git("config", "user.name", "memo operational transport")
        self._git("config", "user.email", "memo@localhost")
        if self.remote is not None:
            current = self._git("remote", "get-url", "origin", check=False)
            if current.returncode == 0:
                if current.stdout.strip() != self.remote:
                    raise _failure("operational Git transport remote binding changed")
            else:
                self._git("remote", "add", "origin", self.remote)
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
            "origin",
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
        local_head = self._git("rev-parse", "--verify", "HEAD", check=False)
        if local_head.returncode != 0:
            self._git("checkout", "--quiet", "-B", _REMOTE_BRANCH, "FETCH_HEAD")
        else:
            self._git("merge", "--quiet", "--no-edit", "FETCH_HEAD")
        return True

    def _push(self) -> None:
        if self.remote is None:
            return
        for attempt in range(2):
            pushed = self._git(
                "push",
                "--quiet",
                "origin",
                f"HEAD:refs/heads/{_REMOTE_BRANCH}",
                check=False,
            )
            if pushed.returncode == 0:
                return
            if attempt == 0:
                self.refresh(required=True)
                continue
            raise _failure(f"operational Git transport push failed: {pushed.stderr.strip()}")

    @staticmethod
    def _write_immutable(path: Path, data: bytes) -> bool:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            if path.read_bytes() != data:
                raise OperationalError(
                    OperationalErrorCode.ANCHOR_CONFLICT,
                    f"immutable operational transport artifact changed: {path}",
                    retryable=False,
                ) from None
            return False
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
        except BaseException:
            path.unlink(missing_ok=True)
            raise
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return True

    @staticmethod
    def _write_head(path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def publish(self, bundle: OriginBundle) -> TransportPublishResult:
        origin = _safe(bundle.anchor.origin_device, "origin")
        epoch = bundle.anchor.ledger_epoch
        if isinstance(epoch, bool) or epoch < 0:
            raise OperationalError(
                OperationalErrorCode.INVALID_EVENT,
                "operational transport ledger epoch is invalid",
                retryable=False,
            )
        published = 0
        duplicates = 0
        with authority_write_lock(self.root / ".memo-operational-transport"):
            self.refresh(required=False)
            anchor_path = (
                self.root
                / "anchors"
                / origin
                / str(epoch)
                / f"{bundle.anchor.anchor_hash}.json"
            )
            checkpoint_path = (
                self.root
                / "checkpoints"
                / origin
                / str(epoch)
                / f"{_safe(bundle.anchor.checkpoint_id, 'checkpoint')}.json"
            )
            self._write_immutable(anchor_path, canonical_json_bytes(bundle.anchor))
            self._write_immutable(checkpoint_path, bundle.checkpoint)
            for event in bundle.events:
                if event.origin_device != origin:
                    raise OperationalError(
                        OperationalErrorCode.INVALID_EVENT,
                        "operational transport bundle mixes origins",
                        retryable=False,
                    )
                segment = (
                    self.root
                    / "events"
                    / origin
                    / str(epoch)
                    / f"{event.origin_sequence:020d}.jsonl"
                )
                if self._write_immutable(segment, canonical_json_bytes(event) + b"\n"):
                    published += 1
                else:
                    duplicates += 1
            head = TransportHead(
                schema=_HEAD_SCHEMA,
                origin_device=origin,
                ledger_epoch=epoch,
                sequence=bundle.head_sequence,
                event_hash=bundle.head_hash,
                anchor_hash=bundle.anchor.anchor_hash,
                checkpoint_id=bundle.anchor.checkpoint_id,
                roster_version=bundle.anchor.roster_version,
                key_id=(
                    bundle.events[-1].key_id
                    if bundle.events
                    else bundle.anchor.key_id
                ),
                signature=(
                    bundle.events[-1].signature
                    if bundle.events
                    else bundle.anchor.signature
                ),
            )
            head_path = self.root / "heads" / f"{origin}.json"
            existing = self.read_head(origin, required=False)
            if existing is not None:
                if existing.sequence > head.sequence:
                    raise OperationalError(
                        OperationalErrorCode.ANCHOR_CONFLICT,
                        "operational transport head cannot regress",
                        retryable=False,
                    )
                if existing.sequence == head.sequence and existing.event_hash != head.event_hash:
                    raise OperationalError(
                        OperationalErrorCode.ANCHOR_CONFLICT,
                        "operational transport head fork detected",
                        retryable=False,
                    )
            self._write_head(head_path, canonical_json_bytes(asdict(head)))
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
        heads = self.root / "heads"
        if not heads.is_dir():
            return ()
        return tuple(sorted(path.stem for path in heads.glob("*.json") if _SAFE_ID.fullmatch(path.stem)))

    def read_head(self, origin: str, *, required: bool = True) -> TransportHead | None:
        path = self.root / "heads" / f"{_safe(origin, 'origin')}.json"
        try:
            encoded = path.read_bytes()
        except FileNotFoundError:
            if not required:
                return None
            raise _failure(f"operational transport head is missing: {origin}") from None
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
        if (
            head.schema != _HEAD_SCHEMA
            or head.origin_device != origin
            or not head.key_id
            or not head.signature
        ):
            raise OperationalError(
                OperationalErrorCode.INVALID_EVENT,
                f"operational transport head identity mismatch: {origin}",
                retryable=False,
            )
        return head

    def read_anchor(self, head: TransportHead) -> bytes:
        return (
            self.root
            / "anchors"
            / _safe(head.origin_device, "origin")
            / str(head.ledger_epoch)
            / f"{head.anchor_hash}.json"
        ).read_bytes()

    def read_checkpoint(self, head: TransportHead) -> bytes:
        return (
            self.root
            / "checkpoints"
            / _safe(head.origin_device, "origin")
            / str(head.ledger_epoch)
            / f"{_safe(head.checkpoint_id, 'checkpoint')}.json"
        ).read_bytes()

    def segment_path(self, head: TransportHead, sequence: int) -> Path:
        return (
            self.root
            / "events"
            / _safe(head.origin_device, "origin")
            / str(head.ledger_epoch)
            / f"{sequence:020d}.jsonl"
        )

    def read_segment(self, head: TransportHead, sequence: int) -> bytes | None:
        try:
            encoded = self.segment_path(head, sequence).read_bytes()
        except FileNotFoundError:
            return None
        if not encoded.endswith(b"\n") or encoded.count(b"\n") != 1:
            raise OperationalError(
                OperationalErrorCode.INVALID_EVENT,
                f"invalid operational transport segment framing: {head.origin_device}/{sequence}",
                retryable=False,
            )
        return encoded[:-1]


__all__ = [
    "GitTransport",
    "TransportHead",
    "TransportPublishResult",
]
