"""Auto-update on memo-mcp start.

When ``MEMO_AUTO_UPDATE`` is enabled, memo-mcp checks (throttled) whether a
newer **tagged** release exists in the git repo and, if so, spawns a detached
``memo update --to-tag <tag>`` in the background. The running process keeps
the old code (you can't hot-swap a live interpreter) — the new version takes
effect on the NEXT memo-mcp start.

Design choices (updated 2026-07-12):
- Trigger is a git **tag** (``vX.Y.Z``), not any commit, so an un-tagged push
  (work in progress / a broken commit) never propagates to the fleet.
- Default OFF: remote checks and installation require explicit opt-in.
- Network + git failures are swallowed: auto-update must never break or delay a
  memo-mcp startup.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import logging
import os
import platform
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol

from memo.config import Config
from memo.flags import flag_bool, flag_int, flag_str

_log = logging.getLogger(__name__)

DEFAULT_REPO = "https://github.com/jagoff/memo.git"
_CHECK_STAMP = "auto_update_check"
_NOTIFY_FILE = "update_available"
_SPAWNED_STAMP = "auto_update_spawned"
_SPAWN_LOCK_FILE = ".auto_update_spawned.lock"
_STARTING_LEASE_TTL_SECONDS = 60


class _UpdateProcess(Protocol):
    pid: int

    def poll(self) -> int | None: ...


@dataclass(frozen=True)
class _SpawnLease:
    tag: str
    pid: int
    state: str
    started_at: float
    process_identity: str | None


_ACTIVE_UPDATES: dict[tuple[str, str], _UpdateProcess] = {}
_SPAWN_LOCK = threading.Lock()


@contextlib.contextmanager
def _spawn_file_lock(cfg: Config) -> Iterator[None]:
    """Serialize every persisted lease transition across memo processes."""

    cfg.state_dir.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(cfg.state_dir / _SPAWN_LOCK_FILE, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        with contextlib.suppress(OSError):
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _parse_semver(tag: str) -> tuple[int, int, int] | None:
    """``v1.2.3`` / ``1.2.3`` → ``(1, 2, 3)``. Anything non-numeric → None.

    Pre-release/build suffixes (``1.2.3-rc1`` / ``1.2.3+build``) are rejected:
    automatic updates track stable release tags only.
    """
    core = tag.strip().lstrip("vV")
    if "-" in core or "+" in core:
        return None
    parts = core.split(".")
    if len(parts) != 3:
        return None
    try:
        return (int(parts[0]), int(parts[1]), int(parts[2]))
    except ValueError:
        return None


def is_newer(remote: str, local: str) -> bool:
    """True iff ``remote`` is a strictly higher semver than ``local``."""
    r, lo = _parse_semver(remote), _parse_semver(local)
    if r is None or lo is None:
        return False
    return r > lo


def latest_remote_tag(repo_url: str, *, timeout: int = 10) -> str | None:
    """Highest ``vX.Y.Z`` tag in the remote repo, or None on any failure.

    Uses ``git ls-remote --tags --refs`` so no clone/fetch is needed and the
    probe stays cheap. Dereferenced (``--refs``) so peeled ``^{}`` lines are
    excluded.
    """
    try:
        cp = subprocess.run(
            ["git", "ls-remote", "--tags", "--refs", repo_url],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        _log.debug("auto-update: ls-remote failed: %s", exc)
        return None
    if cp.returncode != 0:
        _log.debug("auto-update: ls-remote rc=%d: %s", cp.returncode, cp.stderr.strip())
        return None

    best: tuple[int, int, int] | None = None
    best_tag: str | None = None
    for line in cp.stdout.splitlines():
        ref = line.rsplit("refs/tags/", 1)[-1].strip()
        if not ref or ref == line:
            continue
        ver = _parse_semver(ref)
        if ver is not None and (best is None or ver > best):
            best, best_tag = ver, ref
    return best_tag


def _anon_id(cfg: Config) -> str:
    """Anonymous, stable per-install id: sha256 of the persisted device_id.

    The raw ``device_id`` never leaves the machine — only its truncated hash,
    which lets the update endpoint dedupe active installs without identifying
    them. Empty string if device_id is unavailable (no id is sent).
    """
    raw = str(getattr(cfg, "device_id", "") or "")
    if not raw:
        return ""
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def latest_tag_via_endpoint(
    endpoint: str,
    *,
    anon_id: str,
    version: str,
    os_name: str,
    timeout: int = 10,
) -> str | None:
    """Resolve the latest tag from an HTTP update endpoint, or None on failure.

    The GET both (a) returns the latest release tag — its real job, so this is a
    functional version check, not pure telemetry — and (b) lets the endpoint
    record a deduped active-install heartbeat from the query params. All params
    are anonymous: a hashed install id, the current version, and the OS name.
    Any failure returns None so the caller falls back to the git probe.
    """
    from urllib.parse import urlencode

    query = urlencode({"id": anon_id, "v": version, "os": os_name})
    url = f"{endpoint}{'&' if '?' in endpoint else '?'}{query}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": f"memo/{version}"})  # noqa: S310
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        _log.debug("update-endpoint: unreachable (%s)", exc)
        return None
    tag = data.get("latest") if isinstance(data, dict) else None
    if isinstance(tag, str) and _parse_semver(tag) is not None:
        return tag
    return None


def resolve_latest_tag(cfg: Config, repo_url: str, *, timeout: int = 10) -> str | None:
    """Latest tag via the configured HTTP endpoint if set, else git ls-remote.

    When ``MEMO_UPDATE_ENDPOINT`` is empty (the default) this is exactly
    ``latest_remote_tag`` — no network destination changes, no heartbeat. When
    set, the HTTP endpoint is tried first (functional version check + anonymous
    heartbeat) and the git probe is the fallback on any failure.
    """
    endpoint = flag_str("MEMO_UPDATE_ENDPOINT")
    if endpoint:
        from memo import __version__

        tag = latest_tag_via_endpoint(
            endpoint,
            anon_id=_anon_id(cfg),
            version=__version__,
            os_name=platform.system(),
            timeout=timeout,
        )
        if tag is not None:
            return tag
    return latest_remote_tag(repo_url, timeout=timeout)


def tag_is_on_remote_master(repo_url: str, tag: str, *, timeout: int = 60) -> bool:
    """Verify a release tag resolves to a commit reachable from remote master."""
    if _parse_semver(tag) is None:
        return False
    try:
        with tempfile.TemporaryDirectory(prefix="memo-update-provenance-") as directory:
            init = subprocess.run(
                ["git", "init", "--bare", directory],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if init.returncode != 0:
                return False
            fetch = subprocess.run(
                [
                    "git",
                    "-C",
                    directory,
                    "fetch",
                    "--quiet",
                    "--filter=blob:none",
                    repo_url,
                    "+refs/heads/master:refs/heads/master",
                    f"+refs/tags/{tag}:refs/tags/{tag}",
                ],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if fetch.returncode != 0:
                return False
            ancestry = subprocess.run(
                [
                    "git",
                    "-C",
                    directory,
                    "merge-base",
                    "--is-ancestor",
                    f"refs/tags/{tag}",
                    "refs/heads/master",
                ],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return ancestry.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False


def _should_check(cfg: Config, interval_s: int, now: float, force: bool = False) -> bool:
    """Throttle: True if no check stamp or it's older than ``interval_s``.

    If force=True, bypass the throttle (for explicit update --check).
    """
    if force:
        return True
    stamp = cfg.state_dir / _CHECK_STAMP
    try:
        last = float(stamp.read_text().strip())
    except (OSError, ValueError):
        return True
    return (now - last) >= interval_s


def _record_check(cfg: Config, now: float) -> None:
    stamp = cfg.state_dir / _CHECK_STAMP
    try:
        stamp.parent.mkdir(parents=True, exist_ok=True)
        stamp.write_text(str(now))
    except OSError as exc:
        _log.debug("auto-update: could not write check stamp: %s", exc)


def _write_notify(cfg: Config, tag: str) -> None:
    path = cfg.state_dir / _NOTIFY_FILE
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(tag)
    except OSError as exc:
        _log.debug("auto-update: could not write notify file: %s", exc)


def _clear_notify(cfg: Config) -> None:
    path = cfg.state_dir / _NOTIFY_FILE
    with contextlib.suppress(OSError):
        path.unlink(missing_ok=True)


def _spawn_key(cfg: Config, tag: str) -> tuple[str, str]:
    return (str(cfg.state_dir.resolve()), tag)


def _read_spawned_lease(cfg: Config) -> _SpawnLease | None:
    try:
        payload = json.loads((cfg.state_dir / _SPAWNED_STAMP).read_text())
        tag = payload["tag"]
        pid = payload["pid"]
        state = payload["state"]
        started_at = payload["started_at"]
        process_identity = payload.get("process_identity")
        if (
            not isinstance(tag, str)
            or not isinstance(pid, int)
            or pid <= 0
            or state not in {"starting", "running", "succeeded"}
            or not isinstance(started_at, (int, float))
            or (process_identity is not None and not isinstance(process_identity, str))
        ):
            return None
        return _SpawnLease(tag, pid, state, float(started_at), process_identity)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _write_spawned_lease(
    cfg: Config,
    tag: str,
    pid: int,
    state: str,
    *,
    started_at: float | None = None,
    process_identity: str | None = None,
) -> bool:
    path = cfg.state_dir / _SPAWNED_STAMP
    temporary_name: str | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w",
            dir=path.parent,
            prefix=f".{_SPAWNED_STAMP}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            json.dump(
                {
                    "tag": tag,
                    "pid": pid,
                    "state": state,
                    "started_at": time.time() if started_at is None else started_at,
                    "process_identity": process_identity,
                },
                temporary,
                sort_keys=True,
            )
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, path)
        return True
    except OSError as exc:
        _log.debug("auto-update: could not write spawned lease: %s", exc)
        return False
    finally:
        if temporary_name is not None:
            with contextlib.suppress(OSError):
                os.unlink(temporary_name)


def _acquire_spawned_lease_unlocked(cfg: Config, tag: str, *, started_at: float) -> bool:
    """Atomically acquire the cross-process starting lease."""
    path = cfg.state_dir / _SPAWNED_STAMP
    temporary_name: str | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w",
            dir=path.parent,
            prefix=f".{_SPAWNED_STAMP}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            json.dump(
                {
                    "tag": tag,
                    "pid": os.getpid(),
                    "state": "starting",
                    "started_at": started_at,
                    "process_identity": _process_identity(os.getpid()),
                },
                temporary,
                sort_keys=True,
            )
            temporary.flush()
            os.fsync(temporary.fileno())
        # Hard-linking a fully written temporary file is an atomic create: only
        # one memo-mcp process can acquire a missing lease path.
        os.link(temporary_name, path)
        return True
    except FileExistsError:
        return False
    except OSError as exc:
        _log.debug("auto-update: could not acquire spawned lease: %s", exc)
        return False
    finally:
        if temporary_name is not None:
            with contextlib.suppress(OSError):
                os.unlink(temporary_name)


def _acquire_spawned_lease(cfg: Config, tag: str, *, started_at: float) -> bool:
    """Acquire a starting lease while serialized with every lease mutation."""

    with _spawn_file_lock(cfg):
        return _acquire_spawned_lease_unlocked(cfg, tag, started_at=started_at)


def _clear_spawned_stamp_unlocked(
    cfg: Config,
    *,
    tag: str | None = None,
    pid: int | None = None,
) -> None:
    """Release a matching spawn lease without clobbering a newer updater."""
    path = cfg.state_dir / _SPAWNED_STAMP
    if tag is not None or pid is not None:
        lease = _read_spawned_lease(cfg)
        if lease is None:
            # Legacy stamps contained only the tag. Let the matching old child
            # release that format, but never an unrelated tag.
            try:
                raw = path.read_text().strip()
            except OSError:
                return
            if tag is not None and raw != tag:
                return
        elif (tag is not None and lease.tag != tag) or (pid is not None and lease.pid != pid):
            return
    with contextlib.suppress(OSError):
        path.unlink(missing_ok=True)


def _clear_spawned_stamp(
    cfg: Config,
    *,
    tag: str | None = None,
    pid: int | None = None,
) -> None:
    """Release a matching lease as one serialized compare-and-delete."""

    with _spawn_file_lock(cfg):
        _clear_spawned_stamp_unlocked(cfg, tag=tag, pid=pid)


def _mark_spawned_success_unlocked(cfg: Config, tag: str, *, pid: int | None = None) -> None:
    """Persist successful completion so the same tag is not spawned again."""
    lease = _read_spawned_lease(cfg)
    if lease is None or lease.tag != tag or (pid is not None and lease.pid != pid):
        return
    _write_spawned_lease(
        cfg,
        lease.tag,
        lease.pid,
        "succeeded",
        started_at=lease.started_at,
        process_identity=lease.process_identity,
    )


def _mark_spawned_success(cfg: Config, tag: str, *, pid: int | None = None) -> None:
    """Persist success as one serialized compare-and-transition."""

    with _spawn_file_lock(cfg):
        _mark_spawned_success_unlocked(cfg, tag, pid=pid)


def _claim_spawned_lease(
    cfg: Config,
    tag: str,
    *,
    parent_pid: int,
    child_pid: int,
    started_at: float,
) -> bool:
    """Let the spawned worker durably claim a lease if its parent disappears."""

    with _spawn_file_lock(cfg):
        lease = _read_spawned_lease(cfg)
        if lease is None or lease.tag != tag:
            return False
        identity = _process_identity(child_pid)
        if identity is None:
            return False
        if lease.state == "running":
            return lease.pid == child_pid and lease.process_identity == identity
        if lease.state != "starting" or lease.pid != parent_pid or lease.started_at != started_at:
            return False
        return _write_spawned_lease(
            cfg,
            tag,
            child_pid,
            "running",
            started_at=started_at,
            process_identity=identity,
        )


def _process_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _process_identity(pid: int) -> str | None:
    """Return a stable OS start identity so PID reuse cannot preserve a lease."""

    try:
        with open(f"/proc/{pid}/stat") as process_stat:
            stat_fields = process_stat.read().split()
        if len(stat_fields) > 21:
            return f"proc:{stat_fields[21]}"
    except OSError:
        pass
    try:
        result = subprocess.run(
            ["ps", "-o", "lstart=", "-p", str(pid)],
            capture_output=True,
            text=True,
            timeout=1,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    started = result.stdout.strip()
    return f"ps:{started}" if result.returncode == 0 and started else None


def _lease_process_is_active(lease: _SpawnLease) -> bool:
    """Require both a live PID and the same process instance."""

    if not _process_is_alive(lease.pid) or lease.process_identity is None:
        return False
    return _process_identity(lease.pid) == lease.process_identity


def _terminate_untracked_process(proc: _UpdateProcess) -> None:
    """Best-effort stop/reap the detached process group and its descendants."""

    terminate = getattr(proc, "terminate", None)
    wait = getattr(proc, "wait", None)
    kill = getattr(proc, "kill", None)
    with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
        os.killpg(proc.pid, signal.SIGTERM)
    with contextlib.suppress(Exception):
        if callable(terminate):
            terminate()
    if callable(wait):
        with contextlib.suppress(Exception):
            wait(timeout=1)
    # The session leader may exit on TERM while a descendant ignores it. Probe
    # the original process group with KILL even after the leader was reaped.
    with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
        os.killpg(proc.pid, signal.SIGKILL)
    with contextlib.suppress(Exception):
        if callable(kill):
            kill()
    with contextlib.suppress(Exception):
        if callable(wait):
            wait(timeout=1)


def _active_process_blocks_update(cfg: Config, tag: str) -> bool | None:
    """Supervise this process's child; None means no in-memory child exists."""

    key = _spawn_key(cfg, tag)
    proc = _ACTIVE_UPDATES.get(key)
    if proc is None:
        return None
    returncode = proc.poll()
    if returncode is None:
        return True
    _ACTIVE_UPDATES.pop(key, None)
    if returncode == 0:
        _mark_spawned_success_unlocked(cfg, tag, pid=proc.pid)
        return True
    _clear_spawned_stamp_unlocked(cfg, tag=tag, pid=proc.pid)
    return False


def _spawn_lease_blocks_update_unlocked(cfg: Config, tag: str, *, now: float) -> bool:
    """Supervise the active child or recover a stale persisted lease."""
    active_result = _active_process_blocks_update(cfg, tag)
    if active_result is not None:
        return active_result

    lease = _read_spawned_lease(cfg)
    if lease is None:
        # Corrupt and legacy tag-only stamps cannot prove an update is active.
        _clear_spawned_stamp_unlocked(cfg)
        return False
    if lease.state == "starting":
        # The child-side worker will claim this lease after Popen. A fresh
        # starting lease therefore blocks even when the parent died in the
        # tiny Popen→running crash window. Only an unclaimed stale lease clears.
        if now - lease.started_at <= _STARTING_LEASE_TTL_SECONDS:
            return True
        _clear_spawned_stamp_unlocked(cfg, tag=lease.tag, pid=lease.pid)
        return False
    if lease.tag != tag:
        if lease.state != "succeeded" and _lease_process_is_active(lease):
            return True
        _clear_spawned_stamp_unlocked(cfg, tag=lease.tag, pid=lease.pid)
        return False
    if lease.state == "succeeded":
        return True
    if _lease_process_is_active(lease):
        return True
    _clear_spawned_stamp_unlocked(cfg, tag=tag, pid=lease.pid)
    return False


def _spawn_lease_blocks_update(cfg: Config, tag: str, *, now: float) -> bool:
    """Inspect/recover a lease while serialized with other processes."""

    with _spawn_file_lock(cfg):
        return _spawn_lease_blocks_update_unlocked(cfg, tag, now=now)


def pending_update_tag(cfg: Config | None = None) -> str | None:
    """Return the pending update tag if one was recorded, else None.

    Fast (no network) — reads the notify file written by ``notify_if_newer``
    or ``maybe_auto_update``. Used by the startup banner and status commands.
    """
    try:
        cfg = cfg or Config.from_env()
        path = cfg.state_dir / _NOTIFY_FILE
        tag = path.read_text().strip()
        return tag if tag else None
    except (OSError, Exception):
        return None


def notify_if_newer(cfg: Config | None = None, *, force: bool = False) -> str | None:
    """Check git tags and write a notify file if a newer version exists.

    Throttled to once per ``MEMO_AUTO_UPDATE_INTERVAL_S`` (default 6h) unless
    ``force=True``. Returns the newer tag if found, else None. Never raises.
    """
    try:
        cfg = cfg or Config.from_env()
        import time

        now = time.time()
        interval = flag_int("MEMO_AUTO_UPDATE_INTERVAL_S") or 21600
        if not force and not _should_check(cfg, interval, now):
            return pending_update_tag(cfg)
        _record_check(cfg, now)

        from memo import __version__

        repo = flag_str("MEMO_AUTO_UPDATE_REPO") or DEFAULT_REPO
        tag = resolve_latest_tag(cfg, repo)
        if tag and is_newer(tag, __version__) and tag_is_on_remote_master(repo, tag):
            _write_notify(cfg, tag)
            return tag
        else:
            _clear_notify(cfg)
            return None
    except Exception as exc:
        _log.debug("notify-if-newer: skipped (%s)", exc)
        return None


def maybe_auto_update(cfg: Config | None = None) -> bool:
    """Entry point called at memo-mcp startup. Gated, non-blocking.

    Checks for a newer tag on every startup (git ls-remote is cheap). Guards
    against re-spawning the same tag via a per-tag stamp so repeated startups
    during an in-progress install don't pile up subprocess.Popen calls.

    Auto-update is off by default and requires ``MEMO_AUTO_UPDATE=1``.

    Returns True iff a background update was spawned (mainly for tests). Never
    raises — any failure is logged at debug and swallowed so a startup is never
    delayed or broken by the updater.
    """
    try:
        cfg = cfg or Config.from_env()

        if not flag_bool("MEMO_AUTO_UPDATE"):
            return False

        # Ensure state_dir exists for stamps
        with contextlib.suppress(OSError):
            cfg.state_dir.mkdir(parents=True, exist_ok=True)

        from memo import __version__

        repo = flag_str("MEMO_AUTO_UPDATE_REPO") or DEFAULT_REPO
        # Fast path: if notify_if_newer already confirmed a newer version
        # (wrote the update_available file), trust it and skip the network call.
        tag = pending_update_tag(cfg) or resolve_latest_tag(cfg, repo)
        if not tag or not is_newer(tag, __version__) or not tag_is_on_remote_master(repo, tag):
            _clear_notify(cfg)
            return False

        # Homebrew is a user-managed channel: `brew upgrade` owns updates, so we
        # OFFER (write the notify → the banner shows `brew upgrade mlx-memo`)
        # rather than run brew unattended in the background. The pipx/uv/PyPI
        # channels below still self-install.
        from memo.runtime.detect import is_homebrew_install

        if is_homebrew_install():
            _write_notify(cfg, tag)
            return False

        with _SPAWN_LOCK, _spawn_file_lock(cfg):
            # Inspection, stale cleanup, acquisition, spawn and transition
            # are one cross-process critical section. A concurrent cleanup
            # can never delete a newer lease between compare and unlink.
            if _spawn_lease_blocks_update_unlocked(cfg, tag, now=time.time()):
                _write_notify(cfg, tag)  # keep banner visible
                return False

            _write_notify(cfg, tag)
            _log.info("auto-update: %s → %s (spawning background update)", __version__, tag)
            started_at = time.time()
            if not _acquire_spawned_lease_unlocked(cfg, tag, started_at=started_at):
                return False
            log_file = cfg.state_dir / "auto_update.log"
            log_file.parent.mkdir(parents=True, exist_ok=True)
            try:
                with open(log_file, "a") as fh:
                    proc = subprocess.Popen(
                        [
                            sys.executable,
                            "-m",
                            "memo.runtime.autoupdate_worker",
                            str(cfg.state_dir),
                            tag,
                            str(os.getpid()),
                            repr(started_at),
                        ],
                        stdout=fh,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
            except (OSError, ValueError, subprocess.SubprocessError):
                _clear_spawned_stamp_unlocked(cfg, tag=tag, pid=os.getpid())
                raise
            _ACTIVE_UPDATES[_spawn_key(cfg, tag)] = proc
            process_identity = _process_identity(proc.pid)
            if process_identity is None or not _write_spawned_lease(
                cfg,
                tag,
                proc.pid,
                "running",
                started_at=started_at,
                process_identity=process_identity,
            ):
                _ACTIVE_UPDATES.pop(_spawn_key(cfg, tag), None)
                _terminate_untracked_process(proc)
                _clear_spawned_stamp_unlocked(cfg, tag=tag, pid=os.getpid())
                raise OSError("could not persist auto-update child lease")
        return True
    except Exception as exc:  # never break startup
        _log.debug("auto-update: skipped (%s)", exc)
        return False
