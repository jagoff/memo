"""Git-backed cross-machine sync for memo (F4 of memo-sync).

The memorias dir lives inside a git repo (e.g. ~/repos/memo-sync, with `.md`
under `memorias/` and signal under `signal/`). Git is the single source of
truth and transport between Macs; sync runs at session boundaries (pull on
SessionStart, push on Stop), not second-by-second.

Design (decoupled from memflow — its git_sync is not importable here):

  push = export-signal → git add -A → commit (if dirty) → push
  pull = fetch → pre-merge remote signal from the git object into the DB
         (so signal is never lost regardless of how git resolves the file)
         → rebase → auto-resolve signal/*.json with --theirs → reindex
         → re-export signal

The DB is the source of truth for signal; `signal/*.json` is a regenerable
transport, so taking either git side and re-exporting the merged DB is safe.
Real conflicts on a memoria `.md` (same memoria edited on two Macs) are rare;
the rebase aborts and reports rather than guessing.
"""

from __future__ import annotations

import contextlib
import json
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from memo.errors import MemoError
from memo.sync_signal import _FILES, SIGNAL_SCHEMA, export_signal, signal_dir_for

if TYPE_CHECKING:
    from memo.config import Config
    from memo.memory import Memory
    from memo.store import VecStore

_GIT_TIMEOUT = 120


class SyncGitError(MemoError):
    """Git sync failed (not a clean state, conflict needing manual resolution, etc.)."""


def git_root_for(cfg: Config) -> Path:
    """The git repo root holding the memorias — parent of the memorias dir."""
    root = cfg.memory_dir.parent
    if not (root / ".git").exists():
        raise SyncGitError(
            f"{root} is not a git repo (no .git). Run `memo sync clone <url>` or "
            f"point MEMO_DATA_DIR at a memorias dir inside a git clone."
        )
    return root


def _git(root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    cp = subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True,
        text=True,
        timeout=_GIT_TIMEOUT,
    )
    if check and cp.returncode != 0:
        raise SyncGitError(f"git {' '.join(args)} failed: {cp.stderr.strip() or cp.stdout.strip()}")
    return cp


def _current_branch(root: Path) -> str:
    return _git(root, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip() or "main"


def clone_bootstrap(url: str, dest: Path) -> dict:
    """Clone the memo-sync repo to `dest` for a new machine (F6).

    Returns a summary plus the memorias path the caller should point
    MEMO_DATA_DIR at. Does NOT mutate config or reindex — that is the caller's
    explicit next step (config touchpoints vary per machine).
    """
    if dest.exists() and any(dest.iterdir()):
        raise SyncGitError(f"{dest} already exists and is not empty")
    dest.parent.mkdir(parents=True, exist_ok=True)
    cp = subprocess.run(
        ["git", "clone", url, str(dest)],
        capture_output=True,
        text=True,
        timeout=_GIT_TIMEOUT,
    )
    if cp.returncode != 0:
        raise SyncGitError(f"git clone failed: {cp.stderr.strip()}")
    memorias = dest / "memorias"
    if not memorias.exists():
        raise SyncGitError(f"cloned repo has no memorias/ dir at {memorias}")
    n_md = len(list(memorias.rglob("*.md")))
    return {"cloned": str(dest), "memorias_dir": str(memorias), "memorias": n_md}


def bootstrap_clone(url: str, dest: Path, *, config_path: Path | None = None) -> dict:
    """One-step new-machine bootstrap: clone the memo-sync repo (or reuse an
    existing clone at `dest`) and point `config.toml`'s `data_dir` at its
    `memorias/`.

    Idempotent: if `dest` is already a git clone with a `memorias/` dir, it is
    reused (no re-clone, no pull — ongoing sync is `memo sync pull`'s job).
    Existing `[storage]` keys (`vault_path`, `memories_in_vault`, `single_db`)
    are preserved; only `data_dir` is repointed.

    Does NOT reindex or import signal — that needs a `Memory` and is the
    caller's next step (see the `memo sync bootstrap` CLI command).
    """
    from memo.setup.config_io import load_config_file, write_config_file

    memorias = dest / "memorias"
    if (dest / ".git").exists() and memorias.exists():
        n_md = len(list(memorias.rglob("*.md")))
        summary = {"cloned": str(dest), "memorias_dir": str(memorias), "memorias": n_md, "reused": True}
    else:
        summary = clone_bootstrap(url, dest)
        summary["reused"] = False

    existing = (load_config_file(config_path) or {}).get("storage", {})
    vault_path = existing.get("vault_path")
    written = write_config_file(
        data_dir=memorias,
        vault_path=Path(vault_path) if vault_path else None,
        memories_in_vault=bool(existing.get("memories_in_vault")),
        single_db=bool(existing.get("single_db")),
        path=config_path,
    )
    summary["config"] = str(written)
    return summary


def sync_tier(cfg: Config) -> str:
    """Which sync model applies. ``"local"`` = intra-machine only (sessions share
    ``data_dir``/``memvec.db``; visibility is via the shared index, no git).
    ``"remote"`` = a git remote is configured, so cross-machine sync is in play.

    Makes the today-implicit local-vs-cloud decision explicit and single-sourced.
    """
    try:
        root = git_root_for(cfg)
    except SyncGitError:
        return "local"
    has_remote = bool(_git(root, "remote", check=False).stdout.strip())
    return "remote" if has_remote else "local"


def _lock_path(cfg: Config) -> Path:
    return cfg.state_dir / ".sync.lock"


def sync_once(
    cfg: Config,
    store: VecStore,
    mem: Memory,
    *,
    remote: str = "origin",
    do_pull: bool = True,
    do_push: bool = True,
) -> dict:
    """The single, machine-level git step — pull-rebase-before-push, lock-guarded.

    ONE owner per machine: whoever grabs the ``.sync.lock`` flock does the git;
    concurrent same-machine sessions skip (``locked``) because their writes are
    already in the shared store — the lock holder's push carries them. The
    pull-before-push ordering means a remote that advanced (another Mac) rebases
    cleanly instead of rejecting the push.

    No-op (``tier='local'``) when no remote is configured. Never raises for the
    expected cases (not a clone, lock held, offline) — returns a status dict so
    triggers (hooks/daemon) stay quiet; only a genuine `.md` rebase conflict
    surfaces as a ``pending`` marker.
    """
    import fcntl

    if sync_tier(cfg) != "remote":
        return {"tier": "local", "skipped": "no remote"}

    lock_file = _lock_path(cfg)
    try:
        lock_file.parent.mkdir(parents=True, exist_ok=True)
        fh = lock_file.open("w")
    except OSError as exc:
        return {"tier": "remote", "skipped": f"lock open failed: {exc}"}
    try:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            # Another process on this machine owns the git step right now. Our
            # writes are already in the shared store; the holder will carry them.
            return {"tier": "remote", "skipped": "locked"}

        result: dict = {"tier": "remote", "pulled": False, "pushed": False}
        if do_pull:
            try:
                pull_out = sync_pull(cfg, store, mem, remote=remote)
                result["pulled"] = bool(pull_out.get("pulled"))
            except SyncGitError as exc:
                result["pull_error"] = str(exc)
        if do_push:
            try:
                push_out = sync_push(cfg, store, remote=remote)
                result["pushed"] = bool(push_out.get("pushed"))
            except SyncGitError as exc:
                result["push_error"] = str(exc)
        return result
    finally:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        finally:
            fh.close()


def _pending_marker(cfg: Config) -> Path:
    """Sentinel file written when a push fails (offline / auth / remote down) so
    `sync status` / `doctor` can surface stranded local commits and the next
    trigger knows to retry. Lives in the state dir (not the synced repo)."""
    return cfg.state_dir / "sync_pending"


def sync_status(cfg: Config, *, remote: str = "origin", check_remote: bool = False) -> dict:
    """Read-only health of the git-sync repo — never raises, never mutates.

    Surfaces the silent no-op (data_dir not a git clone) and stranded commits
    (committed locally but not pushed: offline/auth). ``ahead``/``behind`` are
    relative to the LAST fetch unless ``check_remote`` (then a network
    ``ls-remote`` probe runs — slower, used by `doctor`, not the hot path).
    """
    try:
        root = git_root_for(cfg)
    except SyncGitError as exc:
        return {"is_git_clone": False, "reason": str(exc)}

    branch = _current_branch(root)
    porcelain = _git(root, "status", "--porcelain", check=False).stdout.strip()
    dirty = len(porcelain.split("\n")) if porcelain else 0
    remote_url = _git(root, "remote", "get-url", remote, check=False).stdout.strip()
    ahead = behind = 0
    counts = _git(
        root, "rev-list", "--left-right", "--count", f"{remote}/{branch}...HEAD", check=False
    )
    if counts.returncode == 0 and counts.stdout.strip():
        parts = counts.stdout.split()
        if len(parts) == 2:
            behind, ahead = int(parts[0]), int(parts[1])
    last = _git(root, "log", "-1", "--format=%cI", check=False).stdout.strip()

    remote_reachable: bool | None = None
    if check_remote and remote_url:
        probe = _git(root, "ls-remote", "--exit-code", remote, "HEAD", check=False)
        remote_reachable = probe.returncode == 0

    return {
        "is_git_clone": True,
        "root": str(root),
        "branch": branch,
        "remote": remote_url,
        "dirty_files": dirty,
        "ahead": ahead,
        "behind": behind,
        "last_commit": last,
        "pending": _pending_marker(cfg).is_file(),
        "remote_reachable": remote_reachable,
    }


def sync_push(cfg: Config, store: VecStore, *, remote: str = "origin") -> dict:
    """Export signal, commit any changes, and push. Returns a summary dict."""
    root = git_root_for(cfg)
    branch = _current_branch(root)
    export_signal(store, signal_dir_for(cfg))

    _git(root, "add", "-A")
    staged = _git(root, "diff", "--cached", "--name-only").stdout.strip()
    n_files = len(staged.splitlines()) if staged else 0
    if staged:
        from memo.identity import current as _identity

        who = _identity(cfg).label
        _git(root, "commit", "-m", f"sync: memo signal + memorias ({n_files} files) [{who}]")

    # Stranded-commit retry: a prior push may have failed AFTER committing
    # (offline/auth), leaving local commits unpushed. Detect that and push even
    # when there's nothing new to commit this round — otherwise the early return
    # would strand the work until the next save.
    unpushed = _git(
        root, "rev-list", "--count", f"{remote}/{branch}..HEAD", check=False
    ).stdout.strip()
    has_unpushed = unpushed.isdigit() and int(unpushed) > 0
    if not staged and not has_unpushed and not _pending_marker(cfg).is_file():
        return {"pushed": False, "reason": "nothing to commit", "branch": branch}

    # push; set upstream on first push
    push = _git(root, "push", remote, branch, check=False)
    if push.returncode != 0:
        push = _git(root, "push", "-u", remote, branch, check=False)
        if push.returncode != 0:
            # Commit landed locally but didn't reach the remote (offline / auth /
            # remote down). Stamp a pending marker so `sync status` / `doctor`
            # flag the stranded commit and the next trigger retries — the work is
            # NOT lost, just not yet shared.
            with contextlib.suppress(OSError):
                _pending_marker(cfg).write_text(branch, encoding="utf-8")
            raise SyncGitError(f"git push failed (commit kept locally, will retry): {push.stderr.strip()}")
    # Pushed — clear any prior pending marker.
    _pending_marker(cfg).unlink(missing_ok=True)
    return {"pushed": True, "committed_files": n_files, "branch": branch}


def _merge_remote_signal_from_git(root: Path, store: VecStore, ref: str) -> None:
    """Read each signal file from a git ref and merge it into the DB.

    Reading from the git object (not the working tree) guarantees the remote's
    signal lands in the DB before any rebase/conflict resolution can drop it.
    Missing files in the ref are skipped.
    """
    payload: dict[str, list[dict]] = {}
    for table, filename in _FILES.items():
        cp = _git(root, "show", f"{ref}:signal/{filename}", check=False)
        if cp.returncode != 0:
            payload[table] = []
            continue
        try:
            doc = json.loads(cp.stdout)
        except json.JSONDecodeError:
            payload[table] = []
            continue
        if doc.get("schema") != SIGNAL_SCHEMA:
            payload[table] = []
            continue
        payload[table] = doc.get("rows") or []
    store.merge_signal(payload)


def sync_pull(cfg: Config, store: VecStore, mem: Memory, *, remote: str = "origin") -> dict:
    """Fetch + rebase + merge remote signal into the DB + reindex. Returns summary."""
    root = git_root_for(cfg)
    branch = _current_branch(root)

    _git(root, "fetch", remote, branch)
    remote_ref = f"{remote}/{branch}"

    # 1) remote signal → DB (loss-proof: from the git object, pre-rebase)
    _merge_remote_signal_from_git(root, store, remote_ref)

    # 2) rebase local commits onto the remote tip
    rebase = _git(root, "rebase", "--autostash", remote_ref, check=False)
    if rebase.returncode != 0:
        conflicts = _git(root, "diff", "--name-only", "--diff-filter=U", check=False).stdout.split()
        non_signal = [c for c in conflicts if not c.startswith("signal/")]
        if non_signal:
            _git(root, "rebase", "--abort", check=False)
            raise SyncGitError(
                "rebase conflict in memorias needs manual resolution: "
                + ", ".join(non_signal)
            )
        # only signal/*.json conflicts — DB already holds the union, take theirs
        for c in conflicts:
            _git(root, "checkout", "--theirs", "--", c, check=False)
            _git(root, "add", "--", c)
        cont = _git(root, "rebase", "--continue", check=False)
        if cont.returncode != 0:
            _git(root, "rebase", "--abort", check=False)
            raise SyncGitError(f"rebase --continue failed: {cont.stderr.strip()}")

    # 3) load any new/changed memorias the pull brought in
    reindexed = mem.reindex()

    # 4) re-export the merged signal so the next push carries the union
    export_signal(store, signal_dir_for(cfg))

    return {"pulled": True, "branch": branch, "reindexed": reindexed}
