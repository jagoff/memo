"""Git-backed cross-machine sync for memo (F4 of memo-sync).

The memories dir lives inside a git repo (e.g. ~/repos/memo-sync, with `.md`
under `memories/` and signal under `signal/`). Git is the single source of
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
Real conflicts on a memory `.md` (same memory edited on two Macs) are rare;
the rebase aborts and reports rather than guessing.
"""

from __future__ import annotations

import contextlib
import json
import os
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
    """The git repo root holding the memories — parent of the memories dir."""
    root = cfg.memory_dir.parent
    if not (root / ".git").exists():
        raise SyncGitError(
            f"{root} is not a git repo (no .git). Run `memo sync clone <url>` or "
            f"point MEMO_DATA_DIR at a memories dir inside a git clone."
        )
    return root


def _git(root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    try:
        cp = subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT,
            # memo drives git non-interactively (daemon / hook / SSH session): a
            # `rebase --continue` after auto-resolving a signal conflict otherwise
            # tries to open an editor and dies with "Terminal is dumb, EDITOR unset",
            # which sync_once swallows → silent perpetual cross-Mac divergence.
            env={**os.environ, "GIT_EDITOR": "true", "GIT_SEQUENCE_EDITOR": "true"},
        )
    except subprocess.TimeoutExpired as exc:
        # A hung git call must surface as the domain error every sync caller
        # already handles (sync_once / --quiet hooks), not a raw traceback.
        # Applies to check=False too: there is no CompletedProcess to return.
        raise SyncGitError(f"git {' '.join(args)} timed out after {_GIT_TIMEOUT}s") from exc
    if check and cp.returncode != 0:
        raise SyncGitError(f"git {' '.join(args)} failed: {cp.stderr.strip() or cp.stdout.strip()}")
    return cp


def _current_branch(root: Path) -> str:
    return _git(root, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip() or "main"


def _git_dir(root: Path) -> Path:
    """Resolve the actual git dir (worktrees have an indirect `.git` file)."""
    out = _git(root, "rev-parse", "--absolute-git-dir", check=False).stdout.strip()
    return Path(out) if out else root / ".git"


def _abort_stale_rebase(root: Path) -> bool:
    """Abort a rebase a PREVIOUS process left mid-flight (SIGKILL on the git
    subprocess timeout, machine sleep, killed Stop hook).

    Resuming someone else's rebase is never safe: git's "already a rebase-merge
    directory" fatal contains the literal `--skip`, which used to false-positive
    the empty-commit recovery in ``sync_pull`` and finish the stale rebase —
    silently dropping local commits. ``--abort`` restores the pre-rebase branch;
    ``--quit`` (drops the state without touching the tree) is the fallback when
    abort itself fails. Returns True if stale state was found and cleared.
    """
    gd = _git_dir(root)
    if not (gd / "rebase-merge").exists() and not (gd / "rebase-apply").exists():
        return False
    if _git(root, "rebase", "--abort", check=False).returncode != 0:
        _git(root, "rebase", "--quit", check=False)
    return True


def _corpus_subdir(root: Path) -> Path:
    """The corpus dir inside a sync repo. New repos use ``memories/``; clones
    created before the rename used ``memorias/`` — honor whichever already
    exists so existing fleets keep working. Defaults to ``memories/`` when
    neither is present (a fresh repo)."""
    new = root / "memories"
    if new.exists():
        return new
    legacy = root / "memorias"
    if legacy.exists():
        return legacy
    return new


def clone_bootstrap(url: str, dest: Path) -> dict:
    """Clone the memo-sync repo to `dest` for a new machine (F6).

    Returns a summary plus the corpus path the caller should point
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
    memories = _corpus_subdir(dest)
    if not memories.exists():
        raise SyncGitError(
            f"cloned repo has no memories/ (or legacy memorias/) dir at {dest}"
        )
    n_md = len(list(memories.rglob("*.md")))
    return {"cloned": str(dest), "memories_dir": str(memories), "memories": n_md}


def bootstrap_clone(url: str, dest: Path, config_path: Path | None = None) -> dict:
    """One-step new-machine bootstrap: clone the memo-sync repo (or reuse an
    existing clone at `dest`) and point `config.toml`'s `data_dir` at its
    `memories/`.

    Idempotent: if `dest` is already a git clone with a `memories/` dir, it is
    reused (no re-clone, no pull — ongoing sync is `memo sync pull`'s job).
    Existing `[storage]` keys (`vault_path`, `memories_in_vault`, `single_db`)
    are preserved; only `data_dir` is repointed.

    Does NOT reindex or import signal — that needs a `Memory` and is the
    caller's next step (see the `memo sync bootstrap` CLI command).
    """
    from memo.setup.config_io import load_config_file, write_config_file

    memories = _corpus_subdir(dest)
    git_ok = (dest / ".git").exists()
    n_md = len(list(memories.rglob("*.md"))) if memories.exists() else 0
    if git_ok and n_md > 0:
        summary = {
            "cloned": str(dest),
            "memories_dir": str(memories),
            "memories": n_md,
            "reused": True,
        }
    elif git_ok:
        # A git clone with zero markdown is a BROKEN corpus, not a fresh one:
        # reusing it then `reindex --rebuild` (the CLI's next step) would truncate
        # the index against an empty disk and wipe it. Refuse with recovery steps.
        raise SyncGitError(
            f"{dest} is a git clone but has no .md under memories/ — refusing to "
            f"bootstrap (a rebuild would wipe the index). Restore tracked files with "
            f"`git -C {dest} restore .`, or `rm -rf {dest}` then re-run bootstrap."
        )
    else:
        summary = clone_bootstrap(url, dest)
        summary["reused"] = False

    existing = (load_config_file(config_path) or {}).get("storage", {})
    vault_path = existing.get("vault_path")
    # Re-read from the summary: on a fresh clone, `memories` was resolved before
    # the clone existed (so it defaulted to memories/); the summary reflects the
    # corpus dir that actually landed (memories/ or legacy memorias/).
    written = write_config_file(
        data_dir=Path(str(summary["memories_dir"])),
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
        fh = lock_file.open("w")  # type: ignore[call-overload]
    except OSError as exc:
        return {"tier": "remote", "skipped": f"lock open failed: {exc}"}
    try:
        try:
            flags = fcntl.fcntl(fh.fileno(), fcntl.F_GETFD)
            fcntl.fcntl(fh.fileno(), fcntl.F_SETFD, flags | fcntl.FD_CLOEXEC)
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            # Another process on this machine owns the git step right now. Our
            # writes are already in the shared store; the holder will carry them.
            return {"tier": "remote", "skipped": "locked"}

        result: dict = {"tier": "remote", "pulled": False, "pushed": False}
        # Commit local work (incl. deletions) BEFORE the pull. An uncommitted
        # delete/edit would be lost to `rebase --autostash` when the remote still
        # has the file; committing first makes it a real commit the rebase merges.
        try:
            _commit_local(cfg, store)
        except SyncGitError as exc:
            result["commit_error"] = str(exc)
        if do_pull:
            try:
                pull_out = sync_pull(cfg, store, mem, remote=remote)
                result["pulled"] = bool(pull_out.get("pulled"))
                result["pull"] = pull_out
            except SyncGitError as exc:
                result["pull_error"] = str(exc)
        if do_push:
            try:
                push_out = sync_push(cfg, store, remote=remote)
                result["pushed"] = bool(push_out.get("pushed"))
                result["push"] = push_out
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


def _stamp_pending(cfg: Config, branch: str, *, reason: str | None = None) -> None:
    """Write the sync_pending sentinel. With `reason` (secret gate) the marker
    is JSON so status/doctor can surface WHY; plain branch text otherwise
    (the legacy format — old markers keep parsing)."""
    payload = json.dumps({"branch": branch, "reason": reason}) if reason else branch
    with contextlib.suppress(OSError):
        _pending_marker(cfg).write_text(payload, encoding="utf-8")


def _read_pending_reason(cfg: Config) -> str | None:
    """The block reason from a JSON sync_pending marker, else None (missing
    marker, legacy plain-text marker, or unreadable)."""
    marker = _pending_marker(cfg)
    if not marker.is_file():
        return None
    try:
        raw = marker.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if raw.startswith("{"):
        with contextlib.suppress(json.JSONDecodeError):
            reason = json.loads(raw).get("reason")
            return str(reason) if reason else None
    return None


def _scan_staged_secrets(root: Path) -> list[str]:
    """Scan ADDED lines of staged `.md` diffs for secrets — pattern tier only
    (never entropy: a push block must not false-positive on the hashes/ids
    that pepper memory bodies).

    Added lines are collected PER FILE and joined with newlines before the
    scan: `_PEM_RE` requires the BEGIN and END markers in the same string, so
    a line-at-a-time scan would let a multi-line private-key block sail
    through (verified: 0 hits per-line vs 1 hit joined on a 4-line PEM).
    Token patterns are single-line, so joining never hides them. Joining may
    concatenate added lines from different hunks of one file — that can only
    over-match (a conservative block), never under-match.

    Returns human-readable findings like
    ``memories/leak.md: github-token ****WXYZ``."""
    from memo.redact import scan_secrets

    diff = _git(root, "diff", "--cached", "--unified=0", "--no-color", "--", "*.md").stdout
    added_by_file: dict[str, list[str]] = {}
    current = ""
    for line in diff.splitlines():
        if line.startswith("+++ b/"):
            current = line[6:]
            continue
        if not line.startswith("+") or line.startswith("+++"):
            continue
        added_by_file.setdefault(current, []).append(line[1:])
    findings: list[str] = []
    for path, lines in added_by_file.items():
        findings.extend(
            f"{path}: {kind} {preview}"
            for kind, preview in scan_secrets("\n".join(lines))
        )
    return findings


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
        "pending_reason": _read_pending_reason(cfg),
        "remote_reachable": remote_reachable,
    }


def _commit_local(cfg: Config, store: VecStore) -> tuple[Path, str, int]:
    """Export signal, stage everything (incl. deletions), and commit if dirty.

    Returns ``(root, branch, n_committed)``. Idempotent — a no-op commit when the
    tree is already clean. Pulled out of ``sync_push`` so the coordinator can
    commit local work BEFORE the pull/rebase: an uncommitted deletion/edit would
    otherwise be lost to ``rebase --autostash`` when the remote still has the
    file (the delete-during-sync bug). Committing first turns local work into a
    real commit the rebase merges correctly.
    """
    root = git_root_for(cfg)
    branch = _current_branch(root)
    # An interrupted merge/rebase leaves unmerged paths whose working-tree
    # files hold raw conflict markers — `add -A` would stage and commit that
    # garbage (and then push it to every machine). Refuse; the module contract
    # is "abort and report rather than guessing".
    unmerged = [
        line
        for line in _git(root, "status", "--porcelain", check=False).stdout.splitlines()
        if line[:2] in ("DD", "AU", "UD", "UA", "DU", "AA", "UU")
    ]
    if unmerged:
        raise SyncGitError(
            "unmerged paths in sync repo (interrupted merge/rebase) — refusing to "
            "commit conflict markers: " + ", ".join(line[3:] for line in unmerged)
        )
    export_signal(store, signal_dir_for(cfg))
    _git(root, "add", "-A")
    staged = _git(root, "diff", "--cached", "--name-only").stdout.strip()
    n_files = len(staged.splitlines()) if staged else 0
    if staged:
        from memo.flags import flag_bool

        if flag_bool("MEMO_SYNC_SECRET_GATE"):
            findings = _scan_staged_secrets(root)
            if findings:
                reason = "secret-scan blocked commit: " + "; ".join(findings[:5])
                _stamp_pending(cfg, branch, reason=reason)
                raise SyncGitError(
                    f"{reason} — remove/mask the secret in the file(s) and re-run "
                    "`memo sync once` (one-shot bypass: MEMO_SYNC_SECRET_GATE=0). "
                    "Nothing was committed; the secret never entered git history."
                )
        from memo.identity import current as _identity

        who = _identity(cfg).label
        _git(root, "commit", "-m", f"sync: memo signal + memories ({n_files} files) [{who}]")
    return root, branch, n_files


def sync_push(cfg: Config, store: VecStore, *, remote: str = "origin") -> dict:
    """Commit any local changes and push. Returns a summary dict."""
    root, branch, n_files = _commit_local(cfg, store)

    # Stranded-commit retry: a prior push may have failed AFTER committing
    # (offline/auth), leaving local commits unpushed. Detect that and push even
    # when there's nothing new to commit this round — otherwise the early return
    # would strand the work until the next save.
    unpushed = _git(
        root, "rev-list", "--count", f"{remote}/{branch}..HEAD", check=False
    ).stdout.strip()
    has_unpushed = unpushed.isdigit() and int(unpushed) > 0
    if n_files == 0 and not has_unpushed and not _pending_marker(cfg).is_file():
        return {"pushed": False, "reason": "nothing to commit", "branch": branch}

    # Stamp pending marker BEFORE push attempt — if we crash between now and
    # the push, the retry mechanism will catch it on next trigger.
    _stamp_pending(cfg, branch)

    # push; set upstream on first push
    push = _git(root, "push", remote, branch, check=False)
    if push.returncode != 0:
        push = _git(root, "push", "-u", remote, branch, check=False)
        if push.returncode != 0:
            # Commit landed locally but didn't reach the remote (offline / auth /
            # remote down). Stamp a pending marker so `sync status` / `doctor`
            # flag the stranded commit and the next trigger retries — the work is
            # NOT lost, just not yet shared.
            _stamp_pending(cfg, branch)
            raise SyncGitError(
                f"git push failed (commit kept locally, will retry): {push.stderr.strip()}"
            )
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
            import logging
            logging.getLogger(__name__).warning(
                "sync: skipping %s signal (schema %r != %r)", table, doc.get("schema"), SIGNAL_SCHEMA
            )
            payload[table] = []
            continue
        payload[table] = doc.get("rows") or []
    store.merge_signal(payload)


def sync_pull(cfg: Config, store: VecStore, mem: Memory, *, remote: str = "origin") -> dict:
    """Fetch + rebase + merge remote signal into the DB + reindex. Returns summary."""
    root = git_root_for(cfg)

    # 0) clean up a rebase a crashed prior sync left mid-flight — BEFORE reading
    # the branch (mid-rebase HEAD is detached) and before starting our own rebase
    # (git would refuse with the '--skip'-bearing fatal that used to false-
    # positive the recovery loop below).
    stale_rebase_aborted = _abort_stale_rebase(root)

    branch = _current_branch(root)

    _git(root, "fetch", remote, branch)
    remote_ref = f"{remote}/{branch}"

    # 1) remote signal → DB (loss-proof: from the git object, pre-rebase)
    _merge_remote_signal_from_git(root, store, remote_ref)

    # 1b) rebuild feedback vectors for newly imported feedback rows
    store.rebuild_feedback_vecs(mem.embedder.embed_query)

    # 2) rebase local commits onto the remote tip. The rebase stops ONCE PER
    # local commit that conflicts; signal/*.json conflicts on essentially every
    # cross-Mac sync (machine-local counters), so resolve-and-continue in a LOOP
    # until the rebase finishes or a real `.md` conflict needs a human. (Handling
    # only the first stop left multi-commit divergence stuck and behind forever.)
    rebase = _git(root, "rebase", "--autostash", remote_ref, check=False)
    while rebase.returncode != 0:
        # A pre-existing rebase state must NEVER be "recovered" with --skip: its
        # fatal mentions `--skip`, but resuming it finishes the STALE rebase and
        # drops local commits. Unreachable after the abort in step 0 — kept as a
        # hard guard so the recovery below only ever touches OUR rebase.
        if "already a rebase" in rebase.stderr or "rebase-merge directory" in rebase.stderr:
            raise SyncGitError(
                "a rebase was already in progress (stale state from an interrupted "
                "sync) — not resuming it: " + rebase.stderr.strip()
            )
        conflicts = _git(root, "diff", "--name-only", "--diff-filter=U", check=False).stdout.split()
        non_signal = [c for c in conflicts if not c.startswith("signal/")]
        if non_signal or not conflicts:
            # Empty commit after resolving all signal conflicts: git rebase --continue
            # exits non-zero with 'nothing to stage' / '--skip'. Handle it before aborting.
            if not non_signal and not conflicts and (
                "--skip" in rebase.stderr or "nothing to stage" in rebase.stderr or "nothing to commit" in rebase.stderr
            ):
                rebase = _git(root, "rebase", "--skip", check=False)
                continue
            # a real memory conflict, or a failure with nothing to resolve
            # (don't loop forever): abort and surface for manual handling.
            _git(root, "rebase", "--abort", check=False)
            raise SyncGitError(
                "rebase conflict needs manual resolution: "
                + (", ".join(non_signal) or rebase.stderr.strip())
            )
        # only signal/*.json conflicts — DB already holds the union, take theirs
        for c in conflicts:
            _git(root, "checkout", "--theirs", "--", c, check=False)
            _git(root, "add", "--", c)
        rebase = _git(root, "rebase", "--continue", check=False)

    # 3) load any new/changed memories the pull brought in
    reindexed = mem.reindex()

    # 3b) prune index rows whose `.md` the pull DELETED. reindex() only adds/
    # updates from existing files — without this, a memory deleted on another
    # Mac stays findable here (orphan row) until a full rebuild. gc(fix=True)
    # drops rows whose `.md` is gone, so a deletion propagates cross-machine.
    pruned: list[str] = []
    try:
        pruned = mem.gc(fix=True).get("orphan_store", [])
    except Exception as exc:  # never let GC break the pull
        import logging
        logging.getLogger(__name__).warning("sync_pull: GC failed (orphan rows may remain): %s", exc)

    # 4) re-export the merged signal so the next push carries the union
    export_signal(store, signal_dir_for(cfg))

    out = {"pulled": True, "branch": branch, "reindexed": reindexed, "pruned": len(pruned)}
    if stale_rebase_aborted:
        out["stale_rebase_aborted"] = True
    return out


def sync_init_home(cfg: Config, private: bool = True) -> dict:
    """Initialize a new memo-sync repo: create GitHub repo + ensure local git + first push.

    Uses `gh repo create` to create the remote, ensures the local memories
    dir is a git repo (running `git init` if needed), and pushes.
    Returns the repo URL for cloning on other machines.
    """
    import subprocess

    root_candidate = cfg.memory_dir.parent
    if not (root_candidate / ".git").exists():
        # First-time setup: initialize the local git repo so git_root_for() works
        # and gh repo create --source can push. Via _git so a failure/timeout
        # raises SyncGitError instead of CalledProcessError/TimeoutExpired.
        _git(root_candidate, "init")

    root = git_root_for(cfg)

    owner = "private" if private else "public"
    repo_name = "memo-sync"

    try:
        gh_cmd = subprocess.run(
            ["gh", "repo", "create", repo_name, f"--{owner}", "--source", str(root), "--push"],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except OSError as exc:
        raise SyncGitError(
            "gh CLI not found — install it: https://cli.github.com or `brew install gh`"
        ) from exc
    if gh_cmd.returncode != 0:
        raise SyncGitError(f"gh repo create failed: {gh_cmd.stderr.strip()}")

    remote_url = ""
    for line in gh_cmd.stdout.split("\n"):
        if "github.com/" in line:
            remote_url = line.strip()
            break

    if not remote_url:
        remote_url = _git(root, "remote", "get-url", "origin").stdout.strip()

    return {"repo_url": remote_url, "branch": _current_branch(root), "local_dir": str(root)}
