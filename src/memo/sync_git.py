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


def sync_push(cfg: Config, store: VecStore, *, remote: str = "origin") -> dict:
    """Export signal, commit any changes, and push. Returns a summary dict."""
    root = git_root_for(cfg)
    branch = _current_branch(root)
    export_signal(store, signal_dir_for(cfg))

    _git(root, "add", "-A")
    staged = _git(root, "diff", "--cached", "--name-only").stdout.strip()
    if not staged:
        return {"pushed": False, "reason": "nothing to commit", "branch": branch}

    n_files = len(staged.splitlines())
    _git(root, "commit", "-m", f"sync: memo signal + memorias ({n_files} files)")

    # push; set upstream on first push
    push = _git(root, "push", remote, branch, check=False)
    if push.returncode != 0:
        push = _git(root, "push", "-u", remote, branch, check=False)
        if push.returncode != 0:
            raise SyncGitError(f"git push failed: {push.stderr.strip()}")
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
