"""F4 — git-backed cross-machine sync (push/pull round trip).

Uses a local bare remote + two clones to prove a memoria + its signal saved
on "Mac A" reach "Mac B" via push → pull, with the signal merged into B's DB.
No network, no MLX (embedder stubbed to 4-dim).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from memo.config import Config
from memo.memory import Memory
from memo.sync_git import (
    SyncGitError,
    clone_bootstrap,
    git_root_for,
    sync_once,
    sync_pull,
    sync_push,
    sync_status,
    sync_tier,
)


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, text=True)


def _stub_embed(self, inputs):
    out = []
    for s in inputs:
        h = sum(ord(c) for c in s) % 4
        v = [0.0] * 4
        v[h] = 1.0
        out.append(v)
    return out


def _make_clone(remote: Path, where: Path) -> Path:
    subprocess.run(
        ["git", "clone", str(remote), str(where)], check=True, capture_output=True, text=True
    )
    _git(where, "config", "user.email", "t@t.t")
    _git(where, "config", "user.name", "t")
    (where / "memorias").mkdir(exist_ok=True)
    return where


def _mem_for(clone: Path, state: Path, monkeypatch) -> Memory:
    monkeypatch.setattr("memo.embedder.MLXEmbedder.embed", _stub_embed)
    cfg = Config(
        data_dir=clone / "memorias",
        state_dir=state,
        embedder_dims=4,
        embedder_model="stub",
        reranker_enabled=False,
    )
    return Memory(cfg)


@pytest.fixture
def remote(tmp_path: Path) -> Path:
    r = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--bare", "-b", "main", str(r)], check=True, capture_output=True)
    # seed an initial commit so clones share history
    seed = tmp_path / "seed"
    subprocess.run(["git", "clone", str(r), str(seed)], check=True, capture_output=True)
    _git(seed, "config", "user.email", "t@t.t")
    _git(seed, "config", "user.name", "t")
    (seed / "memorias").mkdir()
    (seed / "memorias" / ".gitkeep").write_text("")
    _git(seed, "add", "-A")
    _git(seed, "commit", "-m", "seed")
    _git(seed, "push", "origin", "main")
    return r


def test_git_root_requires_git(tmp_path: Path):
    cfg = Config(data_dir=tmp_path / "memorias", state_dir=tmp_path / "state", embedder_dims=4)
    (tmp_path / "memorias").mkdir()
    with pytest.raises(SyncGitError, match="not a git repo"):
        git_root_for(cfg)


def test_push_clean_repo_is_noop(remote: Path, tmp_path: Path, monkeypatch):
    clone = _make_clone(remote, tmp_path / "A")
    mem = _mem_for(clone, tmp_path / "stateA", monkeypatch)
    try:
        # first push writes the signal/*.json files (new) → commits
        first = sync_push(mem.cfg, mem.store)
        assert first["pushed"] is True
        # second push: identical signal, nothing changed → noop
        second = sync_push(mem.cfg, mem.store)
        assert second["pushed"] is False
    finally:
        mem.close()


def test_push_then_pull_propagates_memoria_and_signal(remote: Path, tmp_path: Path, monkeypatch):
    # --- Mac A: save a memoria, build signal, push ---
    clone_a = _make_clone(remote, tmp_path / "A")
    mem_a = _mem_for(clone_a, tmp_path / "stateA", monkeypatch)
    try:
        rec = mem_a.save(content="cross-mac body", title="Crossmac")
        mem_a.store.touch([rec.id], ts="2026-06-01T00:00:00+00:00")
        mem_a.store.boost_roi_batch([rec.id])
        out = sync_push(mem_a.cfg, mem_a.store)
        assert out["pushed"] is True
        rec_id = rec.id
    finally:
        mem_a.close()

    # --- Mac B: pull, expect the memoria + its access signal ---
    clone_b = _make_clone(remote, tmp_path / "B")
    mem_b = _mem_for(clone_b, tmp_path / "stateB", monkeypatch)
    try:
        res = sync_pull(mem_b.cfg, mem_b.store, mem_b)
        assert res["pulled"] is True
        # memoria .md arrived and indexed
        got = mem_b.get(rec_id)
        assert got is not None
        # signal merged into B's DB
        assert mem_b.store.get_access(rec_id)["access_count"] == 1
        assert rec_id in mem_b.store.get_health_batch([rec_id])
    finally:
        mem_b.close()


def test_clone_bootstrap(remote: Path, tmp_path: Path, monkeypatch):
    # seed remote with a memoria so the clone has content
    clone_a = _make_clone(remote, tmp_path / "A")
    mem_a = _mem_for(clone_a, tmp_path / "stateA", monkeypatch)
    try:
        mem_a.save(content="seed body", title="Seed")
        sync_push(mem_a.cfg, mem_a.store)
    finally:
        mem_a.close()

    dest = tmp_path / "fresh"
    out = clone_bootstrap(str(remote), dest)
    assert out["memories"] == 1
    assert Path(out["memories_dir"]) == dest / "memorias"


def test_clone_refuses_nonempty_dest(remote: Path, tmp_path: Path):
    dest = tmp_path / "occupied"
    dest.mkdir()
    (dest / "junk").write_text("x")
    with pytest.raises(SyncGitError, match="not empty"):
        clone_bootstrap(str(remote), dest)


def test_sync_status_reports_clone_state(remote: Path, tmp_path: Path, monkeypatch):
    clone = _make_clone(remote, tmp_path / "A")
    mem = _mem_for(clone, tmp_path / "stateA", monkeypatch)
    try:
        st = sync_status(mem.cfg)
        assert st["is_git_clone"] is True
        assert st["branch"] == "main"
        assert st["ahead"] == 0 and st["pending"] is False
        # a save leaves the working tree dirty until the next push
        mem.save(content="status body here", title="StatusX")
        assert sync_status(mem.cfg)["dirty_files"] > 0
    finally:
        mem.close()


def test_sync_status_not_a_clone(tmp_path: Path):
    cfg = Config(data_dir=tmp_path / "memorias", state_dir=tmp_path / "state", embedder_dims=4)
    (tmp_path / "memorias").mkdir()
    st = sync_status(cfg)
    assert st["is_git_clone"] is False


def test_sync_push_retries_stranded_commit(remote: Path, tmp_path: Path, monkeypatch):
    """A commit that landed locally but never pushed (offline/auth) must be
    pushed by the next sync_push even when there's nothing new to commit."""
    clone = _make_clone(remote, tmp_path / "A")
    mem = _mem_for(clone, tmp_path / "stateA", monkeypatch)
    try:
        sync_push(mem.cfg, mem.store)  # initial push
        # simulate a stranded local commit (committed, not pushed)
        (clone / "memorias" / "stray.md").write_text("---\nid: y\n---\nstray\n")
        _git(clone, "add", "-A")
        _git(clone, "commit", "-m", "stranded")
        assert sync_status(mem.cfg)["ahead"] == 1
        # nothing NEW to commit, but ahead>0 → push must still fire
        out = sync_push(mem.cfg, mem.store)
        assert out["pushed"] is True
        assert sync_status(mem.cfg)["ahead"] == 0
    finally:
        mem.close()


def test_sync_once_commits_local_deletion_before_pull(remote: Path, tmp_path: Path, monkeypatch):
    """Regression: an uncommitted local .md deletion must survive sync_once
    (commit-before-pull), not be resurrected by rebase --autostash from origin."""
    clone = _make_clone(remote, tmp_path / "A")
    mem = _mem_for(clone, tmp_path / "stateA", monkeypatch)
    try:
        rec = mem.save(content="to be deleted cross-mac", title="DeleteMe")
        sync_once(mem.cfg, mem.store, mem)  # push it to origin
        md = next((clone / "memorias").rglob("*.md"))
        # simulate `memo delete`'s file removal WITHOUT committing
        md.unlink()
        assert not md.exists()
        out = sync_once(mem.cfg, mem.store, mem)
        assert out["tier"] == "remote"
        # deletion stuck locally (not resurrected) and reached origin
        assert not md.exists(), "deletion was resurrected by the pull/rebase"
        _git(clone, "fetch", "origin", "main")
        tree = subprocess.run(
            ["git", "-C", str(clone), "ls-tree", "-r", "origin/main", "--name-only"],
            capture_output=True,
            text=True,
        ).stdout
        assert md.name not in tree, "deleted memoria still on origin"
        _ = rec
    finally:
        mem.close()


def test_sync_pull_prunes_remote_deletion(remote: Path, tmp_path: Path, monkeypatch):
    """Receiver side: a memoria deleted on Mac A must DISAPPEAR from Mac B's
    index after pull — reindex only adds, so sync_pull's gc(fix=True) must drop
    the orphan row whose .md the pull removed."""
    # A: save + push a memoria
    clone_a = _make_clone(remote, tmp_path / "A")
    mem_a = _mem_for(clone_a, tmp_path / "stateA", monkeypatch)
    rec_id = None
    try:
        rec_id = mem_a.save(content="cross-mac delete target", title="DelTarget").id
        sync_once(mem_a.cfg, mem_a.store, mem_a)
    finally:
        mem_a.close()
    # B: pull → has it
    clone_b = _make_clone(remote, tmp_path / "B")
    mem_b = _mem_for(clone_b, tmp_path / "stateB", monkeypatch)
    try:
        sync_pull(mem_b.cfg, mem_b.store, mem_b)
        assert mem_b.get(rec_id) is not None
    finally:
        mem_b.close()
    # A: delete it + push the deletion
    mem_a2 = _mem_for(clone_a, tmp_path / "stateA", monkeypatch)
    try:
        mem_a2.delete(rec_id)
        sync_once(mem_a2.cfg, mem_a2.store, mem_a2)
    finally:
        mem_a2.close()
    # B: pull again → it must be GONE from B's index (pruned), not an orphan
    mem_b2 = _mem_for(clone_b, tmp_path / "stateB", monkeypatch)
    try:
        out = sync_pull(mem_b2.cfg, mem_b2.store, mem_b2)
        assert out["pruned"] >= 1
        assert mem_b2.get(rec_id) is None, "deleted memoria still in receiver's index"
    finally:
        mem_b2.close()


def test_cli_pull_quiet_softfails_on_non_git(tmp_path: Path):
    """The SessionStart hook calls `memo sync pull --quiet`; a non-git install
    must exit 0 (not break the session)."""
    from click.testing import CliRunner

    from memo.cli import cli

    data = tmp_path / "data"
    data.mkdir()
    env = {
        "MEMO_CONFIG_FILE": str(tmp_path / "memo-config.toml"),
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_DATA_DIR": str(data),
        "MEMO_STATE_DIR": str(tmp_path / "state"),
        "MEMO_EMBEDDER_DIMS": "4",
        "MEMO_EMBEDDER_MODEL": "stub",
    }
    r = CliRunner().invoke(cli, ["sync", "pull", "--quiet"], env=env)
    assert r.exit_code == 0, r.output
    assert "skipped" in r.output.lower()

    # without --quiet it surfaces the error (non-zero)
    r2 = CliRunner().invoke(cli, ["sync", "pull"], env=env)
    assert r2.exit_code != 0


def test_sync_auto_throttle_and_disable(remote: Path, tmp_path: Path):
    """sync auto: cheap no-op when not due (no Memory/MLX built), and a hard
    skip when MEMO_SYNC_AUTO=0."""
    import json
    import time

    from click.testing import CliRunner

    from memo.cli import cli

    clone = _make_clone(remote, tmp_path / "A")
    state = tmp_path / "stateA"
    state.mkdir()
    env = {
        "MEMO_CONFIG_FILE": str(tmp_path / "memo-config.toml"),
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_DATA_DIR": str(clone / "memorias"),
        "MEMO_STATE_DIR": str(state),
        "MEMO_EMBEDDER_DIMS": "4",
        "MEMO_EMBEDDER_MODEL": "stub",
    }
    # pre-stamp recent timestamps → neither pull nor push due → returns before
    # building Memory (so no MLX needed).
    (state / ".sync_auto_ts").write_text(
        json.dumps({"last_pull": time.time(), "last_push": time.time()})
    )
    r = CliRunner().invoke(cli, ["sync", "auto", "--json"], env=env)
    assert r.exit_code == 0, r.output
    assert '"skipped": "not due"' in r.output

    r2 = CliRunner().invoke(cli, ["sync", "auto", "--json"], env={**env, "MEMO_SYNC_AUTO": "0"})
    assert r2.exit_code == 0
    assert '"skipped": "disabled"' in r2.output


def test_pull_aborts_on_memoria_conflict(remote: Path, tmp_path: Path, monkeypatch):
    """Same memoria path edited divergently on A and B → rebase aborts + reports."""
    # Both clone from the shared seed FIRST so they truly diverge.
    clone_a = _make_clone(remote, tmp_path / "A")
    clone_b = _make_clone(remote, tmp_path / "B")

    # A creates the file and pushes
    (clone_a / "memorias" / "note.md").write_text("---\nid: x\n---\nA version\n")
    _git(clone_a, "add", "-A")
    _git(clone_a, "commit", "-m", "A note")
    _git(clone_a, "push", "origin", "main")

    # B makes a conflicting commit on the same path (without pulling A's)
    (clone_b / "memorias" / "note.md").write_text("---\nid: x\n---\nB version\n")
    _git(clone_b, "add", "-A")
    _git(clone_b, "commit", "-m", "B note")

    mem_b = _mem_for(clone_b, tmp_path / "stateB", monkeypatch)
    try:
        with pytest.raises(SyncGitError, match="manual resolution"):
            sync_pull(mem_b.cfg, mem_b.store, mem_b)
        # repo left clean (rebase aborted), not mid-rebase
        assert not (clone_b / ".git" / "rebase-merge").exists()
    finally:
        mem_b.close()


def test_pull_resolves_multi_commit_signal_conflicts_headless(
    remote: Path, tmp_path: Path, monkeypatch
):
    """Regression: a Mac with MULTIPLE local commits that each conflict on
    signal/*.json must rebase cleanly in a headless env (no usable editor).

    Reproduces the two bugs that left a Mac silently diverged (behind forever):
      1. `rebase --continue` opened an editor → "Terminal is dumb, EDITOR unset".
      2. only the FIRST conflict stop was resolved; a 2nd conflicting commit left
         the rebase stuck → aborted → `sync_once` swallowed it as a no-op.
    """
    clone_a = _make_clone(remote, tmp_path / "A")
    clone_b = _make_clone(remote, tmp_path / "B")

    # A advances origin with two signal files
    (clone_a / "signal").mkdir(exist_ok=True)
    (clone_a / "signal" / "access.json").write_text('{"v": "A"}\n')
    (clone_a / "signal" / "memory_health.json").write_text('{"v": "A"}\n')
    _git(clone_a, "add", "-A")
    _git(clone_a, "commit", "-m", "A signal")
    _git(clone_a, "push", "origin", "main")

    # B makes TWO local commits, each conflicting on a DIFFERENT signal file →
    # the rebase stops twice (the bug only handled the first stop).
    (clone_b / "signal").mkdir(exist_ok=True)
    (clone_b / "signal" / "access.json").write_text('{"v": "B1"}\n')
    _git(clone_b, "add", "-A")
    _git(clone_b, "commit", "-m", "B signal 1")
    (clone_b / "signal" / "memory_health.json").write_text('{"v": "B2"}\n')
    _git(clone_b, "add", "-A")
    _git(clone_b, "commit", "-m", "B signal 2")

    # Force "no usable editor": without the production override `rebase --continue`
    # would run `false` and fail fast (proving the bug) instead of hanging on a
    # real editor. The fix sets GIT_EDITOR=true in git's env regardless of this.
    monkeypatch.setenv("GIT_EDITOR", "false")
    monkeypatch.setenv("GIT_SEQUENCE_EDITOR", "false")

    mem_b = _mem_for(clone_b, tmp_path / "stateB", monkeypatch)
    try:
        res = sync_pull(mem_b.cfg, mem_b.store, mem_b)
        assert res["pulled"] is True
        # rebase finished — not stuck mid-rebase
        assert not (clone_b / ".git" / "rebase-merge").exists()
        assert not (clone_b / ".git" / "rebase-apply").exists()
        # divergence reconciled: origin/main is now an ancestor of B's HEAD
        _git(clone_b, "fetch", "origin", "main")
        anc = subprocess.run(
            ["git", "-C", str(clone_b), "merge-base", "--is-ancestor", "origin/main", "HEAD"],
            capture_output=True,
            text=True,
        )
        assert anc.returncode == 0, "origin/main not in B's history after rebase"
    finally:
        mem_b.close()


def test_sync_tier_remote_clone_vs_local_plain_dir(remote: Path, tmp_path: Path, monkeypatch):
    """`sync_tier` is "remote" only when a git remote is configured; a plain
    (non-git) data dir is "local" and never raises."""
    clone = _make_clone(remote, tmp_path / "A")
    mem = _mem_for(clone, tmp_path / "stateA", monkeypatch)
    try:
        assert sync_tier(mem.cfg) == "remote"
    finally:
        mem.close()

    plain = Config(
        data_dir=tmp_path / "plain",
        state_dir=tmp_path / "state_plain",
        embedder_dims=4,
    )
    (tmp_path / "plain").mkdir()
    assert sync_tier(plain) == "local"


def test_sync_once_pull_rebase_before_push(remote: Path, tmp_path: Path, monkeypatch):
    """Mac A has a divergent local commit while the remote advanced (Mac B
    pushed). `sync_once` on A must pull-rebase A's commit onto B's tip and then
    push without rejection — A ends with ahead==0 and B's memoria indexed."""
    clone_a = _make_clone(remote, tmp_path / "A")
    clone_b = _make_clone(remote, tmp_path / "B")

    # --- Mac B advances the remote with a new memoria ---
    mem_b = _mem_for(clone_b, tmp_path / "stateB", monkeypatch)
    try:
        rec_b_id = mem_b.save(content="from mac B", title="FromB").id
        assert sync_push(mem_b.cfg, mem_b.store)["pushed"] is True
    finally:
        mem_b.close()

    # --- Mac A makes a divergent local commit (committed, not pushed) ---
    mem_a = _mem_for(clone_a, tmp_path / "stateA", monkeypatch)
    try:
        rec_a_id = mem_a.save(content="from mac A", title="FromA").id
        _git(clone_a, "add", "-A")
        _git(clone_a, "commit", "-m", "A local memoria")
        # A is ahead of its last-known remote and unaware of B's commit
        assert sync_status(mem_a.cfg)["ahead"] == 1

        res = sync_once(mem_a.cfg, mem_a.store, mem_a)
        assert res["tier"] == "remote"
        assert res["pulled"] is True
        assert res["pushed"] is True

        # rebased onto B's tip + pushed cleanly → nothing stranded
        assert sync_status(mem_a.cfg)["ahead"] == 0
        # B's memoria arrived via the pull-reindex; A's own survived the rebase
        assert mem_a.get(rec_b_id) is not None
        assert mem_a.get(rec_a_id) is not None
    finally:
        mem_a.close()


def test_sync_once_skips_when_lock_held(remote: Path, tmp_path: Path, monkeypatch):
    """A held machine lock makes `sync_once` skip without doing git mutation —
    the concurrent sibling's writes are already in the shared store, so the lock
    holder carries them."""
    import fcntl

    clone = _make_clone(remote, tmp_path / "A")
    mem = _mem_for(clone, tmp_path / "stateA", monkeypatch)
    try:
        lock_path = mem.cfg.state_dir / ".sync.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        fh = lock_path.open("w")
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            res = sync_once(mem.cfg, mem.store, mem)
            # exact shape proves the pull/push branch never ran
            assert res == {"tier": "remote", "skipped": "locked"}
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            fh.close()
    finally:
        mem.close()


def test_sync_push_commit_carries_identity_label(remote: Path, tmp_path: Path, monkeypatch):
    """A commit produced by the sync path is attributed to the identity label —
    the message ends with `[<label>]`."""
    clone = _make_clone(remote, tmp_path / "A")
    mem = _mem_for(clone, tmp_path / "stateA", monkeypatch)
    try:
        assert sync_push(mem.cfg, mem.store)["pushed"] is True
        subject = subprocess.run(
            ["git", "-C", str(clone), "log", "-1", "--format=%s"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        assert "[" in subject and "]" in subject
    finally:
        mem.close()
