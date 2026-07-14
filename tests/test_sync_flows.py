"""Regression tests for sync flow: gh-missing, flock, no-remote, CLI degradation.

Complements tests/test_sync_git.py (which covers the full git round-trip).
These tests focus on CLI-level degradation paths: what happens when there is
no git remote, when the machine lock is held by a concurrent session, and when
the ``gh`` CLI is absent on a fresh machine.
"""

from __future__ import annotations

import fcntl
import subprocess
import threading
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from memo.cli import cli
from memo.config import Config


def _env(tmp_path: Path, **extra: str) -> dict:
    """Minimal isolated env for sync CLI tests (no MLX, no real config)."""
    return {
        "MEMO_CONFIG_FILE": str(tmp_path / "memo-config.toml"),
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_DATA_DIR": str(tmp_path / "data"),
        "MEMO_STATE_DIR": str(tmp_path / "state"),
        "MEMO_EMBEDDER_DIMS": "4",
        **extra,
    }


# ---------------------------------------------------------------------------
# sync status — no git clone
# ---------------------------------------------------------------------------


def test_sync_status_no_remote_gives_actionable_message(tmp_path: Path) -> None:
    """If data_dir is not a git clone, ``memo sync status`` must explain clearly,
    not traceback. The message should name the fix (bootstrap / clone)."""
    (tmp_path / "data").mkdir()
    (tmp_path / "state").mkdir()

    result = CliRunner().invoke(cli, ["sync", "status"], env=_env(tmp_path))

    assert result.exit_code == 0, f"exit={result.exit_code}\n{result.output}"
    assert "Traceback" not in result.output, f"Traceback in output:\n{result.output}"
    output_lower = result.output.lower()
    # Check for phrases that are actually in the output (ANSI-stripped keywords
    # that appear contiguously in the text). The words come from cli_sync.py and
    # the SyncGitError message from git_root_for():
    #   "NOT syncing — data_dir is not a git clone."
    #   "…is not a git repo (no .git). Run `memo sync clone <url>`…"
    #   "Fix: memo sync bootstrap <url>"
    has_useful_message = any(
        kw in output_lower
        for kw in ["not syncing", "not a git clone", "git clone", "bootstrap", "not a git repo"]
    )
    assert has_useful_message, f"No actionable message in:\n{result.output}"


# ---------------------------------------------------------------------------
# sync once — no remote, with flock held
# ---------------------------------------------------------------------------


def test_sync_once_concurrent_skips_gracefully(tmp_path: Path) -> None:
    """Concurrent ``memo sync once`` calls: the command must return quickly
    (exit 0) even when the machine lock file is held by another process.

    Because data_dir is not a git clone, the command short-circuits before it
    ever touches the lock (``sync_tier`` → "local"). This guarantees no
    blocking regardless of the lock state — the flock path is covered at the
    API level by test_sync_once_skips_when_lock_held in test_sync_git.py.
    """
    (tmp_path / "data").mkdir()
    (tmp_path / "state").mkdir()
    lock_path = tmp_path / "state" / ".sync.lock"
    lock_path.touch()

    results: list[int] = []

    def run_sync() -> None:
        r = CliRunner().invoke(cli, ["sync", "once"], env=_env(tmp_path))
        results.append(r.exit_code)

    # Hold the flock in the main thread to simulate a concurrent session
    lock_fd = lock_path.open("w")
    fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX)
    try:
        t = threading.Thread(target=run_sync)
        t.start()
        t.join(timeout=5)
    finally:
        fcntl.flock(lock_fd.fileno(), fcntl.LOCK_UN)
        lock_fd.close()

    assert not t.is_alive(), "sync once blocked instead of returning quickly"
    assert results, "sync once returned no result"
    assert results[0] == 0, f"sync once non-zero exit: {results[0]}"


def test_sync_once_no_remote_skips_with_message(tmp_path: Path) -> None:
    """Non-git data_dir → ``sync once`` exits 0 and says it skipped."""
    (tmp_path / "data").mkdir()
    (tmp_path / "state").mkdir()

    result = CliRunner().invoke(cli, ["sync", "once"], env=_env(tmp_path))

    assert result.exit_code == 0, f"exit={result.exit_code}\n{result.output}"
    assert "Traceback" not in result.output
    output_lower = result.output.lower()
    # "sync once skipped: local tier (no remote / not a clone)"
    assert "skipped" in output_lower or "local" in output_lower, (
        f"Expected skip message in:\n{result.output}"
    )


# ---------------------------------------------------------------------------
# sync init — gh CLI missing
# ---------------------------------------------------------------------------


def test_sync_init_gh_missing_raises_clear_error(tmp_path: Path) -> None:
    """``memo sync init`` must give a clear ClickException when ``gh`` is not
    installed — not a raw FileNotFoundError traceback.

    Regression for: sync_init_home called subprocess.run(["gh", ...]) without
    catching OSError, leaking a FileNotFoundError to the caller.
    """
    # Create a minimal git-like layout so git_root_for() passes its .git check
    # without a real git init (it only checks dir existence).
    data_dir = tmp_path / "memorias"
    data_dir.mkdir()
    (tmp_path / ".git").mkdir()  # git_root_for checks (data_dir.parent / ".git").exists()
    (tmp_path / "state").mkdir()

    env = {
        "MEMO_CONFIG_FILE": str(tmp_path / "memo-config.toml"),
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_DATA_DIR": str(data_dir),
        "MEMO_STATE_DIR": str(tmp_path / "state"),
        "MEMO_EMBEDDER_DIMS": "4",
    }

    _real_run = subprocess.run

    def _raise_for_gh(args, **kwargs):
        if args and args[0] == "gh":
            raise FileNotFoundError("No such file or directory: 'gh'")
        return _real_run(args, **kwargs)

    with patch("subprocess.run", side_effect=_raise_for_gh):
        result = CliRunner().invoke(cli, ["sync", "init"], env=env)

    # Must give a Click-level error message, not a raw Python traceback
    assert "Traceback" not in result.output, f"Raw traceback from missing gh:\n{result.output}"
    # Should mention gh and the problem clearly
    output_lower = result.output.lower()
    has_clear_error = any(kw in output_lower for kw in ["gh", "not found", "install", "error"])
    assert has_clear_error, f"No clear error about missing gh in:\n{result.output}"
    assert result.exit_code != 0, "Should exit non-zero when gh is missing"


# ---------------------------------------------------------------------------
# sync auto — no remote
# ---------------------------------------------------------------------------


def test_sync_auto_no_remote_exits_cleanly(tmp_path: Path) -> None:
    """``memo sync auto`` must exit 0 silently when data_dir is not a git clone."""
    (tmp_path / "data").mkdir()
    (tmp_path / "state").mkdir()

    result = CliRunner().invoke(cli, ["sync", "auto", "--json"], env=_env(tmp_path))

    assert result.exit_code == 0, f"exit={result.exit_code}\n{result.output}"
    assert "Traceback" not in result.output
    # The json output skips before reaching local-tier check when timestamps are
    # fresh enough or MEMO_SYNC_AUTO=0 — but with no timestamps the debounce
    # fires and reaches the tier check → skipped: "local tier (no remote...)"
    output_lower = result.output.lower()
    assert "traceback" not in output_lower


# ---------------------------------------------------------------------------
# sync setup — guided onboarding wizard
# ---------------------------------------------------------------------------


def test_sync_setup_never_writes_dismiss_stamp(tmp_path):
    from click.testing import CliRunner

    from memo.cli import cli

    env = {
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_DATA_DIR": str(tmp_path / "memorias"),
        "MEMO_STATE_DIR": str(tmp_path / "state"),
    }
    (tmp_path / "memorias").mkdir()
    r = CliRunner().invoke(cli, ["sync", "setup", "--never"], env=env)
    assert r.exit_code == 0
    assert (tmp_path / "state" / ".sync_nudge_dismissed").is_file()


def test_sync_setup_noninteractive_prints_hint_no_prompt(tmp_path):
    from click.testing import CliRunner

    from memo.cli import cli

    env = {
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_DATA_DIR": str(tmp_path / "memorias"),
        "MEMO_STATE_DIR": str(tmp_path / "state"),
    }
    (tmp_path / "memorias").mkdir()
    r = CliRunner().invoke(cli, ["sync", "setup"], env=env)
    assert r.exit_code == 0
    assert "memo sync setup" in r.output  # hint mentions the command


def test_run_sync_setup_create_with_gh_calls_init_home(tmp_path, monkeypatch):
    import memo.cli_sync as cs

    calls = {}

    def fake_sync_init_home(cfg, private=True):
        calls["init"] = {"private": private}
        return {"repo_url": "u", "branch": "main", "local_dir": "d"}

    monkeypatch.setattr(cs, "sync_init_home", fake_sync_init_home)
    cfg = Config(data_dir=tmp_path / "memorias", state_dir=tmp_path / "state", embedder_dims=4)
    out = cs._run_sync_setup(cfg, "1", None, gh_ok=True)
    assert calls["init"]["private"] is True
    assert out["repo_url"] == "u"


def test_run_sync_setup_create_without_gh_calls_byo(tmp_path, monkeypatch):
    import memo.cli_sync as cs

    calls = {}

    def fake_sync_init_home_byo(cfg, url):
        calls["url"] = url
        return {"repo_url": url, "branch": "main", "local_dir": "d"}

    monkeypatch.setattr(cs, "sync_init_home_byo", fake_sync_init_home_byo)
    cfg = Config(data_dir=tmp_path / "memorias", state_dir=tmp_path / "state", embedder_dims=4)
    out = cs._run_sync_setup(cfg, "1", "https://example.com/x.git", gh_ok=False)
    assert calls["url"] == "https://example.com/x.git"
    assert out["repo_url"] == "https://example.com/x.git"


def test_run_sync_setup_join_calls_bootstrap(tmp_path, monkeypatch):
    import memo.cli_sync as cs

    calls = {}

    def fake_bootstrap_clone(url, dest, config_path=None):
        calls["url"] = url
        return {"cloned": str(dest), "memories": 0, "memories_dir": str(dest / "memorias")}

    monkeypatch.setattr(cs, "bootstrap_clone", fake_bootstrap_clone)

    class _FakeStore:
        @staticmethod
        def rebuild_feedback_vecs(*a, **k):
            return 0

    class _FakeMem:
        store = _FakeStore()

        class embedder:
            @staticmethod
            def embed_query(*a, **k):
                return []

        def reindex(self, rebuild=False):
            return 0

    monkeypatch.setattr(cs, "_get_memory", lambda cfg: _FakeMem())
    monkeypatch.setattr(cs, "import_signal", lambda *a, **k: 0)
    monkeypatch.setattr(cs, "signal_dir_for", lambda *a, **k: tmp_path)
    monkeypatch.setattr(
        cs.Config,
        "from_env",
        staticmethod(
            lambda **kw: Config(
                data_dir=tmp_path / "memorias", state_dir=tmp_path / "state", embedder_dims=4
            )
        ),
    )

    cfg = Config(data_dir=tmp_path / "memorias", state_dir=tmp_path / "state", embedder_dims=4)
    out = cs._run_sync_setup(cfg, "2", "https://example.com/memo-sync.git", gh_ok=False)
    assert calls["url"] == "https://example.com/memo-sync.git"
    assert out["reindexed"] == 0


def test_run_sync_setup_cancel_returns_none(tmp_path):
    import memo.cli_sync as cs

    cfg = Config(data_dir=tmp_path / "memorias", state_dir=tmp_path / "state", embedder_dims=4)
    assert cs._run_sync_setup(cfg, "3", None, gh_ok=False) is None
