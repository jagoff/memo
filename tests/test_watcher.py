"""File-watcher — debounce coalescing + plist render."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

from memo.repo_watcher import DebouncedRepoRefresh, _watch_target
from memo.watcher import _PLIST_LABEL, _DebouncedReindex, render_plist


class _SpyMemory:
    def __init__(self) -> None:
        self.calls: list[float] = []
        self.lock = threading.Lock()

    def reindex(self) -> dict[str, int]:
        with self.lock:
            self.calls.append(time.perf_counter())
        return {"checked": 1, "reindexed": 0, "added": 0, "skipped": 0}


def test_debounce_coalesces_burst() -> None:
    spy = _SpyMemory()
    deb = _DebouncedReindex(spy, delay=0.2)
    for _ in range(10):
        deb.schedule(Path("/tmp/x.md"))
        time.sleep(0.02)  # bursts within the debounce window
    time.sleep(0.5)  # wait past the debounce
    assert len(spy.calls) == 1


def test_debounce_fires_again_after_window() -> None:
    spy = _SpyMemory()
    deb = _DebouncedReindex(spy, delay=0.15)
    deb.schedule(Path("/tmp/a.md"))
    time.sleep(0.3)
    deb.schedule(Path("/tmp/b.md"))
    time.sleep(0.3)
    assert len(spy.calls) == 2


def test_debounce_swallows_reindex_errors() -> None:
    class Boom:
        def reindex(self) -> dict[str, int]:
            raise RuntimeError("disk full")

    deb = _DebouncedReindex(Boom(), delay=0.1)
    deb.schedule(Path("/tmp/x.md"))
    time.sleep(0.25)
    # No exception leaked to the caller's thread — the debouncer runs
    # on its own timer thread and swallows the failure.


def test_render_plist_contains_label_and_bin(monkeypatch: Any, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("MEMO_DATA_DIR", "/tmp/memo-data")
    xml = render_plist("/usr/local/bin/memo")
    assert _PLIST_LABEL in xml
    assert "/usr/local/bin/memo" in xml
    assert "<string>watch</string>" in xml
    assert "MEMO_DATA_DIR" in xml
    assert "/tmp/memo-data" in xml
    assert "KeepAlive" in xml


def test_render_plist_log_dir_created(monkeypatch: Any, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    render_plist("/usr/bin/memo")
    assert (tmp_path / "Library" / "Logs" / "memo").is_dir()


def test_repo_debounce_refreshes_incrementally_and_ignores_git(tmp_path: Path) -> None:
    class RepoSpy:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict[str, Any]]] = []

        def repo_index(self, url: str, **kwargs: Any) -> dict[str, Any]:
            self.calls.append((url, kwargs))
            return {"indexed_files": 1, "deleted_files": 0, "unchanged_files": 2}

    spy = RepoSpy()
    source = {
        "url": "https://example/repo.git",
        "name": "repo",
        "ref": "HEAD",
        "extra": {"include": ["*.py"], "exclude": ["tmp/*"], "max_file_bytes": 1000},
    }
    debounce = DebouncedRepoRefresh(spy, source, delay=0.05)
    debounce.schedule(tmp_path / ".git" / "index")
    debounce.schedule(tmp_path / "src" / "a.py")
    debounce.schedule(tmp_path / "src" / "b.py")
    time.sleep(0.2)

    assert len(spy.calls) == 1
    url, kwargs = spy.calls[0]
    assert url == source["url"]
    assert kwargs["refresh"] is True
    assert kwargs["include"] == ["*.py"]


def test_repo_debounce_accepts_git_commit_ref_signal(tmp_path: Path) -> None:
    class RepoSpy:
        def __init__(self) -> None:
            self.calls = 0

        def repo_index(self, url: str, **kwargs: Any) -> dict[str, Any]:
            del url, kwargs
            self.calls += 1
            return {"indexed_files": 0, "deleted_files": 0, "unchanged_files": 3}

    spy = RepoSpy()
    debounce = DebouncedRepoRefresh(
        spy,
        {"url": str(tmp_path), "name": "repo", "ref": "HEAD", "extra": {}},
        delay=0.05,
    )
    debounce.schedule(tmp_path / ".git" / "index")
    debounce.schedule(tmp_path / ".git" / "refs" / "heads" / "master")
    time.sleep(0.2)

    assert spy.calls == 1


def test_repo_watcher_prefers_local_source_checkout(tmp_path: Path) -> None:
    source = tmp_path / "source"
    clone = tmp_path / "managed-clone"
    source.mkdir()
    clone.mkdir()

    target = _watch_target(
        {"url": str(source), "clone_path": str(clone), "name": "repo"},
        "repo",
    )

    assert target == source.resolve()
