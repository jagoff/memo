"""File-watcher — debounce coalescing + plist render."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

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
    assert "MEMO_NONINTERACTIVE" in xml
    assert "<string>1</string>" in xml
    assert "KeepAlive" in xml


def test_render_plist_log_dir_created(monkeypatch: Any, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    render_plist("/usr/bin/memo")
    assert (tmp_path / "Library" / "Logs" / "memo").is_dir()
