"""Incremental repository watcher with debounced, deduplicable refresh jobs."""

from __future__ import annotations

import signal
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from memo.errors import MemoError


class DebouncedRepoRefresh:
    """Coalesce repository file events into one hash-incremental refresh."""

    def __init__(
        self,
        memory: Any,
        source: dict[str, Any],
        *,
        delay: float = 1.0,
        debug: bool = False,
    ) -> None:
        self.memory = memory
        self.source = source
        self.delay = max(0.01, float(delay))
        self.debug = debug
        self._timer: threading.Timer | None = None
        self._lock = threading.Lock()
        self._pending_paths: set[str] = set()
        self._running = False

    def schedule(self, path: Path) -> None:
        value = str(path)
        if "/.git/" in value.replace("\\", "/"):
            return
        with self._lock:
            self._pending_paths.add(value)
            if self._timer is not None:
                self._timer.cancel()
            self._arm_locked()
        if self.debug:
            print(f"# memo repo watch: queued {path}", file=sys.stderr)

    def _arm_locked(self) -> None:
        self._timer = threading.Timer(self.delay, self._run)
        self._timer.daemon = True
        self._timer.start()

    def _run(self) -> None:
        with self._lock:
            if not self._pending_paths:
                return
            if self._running:
                self._arm_locked()
                return
            changed_paths = sorted(self._pending_paths)
            self._pending_paths.clear()
            self._running = True
        started = time.perf_counter()
        result: dict[str, Any] | None = None
        try:
            extra = self.source.get("extra")
            include = extra.get("include") if isinstance(extra, dict) else None
            exclude = extra.get("exclude") if isinstance(extra, dict) else None
            max_file_bytes = extra.get("max_file_bytes") if isinstance(extra, dict) else None
            result = self.memory.repo_index(
                str(self.source["url"]),
                name=str(self.source["name"]),
                ref=str(self.source["ref"]),
                refresh=True,
                include=list(include) if isinstance(include, list) else None,
                exclude=list(exclude) if isinstance(exclude, list) else None,
                max_file_bytes=(int(max_file_bytes) if isinstance(max_file_bytes, int) else None),
            )
        except (
            MemoError,
            OSError,
            RuntimeError,
            ValueError,
            sqlite3.Error,
            subprocess.SubprocessError,
        ) as exc:
            print(f"# memo repo watch: refresh failed: {exc}", file=sys.stderr)
        finally:
            with self._lock:
                self._running = False
                if self._pending_paths:
                    self._arm_locked()
        if result is not None and (
            self.debug
            or result.get("queued")
            or result.get("indexed_files")
            or result.get("deleted_files")
        ):
            elapsed = time.perf_counter() - started
            print(
                f"# memo repo watch: refreshed {len(changed_paths)} paths in {elapsed:.2f}s "
                f"indexed={result.get('indexed_files', 0)} "
                f"deleted={result.get('deleted_files', 0)} "
                f"unchanged={result.get('unchanged_files', 0)} "
                f"job={result.get('job_id', '-')}",
                file=sys.stderr,
            )


def _watch_target(source: dict[str, Any] | None, repo: str) -> Path:
    if source is None:
        raise SystemExit(f"repo not found: {repo}")
    target = Path(str(source["clone_path"]))
    if not target.is_dir():
        raise SystemExit(f"managed clone is missing: {target}")
    return target


def _watch_handler(base: Any, target: Path, refresh: DebouncedRepoRefresh) -> Any:
    class _Handler(base):
        def _maybe(self, event: Any) -> None:
            if event.is_directory:
                return
            path = Path(str(getattr(event, "dest_path", "") or event.src_path))
            try:
                relative = path.relative_to(target)
            except ValueError:
                return
            if relative.parts and relative.parts[0] == ".git":
                return
            refresh.schedule(path)

        def on_modified(self, event: Any) -> None:
            self._maybe(event)

        def on_created(self, event: Any) -> None:
            self._maybe(event)

        def on_deleted(self, event: Any) -> None:
            self._maybe(event)

        def on_moved(self, event: Any) -> None:
            self._maybe(event)

    return _Handler()


def run_repo_watcher(repo: str, *, delay: float = 1.0, debug: bool = False) -> None:
    """Watch one managed clone and refresh only changed file payloads."""
    try:
        from watchdog.events import FileSystemEventHandler
        from watchdog.observers import Observer
    except ImportError as exc:
        raise SystemExit(
            "memo repo watch requires watchdog; reinstall memo or `pip install watchdog`"
        ) from exc

    from memo.config import Config
    from memo.memory import Memory

    memory = Memory(Config.from_env())
    try:
        source = memory.repo_status(repo)
        target = _watch_target(source, repo)
        assert source is not None
        refresh = DebouncedRepoRefresh(memory, source, delay=delay, debug=debug)

        observer = Observer()
        observer.schedule(
            _watch_handler(FileSystemEventHandler, target, refresh),
            str(target),
            recursive=True,
        )
        observer.start()
        stop = threading.Event()

        def _stop(_signo: int, _frame: Any) -> None:
            stop.set()

        signal.signal(signal.SIGINT, _stop)
        signal.signal(signal.SIGTERM, _stop)
        print(
            f"# memo repo watch: watching {source['name']} at {target} "
            f"(debounce {delay}s, Ctrl+C to stop)",
            file=sys.stderr,
        )
        try:
            while not stop.wait(0.5):
                pass
        finally:
            observer.stop()
            observer.join(timeout=3)
            print("# memo repo watch: stopped", file=sys.stderr)
    finally:
        memory.close()


__all__ = ["DebouncedRepoRefresh", "run_repo_watcher"]
