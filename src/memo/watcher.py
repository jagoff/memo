"""File-watcher daemon — auto-reindex on `.md` edit.

`Memory.reindex()` is idempotent and cheap (skips files whose
`body_hash` matches the indexed value), so the watcher just calls it
after a short debounce when a `.md` under the memory directory is
modified, created, or moved.

Two ways to run:

    # Foreground (Ctrl+C to stop)
    memo watch

    # Background via launchd
    memo install-watcher

The watcher is intentionally minimal — no incremental "only re-embed
this one file" path. The reindex pass is already O(n) over the index
and the bottleneck is the embedder forward pass, which only fires on
hash mismatches. For a typical vault (a few thousand `.md` files), the
walk takes <50 ms; the re-embed cost is paid only for files that
actually changed.
"""

from __future__ import annotations

import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any


class _DebouncedReindex:
    """Coalesce bursts of FS events into a single reindex call.

    A debounce window of `delay` seconds is reset on each event, so a
    flurry of saves from Obsidian's autosave only triggers one reindex
    once the dust settles.
    """

    def __init__(self, memory: Any, *, delay: float = 2.0, debug: bool = False) -> None:
        self.memory = memory
        self.delay = delay
        self.debug = debug
        self._timer: threading.Timer | None = None
        self._lock = threading.Lock()
        self._pending = False

    def schedule(self, path: Path) -> None:
        with self._lock:
            self._pending = True
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(self.delay, self._run)
            self._timer.daemon = True
            self._timer.start()
            if self.debug:
                print(f"# memo watch: queued reindex (trigger: {path.name})", file=sys.stderr)

    def _run(self) -> None:
        with self._lock:
            if not self._pending:
                return
            self._pending = False
        t0 = time.perf_counter()
        try:
            res = self.memory.reindex()
        except Exception as exc:
            if self.debug:
                print(f"# memo watch: reindex failed: {exc}", file=sys.stderr)
            return
        dt = time.perf_counter() - t0
        if self.debug or res.get("reindexed", 0) or res.get("added", 0):
            print(
                f"# memo watch: reindex done in {dt:.2f}s — "
                f"checked={res.get('checked', 0)} "
                f"reindexed={res.get('reindexed', 0)} "
                f"added={res.get('added', 0)} "
                f"skipped={res.get('skipped', 0)}",
                file=sys.stderr,
            )


def run_watcher(*, delay: float = 2.0, debug: bool = False) -> None:
    """Block the calling thread, reindex on `.md` change."""
    try:
        from watchdog.events import FileSystemEventHandler
        from watchdog.observers import Observer
    except ImportError as exc:
        raise SystemExit(
            "memo watch requires the `watchdog` package. "
            "Run: pip install watchdog (or reinstall memo-mcp which depends on it)."
        ) from exc

    from memo.config import Config
    from memo.memory import Memory

    cfg = Config.from_env()
    mem = Memory(cfg)
    target = cfg.memory_dir
    target.mkdir(parents=True, exist_ok=True)

    deb = _DebouncedReindex(mem, delay=delay, debug=debug)

    class _Handler(FileSystemEventHandler):
        def _maybe(self, event: Any) -> None:
            if event.is_directory:
                return
            src = getattr(event, "src_path", "") or ""
            if not src.endswith(".md"):
                return
            deb.schedule(Path(src))

        def on_modified(self, event: Any) -> None:
            self._maybe(event)

        def on_created(self, event: Any) -> None:
            self._maybe(event)

        def on_moved(self, event: Any) -> None:
            self._maybe(event)

    obs = Observer()
    obs.schedule(_Handler(), str(target), recursive=True)
    obs.start()
    print(f"# memo watch: watching {target} (debounce {delay}s, Ctrl+C to stop)", file=sys.stderr)

    stop = threading.Event()

    def _sigterm(_signo: int, _frame: Any) -> None:
        stop.set()

    signal.signal(signal.SIGINT, _sigterm)
    signal.signal(signal.SIGTERM, _sigterm)

    try:
        while not stop.is_set():
            time.sleep(0.5)
    finally:
        obs.stop()
        obs.join(timeout=3)
        print("# memo watch: stopped", file=sys.stderr)


# ----------------- launchd plist generation -----------------

_PLIST_LABEL = "com.memo.watch"


def _plist_path() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{_PLIST_LABEL}.plist"


def _log_dir() -> Path:
    return Path.home() / "Library" / "Logs" / "memo"


def render_plist(memo_bin: str) -> str:
    """Return the plist XML for the watcher launchd job.

    `memo_bin` should be the absolute path to the `memo` CLI binary so
    launchd doesn't have to resolve `$PATH` (LaunchAgents inherit a
    minimal env).
    """
    log_dir = _log_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    # Forward MEMO_* env vars from the user's interactive shell so the
    # daemon uses the same vault/state dir as `memo` in a terminal.
    env_vars = {k: v for k, v in os.environ.items() if k.startswith("MEMO_")}
    env_keys = "".join(
        f"        <key>{k}</key>\n        <string>{v}</string>\n" for k, v in env_vars.items()
    )
    env_block = (
        f"    <key>EnvironmentVariables</key>\n    <dict>\n{env_keys}    </dict>\n"
        if env_vars
        else ""
    )
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{_PLIST_LABEL}</string>
    <key>ProgramArguments</key>
    <array>
        <string>{memo_bin}</string>
        <string>watch</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>{log_dir}/watch.out.log</string>
    <key>StandardErrorPath</key>
    <string>{log_dir}/watch.err.log</string>
{env_block}</dict>
</plist>
"""


def install_plist(memo_bin: str) -> Path:
    """Write the plist and return its path. Caller is responsible for
    `launchctl bootstrap` (the CLI does that)."""
    p = _plist_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(render_plist(memo_bin), encoding="utf-8")
    return p


def uninstall_plist() -> bool:
    """Remove the plist file. Returns True if it existed and was deleted."""
    p = _plist_path()
    if p.is_file():
        p.unlink()
        return True
    return False
