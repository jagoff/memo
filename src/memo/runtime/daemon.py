"""launchctl / recall-daemon lifecycle commands (extracted from cli_runtime)."""

from __future__ import annotations

import os

import click
from rich.panel import Panel

from memo.cli_common import console
from memo.config import Config


def _resolve_watcher_binary(memo_bin: str | None) -> str:
    """Resolve an auto-detected watcher binary and reject ephemeral venvs."""
    if memo_bin is not None:
        return memo_bin

    import shutil
    from pathlib import Path

    from memo.runtime.install import _is_project_venv_path

    resolved = shutil.which("memo") or ""
    if not resolved:
        raise click.ClickException(
            "Could not locate `memo` on PATH. Pass --bin /abs/path/to/memo.",
        )
    # A KeepAlive=true plist baked with a project .venv binary crash-loops
    # launchd the moment that venv is removed.
    if _is_project_venv_path(Path(resolved)):
        raise click.ClickException(
            f"auto-detected `memo` resolves inside a project venv: {resolved}\n"
            "A launchd plist must point at a stable isolated runtime. Install "
            "memo as an isolated tool first (`pipx install mlx-memo` or "
            "`uv tool install mlx-memo`), or pass --bin /abs/path/to/memo "
            "explicitly.",
        )
    return resolved


@click.command(name="watch")
@click.option(
    "--delay",
    default=2.0,
    type=float,
    show_default=True,
    help="Debounce window in seconds — coalesces bursts of edits into one reindex.",
)
@click.option("--debug", is_flag=True, help="Print every reindex result to stderr.")
def watch(delay: float, debug: bool) -> None:
    """Auto-reindex on `.md` change. Foreground; Ctrl+C to stop.

    Watches `cfg.memory_dir` recursively. Files saved in Obsidian (or
    any editor) trigger a debounced `Memory.reindex()` call so the
    sqlite-vec index stays in sync without manual `memo reindex` runs.

    Run as a daemon via `memo install-watcher` (launchd plist).
    """
    from memo.watcher import run_watcher

    run_watcher(delay=delay, debug=debug)


@click.command(name="install-watcher")
@click.option(
    "--bin",
    "memo_bin",
    default=None,
    help="Absolute path to the `memo` binary (default: auto-detect via shutil.which).",
)
@click.option("--no-load", is_flag=True, help="Write the plist but don't `launchctl bootstrap` it.")
def install_watcher(memo_bin: str | None, no_load: bool) -> None:
    """Install + load the file-watcher as a launchd daemon.

    Generates `~/Library/LaunchAgents/com.memo.watch.plist`, loads
    it via `launchctl bootstrap`, and verifies it's running. Restart on
    crash is enabled (`KeepAlive=true`). Logs land in
    `~/Library/Logs/memo/`.
    """
    import sys

    if sys.platform != "darwin":
        raise click.ClickException(
            "launchd is macOS-only. On Linux run the foreground watcher "
            "(`memo watch`) under systemd/supervisor instead."
        )
    import subprocess

    from memo.watcher import _PLIST_LABEL, install_plist

    memo_bin = _resolve_watcher_binary(memo_bin)

    plist_path = install_plist(memo_bin)
    console.print(f"[dim]wrote:[/dim] {plist_path}")

    if no_load:
        console.print(
            "[yellow]Skipped load (--no-load). To activate manually:[/yellow]\n"
            f"  launchctl bootstrap gui/$(id -u) {plist_path}",
        )
        return

    uid = os.getuid()
    domain = f"gui/{uid}"
    target = f"{domain}/{_PLIST_LABEL}"

    # Unload first if already present, to pick up plist changes.
    try:
        subprocess.run(
            ["launchctl", "bootout", target],
            check=False,
            capture_output=True,
            timeout=15,
        )
    except subprocess.TimeoutExpired as exc:
        raise click.ClickException(
            f"launchctl bootout timed out (15s) for {target}.",
        ) from exc
    try:
        res = subprocess.run(
            ["launchctl", "bootstrap", domain, str(plist_path)],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except subprocess.TimeoutExpired as exc:
        raise click.ClickException(
            f"launchctl bootstrap timed out (15s) for {target}.",
        ) from exc
    if res.returncode != 0:
        raise click.ClickException(
            f"launchctl bootstrap failed: {res.stderr.strip() or res.stdout.strip()}",
        )

    # Verify.
    try:
        verify = subprocess.run(
            ["launchctl", "print", target],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except subprocess.TimeoutExpired as exc:
        raise click.ClickException(
            f"launchctl print timed out (15s) for {target}.",
        ) from exc
    if verify.returncode != 0:
        raise click.ClickException(
            "Plist loaded but `launchctl print` could not find it. "
            "Inspect `~/Library/Logs/memo/watch.err.log`.",
        )

    console.print(
        Panel.fit(
            f"[bold]watcher loaded[/bold]\n"
            f"[dim]label:[/dim] {_PLIST_LABEL}\n"
            f"[dim]plist:[/dim] {plist_path}\n"
            f"[dim]logs:[/dim] ~/Library/Logs/memo/watch.{{out,err}}.log",
            title="✓ install-watcher",
            border_style="green",
        )
    )


@click.command(name="uninstall-watcher")
def uninstall_watcher_cmd() -> None:
    """Unload + remove the file-watcher launchd job."""
    import sys

    if sys.platform != "darwin":
        raise click.ClickException(
            "launchd is macOS-only — there is no watcher launchd job to remove on Linux."
        )
    import subprocess

    from memo.watcher import _PLIST_LABEL, uninstall_plist

    uid = os.getuid()
    target = f"gui/{uid}/{_PLIST_LABEL}"
    try:
        subprocess.run(
            ["launchctl", "bootout", target],
            check=False,
            capture_output=True,
            timeout=15,
        )
    except subprocess.TimeoutExpired as exc:
        raise click.ClickException(
            f"launchctl bootout timed out (15s) for {target}.",
        ) from exc
    existed = uninstall_plist()
    if existed:
        console.print("[green]✓ watcher uninstalled.[/green]")
    else:
        console.print("[yellow]No plist found to remove.[/yellow]")


@click.command(name="sleep-cycle")
@click.option("--debug", is_flag=True, help="Print activity to stderr.")
def sleep_cycle(debug: bool) -> None:
    """Run autonomous background maintenance.

    Monitors system activity and runs synthesize/consolidate when idle.
    Gated by MEMO_MAINT_SLEEP_CYCLE_ENABLED.
    """
    from memo.flags import flag_bool

    if not flag_bool("MEMO_MAINT_SLEEP_CYCLE_ENABLED"):
        console.print("[yellow]Sleep cycle disabled (MEMO_MAINT_SLEEP_CYCLE_ENABLED=0).[/yellow]")
        return

    from memo.runtime.sleep_cycle import run_sleep_cycle

    run_sleep_cycle(debug=debug)


@click.command(name="prewarm")
@click.option(
    "--download-all",
    "download_all",
    is_flag=True,
    default=False,
    help="Also pre-download the configured chat/helper LLM models. Used by the installer.",
)
def prewarm(download_all: bool) -> None:
    """SessionStart hook — pre-load the MLX embedder so first recall is fast.

    Loads the embedder + optional reranker into memory so subsequent
    recall-hook calls benefit from the OS file cache (cold-load drops
    from ~2 s to ~500 ms). Writes a warm-signal file so the recall-hook
    can detect a cold start and fall back to BM25 instead of timing out.

    With --download-all, also pre-downloads the configured chat/helper LLM models
    so the first `memo ask` doesn't stall on a multi-GB download. Designed
    to be called by the installer immediately after `pipx install`.

    Failures are silent when run as a hook — a hook crash must never block
    Claude Code's prompt submission.
    """
    import sys as _sys

    _warm_embedder(Config.from_env(), download_all=download_all)
    _sys.exit(0)


def _warm_embedder(cfg: Config, *, download_all: bool = False, warm_reranker: bool = True) -> None:
    """Load the embedder (+ optional reranker) and stamp the ``.prewarm_ts`` warm
    signal the recall hook reads. Shared by the ``prewarm`` command and
    ``memo onboard``.

    The stamp is what lets a fresh install's FIRST recall run vec instead of the
    cold-start bm25 fallback: without it the recall hook downgrades to bm25, whose
    relevance score falls under the vec-calibrated ``min_sim`` floor, so the very
    first saved memory is invisible to the very first recall. The stamp is written
    as soon as the EMBEDDER is warm — the recall hook's cold-start check is about
    the embedder's disk cache, not the reranker, so a later reranker failure must
    never suppress it. ``warm_reranker=False`` skips the reranker entirely (used by
    ``memo onboard``, where a fresh install's reranker model may be uncached and
    the reranker is irrelevant to the first-recall path). Best-effort — never
    raises (a SessionStart hook crash must not block prompt submission)."""
    import sys as _sys
    import time as _time

    from memo.flags import flag_bool

    if flag_bool("MEMO_RECALL_DISABLE"):
        return
    try:
        from memo.embedder_select import make_embedder

        # make_embedder picks MLX (Apple Silicon) or the CPU backend (Linux),
        # so prewarm actually warms whichever embedder this host will use.
        emb = make_embedder(cfg)
        emb.embed(["warmup"])  # batch=1; forces the model load + first forward pass
        # Write warm-signal so recall-hook knows disk cache is fresh. Written HERE
        # (right after the embedder warm, before the reranker) so an uncached /
        # failing reranker load can never suppress the signal — the recall hook
        # reads only this file's mtime, and only the embedder gates recall.
        cfg.state_dir.mkdir(parents=True, exist_ok=True)
        warm_signal = cfg.state_dir / ".prewarm_ts"
        warm_signal.write_text(str(_time.time()))
        # Reranker prewarm — same rationale as the embedder. Skipped when disabled
        # (or warm_reranker=False) to keep the SessionStart hook below its 30s
        # budget on machines that opted out of rerank entirely.
        if warm_reranker and cfg.reranker_enabled:
            from memo.reranker import MLXReranker

            r = MLXReranker(
                model_path=cfg.reranker_model,
                revision=cfg.reranker_revision,
            )
            r.warmup()
        if download_all:
            # Pre-download the chat models so the first `memo ask` doesn't
            # stall. snapshot_download is a no-op when the model is already
            # cached, so re-running this is safe.
            try:
                from memo.model_pins import resolve_model_snapshot

                for repo, revision in (
                    (cfg.llm_model, cfg.llm_revision),
                    (cfg.helper_model, cfg.helper_revision),
                ):
                    click.echo(f"[memo] downloading {repo}…")
                    resolve_model_snapshot(repo, revision)
                    click.echo(f"[memo] ready: {repo}")
            except Exception as dl_exc:
                click.echo(f"[memo] chat model download failed: {dl_exc}", err=True)
    except Exception as exc:
        if flag_bool("MEMO_RECALL_DEBUG"):
            print(f"# memo prewarm failed: {exc}", file=_sys.stderr)
