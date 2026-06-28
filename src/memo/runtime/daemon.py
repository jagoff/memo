"""launchctl / recall-daemon lifecycle commands (extracted from cli_runtime)."""

from __future__ import annotations

import os

import click
from rich.panel import Panel

from memo.cli_common import console
from memo.config import Config


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
    import shutil as _shutil
    import subprocess

    from memo.watcher import _PLIST_LABEL, install_plist

    if memo_bin is None:
        memo_bin = _shutil.which("memo") or ""
        if not memo_bin:
            raise click.ClickException(
                "Could not locate `memo` on PATH. Pass --bin /abs/path/to/memo.",
            )

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
    help="Also pre-download the chat LLM models (7B + 3B helper). Used by the installer.",
)
def prewarm(download_all: bool) -> None:
    """SessionStart hook — pre-load the MLX embedder so first recall is fast.

    Loads the embedder + optional reranker into memory so subsequent
    recall-hook calls benefit from the OS file cache (cold-load drops
    from ~2 s to ~500 ms). Writes a warm-signal file so the recall-hook
    can detect a cold start and fall back to BM25 instead of timing out.

    With --download-all, also pre-downloads the chat LLM models (7B + 3B)
    so the first `memo ask` doesn't stall on a multi-GB download. Designed
    to be called by the installer immediately after `pipx install`.

    Failures are silent when run as a hook — a hook crash must never block
    Claude Code's prompt submission.
    """
    import sys as _sys
    import time as _time

    from memo.flags import flag_bool

    if flag_bool("MEMO_RECALL_DISABLE"):
        _sys.exit(0)
    try:
        from memo.embedder import MLXEmbedder

        cfg = Config.from_env()
        emb = MLXEmbedder(model_path=cfg.embedder_model, expected_dims=cfg.embedder_dims)
        emb.embed(["warmup"])  # batch=1; forces MLX load + first forward pass
        # Reranker prewarm — same rationale as the embedder. Skipped
        # when disabled to keep the SessionStart hook below its
        # 30s budget on machines that opted out of rerank entirely.
        if cfg.reranker_enabled:
            from memo.reranker import MLXReranker

            r = MLXReranker(
                model_path=cfg.reranker_model,
                revision=cfg.reranker_revision,
            )
            r.warmup()
        # Write warm-signal so recall-hook knows disk cache is fresh.
        # The file's mtime is the only datum read by recall-hook.
        cfg.state_dir.mkdir(parents=True, exist_ok=True)
        warm_signal = cfg.state_dir / ".prewarm_ts"
        warm_signal.write_text(str(_time.time()))
        if download_all:
            # Pre-download the chat models so the first `memo ask` doesn't
            # stall. snapshot_download is a no-op when the model is already
            # cached, so re-running this is safe.
            try:
                from huggingface_hub import snapshot_download

                for repo in (cfg.llm_model, cfg.helper_model):
                    click.echo(f"[memo] downloading {repo}…")
                    snapshot_download(repo_id=repo)
                    click.echo(f"[memo] ready: {repo}")
            except Exception as dl_exc:
                click.echo(f"[memo] chat model download failed: {dl_exc}", err=True)
    except Exception as exc:
        if flag_bool("MEMO_RECALL_DEBUG"):
            print(f"# memo prewarm failed: {exc}", file=_sys.stderr)
    _sys.exit(0)
