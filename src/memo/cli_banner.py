"""`memo startup-banner` — prints [Memo ver] status at agent launch.

Called by agent shims (installed via `memo install-shims`). Must stay
fast and MLX-free: only git + importlib.metadata, no embedding.
"""
from __future__ import annotations

import sys

import click


@click.command(name="startup-banner")
@click.option("--agent", default="", help="Agent name shown in header (e.g. codex).")
def startup_banner_cmd(agent: str) -> None:
    """Print memo status banner to stderr. Called by agent shims on startup."""
    import importlib.metadata

    try:
        version = importlib.metadata.version("mlx-memo")
    except Exception:
        version = "?"

    sync_str = _fast_sync_state()
    update_tag = _pending_update_tag()
    update_str = f" | ⬆ {update_tag} available — run: memo update" if update_tag else ""
    memo_line = f"[Memo {version}] | sync {sync_str}{update_str}"
    label = f"─── memo / {agent} " if agent else "─── memo "
    width = max(len(memo_line) + 2, len(label) + 4, 44)
    pad = max(0, width - len(label))

    sys.stderr.write(f"{label}{'─' * pad}\n")
    sys.stderr.write(f"{memo_line}\n")
    sys.stderr.write(f"{'─' * width}\n")


def _pending_update_tag() -> str | None:
    """Read the pending update notification file. Fast, no network."""
    try:
        from memo.config import Config
        from memo.runtime.autoupdate import pending_update_tag

        return pending_update_tag(Config.from_env())
    except Exception:
        return None


def _fast_sync_state() -> str:
    """Git-only sync check — no network, no MLX. Returns human-readable label."""
    try:
        from memo.config import Config
        from memo.sync_git import sync_status

        cfg = Config.from_env()
        st = sync_status(cfg)
        if not st.get("is_git_clone"):
            return "local"
        ahead = int(st.get("ahead") or 0)
        return "al día" if ahead == 0 else f"ahead {ahead}"
    except Exception:
        return "-"


@click.command(name="codex-badge")
@click.option("--agent", default="", help="Agent name; accepted for shim symmetry.")
def codex_badge_cmd(agent: str) -> None:
    """Show memo's version badge in Codex/Supacode's top-line notification area."""
    del agent
    from memo.runtime.codex_notify import emit_memo_badge

    emit_memo_badge()
