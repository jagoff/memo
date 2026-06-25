"""`memo startup-banner` — prints [MEMO ver] status at agent launch.

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
    memo_line = f"[MEMO {version}] | sync {sync_str}"
    label = f"─── memo / {agent} " if agent else "─── memo "
    width = max(len(memo_line) + 2, len(label) + 4, 44)
    pad = max(0, width - len(label))

    sys.stderr.write(f"{label}{'─' * pad}\n")
    sys.stderr.write(f"{memo_line}\n")
    sys.stderr.write(f"{'─' * width}\n")


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
