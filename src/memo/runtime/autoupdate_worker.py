"""Crash-safe child wrapper for the detached automatic updater."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from memo.config import Config
from memo.runtime.autoupdate import _claim_spawned_lease


def main(argv: list[str] | None = None) -> int:
    """Claim the parent's starting lease, then replace this process with update."""

    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 4:
        return 2
    state_dir_text, tag, parent_pid_text, started_at_text = args
    try:
        parent_pid = int(parent_pid_text)
        started_at = float(started_at_text)
    except ValueError:
        return 2

    state_dir = Path(state_dir_text).expanduser().resolve()
    cfg = Config(state_dir=state_dir, data_dir=state_dir / ".worker-unused-data")
    if not _claim_spawned_lease(
        cfg,
        tag,
        parent_pid=parent_pid,
        child_pid=os.getpid(),
        started_at=started_at,
    ):
        return 1

    os.execv(  # noqa: S606 - replace with the exact trusted interpreter path
        sys.executable,
        [sys.executable, "-m", "memo.cli", "update", "--to-tag", tag],
    )
    return 1  # pragma: no cover - os.execv only returns on an OS contract breach


if __name__ == "__main__":  # pragma: no cover - exercised through subprocesses
    raise SystemExit(main())
