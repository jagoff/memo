"""Vault → memo re-ingestion orchestration.

Ported from synapse ``ops.run_vault_ingest`` on its deprecation (2026-07-30).
Per vault: best-effort ``git pull --ff-only``, then ``memo ingest <root>
--name <label> --prune`` with the fixed system excludes plus any user-recorded
tombstones (:class:`memo.ingest_exclude.IngestExcludeStore`) — so a note the
user deleted stays out of the index. Args are assembled as a list and run
without a shell, so note paths with spaces or metacharacters can't break
quoting or inject. The synapse-era extras (gbrain import, memflow signal,
overview rebuild) were dropped with synapse.

Runs the installed ``memo`` binary in a subprocess so the WatchPaths launchd
agent (``com.memo.vault-ingest``) stays a thin process that never loads the
embedder itself.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any

from memo.flags import flag_str
from memo.ingest_exclude import IngestExcludeStore

_logger = logging.getLogger("memo.vault_ingest")

_DEFAULT_VAULT_PATHS = (
    Path.home() / "Library/Mobile Documents/iCloud~md~obsidian/Documents/Notes",
    Path.home() / "Library/Mobile Documents/iCloud~md~obsidian/Documents/obsidian-work",
)

# Fixed excludes applied to every vault ingest (system folders memo must never
# index as personal/work notes). User-deleted notes are appended per-vault via
# IngestExcludeStore so re-ingestion can't resurrect them. Archive casing
# variants are enumerated here AND covered by memo's case-insensitive matcher
# (defense in depth).
_FIXED_VAULT_EXCLUDES = (
    "Obsidian/AI/**",
    "04-Archive/**",
    "Archive/**",
    "archive/**",
)


def vault_paths() -> list[Path]:
    """Obsidian vault roots to re-ingest.

    ``MEMO_VAULT_PATHS`` (comma list) overrides; otherwise the known iCloud
    vaults that exist on disk.
    """
    raw = flag_str("MEMO_VAULT_PATHS").strip()
    if raw:
        return [Path(p.strip()).expanduser() for p in raw.split(",") if p.strip()]
    return [p for p in _DEFAULT_VAULT_PATHS if p.is_dir()]


def vault_label(path: Path) -> str:
    """memo ``--name`` label for a vault dir: Notes→notes, obsidian-work→work."""
    name = path.name.lower()
    if name.startswith("obsidian-"):
        name = name[len("obsidian-") :]
    return name or "vault"


def build_ingest_command(memo_bin: str, path: Path, label: str, excludes: list[str]) -> list[str]:
    """Argv for one vault's ``memo ingest`` run (no shell, no quoting issues)."""
    cmd = [memo_bin, "ingest", str(path), "--name", label, "--prune"]
    for glob in excludes:
        cmd += ["--exclude", glob]
    return cmd


def run_vault_ingest(*, memo_bin: str | None = None) -> dict[str, Any]:
    """Re-ingest each vault into memo; returns per-vault results."""
    memo_bin = memo_bin or shutil.which("memo") or "memo"
    git_bin = shutil.which("git") or "git"
    store = IngestExcludeStore()
    results: list[dict[str, Any]] = []
    for p in vault_paths():
        label = vault_label(p)
        if (p / ".git").is_dir():
            subprocess.run(
                [git_bin, "-C", str(p), "pull", "--ff-only"],
                check=False,
                capture_output=True,
            )
        excludes = list(_FIXED_VAULT_EXCLUDES) + store.globs(label)
        proc = subprocess.run(
            build_ingest_command(memo_bin, p, label, excludes),
            check=False,
            capture_output=True,
            text=True,
        )
        results.append(
            {
                "vault": label,
                "path": str(p),
                "returncode": proc.returncode,
                "excludes": excludes,
            }
        )
        if proc.returncode != 0:
            _logger.warning(
                "vault_ingest_failed: %s exit=%s stderr=%s",
                label,
                proc.returncode,
                (proc.stderr or "").strip()[:500],
            )
    return {"ok": all(r["returncode"] == 0 for r in results), "vaults": results}
