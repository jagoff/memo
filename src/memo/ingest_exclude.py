"""Durable vault-ingest exclusion tombstones.

When the user deletes a vault-derived memory, deleting the memo row is futile
on its own: the ``com.memo.vault-ingest`` agent re-runs ``memo ingest`` and the
surviving ``.md`` resurrects the row on the next tick. memo must not destroy
the user's Obsidian file, so instead a *tombstone* is recorded here — a
vault-relative glob fed to ``memo ingest --exclude`` so re-ingestion
permanently skips that note.

State lives under ``<state_dir>/ingest_excludes/<label>.txt`` (one glob per
line, one file per vault ``--name`` label). Ported from synapse
``ingest_exclude.py`` on its deprecation (2026-07-30); the synapse chat-side
helpers (``source_vault_rel``, ``filter_tombstoned_sources``) died with it.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

_logger = logging.getLogger("memo.ingest_exclude")

_SAFE_LABEL_RE = re.compile(r"[^a-z0-9_-]+")


def _safe_label(label: str) -> str:
    """Filesystem-safe vault label (lowercased, non ``[a-z0-9_-]`` collapsed)."""
    cleaned = _SAFE_LABEL_RE.sub("-", str(label or "").strip().lower()).strip("-")
    return cleaned or "vault"


class IngestExcludeStore:
    """Per-vault list of exclude globs (relative to the vault root).

    Append-only-ish text files, deduped on read/write. Pure file management —
    label↔path resolution lives in ``vault_ingest`` to avoid an import cycle.
    """

    def __init__(self, state_dir: Path | str | None = None) -> None:
        if state_dir is None:
            from memo.config import Config

            state_dir = Config.from_env().state_dir
        self.dir = Path(state_dir).expanduser() / "ingest_excludes"

    def _path(self, vault_label: str) -> Path:
        return self.dir / f"{_safe_label(vault_label)}.txt"

    def globs(self, vault_label: str) -> list[str]:
        """Distinct non-empty exclude globs for *vault_label*, order preserved."""
        path = self._path(vault_label)
        if not path.exists():
            return []
        seen: set[str] = set()
        out: list[str] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            glob = line.strip()
            if not glob or glob.startswith("#") or glob in seen:
                continue
            seen.add(glob)
            out.append(glob)
        return out

    def add(self, *, vault_label: str, rel_path: str) -> bool:
        """Record *rel_path* as excluded for *vault_label*. Idempotent.

        Returns True if a new entry was written, False if already present.
        """
        glob = str(rel_path or "").strip()
        if not glob:
            raise ValueError("ingest_exclude.add: rel_path required")
        if glob in self.globs(vault_label):
            return False
        path = self._path(vault_label)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(glob + "\n")
        _logger.info("ingest_exclude_added: %s %s", _safe_label(vault_label), glob)
        return True

    def remove(self, *, vault_label: str, rel_path: str) -> bool:
        """Drop *rel_path* from *vault_label*'s excludes. Returns True if removed."""
        glob = str(rel_path or "").strip()
        path = self._path(vault_label)
        if not glob or not path.exists():
            return False
        existing = self.globs(vault_label)
        kept = [g for g in existing if g != glob]
        if len(kept) == len(existing):
            return False
        path.write_text("".join(f"{g}\n" for g in kept), encoding="utf-8")
        _logger.info("ingest_exclude_removed: %s %s", _safe_label(vault_label), glob)
        return True

    def all_labels(self) -> list[str]:
        """Vault labels that currently have at least one tombstone file."""
        if not self.dir.exists():
            return []
        return sorted(p.stem for p in self.dir.glob("*.txt"))
