"""Private helpers for repo_index.py: constants, git utilities, file/chunk processing.

Extracted to keep repo_index.py under 800 lines. Import via repo_index.py — not
intended for direct use outside the repo-indexing subsystem.
"""

from __future__ import annotations

import fnmatch
import hashlib
import re
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from memo.config import Config
from memo.embedder import MLXEmbedder

# ---------------------------------------------------------------------------
# Public constants (re-exported from repo_index for backward compat)
# ---------------------------------------------------------------------------

DEFAULT_EXCLUDE_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".claude",
        ".codex",
        ".devin",
        ".venv",
        "venv",
        "env",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "node_modules",
        ".next",
        ".nuxt",
        ".turbo",
        "dist",
        "build",
        "target",
        "coverage",
        ".idea",
        ".vscode",
    }
)

DEFAULT_EXCLUDE_GLOBS = (
    "*.png",
    "*.jpg",
    "*.jpeg",
    "*.gif",
    "*.webp",
    "*.ico",
    "*.pdf",
    "*.zip",
    "*.gz",
    "*.bz2",
    "*.xz",
    "*.7z",
    "*.tar",
    "*.sqlite",
    "*.sqlite3",
    "*.db",
    "*.dylib",
    "*.so",
    "*.a",
    "*.pyc",
    "*.class",
    "*.o",
    "*.wasm",
)

DEFAULT_MAX_FILE_BYTES = 2_000_000
DEFAULT_CHUNK_TARGET_CHARS = 3500
DEFAULT_CHUNK_OVERLAP_LINES = 8
# Chunks shorter than this with no heading/link carry almost no semantic
# signal (stray punctuation, empty list items, frontmatter fragments) and
# only add noise + cost to the embedding index. They are dropped at build
# time. Lines are still kept in full in repo_lines for keyword search.
MIN_CHUNK_CHARS = 60
_LINK_RE = re.compile(r"\[\[.+?\]\]|\[.+?\]\(.+?\)|https?://\S+")

DEFAULT_EMBED_BATCH = 64
MIN_EMBED_BATCH = 1
DEFAULT_FLUSH_BATCH = 25
MIN_FLUSH_BATCH = 1
STATUS_INDEXING = "indexing"

ProgressCallback = Callable[[str, dict[str, Any]], None]


# ---------------------------------------------------------------------------
# Internal data class (not part of the public API)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RepoEmbedInput:
    chunk: dict[str, Any]
    text: str
    input_hash: str


# ---------------------------------------------------------------------------
# Noise filter
# ---------------------------------------------------------------------------

def _is_noise_chunk(body: str) -> bool:
    """True for near-empty chunks with no heading and no link/URL."""
    stripped = body.strip()
    if len(stripped) >= MIN_CHUNK_CHARS:
        return False
    if "#" in stripped and re.search(r"^#{1,6}\s+\S", stripped, re.MULTILINE):
        return False  # markdown heading — keep
    # wikilink / md link / URL — keep; otherwise it's noise
    return not _LINK_RE.search(stripped)


# ---------------------------------------------------------------------------
# Git utilities
# ---------------------------------------------------------------------------

def _git_timeout(default: float) -> float:
    """Cap for `subprocess.run` of git commands.

    Why: a network-flaky `git clone` or `ls-files` would otherwise hang
    indefinitely, blocking the indexer thread (and any caller awaiting it).
    Configurable via MEMO_REPO_GIT_TIMEOUT_S (seconds, 0 disables).
    """
    from memo.flags import flag_float

    v = flag_float("MEMO_REPO_GIT_TIMEOUT_S")
    if v is None:
        return default
    return v if v > 0 else 0.0


def _git(args: list[str], *, check: bool = True, timeout: float | None = None) -> str:
    t = _git_timeout(120.0 if timeout is None else timeout)
    try:
        proc = subprocess.run(
            args,
            check=False,
            capture_output=True,
            text=True,
            timeout=t if t > 0 else None,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("`git` not found on PATH") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"git timed out after {t:.0f}s: {' '.join(args)} "
            f"(raise MEMO_REPO_GIT_TIMEOUT_S to extend)"
        ) from exc
    if check and proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(f"git command failed ({proc.returncode}): {' '.join(args)}\n{detail}")
    return proc.stdout


def _emit(progress: ProgressCallback | None, event: str, **data: Any) -> None:
    if progress is not None:
        progress(event, data)


def _tracked_files(clone_path: Path) -> list[str]:
    """Return Git-tracked files without walking generated/untracked trees."""
    t = _git_timeout(60.0)
    try:
        proc = subprocess.run(
            ["git", "-C", str(clone_path), "ls-files", "-z"],
            check=False,
            capture_output=True,
            text=True,
            timeout=t if t > 0 else None,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("`git` not found on PATH") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"git ls-files timed out after {t:.0f}s on {clone_path} "
            f"(raise MEMO_REPO_GIT_TIMEOUT_S to extend)"
        ) from exc
    if proc.returncode == 0:
        paths = [p for p in proc.stdout.split("\0") if p]
        return sorted(paths)
    return sorted(
        p.relative_to(clone_path).as_posix() for p in clone_path.rglob("*") if p.is_file()
    )


# ---------------------------------------------------------------------------
# Embed helpers
# ---------------------------------------------------------------------------

def _repo_embed_input(chunk: dict[str, Any]) -> RepoEmbedInput:
    text = (
        f"repo: {chunk['repo_name']}\npath: {chunk['path']}\n"
        f"lines: {chunk['line_start']}-{chunk['line_end']}\n\n{chunk['body_text']}"
    )
    return RepoEmbedInput(
        chunk=chunk,
        text=text,
        input_hash=hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest(),
    )


def _repo_embed_batch_size() -> int:
    from memo.flags import flag_int

    value = flag_int("MEMO_REPO_EMBED_BATCH")
    return DEFAULT_EMBED_BATCH if value is None else max(MIN_EMBED_BATCH, value)


def _repo_flush_batch_size() -> int:
    """Number of files to accumulate before flushing to the store.

    Lower values trade write overhead for finer-grained resume
    granularity if the run is interrupted.
    """
    from memo.flags import flag_int

    value = flag_int("MEMO_REPO_FLUSH_BATCH")
    return DEFAULT_FLUSH_BATCH if value is None else max(MIN_FLUSH_BATCH, value)


def _embed_cache_model(embedder: MLXEmbedder, cfg: Config) -> str:
    model = getattr(embedder, "model_path", None)
    return str(model or cfg.embedder_model)


# ---------------------------------------------------------------------------
# File/metadata helpers
# ---------------------------------------------------------------------------

def _semantic_status(current: str | None, counts: dict[str, int]) -> str:
    if counts["chunks"] == counts["embedded_chunks"]:
        return "semantic_ready"
    if current == "semantic_indexing":
        return "semantic_indexing"
    return "semantic_pending"


def _derive_repo_name(url: str) -> str:
    s = url.rstrip("/").removesuffix(".git")
    return re.split(r"[/:\s]+", s)[-1] or "repo"


def _safe_repo_name(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9._-]+", "-", name)
    return name.strip("-._")[:80]


def _stable_id(*parts: str) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(part.encode("utf-8", errors="replace"))
        h.update(b"\0")
    return h.hexdigest()[:32]


def _now_iso() -> str:
    return datetime.now(tz=UTC).astimezone().isoformat(timespec="milliseconds")


def _looks_binary(raw: bytes) -> bool:
    sample = raw[:8192]
    return b"\0" in sample


def _is_excluded(rel: Path, rel_posix: str, include: list[str], exclude: list[str]) -> bool:
    if any(part in DEFAULT_EXCLUDE_DIRS for part in rel.parts):
        return True
    if include and not any(fnmatch.fnmatch(rel_posix, pat) for pat in include):
        return True
    return any(fnmatch.fnmatch(rel_posix, pat) for pat in exclude)


# ---------------------------------------------------------------------------
# Chunking helpers
# ---------------------------------------------------------------------------

def _chunk_lines(
    lines: list[str],
    *,
    target_chars: int = DEFAULT_CHUNK_TARGET_CHARS,
    overlap_lines: int = DEFAULT_CHUNK_OVERLAP_LINES,
) -> list[tuple[int, int, int, str]]:
    if not lines:
        return []

    chunks: list[tuple[int, int, int, str]] = []
    seq = 0
    start = 0
    while start < len(lines):
        # Minified/generated files often contain one very long line. Keep
        # the exact full line in repo_lines, but never hand that whole
        # line to the embedder as one sequence.
        if len(lines[start]) > target_chars:
            for part in _split_long_line(lines[start], target_chars):
                chunks.append((seq, start + 1, start + 1, part))
                seq += 1
            start += 1
            continue

        end = start
        chars = 0
        while end < len(lines) and chars < target_chars:
            if len(lines[end]) > target_chars:
                break
            chars += len(lines[end]) + 1
            end += 1
        if end == start:
            # Defensive: the long-line branch above should handle this,
            # but guarantee forward progress if target_chars is tiny.
            end += 1
        body = "\n".join(lines[start:end])
        chunks.append((seq, start + 1, end, body))
        seq += 1
        if end >= len(lines):
            break
        start = max(start + 1, end - overlap_lines)
    return chunks


def _split_long_line(line: str, target_chars: int) -> list[str]:
    target = max(1, target_chars)
    return [line[i : i + target] for i in range(0, len(line), target)]


def _language_for_path(path: str) -> str:
    suffix = Path(path).suffix.lower().lstrip(".")
    return suffix or "text"
