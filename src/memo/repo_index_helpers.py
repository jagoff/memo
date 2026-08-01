"""Private helpers for repo_index.py: constants, git utilities, file/chunk processing.

Extracted to keep repo_index.py under 800 lines. Import via repo_index.py — not
intended for direct use outside the repo-indexing subsystem.
"""

from __future__ import annotations

import fnmatch
import hashlib
import os
import re
import sqlite3
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from memo.config import Config
from memo.embed_base import EmbedderBase

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

# A 64-chunk Qwen3/MLX batch can exceed 19 GiB on real repositories even when
# individual chunks are bounded. Sixteen keeps peak memory practical on common
# Apple Silicon machines; operators with more headroom can still override it.
DEFAULT_EMBED_BATCH = 16
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


# Git's `ext::`/`fd::` remote helpers execute an arbitrary command as part of a
# clone/fetch. GIT_ALLOW_PROTOCOL is an explicit allow-list — any protocol not
# named here (notably ext/fd) is refused by git itself, while the transports memo
# actually uses (including `file` for local-path clones) keep working. Defense in
# depth behind `_validate_clone_url`; disabling the credential prompt keeps a
# missing-auth clone from hanging the indexer thread. Applied to every git call.
_GIT_SAFE_ENV = {
    "GIT_ALLOW_PROTOCOL": "file:git:ssh:http:https:git+ssh",
    "GIT_TERMINAL_PROMPT": "0",
}

_ALLOWED_URL_SCHEMES = ("https://", "http://", "ssh://", "git://", "git+ssh://", "file://")


def _validate_clone_url(url: str) -> None:
    """Reject repo URLs that could reach git's arbitrary-command remote helpers.

    `memo_repo_index` accepts a caller-supplied URL and passes it to `git clone`.
    Without this guard a URL like ``ext::sh -c '<cmd>'`` (git's ``ext`` transport)
    executes ``<cmd>`` as the local user — an RCE reachable from the MCP surface.
    The only arbitrary-command vectors are the ``<scheme>::`` remote-helper syntax
    (``ext::``/``fd::``) and a leading ``-`` (option injection); everything else —
    real network schemes, scp-like ``user@host:path``, and bare local filesystem
    paths — is a normal, safe clone target. An allowed network scheme is checked
    first so an IPv6 literal host (``https://[::1]/x``) is not mistaken for the
    remote-helper syntax.
    """
    u = url.strip()
    if not u or u.startswith("-"):
        raise ValueError(f"unsafe repo url (empty or leading dash): {url!r}")
    if u.lower().startswith(_ALLOWED_URL_SCHEMES):
        return
    if "::" in u:
        raise ValueError(f"unsafe repo url (remote-helper transport not allowed): {url!r}")


def _git(args: list[str], *, check: bool = True, timeout: float | None = None) -> str:
    t = _git_timeout(120.0 if timeout is None else timeout)
    try:
        proc = subprocess.run(
            args,
            check=False,
            capture_output=True,
            text=True,
            timeout=t if t > 0 else None,
            env={**os.environ, **_GIT_SAFE_ENV},
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
    )[:20000]


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


def _embed_cache_model(embedder: EmbedderBase, cfg: Config) -> str:
    # STEmbedder.model_name includes its pinned revision; model_path alone does
    # not. The revision is part of the vector space and therefore of every
    # content-addressed embedding-cache key.
    model_name = getattr(embedder, "model_name", None)
    if model_name:
        return str(model_name)
    model_path = getattr(embedder, "model_path", None)
    if model_path:
        return str(model_path)
    from memo.embedder_select import active_embedder_identity

    return active_embedder_identity(cfg)


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
        # Clamp the overlap to half the chunk's own line span: on long-line
        # files (paragraph-per-line prose, semi-minified) a chunk spans fewer
        # than overlap_lines lines, and a fixed retreat would degrade into a
        # 1-line sliding window of near-duplicate chunks.
        start = max(start + 1, end - min(overlap_lines, (end - start) // 2))
    return chunks


def _split_long_line(line: str, target_chars: int) -> list[str]:
    target = max(1, target_chars)
    return [line[i : i + target] for i in range(0, len(line), target)]


# ---------------------------------------------------------------------------
# Symbol-aligned chunking (codegraph) — MEMO_REPO_CHUNK_SYMBOL_ALIGNED
# ---------------------------------------------------------------------------

# Node kinds whose start/end lines are usable cut boundaries. Deliberately
# excludes structural kinds: 'file' spans the whole file (would collapse the
# file into one giant "symbol") and 'import'/'variable'/'constant' rows are
# line-level noise for chunking purposes.
_SYMBOL_KINDS = ("function", "method", "class", "property")

# A single symbol larger than this factor × target_chars falls back to the
# char-based cutter *within* the symbol, so a giant function never produces
# an unbounded chunk.
SYMBOL_CHUNK_HARD_FACTOR = 2


def _symbol_chunking_enabled() -> bool:
    from memo.flags import flag_bool

    return bool(flag_bool("MEMO_REPO_CHUNK_SYMBOL_ALIGNED"))


def _codegraph_db_for(repo_root: Path) -> Path:
    """Conventional codegraph DB path for a repo checkout.

    The `.codegraph/codegraph.db` layout is owned by `memo.codegraph_loader`;
    derive the relative part from its constants so the convention has a
    single source of truth.
    """
    from memo import codegraph_loader

    rel = codegraph_loader.CODEGRAPH_DB.relative_to(codegraph_loader.CODEGRAPH_DIR.parent)
    return repo_root / rel


class _RepoSymbolSpans:
    """Per-run read-only view over a repo's codegraph symbol spans.

    Resolves the codegraph DB from the given candidate roots (first hit wins)
    and opens ONE sqlite connection per indexing run — lazily, on the first
    lookup — never one per file. Any failure (missing DB, foreign schema)
    degrades to "no symbols", so callers fall back to the char-based chunker.
    """

    def __init__(self, *roots: Path) -> None:
        self._db_path = next(
            (db for db in (_codegraph_db_for(root) for root in roots) if db.is_file()),
            None,
        )
        self._conn: sqlite3.Connection | None = None
        self._failed = False

    def spans_for(self, rel_path: str) -> list[tuple[int, int]]:
        """Symbol (start_line, end_line) spans for a repo-relative file path."""
        if self._db_path is None or self._failed:
            return []
        if self._conn is None:
            try:
                self._conn = sqlite3.connect(f"file:{self._db_path}?mode=ro", uri=True)
            except sqlite3.Error:
                self._failed = True
                return []
        try:
            rows = self._conn.execute(
                "SELECT start_line, end_line FROM nodes "
                "WHERE file_path = ? AND kind IN (?, ?, ?, ?) "
                "ORDER BY start_line, end_line",
                (rel_path, *_SYMBOL_KINDS),
            ).fetchall()
        except sqlite3.Error:
            self._failed = True
            return []
        return [(int(s), int(e)) for s, e in rows]

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None


def _normalize_symbol_spans(
    spans: Sequence[tuple[int, int]], line_count: int
) -> list[tuple[int, int]]:
    """Sort, clamp, and de-nest raw codegraph spans into disjoint cut regions.

    Containers (a class wrapping its methods) are replaced by their leaf
    members — the finest symbol boundaries; container-only lines (class
    header, docstring, attributes) become gap lines. Overlapping siblings
    keep the earliest span. 1-based inclusive in and out.
    """
    clamped = sorted({(start, min(end, line_count)) for start, end in spans if 1 <= start <= end})
    clamped = [(s, e) for s, e in clamped if s <= line_count]
    if not clamped:
        return []
    # Keep leaves: drop any span that contains another (distinct) span.
    leaves = [
        span
        for span in clamped
        if not any(
            other != span and span[0] <= other[0] and other[1] <= span[1] for other in clamped
        )
    ]
    out: list[tuple[int, int]] = []
    last_end = 0
    for start, end in leaves:
        if start <= last_end:
            continue  # overlapping sibling — first one wins
        out.append((start, end))
        last_end = end
    return out


def _chunk_lines_symbol_aligned(
    lines: list[str],
    spans: Sequence[tuple[int, int]],
    *,
    target_chars: int = DEFAULT_CHUNK_TARGET_CHARS,
    overlap_lines: int = DEFAULT_CHUNK_OVERLAP_LINES,
) -> list[tuple[int, int, int, str]] | None:
    """Chunk `lines` cutting at codegraph symbol boundaries.

    Returns None when the spans are unusable (empty after normalization) so
    the caller falls back to `_chunk_lines`. Greedy grouping: consecutive
    segments (symbols, plus the gap lines between them) accumulate until the
    target size; a symbol that alone exceeds the target keeps its own whole
    chunk; a segment beyond SYMBOL_CHUNK_HARD_FACTOR × target falls back to
    the char-based cutter *within* the segment (same target/overlap as the
    default chunker, line numbers re-offset).
    """
    if not lines:
        return None
    norm = _normalize_symbol_spans(spans, len(lines))
    if not norm:
        return None

    # Ordered segments covering every line: (start0, end0) 0-based inclusive,
    # alternating symbol spans and the gaps around them.
    segments: list[tuple[int, int]] = []
    cursor = 0
    for start, end in norm:
        if start - 1 > cursor:
            segments.append((cursor, start - 2))
        segments.append((start - 1, end - 1))
        cursor = end
    if cursor < len(lines):
        segments.append((cursor, len(lines) - 1))

    def _chars(a: int, b: int) -> int:
        return sum(len(lines[i]) + 1 for i in range(a, b + 1))

    out: list[tuple[int, int, int, str]] = []
    seq = 0
    group_start: int | None = None
    group_end = -1
    group_chars = 0

    def _flush() -> None:
        nonlocal seq, group_start, group_chars
        if group_start is None:
            return
        out.append(
            (seq, group_start + 1, group_end + 1, "\n".join(lines[group_start : group_end + 1]))
        )
        seq += 1
        group_start = None
        group_chars = 0

    hard_limit = SYMBOL_CHUNK_HARD_FACTOR * target_chars
    for seg_start, seg_end in segments:
        size = _chars(seg_start, seg_end)
        if size > hard_limit:
            # Giant symbol (or gap): char-based cuts inside the segment.
            _flush()
            for _, sub_start, sub_end, body in _chunk_lines(
                lines[seg_start : seg_end + 1],
                target_chars=target_chars,
                overlap_lines=overlap_lines,
            ):
                out.append((seq, seg_start + sub_start, seg_start + sub_end, body))
                seq += 1
            continue
        if group_start is not None and group_chars + size > target_chars:
            _flush()
        if group_start is None:
            group_start = seg_start
        group_end = seg_end
        group_chars += size
    _flush()
    return out or None


def _language_for_path(path: str) -> str:
    suffix = Path(path).suffix.lower().lstrip(".")
    return suffix or "text"
