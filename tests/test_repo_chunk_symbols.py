"""Symbol-aligned repo chunking (MEMO_REPO_CHUNK_SYMBOL_ALIGNED, default off).

When the flag is on AND the indexed repo has a codegraph index
(`.codegraph/codegraph.db`), repo_index chunk boundaries align to
function/class symbol spans (nodes.start_line/end_line) instead of the
char-based cut. Flag off must stay byte-identical to the current chunker;
missing DB or symbol-less files fall back silently.
"""

from __future__ import annotations

import sqlite3
import subprocess
from pathlib import Path

import pytest

from memo.config import Config
from memo.repo_index import RepoCorpus
from memo.repo_index_helpers import (
    _chunk_lines,
    _chunk_lines_symbol_aligned,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True)


def _make_text_repo(root: Path, name: str, files: dict[str, str]) -> Path:
    repo = root / name
    for rel, text in files.items():
        path = repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    subprocess.run(["git", "init", str(repo)], check=True, capture_output=True, text=True)
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "initial")
    return repo


def _cfg(tmp_path: Path) -> Config:
    return Config(
        data_dir=tmp_path / "data",
        state_dir=tmp_path / "state",
        embedder_dims=4,
        reranker_enabled=False,
    )


def _seed_codegraph(repo: Path, nodes: list[tuple[str, str, int, int]]) -> None:
    """Create a synthetic codegraph.db at `<repo>/.codegraph/codegraph.db`.

    `nodes` rows are (kind, file_path, start_line, end_line). Created AFTER
    the git commit, so it stays untracked and never reaches the clone —
    exercising the source-checkout resolution path.
    """
    db = repo / ".codegraph" / "codegraph.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE nodes ("
        "id TEXT PRIMARY KEY, kind TEXT NOT NULL, name TEXT NOT NULL, "
        "qualified_name TEXT NOT NULL, file_path TEXT NOT NULL, "
        "start_line INTEGER NOT NULL, end_line INTEGER NOT NULL)"
    )
    for i, (kind, file_path, start, end) in enumerate(nodes):
        conn.execute(
            "INSERT INTO nodes VALUES (?, ?, ?, ?, ?, ?, ?)",
            (f"node-{i}", kind, f"sym{i}", f"mod.sym{i}", file_path, start, end),
        )
    conn.commit()
    conn.close()


def _fn_lines(tag: str, n: int) -> list[str]:
    """`n` lines of stable width (~45 chars) so chunk sizes are predictable."""
    return [f"line-{tag}-{i:03d} " + "x" * 33 for i in range(n)]


def _three_symbol_file() -> tuple[str, list[str]]:
    """A 90-line file shaped as three 30-line 'functions' (no gaps).

    Each line is 45 chars (46 with newline): one function ≈ 1380 chars, two
    fit the default 3500-char target, three do not → the symbol-aligned cut
    must land exactly on the second symbol's end_line (60).
    """
    lines = _fn_lines("a", 30) + _fn_lines("b", 30) + _fn_lines("c", 30)
    return "\n".join(lines) + "\n", lines


_THREE_SYMBOL_NODES = [
    ("file", "src/app.py", 1, 90),  # structural — must NOT become a cut span
    ("import", "src/app.py", 1, 1),  # ditto
    ("function", "src/app.py", 1, 30),
    ("function", "src/app.py", 31, 60),
    ("function", "src/app.py", 61, 90),
]


def _index_and_read_chunks(
    tmp_path: Path, repo: Path, name: str
) -> list[tuple[int, int, int, str]]:
    corpus = RepoCorpus(_cfg(tmp_path), embedder=object())
    out = corpus.index(str(repo), name=name, with_embeddings=False)
    rows = corpus.store.repo_pending_chunks(out["repo_id"], force=True)
    return [
        (r["chunk_seq"], r["line_start"], r["line_end"], r["body_text"])
        for r in rows
        if r["path"] == "src/app.py"
    ]


def test_symbol_span_connection_closes_when_indexing_is_interrupted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _make_text_repo(
        tmp_path,
        "interrupt",
        {"src/app.py": "def alpha():\n    return '" + "x" * 100 + "'\n"},
    )

    class AbortIndex(BaseException):
        pass

    class FakeSymbolSpans:
        closed = False

        def spans_for(self, _rel_path: str) -> list[tuple[int, int]]:
            return []

        def close(self) -> None:
            self.closed = True

    spans = FakeSymbolSpans()
    monkeypatch.setattr("memo.repo_index._symbol_chunking_enabled", lambda: True)
    monkeypatch.setattr("memo.repo_index._RepoSymbolSpans", lambda *_roots: spans)
    corpus = RepoCorpus(_cfg(tmp_path), embedder=object())

    def interrupt(event: str, _payload: dict[str, object]) -> None:
        if event == "file_indexed":
            raise AbortIndex

    with pytest.raises(AbortIndex):
        corpus.index(
            str(repo),
            name="interrupt",
            with_embeddings=False,
            progress=interrupt,
        )

    assert spans.closed is True


# ---------------------------------------------------------------------------
# Unit: _chunk_lines_symbol_aligned
# ---------------------------------------------------------------------------


def test_groups_consecutive_symbols_until_target():
    # Arrange — three 4-line symbols, 10-char lines (11 with newline): 44 each.
    lines = ["aaaaaaaaaa"] * 12
    spans = [(1, 4), (5, 8), (9, 12)]

    # Act — two symbols fit 95 chars, three do not.
    chunks = _chunk_lines_symbol_aligned(lines, spans, target_chars=95)

    # Assert — cut lands on the second symbol's end_line.
    assert chunks is not None
    assert [(seq, ls, le) for seq, ls, le, _ in chunks] == [(0, 1, 8), (1, 9, 12)]
    assert chunks[0][3] == "\n".join(lines[0:8])
    assert chunks[1][3] == "\n".join(lines[8:12])


def test_symbol_exceeding_target_stays_whole():
    # Arrange — middle symbol is 66 chars vs a 60-char target (but under 2x).
    lines = ["aaaaaaaaaa"] * 12
    spans = [(1, 2), (3, 8), (9, 12)]

    # Act
    chunks = _chunk_lines_symbol_aligned(lines, spans, target_chars=60)

    # Assert — the oversized symbol is never split across chunks.
    assert chunks is not None
    assert (3, 8) in {(ls, le) for _, ls, le, _ in chunks}


def test_giant_symbol_falls_back_to_char_cuts_within():
    # Arrange — one symbol spanning the whole file at 110 chars vs 2x40=80 cap.
    lines = ["aaaaaaaaaa"] * 10
    spans = [(1, 10)]

    # Act
    chunks = _chunk_lines_symbol_aligned(lines, spans, target_chars=40)

    # Assert — char-based cuts INSIDE the symbol: several chunks, all within
    # the span, together covering every line.
    assert chunks is not None
    assert len(chunks) > 1
    covered: set[int] = set()
    for _, line_start, line_end, _ in chunks:
        assert 1 <= line_start <= line_end <= 10
        covered.update(range(line_start, line_end + 1))
    assert covered == set(range(1, 11))


def test_unusable_spans_return_none_for_fallback():
    lines = ["aaaaaaaaaa"] * 5

    assert _chunk_lines_symbol_aligned(lines, [], target_chars=40) is None
    # Spans entirely past EOF are dropped → fallback too.
    assert _chunk_lines_symbol_aligned(lines, [(100, 120)], target_chars=40) is None
    assert _chunk_lines_symbol_aligned([], [(1, 2)], target_chars=40) is None


def test_nested_spans_cut_at_leaf_boundaries():
    # Arrange — a class (1,10) containing two methods; leaf spans win.
    lines = ["aaaaaaaaaa"] * 10
    spans = [(1, 10), (2, 5), (6, 9)]

    # Act
    chunks = _chunk_lines_symbol_aligned(lines, spans, target_chars=100)

    # Assert — no chunk boundary falls strictly inside a method span.
    assert chunks is not None
    for _, line_start, line_end, _ in chunks:
        for s, e in ((2, 5), (6, 9)):
            assert not (s < line_start <= e), (line_start, line_end)
            assert not (s <= line_end < e), (line_start, line_end)
    covered: set[int] = set()
    for _, line_start, line_end, _ in chunks:
        covered.update(range(line_start, line_end + 1))
    assert covered == set(range(1, 11))


# ---------------------------------------------------------------------------
# Integration: flag off / on / fallback through RepoCorpus.index
# ---------------------------------------------------------------------------


def test_flag_off_chunks_identical_even_with_db(tmp_path: Path, monkeypatch):
    """Default (flag off) → byte-for-byte the current char-based chunker,
    even when a codegraph.db exists for the repo."""
    monkeypatch.delenv("MEMO_REPO_CHUNK_SYMBOL_ALIGNED", raising=False)
    text, lines = _three_symbol_file()
    repo = _make_text_repo(tmp_path, "flag-off-repo", {"src/app.py": text})
    _seed_codegraph(repo, _THREE_SYMBOL_NODES)

    got = _index_and_read_chunks(tmp_path, repo, "flagoff")

    expected = [(seq, ls, le, body) for seq, ls, le, body in _chunk_lines(lines)]
    assert got == expected


def test_flag_on_cuts_at_symbol_end_lines(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("MEMO_REPO_CHUNK_SYMBOL_ALIGNED", "1")
    text, lines = _three_symbol_file()
    repo = _make_text_repo(tmp_path, "flag-on-repo", {"src/app.py": text})
    _seed_codegraph(repo, _THREE_SYMBOL_NODES)

    got = _index_and_read_chunks(tmp_path, repo, "flagon")

    # Two symbols (1-30, 31-60) group under the 3500-char target; the third
    # starts its own chunk — boundaries are exactly the symbols' end_lines.
    assert [(seq, ls, le) for seq, ls, le, _ in got] == [(0, 1, 60), (1, 61, 90)]
    assert got[0][3] == "\n".join(lines[0:60])
    assert got[1][3] == "\n".join(lines[60:90])


def test_flag_on_without_db_falls_back_silently(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("MEMO_REPO_CHUNK_SYMBOL_ALIGNED", "1")
    text, lines = _three_symbol_file()
    repo = _make_text_repo(tmp_path, "no-db-repo", {"src/app.py": text})
    # No .codegraph/codegraph.db anywhere.

    got = _index_and_read_chunks(tmp_path, repo, "nodb")

    expected = [(seq, ls, le, body) for seq, ls, le, body in _chunk_lines(lines)]
    assert got == expected


def test_flag_on_file_without_symbols_falls_back(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("MEMO_REPO_CHUNK_SYMBOL_ALIGNED", "1")
    text, lines = _three_symbol_file()
    repo = _make_text_repo(tmp_path, "no-syms-repo", {"src/app.py": text})
    # DB exists but has no symbol rows for src/app.py.
    _seed_codegraph(repo, [("function", "src/other.py", 1, 10)])

    got = _index_and_read_chunks(tmp_path, repo, "nosyms")

    expected = [(seq, ls, le, body) for seq, ls, le, body in _chunk_lines(lines)]
    assert got == expected


def test_flag_on_opens_one_db_connection_per_run(tmp_path: Path, monkeypatch):
    """The codegraph DB is opened once per indexing run, never per file."""
    monkeypatch.setenv("MEMO_REPO_CHUNK_SYMBOL_ALIGNED", "1")
    text_a, _ = _three_symbol_file()
    repo = _make_text_repo(
        tmp_path,
        "multi-file-repo",
        {"src/app.py": text_a, "src/lib.py": text_a, "README.md": "# Sample\n\nzebra corpus.\n"},
    )
    _seed_codegraph(
        repo,
        [
            *_THREE_SYMBOL_NODES,
            ("function", "src/lib.py", 1, 30),
            ("function", "src/lib.py", 31, 60),
            ("function", "src/lib.py", 61, 90),
        ],
    )

    real_connect = sqlite3.connect
    codegraph_connects: list[str] = []

    def counting_connect(target, *args, **kwargs):
        if "codegraph.db" in str(target):
            codegraph_connects.append(str(target))
        return real_connect(target, *args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", counting_connect)

    corpus = RepoCorpus(_cfg(tmp_path), embedder=object())
    out = corpus.index(str(repo), name="multi", with_embeddings=False)

    assert out["indexed_files"] == 3
    assert len(codegraph_connects) == 1
