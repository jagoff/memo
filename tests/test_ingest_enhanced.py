"""Enhanced `memo ingest` pipeline — OCR, chunking, PDF, orphan images.

These tests exercise the click command end-to-end with a stubbed
embedder and mocked OCR/PDF tools so they run on any platform without
loading MLX or invoking Apple Vision.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from memo.cli import cli
from memo.store import VecStore


@pytest.fixture
def runner_env(tmp_path: Path) -> dict[str, str]:
    return {
        "MEMO_CONFIG_FILE": str(tmp_path / "memo-config.toml"),
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_DATA_DIR": str(tmp_path / "data"),
        "MEMO_STATE_DIR": str(tmp_path / "state"),
        "MEMO_INGEST_MIN_CHARS": "10",  # keep small test files in
        "MEMO_OCR_ENABLED": "0",  # not used by vault ingest (uses flag) but belt+braces
        "MEMO_EMBEDDER_DIMS": "4",
    }


def _stub_embed(self, inputs):
    """Deterministic 4-dim embedder. Hashes input → bucket vector."""
    out = []
    for s in inputs:
        h = sum(ord(c) for c in (s or "")) % 4
        v = [0.0] * 4
        v[h] = 1.0
        out.append(v)
    return out


@pytest.fixture(autouse=True)
def _stub_mlx(monkeypatch):
    """Replace MLXEmbedder.embed + skip model load so tests don't pull MLX."""
    monkeypatch.setattr("memo.embedder.MLXEmbedder.embed", _stub_embed)
    # Skip the real model load — MLXEmbedder.__init__ checks the model exists.
    monkeypatch.setattr("memo.embedder.MLXEmbedder.__init__", lambda self, **kw: None)


def _build_vault(root: Path, files: dict[str, str]) -> Path:
    """Create a temp vault with the given relative-path → content map."""
    root.mkdir(parents=True, exist_ok=True)
    for rel, content in files.items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    return root


def _open_store(env: dict[str, str]) -> VecStore:
    db_path = Path(env["MEMO_STATE_DIR"]) / "memvec.db"
    return VecStore(db_path, dims=4)


def _all_rows(store: VecStore) -> list[dict]:
    rows = store._conn.execute(
        "SELECT meta.id, meta.path, meta.title, meta.tags, meta.extra_json, fts.body "
        "FROM meta LEFT JOIN fts ON meta.id = fts.id ORDER BY meta.path"
    ).fetchall()
    out = []
    import json as _j
    for r in rows:
        d = dict(r)
        if isinstance(d.get("tags"), str):
            try:
                d["tags"] = _j.loads(d["tags"])
            except Exception:
                # tags column actually stores space-separated string
                d["tags"] = d["tags"].split()
        out.append(d)
    return out


def test_ingest_chunks_long_markdown(tmp_path: Path, runner_env):
    """A note larger than chunk_chars produces multiple meta rows, each
    with parent_path + chunk_seq metadata."""
    long_body = (
        "# Big Note\n\n"
        + "\n\n".join(f"## Section {i}\n\n" + ("filler " * 200) for i in range(4))
    )
    vault = _build_vault(tmp_path / "vault", {"long.md": long_body})

    result = CliRunner().invoke(
        cli, ["ingest", str(vault), "--name", "v", "--chunk", "--chunk-chars", "1500", "--chunk-overlap", "250", "--no-include-pdf", "--no-include-orphan-images", "--no-ocr"],
        env=runner_env,
    )
    assert result.exit_code == 0, result.output

    store = _open_store(runner_env)
    rows = _all_rows(store)
    chunk_rows = [r for r in rows if "#chunk-" in r["path"]]
    assert len(chunk_rows) >= 2, f"expected multi-chunk, got rows: {[r['path'] for r in rows]}"
    parent_paths = {r["extra_json"] for r in chunk_rows if r.get("extra_json")}
    assert parent_paths, "chunks must carry extra_json with parent_path"


def test_ingest_skips_chunking_for_short_doc(tmp_path: Path, runner_env):
    """Short doc (< chunk_chars) stores a single row, no chunk suffix."""
    vault = _build_vault(tmp_path / "vault", {"short.md": "# Short\n\nA tiny note about cats and dogs."})

    result = CliRunner().invoke(
        cli, ["ingest", str(vault), "--name", "v", "--chunk", "--no-include-pdf", "--no-include-orphan-images", "--no-ocr"],
        env=runner_env,
    )
    assert result.exit_code == 0, result.output

    store = _open_store(runner_env)
    rows = _all_rows(store)
    assert len(rows) == 1
    assert "#chunk-" not in rows[0]["path"]


def test_ingest_ocr_embedded_image(tmp_path: Path, runner_env):
    """Note with `![[fake.png]]` gets OCR'd content appended to body."""
    vault = tmp_path / "vault"
    (vault / "attachments").mkdir(parents=True)
    fake_png = vault / "attachments" / "screenshot.png"
    fake_png.write_bytes(b"\x89PNG\r\n\x1a\nfake-png-bytes")
    (vault / "note.md").write_text(
        "# Note\n\nSee the screenshot: ![[screenshot.png]]\n\nThat's the panel.",
        encoding="utf-8",
    )

    def fake_ocr(img_path, cache_dir=None):
        return "OCR_TEXT_PANEL_REVENUE_QUARTERLY"

    with patch("memo.ingest_helpers.extract_text_cached", side_effect=fake_ocr):
        result = CliRunner().invoke(
            cli, ["ingest", str(vault), "--name", "v", "--ocr", "--no-chunk", "--no-include-pdf", "--no-include-orphan-images"],
            env=runner_env,
        )
    assert result.exit_code == 0, result.output

    store = _open_store(runner_env)
    rows = _all_rows(store)
    note_row = next(r for r in rows if r["path"].endswith("note.md"))
    assert "OCR_TEXT_PANEL_REVENUE_QUARTERLY" in (note_row["body"] or "")


def test_ingest_orphan_image_becomes_memoria(tmp_path: Path, runner_env):
    """Image not referenced by any note → standalone memoria with OCR body."""
    vault = tmp_path / "vault"
    (vault / "attachments").mkdir(parents=True)
    orphan = vault / "attachments" / "orphan.png"
    orphan.write_bytes(b"\x89PNG\r\n\x1a\norphan-bytes")
    (vault / "unrelated.md").write_text("# Unrelated\n\nDoes not reference anything.", encoding="utf-8")

    def fake_ocr(img_path, cache_dir=None):
        return "ORPHAN_SCREENSHOT_AWS_BUDGET_2026"

    with patch("memo.ingest_helpers.extract_text_cached", side_effect=fake_ocr), \
         patch("memo.ocr.extract_text_cached", side_effect=fake_ocr):
        result = CliRunner().invoke(
            cli, ["ingest", str(vault), "--name", "v", "--ocr", "--no-chunk", "--no-include-pdf", "--include-orphan-images"],
            env=runner_env,
        )
    assert result.exit_code == 0, result.output

    store = _open_store(runner_env)
    rows = _all_rows(store)
    paths = [r["path"] for r in rows]
    assert any(p.endswith("orphan.png") for p in paths), f"orphan not ingested: {paths}"
    orphan_row = next(r for r in rows if r["path"].endswith("orphan.png"))
    assert "ORPHAN_SCREENSHOT_AWS_BUDGET_2026" in (orphan_row["body"] or "")
    assert "standalone-image" in (orphan_row["tags"] or "")


def test_ingest_pdf_chunked(tmp_path: Path, runner_env):
    """PDF with mocked extracted text > chunk_chars produces multiple rows."""
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "doc.pdf").write_bytes(b"%PDF-1.4\nfake")
    (vault / "filler.md").write_text("# Filler\n\nA tiny markdown so the ingest has at least one doc.", encoding="utf-8")

    long_text = "\n\n".join(f"## Section {i}\n" + ("line of pdf text " * 100) for i in range(3))

    with patch("memo.ingest_helpers.extract_pdf_text", return_value=long_text), \
         patch("memo.ingest_helpers.pdftotext_available", return_value=True):
        result = CliRunner().invoke(
            cli, ["ingest", str(vault), "--name", "v", "--chunk", "--chunk-chars", "1500", "--include-pdf", "--no-include-orphan-images", "--no-ocr"],
            env=runner_env,
        )
    assert result.exit_code == 0, result.output

    store = _open_store(runner_env)
    rows = _all_rows(store)
    pdf_rows = [r for r in rows if r["path"].endswith(".pdf") or "doc.pdf#chunk-" in r["path"]]
    assert len(pdf_rows) >= 2, f"expected PDF chunked into multiple rows, got: {[r['path'] for r in rows]}"


def test_dedup_strips_chunk_suffix(mock_memory):
    """`_norm_dedup_path` collapses `path#chunk-N` to `path` so multi-chunk
    memorias dedup against repo hits / each other in ask context."""
    from memo.memory import _norm_dedup_path

    assert _norm_dedup_path("Notes/foo.md") == "notes/foo.md"
    assert _norm_dedup_path("Notes/foo.md#chunk-0") == "notes/foo.md"
    assert _norm_dedup_path("Notes/foo.md#chunk-12") == "notes/foo.md"
    # Both sides normalise identically — what matters for dedup is that
    # `path` and `path#chunk-N` collapse to the same key.
    assert _norm_dedup_path("Notes/foo.md") == _norm_dedup_path("Notes/foo.md#chunk-5")
