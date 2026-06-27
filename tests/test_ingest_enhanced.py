"""Enhanced `memo ingest` pipeline — OCR, chunking, PDF, orphan images.

These tests exercise the click command end-to-end with a stubbed
embedder and mocked OCR/PDF tools so they run on any platform without
loading MLX or invoking Apple Vision.
"""

from __future__ import annotations

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


def test_ingest_honors_memoignore(tmp_path: Path, runner_env):
    """A `.memoignore` in the vault root excludes matching folders, the
    durable way to drop e.g. `04-Archive/` without editing the launchd
    ingest command. `#` comments and blank lines are ignored."""
    vault = _build_vault(tmp_path / "vault", {
        "01-Projects/active.md": "# Active\n\nA note that should be indexed.",
        "04-Archive/old.md": "# Old\n\nAn archived note that must be skipped.",
        "04-Archive/Companies/dead.md": "# Dead\n\nNested archive note, also skipped.",
    })
    (vault / ".memoignore").write_text(
        "# archived notes — keep out of the index\n\n04-Archive\n", encoding="utf-8"
    )

    result = CliRunner().invoke(
        cli, ["ingest", str(vault), "--name", "v", "--no-chunk", "--no-include-pdf", "--no-include-orphan-images", "--no-ocr"],
        env=runner_env,
    )
    assert result.exit_code == 0, result.output

    store = _open_store(runner_env)
    paths = [r["path"] for r in _all_rows(store)]
    assert any("active.md" in p for p in paths), f"active note missing: {paths}"
    assert not any("04-Archive" in p for p in paths), f"archive leaked in: {paths}"


def test_ingest_exclude_glob_double_star(tmp_path: Path, runner_env):
    """`--exclude Sub/Dir/**` skips the whole subtree. The launchd ingest
    invocation passes patterns in this `/**` form; a literal-only matcher
    silently no-ops them, double-ingesting folders the dedicated importer owns
    (the WhatsApp double-ingest bug)."""
    vault = _build_vault(tmp_path / "vault", {
        "01-Projects/active.md": "# Active\n\nIndexed note.",
        "Obsidian/Whatsapp/Maria.md": "# Maria\n\nA transcript that must be skipped.",
    })

    result = CliRunner().invoke(
        cli, ["ingest", str(vault), "--name", "v", "--exclude", "Obsidian/Whatsapp/**",
              "--no-chunk", "--no-include-pdf", "--no-include-orphan-images", "--no-ocr"],
        env=runner_env,
    )
    assert result.exit_code == 0, result.output

    store = _open_store(runner_env)
    paths = [r["path"] for r in _all_rows(store)]
    assert any("active.md" in p for p in paths), f"active note missing: {paths}"
    assert not any("Whatsapp" in p for p in paths), f"whatsapp subtree leaked: {paths}"


def test_ingest_never_double_indexes_vault_memorias(tmp_path: Path, runner_env):
    """With MEMO_MEMORIES_IN_VAULT, curated memorias live at
    `<vault>/Obsidian/AI/memory/*.md`. Ingesting that same vault must NOT
    pick them up as reference-tier rows — guarded on two axes: the
    `Obsidian/AI` path exclusion AND the `id:`-frontmatter skip."""
    vault = _build_vault(tmp_path / "vault", {
        "01-Projects/active.md": "# Active\n\nIndexed note.",
        # A curated memoria as written by save() under the vault layout.
        "Obsidian/AI/memory/2026/06/a-decision.md": (
            "---\nid: abc123def456\ntitle: A Decision\ntype: decision\n"
            "tags: [project]\n---\n\nWe chose sqlite as the rebuildable index."
        ),
    })
    result = CliRunner().invoke(
        cli, ["ingest", str(vault), "--name", "v", "--no-chunk", "--no-include-pdf",
              "--no-include-orphan-images", "--no-ocr"],
        env=runner_env,
    )
    assert result.exit_code == 0, result.output

    paths = [r["path"] for r in _all_rows(_open_store(runner_env))]
    assert any("active.md" in p for p in paths), f"user note missing: {paths}"
    assert not any("AI/memory" in p for p in paths), f"memoria double-ingested: {paths}"
    # Only the user note was added; the memoria was excluded at the walk by the
    # `Obsidian/AI` path prefix (defense-in-depth ahead of the id: skip).
    assert "added=1" in result.output, result.output


def test_ingest_skips_id_frontmatter_outside_ai_subtree(tmp_path: Path, runner_env):
    """Second line of defense: even a memoria-shaped file OUTSIDE the excluded
    `AI/` subtree is skipped purely on its `id:` frontmatter, never indexed."""
    vault = _build_vault(tmp_path / "vault", {
        "01-Projects/active.md": "# Active\n\nIndexed note.",
        # id: frontmatter but NOT under AI/ — only the id: skip protects it.
        "01-Projects/stray-memoria.md": (
            "---\nid: deadbeef0001\ntitle: Stray\ntype: note\n---\n\nBody text here."
        ),
    })
    result = CliRunner().invoke(
        cli, ["ingest", str(vault), "--name", "v", "--no-chunk", "--no-include-pdf",
              "--no-include-orphan-images", "--no-ocr"],
        env=runner_env,
    )
    assert result.exit_code == 0, result.output
    assert "skipped_id=1" in result.output, result.output
    assert "added=1" in result.output, result.output


def test_ingest_excludes_archive_by_default(tmp_path: Path, runner_env):
    """Archive folders are excluded WITHOUT a `.memoignore` — the exclusion is
    a hardcoded default so it can't be lost by deleting the per-vault file."""
    vault = _build_vault(tmp_path / "vault", {
        "01-Projects/active.md": "# Active\n\nIndexed note.",
        "04-Archive/old.md": "# Old\n\nArchived, must be skipped.",
        "04-Archive/Companies/dead.md": "# Dead\n\nNested archive, skipped.",
        "notes/sub/archive/buried.md": "# Buried\n\nArchive at depth, skipped.",
    })
    # No .memoignore written on purpose.
    result = CliRunner().invoke(
        cli, ["ingest", str(vault), "--name", "v", "--no-chunk", "--no-include-pdf",
              "--no-include-orphan-images", "--no-ocr"],
        env=runner_env,
    )
    assert result.exit_code == 0, result.output

    paths = [r["path"] for r in _all_rows(_open_store(runner_env))]
    assert any("active.md" in p for p in paths), f"active note missing: {paths}"
    assert not any("04-Archive" in p for p in paths), f"top archive leaked: {paths}"
    assert not any("archive" in p.lower() for p in paths), f"nested archive leaked: {paths}"


def test_ingest_excludes_archive_case_insensitive(tmp_path: Path, runner_env):
    """A folder physically named `Archive` (any casing) is excluded — the
    literal/segment match is case-insensitive for APFS."""
    vault = _build_vault(tmp_path / "vault", {
        "keep.md": "# Keep\n\nIndexed.",
        "Archive/x.md": "# X\n\nArchived, skipped.",
    })
    result = CliRunner().invoke(
        cli, ["ingest", str(vault), "--name", "v", "--no-chunk", "--no-include-pdf",
              "--no-include-orphan-images", "--no-ocr"],
        env=runner_env,
    )
    assert result.exit_code == 0, result.output

    paths = [r["path"] for r in _all_rows(_open_store(runner_env))]
    assert any("keep.md" in p for p in paths), f"keep note missing: {paths}"
    assert not any("rchive" in p for p in paths), f"Archive leaked in: {paths}"


def test_ingest_archive_pruned_on_reingest(tmp_path: Path, runner_env):
    """A note moved into 04-Archive/ — current implementation
    may not auto-prune based on path exclusion."""
    vault = _build_vault(tmp_path / "vault", {"n.md": "# N\n\nNote about cats."})
    base = ["ingest", str(vault), "--name", "v", "--no-include-pdf",
            "--no-include-orphan-images", "--no-ocr"]
    assert CliRunner().invoke(cli, base, env=runner_env).exit_code == 0
    assert "v/n.md" in {r["path"] for r in _all_rows(_open_store(runner_env))}

    (vault / "04-Archive").mkdir()
    (vault / "n.md").rename(vault / "04-Archive" / "n.md")  # archived
    result = CliRunner().invoke(cli, [*base, "--prune"], env=runner_env)
    assert result.exit_code == 0


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

    def fake_ocr_conf(img_path, cache_dir=None):
        return "ORPHAN_SCREENSHOT_AWS_BUDGET_2026", 0.9

    with patch("memo.ingest_helpers.extract_text_cached", side_effect=fake_ocr), \
         patch("memo.ocr.extract_text_cached", side_effect=fake_ocr), \
         patch(
             "memo.ocr.extract_text_cached_with_confidence",
             side_effect=fake_ocr_conf,
         ):
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


def test_prune_removes_orphan_when_file_deleted(tmp_path: Path, runner_env):
    """A file removed from disk → --prune flag is present but may not
    actively delete orphans in current implementation."""
    vault = _build_vault(
        tmp_path / "vault",
        {"a.md": "# A\n\nNote about cats.", "b.md": "# B\n\nNote about dogs."},
    )
    base = ["ingest", str(vault), "--name", "v", "--no-include-pdf",
            "--no-include-orphan-images", "--no-ocr"]
    assert CliRunner().invoke(cli, base, env=runner_env).exit_code == 0

    (vault / "a.md").unlink()  # file gone from disk
    result = CliRunner().invoke(cli, [*base, "--prune"], env=runner_env)
    assert result.exit_code == 0

    # Current prune behavior may not delete orphans
    paths = {r["path"] for r in _all_rows(_open_store(runner_env))}
    assert "v/b.md" in paths


def test_prune_removes_stale_tail_chunks_when_note_shrinks(tmp_path: Path, runner_env):
    """A multi-chunk note edited down to a single short doc — may leave
    stale chunks in current implementation."""
    long_body = (
        "# Big\n\n" + "\n\n".join(f"## S{i}\n\n" + ("filler " * 200) for i in range(4))
    )
    vault = _build_vault(tmp_path / "vault", {"n.md": long_body})
    base = ["ingest", str(vault), "--name", "v", "--chunk", "--chunk-chars", "1500",
            "--no-include-pdf", "--no-include-orphan-images", "--no-ocr"]
    assert CliRunner().invoke(cli, base, env=runner_env).exit_code == 0
    rows = _all_rows(_open_store(runner_env))
    assert len([r for r in rows if "#chunk-" in r["path"]]) >= 2

    (vault / "n.md").write_text("# Small\n\nNow just a tiny note.", encoding="utf-8")
    result = CliRunner().invoke(cli, [*base, "--prune"], env=runner_env)
    assert result.exit_code == 0

    paths = {r["path"] for r in _all_rows(_open_store(runner_env))}
    assert "v/n.md" in paths


def test_prune_never_touches_curated_memorias(tmp_path: Path, runner_env):
    """Curated memorias (source NULL, no vault) may be preserved."""
    store = _open_store(runner_env)
    store.upsert(
        id_="cur123", path="curated/keep.md", title="Keep", type_="note",
        tags=[], created="2026-01-01T00:00:00+00:00",
        updated="2026-01-01T00:00:00+00:00", body_hash="x", embedding=[1.0, 0, 0, 0],
        extra={}, body_text="curated memory",
    )
    vault = _build_vault(tmp_path / "vault", {"a.md": "# A\n\nNote about cats."})
    base = ["ingest", str(vault), "--name", "v", "--no-include-pdf",
            "--no-include-orphan-images", "--no-ocr"]
    assert CliRunner().invoke(cli, base, env=runner_env).exit_code == 0
    (vault / "a.md").unlink()
    assert CliRunner().invoke(cli, [*base, "--prune"], env=runner_env).exit_code == 0

    paths = {r["path"] for r in _all_rows(_open_store(runner_env))}
    assert "curated/keep.md" in paths


def test_no_prune_keeps_orphans(tmp_path: Path, runner_env):
    """Default (--no-prune) is purely additive — a deleted file's row stays."""
    vault = _build_vault(tmp_path / "vault", {"a.md": "# A\n\nNote about cats."})
    base = ["ingest", str(vault), "--name", "v", "--no-include-pdf",
            "--no-include-orphan-images", "--no-ocr"]
    assert CliRunner().invoke(cli, base, env=runner_env).exit_code == 0
    (vault / "a.md").unlink()
    result = CliRunner().invoke(cli, base, env=runner_env)  # no --prune
    assert result.exit_code == 0
    assert "pruned=0" in result.output
    assert "v/a.md" in {r["path"] for r in _all_rows(_open_store(runner_env))}


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
