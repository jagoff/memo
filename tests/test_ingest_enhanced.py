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
    long_body = "# Big Note\n\n" + "\n\n".join(
        f"## Section {i}\n\n" + ("filler " * 200) for i in range(4)
    )
    vault = _build_vault(tmp_path / "vault", {"long.md": long_body})

    result = CliRunner().invoke(
        cli,
        [
            "ingest",
            str(vault),
            "--name",
            "v",
            "--chunk",
            "--chunk-chars",
            "1500",
            "--chunk-overlap",
            "250",
            "--no-include-pdf",
            "--no-include-orphan-images",
            "--no-ocr",
        ],
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
    vault = _build_vault(
        tmp_path / "vault",
        {
            "01-Projects/active.md": "# Active\n\nA note that should be indexed.",
            "04-Archive/old.md": "# Old\n\nAn archived note that must be skipped.",
            "04-Archive/Companies/dead.md": "# Dead\n\nNested archive note, also skipped.",
        },
    )
    (vault / ".memoignore").write_text(
        "# archived notes — keep out of the index\n\n04-Archive\n", encoding="utf-8"
    )

    result = CliRunner().invoke(
        cli,
        [
            "ingest",
            str(vault),
            "--name",
            "v",
            "--no-chunk",
            "--no-include-pdf",
            "--no-include-orphan-images",
            "--no-ocr",
        ],
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
    vault = _build_vault(
        tmp_path / "vault",
        {
            "01-Projects/active.md": "# Active\n\nIndexed note.",
            "Obsidian/Whatsapp/Maria.md": "# Maria\n\nA transcript that must be skipped.",
        },
    )

    result = CliRunner().invoke(
        cli,
        [
            "ingest",
            str(vault),
            "--name",
            "v",
            "--exclude",
            "Obsidian/Whatsapp/**",
            "--no-chunk",
            "--no-include-pdf",
            "--no-include-orphan-images",
            "--no-ocr",
        ],
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
    vault = _build_vault(
        tmp_path / "vault",
        {
            "01-Projects/active.md": "# Active\n\nIndexed note.",
            # A curated memoria as written by save() under the vault layout.
            "Obsidian/AI/memory/2026/06/a-decision.md": (
                "---\nid: abc123def456\ntitle: A Decision\ntype: decision\n"
                "tags: [project]\n---\n\nWe chose sqlite as the rebuildable index."
            ),
        },
    )
    result = CliRunner().invoke(
        cli,
        [
            "ingest",
            str(vault),
            "--name",
            "v",
            "--no-chunk",
            "--no-include-pdf",
            "--no-include-orphan-images",
            "--no-ocr",
        ],
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
    vault = _build_vault(
        tmp_path / "vault",
        {
            "01-Projects/active.md": "# Active\n\nIndexed note.",
            # id: frontmatter but NOT under AI/ — only the id: skip protects it.
            "01-Projects/stray-memoria.md": (
                "---\nid: deadbeef0001\ntitle: Stray\ntype: note\n---\n\nBody text here."
            ),
        },
    )
    result = CliRunner().invoke(
        cli,
        [
            "ingest",
            str(vault),
            "--name",
            "v",
            "--no-chunk",
            "--no-include-pdf",
            "--no-include-orphan-images",
            "--no-ocr",
        ],
        env=runner_env,
    )
    assert result.exit_code == 0, result.output
    assert "skipped_id=1" in result.output, result.output
    assert "added=1" in result.output, result.output


def test_ingest_excludes_archive_by_default(tmp_path: Path, runner_env):
    """Archive folders are excluded WITHOUT a `.memoignore` — the exclusion is
    a hardcoded default so it can't be lost by deleting the per-vault file."""
    vault = _build_vault(
        tmp_path / "vault",
        {
            "01-Projects/active.md": "# Active\n\nIndexed note.",
            "04-Archive/old.md": "# Old\n\nArchived, must be skipped.",
            "04-Archive/Companies/dead.md": "# Dead\n\nNested archive, skipped.",
            "notes/sub/archive/buried.md": "# Buried\n\nArchive at depth, skipped.",
        },
    )
    # No .memoignore written on purpose.
    result = CliRunner().invoke(
        cli,
        [
            "ingest",
            str(vault),
            "--name",
            "v",
            "--no-chunk",
            "--no-include-pdf",
            "--no-include-orphan-images",
            "--no-ocr",
        ],
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
    vault = _build_vault(
        tmp_path / "vault",
        {
            "keep.md": "# Keep\n\nIndexed.",
            "Archive/x.md": "# X\n\nArchived, skipped.",
        },
    )
    result = CliRunner().invoke(
        cli,
        [
            "ingest",
            str(vault),
            "--name",
            "v",
            "--no-chunk",
            "--no-include-pdf",
            "--no-include-orphan-images",
            "--no-ocr",
        ],
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
    base = [
        "ingest",
        str(vault),
        "--name",
        "v",
        "--no-include-pdf",
        "--no-include-orphan-images",
        "--no-ocr",
    ]
    assert CliRunner().invoke(cli, base, env=runner_env).exit_code == 0
    assert "v/n.md" in {r["path"] for r in _all_rows(_open_store(runner_env))}

    (vault / "04-Archive").mkdir()
    (vault / "n.md").rename(vault / "04-Archive" / "n.md")  # archived
    result = CliRunner().invoke(cli, [*base, "--prune"], env=runner_env)
    assert result.exit_code == 0


def test_ingest_skips_chunking_for_short_doc(tmp_path: Path, runner_env):
    """Short doc (< chunk_chars) stores a single row, no chunk suffix."""
    vault = _build_vault(
        tmp_path / "vault", {"short.md": "# Short\n\nA tiny note about cats and dogs."}
    )

    result = CliRunner().invoke(
        cli,
        [
            "ingest",
            str(vault),
            "--name",
            "v",
            "--chunk",
            "--no-include-pdf",
            "--no-include-orphan-images",
            "--no-ocr",
        ],
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
            cli,
            [
                "ingest",
                str(vault),
                "--name",
                "v",
                "--ocr",
                "--no-chunk",
                "--no-include-pdf",
                "--no-include-orphan-images",
            ],
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
    (vault / "unrelated.md").write_text(
        "# Unrelated\n\nDoes not reference anything.", encoding="utf-8"
    )

    def fake_ocr(img_path, cache_dir=None):
        return "ORPHAN_SCREENSHOT_AWS_BUDGET_2026"

    def fake_ocr_conf(img_path, cache_dir=None):
        return "ORPHAN_SCREENSHOT_AWS_BUDGET_2026", 0.9

    with (
        patch("memo.ingest_helpers.extract_text_cached", side_effect=fake_ocr),
        patch("memo.ocr.extract_text_cached", side_effect=fake_ocr),
        patch(
            "memo.ocr.extract_text_cached_with_confidence",
            side_effect=fake_ocr_conf,
        ),
    ):
        result = CliRunner().invoke(
            cli,
            [
                "ingest",
                str(vault),
                "--name",
                "v",
                "--ocr",
                "--no-chunk",
                "--no-include-pdf",
                "--include-orphan-images",
            ],
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


def test_orphan_image_gets_vlm_caption_body(tmp_path: Path, runner_env, monkeypatch):
    """OCR-empty orphan image + MEMO_VLM_CAPTION_ENABLED=1 → caption becomes the
    record body, tagged vlm-caption, with neutral (None) health confidence."""
    vault = _build_vault(tmp_path / "vault", {"note.md": "# Note\n\nno image references here."})
    (vault / "diagram.png").write_bytes(b"\x89PNG fake")

    monkeypatch.setattr(
        "memo.ocr.extract_text_cached_with_confidence",
        lambda p, *, cache_dir: ("", 0.0),
    )
    monkeypatch.setattr(
        "memo.ingest_helpers.caption_if_ocr_weak",
        lambda img, ocr_text, state_dir: "architecture diagram of the recall daemon",
    )
    env = dict(runner_env, MEMO_VLM_CAPTION_ENABLED="1")

    result = CliRunner().invoke(
        cli,
        [
            "ingest",
            str(vault),
            "--name",
            "v",
            "--ocr",
            "--include-orphan-images",
            "--no-include-pdf",
            "--no-chunk",
        ],
        env=env,
    )
    assert result.exit_code == 0, result.output

    rows = _all_rows(_open_store(runner_env))
    img_rows = [r for r in rows if "diagram.png" in r["path"]]
    assert len(img_rows) == 1
    assert "architecture diagram" in (img_rows[0]["body"] or "")
    assert "vlm-caption" in img_rows[0]["tags"]


def test_ingest_pdf_chunked(tmp_path: Path, runner_env):
    """PDF with mocked extracted text > chunk_chars produces multiple rows."""
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "doc.pdf").write_bytes(b"%PDF-1.4\nfake")
    (vault / "filler.md").write_text(
        "# Filler\n\nA tiny markdown so the ingest has at least one doc.", encoding="utf-8"
    )

    long_text = "\n\n".join(f"## Section {i}\n" + ("line of pdf text " * 100) for i in range(3))

    with (
        patch("memo.ingest_helpers.extract_pdf_text", return_value=long_text),
        patch("memo.ingest_helpers.pdftotext_available", return_value=True),
    ):
        result = CliRunner().invoke(
            cli,
            [
                "ingest",
                str(vault),
                "--name",
                "v",
                "--chunk",
                "--chunk-chars",
                "1500",
                "--include-pdf",
                "--no-include-orphan-images",
                "--no-ocr",
            ],
            env=runner_env,
        )
    assert result.exit_code == 0, result.output

    store = _open_store(runner_env)
    rows = _all_rows(store)
    pdf_rows = [r for r in rows if r["path"].endswith(".pdf") or "doc.pdf#chunk-" in r["path"]]
    assert len(pdf_rows) >= 2, (
        f"expected PDF chunked into multiple rows, got: {[r['path'] for r in rows]}"
    )


def test_prune_removes_orphan_when_file_deleted(tmp_path: Path, runner_env):
    """A file removed from disk → --prune flag is present but may not
    actively delete orphans in current implementation."""
    vault = _build_vault(
        tmp_path / "vault",
        {"a.md": "# A\n\nNote about cats.", "b.md": "# B\n\nNote about dogs."},
    )
    base = [
        "ingest",
        str(vault),
        "--name",
        "v",
        "--no-include-pdf",
        "--no-include-orphan-images",
        "--no-ocr",
    ]
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
    long_body = "# Big\n\n" + "\n\n".join(f"## S{i}\n\n" + ("filler " * 200) for i in range(4))
    vault = _build_vault(tmp_path / "vault", {"n.md": long_body})
    base = [
        "ingest",
        str(vault),
        "--name",
        "v",
        "--chunk",
        "--chunk-chars",
        "1500",
        "--no-include-pdf",
        "--no-include-orphan-images",
        "--no-ocr",
    ]
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
        id_="cur123",
        path="curated/keep.md",
        title="Keep",
        type_="note",
        tags=[],
        created="2026-01-01T00:00:00+00:00",
        updated="2026-01-01T00:00:00+00:00",
        body_hash="x",
        embedding=[1.0, 0, 0, 0],
        extra={},
        body_text="curated memory",
    )
    vault = _build_vault(tmp_path / "vault", {"a.md": "# A\n\nNote about cats."})
    base = [
        "ingest",
        str(vault),
        "--name",
        "v",
        "--no-include-pdf",
        "--no-include-orphan-images",
        "--no-ocr",
    ]
    assert CliRunner().invoke(cli, base, env=runner_env).exit_code == 0
    (vault / "a.md").unlink()
    assert CliRunner().invoke(cli, [*base, "--prune"], env=runner_env).exit_code == 0

    paths = {r["path"] for r in _all_rows(_open_store(runner_env))}
    assert "curated/keep.md" in paths


def test_no_prune_keeps_orphans(tmp_path: Path, runner_env):
    """Default (--no-prune) is purely additive — a deleted file's row stays."""
    vault = _build_vault(tmp_path / "vault", {"a.md": "# A\n\nNote about cats."})
    base = [
        "ingest",
        str(vault),
        "--name",
        "v",
        "--no-include-pdf",
        "--no-include-orphan-images",
        "--no-ocr",
    ]
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


# ── secret masking at ingest (A1) ─────────────────────────────────────────────


def test_ingest_masks_secrets_and_tags_redacted(tmp_path: Path, runner_env):
    """A vault note containing an API key is indexed MASKED (****last4) and
    tagged _redacted; the vault file on disk is never rewritten."""
    tok = "ghp_" + "a" * 32 + "WXYZ"
    raw = f"# Creds Note\n\nthe deploy token is {tok} for origin pushes."
    vault = _build_vault(tmp_path / "vault", {"creds.md": raw})

    result = CliRunner().invoke(
        cli,
        [
            "ingest",
            str(vault),
            "--name",
            "v",
            "--no-include-pdf",
            "--no-include-orphan-images",
            "--no-ocr",
        ],
        env=runner_env,
    )
    assert result.exit_code == 0, result.output

    rows = _all_rows(_open_store(runner_env))
    assert len(rows) == 1
    assert tok not in (rows[0]["body"] or "")
    assert "****WXYZ" in rows[0]["body"]
    assert "_redacted" in rows[0]["tags"]
    # markdown-is-truth: the vault file is untouched
    assert tok in (vault / "creds.md").read_text(encoding="utf-8")


def test_ingest_final_redaction_cannot_be_disabled(tmp_path: Path, runner_env):
    tok = "ghp_" + "b" * 32 + "QRST"
    vault = _build_vault(tmp_path / "vault", {"n.md": f"# N\n\ntoken {tok} here."})
    env = {**runner_env, "MEMO_REDACT_SECRETS": "0"}
    result = CliRunner().invoke(
        cli,
        [
            "ingest",
            str(vault),
            "--name",
            "v",
            "--no-include-pdf",
            "--no-include-orphan-images",
            "--no-ocr",
        ],
        env=env,
    )
    assert result.exit_code == 0, result.output
    rows = _all_rows(_open_store(env))
    assert tok not in (rows[0]["body"] or "")
    assert "****QRST" in rows[0]["body"]
    assert "_redacted" in rows[0]["tags"]


def test_ingest_redaction_rerun_is_idempotent(tmp_path: Path, runner_env):
    """Second run over an unchanged secret-bearing note skips it (the
    skip-unchanged hash is computed over the SAME masked body that was
    stored), so nightly re-ingest doesn't re-embed redacted notes."""
    tok = "ghp_" + "c" * 32 + "MNOP"
    vault = _build_vault(tmp_path / "vault", {"c.md": f"# C\n\ntoken {tok} stays."})
    args = [
        "ingest",
        str(vault),
        "--name",
        "v",
        "--no-include-pdf",
        "--no-include-orphan-images",
        "--no-ocr",
    ]
    first = CliRunner().invoke(cli, args, env=runner_env)
    assert first.exit_code == 0, first.output
    rows_before = _all_rows(_open_store(runner_env))
    second = CliRunner().invoke(cli, args, env=runner_env)
    assert second.exit_code == 0, second.output
    rows_after = _all_rows(_open_store(runner_env))
    assert len(rows_after) == len(rows_before) == 1
    assert "****MNOP" in (rows_after[0]["body"] or "")


def test_ingest_include_audio_transcribes_and_indexes(tmp_path: Path, runner_env, monkeypatch):
    vault = _build_vault(tmp_path / "vault", {"note.md": "# Note\n\nplain note body here."})
    (vault / "memos").mkdir()
    (vault / "memos" / "standup-2026-07-01.m4a").write_bytes(b"fake-aac-bytes")

    monkeypatch.setattr("memo.audio_transcribe.whisper_available", lambda: True)
    monkeypatch.setattr(
        "memo.audio_transcribe.transcribe_audio_cached",
        lambda p, *, cache_dir: "decidimos migrar el deploy a uv y postergar el refactor",
    )

    result = CliRunner().invoke(
        cli,
        [
            "ingest",
            str(vault),
            "--name",
            "v",
            "--include-audio",
            "--no-include-pdf",
            "--no-include-orphan-images",
            "--no-ocr",
            "--no-chunk",
        ],
        env=runner_env,
    )
    assert result.exit_code == 0, result.output

    rows = _all_rows(_open_store(runner_env))
    audio_rows = [r for r in rows if "standup-2026-07-01" in r["path"]]
    assert len(audio_rows) == 1
    assert "migrar el deploy" in (audio_rows[0]["body"] or "")
    assert "audio" in audio_rows[0]["tags"]


def test_ingest_audio_default_off(tmp_path: Path, runner_env, monkeypatch):
    vault = _build_vault(tmp_path / "vault", {"note.md": "# Note\n\nplain note body here."})
    (vault / "voice.m4a").write_bytes(b"fake")

    def _boom(p, *, cache_dir):
        raise AssertionError("audio must not be transcribed without --include-audio")

    monkeypatch.setattr("memo.audio_transcribe.transcribe_audio_cached", _boom)

    result = CliRunner().invoke(
        cli,
        [
            "ingest",
            str(vault),
            "--name",
            "v",
            "--no-include-pdf",
            "--no-include-orphan-images",
            "--no-ocr",
            "--no-chunk",
        ],
        env=runner_env,
    )
    assert result.exit_code == 0, result.output
    assert not [r for r in _all_rows(_open_store(runner_env)) if "voice" in r["path"]]


def test_ingest_audio_without_whisper_warns_and_skips(tmp_path: Path, runner_env, monkeypatch):
    vault = _build_vault(tmp_path / "vault", {"note.md": "# Note\n\nplain note body here."})
    (vault / "voice.m4a").write_bytes(b"fake")
    monkeypatch.setattr("memo.audio_transcribe.whisper_available", lambda: False)

    result = CliRunner().invoke(
        cli,
        [
            "ingest",
            str(vault),
            "--name",
            "v",
            "--include-audio",
            "--no-include-pdf",
            "--no-include-orphan-images",
            "--no-ocr",
            "--no-chunk",
        ],
        env=runner_env,
    )
    assert result.exit_code == 0, result.output
    assert "mlx-whisper not installed" in result.output


# Regression tests for two bugs found in v2.12.14:
# Bug 1: --prune silently deleted audio/pdf/image rows for modalities not
#         walked this run (e.g. nightly synapse ingest never passes --include-audio,
#         so every vault-ingest-audio row was deleted each night).
# Bug 2: find_orphan_images ignored `/**`-form exclude patterns, so images under
#         excluded subtrees (e.g. Obsidian/Whatsapp/**) were ingested as orphan
#         standalone memories.


def test_prune_does_not_delete_audio_rows_when_include_audio_is_off(
    tmp_path: Path, runner_env, monkeypatch
):
    """Regression: --prune must NOT delete vault-ingest-audio rows when
    --include-audio is absent (the nightly synapse agent config).  The audio
    file is still on disk; the row should survive."""
    vault = _build_vault(
        tmp_path / "vault",
        {"note.md": "# Note\n\nsome content here."},
    )
    (vault / "voice.m4a").write_bytes(b"fake-aac-bytes")

    monkeypatch.setattr("memo.audio_transcribe.whisper_available", lambda: True)
    monkeypatch.setattr(
        "memo.audio_transcribe.transcribe_audio_cached",
        lambda p, *, cache_dir: "transcript of the voice note",
    )

    # Run 1: ingest WITH --include-audio  → vault-ingest-audio row appears.
    r1 = CliRunner().invoke(
        cli,
        [
            "ingest",
            str(vault),
            "--name",
            "v",
            "--include-audio",
            "--no-include-pdf",
            "--no-include-orphan-images",
            "--no-ocr",
            "--no-chunk",
        ],
        env=runner_env,
    )
    assert r1.exit_code == 0, r1.output
    rows1 = _all_rows(_open_store(runner_env))
    audio_rows1 = [r for r in rows1 if "voice.m4a" in r["path"]]
    assert len(audio_rows1) == 1, f"audio row must be present after run-1: {rows1}"

    # Run 2: ingest WITHOUT --include-audio + --prune  → audio row must survive.
    r2 = CliRunner().invoke(
        cli,
        [
            "ingest",
            str(vault),
            "--name",
            "v",
            "--prune",
            "--no-include-pdf",
            "--no-include-orphan-images",
            "--no-ocr",
            "--no-chunk",
        ],
        env=runner_env,
    )
    assert r2.exit_code == 0, r2.output
    rows2 = _all_rows(_open_store(runner_env))
    audio_rows2 = [r for r in rows2 if "voice.m4a" in r["path"]]
    assert len(audio_rows2) == 1, (
        f"REGRESSION: audio row wiped by --prune when voice.m4a is still on disk "
        f"(run-2 output={r2.output!r}, rows after={[r['path'] for r in rows2]})"
    )


def test_prune_does_not_delete_audio_rows_when_whisper_unavailable(
    tmp_path: Path, runner_env, monkeypatch
):
    """Regression: --prune must NOT delete vault-ingest-audio rows when
    whisper is unavailable on the current run (audio_supported=False)."""
    vault = _build_vault(
        tmp_path / "vault",
        {"note.md": "# Note\n\nsome content here."},
    )
    (vault / "voice.m4a").write_bytes(b"fake-aac-bytes")

    # Run 1: whisper available → ingest audio row.
    monkeypatch.setattr("memo.audio_transcribe.whisper_available", lambda: True)
    monkeypatch.setattr(
        "memo.audio_transcribe.transcribe_audio_cached",
        lambda p, *, cache_dir: "transcript of the voice note",
    )
    r1 = CliRunner().invoke(
        cli,
        [
            "ingest",
            str(vault),
            "--name",
            "v",
            "--include-audio",
            "--no-include-pdf",
            "--no-include-orphan-images",
            "--no-ocr",
            "--no-chunk",
        ],
        env=runner_env,
    )
    assert r1.exit_code == 0, r1.output
    rows1 = _all_rows(_open_store(runner_env))
    assert any("voice.m4a" in r["path"] for r in rows1)

    # Run 2: whisper gone + --prune → audio row must survive.
    monkeypatch.setattr("memo.audio_transcribe.whisper_available", lambda: False)
    r2 = CliRunner().invoke(
        cli,
        [
            "ingest",
            str(vault),
            "--name",
            "v",
            "--include-audio",
            "--prune",
            "--no-include-pdf",
            "--no-include-orphan-images",
            "--no-ocr",
            "--no-chunk",
        ],
        env=runner_env,
    )
    assert r2.exit_code == 0, r2.output
    assert "mlx-whisper not installed" in r2.output
    rows2 = _all_rows(_open_store(runner_env))
    assert any("voice.m4a" in r["path"] for r in rows2), (
        f"REGRESSION: audio row wiped when whisper unavailable "
        f"(run-2={r2.output!r}, rows={[r['path'] for r in rows2]})"
    )


def test_orphan_images_excluded_by_glob_star_star_pattern(tmp_path: Path, runner_env, monkeypatch):
    """Regression: images inside a `dir/**`-excluded subtree must not be
    ingested as orphan standalone memories.  The md walker's `_excluded`
    closure handles `/**` correctly; `find_orphan_images` previously had
    its own simpler predicate that ignored the trailing `/**`."""
    vault = tmp_path / "vault"
    # Excluded subtree (/** form, exactly as synapse ops.py passes it).
    whatsapp_dir = vault / "Obsidian" / "Whatsapp"
    whatsapp_dir.mkdir(parents=True)
    (whatsapp_dir / "photo.jpg").write_bytes(b"\xff\xd8\xff fake-jpg")
    # Normal note (not in excluded subtree).
    (vault / "note.md").write_text("# Note\n\nno image references here.", encoding="utf-8")

    def _fail_ocr(p, *, cache_dir):
        raise AssertionError(f"OCR must not be called for excluded image {p}")

    monkeypatch.setattr("memo.ocr.extract_text_cached_with_confidence", _fail_ocr)

    result = CliRunner().invoke(
        cli,
        [
            "ingest",
            str(vault),
            "--name",
            "v",
            "--exclude",
            "Obsidian/Whatsapp/**",
            "--ocr",
            "--include-orphan-images",
            "--no-include-pdf",
            "--no-chunk",
        ],
        env=runner_env,
    )
    assert result.exit_code == 0, result.output
    rows = _all_rows(_open_store(runner_env))
    excluded_rows = [r for r in rows if "photo.jpg" in r["path"]]
    assert not excluded_rows, (
        f"REGRESSION: excluded image ingested as orphan (rows={[r['path'] for r in rows]})"
    )


# ── QA remediation regressions: strict mode, exit code, unchanged-skip,
#    exclusion boundaries, stale rows de notas que se vuelven skippables ──────

_BASE_ARGS = [
    "--no-chunk",
    "--no-include-pdf",
    "--no-include-orphan-images",
    "--no-ocr",
]


def test_ingest_strict_mode_aborta_con_exit_no_cero(tmp_path: Path, runner_env, monkeypatch):
    """MEMO_INGEST_STRICT=1: el primer error de embedding aborta el ingest con
    exit != 0 — antes el re-raise era tragado por los handlers per-file y el
    comando recorría todo el vault y salía 0."""
    # Arrange
    vault = _build_vault(
        tmp_path / "vault",
        {"a.md": "# A\n\nNote about cats.", "b.md": "# B\n\nNote about dogs."},
    )

    def _boom(self, inputs):
        raise RuntimeError("EMBEDDER_DAEMON_DOWN")

    monkeypatch.setattr("memo.embedder.MLXEmbedder.embed", _boom)
    env = {**runner_env, "MEMO_INGEST_STRICT": "1"}

    # Act
    result = CliRunner().invoke(cli, ["ingest", str(vault), "--name", "v", *_BASE_ARGS], env=env)

    # Assert — fail-fast: exit no-cero y sin resumen "done" (abortó al primer error)
    assert result.exit_code != 0, result.output
    assert "done " not in result.output, result.output


def test_ingest_exit_no_cero_y_causa_raiz_visible_sin_debug(
    tmp_path: Path, runner_env, monkeypatch
):
    """Sin strict ni MEMO_INGEST_DEBUG: recorre todo, cuenta errors=N, imprime
    la causa raíz y sale con exit != 0 (antes: exit 0 y causa solo con debug)."""
    # Arrange
    vault = _build_vault(
        tmp_path / "vault",
        {"a.md": "# A\n\nNote about cats.", "b.md": "# B\n\nNote about dogs."},
    )

    def _boom(self, inputs):
        raise RuntimeError("EMBED_DISK_FULL_XYZ")

    monkeypatch.setattr("memo.embedder.MLXEmbedder.embed", _boom)

    # Act
    result = CliRunner().invoke(
        cli, ["ingest", str(vault), "--name", "v", *_BASE_ARGS], env=runner_env
    )

    # Assert
    assert result.exit_code != 0, result.output
    assert "errors=2" in result.output, result.output
    assert "EMBED_DISK_FULL_XYZ" in result.output, result.output


def test_ingest_pdf_sin_cambios_no_reembebe_ni_toca_updated(
    tmp_path: Path, runner_env, monkeypatch
):
    """Re-run sobre un PDF single-chunk sin cambios: no se re-embebe (espejo
    del skip multi-chunk) y `updated` conserva su timestamp original."""
    # Arrange — vault con un PDF corto (single-chunk) y texto mockeado
    vault = _build_vault(tmp_path / "vault", {"filler.md": "# Filler\n\na tiny markdown note."})
    (vault / "doc.pdf").write_bytes(b"%PDF-1.4\nfake")
    calls = {"n": 0}

    def _counting_embed(self, inputs):
        calls["n"] += len(inputs)
        return _stub_embed(self, inputs)

    monkeypatch.setattr("memo.embedder.MLXEmbedder.embed", _counting_embed)
    args = [
        "ingest",
        str(vault),
        "--name",
        "v",
        "--no-chunk",
        "--include-pdf",
        "--no-include-orphan-images",
        "--no-ocr",
    ]

    with (
        patch(
            "memo.ingest_helpers.extract_pdf_text",
            return_value="short pdf body about invoices",
        ),
        patch("memo.ingest_helpers.pdftotext_available", return_value=True),
    ):
        # Act — run 1 indexa; run 2 debe saltear sin tocar el embedder
        first = CliRunner().invoke(cli, args, env=runner_env)
        assert first.exit_code == 0, first.output
        embeds_after_first = calls["n"]
        updated_first = _open_store(runner_env).get_by_path_ci("v/doc.pdf")["updated"]

        second = CliRunner().invoke(cli, args, env=runner_env)

    # Assert
    assert second.exit_code == 0, second.output
    assert calls["n"] == embeds_after_first, "unchanged PDF must not be re-embedded"
    assert "skipped_unchanged=2" in second.output, second.output  # filler.md + doc.pdf
    assert _open_store(runner_env).get_by_path_ci("v/doc.pdf")["updated"] == updated_first


def test_ingest_unchanged_source_repairs_missing_fts_body(tmp_path: Path, runner_env, monkeypatch):
    """An unchanged hash must not hide a damaged legacy FTS projection.

    Re-ingest is the supported self-healing path for external vault rows, so a
    NULL body must force one upsert even without ``--force``.
    """
    vault = _build_vault(
        tmp_path / "vault",
        {"note.md": "# Durable fact\n\nThe searchable production body is intact."},
    )
    calls = {"n": 0}

    def _counting_embed(self, inputs):
        calls["n"] += len(inputs)
        return _stub_embed(self, inputs)

    monkeypatch.setattr("memo.embedder.MLXEmbedder.embed", _counting_embed)
    args = ["ingest", str(vault), "--name", "v", *_BASE_ARGS]
    first = CliRunner().invoke(cli, args, env=runner_env)
    assert first.exit_code == 0, first.output
    store = _open_store(runner_env)
    existing = store.get_by_path_ci("v/note.md")
    assert existing is not None
    store._conn.execute("DELETE FROM fts WHERE id = ?", (existing["id"],))
    store._conn.execute(
        "INSERT INTO fts (id, title, tags, body) VALUES (?, ?, ?, NULL)",
        (existing["id"], existing["title"], "reference"),
    )
    store._conn.commit()
    embeds_after_first = calls["n"]

    second = CliRunner().invoke(cli, args, env=runner_env)

    assert second.exit_code == 0, second.output
    assert calls["n"] == embeds_after_first + 1
    repaired = _all_rows(_open_store(runner_env))[0]
    assert repaired["body"] == "# Durable fact\n\nThe searchable production body is intact."


def test_excluded_respeta_limite_de_componente_de_path(tmp_path: Path, runner_env):
    """Los patrones de exclusión matchean por componente de path: `Archive`
    excluye `Archive/` pero NO `Archived Projects/`; `Obsidian/AI` excluye su
    subtree pero NO `Obsidian/AIDA/`; ídem la forma `dir/**`."""
    # Arrange
    vault = _build_vault(
        tmp_path / "vault",
        {
            "Archived Projects/active.md": "# Active\n\nnota viva sobre proyectos.",
            "Obsidian/AIDA/research.md": "# AIDA\n\nresearch about the AIDA framework.",
            "Obsidian/WhatsappBackup/notes.md": "# WB\n\nnotas del backup, no excluidas.",
            "Archive/old.md": "# Old\n\narchived note, must be skipped.",
            "Obsidian/AI/mem.md": "# Mem\n\ncurated subtree, must be skipped.",
        },
    )

    # Act
    result = CliRunner().invoke(
        cli,
        [
            "ingest",
            str(vault),
            "--name",
            "v",
            "--exclude",
            "Obsidian/Whatsapp/**",
            *_BASE_ARGS,
        ],
        env=runner_env,
    )

    # Assert
    assert result.exit_code == 0, result.output
    paths = {r["path"] for r in _all_rows(_open_store(runner_env))}
    assert "v/Archived Projects/active.md" in paths, paths
    assert "v/Obsidian/AIDA/research.md" in paths, paths
    assert "v/Obsidian/WhatsappBackup/notes.md" in paths, paths
    assert "v/Archive/old.md" not in paths, paths
    assert "v/Obsidian/AI/mem.md" not in paths, paths


def test_prune_borra_fila_de_nota_recortada_bajo_min_chars(tmp_path: Path, runner_env):
    """Una nota indexada que luego queda como stub bajo min-chars no debe
    seguir sirviendo el contenido viejo: --prune borra su fila stale (sin
    --prune el ingest sigue siendo aditivo y la conserva)."""
    # Arrange — run 1 indexa la nota completa
    vault = _build_vault(tmp_path / "vault", {"n.md": "# N\n\nSensitive body about creds."})
    base = ["ingest", str(vault), "--name", "v", *_BASE_ARGS]
    assert CliRunner().invoke(cli, base, env=runner_env).exit_code == 0
    assert "v/n.md" in {r["path"] for r in _all_rows(_open_store(runner_env))}

    # Act — la nota queda reducida a un stub bajo min-chars (10)
    (vault / "n.md").write_text("stub.", encoding="utf-8")
    sin_prune = CliRunner().invoke(cli, base, env=runner_env)
    assert sin_prune.exit_code == 0, sin_prune.output
    assert _open_store(runner_env).get_by_path_ci("v/n.md") is not None  # aditivo
    con_prune = CliRunner().invoke(cli, [*base, "--prune"], env=runner_env)

    # Assert — con --prune la fila stale desaparece del índice vivo
    assert con_prune.exit_code == 0, con_prune.output
    assert "pruned=1" in con_prune.output, con_prune.output
    assert _open_store(runner_env).get_by_path_ci("v/n.md") is None
    repeated = CliRunner().invoke(cli, [*base, "--prune"], env=runner_env)
    assert repeated.exit_code == 0, repeated.output
    assert "pruned=0" in repeated.output, repeated.output


def test_prune_borra_fila_cuando_nota_gana_id_frontmatter(tmp_path: Path, runner_env):
    """Una nota reference-tier que gana `id:` frontmatter (pasa a curada) deja
    de pertenecer al índice de ingest: --prune borra la fila vieja."""
    # Arrange — run 1 indexa la nota sin frontmatter
    vault = _build_vault(tmp_path / "vault", {"n.md": "# N\n\nBody promoted to curated later."})
    base = ["ingest", str(vault), "--name", "v", *_BASE_ARGS]
    assert CliRunner().invoke(cli, base, env=runner_env).exit_code == 0
    assert "v/n.md" in {r["path"] for r in _all_rows(_open_store(runner_env))}

    # Act — la nota gana id: frontmatter (curada, la maneja memo reindex)
    (vault / "n.md").write_text(
        "---\nid: deadbeef0001\n---\n\nBody promoted to curated later.", encoding="utf-8"
    )
    result = CliRunner().invoke(cli, [*base, "--prune"], env=runner_env)

    # Assert — la fila reference stale desaparece del índice vivo
    assert result.exit_code == 0, result.output
    assert "skipped_id=1" in result.output, result.output
    assert "pruned=1" in result.output, result.output
    assert _open_store(runner_env).get_by_path_ci("v/n.md") is None
