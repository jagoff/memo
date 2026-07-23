"""`memo ingest` — bulk vault intake into the corpus.

Extracted from cli_capture.py (god-module decomposition). Registered onto the
root group in cli.py. Carries its own ingest helpers (`_resolve_ingest_row`,
`_is_high_signal`, `_extract_first_h1`) and the high-signal tag/URL constants,
which nothing outside this command uses.
"""

from __future__ import annotations

import contextlib
import hashlib
import re
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

import click

from memo.cli_common import console
from memo.config import Config

# Cap on the error root causes echoed in the final summary (full per-file
# detail stays behind MEMO_INGEST_DEBUG).
_MAX_ERROR_SAMPLES = 5


@contextlib.contextmanager
def _ingest_error_boundary(
    context: str,
    *,
    strict: bool,
    debug: bool,
    debug_label: str,
    note_error: Callable[[str, Exception], None],
    on_done: Callable[[], None],
) -> Iterator[None]:
    """Apply the command's shared strict/debug policy to one ingest item."""
    try:
        yield
    except Exception as exc:
        if strict:
            raise
        note_error(context, exc)
        if debug:
            console.print(f"[red]{debug_label}[/] {context}: {exc}")
    finally:
        on_done()


def _chunk_title(title: str, heading: str, seq: int, count: int) -> str:
    """Return the stable display title for one emitted chunk."""
    if heading:
        return f"{title} — {heading}"
    return f"{title} (§{seq + 1}/{count})"


def _resolve_ingest_row(store, path_str):
    """Resolve the (id, existing-row) an ingest path should write to.

    The vault lives on a case-insensitive filesystem (APFS), so the same
    file can be walked under different directory casing (`notes/Foo.md`
    vs `Notes/Foo.md`). A fresh sha256(path) id would differ per casing
    and mint duplicate rows on re-ingest. So first look for an existing
    row by case-insensitive path and reuse ITS id — the upsert then
    updates that row in place instead of inserting a duplicate. This
    needs no id migration: existing rows keep their original id. Only a
    genuinely new file (no row under any casing) mints a fresh id.
    """

    existing = store.get_by_path_ci(path_str)
    if existing is not None:
        return existing["id"], existing
    id_ = hashlib.sha256(path_str.encode("utf-8")).hexdigest()[:32]
    return id_, store.get(id_)


@click.command(name="ingest")
@click.argument("vault_path", type=click.Path(exists=True, file_okay=False, resolve_path=True))
@click.option(
    "--name", default=None, help="Vault label (default: dirname). Used as path prefix in store."
)
@click.option("--force", is_flag=True, help="Re-embed even if body unchanged.")
@click.option("--dry-run", is_flag=True, help="Walk + report counts, don't embed/write.")
@click.option(
    "--exclude",
    multiple=True,
    help="Glob to exclude (relative to vault). Repeat. Default: .obsidian/.git/.trash/.makemd/.smart-env/.space/Obsidian/AI/",
)
@click.option(
    "--ocr/--no-ocr",
    default=True,
    help="Run OCR on ![[image]] embeds inside notes (Apple Vision). Default on.",
)
@click.option(
    "--chunk/--no-chunk",
    default=True,
    help="Semantically chunk markdown/PDF bodies for better retrieval precision. Default on.",
)
@click.option(
    "--chunk-chars",
    default=1500,
    show_default=True,
    type=int,
    help="Target chunk size in characters.",
)
@click.option(
    "--chunk-overlap",
    default=250,
    show_default=True,
    type=int,
    help="Overlap between consecutive chunks.",
)
@click.option(
    "--include-pdf/--no-include-pdf",
    default=True,
    help="Extract text from .pdf via pdftotext + chunk + embed.",
)
@click.option(
    "--include-orphan-images/--no-include-orphan-images",
    default=True,
    help="OCR images not referenced by any note and ingest them as standalone memories.",
)
@click.option(
    "--include-audio/--no-include-audio",
    default=False,
    help="Transcribe vault audio (m4a/mp3/wav/aac/ogg/flac/opus) via mlx-whisper "
    "+ chunk + embed. Requires the optional mlx-whisper dep "
    "(pip install 'mlx-memo[multimodal]'). Default off.",
)
@click.option(
    "--prune/--no-prune",
    default=False,
    help="Delete stale vault-ingest chunks under this label: files moved/renamed/deleted (abs_path gone) and leftover chunks of notes edited down to fewer chunks. Default off (ingest is purely additive); scheduled ingestion should pass --prune so the index self-heals.",
)
def ingest(
    vault_path: str,
    name: str | None,
    force: bool,
    dry_run: bool,
    exclude: tuple[str, ...],
    ocr: bool,
    chunk: bool,
    chunk_chars: int,
    chunk_overlap: int,
    include_pdf: bool,
    include_orphan_images: bool,
    include_audio: bool,
    prune: bool,
) -> None:
    """Bulk-ingest all .md from a vault into the memo index.

    Walks `<vault_path>/**/*.md`, embeds each, stores under path
    `<name>/<rel-path>`. Files with `id:` in frontmatter are skipped
    (those are curated memories managed by `memo reindex`).

    The user's .md files are NOT modified — we synthesize ids from
    path hash and write only to `~/.local/share/memo/memvec.db`.

    Idempotent: re-running skips files whose body_hash matches the
    indexed value. Use --force to re-embed everything (e.g. after
    embedder model swap).

    Default exclusions skip Obsidian system dirs (.obsidian/, .trash/,
    etc.) and memo's own memory subtree (`<SYSTEM_DIR>/AI/`) so we
    don't double-index curated memories. Note: sibling user content
    under `<SYSTEM_DIR>/` — `Contacts/`, `99-Forms/`, `99-Templates/`
    — IS indexed (e.g. `<SYSTEM_DIR>/Contacts/Grecia.md`).

    A `.memoignore` file in the vault root adds further exclusions (one
    pattern per line, `#` comments allowed) — the durable way to drop a
    folder like `04-Archive/` without editing the launchd ingest command.
    """
    from pathlib import Path

    import frontmatter

    from memo.audio_transcribe import (
        AUDIO_EXTENSIONS,
        transcribe_audio_cached,
        whisper_available,
    )
    from memo.chunker import chunk_markdown
    from memo.embedder import assert_valid_embedding
    from memo.embedder_select import make_embedder
    from memo.flags import flag_str as _flag_str
    from memo.ingest_helpers import (
        IMAGE_EXTENSIONS,
        caption_if_ocr_weak,
        enrich_with_ocr,
        extract_pdf_text,
        find_orphan_images,
        pdftotext_available,
    )
    from memo.store import VecStore

    cfg = Config.from_env()
    cfg.ensure_dirs()

    vault = Path(vault_path).resolve()
    # `cfg.vault_path` is the user's "primary" Obsidian vault (set via
    # `memo init`'s Obsidian branch, or `MEMO_VAULT_PATH`). When we're
    # ingesting that exact vault, paths are stored without a label
    # prefix (e.g. `01-Projects/foo.md`); external vaults get a
    # `<label>/` prefix so multiple vaults coexist in one store.
    is_principal_vault = cfg.vault_path is not None and vault == cfg.vault_path
    label = "" if is_principal_vault else (name or vault.name)

    default_excludes = (
        ".obsidian",
        ".git",
        ".trash",
        ".makemd",
        ".smart-env",
        ".space",
        ".claude",
        ".devin",
        (_flag_str("MEMO_VAULT_SYSTEM_DIR") or "Obsidian") + "/AI",
        # Archived notes are not durable knowledge — they pollute recall. Kept
        # out by default (any depth, case-insensitive — see `_excluded`) so the
        # exclusion does not depend on a per-vault `.memoignore` surviving.
        "04-Archive",
        "Archive",
        "archive",
    )
    # `.memoignore` in the vault root lets the user exclude folders durably,
    # without touching the (auto-regenerated) launchd ingest invocation. One
    # pattern per line; `#` comments and blank lines ignored. Patterns match
    # like --exclude: a path prefix or a `/segment/` anywhere in the rel path.
    memoignore_patterns: list[str] = []
    memoignore = vault / ".memoignore"
    if memoignore.is_file():
        for line in memoignore.read_text(encoding="utf-8").splitlines():
            pat = line.split("#", 1)[0].strip().strip("/")
            if pat:
                memoignore_patterns.append(pat)
    exclude_patterns = list(exclude) + memoignore_patterns + list(default_excludes)

    def _excluded(rel: Path) -> bool:
        s = str(rel)
        padded = f"/{s}/"
        # Lowercased forms for the case-insensitive literal/segment match below.
        # APFS is case-insensitive, so a folder physically named `Archive` can be
        # walked as `archive`/`ARCHIVE`; a case-sensitive compare would miss it.
        s_low = s.lower()
        padded_low = padded.lower()
        for pat in exclude_patterns:
            # A trailing `/**` means "this directory and everything under it".
            # The launchd ingest invocation passes patterns in this form
            # (`Obsidian/Whatsapp/**`); without this they silently no-op and the
            # subtree gets double-ingested by both the generic and dedicated
            # importers. Kept case-sensitive — user globs are explicit.
            if pat.endswith("/**"):
                base = pat[:-3]
                if s == base or s.startswith(base + "/") or f"/{base}/" in padded:
                    return True
                continue
            # Literal path prefix (component boundary — `Archive` must exclude
            # `Archive/x.md` but NOT `Archived Projects/x.md`) or `/segment/`
            # anywhere in the rel path — matched case-insensitively so
            # archive-folder casing variants are caught.
            pat_low = pat.lower()
            if (
                s == pat
                or s.startswith(pat + "/")
                or f"/{pat}/" in padded
                or s_low == pat_low
                or s_low.startswith(pat_low + "/")
                or f"/{pat_low}/" in padded_low
            ):
                return True
            # General globs (`*.tmp`, `a/*/b`) match against the full rel path.
            # Kept case-sensitive — user globs are explicit.
            if ("*" in pat or "?" in pat or "[" in pat) and fnmatch(s, pat):
                return True
        return False

    md_files: list[Path] = []
    pdf_files: list[Path] = []
    audio_files: list[Path] = []
    for p in vault.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(vault)
        if _excluded(rel):
            continue
        suffix = p.suffix.lower()
        if suffix == ".md":
            md_files.append(p)
        elif suffix == ".pdf" and include_pdf:
            pdf_files.append(p)
        elif suffix in AUDIO_EXTENSIONS and include_audio:
            audio_files.append(p)
    md_files.sort()
    pdf_files.sort()
    audio_files.sort()

    pdf_supported = include_pdf and pdftotext_available()
    if include_pdf and not pdf_supported:
        console.print("[yellow]pdftotext not found on PATH — skipping PDFs[/yellow]")
        pdf_files = []

    audio_supported = include_audio and whisper_available()
    if include_audio and not audio_supported:
        console.print("[yellow]mlx-whisper not installed — skipping audio[/yellow]")
        audio_files = []

    console.print(
        f"[cyan]found[/cyan] {len(md_files)} .md, {len(pdf_files)} .pdf, "
        f"{len(audio_files)} audio in {label} (after exclusions)"
    )

    if dry_run:
        console.print("[dim](dry-run — exiting before embed/write)[/dim]")
        for p in md_files[:5]:
            console.print(f"  · {p.relative_to(vault)}")
        if len(md_files) > 5:
            console.print(f"  · …and {len(md_files) - 5} more")
        if pdf_files:
            console.print(f"  · PDFs: {len(pdf_files)}")
        # Pre-flight: warn if existing index has different dims than current
        # config — a real ingest would fail at the first upsert with a
        # cryptic "dimension mismatch" error. Catching it here makes dry-run
        # actually useful for smoke-testing before a production ingest.
        from memo.config import _index_embedder_profile

        profile = _index_embedder_profile(cfg.db_path)
        if profile is not None:
            _, index_dims = profile
            if index_dims != cfg.embedder_dims:
                console.print(
                    f"[red]⚠ dims mismatch:[/red] index is {index_dims}D "
                    f"but config expects {cfg.embedder_dims}D — "
                    f"run 'memo reindex --rebuild' before ingesting."
                )
        return

    embedder = make_embedder(cfg)
    from memo.embedder_select import active_embedder_identity

    store = VecStore(
        cfg.db_path,
        dims=cfg.embedder_dims,
        embedder_model=active_embedder_identity(cfg),
        vec_quant=_flag_str("MEMO_VEC_QUANTIZE"),
    )

    skipped_id = skipped_empty = skipped_unchanged = added = updated = errors = 0
    skipped_pdf_empty = pdf_added = orphan_added = orphan_skipped = 0
    audio_added = skipped_audio_empty = 0
    chunks_emitted = pruned = 0
    referenced_images: set[Path] = set()
    # Abs-paths of every file seen on disk this walk (md + pdf + orphan imgs),
    # added BEFORE any skip so existing-but-skipped files are kept. The --prune
    # sweep deletes label rows whose abs_path is NOT here (file gone from disk).
    # Content-state skips (gained id:, emptied, trimmed under min-chars)
    # deliberately withdraw the file again so their now-stale rows get pruned.
    seen_abs: set[str] = set()

    def _reconcile_file(store_path: str, valid_paths: set[str]) -> None:
        """Drop rows of one just-emitted file whose path is no longer valid —
        e.g. a multi-chunk note edited down to fewer chunks leaves stale
        `#chunk-N` rows, or a single↔multi flip leaves the other shape."""
        nonlocal pruned
        for row in store.file_rows(store_path):
            if row["path"] not in valid_paths and store.delete(row["id"]):
                pruned += 1

    from memo.flags import flag_bool, flag_int

    min_chars_flag = flag_int("MEMO_INGEST_MIN_CHARS")
    min_chars = min_chars_flag if min_chars_flag is not None else 200
    strict_mode = flag_bool("MEMO_INGEST_STRICT")
    debug_mode = flag_bool("MEMO_INGEST_DEBUG")

    # First few error root causes, echoed in the final summary so `errors=N`
    # is diagnosable without MEMO_INGEST_DEBUG.
    error_samples: list[str] = []

    def _note_error(context: str, exc: Exception) -> None:
        nonlocal errors
        errors += 1
        if len(error_samples) < _MAX_ERROR_SAMPLES:
            error_samples.append(f"{context}: {exc}")

    from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeRemainingColumn

    def _emit_record(
        *,
        store_path: str,
        title: str,
        tags: list[str],
        body: str,
        abs_path: Path,
        source: str,
        extra_meta: dict | None = None,
        health_confidence: float | None = None,
    ) -> str | None:
        """Embed `title + body` (chunked if --chunk and large) and upsert
        one row per chunk. Returns "added" / "updated" / None on error
        or unchanged-skip.

        Single-chunk path keeps the canonical store_path so dedup +
        idempotence keep working. Multi-chunk path suffixes
        `#chunk-N` to the store_path so each chunk is its own row.
        """
        nonlocal chunks_emitted, skipped_unchanged
        body, tags = _redact_secrets_for_index(body, tags)
        # Universal text-quality gate: down-weight garbled records (mojibake from
        # any source — pdftotext, broken encodings, future OCR) so they rank below
        # clean notes. Combined with any per-source signal already passed (e.g.
        # OCR mean-confidence for images): keep the LOWER confidence.
        from .text_quality import text_health_confidence as _text_conf

        _tq = _text_conf(body)
        if _tq is not None:
            health_confidence = _tq if health_confidence is None else min(health_confidence, _tq)
        composed_full = f"{title}\n\n{body}"
        if chunk and len(composed_full) > chunk_chars:
            pieces = chunk_markdown(
                composed_full, target_chars=chunk_chars, overlap_chars=chunk_overlap
            )
        else:
            pieces = None  # single-vector path

        if pieces is None or len(pieces) <= 1:
            composed = composed_full[: cfg.max_content_chars]
            id_, existing = _resolve_ingest_row(store, store_path)
            body_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]
            # Mirror of the multi-chunk skip: an unchanged body (PDF/audio/
            # orphan-image re-run) is neither re-embedded nor upserted, so
            # `updated` keeps its original timestamp.
            if existing and existing["body_hash"] == body_hash and not force:
                skipped_unchanged += 1
                return None
            try:
                emb = embedder.embed([composed])[0]
                assert_valid_embedding(emb, cfg.embedder_dims, context=str(abs_path))
            except Exception as exc:
                _note_error(str(abs_path), exc)
                if strict_mode:
                    raise
                if debug_mode:
                    console.print(f"[red]reject:[/] {exc}")
                return None
            now = datetime.now(UTC).isoformat()
            extra: dict[str, Any] = {"source": source, "vault": label, "abs_path": str(abs_path)}
            if extra_meta:
                extra.update(extra_meta)
            store.upsert(
                id_=id_,
                path=store_path,
                title=title[:200],
                type_="reference",
                tags=tags,
                created=existing["created"] if existing else now,
                updated=now,
                body_hash=body_hash,
                embedding=emb,
                extra=extra,
                body_text=body,
            )
            chunks_emitted += 1
            if health_confidence is not None:
                store.set_confidence_batch([(id_, health_confidence)])
            if prune:
                _reconcile_file(store_path, {store_path})
            return "updated" if existing else "added"

        # Multi-chunk path. Each chunk = own meta row; parent_path lets
        # the chat-ask dedup collapse chunks back to one source.
        any_added = any_updated = False
        emitted_ids: list[str] = []
        for piece in pieces:
            seq = piece["seq"]
            heading = piece["heading"]
            chunk_body = piece["body"]
            chunk_path = f"{store_path}#chunk-{seq}"
            id_, existing = _resolve_ingest_row(store, chunk_path)
            chunk_body_hash = hashlib.sha256(chunk_body.encode("utf-8")).hexdigest()[:16]
            if existing and existing["body_hash"] == chunk_body_hash and not force:
                continue
            chunk_composed = chunk_body[: cfg.max_content_chars]
            try:
                emb = embedder.embed([chunk_composed])[0]
                assert_valid_embedding(emb, cfg.embedder_dims, context=f"{abs_path}#chunk-{seq}")
            except Exception as exc:
                _note_error(f"{abs_path}#chunk-{seq}", exc)
                if strict_mode:
                    raise
                if debug_mode:
                    console.print(f"[red]reject:[/] {exc}")
                continue
            now = datetime.now(UTC).isoformat()
            chunk_title = _chunk_title(title, heading, seq, len(pieces))
            extra = {
                "source": source,
                "vault": label,
                "abs_path": str(abs_path),
                "parent_path": store_path,
                "chunk_seq": seq,
                "chunk_count": len(pieces),
                "chunk_heading": heading,
            }
            if extra_meta:
                extra.update(extra_meta)
            store.upsert(
                id_=id_,
                path=chunk_path,
                title=chunk_title[:200],
                type_="reference",
                tags=[*tags, "chunk"],
                created=existing["created"] if existing else now,
                updated=now,
                body_hash=chunk_body_hash,
                embedding=emb,
                extra=extra,
                body_text=chunk_body,
            )
            chunks_emitted += 1
            emitted_ids.append(id_)
            if existing:
                any_updated = True
            else:
                any_added = True
        if health_confidence is not None and emitted_ids:
            store.set_confidence_batch([(i, health_confidence) for i in emitted_ids])
        if prune:
            _reconcile_file(
                store_path,
                {f"{store_path}#chunk-{p['seq']}" for p in pieces},
            )
        if any_added:
            return "added"
        if any_updated:
            return "updated"
        return None

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeRemainingColumn(),
    ) as progress:
        task_id = progress.add_task(f"embed {label}", total=len(md_files))

        for path in md_files:
            with _ingest_error_boundary(
                str(path),
                strict=strict_mode,
                debug=debug_mode,
                debug_label="err",
                note_error=_note_error,
                on_done=lambda: progress.advance(task_id),
            ):
                rel = path.relative_to(vault)
                store_path = f"{label}/{rel}" if label else str(rel)
                seen_abs.add(str(path))

                raw = path.read_text(encoding="utf-8", errors="replace")

                try:
                    fm = frontmatter.loads(raw)
                except Exception as _fm_exc:
                    # Malformed YAML frontmatter often means a corrupt/partial
                    # file; skip it rather than indexing the raw `---` block as
                    # the document body (which pollutes recall with YAML syntax).
                    console.print(
                        f"[yellow]skip (frontmatter parse error):[/yellow] {path.name} — {_fm_exc}"
                    )
                    skipped_empty += 1
                    continue

                # Skip curated memories (have explicit id). This and the two
                # content-state skips below withdraw the file from seen_abs so
                # a previously-indexed row whose note became skippable (gained
                # id:, emptied, trimmed under min-chars) is dropped by the
                # --prune sweep instead of serving stale content forever. The
                # frontmatter-parse-error skip above deliberately stays in
                # seen_abs — a corrupt/partial read must not nuke a good row.
                if fm.metadata.get("id"):
                    skipped_id += 1
                    seen_abs.discard(str(path))
                    continue

                body = fm.content.strip()
                if not body:
                    skipped_empty += 1
                    seen_abs.discard(str(path))
                    continue

                if len(body) < min_chars and not _is_high_signal(body, fm.metadata.get("tags")):
                    skipped_empty += 1
                    seen_abs.discard(str(path))
                    continue

                title = (
                    fm.metadata.get("title")
                    or _extract_first_h1(body)
                    or path.stem.replace("-", " ").replace("_", " ")
                )
                title = str(title).strip() or path.stem

                tags: list[str] = []
                fm_tags: Any = fm.metadata.get("tags") or []
                if isinstance(fm_tags, str):
                    fm_tags = [t.strip() for t in fm_tags.split(",")]
                for t in fm_tags:
                    if t and str(t) not in tags:
                        tags.append(str(t))
                for part in rel.parent.parts:
                    if part and part not in tags:
                        tags.append(part)

                # OCR enrichment — appends <!-- OCR: img.png -->\n<text>
                # blocks for every ![[image]] embed in the note. Tracks
                # resolved image paths so the orphan-image pass below
                # knows which files are already claimed.
                if ocr:
                    enriched, resolved, _ = enrich_with_ocr(
                        body,
                        path,
                        vault,
                        cfg.state_dir,
                    )
                    referenced_images.update(resolved)
                    body = enriched

                body, tags = _redact_secrets_for_index(body, tags)
                body_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]
                _, existing = _resolve_ingest_row(store, store_path)
                if existing and existing["body_hash"] == body_hash and not force:
                    skipped_unchanged += 1
                    continue

                outcome = _emit_record(
                    store_path=store_path,
                    title=title,
                    tags=tags,
                    body=body,
                    abs_path=path,
                    source="vault-ingest",
                )
                if outcome == "added":
                    added += 1
                elif outcome == "updated":
                    updated += 1

        if pdf_files:
            pdf_task = progress.add_task(f"PDF {label}", total=len(pdf_files))
            for pdf_path in pdf_files:
                with _ingest_error_boundary(
                    str(pdf_path),
                    strict=strict_mode,
                    debug=debug_mode,
                    debug_label="err pdf",
                    note_error=_note_error,
                    on_done=lambda: progress.advance(pdf_task),
                ):
                    rel = pdf_path.relative_to(vault)
                    seen_abs.add(str(pdf_path))
                    text = extract_pdf_text(pdf_path).strip()
                    if not text:
                        skipped_pdf_empty += 1
                        continue
                    store_path = f"{label}/{rel}" if label else str(rel)
                    title = pdf_path.stem.replace("-", " ").replace("_", " ")
                    tags = [p for p in rel.parent.parts if p] + ["pdf"]
                    outcome = _emit_record(
                        store_path=store_path,
                        title=title,
                        tags=tags,
                        body=text,
                        abs_path=pdf_path,
                        source="vault-ingest-pdf",
                    )
                    if outcome == "added":
                        pdf_added += 1

        if audio_files:
            audio_task = progress.add_task(f"Audio {label}", total=len(audio_files))
            audio_cache = cfg.state_dir / "audio_cache"
            for audio_path in audio_files:
                with _ingest_error_boundary(
                    str(audio_path),
                    strict=strict_mode,
                    debug=debug_mode,
                    debug_label="err audio",
                    note_error=_note_error,
                    on_done=lambda: progress.advance(audio_task),
                ):
                    rel = audio_path.relative_to(vault)
                    seen_abs.add(str(audio_path))
                    text = transcribe_audio_cached(audio_path, cache_dir=audio_cache).strip()
                    if not text:
                        skipped_audio_empty += 1
                        continue
                    store_path = f"{label}/{rel}" if label else str(rel)
                    title = audio_path.stem.replace("-", " ").replace("_", " ")
                    tags = [p for p in rel.parent.parts if p] + ["audio"]
                    outcome = _emit_record(
                        store_path=store_path,
                        title=title,
                        tags=tags,
                        body=text,
                        abs_path=audio_path,
                        source="vault-ingest-audio",
                        extra_meta={"audio_ext": audio_path.suffix.lower()},
                    )
                    if outcome == "added":
                        audio_added += 1

        if include_orphan_images and ocr:
            orphans = find_orphan_images(vault, referenced_images, excluded_fn=_excluded)
            # Filter image extensions we actually OCR (Apple Vision covers png/jpg/webp/heic).
            orphans = [o for o in orphans if o.suffix.lower() in IMAGE_EXTENSIONS]
            if orphans:
                orphan_task = progress.add_task(f"OCR orphan imgs {label}", total=len(orphans))
                from memo.ocr import (
                    extract_text_cached_with_confidence,
                    image_health_confidence,
                )

                cache_dir = cfg.state_dir / "ocr_cache"
                for img_path in orphans:
                    with _ingest_error_boundary(
                        str(img_path),
                        strict=strict_mode,
                        debug=debug_mode,
                        debug_label="err img",
                        note_error=_note_error,
                        on_done=lambda: progress.advance(orphan_task),
                    ):
                        seen_abs.add(str(img_path))
                        ocr_text_raw, ocr_conf = extract_text_cached_with_confidence(
                            img_path, cache_dir=cache_dir
                        )
                        ocr_text = (ocr_text_raw or "").strip()
                        caption = caption_if_ocr_weak(img_path, ocr_text, cfg.state_dir)
                        if not ocr_text and not caption:
                            orphan_skipped += 1
                            continue
                        if caption and ocr_text:
                            body_text = f"{ocr_text}\n\n<!-- VLM caption -->\n{caption}"
                        else:
                            body_text = ocr_text or caption
                        rel = img_path.relative_to(vault)
                        store_path = f"{label}/{rel}" if label else str(rel)
                        title = img_path.stem.replace("-", " ").replace("_", " ")
                        tags = [p for p in rel.parent.parts if p] + ["standalone-image"]
                        if caption:
                            tags.append("vlm-caption")
                        outcome = _emit_record(
                            store_path=store_path,
                            title=title,
                            tags=tags,
                            body=body_text,
                            abs_path=img_path,
                            source="vault-ingest-image",
                            extra_meta={"image_ext": img_path.suffix.lower()},
                            # Down-weight low-quality screenshot OCR so garbled
                            # captures rank below clean text notes. A caption-only
                            # body is clean model text — leave it neutral (None).
                            health_confidence=(
                                image_health_confidence(ocr_conf) if ocr_text else None
                            ),
                        )
                        if outcome == "added":
                            orphan_added += 1

    # Prune: drop vault-ingest rows under this label whose source file is
    # gone from disk (moved/renamed/deleted). Per-file chunk reconciliation
    # already ran in _emit_record for re-emitted files; this catches whole
    # files that disappeared. source-filtered, so curated memories are safe.
    #
    # Guard: only prune a row if its modality was actually walked this run.
    # A row for an un-walked modality (audio when --include-audio is off,
    # pdf when pdftotext is absent, images when --no-include-orphan-images
    # or --no-ocr) means "not walked, not in seen_abs" — that MUST NOT be
    # treated as "file gone from disk". Scheduled ingestion commonly omits
    # --include-audio, so every vault-ingest-audio row would
    # otherwise be silently deleted every night.
    if prune:
        walked_audio = include_audio and audio_supported
        walked_pdf = include_pdf and pdf_supported
        walked_imgs = include_orphan_images and ocr
        for row in store.vault_ingest_rows(label):
            abs_path = row.get("abs_path")
            if not abs_path or abs_path in seen_abs:
                continue
            suffix = Path(abs_path).suffix.lower()
            if suffix in AUDIO_EXTENSIONS and not walked_audio:
                continue
            if suffix == ".pdf" and not walked_pdf:
                continue
            if suffix in IMAGE_EXTENSIONS and not walked_imgs:
                continue
            if store.delete(row["id"]):
                pruned += 1

    # Bump on-disk schema version so the legacy-paths probe in
    # `Memory._maybe_warn_legacy_paths` doesn't fire for ingest-only
    # vaults (vault-ingest rows live outside `cfg.data_dir` and the
    # probe can't resolve them; setting user_version=1 marks "this DB
    # is post-init, the legacy fallback is no longer relevant").
    if added or updated or pdf_added or orphan_added or audio_added:
        with contextlib.suppress(Exception):
            store.set_user_version(1)
        # Flush WAL after a productive ingest so a subsequent crash doesn't
        # leave a large un-checkpointed WAL (~1MB per ~100 records written).
        with contextlib.suppress(Exception):
            store._checkpoint()

    click.echo(
        "done "
        f"added={added} updated={updated} "
        f"skipped_unchanged={skipped_unchanged} "
        f"skipped_id={skipped_id} skipped_empty={skipped_empty} "
        f"pdf_added={pdf_added} pdf_empty={skipped_pdf_empty} "
        f"orphan_added={orphan_added} orphan_skipped={orphan_skipped} "
        f"audio_added={audio_added} skipped_audio_empty={skipped_audio_empty} "
        f"chunks_emitted={chunks_emitted} pruned={pruned} "
        f"errors={errors}"
    )
    if errors:
        for sample in error_samples:
            console.print(f"[red]error:[/red] {sample}")
        if errors > len(error_samples):
            console.print(
                f"[red]…and {errors - len(error_samples)} more "
                f"(MEMO_INGEST_DEBUG=1 for full detail)[/red]"
            )
        raise SystemExit(1)


_HIGH_SIGNAL_TAGS = frozenset(
    {
        # Notes pinned to lookup-style facts. Lowercase compare; surface
        # forms like "Link" / "LINKS" / "Pago" all match. Spanish + English
        # variants because the vault mixes both.
        "link",
        "links",
        "url",
        "urls",
        "dato",
        "datos",
        "data",
        "ref",
        "refs",
        "referencia",
        "referencias",
        "reference",
        "comando",
        "comandos",
        "command",
        "commands",
        "cmd",
        "snippet",
        "pago",
        "pagos",
        "payment",
        "credencial",
        "credenciales",
        "credential",
        "credentials",
        "endpoint",
        "endpoints",
        "api",
        "telefono",
        "teléfono",
        "phone",
        "tel",
        "cbu",
        "alias",
        "iban",
    }
)

# Match http(s):// URLs — anchored end on whitespace, ), >, ], or "
# (common markdown wrappers). Permissive enough to catch trailing
# punctuation cases without dragging adjacent text in.

_URL_RE = re.compile(r"https?://[^\s)>\]\"]+")


def _redact_secrets_for_index(body: str, tags: list[str]) -> tuple[str, list[str]]:
    """Mask secrets in the text that goes INTO THE INDEX (embedding + fts +
    body_hash). The vault `.md` on disk is never rewritten — markdown stays
    the source of truth; the index is derived, and redaction is deterministic
    so reindex/re-ingest reproduce the same masked rows. Returns the (possibly
    masked) body and a NEW tags list with `_redacted` appended on a hit.
    Pattern/private sanitization is a final persistence invariant; the legacy
    early-redaction flag cannot disable this boundary."""
    from memo.flags import flag_bool
    from memo.redact import sanitize_persisted_text

    res = sanitize_persisted_text(body, entropy=flag_bool("MEMO_REDACT_ENTROPY"))
    if not res.found:
        return body, tags
    out_tags = list(tags)
    if "_redacted" not in out_tags:
        out_tags.append("_redacted")
    return res.text, out_tags


def _is_high_signal(body: str, fm_tags: Any) -> bool:
    """Short notes worth indexing despite being below MIN_CHARS.

    A note is high-signal if any of:
    - frontmatter tags include `link` / `dato` / `ref` / `comando` /
      `pago` / `endpoint` / `cbu` / etc.
    - body contains an http(s) URL
    - body contains a fenced code block (```)

    The user uses these notes as atomic-fact pins (a payment URL, a
    CBU, a one-off shell command). Filtering them by char count
    dropped them from the index even when their title perfectly
    matched a future query. Real example: `Pagar escuela Grecia.md`
    with a 67-char body containing the payment URL.
    """
    if not body:
        return False

    raw_tags: list[str] = []
    if isinstance(fm_tags, list):
        raw_tags = [str(t).strip().lower() for t in fm_tags if t]
    elif isinstance(fm_tags, str):
        raw_tags = [t.strip().lower() for t in fm_tags.split(",") if t.strip()]
    if any(t in _HIGH_SIGNAL_TAGS for t in raw_tags):
        return True

    if _URL_RE.search(body):
        return True

    return "```" in body


def _extract_first_h1(body: str) -> str | None:
    """Return text of the first `# H1` line, or None."""
    for line in body.splitlines():
        s = line.strip()
        if s.startswith("# ") and not s.startswith("##"):
            return s[2:].strip()
        if s and not s.startswith("#"):
            # First non-heading line of content — no H1.
            return None
    return None
