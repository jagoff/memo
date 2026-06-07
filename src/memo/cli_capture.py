"""`memo mine-history` / `ingest` / `capture-stop` / `resume` — corpus intake.

Extracted from cli.py (2b god-module decomposition). Registered onto the root
group in cli.py via `cli.add_command(...)`. Carries its own ingest helpers
(`_resolve_ingest_row`, `_is_high_signal`, `_extract_first_h1`) and the
high-signal tag/URL constants, which nothing outside this cluster uses.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import sys
from datetime import UTC, datetime
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

import click
from rich.panel import Panel
from rich.table import Table

from memo.cli_common import console
from memo.config import AI_SUBDIR, Config


@click.command(name="mine-history")
@click.option("--path", "root_path", default=None,
              help="Transcripts root (default: ~/.claude/projects).")
@click.option("--since", "since_days", type=int, default=None,
              help="Only process transcripts modified in the last N days.")
@click.option("--limit", "file_limit", type=int, default=None,
              help="Cap on number of transcripts to process (newest first).")
@click.option("--dry-run", is_flag=True,
              help="Walk + extract, don't save. Useful for cost estimation.")
@click.option("--debug", is_flag=True, help="Print per-file/per-candidate info to stderr.")
@click.option("--json", "as_json", is_flag=True, help="Emit JSON summary instead of a panel.")
def mine_history(
    root_path: str | None, since_days: int | None, file_limit: int | None,
    dry_run: bool, debug: bool, as_json: bool,
) -> None:
    """Mine past Claude Code conversations for actionable insights.

    Walks `~/.claude/projects/<hash>/*.jsonl`, runs the same prefilter +
    helper-LLM extraction + embedding-based dedup as the live capture
    hook, and saves what's new. Resumable: per-file processed-line
    counts are tracked under `~/.local/share/memo/mine-history.json`.

    Tips:
        - First run on a long history is slow (helper LLM is the bottleneck).
          Use `--limit 10 --since 30` to start with the freshest sessions.
        - `--dry-run` reports candidate counts without writing.
    """
    from pathlib import Path as _Path

    from memo.transcript_miner import mine_transcripts

    root = _Path(root_path).expanduser() if root_path else None

    console_progress = None
    if not as_json:
        from rich.progress import (
            BarColumn,
            Progress,
            SpinnerColumn,
            TextColumn,
            TimeElapsedColumn,
        )
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        )
        progress.start()
        task = progress.add_task("mining transcripts", total=None)

        def cb(idx: int, total: int, p: _Path) -> None:
            progress.update(
                task, total=total, completed=idx, description=f"[{idx + 1}/{total}] {p.name}",
            )

        console_progress = (progress, task, cb)

    try:
        summary = mine_transcripts(
            root=root, since_days=since_days, file_limit=file_limit,
            dry_run=dry_run, debug=debug,
            progress_cb=console_progress[2] if console_progress else None,
        )
    finally:
        if console_progress:
            console_progress[0].stop()

    if as_json:
        click.echo(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    status = summary.get("status")
    if status == "no_files":
        console.print(f"[yellow]No transcripts found under {summary['root']}.[/yellow]")
        return

    saved = summary.get("saved", [])
    body = (
        f"[dim]root:[/dim] {summary['root']}\n"
        f"[dim]files:[/dim] {summary['files_processed']}/{summary['files_total']} processed"
        f" ([dim]{summary['files_skipped']} skipped — already mined[/dim])\n"
        f"[dim]candidates:[/dim] {summary['candidates']}\n"
        f"[bold green]saved:[/bold green] {len(saved)}"
        f"{' [yellow](dry-run)[/yellow]' if summary['dry_run'] else ''}\n"
        f"[dim]skipped duplicates:[/dim] {summary['skipped_dup']}"
    )
    console.print(Panel.fit(body, title="✓ mine-history", border_style="green"))


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
@click.option("--name", default=None, help="Vault label (default: dirname). Used as path prefix in store.")
@click.option("--force", is_flag=True, help="Re-embed even if body unchanged.")
@click.option("--dry-run", is_flag=True, help="Walk + report counts, don't embed/write.")
@click.option("--exclude", multiple=True, help="Glob to exclude (relative to vault). Repeat. Default: .obsidian/.git/.trash/.makemd/.smart-env/.space/Obsidian/AI/")
@click.option("--ocr/--no-ocr", default=True, help="Run OCR on ![[image]] embeds inside notes (Apple Vision). Default on.")
@click.option("--chunk/--no-chunk", default=True, help="Semantically chunk markdown/PDF bodies for better retrieval precision. Default on.")
@click.option("--chunk-chars", default=1500, show_default=True, type=int, help="Target chunk size in characters.")
@click.option("--chunk-overlap", default=250, show_default=True, type=int, help="Overlap between consecutive chunks.")
@click.option("--include-pdf/--no-include-pdf", default=True, help="Extract text from .pdf via pdftotext + chunk + embed.")
@click.option("--include-orphan-images/--no-include-orphan-images", default=True, help="OCR images not referenced by any note and ingest them as standalone memorias.")
@click.option("--prune/--no-prune", default=False, help="Delete stale vault-ingest chunks under this label: files moved/renamed/deleted (abs_path gone) and leftover chunks of notes edited down to fewer chunks. Default off (ingest is purely additive); the synapse vault-ingest agent passes --prune so the index self-heals.")
def ingest(
    vault_path: str, name: str | None, force: bool, dry_run: bool, exclude: tuple[str, ...],
    ocr: bool, chunk: bool, chunk_chars: int, chunk_overlap: int,
    include_pdf: bool, include_orphan_images: bool, prune: bool,
) -> None:
    """Bulk-ingest all .md from a vault into the memo index.

    Walks `<vault_path>/**/*.md`, embeds each, stores under path
    `<name>/<rel-path>`. Files with `id:` in frontmatter are skipped
    (those are curated memorias managed by `memo reindex`).

    The user's .md files are NOT modified — we synthesize ids from
    path hash and write only to `~/.local/share/memo/memvec.db`.

    Idempotent: re-running skips files whose body_hash matches the
    indexed value. Use --force to re-embed everything (e.g. after
    embedder model swap).

    Default exclusions skip Obsidian system dirs (.obsidian/, .trash/,
    etc.) and memo's own memory subtree (`<SYSTEM_DIR>/AI/`) so we
    don't double-index curated memorias. Note: sibling user content
    under `<SYSTEM_DIR>/` — `Contacts/`, `99-Forms/`, `99-Templates/`
    — IS indexed (e.g. `<SYSTEM_DIR>/Contacts/Grecia.md`).

    A `.memoignore` file in the vault root adds further exclusions (one
    pattern per line, `#` comments allowed) — the durable way to drop a
    folder like `04-Archive/` without editing the launchd ingest command.
    """
    import os as _os_min
    from pathlib import Path

    import frontmatter

    from memo.chunker import chunk_markdown
    from memo.embedder import MLXEmbedder, assert_valid_embedding
    from memo.ingest_helpers import (
        IMAGE_EXTENSIONS,
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
        ".obsidian", ".git", ".trash", ".makemd", ".smart-env", ".space",
        ".claude", ".devin", AI_SUBDIR,
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
        for pat in exclude_patterns:
            # A trailing `/**` means "this directory and everything under it".
            # The launchd ingest invocation passes patterns in this form
            # (`Obsidian/Whatsapp/**`); without this they silently no-op and the
            # subtree gets double-ingested by both the generic and dedicated
            # importers.
            if pat.endswith("/**"):
                base = pat[:-3]
                if s.startswith(base) or f"/{base}/" in padded:
                    return True
                continue
            # Literal prefix or `/segment/` anywhere in the rel path.
            if s.startswith(pat) or f"/{pat}/" in padded:
                return True
            # General globs (`*.tmp`, `a/*/b`) match against the full rel path.
            if ("*" in pat or "?" in pat or "[" in pat) and fnmatch(s, pat):
                return True
        return False

    md_files: list[Path] = []
    pdf_files: list[Path] = []
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
    md_files.sort()
    pdf_files.sort()

    pdf_supported = include_pdf and pdftotext_available()
    if include_pdf and not pdf_supported:
        console.print("[yellow]pdftotext not found on PATH — skipping PDFs[/yellow]")
        pdf_files = []

    console.print(
        f"[cyan]found[/cyan] {len(md_files)} .md, {len(pdf_files)} .pdf in {label} "
        f"(after exclusions)"
    )

    if dry_run:
        console.print("[dim](dry-run — exiting before embed/write)[/dim]")
        for p in md_files[:5]:
            console.print(f"  · {p.relative_to(vault)}")
        if len(md_files) > 5:
            console.print(f"  · …and {len(md_files) - 5} more")
        if pdf_files:
            console.print(f"  · PDFs: {len(pdf_files)}")
        return

    embedder = MLXEmbedder(model_path=cfg.embedder_model, expected_dims=cfg.embedder_dims)
    store = VecStore(cfg.db_path, dims=cfg.embedder_dims)

    skipped_id = skipped_empty = skipped_unchanged = added = updated = errors = 0
    skipped_pdf_empty = pdf_added = orphan_added = orphan_skipped = 0
    chunks_emitted = pruned = 0
    referenced_images: set[Path] = set()
    # Abs-paths of every file seen on disk this walk (md + pdf + orphan imgs),
    # added BEFORE any skip so existing-but-skipped files are kept. The --prune
    # sweep deletes label rows whose abs_path is NOT here (file gone from disk).
    seen_abs: set[str] = set()

    def _reconcile_file(store_path: str, valid_paths: set[str]) -> None:
        """Drop rows of one just-emitted file whose path is no longer valid —
        e.g. a multi-chunk note edited down to fewer chunks leaves stale
        `#chunk-N` rows, or a single↔multi flip leaves the other shape."""
        nonlocal pruned
        for row in store.file_rows(store_path):
            if row["path"] not in valid_paths and store.delete(row["id"]):
                pruned += 1

    min_chars = int(_os_min.environ.get("MEMO_INGEST_MIN_CHARS", "200"))
    strict_mode = _os_min.environ.get("MEMO_INGEST_STRICT") == "1"
    debug_mode = _os_min.environ.get("MEMO_INGEST_DEBUG") == "1"

    from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeRemainingColumn

    def _emit_record(
        *, store_path: str, title: str, tags: list[str], body: str,
        abs_path: Path, source: str, extra_meta: dict | None = None,
    ) -> str | None:
        """Embed `title + body` (chunked if --chunk and large) and upsert
        one row per chunk. Returns "added" / "updated" / None on error.

        Single-chunk path keeps the canonical store_path so dedup +
        idempotence keep working. Multi-chunk path suffixes
        `#chunk-N` to the store_path so each chunk is its own row.
        """
        nonlocal errors, chunks_emitted
        composed_full = f"{title}\n\n{body}"
        if chunk and len(composed_full) > chunk_chars:
            pieces = chunk_markdown(composed_full, target_chars=chunk_chars, overlap_chars=chunk_overlap)
        else:
            pieces = None  # single-vector path

        if pieces is None or len(pieces) <= 1:
            composed = composed_full[: cfg.max_content_chars]
            try:
                emb = embedder.embed([composed])[0]
                assert_valid_embedding(emb, cfg.embedder_dims, context=str(abs_path))
            except Exception as exc:
                errors += 1
                if strict_mode:
                    raise
                if debug_mode:
                    console.print(f"[red]reject:[/] {exc}")
                return None
            now = datetime.now(UTC).isoformat()
            id_, existing = _resolve_ingest_row(store, store_path)
            body_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]
            extra: dict[str, Any] = {"source": source, "vault": label, "abs_path": str(abs_path)}
            if extra_meta:
                extra.update(extra_meta)
            store.upsert(
                id_=id_, path=store_path, title=title[:200], type_="reference",
                tags=tags, created=existing["created"] if existing else now,
                updated=now, body_hash=body_hash, embedding=emb,
                extra=extra, body_text=body,
            )
            chunks_emitted += 1
            if prune:
                _reconcile_file(store_path, {store_path})
            return "updated" if existing else "added"

        # Multi-chunk path. Each chunk = own meta row; parent_path lets
        # the chat-ask dedup collapse chunks back to one source.
        any_added = any_updated = False
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
                errors += 1
                if strict_mode:
                    raise
                if debug_mode:
                    console.print(f"[red]reject:[/] {exc}")
                continue
            now = datetime.now(UTC).isoformat()
            chunk_title = f"{title} (§{seq+1}/{len(pieces)})"
            if heading:
                chunk_title = f"{title} — {heading}"
            extra = {
                "source": source, "vault": label, "abs_path": str(abs_path),
                "parent_path": store_path, "chunk_seq": seq,
                "chunk_count": len(pieces), "chunk_heading": heading,
            }
            if extra_meta:
                extra.update(extra_meta)
            store.upsert(
                id_=id_, path=chunk_path, title=chunk_title[:200], type_="reference",
                tags=[*tags, "chunk"], created=existing["created"] if existing else now,
                updated=now, body_hash=chunk_body_hash, embedding=emb,
                extra=extra, body_text=chunk_body,
            )
            chunks_emitted += 1
            if existing:
                any_updated = True
            else:
                any_added = True
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
            try:
                rel = path.relative_to(vault)
                store_path = f"{label}/{rel}" if label else str(rel)
                seen_abs.add(str(path))

                raw = path.read_text(encoding="utf-8", errors="replace")

                try:
                    fm = frontmatter.loads(raw)
                except Exception:
                    fm = frontmatter.Post(raw)

                # Skip curated memorias (have explicit id).
                if fm.metadata.get("id"):
                    skipped_id += 1
                    continue

                body = fm.content.strip()
                if not body:
                    skipped_empty += 1
                    continue

                if len(body) < min_chars and not _is_high_signal(body, fm.metadata.get("tags")):
                    skipped_empty += 1
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
                        body, path, vault, cfg.state_dir,
                    )
                    referenced_images.update(resolved)
                    body = enriched

                body_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]
                _, existing = _resolve_ingest_row(store, store_path)
                if existing and existing["body_hash"] == body_hash and not force:
                    skipped_unchanged += 1
                    continue

                outcome = _emit_record(
                    store_path=store_path, title=title, tags=tags, body=body,
                    abs_path=path, source="vault-ingest",
                )
                if outcome == "added":
                    added += 1
                elif outcome == "updated":
                    updated += 1
            except Exception as exc:
                errors += 1
                if debug_mode:
                    console.print(f"[red]err[/] {path}: {exc}")
            finally:
                progress.advance(task_id)

        if pdf_files:
            pdf_task = progress.add_task(f"PDF {label}", total=len(pdf_files))
            for pdf_path in pdf_files:
                try:
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
                        store_path=store_path, title=title, tags=tags, body=text,
                        abs_path=pdf_path, source="vault-ingest-pdf",
                    )
                    if outcome == "added":
                        pdf_added += 1
                except Exception as exc:
                    errors += 1
                    if debug_mode:
                        console.print(f"[red]err pdf[/] {pdf_path}: {exc}")
                finally:
                    progress.advance(pdf_task)

        if include_orphan_images and ocr:
            orphans = find_orphan_images(vault, referenced_images, excluded_dirs=tuple(exclude_patterns))
            # Filter image extensions we actually OCR (Apple Vision covers png/jpg/webp/heic).
            orphans = [o for o in orphans if o.suffix.lower() in IMAGE_EXTENSIONS]
            if orphans:
                orphan_task = progress.add_task(f"OCR orphan imgs {label}", total=len(orphans))
                from memo.ocr import extract_text_cached
                cache_dir = cfg.state_dir / "ocr_cache"
                for img_path in orphans:
                    try:
                        seen_abs.add(str(img_path))
                        ocr_text = (extract_text_cached(img_path, cache_dir=cache_dir) or "").strip()
                        if not ocr_text:
                            orphan_skipped += 1
                            continue
                        rel = img_path.relative_to(vault)
                        store_path = f"{label}/{rel}" if label else str(rel)
                        title = img_path.stem.replace("-", " ").replace("_", " ")
                        tags = [p for p in rel.parent.parts if p] + ["standalone-image"]
                        outcome = _emit_record(
                            store_path=store_path, title=title, tags=tags, body=ocr_text,
                            abs_path=img_path, source="vault-ingest-image",
                            extra_meta={"image_ext": img_path.suffix.lower()},
                        )
                        if outcome == "added":
                            orphan_added += 1
                    except Exception as exc:
                        errors += 1
                        if debug_mode:
                            console.print(f"[red]err img[/] {img_path}: {exc}")
                    finally:
                        progress.advance(orphan_task)

    # Prune: drop vault-ingest rows under this label whose source file is
    # gone from disk (moved/renamed/deleted). Per-file chunk reconciliation
    # already ran in _emit_record for re-emitted files; this catches whole
    # files that disappeared. source-filtered, so curated memorias are safe.
    if prune:
        for row in store.vault_ingest_rows(label):
            abs_path = row.get("abs_path")
            if abs_path and abs_path not in seen_abs and store.delete(row["id"]):
                pruned += 1

    # Bump on-disk schema version so the legacy-paths probe in
    # `Memory._maybe_warn_legacy_paths` doesn't fire for ingest-only
    # vaults (vault-ingest rows live outside `cfg.data_dir` and the
    # probe can't resolve them; setting user_version=1 marks "this DB
    # is post-init, the legacy fallback is no longer relevant").
    if added or updated or pdf_added or orphan_added:
        with contextlib.suppress(Exception):
            store.set_user_version(1)

    console.print(
        f"\n[green]done[/] "
        f"added={added} updated={updated} "
        f"skipped_unchanged={skipped_unchanged} "
        f"skipped_id={skipped_id} skipped_empty={skipped_empty} "
        f"pdf_added={pdf_added} pdf_empty={skipped_pdf_empty} "
        f"orphan_added={orphan_added} orphan_skipped={orphan_skipped} "
        f"chunks_emitted={chunks_emitted} pruned={pruned} "
        f"errors={errors}"
    )


_HIGH_SIGNAL_TAGS = frozenset({
    # Notes pinned to lookup-style facts. Lowercase compare; surface
    # forms like "Link" / "LINKS" / "Pago" all match. Spanish + English
    # variants because the vault mixes both.
    "link", "links", "url", "urls",
    "dato", "datos", "data",
    "ref", "refs", "referencia", "referencias", "reference",
    "comando", "comandos", "command", "commands", "cmd", "snippet",
    "pago", "pagos", "payment",
    "credencial", "credenciales", "credential", "credentials",
    "endpoint", "endpoints", "api",
    "telefono", "teléfono", "phone", "tel",
    "cbu", "alias", "iban",
})

# Match http(s):// URLs — anchored end on whitespace, ), >, ], or "
# (common markdown wrappers). Permissive enough to catch trailing
# punctuation cases without dragging adjacent text in.
_URL_RE = re.compile(r"https?://[^\s)>\]\"]+")


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


@click.command(name="capture-stop")
def capture_stop() -> None:
    """Stop hook — passive auto-extract of insights from the last turn.

    Reads the Stop hook payload from stdin (Claude Code format), pulls
    the last (user, assistant) exchange from the transcript, asks the
    helper LLM (Qwen2.5-3B) to extract any actionable insights, dedups
    against the existing corpus, and saves survivors via Memory.save().

    Hook input (stdin, JSON):
      {"transcript_path": "/path/to/...jsonl", ...}

    Hook output (stdout):
      `{}`  — always. Capture is silent; the user discovers new
      memorias via `memo list` or the next ambient recall.

    Env vars:
      MEMO_CAPTURE_DISABLE  — set to "1" to make this a no-op.
      MEMO_CAPTURE_DEBUG    — set to "1" to print extraction progress
                              to stderr (helpful while tuning the
                              extraction prompt or trigger keywords).

    Failure modes are absorbed. The hook never blocks the user — at
    worst you get no auto-save for that turn.
    """
    import json as _json
    import sys as _sys
    from pathlib import Path

    if os.environ.get("MEMO_CAPTURE_DISABLE") == "1":
        print("{}")
        _sys.exit(0)

    debug = os.environ.get("MEMO_CAPTURE_DEBUG") == "1"

    try:
        raw = _sys.stdin.read()
        payload = _json.loads(raw) if raw.strip() else {}
    except _json.JSONDecodeError:
        print("{}")
        _sys.exit(0)

    transcript_path = payload.get("transcript_path")
    if not transcript_path:
        print("{}")
        _sys.exit(0)

    try:
        from memo.capture import run_capture
        run_capture(Path(transcript_path), debug=debug)
    except Exception as exc:
        if debug:
            print(f"# memo capture-stop failed: {exc}", file=_sys.stderr)

    # Grounding (P0): score how much the answer used this turn's recalled
    # memorias → grounding.log (the outcome-based utility signal). Best-effort,
    # budget-guarded inside score_turn, never fails the turn.
    try:
        from memo import grounding
        from memo.config import Config
        summary = grounding.score_turn(Config.from_env().state_dir, payload)
        if debug and summary:
            print(f"# memo grounding: {summary}", file=_sys.stderr)
    except Exception as exc:
        if debug:
            print(f"# memo grounding failed: {exc}", file=_sys.stderr)

    print("{}")
    _sys.exit(0)


# ── Session reflection (v0.5.0) ─────────────────────────────────────────────
#
# `memo reflect` — read a full session transcript, extract durable insights
# (decisions, facts, bugs, follow-ups), and save them as memorias + a session
# arc nota. Auto-idempotent via `reflected_at` stamp in the session snapshot.


_REFLECT_TRANSCRIPT_WORD_BUDGET = 8000


def _read_full_transcript(transcript_path: Path) -> list[tuple[str, str]]:
    """Return all (role, text) pairs from a JSONL transcript. role ∈ {"user", "assistant"}.
    Skips tool-use blocks, system lines, and empty turns."""
    if not transcript_path.is_file():
        return []
    try:
        raw = transcript_path.read_text(encoding="utf-8")
    except OSError:
        return []

    exchanges: list[tuple[str, str]] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        role = obj.get("type") or obj.get("role")
        if role not in ("user", "assistant"):
            continue
        msg = obj.get("message", obj)
        content = msg.get("content") if isinstance(msg, dict) else None
        if content is None:
            continue
        if isinstance(content, str):
            text = content.strip()
        elif isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text":
                    t = (block.get("text") or "").strip()
                    if t:
                        parts.append(t)
            text = "\n\n".join(parts).strip()
        else:
            text = ""
        if text:
            exchanges.append((role, text))
    return exchanges


def _build_reflect_prompt(
    exchanges: list[tuple[str, str]],
    *,
    cwd: str | None = None,
    branch: str | None = None,
    running_summary: str | None = None,
) -> str:
    """Build the transcript block for the reflect LLM call.
    Word-budgeted to ~8k words: keep last N exchanges that fit.
    """
    # Build context header.
    ctx_parts: list[str] = []
    if cwd:
        ctx_parts.append(f"cwd: {cwd}")
    if branch:
        ctx_parts.append(f"branch: {branch}")
    if running_summary:
        ctx_parts.append(f"session summary: {running_summary}")
    header = " | ".join(ctx_parts)

    # Word-budget the transcript (most recent exchanges preferred).
    budget = _REFLECT_TRANSCRIPT_WORD_BUDGET
    blocks: list[str] = []
    for role, text in reversed(exchanges):
        label = "User" if role == "user" else "Assistant"
        # Truncate per-turn to avoid a single monster turn eating the budget.
        snippet = text[:2000]
        block = f"[{label}] {snippet}"
        words = len(block.split())
        if words > budget:
            break
        blocks.append(block)
        budget -= words
    blocks.reverse()

    transcript = "\n\n".join(blocks)
    if header:
        return f"Context: {header}\n\nTranscript:\n{transcript}"
    return f"Transcript:\n{transcript}"


def _reflect_session(
    session_id: str,
    mem: Any,
    cfg: Any,
    *,
    dry_run: bool = False,
    debug: bool = False,
) -> dict[str, Any]:
    """Core reflect logic. Returns a result dict.

    Loads the session snapshot, reads the full transcript, calls the LLM,
    saves memorias, stamps `reflected_at`. All heavy imports are deferred.
    """
    from memo.session import get_session, mark_reflected

    snap = get_session(cfg.state_dir, session_id)
    if snap is None:
        return {"status": "not_found", "session_id": session_id}

    transcript_path_str = snap.get("transcript_path")
    if not transcript_path_str:
        return {"status": "no_transcript", "session_id": session_id}

    transcript_path = Path(transcript_path_str).expanduser()
    exchanges = _read_full_transcript(transcript_path)

    user_turns = [e for e in exchanges if e[0] == "user"]
    if len(user_turns) < 3:
        return {"status": "too_short", "session_id": session_id, "user_turns": len(user_turns)}

    prompt = _build_reflect_prompt(
        exchanges,
        cwd=snap.get("cwd"),
        branch=snap.get("branch"),
        running_summary=snap.get("running_summary"),
    )

    # LLM call — use the configured llm_model (7B default).
    try:
        from memo.memory.record import _REFLECT_SYSTEM_PROMPT, strip_llm_output
        result = mem._chat.chat(
            cfg.llm_model,
            [
                {"role": "system", "content": _REFLECT_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            options={"temperature": 0.0, "num_predict": 1024},
        )
        raw_json = (result.get("message") or {}).get("content") or ""
    except Exception as exc:
        if debug:
            print(f"# memo reflect: LLM failed: {exc}", file=sys.stderr)
        return {"status": "llm_error", "session_id": session_id, "error": str(exc)}

    # Strip <think> traces + markdown fences (shared helper).
    raw_json = strip_llm_output(raw_json)
    try:
        parsed = json.loads(raw_json)
    except json.JSONDecodeError:
        if debug:
            print(f"# memo reflect: JSON parse failed: {raw_json[:200]}", file=sys.stderr)
        parsed = {}

    session_title = (parsed.get("session_title") or "").strip()
    arc_summary = (parsed.get("summary") or "").strip()

    # Gather all items across categories.
    from memo.capture import is_near_duplicate

    saved_ids: list[str] = []
    skipped_dup = 0

    category_type_map = {
        "decisions": "decision",
        "facts": "fact",
        "bugs": "bug",
        "followups": "note",
    }

    for cat_key, mem_type in category_type_map.items():
        items = parsed.get(cat_key) or []
        for item in items:
            if not isinstance(item, dict):
                continue
            title = (item.get("title") or "").strip()[:80]
            body = (item.get("body") or "").strip()[:300]
            tags = [str(t).lower().strip() for t in (item.get("tags") or []) if t]
            if not title or not body:
                continue

            cand = {"title": title, "body": body}
            if not dry_run and is_near_duplicate(mem, cand):
                skipped_dup += 1
                if debug:
                    print(f"# memo reflect: skip dup '{title}'", file=sys.stderr)
                continue

            if not dry_run:
                try:
                    rec = mem.save(content=body, title=title, type_=mem_type, tags=tags)
                    saved_ids.append(rec.id)
                    if debug:
                        print(f"# memo reflect: saved [{rec.id[:8]}] {rec.title}", file=sys.stderr)
                except Exception as exc:
                    if debug:
                        print(f"# memo reflect: save failed: {exc}", file=sys.stderr)
            else:
                saved_ids.append(f"(dry-run) {title}")

    # Arc nota — a single nota linking the session narrative.
    arc_id: str | None = None
    if arc_summary and not dry_run:
        project = snap.get("project") or "unknown"
        branch_str = snap.get("branch") or ""
        id_refs = " ".join(f"[{i[:8]}]" for i in saved_ids if not i.startswith("("))
        arc_body = f"{arc_summary}"
        if id_refs:
            arc_body += f"\n\nInsights: {id_refs}"
        arc_title = session_title or f"{project} session"
        try:
            arc_tags = ["session-arc", f"project:{project}"]
            if branch_str:
                arc_tags.append(f"branch:{branch_str}")
            arc_rec = mem.save(content=arc_body, title=arc_title, type_="note", tags=arc_tags)
            arc_id = arc_rec.id
            if debug:
                print(f"# memo reflect: arc nota [{arc_id[:8]}] {arc_title}", file=sys.stderr)
        except Exception as exc:
            if debug:
                print(f"# memo reflect: arc save failed: {exc}", file=sys.stderr)

    if not dry_run:
        mark_reflected(cfg.state_dir, session_id)

    return {
        "status": "ok",
        "session_id": session_id,
        "session_title": session_title,
        "saved": saved_ids,
        "skipped_dup": skipped_dup,
        "arc_id": arc_id,
        "dry_run": dry_run,
    }


@click.command(name="reflect")
@click.argument("session_id", required=False)
@click.option("--last", is_flag=True, default=False,
              help="Reflect on the most recent completed session (default if no SESSION_ID).")
@click.option("--if-due", is_flag=True, default=False,
              help="Skip if the session was already reflected (idempotent).")
@click.option("--quiet", is_flag=True, default=False,
              help="Output JSON only (for hook use).")
@click.option("--dry-run", is_flag=True, default=False,
              help="Show what would be saved without saving.")
@click.option("--debug", is_flag=True, default=False,
              help="Print extraction progress to stderr.")
def reflect(
    session_id: str | None, last: bool, if_due: bool,
    quiet: bool, dry_run: bool, debug: bool,
) -> None:
    """Synthesize a session transcript into durable memorias.

    Reads the full session transcript (not just the last 3 turns),
    extracts decisions/facts/bugs/follow-ups, deduplicates against the
    existing corpus, and saves survivors plus a session arc nota.

    Idempotent: a `reflected_at` stamp prevents re-processing the same
    session. Pass `--if-due` to skip cleanly when already reflected.

    Examples:

      memo reflect --last            # reflect on the most recent session
      memo reflect --last --if-due   # no-op if already reflected (hook use)
      memo reflect <id>              # reflect on a specific session
      memo reflect --last --dry-run  # preview without saving
    """
    from memo.config import Config
    from memo.session import get_session, list_sessions

    if os.environ.get("MEMO_CAPTURE_DISABLE") == "1":
        if quiet:
            click.echo(json.dumps({"status": "disabled"}))
        return

    cfg = Config.from_env()

    # Resolve which session to reflect on.
    target_id: str | None = session_id
    if not target_id:
        sessions = list_sessions(cfg.state_dir, limit=2)
        if not sessions:
            result = {"status": "no_sessions"}
            if quiet:
                click.echo(json.dumps(result))
            else:
                console.print("[yellow]No sessions found.[/yellow]")
            return
        # `--last` or no arg: use the most recent session.
        # If session is still "active" (no reflected_at, recent), use it anyway.
        target_id = sessions[0].get("session_id") or ""

    if not target_id:
        result = {"status": "no_session_id"}
        click.echo(json.dumps(result) if quiet else "")
        return

    # Idempotence guard.
    if if_due:
        snap = get_session(cfg.state_dir, target_id)
        if snap and snap.get("reflected_at"):
            result = {"status": "already_reflected", "session_id": target_id,
                      "reflected_at": snap["reflected_at"]}
            if quiet:
                click.echo(json.dumps(result))
            else:
                console.print(f"[dim]already reflected: {target_id[:8]}[/dim]")
            return

    # Load Memory with LLM warmed.
    from memo.memory import Memory

    mem = Memory(cfg)
    if mem._chat is None:  # type: ignore[attr-defined]
        from memo.llm import MLXChat
        mem._chat = MLXChat()  # type: ignore[attr-defined]

    result = _reflect_session(target_id, mem, cfg, dry_run=dry_run, debug=debug)

    if quiet:
        click.echo(json.dumps(result, ensure_ascii=False))
        return

    status = result.get("status")
    if status == "not_found":
        console.print(f"[red]session not found:[/red] {target_id}")
        sys.exit(1)
    if status == "no_transcript":
        console.print(f"[yellow]no transcript for session:[/yellow] {target_id[:8]}")
        return
    if status == "too_short":
        console.print(
            f"[dim]session too short ({result.get('user_turns')} user turns) — skipping[/dim]",
        )
        return
    if status == "llm_error":
        console.print(f"[red]LLM error:[/red] {result.get('error')}")
        sys.exit(1)
    if status == "already_reflected":
        console.print(f"[dim]already reflected: {target_id[:8]}[/dim]")
        return

    saved = list(result.get("saved") or [])
    skipped = result.get("skipped_dup") or 0
    arc_id = result.get("arc_id")
    dry_label = " [yellow](dry-run)[/yellow]" if dry_run else ""
    title = result.get("session_title") or target_id[:8]

    body = (
        f"[dim]session:[/dim] {target_id[:8]}\n"
        f"[dim]title:[/dim]   {title}\n"
        f"[bold green]saved:[/bold green]   {len(saved)}{dry_label}\n"
        f"[dim]dup skip:[/dim] {skipped}\n"
        f"[dim]arc:[/dim]     {arc_id[:8] if arc_id else '—'}"
    )
    console.print(Panel.fit(body, title="✓ reflect", border_style="green"))


# ── Session checkpoints (v0.4.0) ───────────────────────────────────────────
#
# `memo session ...` — short-lived "what was I working on" snapshots, written
# on every Claude Code Stop hook. Survive a closed/crashed session so the
# next SessionStart can show a picker of recent work. Storage is sidecar
# JSON in `state_dir/sessions/`, NOT memorias (different lifecycle, different
# query pattern — looked up by recency, never by semantic similarity).




@click.command(name="resume")
@click.argument("session_id", required=False)
@click.option("--limit", default=10, type=int, show_default=True,
              help="Max sessions to show (only used when SESSION_ID is omitted).")
@click.option("--project", default=None, help="Filter to one project basename.")
@click.option("--cwd", "cwd_filter", default=None,
              help="Filter to sessions for this exact cwd (resolved). "
                   "Used by the shell wrapper to ask 'what was open here?' "
                   "without manual path comparison.")
@click.option("--json", "as_json", is_flag=True)
def resume(
    session_id: str | None, limit: int,
    project: str | None, cwd_filter: str | None, as_json: bool,
) -> None:
    """Recent sessions to retomar — picker for the SessionStart flow.

    With no argument, prints a table of the most recent sessions
    (cwd / branch / summary / id). Pass SESSION_ID (full or unique
    prefix ≥4 chars) to inspect one session in detail.

    Storage is sidecar JSON under `~/.local/share/memo/sessions/`,
    auto-written by the Stop hook (`memo session checkpoint`) and
    LRU-capped at 50.
    """
    from memo.session import format_relative, get_session, list_sessions

    cfg = Config.from_env()

    # Detail view — one session.
    if session_id:
        snap = get_session(cfg.state_dir, session_id)
        if snap is None:
            console.print(f"[red]not found:[/red] {session_id}")
            sys.exit(1)
        if as_json:
            click.echo(json.dumps(snap, ensure_ascii=False, indent=2))
            return
        mods = snap.get("modified_files") or []
        mods_line = ", ".join(mods[:5])
        if len(mods) > 5:
            mods_line += f", …(+{len(mods) - 5})"
        sid = snap.get("session_id") or ""
        console.print(Panel.fit(
            f"[bold]{snap.get('summary') or snap.get('last_user_msg') or 'session'}[/bold]\n"
            f"[dim]session_id:[/dim] {sid}\n"
            f"[dim]project:[/dim]    {snap.get('project') or '—'}\n"
            f"[dim]cwd:[/dim]        {snap.get('cwd') or '—'}\n"
            f"[dim]branch:[/dim]     {snap.get('branch') or '—'}\n"
            f"[dim]head:[/dim]       {snap.get('head_commit') or '—'}\n"
            f"[dim]modified:[/dim]   {mods_line or '—'}\n"
            f"[dim]transcript:[/dim] {snap.get('transcript_path') or '—'}\n"
            f"[dim]created:[/dim]    {snap.get('created')}  ({format_relative(snap.get('created'))})\n"
            f"[dim]updated:[/dim]    {snap.get('updated')}  ({format_relative(snap.get('updated'))})\n"
            f"[dim]turns:[/dim]      {snap.get('turn_count')}\n\n"
            f"{snap.get('last_user_msg') or ''}",
            title="session", border_style="cyan",
        ))
        if sid:
            console.print(
                f"\n[bold green]Para retomar:[/bold green]  "
                f"[cyan]claude --resume {sid}[/cyan]\n"
                f"[dim](copy-paste; corré el comando desde "
                f"`{snap.get('cwd') or '?'}`)[/dim]",
            )
        return

    # List view — picker.
    rows = list_sessions(
        cfg.state_dir, limit=limit, project=project, cwd=cwd_filter,
    )
    if as_json:
        click.echo(json.dumps(rows, ensure_ascii=False, indent=2))
        return
    if not rows:
        console.print("[dim]no sessions yet — run a checkpoint first[/dim]")
        return

    # When the caller passed an explicit --cwd, the list is already
    # filtered to that cwd — printing a "Última en este proyecto"
    # banner on top of a homogeneous list would be redundant.
    if cwd_filter:
        same_cwd = []
    else:
        # Bias: if there's a session for the current cwd, surface it on top
        # with the exact resume command. The whole point of the picker is
        # crash recovery — if you crashed and reopened terminal in the same
        # project, the very first thing you want to see is "click here to
        # resume", not a generic chronological list.
        import os as _os
        from pathlib import Path as _Path
        cur_cwd = str(_Path(_os.getcwd()).resolve())
        same_cwd = [r for r in rows if (r.get("cwd") or "") == cur_cwd]
    if same_cwd:
        top = same_cwd[0]
        sid = top.get("session_id") or ""
        console.print(
            f"[bold green]Última en este proyecto[/bold green]  "
            f"[dim]({format_relative(top.get('updated'))})[/dim]: "
            f"{(top.get('summary') or top.get('last_user_msg') or '—')[:80]}",
        )
        console.print(
            f"[bold green]Para retomar:[/bold green]  "
            f"[cyan]claude --resume {sid}[/cyan]\n",
        )

    tbl = Table(show_lines=False, expand=True)
    tbl.add_column("when", width=10)
    tbl.add_column("project", width=14, overflow="fold")
    tbl.add_column("branch", width=14, overflow="fold")
    tbl.add_column("turns", justify="right", width=5)
    tbl.add_column("summary", overflow="fold")
    tbl.add_column("session_id", overflow="fold")
    for r in rows:
        tbl.add_row(
            format_relative(r.get("updated")),
            r.get("project") or "—",
            r.get("branch") or "—",
            str(r.get("turn_count") or 0),
            (r.get("summary") or r.get("last_user_msg") or "—")[:80],
            r.get("session_id") or "—",
        )
    console.print(tbl)
    console.print(
        "[dim]Detalle: `memo resume <id|prefix>`  ·  "
        "Retomar: `claude --resume <session_id>` (copy desde la tabla).[/dim]",
    )

