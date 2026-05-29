"""Core memory verbs for the memo CLI — CRUD + retrieval.

Extracted from cli.py (3a follow-up: top-level command grouping). Each
command is a standalone @click.command registered onto the root group in
cli.py via cli.add_command(...). Commands: save, search, ask, embed,
chat-ask, rerank, list, get, update, reindex, delete, history, ocr-image,
provenance, extract-entities, entities, entity, lint, restore.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import click
from rich.panel import Panel
from rich.table import Table

from memo.cli_common import console
from memo.config import Config


def _resolved(thunk):
    """Run `thunk()` translating `AmbiguousIdError` into a friendly print
    + exit code 2. Used by every CLI verb that takes an id-or-prefix
    argument (`get`, `update`, `delete`).
    """
    from memo.memory import AmbiguousIdError

    try:
        return thunk()
    except AmbiguousIdError as exc:
        console.print(f"[red]ambiguous id prefix[/red] {exc.prefix!r} matches:")
        for m in exc.matches[:8]:
            console.print(f"  · {m}")
        if len(exc.matches) > 8:
            console.print(f"  · …and {len(exc.matches) - 8} more")
        sys.exit(2)


@click.command()
@click.argument("content")
@click.option("--title", default=None, help="Short title (default: first line of content)")
@click.option(
    "--type", "type_",
    type=click.Choice(
        ["decision", "fact", "bug", "feedback", "preference", "note", "manual"],
    ),
    default="note", show_default=True,
)
@click.option("--tag", "-t", "tags", multiple=True, help="Repeatable. Lower-cased + de-duplicated.")
@click.option("--auto-derive", is_flag=True,
              help="When title/type/tags missing, ask Qwen2.5-3B helper to derive them. "
                   "Adds ~1-2s latency on first call.")
@click.option("--no-project-tag", "no_project_tag", is_flag=True,
              help="Skip the auto `project:<repo>` tag derived from the current git toplevel.")
@click.option("--defer-embed", is_flag=True,
              help="Save markdown + BM25 index only; run `memo reindex` later for semantic search.")
@click.option("--meta", "meta_pairs", multiple=True, metavar="KEY=VALUE",
              help="Repeatable. Adds an entry to the `extra` metadata bag persisted "
                   "to frontmatter + meta.extra_json. Synapse uses this to attach "
                   "provenance (`--meta synapse_trace_id=...`, `--meta synapse_agent_id=...`).")
@click.option("--json", "as_json", is_flag=True, help="Emit JSON instead of a panel.")
def save(content: str, title: str | None, type_: str, tags: tuple[str, ...],
         auto_derive: bool, no_project_tag: bool, defer_embed: bool,
         meta_pairs: tuple[str, ...], as_json: bool) -> None:
    """Persist CONTENT to the vault + index. Pass `-` to read CONTENT from stdin."""
    from memo.memory import Memory

    if content == "-":
        content = sys.stdin.read()
    extra: dict[str, Any] | None = None
    if meta_pairs:
        extra = {}
        for pair in meta_pairs:
            if "=" not in pair:
                raise click.BadParameter(
                    f"--meta expects KEY=VALUE, got {pair!r}", param_hint="--meta",
                )
            key, _, value = pair.partition("=")
            key = key.strip()
            if not key:
                raise click.BadParameter(
                    f"--meta key cannot be empty: {pair!r}", param_hint="--meta",
                )
            extra[key] = value
    mem = Memory(Config.from_env())
    rec = mem.save(content=content, title=title, type_=type_,
                   tags=list(tags), auto_derive=auto_derive,
                   auto_project=not no_project_tag,
                   defer_embed=defer_embed, extra=extra)
    if as_json:
        click.echo(json.dumps(rec.to_dict(), ensure_ascii=False, indent=2))
        return
    console.print(Panel.fit(
        f"[bold]{rec.title}[/bold]\n"
        f"[dim]id:[/dim] {rec.id}\n"
        f"[dim]path:[/dim] {rec.path}\n"
        f"[dim]type:[/dim] {rec.type}  [dim]tags:[/dim] {', '.join(rec.tags) or '—'}",
        title="✓ saved", border_style="green",
    ))


@click.command()
@click.argument("query")
@click.option("--limit", default=10, type=int, show_default=True)
@click.option("--type", "type_", default=None, help="Filter by record type.")
@click.option("--mode", default="hybrid",
              type=click.Choice(["hybrid", "vec", "bm25"]), show_default=True,
              help="hybrid = RRF fusion of vec + bm25 (default). vec = semantic only. bm25 = keyword only.")
@click.option("--json", "as_json", is_flag=True)
def search(query: str, limit: int, type_: str | None, mode: str, as_json: bool) -> None:
    """Top-k search — hybrid (semantic + keyword) by default."""
    from memo.memory import Memory

    mem = Memory(Config.from_env())
    hits = mem.search(query, limit=limit, type_=type_, mode=mode)
    if as_json:
        click.echo(json.dumps([h.to_dict() for h in hits], ensure_ascii=False, indent=2))
        return
    if not hits:
        console.print("[dim]no results[/dim]")
        return
    tbl = Table(show_lines=False, expand=True)
    tbl.add_column("score", justify="right", width=6)
    tbl.add_column("type", width=10)
    tbl.add_column("title", overflow="fold")
    tbl.add_column("tags", overflow="fold")
    for h in hits:
        tbl.add_row(
            f"{h.score:.3f}" if h.score is not None else "—",
            h.type,
            h.title,
            ", ".join(h.tags) or "—",
        )
    console.print(tbl)


@click.command()
@click.argument("question")
@click.option("--k", default=5, type=int, show_default=True,
              help="Top-K memorias to feed the LLM as context.")
@click.option("--type", "type_", default=None, help="Restrict the retrieval to one record type.")
@click.option("--json", "as_json", is_flag=True)
def ask(question: str, k: int, type_: str | None, as_json: bool) -> None:
    """RAG over the memory archive — synthesises a prose answer with
    inline `[id]` citations using MLXChat 7B over the top-K hybrid hits.
    """
    from memo.memory import Memory

    mem = Memory(Config.from_env())
    out = mem.ask(question, k=k, type_=type_)
    if as_json:
        click.echo(json.dumps(out, ensure_ascii=False, indent=2))
        return
    console.print(Panel.fit(
        out["answer"] or "[dim](sin respuesta)[/dim]",
        title=f"❓ {question[:60]}", border_style="cyan",
    ))
    if out["sources"]:
        console.print("[dim]fuentes:[/dim]")
        for s in out["sources"]:
            console.print(
                f"  [dim][{s['id_short']}][/dim] {s['title'][:60]}  "
                f"[dim](score {s['score']:.3f})[/dim]"
            )


@click.command(name="embed")
@click.argument("text", required=False)
@click.option(
    "--batch-json", type=click.File("r"), default=None,
    help="Read JSON list of texts from this path (or '-' for stdin). "
         "Each text is embedded with the SYMMETRIC (document) prefix. "
         "Mutually exclusive with positional TEXT.",
)
def embed_cmd(text: str | None, batch_json) -> None:
    """Compute embedding vector(s) using memo's MLX embedder.

    Single (asymmetric query prefix):
        memo embed "hablame de Grecia"

    Batch (symmetric document prefix, single MLX forward pass):
        echo '["alpha","beta","gamma"]' | memo embed --batch-json -

    Output: one JSON object per invocation (no indent), written to
    stdout. Shape:
      single: {"vector": [...], "dim": int, "model": "..."}
      batch:  {"vectors": [[...], ...], "dim": int, "model": "..."}

    Synapse consumes this as its unified embed RPC (replaces a separate
    Ollama embedder so query/document vectors share memo's space).
    """
    from memo.memory import Memory

    if batch_json is not None and text:
        raise click.UsageError("--batch-json and TEXT are mutually exclusive")
    if batch_json is None and not text:
        raise click.UsageError("provide TEXT or --batch-json")

    mem = Memory(Config.from_env())
    if batch_json is not None:
        try:
            texts = json.load(batch_json)
        except json.JSONDecodeError as exc:
            raise click.UsageError(f"--batch-json: invalid JSON: {exc}") from exc
        if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
            raise click.UsageError("--batch-json: expected JSON list of strings")
        vecs = mem.embedder.embed(texts)
        dim = len(vecs[0]) if vecs else 0
        out = {"vectors": vecs, "dim": dim, "model": mem.cfg.embedder_model}
    else:
        assert text is not None
        vec = mem.embedder.embed_query(text)
        out = {"vector": vec, "dim": len(vec), "model": mem.cfg.embedder_model}
    sys.stdout.write(json.dumps(out, ensure_ascii=False) + "\n")
    sys.stdout.flush()


@click.command(name="chat-ask")
@click.argument("question")
@click.option("--k", default=7, type=int, show_default=True,
              help="Top-K memorias to feed the LLM as context.")
@click.option("--type", "type_", default=None, help="Restrict the retrieval to one record type.")
@click.option("--history-json", type=click.File("r"), default=None,
              help="Conversation history JSON: list of {role,text}. '-' reads stdin.")
@click.option("--context-json", type=click.File("r"), default=None,
              help="Caller-supplied federation context (e.g. Synapse packet) for richer synthesis.")
@click.option("--json", "as_json", is_flag=True)
@click.option(
    "--stream", "as_stream", is_flag=True,
    help=(
        "Emit one NDJSON event per line (context/token/done) flushed "
        "immediately. Forces JSON output and disables the panel."
    ),
)
def chat_ask(
    question: str,
    k: int,
    type_: str | None,
    history_json,
    context_json,
    as_json: bool,
    as_stream: bool,
) -> None:
    """Chat-shaped RAG over memo."""
    from memo.memory import Memory

    history: list[dict] = []
    if history_json is not None:
        try:
            raw = json.load(history_json)
            if isinstance(raw, list):
                history = [h for h in raw if isinstance(h, dict)]
        except Exception:
            history = []

    context: dict = {}
    if context_json is not None:
        try:
            ctx = json.load(context_json)
            if isinstance(ctx, dict):
                context = ctx
        except Exception:
            context = {}

    mem = Memory(Config.from_env())

    if as_stream:
        import sys
        for event in mem.chat_ask_stream(
            question,
            k=k,
            type_=type_,
            history=history,
            context=context,
        ):
            sys.stdout.write(json.dumps(event, ensure_ascii=False) + "\n")
            sys.stdout.flush()
        return

    envelope = mem.chat_ask(
        question,
        k=k,
        type_=type_,
        history=history,
        context=context,
    )
    if as_json:
        click.echo(json.dumps(envelope, ensure_ascii=False, indent=2))
        return
    console.print(Panel.fit(
        envelope["answer"] or "[dim](sin respuesta)[/dim]",
        title=f"❓ {question[:60]}", border_style="magenta",
    ))
    if envelope["sources"]:
        console.print("[dim]fuentes:[/dim]")
        for s in envelope["sources"]:
            console.print(
                f"  [dim][{s.get('id_short','?')}][/dim] {(s.get('title','') or '')[:60]}  "
                f"[dim](score {s.get('score', 0):.3f})[/dim]"
            )


@click.command(name="rerank")
@click.option("--query", "query", required=True, help="The search query.")
@click.option(
    "--hits-file", type=click.Path(exists=True, dir_okay=False),
    default=None, help="JSON file with hits array. If omitted, reads stdin.",
)
@click.option(
    "--top-n", type=int, default=None,
    help="Truncate output to top-N after reranking.",
)
@click.option(
    "--body-chars", type=int, default=1200, show_default=True,
    help="Per-hit body truncation before scoring.",
)
@click.option("--trace-id", default="", help="Trace ID for provenance (unused locally).")
def rerank_cmd(
    query: str,
    hits_file: str | None,
    top_n: int | None,
    body_chars: int,
    trace_id: str,
) -> None:
    """Rerank externally supplied hits via MLXReranker.

    Reads a JSON array of hits ({title, snippet|body, ...}) from
    ``--hits-file`` or stdin, scores each (query, hit) pair with the
    configured ``reranker_model``, and writes the reordered array to
    stdout with a new ``rerank_score`` field per hit. Original fields
    are preserved verbatim.

    Designed for Synapse to delegate rerank to memo's already-cached
    Qwen3-Reranker without loading the model in a second process.
    """
    import sys

    del trace_id  # accepted for compat; not used locally

    if hits_file:
        with open(hits_file, encoding="utf-8") as f:
            hits = json.load(f)
    else:
        hits = json.load(sys.stdin)

    if not isinstance(hits, list) or not hits:
        click.echo(json.dumps([], ensure_ascii=False))
        return

    cfg = Config.from_env()
    if not cfg.reranker_enabled:
        # Pass-through when reranker is disabled in this memo install,
        # so callers can rely on the subcommand always returning a list.
        click.echo(json.dumps(hits, ensure_ascii=False))
        return

    from memo.reranker import MLXReranker

    r = MLXReranker(
        model_path=cfg.reranker_model,
        revision=cfg.reranker_revision,
    )

    scored: list[tuple[float, dict]] = []
    for h in hits:
        if not isinstance(h, dict):
            continue
        title = str(h.get("title") or "")
        body_src = str(h.get("snippet") or h.get("body") or "")[
            : max(0, body_chars)
        ]
        doc = f"{title}\n\n{body_src}" if body_src else title
        try:
            p = float(r.score(query, doc))
        except Exception:
            p = 0.0
        scored.append((p, h))

    scored.sort(key=lambda t: t[0], reverse=True)
    out: list[dict] = []
    for p, h in scored:
        new = dict(h)
        new["rerank_score"] = p
        out.append(new)

    if top_n is not None and top_n > 0:
        out = out[:top_n]

    click.echo(json.dumps(out, ensure_ascii=False))


@click.command(name="list")
@click.option("--limit", default=20, type=int, show_default=True)
@click.option("--type", "type_", default=None)
@click.option("--json", "as_json", is_flag=True)
def list_cmd(limit: int, type_: str | None, as_json: bool) -> None:
    """Recent memories by `updated` desc."""
    from memo.memory import Memory

    mem = Memory(Config.from_env())
    items = mem.list(limit=limit, type_=type_)
    if as_json:
        click.echo(json.dumps([r.to_dict() for r in items], ensure_ascii=False, indent=2))
        return
    if not items:
        console.print("[dim]vacío[/dim]")
        return
    tbl = Table(show_lines=False, expand=True)
    tbl.add_column("updated", width=20)
    tbl.add_column("type", width=10)
    tbl.add_column("title", overflow="fold")
    tbl.add_column("tags", overflow="fold")
    for r in items:
        tbl.add_row(r.updated[:19], r.type, r.title, ", ".join(r.tags) or "—")
    console.print(tbl)


@click.command()
@click.argument("id_")
@click.option("--json", "as_json", is_flag=True)
def get(id_: str, as_json: bool) -> None:
    """Fetch one memory by id."""
    from memo.memory import Memory

    mem = Memory(Config.from_env())
    rec = _resolved(lambda: mem.get(id_))
    if rec is None:
        console.print(f"[red]not found:[/red] {id_}")
        sys.exit(1)
    if as_json:
        click.echo(json.dumps(rec.to_dict(), ensure_ascii=False, indent=2))
        return
    console.print(Panel.fit(
        f"[bold]{rec.title}[/bold]\n"
        f"[dim]id:[/dim] {rec.id}  [dim]type:[/dim] {rec.type}\n"
        f"[dim]tags:[/dim] {', '.join(rec.tags) or '—'}\n"
        f"[dim]created:[/dim] {rec.created}\n"
        f"[dim]updated:[/dim] {rec.updated}\n\n"
        f"{rec.body}",
        title=rec.title, border_style="cyan",
    ))


@click.command()
@click.argument("id_")
@click.option("--title", default=None)
@click.option(
    "--type", "type_",
    type=click.Choice(
        ["decision", "fact", "bug", "feedback", "preference", "note", "manual"],
    ),
    default=None,
)
@click.option("--tag", "-t", "tags", multiple=True, help="Replaces existing tags.")
@click.option(
    "--content", default=None,
    help="Replace body. Use '-' to read from stdin.",
)
@click.option("--json", "as_json", is_flag=True)
def update(
    id_: str,
    title: str | None,
    type_: str | None,
    tags: tuple[str, ...],
    content: str | None,
    as_json: bool,
) -> None:
    """Patch fields on an existing memory. Re-embeds only if body changed."""
    from memo.memory import Memory

    if content == "-":
        content = sys.stdin.read()

    mem = Memory(Config.from_env())
    rec = _resolved(lambda: mem.update(
        id_,
        title=title,
        type_=type_,
        tags=list(tags) if tags else None,
        content=content,
    ))
    if rec is None:
        console.print(f"[red]not found:[/red] {id_}")
        sys.exit(1)
    if as_json:
        click.echo(json.dumps(rec.to_dict(), ensure_ascii=False, indent=2))
        return
    console.print(Panel.fit(
        f"[bold]{rec.title}[/bold]\n"
        f"[dim]id:[/dim] {rec.id}  [dim]type:[/dim] {rec.type}\n"
        f"[dim]tags:[/dim] {', '.join(rec.tags) or '—'}\n"
        f"[dim]updated:[/dim] {rec.updated}",
        title="✓ updated", border_style="yellow",
    ))


@click.command()
@click.option("--force", is_flag=True,
              help="Re-embed ALL indexed entries regardless of body_hash. "
                   "Use after embedder swap or composition change.")
@click.option("--json", "as_json", is_flag=True)
def reindex(force: bool, as_json: bool) -> None:
    """Re-scan memory dir, re-embed entries with body_hash mismatch.

    Run after editing memory `.md` files directly in Obsidian, or after
    restoring memories from a backup. Use `--force` to re-embed every
    entry (slower; needed after model/composition changes).
    """
    from memo.memory import Memory

    mem = Memory(Config.from_env())
    counts = mem.reindex(force=force)
    if as_json:
        click.echo(json.dumps(counts, indent=2))
        return
    console.print(
        f"checked: [cyan]{counts['checked']}[/cyan]  "
        f"reindexed: [yellow]{counts['reindexed']}[/yellow]  "
        f"added: [green]{counts['added']}[/green]  "
        f"skipped: [dim]{counts['skipped']}[/dim]",
    )






@click.command()
@click.argument("id_")
@click.option("--yes", is_flag=True, help="Skip confirmation.")
def delete(id_: str, yes: bool) -> None:
    """Delete one memory by id."""
    from memo.memory import Memory

    mem = Memory(Config.from_env())
    if not yes:
        click.confirm(f"Delete memory {id_!r}? This removes the .md and the index entry.", abort=True)
    ok = _resolved(lambda: mem.delete(id_))
    console.print(f"[{'green' if ok else 'red'}]{'✓ deleted' if ok else 'not found'}[/]: {id_}")


@click.command()
@click.option("--limit", default=20, type=int, show_default=True)
@click.option("--op", default=None,
              type=click.Choice(["save", "update", "delete"]),
              help="Filter to one op type.")
@click.option("--id", "record_id", default=None,
              help="Filter to events for one record (full id or unique prefix).")
@click.option("--json", "as_json", is_flag=True)
def history(limit: int, op: str | None, record_id: str | None, as_json: bool) -> None:
    """Recent save/update/delete events. Append-only audit log."""
    from memo.memory import Memory

    mem = Memory(Config.from_env())
    if record_id and len(record_id) < 32:
        # Resolve prefix → full id (audit log stores full ids).
        resolved = mem.resolve_id(record_id)
        if resolved is None:
            console.print(f"[red]not found:[/red] {record_id}")
            sys.exit(1)
        record_id = resolved
    rows = mem.history.list_recent(limit=limit, op=op, record_id=record_id)
    if as_json:
        click.echo(json.dumps(rows, ensure_ascii=False, indent=2, default=str))
        return
    if not rows:
        console.print("[dim]no events[/dim]")
        return
    tbl = Table(show_lines=False, expand=True)
    tbl.add_column("ts", width=20)
    tbl.add_column("op", width=7)
    tbl.add_column("id", width=10)
    tbl.add_column("title", overflow="fold")
    tbl.add_column("delta", overflow="fold")
    for r in rows:
        delta = ""
        if r.get("delta"):
            delta = ", ".join(f"{k}" for k in r["delta"])
        tbl.add_row(
            (r["ts"] or "")[:19], r["op"], (r["record_id"] or "")[:8],
            r["title"] or "—", delta or "—",
        )
    console.print(tbl)


@click.command(name="ocr-image")
@click.argument("image_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--json", "as_json", is_flag=True)
def ocr_image(image_path: str, as_json: bool) -> None:
    """Extract text from an image using Apple Vision OCR.

    Results cached by SHA256 under `<state_dir>/ocr_cache`. Returns the
    raw extracted text on stdout (or JSON envelope with `--json`).
    Empty output indicates Vision unavailable or no text recognized.
    """

    from memo.ocr import extract_text_cached, vision_available

    cfg = Config.from_env()
    cache_dir = cfg.state_dir / "ocr_cache"
    if not vision_available():
        if as_json:
            click.echo(json.dumps({"text": "", "error": "vision unavailable"}))
        else:
            console.print("[yellow]Apple Vision not available[/yellow]")
        return
    text = extract_text_cached(Path(image_path), cache_dir=cache_dir)
    if as_json:
        click.echo(json.dumps({"text": text, "cached": True}))
    else:
        click.echo(text)


@click.command()
@click.argument("id_", metavar="ID")
@click.option("--json", "as_json", is_flag=True)
def provenance(id_: str, as_json: bool) -> None:
    """Provenance trail for one memoria.

    Returns the current synapse_*/agent_* keys plus every save/update
    event carrying its own provenance snapshot. Useful to audit which
    agent / trace_id / route_reason produced each version of a memoria.
    """
    from memo.memory import Memory

    mem = Memory(Config.from_env())
    payload = mem.provenance(id_)
    if payload is None:
        console.print(f"[red]not found:[/red] {id_}")
        sys.exit(1)
    if as_json:
        click.echo(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
        return
    cur = payload.get("current") or {}
    if cur:
        console.print(Panel.fit(
            "\n".join(f"[dim]{k}:[/dim] {v}" for k, v in cur.items()),
            title=f"provenance {payload['id'][:8]}", border_style="cyan",
        ))
    else:
        console.print(f"[dim]no provenance for {payload['id'][:8]} (current state)[/dim]")
    events = payload.get("events") or []
    if not events:
        return
    tbl = Table(show_lines=False, expand=True, title="history")
    tbl.add_column("ts", width=20)
    tbl.add_column("op", width=7)
    tbl.add_column("provenance", overflow="fold")
    for ev in events:
        prov = ev.get("provenance") or {}
        prov_str = ", ".join(f"{k}={v}" for k, v in prov.items()) if prov else "—"
        tbl.add_row((ev.get("ts") or "")[:19], ev.get("op") or "", prov_str)
    console.print(tbl)


@click.command(name="extract-entities")
@click.option("--all", "all_", is_flag=True, help="Process every memoria in the store.")
@click.option("--id", "id_", default=None, multiple=True,
              help="Repeatable. Process specific memoria id(s) (full or prefix).")
@click.option("--force", is_flag=True,
              help="Re-extract even if memoria already has entity links (default skips).")
@click.option("--json", "as_json", is_flag=True)
def extract_entities(all_: bool, id_: tuple[str, ...], force: bool, as_json: bool) -> None:
    """Extract named entities (person/project/technology/file/org/concept)
    from memoria bodies via Qwen2.5-3B and write them to the graph DB.

    Cost: ~0.5-1s per memoria. 223-doc corpus ≈ 2-4 min.
    """
    from memo.memory import Memory

    if not all_ and not id_:
        click.echo("pass --all or one or more --id <prefix>", err=True)
        sys.exit(2)

    mem = Memory(Config.from_env())
    resolved_ids: list[str] | None = None
    if id_:
        resolved_ids = []
        for raw in id_:
            r = _resolved(lambda raw=raw: mem.resolve_id(raw))
            if r is None:
                console.print(f"[red]not found:[/red] {raw}")
                sys.exit(1)
            resolved_ids.append(r)

    counts = mem.extract_entities(
        ids=resolved_ids, all_=all_, skip_already_indexed=not force,
    )
    if as_json:
        click.echo(json.dumps(counts, indent=2))
        return
    console.print(
        f"processed: [cyan]{counts['processed']}[/cyan]  "
        f"entities: [green]{counts['entities_extracted']}[/green]  "
        f"links: [green]{counts['links_written']}[/green]  "
        f"skipped: [dim]{counts['skipped']}[/dim]  "
        f"errors: [red]{counts['errors']}[/red]",
    )


@click.command()
@click.option("--limit", default=30, type=int, show_default=True)
@click.option("--type", "type_", default=None,
              type=click.Choice(["person", "project", "technology", "file", "org", "concept"]),
              help="Filter by entity type.")
@click.option("--json", "as_json", is_flag=True)
def entities(limit: int, type_: str | None, as_json: bool) -> None:
    """Top entities by mention count."""
    from memo.memory import Memory

    mem = Memory(Config.from_env())
    rows = mem.graph.top_entities(limit=limit, type_=type_)
    if as_json:
        click.echo(json.dumps(rows, indent=2))
        return
    if not rows:
        console.print("[dim]no entities indexed — run `memo extract-entities --all` first[/dim]")
        return
    tbl = Table(show_lines=False, expand=True)
    tbl.add_column("count", justify="right", width=6)
    tbl.add_column("type", width=12)
    tbl.add_column("name", overflow="fold")
    tbl.add_column("first_seen", width=10)
    tbl.add_column("last_seen", width=10)
    for r in rows:
        tbl.add_row(
            str(r["mention_count"]), r["type"], r["name"],
            (r["first_seen"] or "")[:10], (r["last_seen"] or "")[:10],
        )
    console.print(tbl)


@click.command()
@click.argument("name")
@click.option("--type", "type_", default=None,
              type=click.Choice(["person", "project", "technology", "file", "org", "concept"]))
@click.option("--json", "as_json", is_flag=True)
def entity(name: str, type_: str | None, as_json: bool) -> None:
    """Memorias that mention an entity."""
    from memo.memory import Memory

    mem = Memory(Config.from_env())
    ids = mem.graph.entity_memorias(name, type_=type_)
    if as_json:
        click.echo(json.dumps(ids, indent=2))
        return
    if not ids:
        console.print(f"[dim]no memorias mention {name!r}{f' ({type_})' if type_ else ''}[/dim]")
        return
    console.print(f"[bold]{len(ids)}[/bold] memoria(s) mention [cyan]{name}[/cyan]:")
    for mid in ids[:50]:
        rec = mem.store.get(mid)
        if rec:
            console.print(f"  · [{mid[:8]}] {rec['title'][:60]} [dim]({rec['updated'][:10]})[/dim]")
    if len(ids) > 50:
        console.print(f"  · …and {len(ids) - 50} more")


@click.command()
@click.option("--category", default=None,
              type=click.Choice(["legacy_extra", "few_tags", "body_skinny", "untitled"]),
              help="Show only one category. Default: summary of all.")
@click.option("--limit", default=20, type=int, show_default=True,
              help="Max entries per category in the report.")
@click.option("--json", "as_json", is_flag=True)
def lint(category: str | None, limit: int, as_json: bool) -> None:
    """Surface memorias with quality issues. Read-only — does not edit
    anything. Use to plan a manual cleanup pass.
    """
    from memo.memory import Memory

    mem = Memory(Config.from_env())
    report = mem.lint()
    if category:
        report = {category: report.get(category, [])}
    if as_json:
        click.echo(json.dumps(report, ensure_ascii=False, indent=2, default=str))
        return
    for cat, rows in report.items():
        n = len(rows)
        if n == 0:
            console.print(f"[green]✓[/green] {cat}: 0")
            continue
        console.print(f"[yellow]{cat}[/yellow]: {n}")
        for entry in rows[:limit]:
            console.print(
                f"  · {entry['id'][:8]} · {entry['title'][:60]} · [dim]{entry['reason']}[/dim]"
            )
        if n > limit:
            console.print(f"  · …and {n - limit} more")




@click.command()
@click.argument("zip_path", type=click.Path(exists=True))
@click.option("--reindex", is_flag=True,
              help="After restoring .md files, run `memo reindex` to "
                   "rebuild the index from disk (use when restoring without "
                   "the bundled state DBs, or across embedder model versions).")
@click.option("--yes", is_flag=True, help="Skip confirmation.")
def restore(zip_path: str, reindex: bool, yes: bool) -> None:
    """Restore from a backup zip created by `memo backup`.

    Extracts memory `.md` files into the vault and (optionally) the
    state DBs. **Will overwrite** matching files in the vault and
    state dir — confirmation required unless `--yes`.
    """
    import zipfile

    cfg = Config.from_env()
    cfg.ensure_dirs()

    with zipfile.ZipFile(zip_path, "r") as zf:
        try:
            manifest = json.loads(zf.read("manifest.json"))
        except KeyError:
            manifest = None
        if manifest:
            console.print(
                f"backup created: {manifest.get('created')}  "
                f"memorias: {manifest.get('n_md')}  "
                f"embedder: {manifest.get('embedder_model')}",
            )
        if not yes:
            click.confirm(
                f"Extract into {cfg.data_dir} + {cfg.state_dir}? "
                "Existing files will be overwritten.", abort=True,
            )
        # Stream entries.
        n_md = n_db = 0
        for info in zf.infolist():
            if info.filename == "manifest.json":
                continue
            data = zf.read(info)
            if info.filename.startswith("memory/"):
                rel = info.filename[len("memory/"):]
                dest = cfg.data_dir / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(data)
                n_md += 1
            elif info.filename.startswith("state/"):
                rel = info.filename[len("state/"):]
                dest = cfg.state_dir / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(data)
                n_db += 1

    console.print(
        f"[green]✓[/green] restored {n_md} memorias + {n_db} state DB(s) "
        f"into {cfg.data_dir}",
    )

    if reindex:
        from memo.memory import Memory
        mem = Memory(Config.from_env())
        # Force re-embed in case the bundled DB is from a different
        # embedder model — rebuilds vectors from .md authoritative state.
        counts = mem.reindex(force=True)
        console.print(
            f"reindex: checked {counts['checked']}  reindexed {counts['reindexed']}  "
            f"added {counts['added']}  skipped {counts['skipped']}",
        )
