"""Retrieval verbs for the memo CLI — search / ask / embed / chat-ask / rerank.

Extracted from cli_memory.py (god-module decomposition). Each command is a
standalone @click.command registered onto the root group in cli.py.
"""

from __future__ import annotations

import json
import sys

import click
from rich.panel import Panel
from rich.table import Table

from memo.cli_common import console
from memo.cli_common import get_memory as _get_memory
from memo.config import Config


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

    mem = _get_memory(Config.from_env())
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

    mem = _get_memory(Config.from_env())
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

    if batch_json is not None and text:
        raise click.UsageError("--batch-json and TEXT are mutually exclusive")
    if batch_json is None and not text:
        raise click.UsageError("provide TEXT or --batch-json")

    mem = _get_memory(Config.from_env())
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

    mem = _get_memory(Config.from_env())

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
