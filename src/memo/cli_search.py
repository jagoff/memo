"""Retrieval verbs for the memo CLI — search / ask / embed / chat-ask / rerank.

Extracted from cli_memory.py (god-module decomposition). Each command is a
standalone @click.command registered onto the root group in cli.py.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict

import click
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from memo.cli_common import console, log_cli_consult
from memo.cli_common import get_memory as _get_memory
from memo.config import Config
from memo.errors import MemoError
from memo.memory.record import MemoryRecord


def _format_source_score(score: object) -> str:
    return f"{score:.3f}" if isinstance(score, (int, float)) else "—"


def _sources_as_hits(out: dict) -> list[dict]:
    """Map an ask/chat-ask answer envelope's ``sources`` to recall-log hit dicts
    so the consult records which memories backed the answer."""
    hits: list[dict] = []
    for s in out.get("sources") or []:
        if not isinstance(s, dict):
            continue
        hits.append(
            {
                "id": s.get("id") or s.get("id_short") or "",
                "score": s.get("score"),
                "title": s.get("title") or "",
            }
        )
    return hits


def _compact_hit_dicts(hits: list[dict], body_chars: int) -> list[dict]:
    compact: list[dict] = []
    for hit in hits:
        if not isinstance(hit, dict):
            continue
        if body_chars < 0:
            compact.append(hit)
            continue
        body = str(hit.get("body") or "")
        if len(body) > body_chars:
            trimmed = dict(hit)
            trimmed["body"] = body[:body_chars].rstrip() + "…"
            trimmed["body_truncated"] = True
            compact.append(trimmed)
            continue
        compact.append(hit)
    return compact


def _fact_badge(hit: dict) -> str:
    extra = hit.get("extra") if isinstance(hit.get("extra"), dict) else {}
    facts = extra.get("related_fact_edges") if isinstance(extra, dict) else None
    return f" facts:{len(facts)}" if isinstance(facts, list) and facts else ""


def _require_valid_as_of(as_of: str | None) -> None:
    """Refuse a malformed `--as-of` before the query runs.

    The store keeps every row on a bound it cannot parse, so without this the
    query answers from the PRESENT while looking like it honoured `--as-of`.
    """
    from memo.asof import validate_as_of

    try:
        validate_as_of(as_of)
    except MemoError as exc:
        raise click.ClickException(str(exc)) from exc


def _default_search_json_body_chars() -> int:
    from memo.flags import flag_int

    value = flag_int("MEMO_SEARCH_JSON_BODY_CHARS")
    return 280 if value is None else value


def _render_search_table(
    hits: list[MemoryRecord],
    hit_dicts: list[dict],
    *,
    explain: bool,
    trace: list[dict] | None,
) -> None:
    """Render the `memo search` results table plus the optional --explain
    ranking-reason table. Pure presentation — no query/consult logic."""
    tbl = Table(show_lines=False, expand=True)
    tbl.add_column("score", justify="right", width=6)
    tbl.add_column("type", width=10)
    tbl.add_column("title", overflow="fold")
    tbl.add_column("tags", overflow="fold")
    for h in hits:
        h_dict = h.to_dict()
        tbl.add_row(
            f"{h.score:.3f}" if h.score is not None else "—",
            escape(h.type),
            escape(h.title + _fact_badge(h_dict)),
            escape(", ".join(h.tags) or "—"),
        )
    console.print(tbl)
    if explain and trace is not None:
        reason_tbl = Table(title="Why these ranked", show_header=False, expand=True)
        reason_tbl.add_column("hit", width=10, style="dim")
        reason_tbl.add_column("reason", overflow="fold")
        for hit in hit_dicts:
            exp = hit.get("explain") if isinstance(hit.get("explain"), dict) else {}
            why = exp.get("why") if isinstance(exp, dict) else None
            if not why:
                continue
            reason_tbl.add_row(str(hit.get("id") or "")[:8], "; ".join(str(w) for w in why[:3]))
        if reason_tbl.row_count:
            console.print(reason_tbl)
        console.print(
            f"[dim]search trace stages: {', '.join(str(t.get('stage')) for t in trace)}[/dim]"
        )


@click.command()
@click.argument("query")
@click.option("--limit", default=10, type=click.IntRange(1, 500), show_default=True)
@click.option("--type", "type_", default=None, help="Filter by record type.")
@click.option(
    "--mode",
    default="hybrid",
    type=click.Choice(["hybrid", "vec", "bm25", "exact"]),
    show_default=True,
    help="hybrid = RRF fusion of vec + bm25 (default). vec = semantic only. "
    "bm25 = keyword only. exact = strict keyword AND with tag/title boost.",
)
@click.option(
    "--rerank/--no-rerank",
    "use_rerank",
    default=None,
    help="Force enable (--rerank) or disable (--no-rerank) cross-encoder reranking "
    "for this invocation, overriding MEMO_RERANKER_ENABLED. Only meaningful with "
    "--mode hybrid.",
)
@click.option(
    "--body-chars",
    default=None,
    type=int,
    help="Preview length for JSON bodies (use -1 for full bodies). "
    "Default: MEMO_SEARCH_JSON_BODY_CHARS (280).",
)
@click.option("--json", "as_json", is_flag=True)
@click.option("--explain", is_flag=True, help="Include per-hit ranking explanation details.")
@click.option(
    "--as-of",
    "as_of",
    default=None,
    help="Valid-time recall: an ISO date (YYYY-MM-DD) or full timestamp. Returns "
    "records as their world-validity stood at that time — a since-superseded "
    "fact resurfaces — overriding the default currently-valid gate.",
)
@click.option(
    "--source",
    default=None,
    help="Identify the calling client so the consult is "
    "attributed in `memo usefulness`. Falls back to the MEMO_SOURCE env var.",
)
def search(
    query: str,
    limit: int,
    type_: str | None,
    mode: str,
    use_rerank: bool | None,
    body_chars: int | None,
    as_json: bool,
    explain: bool,
    as_of: str | None,
    source: str | None,
) -> None:
    """Top-k search — hybrid (semantic + keyword) by default."""
    import time

    _require_valid_as_of(as_of)

    if body_chars is None:
        body_chars = _default_search_json_body_chars()

    cfg = Config.from_env()
    if use_rerank is True:
        from memo.platform_detect import is_apple_silicon

        if is_apple_silicon():
            cfg = cfg.model_copy(update={"reranker_enabled": True})
        else:
            click.echo(
                "note: --rerank ignored — the cross-encoder reranker requires "
                "the MLX runtime (Apple Silicon).",
                err=True,
            )
    mem = _get_memory(cfg)
    disable_reranker = use_rerank is False
    t0 = int(time.time() * 1000)
    trace = None
    degraded: list[str] = []
    if explain:
        from memo.search_explain import build_search_explanations

        envelope = mem.search_with_trace(
            query,
            limit=limit,
            type_=type_,
            mode=mode,
            disable_reranker=disable_reranker,
            quality_rerank=True,
            as_of=as_of,
            _degraded=degraded,
        )
        hits = envelope["hits"]
        trace = envelope.get("trace") or []
        explanations = build_search_explanations(hits, trace)
    else:
        hits = mem.search(
            query,
            limit=limit,
            type_=type_,
            mode=mode,
            disable_reranker=disable_reranker,
            quality_rerank=True,
            as_of=as_of,
            _degraded=degraded,
        )
        explanations = {}
    hit_dicts = _compact_hit_dicts([h.to_dict() for h in hits], body_chars)
    if explain:
        for hit in hit_dicts:
            hit["explain"] = explanations.get(str(hit.get("id") or ""), {})
    log_cli_consult(cfg, verb="search", query=query, hits=hit_dicts, t0_ms=t0, source=source)
    if degraded:
        # Human-readable note on stderr only, so it never contaminates piped
        # or --json stdout. `--json` stays a bare array unconditionally (the
        # top-level type must not depend on a runtime condition the caller
        # cannot predict) -- stderr is the only degradation channel for now.
        click.secho(f"degraded: {', '.join(degraded)} (search budget)", dim=True, err=True)
    if as_json:
        click.echo(json.dumps(hit_dicts, ensure_ascii=False, indent=2))
        return
    if not hits:
        console.print("[dim]no results[/dim]")
        return
    _render_search_table(hits, hit_dicts, explain=explain, trace=trace)


@click.command(name="context")
@click.argument("question")
@click.option("--k", default=7, type=int, show_default=True, help="Top-K memories to include.")
@click.option("--type", "type_", default=None, help="Restrict retrieval to one record type.")
@click.option(
    "--snippet-chars",
    default=700,
    type=int,
    show_default=True,
    help="Per-memory snippet length.",
)
@click.option(
    "--budget-chars",
    default=6000,
    type=int,
    show_default=True,
    help="Approximate prompt text budget.",
)
@click.option("--profile/--no-profile", "include_profile", default=True, show_default=True)
@click.option("--dynamic/--no-dynamic", "include_dynamic", default=True, show_default=True)
@click.option("--json", "as_json", is_flag=True)
@click.option(
    "--source",
    default=None,
    help="Identify the calling layer so the consult is attributed in `memo usefulness`.",
)
def context_cmd(
    question: str,
    k: int,
    type_: str | None,
    snippet_chars: int,
    budget_chars: int,
    include_profile: bool,
    include_dynamic: bool,
    as_json: bool,
    source: str | None,
) -> None:
    """Build prompt-ready memory context without calling the answer LLM."""
    import time

    from memo.context_surface import build_context_surface, consult_hits_from_context
    from memo.flags import flag_bool

    if not flag_bool("MEMO_CONTEXT_SURFACE"):
        raise click.ClickException("memo context is disabled by MEMO_CONTEXT_SURFACE=0.")
    cfg = Config.from_env()
    mem = _get_memory(cfg)
    t0 = int(time.time() * 1000)
    payload = build_context_surface(
        mem,
        question,
        k=k,
        type_=type_,
        snippet_chars=snippet_chars,
        budget_chars=budget_chars,
        include_profile=include_profile,
        include_dynamic=include_dynamic,
    )
    log_cli_consult(
        cfg,
        verb="context",
        query=question,
        hits=consult_hits_from_context(payload),
        t0_ms=t0,
        source=source,
    )
    if as_json:
        click.echo(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    console.print(
        Panel.fit(
            escape(payload["prompt"]) if payload["prompt"] else "[dim](no memory context)[/dim]",
            title=f"context: {escape(question[:60])}",
            border_style="cyan",
        )
    )


@click.command()
@click.argument("question")
@click.option(
    "--k",
    default=5,
    type=click.IntRange(1, 500),
    show_default=True,
    help="Top-K memories to feed the LLM as context.",
)
@click.option("--type", "type_", default=None, help="Restrict the retrieval to one record type.")
@click.option(
    "--snippet-chars",
    default=None,
    type=int,
    show_default="MEMO_ASK_SNIPPET_CHARS or 800",
    help="Preview length for retrieved memory snippets.",
)
@click.option("--json", "as_json", is_flag=True)
@click.option(
    "--source",
    default=None,
    help="Identify the calling layer so the consult is attributed in "
    "`memo usefulness` (falls back to MEMO_SOURCE).",
)
def ask(
    question: str,
    k: int,
    type_: str | None,
    snippet_chars: int | None,
    as_json: bool,
    source: str | None,
) -> None:
    """RAG over the memory archive — synthesises a prose answer with
    inline `[id]` citations using MLXChat 7B over the top-K hybrid hits.
    """
    import time

    if not question.strip():
        raise click.ClickException("`question` must be non-empty")

    cfg = Config.from_env()
    mem = _get_memory(cfg)
    t0 = int(time.time() * 1000)
    out = mem.ask(question, k=k, type_=type_, snippet_chars=snippet_chars)
    log_cli_consult(
        cfg, verb="ask", query=question, hits=_sources_as_hits(out), t0_ms=t0, source=source
    )
    if as_json:
        click.echo(json.dumps(out, ensure_ascii=False, indent=2))
        return
    console.print(
        Panel.fit(
            escape(out["answer"]) if out["answer"] else "[dim](no answer)[/dim]",
            title=f"❓ {question[:60]}",
            border_style="cyan",
        )
    )
    if out["sources"]:
        console.print("[dim]sources:[/dim]")
        for s in out["sources"]:
            id_short = s["id_short"]
            console.print(
                f"  [dim]{escape(f'[{id_short}]')}[/dim] {escape(s['title'][:60])}  "
                f"[dim](score {_format_source_score(s.get('score'))})[/dim]"
            )


@click.command(name="context-pack")
@click.argument("question")
@click.option(
    "--k",
    default=5,
    type=int,
    show_default=True,
    help="Top-K memories to interpret as a context pack.",
)
@click.option("--type", "type_", default=None, help="Restrict the retrieval to one record type.")
@click.option(
    "--snippet-chars",
    default=800,
    type=int,
    show_default=True,
    help="Preview length for retrieved memory snippets.",
)
@click.option(
    "--code",
    default=None,
    help="Anchor the pack on code: a symbol name, or a file path when it "
    "contains '/'. Adds a '## Código relacionado' section (1-hop codegraph "
    "neighbors + memories citing them); silently omitted when no codegraph "
    "index is available.",
)
@click.option("--json", "as_json", is_flag=True)
@click.option(
    "--source",
    default=None,
    help="Identify the calling layer so the consult is attributed in "
    "`memo usefulness` (falls back to MEMO_SOURCE).",
)
def context_pack_cmd(
    question: str,
    k: int,
    type_: str | None,
    snippet_chars: int,
    code: str | None,
    as_json: bool,
    source: str | None,
) -> None:
    """Build an explicit composed context pack without running the LLM."""
    import time

    from memo.context_pack import (
        DEFAULT_BUDGET_CHARS,
        build_context_pack,
        code_related_section,
        consult_hits_from_pack,
    )
    from memo.flags import flag_bool

    cfg = Config.from_env()
    if not flag_bool("MEMO_CONTEXT_PACK"):
        raise click.ClickException(
            "context-pack is disabled. Set MEMO_CONTEXT_PACK=1 to enable explicit context-pack tools."
        )
    mem = _get_memory(cfg)
    t0 = int(time.time() * 1000)
    hits = mem.search(
        question,
        limit=k,
        type_=type_,
        mode="hybrid",
        disable_reranker=True,
        read_through=False,
        quality_rerank=True,
    )
    code_section = code_related_section(code, mem.store._conn) if code else ""
    budget_chars = DEFAULT_BUDGET_CHARS
    if code_section:
        budget_chars = max(1, budget_chars - len(code_section) - 2)
    pack = build_context_pack(
        question, hits, snippet_chars=snippet_chars, budget_chars=budget_chars
    )
    log_cli_consult(
        cfg,
        verb="context_pack",
        query=question,
        hits=consult_hits_from_pack(pack),
        t0_ms=t0,
        source=source,
    )
    payload = asdict(pack)
    if code_section:
        payload["code_context"] = code_section
    if as_json:
        click.echo(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    console.print(
        Panel.fit(
            "\n\n".join(part for part in (pack.to_prompt(), code_section) if part),
            title=f"context-pack: {question[:60]}",
            border_style="cyan",
        )
    )


@click.command(name="embed")
@click.argument("text", required=False)
@click.option(
    "--batch-json",
    type=click.File("r"),
    default=None,
    help="Read JSON list of texts from this path (or '-' for stdin). "
    "Each text is embedded with the SYMMETRIC (document) prefix. "
    "Mutually exclusive with positional TEXT.",
)
def embed_cmd(text: str | None, batch_json) -> None:
    """Compute embedding vector(s) using memo's MLX embedder.

    Single (asymmetric query prefix):
        memo embed "tell me about Greece"

    Batch (symmetric document prefix, single MLX forward pass):
        echo '["alpha","beta","gamma"]' | memo embed --batch-json -

    Output: one JSON object per invocation (no indent), written to
    stdout. Shape:
      single: {"vector": [...], "dim": int, "model": "..."}
      batch:  {"vectors": [[...], ...], "dim": int, "model": "..."}

    This native RPC lets every client share Memo's exact query/document
    embedding space.
    """

    if batch_json is not None and text:
        raise click.UsageError("--batch-json and TEXT are mutually exclusive")
    if batch_json is None and not text:
        raise click.UsageError("provide TEXT or --batch-json")

    if batch_json is not None:
        try:
            texts = json.load(batch_json)
        except json.JSONDecodeError as exc:
            raise click.UsageError(f"--batch-json: invalid JSON: {exc}") from exc
        if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
            raise click.UsageError("--batch-json: expected JSON list of strings")
        if not texts:
            raise click.UsageError("--batch-json: list must not be empty")
    else:
        assert text is not None, "internal error: text is None in embed-text mode"

    mem = _get_memory(Config.from_env())
    try:
        if batch_json is not None:
            vecs = mem.embedder.embed(texts)
            dim = len(vecs[0]) if vecs else 0
            out = {"vectors": vecs, "dim": dim, "model": mem.store.embedder_model}
        else:
            vec = mem.embedder.embed_query(text)
            out = {"vector": vec, "dim": len(vec), "model": mem.store.embedder_model}
    finally:
        mem.close()
    sys.stdout.write(json.dumps(out, ensure_ascii=False) + "\n")
    sys.stdout.flush()


@click.command(name="chat-ask")
@click.argument("question")
@click.option(
    "--k", default=7, type=int, show_default=True, help="Top-K memories to feed the LLM as context."
)
@click.option("--type", "type_", default=None, help="Restrict the retrieval to one record type.")
@click.option(
    "--snippet-chars",
    default=None,
    type=int,
    show_default="MEMO_ASK_SNIPPET_CHARS or 800",
    help="Preview length for retrieved memory snippets.",
)
@click.option(
    "--history-json",
    type=click.File("r"),
    default=None,
    help="Conversation history JSON: list of {role,text}. '-' reads stdin.",
)
@click.option(
    "--context-json",
    type=click.File("r"),
    default=None,
    help="Caller-supplied federation context (for example a verified Memo bundle).",
)
@click.option("--json", "as_json", is_flag=True)
@click.option(
    "--stream",
    "as_stream",
    is_flag=True,
    help=(
        "Emit one NDJSON event per line (context/token/done) flushed "
        "immediately. Forces JSON output and disables the panel."
    ),
)
@click.option(
    "--source",
    default=None,
    help="Identify the calling layer so the consult is attributed in "
    "`memo usefulness` (falls back to MEMO_SOURCE).",
)
def chat_ask(
    question: str,
    k: int,
    type_: str | None,
    snippet_chars: int | None,
    history_json,
    context_json,
    as_json: bool,
    as_stream: bool,
    source: str | None,
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

    import time

    cfg = Config.from_env()
    mem = _get_memory(cfg)
    t0 = int(time.time() * 1000)

    if as_stream:
        import sys

        stream_hits: list[dict] = []
        for event in mem.chat_ask_stream(
            question,
            k=k,
            type_=type_,
            history=history,
            context=context,
            snippet_chars=snippet_chars,
        ):
            if isinstance(event, dict) and event.get("sources") and not stream_hits:
                stream_hits = _sources_as_hits(event)
            sys.stdout.write(json.dumps(event, ensure_ascii=False) + "\n")
            sys.stdout.flush()
        log_cli_consult(
            cfg, verb="chat_ask", query=question, hits=stream_hits, t0_ms=t0, source=source
        )
        return

    envelope = mem.chat_ask(
        question,
        k=k,
        type_=type_,
        history=history,
        context=context,
        snippet_chars=snippet_chars,
    )
    log_cli_consult(
        cfg,
        verb="chat_ask",
        query=question,
        hits=_sources_as_hits(envelope),
        t0_ms=t0,
        source=source,
    )
    if as_json:
        click.echo(json.dumps(envelope, ensure_ascii=False, indent=2))
        return
    console.print(
        Panel.fit(
            escape(envelope["answer"]) if envelope["answer"] else "[dim](no answer)[/dim]",
            title=f"❓ {question[:60]}",
            border_style="magenta",
        )
    )
    if envelope["sources"]:
        console.print("[dim]sources:[/dim]")
        for s in envelope["sources"]:
            id_short = s.get("id_short", "?")
            console.print(
                f"  [dim]{escape(f'[{id_short}]')}[/dim] "
                f"{escape((s.get('title', '') or '')[:60])}  "
                f"[dim](score {_format_source_score(s.get('score'))})[/dim]"
            )


@click.command(name="recall")
@click.argument("query")
@click.option("--limit", default=5, type=click.IntRange(1, 500), show_default=True)
@click.option("--type", "type_", default=None, help="Filter by record type.")
@click.option("--json", "as_json", is_flag=True, help='Emit {"results": [...]} for callers.')
@click.option(
    "--body-chars",
    default=None,
    type=int,
    help="Preview length for JSON bodies (use -1 for full bodies). "
    "Default: MEMO_SEARCH_JSON_BODY_CHARS (280).",
)
@click.option(
    "--source",
    default=None,
    help="Identify the calling layer so the consult is attributed in "
    "`memo usefulness` (falls back to MEMO_SOURCE).",
)
def recall(
    query: str,
    limit: int,
    type_: str | None,
    as_json: bool,
    body_chars: int | None,
    source: str | None,
) -> None:
    """Hybrid recall for programmatic callers.

    Same retrieval as ``memo search`` but emits ``{"results": [...]}`` and
    attributes the consult so the client stops showing as silent in
    ``memo usefulness``.
    """
    import time

    if body_chars is None:
        body_chars = _default_search_json_body_chars()

    cfg = Config.from_env()
    mem = _get_memory(cfg)
    t0 = int(time.time() * 1000)
    hits = mem.search(query, limit=limit, type_=type_, mode="hybrid")
    results = []
    for h in hits:
        d = h.to_dict()
        d["kind"] = d.get("type")  # transport alias; Memo stores `type`
        results.append(d)
    results = _compact_hit_dicts(results, body_chars)
    log_cli_consult(cfg, verb="recall", query=query, hits=results, t0_ms=t0, source=source)
    if as_json:
        click.echo(json.dumps({"results": results}, ensure_ascii=False))
        return
    if not results:
        console.print("[dim]no results[/dim]")
        return
    for d in results:
        score = d.get("score")
        console.print(
            f"  [dim][{(d.get('id') or '')[:8]}][/dim] {(d.get('title') or '')[:60]}  "
            f"[dim](score {score:.3f})[/dim]"
            if isinstance(score, (int, float))
            else f"  [dim][{(d.get('id') or '')[:8]}][/dim] {(d.get('title') or '')[:60]}"
        )


@click.command(name="rerank")
@click.option("--query", "query", required=True, help="The search query.")
@click.option(
    "--hits-file",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="JSON file with hits array. If omitted, reads stdin.",
)
@click.option(
    "--top-n",
    type=int,
    default=None,
    help="Truncate output to top-N after reranking.",
)
@click.option(
    "--body-chars",
    type=int,
    default=1200,
    show_default=True,
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

    Lets any client reuse Memo's already-cached Qwen3-Reranker without
    loading the model in a second process.
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
        body_src = str(h.get("snippet") or h.get("body") or "")[: max(0, body_chars)]
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
