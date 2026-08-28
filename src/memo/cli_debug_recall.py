"""`memo debug-recall` — reproduce the recall pipeline for one prompt, outside a session.

Runs the same search + rank path the recall hook uses (`Memory.search` →
`recall_logic.rank_hits`) and prints a per-candidate breakdown: vec similarity,
BM25 leg score, search/rerank score, boosts applied, final rank, and whether
the hit cleared the min_sim floor — plus the active thresholds. Diagnostic
only, NOT the hook path, so loading MLX here is fine (imports stay deferred
per repo convention). `--json` for machine output.
"""

from __future__ import annotations

import contextlib
import json as _json
import os
from typing import Any

import click
from rich.table import Table

from memo.cli_common import console, get_memory
from memo.config import Config
from memo.flags import flag_bool, flag_float, flag_int, flag_str


def _run_debug_recall(prompt: str, cwd: str | None) -> dict[str, Any]:
    """Reproduce the recall ranking for ``prompt`` with a per-hit breakdown.

    Mirrors ``_recall_logic``'s flag resolution (same names, same defaults) and
    reuses its ranking core (``rank_hits`` + ``make_vec_cosine``) — the ranking
    itself is never reimplemented here.
    """
    from memo.recall_logic import RankKnobs, make_vec_cosine, rank_hits
    from memo.tiers import REFERENCE_TYPES

    cfg = Config.from_env()
    mem = get_memory(cfg)
    try:
        top_k = 3 if (_tk := flag_int("MEMO_RECALL_TOP_K")) is None else _tk
        _ms = flag_float("MEMO_RECALL_MIN_SIM")
        min_sim = 0.5 if _ms is None else _ms
        _pb = flag_float("MEMO_RECALL_PROJECT_BOOST")
        project_boost = 0.25 if _pb is None else _pb
        _gb = flag_float("MEMO_RECALL_GLOBAL_BOOST")
        global_boost = 0.10 if _gb is None else _gb
        mode = flag_str("MEMO_RECALL_MODE") or "vec"
        _mbc = flag_int("MEMO_RECALL_MIN_BODY_CHARS")
        min_body_chars = 40 if _mbc is None else _mbc
        contextual = flag_bool("MEMO_RECALL_CONTEXTUAL")
        mmr_lambda = flag_float("MEMO_RECALL_MMR_LAMBDA") or 0.0
        synthesis_boost = flag_float("MEMO_RECALL_SYNTHESIS_BOOST") or 0.0

        project_tag = None
        if project_boost > 0 and cwd:
            with contextlib.suppress(Exception):
                from memo.project import current_project_tag

                project_tag = current_project_tag(cwd)

        search_k = top_k * 3 if (project_tag or contextual) else top_k
        # Resolve exclusions through the SAME helper the live hook uses. This
        # used to read MEMO_RECALL_EXCLUDE_REFERENCE alone, missing the
        # Negative Recall branch that drops failure_pattern from the normal
        # section — so the diagnostic reported "● injected" for memories the
        # hook suppresses, on a channel it does not use.
        from memo.recall_logic import _recall_excluded_types

        excluded_types = _recall_excluded_types()
        exclude_types = excluded_types or None

        knobs = RankKnobs(
            top_k=top_k,
            min_sim=min_sim,
            min_body_chars=min_body_chars,
            mode=mode,
            project_tag=project_tag,
            project_boost=project_boost,
            global_boost=global_boost,
            contextual=contextual,
            mmr_lambda=mmr_lambda,
            synthesis_boost=synthesis_boost,
        )

        prefs: Any | None = None
        if contextual:
            with contextlib.suppress(Exception):
                prefs = mem.contextual.context.get_preferences()

        # Diagnostic-only, not a user-visible retrieval: without this, running
        # `memo debug-recall` writes an access-log row (search_ops.py's
        # `_stage_record_usage`) for whatever it surfaces, inflating
        # `access_count` on the memories being inspected — the same signal
        # `memo usefulness` / `dead_weight()` read to decide what's noise.
        traced = mem.search_with_trace(
            prompt,
            limit=search_k,
            mode=mode,
            recency=True,
            exclude_types=exclude_types,
            _track_usage=False,
        )
        candidates, trace = traced["hits"], traced["trace"]
        reranker_ran = any(t.get("stage") == "rerank" for t in trace)

        vec_cosine = make_vec_cosine(mem, prompt)
        explain: dict[str, dict[str, Any]] = {}
        ranked = rank_hits(
            candidates,
            knobs,
            vec_cosine=vec_cosine,
            preferences=prefs,
            explain=explain,
            query=prompt,
        )

        # Post-rank_hits output filters — the hook path (recall_logic._recall_logic)
        # applies MEMO_RECALL_SKIP_BELOW / MEMO_RECALL_GAP_THRESHOLD AFTER
        # rank_hits, so `injected` must honor them or debug-recall shows
        # "● injected" for a hit the real hook suppressed. Same resolution
        # (`or 0.0`) and same checks as _recall_logic.
        skip_below = flag_float("MEMO_RECALL_SKIP_BELOW") or 0.0
        gap_threshold = flag_float("MEMO_RECALL_GAP_THRESHOLD") or 0.0
        qualifying = list(ranked)
        skip_below_triggered = bool(
            skip_below > 0 and qualifying and (qualifying[0].score or 0.0) < skip_below
        )
        if skip_below_triggered:
            qualifying = []
        elif (
            gap_threshold > 0
            and len(qualifying) > 1
            and qualifying[0].score is not None
            and qualifying[1].score is not None
            and (qualifying[0].score - qualifying[1].score) > gap_threshold
        ):
            qualifying = qualifying[:1]
        injected_ids = {h.id for h in qualifying[:top_k]}

        # Supplementary display columns (never affect ranking): BM25 leg score
        # per hit when a keyword leg ran.
        bm25_scores: dict[str, float | None] = {}
        if mode in ("hybrid", "bm25", "exact"):
            with contextlib.suppress(Exception):
                rows = mem.store.search_bm25(
                    prompt, limit=max(search_k * 2, 20), exclude_types=exclude_types
                )
                bm25_scores = {r["id"]: r.get("score") for r in rows}

        hits_out: list[dict[str, Any]] = []
        for h in candidates:
            e = explain.get(h.id, {})
            hits_out.append(
                {
                    "id": h.id,
                    "id8": h.id[:8],
                    "title": h.title,
                    # `--json` only. A tenth table column pushes the boosts
                    # cell into wrapping at the 80-column width the module
                    # console falls back to when there is no TTY, which is
                    # what CI renders at.
                    "type": getattr(h, "type", None),
                    "vec_sim": vec_cosine(h),
                    "bm25": bm25_scores.get(h.id),
                    "search_score": e.get("raw_score", h.score),
                    "tier_boost": e.get("tier_boost"),
                    "preference_boost": e.get("preference_boost"),
                    "synthesis_boost": e.get("synthesis_boost"),
                    "mmr": e.get("mmr"),
                    "final_score": e.get("final_score"),
                    "gate_value": e.get("gate_value"),
                    "passed_min_sim": e.get("passed_min_sim"),
                    "passed_min_body": e.get("passed_min_body"),
                    "dropped": e.get("dropped"),
                    "rank": e.get("rank"),
                    "injected": h.id in injected_ids,
                }
            )
        hits_out.sort(
            key=lambda r: (
                r["rank"] is None,
                r["rank"] if r["rank"] is not None else 0,
                -(r["final_score"] or 0.0),
            )
        )

        config_out: dict[str, Any] = {
            "min_sim": min_sim,
            "top_k": top_k,
            "mode": mode,
            "search_k": search_k,
            "skip_below": skip_below,
            "gap_threshold": gap_threshold,
            "skip_below_triggered": skip_below_triggered,
            "reranker_enabled": bool(cfg.reranker_enabled),
            "reranker_ran": reranker_ran,
            "min_body_chars": min_body_chars,
            "project_tag": project_tag,
            "project_boost": project_boost,
            "global_boost": global_boost,
            "graph_signal": next(
                (item for item in trace if item.get("stage") == "graph_signal"),
                None,
            ),
            "mmr_lambda": mmr_lambda,
            "synthesis_boost": synthesis_boost,
            "contextual": contextual,
            "exclude_reference": bool(REFERENCE_TYPES & excluded_types),
            "excluded_types": sorted(excluded_types),
            "candidates": len(candidates),
            "qualifying": len(ranked),
        }
        return {"hits": hits_out, "config": config_out}
    finally:
        mem.close()


def _fmt_score(v: Any) -> str:
    return f"{v:.3f}" if isinstance(v, int | float) else "—"


def _boosts_cell(row: dict[str, Any]) -> str:
    parts = []
    for key, label in (
        ("tier_boost", "tier"),
        ("preference_boost", "pref"),
        ("synthesis_boost", "synth"),
    ):
        v = row.get(key)
        if v:
            parts.append(f"{label}{v:+.2f}")
    # MMR is a diversity re-ORDER, not a score delta — show the greedy
    # selection score so the reordering is visible in the breakdown.
    mmr = row.get("mmr")
    if isinstance(mmr, dict) and isinstance(mmr.get("mmr_score"), int | float):
        parts.append(f"mmr={mmr['mmr_score']:.2f}")
    return " ".join(parts) or "—"


def _floor_cell(row: dict[str, Any]) -> str:
    dropped = row.get("dropped")
    if dropped == "dedup":
        return "[dim]dup[/dim]"
    if dropped == "synthesis_covered":
        return "[dim]syn-covered[/dim]"
    if row.get("passed_min_sim") is False:
        return "[red]✗ min_sim[/red]"
    if row.get("passed_min_body") is False:
        return "[red]✗ min_body[/red]"
    return "[green]✓[/green]"


def _render(result: dict[str, Any], prompt: str) -> None:
    cfg = result["config"]
    console.print(
        f"[bold]memo debug-recall[/bold] · [cyan]{prompt}[/cyan]\n"
        f"[dim]mode={cfg['mode']} · min_sim={cfg['min_sim']} · top_k={cfg['top_k']} "
        f"(search_k={cfg['search_k']}) · reranker="
        f"{'on' if cfg['reranker_enabled'] else 'off'}"
        f"{' (ran)' if cfg['reranker_ran'] else ''}"
        f" · min_body_chars={cfg['min_body_chars']} · project_tag={cfg['project_tag'] or '—'}"
        f" · boosts proj+{cfg['project_boost']}/glob+{cfg['global_boost']}"
        f"/synth+{cfg['synthesis_boost']} · mmr_lambda={cfg['mmr_lambda']}"
        f" · skip_below={cfg['skip_below']} · gap_threshold={cfg['gap_threshold']}[/dim]"
    )
    if cfg.get("skip_below_triggered"):
        console.print(
            f"[yellow]skip_below triggered — best score < {cfg['skip_below']}, "
            "the real hook injects nothing[/yellow]"
        )
    hits = result["hits"]
    if not hits:
        console.print("[yellow]no candidates returned by search[/yellow]")
        return

    show_bm25 = cfg["mode"] in ("hybrid", "bm25", "exact")
    score_hdr = "rerank·fused" if cfg["reranker_ran"] else "search"
    table = Table(show_lines=False)
    table.add_column("rank", justify="right", no_wrap=True)
    table.add_column("id", no_wrap=True)
    table.add_column("title", max_width=30, overflow="ellipsis")
    table.add_column("vec", justify="right")
    if show_bm25:
        table.add_column("bm25", justify="right")
    table.add_column(score_hdr, justify="right")
    table.add_column("boosts")
    table.add_column("final", justify="right")
    table.add_column("floor")
    for row in hits:
        rank = row.get("rank")
        rank_cell = f"[bold]{rank}[/bold]" if rank is not None else "—"
        if row.get("injected"):
            rank_cell += " [green]●[/green]"
        cells = [
            rank_cell,
            row["id8"],
            row["title"] or "",
            _fmt_score(row.get("vec_sim")),
        ]
        if show_bm25:
            cells.append(_fmt_score(row.get("bm25")))
        cells.extend(
            [
                _fmt_score(row.get("search_score")),
                _boosts_cell(row),
                _fmt_score(row.get("final_score")),
                _floor_cell(row),
            ]
        )
        table.add_row(*cells)
    console.print(table)
    console.print(
        f"[dim]{cfg['candidates']} candidates → {cfg['qualifying']} qualifying · "
        f"● = injected (top {cfg['top_k']})[/dim]"
    )


@click.command(name="debug-recall")
@click.argument("prompt")
@click.option("--json", "as_json", is_flag=True, help="Machine-readable output.")
def debug_recall_cmd(prompt: str, *, as_json: bool = False) -> None:
    """Reproduce the recall pipeline for PROMPT and show the ranking breakdown.

    Diagnostic command (not the hook path): runs the same search + rank_hits
    pipeline the recall hook uses and shows, per candidate, the vec similarity,
    BM25 score, boosts applied, final rank and whether it passed the min_sim
    floor — so a bad recall can be diagnosed as absence, noise, or ranking.
    """
    result = _run_debug_recall(prompt, os.getcwd())
    if as_json:
        click.echo(_json.dumps(result, ensure_ascii=False, indent=2))
        return
    _render(result, prompt)
