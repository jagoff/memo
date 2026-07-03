"""`memo dream` maintenance passes — the per-phase implementations.

Extracted verbatim from `cli_dream.py` to keep that module focused on the
Click command wiring. Each `_run_*` helper performs one maintenance pass over
the corpus and returns a compact summary; `_build_orientation` is the
read-only pre-mutation inventory. All are imported (and re-exported) by
`cli_dream` so `from memo.cli_dream import _run_eviction` keeps working.
"""

from __future__ import annotations

import json
import logging as _logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from memo.cli_common import console
from memo.transcript_miner import mine_transcripts

if TYPE_CHECKING:
    from memo.config import Config
    from memo.memory.facade import Memory

_log = _logging.getLogger(__name__)


def _build_orientation(mem: Memory) -> dict:
    """Read-only corpus inventory — runs before any mutation."""
    conn = mem.store._conn
    result: dict = {
        "total": 0,
        "by_type": {},
        "low_roi": 0,
        "stale_candidates": 0,
        "open_contradictions": 0,
        "unindexed_entities": 0,
    }
    try:
        row = conn.execute("SELECT COUNT(*) AS n FROM meta WHERE type != 'reference'").fetchone()
        result["total"] = int(row["n"]) if row else 0
    except Exception:  # noqa: S110
        pass

    try:
        rows = conn.execute(
            "SELECT type, COUNT(*) AS n FROM meta WHERE type != 'reference' GROUP BY type"
        ).fetchall()
        result["by_type"] = {r["type"]: int(r["n"]) for r in rows}
    except Exception:  # noqa: S110
        pass

    try:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM meta m "
            "LEFT JOIN memory_health h ON h.id = m.id "
            "WHERE COALESCE(h.roi_score, 1.0) < 0.3 AND m.type != 'reference'"
        ).fetchone()
        result["low_roi"] = int(row["n"]) if row else 0
    except Exception:  # noqa: S110
        pass

    try:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM meta m "
            "LEFT JOIN access a ON a.id = m.id "
            "WHERE m.updated < datetime('now', '-365 days') "
            "AND COALESCE(a.access_count, 0) = 0 "
            "AND m.type != 'reference'"
        ).fetchone()
        result["stale_candidates"] = int(row["n"]) if row else 0
    except Exception:  # noqa: S110
        pass

    try:
        pairs = mem.contradict_store.list_open()
        result["open_contradictions"] = len(pairs)
    except Exception:  # noqa: S110
        pass

    try:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM meta m "
            "WHERE m.type != 'reference' "
            "AND NOT EXISTS ("
            "  SELECT 1 FROM entity_memory em "
            "  JOIN entities e ON e.id = em.entity_id "
            "  WHERE em.memory_id = m.id"
            ")"
        ).fetchone()
        result["unindexed_entities"] = int(row["n"]) if row else 0
    except Exception:  # noqa: S110
        pass

    return result


def _run_signal_gather(since_days: float, file_limit: int = 20) -> dict:
    """Run transcript mining and return a compact summary.

    Never raises — exceptions are captured in the returned dict.
    """
    try:
        res = mine_transcripts(since_days=since_days, file_limit=file_limit)
        return {
            "files_processed": res.get("files_processed", 0),
            "memories_saved": len(res.get("saved") or []),
            "skipped_dup": res.get("skipped_dup", 0),
        }
    except Exception as exc:
        return {"files_processed": 0, "memories_saved": 0, "skipped_dup": 0, "error": str(exc)}


def _run_prune_floor(
    mem: Memory,
    roi_floor: float,
    min_age_days: int,
    dry_run: bool,
) -> list[dict]:
    """Archive memories below roi_floor with zero access and age >= min_age_days.

    Returns list of {id, roi_score, days_old} candidates (even in dry-run).
    """
    candidates = mem.store.prune_floor_candidates(roi_floor=roi_floor, min_age_days=min_age_days)
    if not dry_run:
        for c in candidates:
            try:
                mem.lifecycle.archive_memory(c["id"])
            except Exception as exc:
                _log.warning("prune_floor: archive failed for %s: %s", c["id"], exc)
    return candidates


def _run_eviction(mem: Memory, max_count: int, dry_run: bool) -> list[dict]:
    """Archive LFU candidates until corpus size <= max_count.

    Returns list of {id, access_count} archived (or would-archive in dry-run).
    """
    # No defensive except here: a DB error must propagate to the cli_dream
    # caller (which records it in receipt["errors"]), not read as "evicted: 0".
    conn = mem.store._conn
    total_row = conn.execute(
        "SELECT COUNT(*) AS n FROM meta WHERE type != 'reference'"
    ).fetchone()
    total = int(total_row["n"]) if total_row else 0

    excess = total - max_count
    if excess <= 0:
        return []

    candidates = mem.store.eviction_candidates(
        policy="lfu",
        limit=excess,
        exclude_types={"reference", "synthesis"},
    )
    if not dry_run:
        for c in candidates:
            try:
                mem.lifecycle.archive_memory(c["id"])
            except Exception as exc:
                _log.warning("eviction: archive failed for %s: %s", c["id"], exc)
    return [{"id": c["id"], "access_count": c.get("access_count", 0)} for c in candidates]


def _run_compress(mem: Memory, threshold: int, dry_run: bool) -> list[dict]:
    """Compress verbose memories (body > threshold chars) to 2-3 sentences.

    Returns list of {id, original_len, compressed_len}.
    """
    # No defensive except here: a DB error must propagate to the cli_dream
    # caller (which records it in receipt["errors"]), not read as "compressed: 0".
    conn = mem.store._conn
    rows = conn.execute(
        "SELECT m.id, m.path FROM meta m "
        "JOIN fts ON fts.id = m.id "
        "WHERE m.type NOT IN ('reference','synthesis') "
        "AND length(fts.body) > ?",
        (threshold,),
    ).fetchall()

    if not rows:
        return []

    from memo.memory.record import chat_with_timeout

    chat = mem._ensure_chat()
    results = []
    for row in rows:
        mid = row["id"]
        try:
            rec = mem.get(mid)
            if not rec or not rec.body:
                continue
            body_len = len(rec.body)
            if body_len <= threshold:
                continue
            user_prompt = (
                "Compress the following memory note to 2-3 concise sentences "
                "preserving all key facts, decisions, and context. "
                "Output ONLY the compressed text, no preamble.\n\n" + rec.body[:4000]
            )
            chat_out = chat_with_timeout(
                chat,
                timeout=30,
                model=mem.cfg.helper_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a concise technical writer. Compress memory notes.",
                    },
                    {"role": "user", "content": user_prompt},
                ],
                options={"temperature": 0.0, "max_tokens": 256, "thinking": False},
            )
            if chat_out is None:
                continue
            compressed = ((chat_out.get("message") or {}).get("content") or "").strip()
            if not compressed or len(compressed) >= body_len:
                continue
            if not dry_run:
                mem.update(mid, content=compressed)
            results.append({"id": mid, "original_len": body_len, "compressed_len": len(compressed)})
        except Exception as exc:
            _log.warning("compress: failed for %s: %s", mid, exc)
    return results


def _run_prewarm_queries(cfg: Any, mem: Memory, n: int) -> dict:
    """Pre-embed the n most recent unique queries from recall.log.

    Warms the LRU embed cache so the next recall-hook invocation hits cached
    embeddings instead of recomputing them. Never raises.
    """
    try:
        from memo.dashboard_logs import read_recall_log

        entries = read_recall_log(cfg.state_dir, limit=n * 3)
        seen: list[str] = []
        for e in entries:
            q = (e.get("prompt") or "").strip()
            if q and q not in seen:
                seen.append(q)
            if len(seen) >= n:
                break

        warmed = 0
        for q in seen:
            try:
                mem.embedder.embed_query(q)
                warmed += 1
            except Exception:  # noqa: S110
                pass
        return {"queries_warmed": warmed, "queries_available": len(seen)}
    except Exception as exc:
        return {"queries_warmed": 0, "queries_available": 0, "error": str(exc)}


def _run_presynthesis(cfg: Any, mem: Memory, top_n: int, dry_run: bool) -> list[dict]:
    """Pre-synthesize clusters for the top recurring queries.

    Reads recall.log, picks the top_n most frequent queries, runs a focused
    synthesis pass on the memories each query surfaces. Returns a list of
    synthesis results per query.
    """
    try:
        from collections import Counter

        from memo.dashboard_logs import read_recall_log

        entries = read_recall_log(cfg.state_dir, limit=200)
        counts: Counter = Counter()
        for e in entries:
            q = (e.get("prompt") or "").strip()
            if q:
                counts[q] += 1

        top_queries = [q for q, _ in counts.most_common(top_n)]
        if not top_queries:
            return []

        all_results = []
        for query in top_queries:
            try:
                hits = mem.search(query, limit=20, disable_reranker=True)
                if len(hits) < 3:
                    continue
                # NOTE: synthesize_cross_cluster takes no source_ids/cluster param —
                # synthesis runs GLOBALLY over all clusters, not scoped to these hits.
                result = mem.synthesize_cross_cluster(
                    dry_run=dry_run, min_cluster_size=3, max_clusters=1
                )
                if result:
                    all_results.append(
                        {
                            "query": query[:80],
                            "hits": len(hits),
                            "synthesized": len(result),
                        }
                    )
            except Exception as exc:
                _log.warning("presynthesis: failed for query %r: %s", query[:50], exc)
        return all_results
    except Exception as exc:
        return [{"error": str(exc)}]


def _iso_now() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat(timespec="seconds")


def _harvested_labels_path(cfg: Config) -> Path:
    return cfg.state_dir / "eval" / "harvested_labels.json"


def _run_harvest_labels(cfg: Config) -> dict:
    """Mine ground-truth recall labels from grounding.log and merge them into
    ``state_dir/eval/harvested_labels.json``.

    Dedup is by prompt (token-Jaccard, via ``merge_label_prompts``): a
    re-harvested prompt unions its ``expect_ids`` into the existing entry
    instead of duplicating it. New entries are stamped ``harvested_ts`` so the
    eval pass can cap to the most recent N. Returns ``{"new", "total"}``;
    raises on failure (the cli_dream caller records it in receipt["errors"]).
    """
    from memo.eval_recall import LABELS_SCHEMA, harvest_labels, merge_label_prompts

    path = _harvested_labels_path(cfg)
    existing: list[dict] = []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict) and isinstance(raw.get("prompts"), list):
            existing = [p for p in raw["prompts"] if isinstance(p, dict) and p.get("text")]
    except (OSError, json.JSONDecodeError):
        existing = []

    harvested = harvest_labels(cfg.state_dir)
    merged = merge_label_prompts(existing, harvested)
    now = _iso_now()
    for p in merged:
        p.setdefault("harvested_ts", now)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"schema": LABELS_SCHEMA, "prompts": merged}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {"new": len(merged) - len(existing), "total": len(merged)}


def _run_eval_recall(cfg: Config, mem: Memory, *, k: int = 5, max_labels: int = 200) -> dict:
    """Nightly retrieval-only eval (vec mode, no reranker — the same fast path
    ``memo eval recall`` uses) over curated + harvested labels.

    Curated labels (``regression_labels.json``: state_dir first, repo checkout
    second) are always included; harvested labels are cross-deduped against
    them (token-Jaccard >= 0.6, curated absorbs the duplicate's expect_ids) and
    the most recent survivors fill the remaining room up to ``max_labels``.
    Appends one trend line to
    ``state_dir/eval/history.jsonl`` (skipped when there are no labels, so an
    empty run never pollutes the trend) and returns the receipt fragment.
    Raises on failure (the cli_dream caller records it in receipt["errors"]).
    """
    from memo.eval_recall import Cfg as EvalCfg
    from memo.eval_recall import (
        LabelSet,
        Prompt,
        evaluate,
        gate_metrics,
        load_labels,
        merge_label_prompts,
    )
    from memo.flags import flag_float

    curated = None
    for cp in (
        cfg.state_dir / "eval" / "regression_labels.json",
        Path(__file__).resolve().parent.parent.parent / "eval" / "regression_labels.json",
    ):
        try:
            curated = load_labels(cp)
            break
        except ValueError:
            continue

    harvested_raw: list[dict] = []
    try:
        raw = json.loads(_harvested_labels_path(cfg).read_text(encoding="utf-8"))
        if isinstance(raw, dict) and isinstance(raw.get("prompts"), list):
            harvested_raw = [p for p in raw["prompts"] if isinstance(p, dict) and p.get("text")]
    except (OSError, json.JSONDecodeError):
        harvested_raw = []
    harvested_raw.sort(key=lambda p: str(p.get("harvested_ts") or ""), reverse=True)

    curated_used = list(curated.prompts)[:max_labels] if curated else []

    # Cross-dedup harvested AGAINST curated (curated wins/absorbs): a harvested
    # prompt token-Jaccard-similar (>=0.6) to a curated one would otherwise be
    # counted twice in prec@K/noise@K. ``merge_label_prompts`` unions its
    # expect_ids into the curated entry and drops the duplicate; only the
    # surviving (genuinely new) harvested prompts fill the remaining room, so
    # the receipt's ``harvested`` count is post-dedup.
    curated_dicts = [
        {
            "text": p.text,
            "relevant": p.relevant,
            "expect_ids": [str(x) for x in p.expect_ids],
            "expect_associative_ids": list(p.expect_associative_ids),
        }
        for p in curated_used
    ]
    harvested_dicts = [
        {
            "text": str(p["text"]),
            "relevant": bool(p.get("relevant", False)),
            "expect_ids": [str(x) for x in (p.get("expect_ids") or [])],
        }
        for p in harvested_raw
    ]
    merged = merge_label_prompts(curated_dicts, harvested_dicts)
    survivors = merged[len(curated_dicts) :]
    deduped_out = len(harvested_dicts) - len(survivors)

    curated_prompts = [
        Prompt(
            text=str(d["text"]),
            relevant=bool(d.get("relevant", False)),
            expect_ids=[str(x) for x in (d.get("expect_ids") or [])],
            expect_associative_ids=tuple(str(x) for x in (d.get("expect_associative_ids") or ())),
        )
        for d in merged[: len(curated_dicts)]
    ]
    room = max(0, max_labels - len(curated_prompts))
    harvested_used = [
        Prompt(
            text=str(p["text"]),
            relevant=bool(p.get("relevant", False)),
            expect_ids=[str(x) for x in (p.get("expect_ids") or [])],
        )
        for p in survivors[:room]
    ]
    prompts = curated_prompts + harvested_used
    fragment = {
        "prec_at_k": 0.0,
        "noise_at_k": 0.0,
        "k": k,
        "labels_total": len(prompts),
        "harvested": len(harvested_used),
        "harvested_deduped_out": deduped_out,
        "curated": len(curated_prompts),
    }
    if not prompts:
        return fragment

    labels = LabelSet(
        prompts=prompts,
        relevant_terms=set(curated.relevant_terms) if curated else set(),
        noise_tags=set(curated.noise_tags) if curated else set(),
        noise_path_fragments=tuple(curated.noise_path_fragments) if curated else (),
    )
    floor = flag_float("MEMO_RECALL_MIN_SIM")
    floor = 0.5 if floor is None else floor
    rows = evaluate(
        mem,
        k=k,
        labels=labels,
        configs=[EvalCfg(name="vec", mode="vec", floor=floor, exclude_archived=True)],
    )
    metrics = gate_metrics(rows)
    fragment["prec_at_k"] = metrics["precision_at_k"]
    fragment["noise_at_k"] = metrics["noise_at_k"]

    hist = cfg.state_dir / "eval" / "history.jsonl"
    hist.parent.mkdir(parents=True, exist_ok=True)
    line = {
        "ts": _iso_now(),
        "prec_at_k": fragment["prec_at_k"],
        "noise_at_k": fragment["noise_at_k"],
        "k": k,
        "labels": len(prompts),
        "source": "dream",
    }
    with hist.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(line, ensure_ascii=False) + "\n")
    return fragment


def _run_capture_weights(cfg: Config, mem: Memory) -> dict:
    """Nightly citation-type feedback: join grounding.log citations to memory
    types and refresh ``state_dir/capture/type_weights.json`` (consumed at
    capture time when MEMO_CAPTURE_TYPE_FEEDBACK is on — see
    ``memo.capture_weights``). Returns the receipt fragment
    ``{"types", "top"}``; raises on failure (the cli_dream caller records it
    in receipt["errors"])."""
    from memo.capture_weights import compute_type_citation_stats

    payload = compute_type_citation_stats(cfg, mem)
    weights: dict[str, float] = payload.get("weights") or {}
    top = None
    if weights:
        t, w = max(weights.items(), key=lambda kv: kv[1])
        top = f"{t}:{w:g}"
    return {"types": len(weights), "top": top}


def _state_path(cfg: Config):
    return cfg.state_dir / "dream"


def _older_id(mem: Any, id_a: str, id_b: str) -> tuple[str, str]:
    ra, rb = mem.get(id_a), mem.get(id_b)
    ua = getattr(ra, "updated", "") or ""
    ub = getattr(rb, "updated", "") or ""
    if ua and ub:
        return (id_a, id_b) if ua <= ub else (id_b, id_a)
    return id_a, id_b


def _corpus_fingerprint(mem: Memory) -> str | None:
    """A cheap change-signal: (row count, latest update timestamp) of the
    canonical `meta` table. Any save/edit/delete moves at least one."""
    try:
        row = mem.store._conn.execute(
            "SELECT COUNT(*), COALESCE(MAX(updated), '') FROM meta"
        ).fetchone()
        return f"{row[0]}:{row[1]}"
    except Exception:
        return None


def _make_progress() -> Progress:
    import sys

    from memo.flags import flag_bool

    # Non-interactive runs (launchd dream, piped output) still get the live-render
    # ANSI control stream from Rich — ~2MB of escapes per run. Disable the bar
    # there; the per-pass `console.print` summary at the end still emits.
    disable = flag_bool("MEMO_NONINTERACTIVE") or not sys.stderr.isatty()
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=24),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=False,
        disable=disable,
    )


def _render_run_summary(receipt: dict[str, Any], dry_run: bool) -> None:
    """Human-readable summary of a dream run (the non-JSON output path)."""
    tag = "[dim](dry-run)[/dim] " if dry_run else ""
    console.print(f"{tag}[bold]memo dream[/bold]")
    console.print(
        f"  contradictions superseded: {len(receipt['superseded'])}, "
        f"evolutions: {len(receipt['evolved'])}, "
        f"confidence penalized: {receipt['confidence_penalized']}"
    )
    console.print(f"  duplicate clusters merged: {len(receipt['merged'])}")
    console.print(f"  stale memories archived:   {len(receipt['archived_stale'])}")
    if receipt["synthesized"]:
        saved = sum(1 for s in receipt["synthesized"] if s.get("saved"))
        console.print(
            f"  emergent syntheses:        {saved} saved, {len(receipt['synthesized'])} proposed"
        )
    console.print(f"  entities extracted:        {receipt['entities_extracted']}")
    console.print(
        f"  roi reconciled (grounding):{receipt['roi_reconciled']} rescored, "
        f"{len(receipt['dead_archived'])} dead-archived"
    )
    console.print(f"  roi rows decayed:          {receipt['roi_decayed']}")
    console.print(f"  quality-floor pruned:      {len(receipt['pruned_floor'])}")
    if receipt.get("evicted"):
        console.print(f"  evicted (LFU):             {len(receipt['evicted'])}")
    if receipt.get("compressed"):
        console.print(f"  compressed:                {len(receipt['compressed'])}")
    pw = receipt.get("prewarm", {})
    if pw.get("queries_warmed"):
        console.print(f"  cache pre-warmed:          {pw['queries_warmed']} queries")
    if receipt.get("presynthesis"):
        console.print(f"  pre-syntheses:             {len(receipt['presynthesis'])} clusters")
    sg = receipt.get("signal_gathered", {})
    if sg.get("files_processed") or sg.get("memories_saved"):
        console.print(
            f"  signal gather:             {sg['files_processed']} files, "
            f"{sg['memories_saved']} saved, {sg.get('skipped_dup', 0)} dup skipped"
        )
    if receipt["errors"]:
        for e in receipt["errors"]:
            console.print(f"  [yellow]warn:[/yellow] {e}")
