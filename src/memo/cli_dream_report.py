"""Read-only dream reporting commands — `memo dream status/ledger/...`.

Pure reporting over the pipeline's persisted receipts and derived state:
no phase execution. Split from `cli_dream.py` so the 2.5k-line dream module
keeps pipeline-orchestration concerns only; this module is stateless and
imports no `DreamCheckpoint`/`PhaseRecorder` machinery.

Each command is a `@click.command` — registration is implicit on import.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, cast

import click
from rich.markup import escape

from memo.cli_common import console
from memo.cli_common import get_memory as _get_memory
from memo.config import Config
from memo.dream_utils import _state_path
from memo.flags import flag_bool, flag_float, flag_int

_log = logging.getLogger(__name__)


def _render_graph_projection_status(data: dict[str, Any]) -> None:
    projection = data.get("graph_projection")
    if projection:
        console.print(f"  graph:      {projection.get('status')}")


@click.command(name="status")
def dream_status() -> None:
    """Show when dream last ran and what it changed."""
    cfg = Config.from_env()
    last_json = _state_path(cfg) / "last.json"
    if not last_json.exists():
        console.print("[dim]dream has never run[/dim]")
        return
    try:
        data = json.loads(last_json.read_text(encoding="utf-8"))
    except Exception as exc:
        console.print(f"[yellow]could not read last receipt: {exc}[/yellow]")
        return
    ts = data.get("ts")
    import datetime

    when = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M") if ts else "?"
    console.print(f"[bold]last dream run:[/bold] {when}")
    console.print(f"  superseded: {len(data.get('superseded', []))}")
    console.print(f"  merged:     {len(data.get('merged', []))}")
    console.print(f"  stale:      {len(data.get('archived_stale', []))}")
    console.print(f"  syntheses:  {len(data.get('synthesized', []))}")
    console.print(f"  entities:   {data.get('entities_extracted', 0)}")
    _render_graph_projection_status(data)
    console.print(f"  roi decay:  {data.get('roi_decayed', 0)} rows")
    if data.get("tuner"):
        t = data["tuner"]
        extra = (
            f" (min_sim {t.get('floor_before')}→{t.get('floor_after')})"
            if t.get("status") == "applied"
            else ""
        )
        console.print(f"  tuner:      {t.get('status')}{extra}")
    if data.get("tuner", {}).get("online", {}).get("status") == "waiting":
        o = data["tuner"]["online"]
        console.print(f"  tuner online: waiting (cohort {o.get('n_after')}/{o.get('min_cohort')})")
    from memo.dream_tune_online import read_ledger

    _proof = read_ledger(cfg.state_dir, limit=3)
    if _proof:
        console.print("  [bold]proof loop[/bold] (realized online impact):")
        for e in _proof:
            mark = (
                "[green]✓ confirmed[/green]"
                if e.get("verdict") == "confirmed"
                else "[red]✗ reverted[/red]"
            )
            _d = e.get("realized_delta")
            _ds = f"{_d:+g}" if isinstance(_d, (int, float)) else "—"
            _knob = (e.get("knob") or "MEMO_RECALL_MIN_SIM").replace("MEMO_RECALL_", "").lower()
            console.print(
                f"    {mark}  {_knob} {e.get('floor_before')}→{e.get('floor_after')}  "
                f"online {e.get('online_before')}→{e.get('online_after')} "
                f"(Δ{_ds}) n={e.get('n_after')}"
            )

    _gk = 5 if (_gk_flag := flag_int("MEMO_DREAM_TUNE_GRADUATION_K")) is None else _gk_flag
    if read_ledger(cfg.state_dir, limit=max(_gk * 4, 20)):
        from memo.dream_tune_online import graduation_status

        _gs = graduation_status(cfg.state_dir, k=_gk)
        _verdict = (
            "[green]✓ ready to enable by default[/green]" if _gs["graduated"] else "accumulating"
        )
        console.print(f"  graduation: {_gs['streak']}/{_gk} confirmed nights — {_verdict}")
    if data.get("anticipated"):
        from memo.dream_anticipate import briefing_line

        console.print(f"  {briefing_line(data['anticipated'])}")
    if data.get("consolidated_episodes"):
        ce = data["consolidated_episodes"]
        saved = sum(1 for d in ce.get("consolidated", []) if d.get("status") == "saved")
        console.print(f"  consolidate: {ce.get('status')} — {saved} cross-session memo(s)")
    if data.get("harvest_labels"):
        hl = data["harvest_labels"]
        console.print(f"  labels harvested: +{hl.get('new', 0)} (total {hl.get('total', 0)})")
    if data.get("eval_recall"):
        ev = data["eval_recall"]
        console.print(
            f"  eval recall: prec@{ev.get('k')} {ev.get('prec_at_k')} · "
            f"noise@{ev.get('k')} {ev.get('noise_at_k')} · {ev.get('labels_total')} labels "
            f"({ev.get('harvested')} harvested + {ev.get('curated')} curated)"
        )
    if data.get("capture_weights"):
        cw = data["capture_weights"]
        _cw_top = f" · top {cw.get('top')}" if cw.get("top") else ""
        console.print(f"  capture weights: {cw.get('types', 0)} type(s){_cw_top}")
    if data.get("distilled"):
        _di = data["distilled"]
        _n = sum(1 for d in _di.get("distilled", []) if d.get("status") == "saved")
        console.print(f"  distill: {_di.get('status')} ({_n} saved)")
    if data.get("errors"):
        for e in data["errors"]:
            console.print(f"  [yellow]warn:[/yellow] {e}")


@click.option("--limit", type=int, default=30, help="Most recent ledger entries to show.")
@click.option(
    "--open", "open_only", is_flag=True, help="Show only still-open actions (no outcome yet)."
)
@click.option(
    "--json", "as_json", is_flag=True, help="Emit the ledger entries/summary as raw JSON."
)
@click.command(name="ledger")
def dream_ledger_cmd(limit: int, open_only: bool, as_json: bool) -> None:
    """Auditable learning ledger (Fase 3): the chain of dream mutations
    (supersede/merge/archive) and their later reinforced/rollback outcomes.

    Populated only when ``MEMO_DREAM_LEDGER_ENABLED`` is on for the nightly run.
    """
    from memo import dream_ledger

    cfg = Config.from_env()
    summary = dream_ledger.summarize(cfg.state_dir)
    rows = (
        dream_ledger.open_actions(cfg.state_dir, limit=limit)
        if open_only
        else dream_ledger.read_ledger(cfg.state_dir, limit=limit)
    )
    if as_json:
        click.echo(json.dumps({"summary": summary, "entries": rows}, indent=2, ensure_ascii=False))
        return
    console.print(
        f"[bold]dream ledger:[/bold] {summary['actions']} actions · {summary['outcomes']} outcomes "
        f"· {summary['open']} open · {summary['rollback_candidates']} rollback-candidates"
    )
    if summary["by_action"]:
        parts = ", ".join(f"{k}={v}" for k, v in sorted(summary["by_action"].items()))
        console.print(f"  by action: {parts}")
    if not rows:
        console.print(
            "  [dim](no entries — enable MEMO_DREAM_LEDGER_ENABLED for the nightly run)[/dim]"
        )
        return
    for r in rows:
        if r.get("kind") == "outcome":
            console.print(
                f"  [dim]{r.get('ts')}[/dim] outcome→{r.get('outcome')} "
                f"({r.get('verdict')}) for {str(r.get('action_id'))[:8]}"
            )
        else:
            aff = ",".join(str(a)[:8] for a in (r.get("affected_ids") or [])) or "-"
            pass_name = escape(f"[{r.get('pass_name')}]")
            console.print(
                f"  [dim]{r.get('ts')}[/dim] {r.get('action')} {pass_name} "
                f"→ {aff}  ({str(r.get('entry_id'))[:8]})"
            )


@click.option("--repair", is_flag=True, help="Delete derived orphans (never touches .md).")
@click.option("--json", "as_json", is_flag=True, help="Emit the full report as JSON.")
@click.command(name="index-health")
def dream_index_health_cmd(repair: bool, as_json: bool) -> None:
    """Fase 6: derived-index health check — detect (and optionally repair)
    divergence between the Markdown source of truth and the sqlite index.
    Repair only ever removes derived orphans; canonical .md files are untouched.
    """
    from memo.store.index_health import check_index_health

    cfg = Config.from_env()
    mem = _get_memory(cfg)
    res = check_index_health(cfg, mem, repair=repair)
    if as_json:
        click.echo(json.dumps(res, indent=2, ensure_ascii=False))
        return
    console.print(f"[bold]index health:[/bold] {res.get('status')}")
    for name, v in (res.get("checks") or {}).items():
        count = v.get("count", 0)
        style = "green" if count == 0 else "yellow"
        console.print(f"  {name}: [{style}]{count}[/{style}]")
    if res.get("repaired"):
        console.print(f"  [cyan]repaired:[/cyan] {res['repaired']}")
    for e in res.get("errors") or []:
        console.print(f"  [red]error:[/red] {e}")


@click.option(
    "--resume",
    "do_resume",
    is_flag=True,
    help="Re-apply parked proposals whose conflicts are resolved.",
)
@click.option("--drop", "drop_id", default=None, help="Drop a staged proposal by id.")
@click.option("--json", "as_json", is_flag=True, help="Emit as raw JSON.")
@click.command(name="staging")
def dream_staging_cmd(do_resume: bool, drop_id: str | None, as_json: bool) -> None:
    """Fase 7: dream conflict-staging — dream-minted memories parked by a write
    conflict, awaiting human conflict resolution (MEMO_DREAM_STAGING_ENABLED).
    """
    from memo import dream_staging

    cfg = Config.from_env()
    if drop_id:
        ok = dream_staging.drop_staged(cfg, drop_id)
        click.echo(
            json.dumps({"dropped": ok})
            if as_json
            else (f"dropped {drop_id}" if ok else "not found")
        )
        return
    if do_resume:
        mem = _get_memory(cfg)
        res = dream_staging.resume_staged_proposals(cfg, mem)
        click.echo(json.dumps(res, indent=2) if as_json else f"resumed: {res}")
        return
    staged = dream_staging.list_staged(cfg)
    if as_json:
        click.echo(json.dumps([p.to_dict() for p in staged], indent=2, ensure_ascii=False))
        return
    if not staged:
        console.print(
            "[dim]no staged proposals (enable MEMO_DREAM_STAGING_ENABLED for the nightly run)[/dim]"
        )
        return
    console.print(f"[bold]dream staging:[/bold] {len(staged)} parked proposal(s)")
    for p in staged:
        console.print(f"  [bold]{p.proposal_id}[/bold] ({p.kind}) attempts={p.attempts}")
        console.print(f"    conflict: {p.conflict_summary or '-'}")
        console.print(f"    resolve:  {dream_staging.resolve_command(p)}")


@click.option(
    "--status", "show_status", is_flag=True, help="Per shadow-flag review rollup (default view)."
)
@click.option("--promote", "promote_flag", default=None, help="Promote a review-ready shadow flag.")
@click.option(
    "--reject", "reject_flag", default=None, help="Reject a shadowed flag (needs --reason)."
)
@click.option("--reason", default="", help="Reason for --reject.")
@click.option(
    "--apply", "do_apply", is_flag=True, help="With --promote, persist the config change."
)
@click.option("--force-latency", is_flag=True, help="With --promote, override the latency ceiling.")
@click.option("--json", "as_json", is_flag=True, help="Emit as raw JSON.")
@click.command(name="shadow")
def dream_shadow_cmd(
    show_status: bool,
    promote_flag: str | None,
    reject_flag: str | None,
    reason: str,
    do_apply: bool,
    force_latency: bool,
    as_json: bool,
) -> None:
    """Fase 8: shadow mode — measure-only evidence for opt-in phases without
    mutating production; a human promotes a flag only after enough consecutive
    clean nights (MEMO_DREAM_SHADOW_ENABLED).
    """
    from memo import dream_shadow
    from memo.dream_flags import GATES

    cfg = Config.from_env()
    if reject_flag:
        ok = dream_shadow.reject(cfg, reject_flag, reason or "manual")
        click.echo(json.dumps({"rejected": ok}) if as_json else f"rejected {reject_flag}: {ok}")
        return
    if promote_flag:
        res = dream_shadow.promote(cfg, promote_flag, force_latency=force_latency, apply=do_apply)
        click.echo(json.dumps(res, indent=2, ensure_ascii=False) if as_json else str(res))
        return
    rows = dream_shadow.review_rows(cfg.state_dir, GATES)
    if as_json:
        click.echo(json.dumps(rows, indent=2, ensure_ascii=False))
        return
    if not rows:
        console.print(
            "[dim]no shadow-kind flags declared (classify a gate kind='shadow' to populate)[/dim]"
        )
        return
    console.print(f"[bold]dream shadow:[/bold] {len(rows)} shadow flag(s)")
    for r in rows:
        ready = "[green]review-ready[/green]" if r["review_ready"] else "[dim]accruing[/dim]"
        console.print(
            f"  [bold]{r['flag']}[/bold] {ready}  streak {r['streak']}/{r['review_nights']} "
            f"· meanΔ {r['mean_delta']} · cost_p50 {r['cost_p50']}ms · {r['last_verdict']}"
        )


@click.option(
    "--limit", type=int, default=20, help="Most recent proof-loop entries to show (default: 20)."
)
@click.option("--json", "as_json", is_flag=True, help="Emit the ledger entries as raw JSON.")
@click.command(name="timeline")
def dream_timeline(limit: int, as_json: bool) -> None:
    """Proof-loop timeline: how the recall self-tuner changed over time and
    whether each change actually improved real grounding (realized online impact).
    """
    from memo.dream_tune_online import graduation_streak, read_ledger

    cfg = Config.from_env()
    entries = read_ledger(cfg.state_dir, limit=limit)
    if as_json:
        click.echo(json.dumps(entries, ensure_ascii=False, indent=2))
        return
    if not entries:
        console.print(
            "[dim]no proof-loop history yet (the recall self-tuner has applied no change)[/dim]"
        )
        return

    console.print("[bold]recall self-tuner — proof-loop timeline[/bold]")
    for e in entries:
        verdict = e.get("verdict") or ""
        mark = {
            "confirmed": "[green]✓ kept[/green]",
            "reverted": "[red]✗ reverted[/red]",
            "expired": "[yellow]~ expired[/yellow]",
        }.get(verdict, verdict)
        ts = (e.get("resolved_ts") or "")[:16]
        delta = e.get("realized_delta")
        delta_s = f"{delta:+g}" if isinstance(delta, (int, float)) else "—"
        _knob = (e.get("knob") or "MEMO_RECALL_MIN_SIM").replace("MEMO_RECALL_", "").lower()
        console.print(
            f"  {ts}  {mark}  {_knob} {e.get('floor_before')}→{e.get('floor_after')}  "
            f"online {e.get('online_before')}→{e.get('online_after')} (Δ{delta_s}) n={e.get('n_after')}"
        )

    confirmed = sum(1 for e in entries if e.get("verdict") == "confirmed")
    reverted = sum(1 for e in entries if e.get("verdict") == "reverted")
    expired = sum(1 for e in entries if e.get("verdict") == "expired")
    net = sum(
        e["realized_delta"]
        for e in entries
        if e.get("verdict") == "confirmed" and isinstance(e.get("realized_delta"), (int, float))
    )
    console.print(
        f"  [dim]—[/dim] {confirmed} kept · {reverted} reverted · {expired} expired · "
        f"net realized Δ{net:+g} · streak {graduation_streak(entries)}"
    )


@click.option(
    "--json", "as_json", is_flag=True, help="Emit the anticipated-needs fragment as JSON."
)
@click.command(name="anticipate")
def dream_anticipate_cmd(as_json: bool) -> None:
    """Anticipatory pass — surface recurring unmet gaps + hot queries (no fabrication)."""
    from memo import dream_anticipate

    cfg = Config.from_env()
    mem = _get_memory(cfg)
    frag = dream_anticipate.anticipate(
        cfg, mem, top_gaps=5 if (_tg := flag_int("MEMO_DREAM_ANTICIPATE_TOP_GAPS")) is None else _tg
    )
    if as_json:
        click.echo(json.dumps(frag, indent=2, ensure_ascii=False))
        return
    console.print(f"[bold]{dream_anticipate.briefing_line(frag)}[/bold]")
    for g in frag.get("gaps", []):
        console.print(f"  gap (x{g['count']}): {g['prompt']}")
    if frag.get("prewarmed"):
        console.print(f"  prewarmed {frag['prewarmed']} queries")


@click.option("--dry-run", is_flag=True, help="Cluster + synthesize, save nothing.")
@click.option("--json", "as_json", is_flag=True, help="Emit the consolidation fragment as JSON.")
@click.command(name="consolidate-episodes")
def dream_consolidate_cmd(dry_run: bool, as_json: bool) -> None:
    """Episodic→semantic — abstract recurring cross-session work into durable memos."""
    from memo import dream_consolidate

    cfg = Config.from_env()
    mem = _get_memory(cfg)
    res = dream_consolidate.run_consolidate_episodes(
        cfg,
        mem,
        min_sessions=2 if (_ms := flag_int("MEMO_DREAM_CONSOLIDATE_MIN_SESSIONS")) is None else _ms,
        dry_run=dry_run,
    )
    if as_json:
        click.echo(json.dumps(res, indent=2, ensure_ascii=False))
        return
    console.print(f"[bold]consolidate-episodes:[/bold] {res.get('status')}")
    for d in res.get("consolidated", []):
        console.print(f"  [{d['status']}] {d.get('project')}: {d.get('title', '')}", markup=False)


@click.option(
    "--day", "day", default=None, help="Day to chronicle (YYYY-MM-DD, default: last finished day)."
)
@click.option("--dry-run", is_flag=True, help="Compute + narrate, don't write.")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON.")
@click.command(name="chronicle")
def dream_chronicle_cmd(day: str | None, dry_run: bool, as_json: bool) -> None:
    """Write the engineering diary for one day (see MEMO_DREAM_CHRONICLE_ENABLED)."""
    from memo import dream_chronicle

    cfg = Config.from_env()
    mem = _get_memory(cfg)
    res = dream_chronicle.run_chronicle_pass(
        cfg, mem, day=day, weekly=flag_bool("MEMO_CHRONICLE_WEEKLY"), dry_run=dry_run
    )
    if as_json:
        click.echo(json.dumps(res, indent=2, ensure_ascii=False))
        return
    console.print(f"[bold]chronicle:[/bold] {res.get('status')} {res.get('path', '')}")


@click.option("--dry-run", is_flag=True, help="Compute the backlog, generate nothing.")
@click.option(
    "--reembed",
    is_flag=True,
    help=(
        "Re-embed stored HyPE questions into the currently active variant "
        "(see MEMO_HYPE_EMBED_RAW) instead of running the generation pass."
    ),
)
@click.option("--json", "as_json", is_flag=True, help="Emit the pass receipt as JSON.")
@click.command(name="hype")
def dream_hype_cmd(dry_run: bool, reembed: bool, as_json: bool) -> None:
    """Nightly HyPE pass — generate + index hypothetical questions per memory (see MEMO_DREAM_HYPE_ENABLED)."""
    if dry_run and reembed:
        raise click.UsageError("--dry-run cannot be combined with --reembed")

    from memo import dream_hype

    cfg = Config.from_env()
    mem = _get_memory(cfg)
    if reembed:
        res = dream_hype.run_hype_reembed(cfg, mem)
        if as_json:
            click.echo(json.dumps(res, indent=2, ensure_ascii=False))
            return
        console.print(f"[bold]hype reembed:[/bold] {res.get('status')}")
        console.print(
            f"  reembedded: {res.get('reembedded', 0)} · skipped: {res.get('skipped', 0)}"
        )
        return
    res = dream_hype.run_hype_pass(
        cfg,
        mem,
        questions_per_memory=3
        if (_qpm := flag_int("MEMO_HYPE_QUESTIONS_PER_MEMORY")) is None
        else _qpm,
        night_cap=400 if (_nc := flag_int("MEMO_HYPE_NIGHT_CAP")) is None else _nc,
        budget_s=flag_float("MEMO_DREAM_HYPE_BUDGET_S") or None,
        dry_run=dry_run,
    )
    if as_json:
        click.echo(json.dumps(res, indent=2, ensure_ascii=False))
        return
    console.print(f"[bold]hype:[/bold] {res.get('status')}")
    console.print(f"  generated: {res.get('generated', 0)} · memories: {res.get('memories', 0)}")


@click.option("--dry-run", is_flag=True, help="Detect + preview communities, save nothing.")
@click.option("--json", "as_json", is_flag=True, help="Emit the synthesis fragment as JSON.")
@click.command(name="communities")
def dream_communities_cmd(dry_run: bool, as_json: bool) -> None:
    """Graph→semantic — abstract each entity-graph community into a synthesis memo."""
    from memo import dream_communities

    cfg = Config.from_env()
    mem = _get_memory(cfg)
    res = dream_communities.run_synthesize_communities(
        cfg,
        mem,
        min_size=4 if (_msz := flag_int("MEMO_DREAM_COMMUNITIES_MIN_SIZE")) is None else _msz,
        dry_run=dry_run,
    )
    if as_json:
        click.echo(json.dumps(res, indent=2, ensure_ascii=False))
        return
    console.print(f"[bold]communities:[/bold] {res.get('status')}")
    for d in res.get("synthesized", []):
        rep = d.get("representative") or ""
        console.print(f"  [{d['status']}] {rep}: {d.get('title', '')}", markup=False)


@click.option("--dry-run", is_flag=True, help="Cluster + gate + preview, save nothing.")
@click.option("--json", "as_json", is_flag=True, help="Emit the distillation fragment as JSON.")
@click.command(name="distill")
def dream_distill_cmd(dry_run: bool, as_json: bool) -> None:
    """Upward re-abstraction — distill each mature durable cluster into a principle."""
    from memo import dream_distill

    cfg = Config.from_env()
    mem = _get_memory(cfg)
    res = dream_distill.run_distill(
        cfg,
        mem,
        min_cluster=cast(int, flag_int("MEMO_DREAM_DISTILL_MIN_CLUSTER")),
        min_support=cast(int, flag_int("MEMO_DREAM_DISTILL_MIN_SUPPORT")),
        min_age_days=cast(int, flag_int("MEMO_DREAM_DISTILL_MIN_AGE_DAYS")),
        max_clusters=cast(int, flag_int("MEMO_DREAM_DISTILL_MAX")),
        threshold=cast(float, flag_float("MEMO_DREAM_DISTILL_THRESHOLD")),
        min_confidence=cast(float, flag_float("MEMO_DREAM_DISTILL_MIN_CONFIDENCE")),
        dry_run=dry_run,
    )
    if as_json:
        click.echo(json.dumps(res, indent=2, ensure_ascii=False))
        return
    console.print(f"[bold]distill:[/bold] {res.get('status')}")
    for d in res.get("distilled", []):
        console.print(f"  [{d['status']}] {d.get('title', '')}", markup=False)


@click.option("--dry-run", is_flag=True, help="Propose merges, change nothing.")
@click.option("--json", "as_json", is_flag=True, help="Emit the pass receipt as JSON.")
@click.command(name="entity-canon")
def dream_entity_canon_cmd(dry_run: bool, as_json: bool) -> None:
    """Graph hygiene — MinHash+LSH-blocked LLM entity merge (measures calls saved)."""
    from memo import dream_entity_canon

    cfg = Config.from_env()
    mem = _get_memory(cfg)
    res = dream_entity_canon.run_entity_canon(
        cfg,
        mem,
        max_pairs=30 if (_mp := flag_int("MEMO_DREAM_ENTITY_CANON_MAX_PAIRS")) is None else _mp,
        dry_run=dry_run,
    )
    if as_json:
        click.echo(json.dumps(res, indent=2, ensure_ascii=False))
        return
    console.print(f"[bold]entity-canon:[/bold] {res.get('status')}")
    console.print(
        f"  pairs: naive {res.get('pairs_naive')} → blocked {res.get('pairs_blocked')}"
        f" → LLM calls {res.get('llm_calls')}"
    )
    for m in res.get("merged", []):
        console.print(f"  merged '{m['drop']}' → '{m['keep']}' (est {m['est']:.2f})")


@click.option("--dry-run", is_flag=True, help="Group + preview folders, save nothing.")
@click.option("--json", "as_json", is_flag=True, help="Emit the pass receipt as JSON.")
@click.command(name="folder-abstracts")
def dream_folder_abstracts_cmd(dry_run: bool, as_json: bool) -> None:
    """Reference tier — abstract each vault folder into one synthesis memo."""
    from memo import dream_folder_abstracts

    cfg = Config.from_env()
    mem = _get_memory(cfg)
    res = dream_folder_abstracts.run_folder_abstracts(
        cfg,
        mem,
        min_members=5
        if (_mm := flag_int("MEMO_DREAM_FOLDER_ABSTRACTS_MIN_MEMBERS")) is None
        else _mm,
        max_folders=5 if (_mf := flag_int("MEMO_DREAM_FOLDER_ABSTRACTS_MAX")) is None else _mf,
        dry_run=dry_run,
    )
    if as_json:
        click.echo(json.dumps(res, indent=2, ensure_ascii=False))
        return
    console.print(f"[bold]folder-abstracts:[/bold] {res.get('status')}")
    for a in res.get("abstracts", []):
        status = a["status"]
        console.print(f"  {escape(f'[{status}]')} {a['folder'] or '(root)'}")


@click.option("--dry-run", is_flag=True, help="Decide + preview promotions, write nothing.")
@click.option("--json", "as_json", is_flag=True, help="Emit the retag fragment as JSON.")
@click.command(name="retag")
def dream_retag_cmd(dry_run: bool, as_json: bool) -> None:
    """Scope — retag project memories proven general (cross-project grounding) to global."""
    from memo import dream_retag

    cfg = Config.from_env()
    mem = _get_memory(cfg)
    res = dream_retag.run_retag_global(
        cfg,
        mem,
        min_other_projects=2
        if (_mop := flag_int("MEMO_DREAM_RETAG_MIN_PROJECTS")) is None
        else _mop,
        dry_run=dry_run,
    )
    if as_json:
        click.echo(json.dumps(res, indent=2, ensure_ascii=False))
        return
    console.print(
        f"[bold]retag:[/bold] {res.get('status')} "
        f"({len(res.get('retagged', []))} promoted, {res.get('candidates', 0)} candidates)"
    )
    for d in res.get("retagged", []):
        console.print(
            f"  [{d['status']}] {d['id'][:8]} -{','.join(d['dropped'])} "
            f"← {','.join(d['evidence_projects'])}",
            markup=False,
        )


@click.option("--dry-run", is_flag=True, help="Measure + search, write nothing.")
@click.option("--rollback", "do_rollback", is_flag=True, help="Restore the previous tuned params.")
@click.option("--status", "show_status", is_flag=True, help="Show the overlay + baseline.")
@click.command(name="tune")
def dream_tune_cmd(dry_run: bool, do_rollback: bool, show_status: bool) -> None:
    """Self-improving recall tuner (MEMO_RECALL_MIN_SIM) — gated, reversible."""
    from memo import dream_tune, tuned_overlay

    cfg = Config.from_env()
    if show_status:
        click.echo(
            json.dumps(
                {
                    "overlay": tuned_overlay.read_overlay(cfg.state_dir),
                    "baseline": dream_tune.load_baseline(cfg.state_dir),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return
    if do_rollback:
        restored = tuned_overlay.rollback_overlay(cfg.state_dir)
        click.echo(json.dumps({"rolled_back": restored}, ensure_ascii=False))
        return
    mem = _get_memory(cfg)
    res = dream_tune.run_tuning_pass(
        cfg,
        mem,
        k=5 if (_k := flag_int("MEMO_DREAM_TUNE_K")) is None else _k,
        max_evals=20 if (_me := flag_int("MEMO_DREAM_TUNE_MAX_EVALS")) is None else _me,
        min_used_score=0.5
        if (_mus := flag_float("MEMO_DREAM_MINE_MIN_USED_SCORE")) is None
        else _mus,
        dry_run=dry_run,
    )
    click.echo(json.dumps(res, indent=2, ensure_ascii=False))


@click.option("--dry-run", is_flag=True, help="Measure, write nothing (no state, no overlay).")
@click.option("--status", "show_status", is_flag=True, help="Inventory: every dark flag + verdict.")
@click.option("--json", "as_json", is_flag=True, help="Emit result as JSON.")
@click.command(name="graduate-flags")
def dream_graduate_flags_cmd(dry_run: bool, show_status: bool, as_json: bool) -> None:
    """Dark-feature graduation: A/B-measure default-off *_ENABLED flags and
    flip winners ON via the tuned overlay (reversible); report cull candidates."""
    from memo import dream_flags

    cfg = Config.from_env()
    if show_status:
        rows = dream_flags.status_rows(cfg)
        if as_json:
            click.echo(json.dumps(rows, indent=2, ensure_ascii=False))
            return
        from rich.table import Table

        table = Table(title="dark-flag graduation")
        for col in ("flag", "kind", "status", "streak", "days left", "reason"):
            table.add_column(col)
        for r in rows:
            style = {
                "graduated": "green",
                "human_graduated": "green",
                "cull_candidate": "red",
                "reverted": "yellow",
            }.get(str(r["status"]), "")
            table.add_row(
                r["flag"],
                r["kind"],
                f"[{style}]{r['status']}[/{style}]" if style else str(r["status"]),
                str(r["streak"]),
                "-" if r["days_left"] is None else str(r["days_left"]),
                r["reason"],
            )
        console.print(table)
        return
    mem = _get_memory(cfg)
    res = dream_flags.run_flag_graduation_pass(
        cfg,
        mem,
        k=5 if (_k := flag_int("MEMO_DREAM_TUNE_K")) is None else _k,
        min_used_score=0.5
        if (_mus := flag_float("MEMO_DREAM_MINE_MIN_USED_SCORE")) is None
        else _mus,
        dry_run=dry_run,
    )
    click.echo(json.dumps(res, indent=2, ensure_ascii=False))


@click.option("--json", "as_json", is_flag=True, help="Emit result as JSON.")
@click.command(name="consolidate-reuse")
def dream_consolidate_reuse_cmd(as_json: bool) -> None:
    """F4 metric: of synthesis memories consolidate created, how many are
    actually grounded (reused) in real recall.
    """
    from memo import dream_reuse

    cfg = Config.from_env()
    mem = _get_memory(cfg)
    result = dream_reuse.consolidated_reuse(mem)
    if as_json:
        click.echo(json.dumps(result, indent=2, ensure_ascii=False))
        return
    n = result["n_consolidated"]
    r = result["n_reused"]
    pct = result["reuse_fraction"] * 100
    cs = result["cross_session"]
    suffix = f" · {cs} cross-session" if cs else ""
    console.print(
        f"[bold]consolidate-reuse:[/bold] {n} consolidated · {r} reused ({pct:.0f}%){suffix}"
    )


@click.command(name="if-due")
def dream_if_due() -> None:
    """Spawn a background dream run if > 24h since last run (for launchd)."""
    import os as _os
    import subprocess as _sp

    cfg = Config.from_env()
    # Own debounce file — do NOT reuse .last_run_ts: dream run reads that to
    # size its signal-gather lookback, and clobbering it here collapses the
    # lookback to ~0. This file only guards against double-spawn within 24h.
    ts_file = _state_path(cfg) / ".last_if_due_ts"
    try:
        last = float(ts_file.read_text().strip())
    except Exception:
        last = 0.0
    if (time.time() - last) < 24 * 3600:
        return
    try:
        _state_path(cfg).mkdir(parents=True, exist_ok=True)
        ts_file.write_text(str(time.time()), encoding="utf-8")
        _sp.Popen(
            ["memo", "dream", "run"],
            stdin=_sp.DEVNULL,
            stdout=_sp.DEVNULL,
            stderr=_sp.DEVNULL,
            start_new_session=True,
            env={**_os.environ, "MEMO_NONINTERACTIVE": "1"},
        )
    except Exception as exc:
        _log.warning("dream --if-due: failed to spawn background dream: %s", exc)


def register_dream_report_commands(group: click.Group) -> None:
    """Attach the read-only dream reporting commands to a click group.

    Each function in this module is already its own `@click.command` (with
    the `@click.option` decorators applied inline); this re-uses those
    commands under their historical names so `cli_dream` keeps the surface
    identical without hosting the bodies.
    """
    _ROSTER: tuple[tuple[str, str], ...] = (
        ("dream_status", "status"),
        ("dream_ledger_cmd", "ledger"),
        ("dream_index_health_cmd", "index-health"),
        ("dream_staging_cmd", "staging"),
        ("dream_shadow_cmd", "shadow"),
        ("dream_timeline", "timeline"),
        ("dream_anticipate_cmd", "anticipate"),
        ("dream_consolidate_cmd", "consolidate-episodes"),
        ("dream_chronicle_cmd", "chronicle"),
        ("dream_hype_cmd", "hype"),
        ("dream_communities_cmd", "communities"),
        ("dream_distill_cmd", "distill"),
        ("dream_entity_canon_cmd", "entity-canon"),
        ("dream_folder_abstracts_cmd", "folder-abstracts"),
        ("dream_retag_cmd", "retag"),
        ("dream_tune_cmd", "tune"),
        ("dream_graduate_flags_cmd", "graduate-flags"),
        ("dream_consolidate_reuse_cmd", "consolidate-reuse"),
        ("dream_if_due", "if-due"),
    )
    for fn_name, cmd_name in _ROSTER:
        fn = globals().get(fn_name)
        if fn is not None:
            group.add_command(fn, name=cmd_name)
