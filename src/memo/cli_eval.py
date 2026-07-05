"""`memo eval` command group — recall-quality measurement.

Promotes the old `scripts/recall_eval.py` harness to a first-class verb so
recall tuning is observable instead of a script hack. `memo eval recall` runs
the labeled-prompt eval (see `eval_recall.py`), prints precision@K / noise@K /
p50 latency per config, recommends the winning knobs, and caches results
(keyed by corpus fingerprint + label set + K) so repeat runs are instant.

NOTE: there is deliberately no `memo autoloop`. The autonomous tuning loop in
`scripts/autoloop/run.py` drives the *synapse* chat pipeline (`synapse
eval-chat`, `SYNAPSE_*` knobs, Ollama) — it belongs in synapse, not memo.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import click

from memo import eval_baseline, eval_recall
from memo.cli_common import console
from memo.cli_common import get_memory as _get_memory
from memo.config import Config

_CACHE_TTL_S = 24 * 3600


def _cache_path(cfg: Config) -> Path:
    return cfg.state_dir / "eval" / "recall.json"


def _baseline_path(cfg: Config) -> Path:
    # Machine-local: the gate runs against THIS machine's live index, so the
    # baseline can't be a committed repo file — it lives under state_dir.
    return cfg.state_dir / "eval" / "recall_baseline.json"


def _load_cache(cfg: Config) -> dict:
    p = _cache_path(cfg)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _save_cache(cfg: Config, cache: dict) -> None:
    p = _cache_path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


@click.group(name="eval")
def eval_group() -> None:
    """Measure recall quality against the live corpus."""
    pass


@eval_group.command(name="recall")
@click.option("--k", type=int, default=3, help="Top-K to score (default: 3).")
@click.option(
    "--labels",
    "labels_path",
    type=click.Path(exists=True, dir_okay=False),
    help="JSON label set (schema memo.eval_recall.labels.v1). "
    "Defaults to the built-in example — supply your own corpus "
    "for meaningful numbers.",
)
@click.option("--json", "as_json", is_flag=True, help="Emit raw JSON.")
@click.option("--detail", is_flag=True, help="Print per-prompt top-K.")
@click.option("--force", is_flag=True, help="Ignore cached results and re-run.")
@click.option("--no-cache", is_flag=True, help="Neither read nor write the cache.")
@click.option(
    "--quick",
    is_flag=True,
    help="Fast smoke: run only config A and cap prompts unless --max-prompts is set.",
)
@click.option(
    "--config",
    "config_names",
    multiple=True,
    help="Config to run (A, B, C, D, or full name). Repeat for multiple configs.",
)
@click.option(
    "--max-prompts",
    type=click.IntRange(min=1),
    default=None,
    help="Evaluate only the first N prompts from the label set.",
)
@click.option("--progress", is_flag=True, help="Print progress before each prompt search.")
@click.option(
    "--gate",
    is_flag=True,
    help="Regression gate: exit non-zero if precision@K dropped or noise@K rose "
    "vs the saved baseline. Re-runs fresh (no cache).",
)
@click.option(
    "--update-baseline",
    is_flag=True,
    help="Save the current best precision@K / noise@K as the gate baseline.",
)
def eval_recall_cmd(
    k: int,
    labels_path: str | None,
    as_json: bool,
    detail: bool,
    force: bool,
    no_cache: bool,
    quick: bool,
    config_names: tuple[str, ...],
    max_prompts: int | None,
    progress: bool,
    gate: bool,
    update_baseline: bool,
) -> None:
    """Precision@K / noise@K per retrieval config over labeled prompts.

    Example: memo eval recall --k 3 --labels mylabels.json --json

    Gate (local pre-commit, runs against the live index):
      memo eval recall --labels eval/regression_labels.json --update-baseline
      memo eval recall --labels eval/regression_labels.json --gate
    """
    cfg = Config.from_env()
    # The gate compares fresh numbers — never trust a stale cache for a pass/fail.
    if gate or update_baseline:
        force = True

    if labels_path:
        try:
            labels = eval_recall.load_labels(Path(labels_path))
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc
    else:
        labels = eval_recall.DEFAULT_LABELS
        if not as_json:
            console.print(
                "[dim]Using the built-in example label set; pass "
                "--labels for your own corpus.[/dim]"
            )
    if quick and max_prompts is None:
        max_prompts = 12
    labels = eval_recall.limit_label_set(labels, max_prompts)
    try:
        selected_configs = eval_recall.select_configs(list(config_names), quick=quick)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    mem = _get_memory(cfg)
    corpus_fp = eval_recall.fingerprint_corpus(mem)
    configs_fp = ",".join(c.name for c in selected_configs)
    cache_key = f"{corpus_fp}:{labels.fingerprint()}:{configs_fp}:{k}"

    rows = None
    cached = False
    if not (force or no_cache):
        entry = _load_cache(cfg).get(cache_key)
        if entry and (time.time() - entry.get("ts", 0)) < _CACHE_TTL_S:
            rows = [eval_recall.Row(**r) for r in entry["rows"]]
            cached = True

    if rows is None:
        if quick and not as_json:
            progress = True

        def _progress(cfg_: eval_recall.Cfg, index: int, total: int) -> None:
            if progress and not as_json:
                console.print(f"[dim]eval {cfg_.name}: prompt {index}/{total}[/dim]")

        rows = eval_recall.evaluate(
            mem,
            k=k,
            labels=labels,
            configs=selected_configs,
            progress=_progress if progress and not as_json else None,
        )
        if not no_cache:
            cache = _load_cache(cfg)
            cache[cache_key] = {"ts": time.time(), "k": k, "rows": [r.__dict__ for r in rows]}
            _save_cache(cfg, cache)

    if update_baseline:
        metrics = eval_recall.gate_metrics(rows)
        payload = {**metrics, "k": k, "labels_fingerprint": labels.fingerprint()}
        bp = _baseline_path(cfg)
        bp.parent.mkdir(parents=True, exist_ok=True)
        bp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        console.print(
            f"[green]✓[/green] baseline saved: prec@{k} {metrics['precision_at_k']} / "
            f"noise@{k} {metrics['noise_at_k']} → {bp}"
        )
        return

    if gate:
        bp = _baseline_path(cfg)
        if not bp.exists():
            raise click.ClickException(
                f"no gate baseline at {bp} — seed it once with "
                "`memo eval recall --labels <set> --update-baseline`"
            )
        try:
            baseline = json.loads(bp.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise click.ClickException(f"unreadable baseline {bp}: {exc}") from exc
        result = eval_recall.check_gate(rows, baseline)
        if as_json:
            click.echo(json.dumps(result.__dict__, ensure_ascii=False, indent=2))
        else:
            color = "green" if result.passed else "red"
            mark = "✓" if result.passed else "✗"
            console.print(f"[{color}]{mark}[/{color}] recall gate: {result.message}")
        sys.exit(0 if result.passed else 1)

    if as_json:
        click.echo(
            json.dumps(
                {
                    "k": k,
                    "cached": cached,
                    "corpus": corpus_fp,
                    "labels_fingerprint": labels.fingerprint(),
                    "configs": [c.name for c in selected_configs],
                    "rows": [r.__dict__ for r in rows],
                    "recommendation": eval_recall.recommend(rows),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    console.print(eval_recall.rows_to_table(rows, k))
    if cached:
        console.print("[dim](cached — use --force to re-run)[/dim]")
    console.print()
    console.print(f"[bold]Recommendation:[/bold] {eval_recall.recommend(rows)}")

    if detail:
        for r in rows:
            console.print(f"\n[bold]### {r.config}[/bold]")
            for d in r.detail:
                tag = "scored" if d["scored"] else "probe"
                console.print(f"  [{tag}] {d['prompt']}")
                for h in d["top"]:
                    flag = "NOISE" if h["noise"] else ("rel" if h["relevant"] else "—")
                    console.print(f"      {h['score']:>5}  {flag:<5}  {h['title']}")


@eval_group.command(name="baseline")
@click.option("--k", type=int, default=5, help="Top-K for the offline recall metrics (default: 5).")
@click.option(
    "--labels",
    "labels_path",
    default="eval/regression_labels.json",
    help="Label set for the offline metrics (schema memo.eval_recall.labels.v1).",
)
@click.option("--json", "as_json", is_flag=True, help="Emit the snapshot as raw JSON.")
def eval_baseline_cmd(k: int, labels_path: str, as_json: bool) -> None:
    """Freeze a baseline snapshot for self-improvement comparison.

    Captures offline prec@K / noise@K, online grounded + tokens (7d / 30d), and
    the active tuned-params version, to state_dir/eval/baseline_snapshot.json.
    """
    cfg = Config.from_env()
    try:
        labels = eval_recall.load_labels(Path(labels_path))
    except (ValueError, OSError) as exc:
        raise click.ClickException(f"labels: {exc}") from exc

    mem = _get_memory(cfg)
    rows = eval_recall.evaluate(mem, k=k, labels=labels, configs=eval_recall.select_configs(None))
    offline = eval_recall.gate_metrics(rows)

    snap = eval_baseline.build_baseline_snapshot(cfg.state_dir, offline)
    path = eval_baseline.snapshot_path(cfg.state_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(snap, ensure_ascii=False, indent=2), encoding="utf-8")
    import os
    os.replace(tmp, path)

    if as_json:
        console.print_json(data=snap)
    else:
        w7 = snap["online"]["window_7d"]
        console.print(
            f"[green]✓[/green] baseline snapshot → {path}\n"
            f"  offline prec@{k} {snap['offline']['precision_at_k']} / "
            f"noise@{k} {snap['offline']['noise_at_k']}\n"
            f"  online 7d grounded {w7['grounded']} (~{w7['tokens']} tok) · "
            f"params {snap['params_version']}"
        )


@eval_group.command(name="harvest")
@click.option(
    "--out",
    "out_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("eval/auto_labels.json"),
    show_default=True,
    help="Where to write the harvested label set.",
)
@click.option(
    "--strong",
    type=float,
    default=0.5,
    show_default=True,
    help="Minimum used_score for a grounding row to become ground truth.",
)
@click.option(
    "--margin",
    type=float,
    default=0.0,
    show_default=True,
    help="Minimum specific_score (when present) to keep a row.",
)
@click.option(
    "--max",
    "max_labels",
    type=int,
    default=200,
    show_default=True,
    help="Cap on harvested labels (most recent first).",
)
@click.option(
    "--merge/--overwrite",
    default=True,
    show_default=True,
    help="Merge into an existing label file (union expect_ids) or replace it.",
)
@click.option("--as-json", is_flag=True, help="Emit the resulting label set as JSON.")
@click.option(
    "--negatives/--no-negatives",
    default=False,
    show_default=True,
    help="Also mine avoid_ids labels from verdict.log (next-turn user verdicts).",
)
def harvest_cmd(
    out_path: Path,
    strong: float,
    margin: float,
    max_labels: int,
    merge: bool,
    as_json: bool,
    negatives: bool,
) -> None:
    """Grow the eval label set from grounding.log (ground truth by construction).

    Every recalled memory the answer demonstrably USED becomes a labeled
    prompt with its grounded id as expect_ids — no hand-labeling. Feed the
    result to the gate alongside the curated set:

      memo eval harvest --out eval/auto_labels.json
      memo eval recall --labels eval/auto_labels.json --k 5 --update-baseline
    """
    cfg = Config.from_env()
    harvested = eval_recall.harvest_labels(
        cfg.state_dir, strong=strong, specific_margin=margin, max_labels=max_labels
    )
    if negatives:
        harvested = eval_recall.merge_label_prompts(
            harvested, eval_recall.harvest_negative_labels(cfg.state_dir)
        )

    existing: list[dict] = []
    if merge and out_path.exists():
        try:
            prior = json.loads(out_path.read_text(encoding="utf-8"))
            if isinstance(prior, dict) and isinstance(prior.get("prompts"), list):
                existing = [p for p in prior["prompts"] if isinstance(p, dict)]
        except (OSError, json.JSONDecodeError):
            existing = []

    prompts = eval_recall.merge_label_prompts(existing, harvested) if existing else harvested
    label_set = {
        "schema": "memo.eval_recall.labels.v1",
        "_doc": (
            "AUTO-HARVESTED from grounding.log by `memo eval harvest`. Each prompt "
            "is a question whose answer demonstrably USED the memory in expect_ids "
            "(ground truth by construction). Safe to regenerate; merge keeps prior "
            "entries and unions expect_ids."
        ),
        "prompts": prompts,
    }

    if as_json:
        click.echo(json.dumps(label_set, ensure_ascii=False, indent=2))
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(label_set, ensure_ascii=False, indent=2), encoding="utf-8")
    answerable = sum(1 for p in prompts if p.get("expect_ids"))
    console.print(
        f"[green]✓[/green] harvested {len(harvested)} grounded label(s); "
        f"{len(prompts)} total ({answerable} with expect_ids) → {out_path}"
    )


@eval_group.command(name="grounding")
@click.option(
    "--labels",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Ground-truth label set (memo.eval_grounding.labels.v1).",
)
@click.option("--as-json", is_flag=True, help="Emit raw JSON.")
def grounding_eval_cmd(labels: Path, as_json: bool) -> None:
    """Score the grounding detector's 'used memo' decision against hand labels."""
    from memo import eval_grounding
    from memo.dashboard import read_grounding_log

    cfg = Config.from_env()
    try:
        label_set = eval_grounding.load_labels(labels)
    except (KeyError, ValueError, json.JSONDecodeError) as exc:
        raise click.ClickException(f"could not load grounding labels {labels}: {exc}") from exc
    rows = read_grounding_log(cfg.state_dir, limit=4000)
    result = eval_grounding.evaluate(rows, label_set)

    if as_json:
        click.echo(json.dumps(result, ensure_ascii=False, indent=2))
        return

    console.print(
        f"grounding detector vs {result['scored']}/{result['labels']} labels "
        f"({result['missing']} not in grounding.log)\n"
    )
    console.print(
        f"  precision {result['precision']}  recall {result['recall']}  f1 {result['f1']}"
    )
    console.print(f"  tp={result['tp']} fp={result['fp']} fn={result['fn']} tn={result['tn']}")
    if result["false_positives"]:
        console.print(
            f"\n  [yellow]false positives (detector said used, label says no):[/yellow] {len(result['false_positives'])}"
        )
    if result["false_negatives"]:
        console.print(
            f"  [yellow]false negatives (detector missed a real use):[/yellow] {len(result['false_negatives'])}"
        )
