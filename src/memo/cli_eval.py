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
import time
from pathlib import Path

import click

from memo import eval_recall
from memo.cli_common import console
from memo.cli_common import get_memory as _get_memory
from memo.config import Config

_CACHE_TTL_S = 24 * 3600


def _cache_path(cfg: Config) -> Path:
    return cfg.state_dir / "eval" / "recall.json"


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
def eval_recall_cmd(
    k: int, labels_path: str | None, as_json: bool, detail: bool, force: bool, no_cache: bool
) -> None:
    """Precision@K / noise@K per retrieval config over labeled prompts.

    Example: memo eval recall --k 3 --labels mylabels.json --json
    """
    cfg = Config.from_env()

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

    mem = _get_memory(cfg)
    corpus_fp = eval_recall.fingerprint_corpus(mem)
    cache_key = f"{corpus_fp}:{labels.fingerprint()}:{k}"

    rows = None
    cached = False
    if not (force or no_cache):
        entry = _load_cache(cfg).get(cache_key)
        if entry and (time.time() - entry.get("ts", 0)) < _CACHE_TTL_S:
            rows = [eval_recall.Row(**r) for r in entry["rows"]]
            cached = True

    if rows is None:
        rows = eval_recall.evaluate(mem, k=k, labels=labels)
        if not no_cache:
            cache = _load_cache(cfg)
            cache[cache_key] = {"ts": time.time(), "k": k, "rows": [r.__dict__ for r in rows]}
            _save_cache(cfg, cache)

    if as_json:
        click.echo(
            json.dumps(
                {
                    "k": k,
                    "cached": cached,
                    "corpus": corpus_fp,
                    "labels_fingerprint": labels.fingerprint(),
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
