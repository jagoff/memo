"""`memo eval` command group — recall-quality measurement.

Promotes the old `scripts/recall_eval.py` harness to a first-class verb so
recall tuning is observable instead of a script hack. `memo eval recall` runs
the labeled-prompt eval (see `eval_recall.py`), prints precision@K / noise@K /
p50 latency per config, recommends the winning knobs, and caches results
(keyed by corpus fingerprint + label set + K) so repeat runs are instant.

NOTE: there is deliberately no `memo autoloop`. The autonomous tuning loop in
Evaluation in this module is self-contained and exercises Memo's own retrieval,
grounding, and answer surfaces.
"""

from __future__ import annotations

import contextlib
import inspect
import json
import re
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import click

from memo import eval_baseline, eval_recall
from memo.cli_common import console
from memo.cli_common import get_memory as _get_memory
from memo.cli_eval_bench import bench_group
from memo.config import Config

if TYPE_CHECKING:
    from memo.eval_against import AgainstResult

_CACHE_TTL_S = 24 * 3600


def _cache_path(cfg: Config) -> Path:
    return cfg.state_dir / "eval" / "recall.json"


# Which command a baseline belongs to. `eval memory` and `eval recall` measure
# different pipelines and used to share ONE file. Only `eval recall` ever wrote
# it, so the bug was on the READ side: `eval memory --gate` silently compared
# its own pipeline against a baseline recorded from recall's. The k guard in
# check_gate catches the case where the two ran at different top-K and nothing
# else — at equal k the comparison looked healthy and meant nothing.
#
# `eval memory` therefore also gains `--update-baseline`; without it, refusing
# the foreign baseline would leave its gate with no way to seed one at all.
GATE_RECALL = "recall"
GATE_MEMORY = "memory"


def _baseline_path(cfg: Config, command: str = GATE_RECALL) -> Path:
    # Machine-local: the gate runs against THIS machine's live index, so the
    # baseline can't be a committed repo file — it lives under state_dir.
    # `recall` keeps the historical filename so existing baselines stay valid.
    name = "recall_baseline.json" if command == GATE_RECALL else f"{command}_baseline.json"
    return cfg.state_dir / "eval" / name


def _reject_foreign_baseline(baseline: dict, expected: str, path: Path) -> None:
    """Refuse a baseline another command wrote.

    Absent stamp = written before baselines were separated; those files only
    ever came from `eval recall`, so treat them as such rather than forcing a
    re-seed on upgrade.
    """
    owner = baseline.get("gate_command") or GATE_RECALL
    if owner != expected:
        raise click.ClickException(
            f"baseline at {path} was written by `memo eval {owner}`, not "
            f"`memo eval {expected}` — re-seed with "
            f"`memo eval {expected} ... --update-baseline`"
        )


def _atomic_write_json(path: Path, payload: object) -> None:
    """Write JSON via tmp+os.replace so a kill (or a concurrent
    --update-baseline in the shared-worktree/multi-agent setup) can't leave a
    truncated baseline that fails every later --gate run."""
    import os

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _live_knob_fingerprint(cfg: Any) -> str:
    """Short hash of the ranking knobs the run will actually inherit.

    Covers the tuned overlay (via its own version helper) plus the resolved
    `RankKnobs`, so a cached row can never belong to a different configuration
    than the one being asked for.
    """
    import hashlib
    import json as _json

    from memo.recall_logic import knobs_from_flags
    from memo.tuned_overlay import params_version

    knobs = knobs_from_flags()
    payload = _json.dumps(
        {k: v for k, v in vars(knobs).items() if not k.startswith("_")},
        sort_keys=True,
        default=str,
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    return f"{params_version(cfg.state_dir)}:{digest}"


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
    # Drop entries the reader can no longer use. The TTL was only ever checked
    # on read, so every nightly A/B appended a key that stayed forever: this
    # file reached 211 MB on a live install — larger than the memory index it
    # was measuring.
    now = time.time()

    def _is_live(entry: object) -> bool:
        if not isinstance(entry, dict):
            return True  # cannot date it; not evidence of staleness
        try:
            ts = float(entry.get("ts") or 0)
        except (TypeError, ValueError):
            # A malformed timestamp must not abort a save that just cost a full
            # retrieval run — keep the entry and let the reader reject it.
            return True
        return (now - ts) < _CACHE_TTL_S

    cache = {key: entry for key, entry in cache.items() if _is_live(entry)}
    p = _cache_path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_baseline(cfg: Config, command: str = GATE_RECALL) -> dict:
    """Load the machine-local gate baseline for `command`, if one exists."""
    p = _baseline_path(cfg, command)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


@click.group(name="eval")
def eval_group() -> None:
    """Measure recall quality against the live corpus."""
    pass


eval_group.add_command(bench_group)


@eval_group.command(name="repo-search")
@click.option(
    "--labels",
    "labels_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="JSON labels using schema memo.eval.repo_search.labels.v1.",
)
@click.option("--repo", default=None, help="Override the repo filter in every label.")
@click.option("--k", type=click.IntRange(min=1, max=100), default=10, show_default=True)
@click.option("--json", "as_json", is_flag=True)
@click.option(
    "--gate",
    is_flag=True,
    help="Fail when graph-first recall is below grep-first or any search call failed.",
)
def eval_repo_search_cmd(
    labels_path: Path,
    repo: str | None,
    k: int,
    as_json: bool,
    gate: bool,
) -> None:
    """Compare lexical-first and graph-enriched repo retrieval symmetrically."""
    from memo.memory import Memory
    from memo.repo_eval import evaluate_repo_search, load_repo_eval_labels

    try:
        labels = load_repo_eval_labels(labels_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise click.ClickException(str(exc)) from exc
    memory = Memory(Config.from_env())
    try:
        report = evaluate_repo_search(memory, labels, k=k, repo=repo)
    finally:
        memory.close()
    payload = report.to_dict()
    if as_json:
        click.echo(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        console.print(
            f"repo-search eval · {report.labels} labels · "
            f"{report.total_search_calls} calls · {report.total_session_elapsed_ms:.1f}ms"
        )
        for strategy in report.strategies:
            console.print(
                f"  {strategy.strategy}: recall@{k}={strategy.recall_at_k:.3f} "
                f"precision@{k}={strategy.precision_at_k:.3f} mrr={strategy.mrr:.3f} "
                f"zero={strategy.zero_results} failures={strategy.failures} "
                f"search_ms={strategy.search_elapsed_ms:.1f}"
            )
    by_name = {row.strategy: row for row in report.strategies}
    grep_first = by_name["grep-first"]
    graph_first = by_name["graph-first"]
    gate_failed = report.failures > 0 or graph_first.recall_at_k < grep_first.recall_at_k
    if gate and gate_failed:
        raise click.exceptions.Exit(1)


@eval_group.command(name="memory")
@click.option(
    "--update-baseline",
    is_flag=True,
    help="Record THIS run as the memory gate's baseline (its own file, never recall's).",
)
@click.option("--k", type=click.IntRange(min=1, max=50), default=5, show_default=True)
@click.option(
    "--labels",
    "labels_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=Path("eval/regression_labels.json"),
    show_default=True,
)
@click.option(
    "--profile",
    "eval_profile",
    type=click.Choice(["quick", "pre-push", "default", "matrix"]),
    default="quick",
    show_default=True,
)
@click.option("--max-prompts", type=click.IntRange(min=1), default=None)
@click.option("--json", "as_json", is_flag=True)
@click.option(
    "--gate", is_flag=True, help="Compare precision/noise with the saved recall baseline."
)
def eval_memory_cmd(
    update_baseline: bool,
    k: int,
    labels_path: Path,
    eval_profile: str,
    max_prompts: int | None,
    as_json: bool,
    gate: bool,
) -> None:
    """Run the memory-quality suite (retrieval, staleness, evidence and latency)."""
    try:
        labels = eval_recall.load_labels(labels_path)
        labels = eval_recall.limit_label_set(labels, max_prompts)
        configs = eval_recall.profile_configs(cast(eval_recall.EvalProfile, eval_profile))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise click.ClickException(str(exc)) from exc

    mem = _get_memory(Config.from_env())
    try:
        rows = eval_recall.evaluate(mem, k=k, labels=labels, configs=configs)
        # Taken while the store is open: check_gate uses it to tell a code
        # regression apart from corpus drift.
        corpus_fp = eval_recall.fingerprint_corpus(mem)
    finally:
        mem.close()
    payload: dict[str, Any] = {
        "schema": "memo.eval.memory.v1",
        "k": k,
        "profile": eval_profile,
        "labels_fingerprint": labels.fingerprint(),
        "metrics": [
            {
                "config": row.config,
                "precision_at_k": row.precision_at_k,
                "recall_at_k": row.recall_at_k,
                "ndcg_at_k": row.ndcg_at_k,
                "mrr": row.mrr,
                "noise_at_k": row.noise_at_k,
                "stale_at_k": row.stale_at_k,
                "canonical_hit_at_k": row.canonical_hit_at_k,
                "latency_ms_p50": row.latency_ms_p50,
                "graph_recall_gain": row.graph_recall_gain,
                "graph_noise_rate": row.graph_noise_rate,
                "graph_explanation_coverage": row.graph_explanation_coverage,
            }
            for row in rows
        ],
    }
    if update_baseline:
        _cfg = Config.from_env()
        _payload = eval_recall.baseline_payload(
            rows, k=k, labels_fingerprint=labels.fingerprint(), corpus_fingerprint=corpus_fp
        )
        _payload["gate_command"] = GATE_MEMORY
        _bp = _baseline_path(_cfg, GATE_MEMORY)
        _atomic_write_json(_bp, _payload)
        console.print(f"[green]✓[/green] memory baseline saved → {_bp}")
        return

    if gate:
        _cfg = Config.from_env()
        baseline = _load_baseline(_cfg, GATE_MEMORY)
        if not baseline:
            raise click.ClickException(
                f"no memory gate baseline at {_baseline_path(_cfg, GATE_MEMORY)} — seed it "
                "with `memo eval memory --update-baseline`. It deliberately no longer falls "
                "back to `eval recall`'s baseline: that measured a different pipeline."
            )
        _reject_foreign_baseline(baseline, GATE_MEMORY, _baseline_path(_cfg, GATE_MEMORY))
        # k is still passed so a baseline seeded at a different top-K is refused
        # rather than compared silently.
        result = eval_recall.check_gate(
            rows, baseline, labels_fingerprint=labels.fingerprint(), k=k
        )
        payload["gate"] = {
            "passed": result.passed,
            "message": result.message,
            "baseline_precision_at_k": result.baseline_precision,
            "baseline_noise_at_k": result.baseline_noise,
        }
    if as_json:
        click.echo(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        console.print(f"memory eval · {len(labels.prompts)} prompts · k={k}")
        for metric in payload["metrics"]:
            console.print(
                f"  {metric['config']}: prec={metric['precision_at_k']:.3f} "
                f"recall={metric['recall_at_k']:.3f} noise={metric['noise_at_k']:.3f} "
                f"stale={metric['stale_at_k']:.3f} p50={metric['latency_ms_p50']:.1f}ms"
            )
        if "gate" in payload:
            console.print(payload["gate"]["message"])
    if gate and not payload["gate"]["passed"]:
        raise click.exceptions.Exit(1)


@eval_group.command(name="relations")
@click.option(
    "--labels",
    "labels_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=Path("eval/relation_candidate_labels.json"),
    show_default=True,
)
@click.option("--gate", is_flag=True, help="Exit non-zero when the fixture thresholds fail.")
@click.option("--json", "as_json", is_flag=True, help="Emit raw JSON.")
def eval_relations_cmd(labels_path: Path, gate: bool, as_json: bool) -> None:
    """Measure deterministic relation eligibility and namespace leakage."""
    from dataclasses import asdict

    from memo.eval_relations import evaluate

    try:
        result = evaluate(labels_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise click.ClickException(str(exc)) from exc
    payload = asdict(result)
    if as_json:
        click.echo(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        mark = "✓" if result.passed else "✗"
        color = "green" if result.passed else "red"
        console.print(
            f"[{color}]{mark}[/{color}] relation gate: recall={result.recall:.3f} "
            f"precision={result.precision:.3f} noise={result.noise:.3f} "
            f"({result.cases} cases, {result.elapsed_ms:.3f} ms)"
        )
    if gate and not result.passed:
        raise click.exceptions.Exit(1)


@eval_group.command(name="behavior")
@click.option(
    "--scenarios",
    "scenarios_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=Path("eval/behavior_scenarios.json"),
    show_default=True,
)
@click.option(
    "--recall-only",
    is_flag=True,
    help="Score only the recall-layer gates. No LLM is loaded — deterministic and fast.",
)
@click.option("--only", default=None, help="Run a single scenario by id.")
@click.option("--gate", is_flag=True, help="Exit non-zero when any scenario fails.")
@click.option("--json", "as_json", is_flag=True, help="Emit raw JSON.")
def eval_behavior_cmd(
    scenarios_path: Path, recall_only: bool, only: str | None, gate: bool, as_json: bool
) -> None:
    """Measure whether a recalled memory actually steers the answer.

    Seeds an isolated store, runs the real `memo recall-hook` against it, then
    judges the answer a model gives with that injected block. A red gate means
    the injected payload under-steers — not that a specific agent misbehaved.
    """
    from memo.eval_behavior import load_scenarios, run_scenarios

    try:
        scenarios = load_scenarios(scenarios_path)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    if only:
        scenarios = [s for s in scenarios if s.scenario_id == only]
        if not scenarios:
            raise click.ClickException(f"no scenario with id {only!r} in {scenarios_path}")

    results = run_scenarios(scenarios, recall_only=recall_only)
    passed = sum(1 for r in results if r.passed)

    if as_json:
        click.echo(
            json.dumps(
                {
                    "schema_version": "memo.eval_behavior.report.v1",
                    "recall_only": recall_only,
                    "scenarios": len(results),
                    "passed": passed,
                    "results": [r.as_dict() for r in results],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        for result in results:
            mark = "✓" if result.passed else "✗"
            color = "green" if result.passed else "red"
            console.print(f"[{color}]{mark}[/{color}] {result.scenario_id}")
            if result.error:
                console.print(f"    [red]error:[/red] {result.error}")
            for gate_result in result.gates:
                if not gate_result.passed:
                    console.print(f"    [red]✗[/red] {gate_result.gate.kind}: {gate_result.detail}")
        mode = "recall-layer only" if recall_only else "full"
        console.print(f"\n{passed}/{len(results)} scenarios passed ({mode})")

    if gate and passed != len(results):
        raise click.exceptions.Exit(1)


def _run_gate(
    rows: list[Any],
    cfg: Config,
    *,
    labels_fingerprint: str,
    k: int,
    corpus_fingerprint: str,
) -> eval_recall.GateResult:
    """Load the machine-local baseline and check `rows` against it.

    `corpus_fingerprint` is what lets the result distinguish a code regression
    from corpus drift; see `eval_recall.check_gate`.
    """
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
    _reject_foreign_baseline(baseline, GATE_RECALL, bp)
    return eval_recall.check_gate(
        rows,
        baseline,
        labels_fingerprint=labels_fingerprint,
        k=k,
        corpus_fingerprint=corpus_fingerprint,
    )


def _resolve_cache_flags(
    *,
    gate: bool,
    update_baseline: bool,
    graph_ab: bool,
    against_ref: str | None,
    force: bool,
    no_cache: bool,
) -> tuple[bool, bool]:
    """Fresh numbers are mandatory for anything that renders a verdict.

    `--gate`/`--update-baseline`/`--graph-ab` need FORCE (skip the cache
    read; a plain run may still write it). `--against` needs the stronger
    NO_CACHE (skip the read AND the write) — `force` alone would still let
    the current side's experimental numbers poison the shared cache that
    every other worktree on this machine reads from.
    """
    if gate or update_baseline or graph_ab:
        force = True
    if against_ref:
        no_cache = True
    return force, no_cache


def _validate_against_flags(
    against_ref: str | None,
    quick: bool,
    max_prompts: int | None,
    gate: bool,
    update_baseline: bool,
    graph_ab: bool,
) -> None:
    """--against requires both sides to score the same prompt set, and it
    exits via `sys.exit` before --gate's, --update-baseline's, or
    --graph-ab's own logic ever runs — combined with any of them, it
    silently discards the flag's effect the user actually asked for (no
    baseline written, no gate check performed, no graph A/B table printed).
    """
    if not against_ref:
        return
    if quick or max_prompts is not None:
        bad = "--quick" if quick else "--max-prompts"
        raise click.ClickException(
            f"{bad} cannot be combined with --against — both sides of the "
            "comparison must evaluate the same prompt set, or the delta "
            "silently stops meaning anything."
        )
    if gate or update_baseline:
        bad = "--gate" if gate else "--update-baseline"
        raise click.ClickException(
            f"{bad} cannot be combined with --against — --against prints its "
            f"own comparison and exits before {bad}'s effect would run, so "
            f"{bad} would silently do nothing."
        )
    if graph_ab:
        raise click.ClickException(
            "--graph-ab cannot be combined with --against — the graph A/B "
            "sweep runs before --against's comparison and exits before its "
            "table is ever printed, so --graph-ab would silently burn two "
            "extra eval sweeps for output nobody sees."
        )


def _run_against(
    against_ref: str,
    *,
    rows: list[Any],
    labels_path: str | None,
    k: int,
    profile: str | None,
    config_names: tuple[str, ...],
    labels_fingerprint: str,
) -> AgainstResult:
    """Run the ref side and compare it against `rows` (the current side).

    `resolve_repo_root`/`run_against` can raise `eval_against.AgainstError`
    for a bad ref, a missing git repo, or a polluted/non-JSON subprocess
    payload (see m3). Translated here into `click.ClickException` so the
    failure reads as a controlled CLI error instead of a raw traceback —
    otherwise `exit=1` from a broken pipeline is indistinguishable from a
    genuine ranking regression, which undercuts the point of surfacing the
    real reason at all.
    """
    from memo import eval_against

    try:
        # Anchor on the INVOCATION cwd, not on this module's directory: under
        # the isolated uv-tool/pipx install `__file__` lives in site-packages,
        # outside any git worktree, so `--against` died with `fatal: not a git
        # repository` on precisely the install the pre-push gate's own failure
        # message tells you to run it from.
        repo_root = eval_against.resolve_repo_root(Path.cwd())
        ref_argv = eval_against.build_eval_argv(
            labels_path=labels_path, k=k, profile=profile, configs=config_names
        )
        ref_run = eval_against.run_against(against_ref, repo_root=repo_root, argv=ref_argv)
    except eval_against.AgainstError as exc:
        raise click.ClickException(str(exc)) from exc
    return eval_against.compare_rows(
        [r.__dict__ for r in rows],
        ref_run.rows,
        current_labels_fingerprint=labels_fingerprint,
        ref_labels_fingerprint=ref_run.labels_fingerprint,
    )


@eval_group.command(name="recall")
@click.option("--k", type=int, default=3, help="Top-K to score (default: 3).")
@click.option(
    "--labels",
    "labels_path",
    # resolve_path=True: --against forwards this path verbatim into a
    # subprocess whose cwd is a *different* worktree — a relative path would
    # resolve against the wrong directory and silently score the two sides
    # against different label sets.
    type=click.Path(exists=True, dir_okay=False, resolve_path=True),
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
    "--profile",
    type=click.Choice(["quick", "default", "pre-push", "matrix", "expensive", "hype"]),
    default=None,
    help="Named config profile. Explicit --config values override this.",
)
@click.option(
    "--config",
    "config_names",
    multiple=True,
    help="Config to run (A-J, or full name). Repeat for multiple configs.",
)
@click.option(
    "--max-prompts",
    type=click.IntRange(min=1),
    default=None,
    help="Evaluate only the first N prompts from the label set.",
)
@click.option("--progress", is_flag=True, help="Print progress before each prompt search.")
@click.option(
    "--graph-ab",
    is_flag=True,
    help="Also run graph-off and graph-on variants of the selected configs and report deltas.",
)
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
@click.option(
    "--against",
    "against_ref",
    default=None,
    help="Compare this worktree's code against a git ref on the SAME live corpus. "
    "Both runs are uncached, so the corpus term cancels and the delta is the diff. "
    "This — not --gate — is the check a ranking change has to clear.",
)
def eval_recall_cmd(
    k: int,
    labels_path: str | None,
    as_json: bool,
    detail: bool,
    force: bool,
    no_cache: bool,
    quick: bool,
    profile: str | None,
    config_names: tuple[str, ...],
    max_prompts: int | None,
    progress: bool,
    graph_ab: bool,
    gate: bool,
    update_baseline: bool,
    against_ref: str | None,
) -> None:
    """Precision@K / noise@K per retrieval config over labeled prompts.

    Example: memo eval recall --k 3 --labels mylabels.json --json

    Gate (local pre-commit, runs against the live index):
      memo eval recall --labels eval/regression_labels.json --update-baseline
      memo eval recall --labels eval/regression_labels.json --gate
    """
    _validate_against_flags(against_ref, quick, max_prompts, gate, update_baseline, graph_ab)

    cfg = Config.from_env()
    force, no_cache = _resolve_cache_flags(
        gate=gate,
        update_baseline=update_baseline,
        graph_ab=graph_ab,
        against_ref=against_ref,
        force=force,
        no_cache=no_cache,
    )

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
        if config_names:
            selected_configs = eval_recall.select_configs(list(config_names), quick=quick)
        elif profile is not None:
            selected_configs = eval_recall.profile_configs(cast(eval_recall.EvalProfile, profile))
        else:
            selected_configs = eval_recall.select_configs(None, quick=quick)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    mem = _get_memory(cfg)
    corpus_fp = eval_recall.fingerprint_corpus(mem)
    configs_fp = ",".join(c.name for c in selected_configs)
    # The config NAME only carries mode + floor ("L live/vec/0.5"); every other
    # knob that governs ranking — the boosts, skip_below, gap_threshold,
    # dedup_collapse, mmr_lambda … — is inherited from the env/overlay chain and
    # would otherwise be invisible to the key, so an A/B that changed one of
    # them silently replayed the other arm's cached rows.
    cache_key = f"{corpus_fp}:{labels.fingerprint()}:{configs_fp}:{k}:{_live_knob_fingerprint(cfg)}"

    rows = None
    cached = False
    if not (force or no_cache):
        entry = _load_cache(cfg).get(cache_key)
        if entry and (time.time() - entry.get("ts", 0)) < _CACHE_TTL_S:
            rows = [eval_recall.Row(**r) for r in entry["rows"]]
            cached = True

    if rows is None:
        if not as_json:
            progress = True
            total_searches = len(selected_configs) * len(labels.prompts)
            console.print(
                f"[dim]Running recall eval: {len(selected_configs)} config(s) x "
                f"{len(labels.prompts)} prompt(s) = {total_searches} search(es).[/dim]"
            )

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

    graph_ab_payload: dict[str, object] | None = None
    if graph_ab:
        off_configs, on_configs = eval_recall.graph_ab_configs(selected_configs)

        def _ab_progress(cfg_: eval_recall.Cfg, index: int, total: int) -> None:
            if progress and not as_json:
                console.print(f"[dim]graph A/B {cfg_.name}: prompt {index}/{total}[/dim]")

        if not as_json:
            console.print(
                f"[dim]Running graph A/B: {len(selected_configs) * 2} config run(s).[/dim]"
            )
        off_rows = eval_recall.evaluate(
            mem,
            k=k,
            labels=labels,
            configs=off_configs,
            progress=_ab_progress if progress and not as_json else None,
        )
        on_rows = eval_recall.evaluate(
            mem,
            k=k,
            labels=labels,
            configs=on_configs,
            progress=_ab_progress if progress and not as_json else None,
        )
        comparison = eval_recall.graph_ab_compare(off_rows, on_rows)
        graph_ab_payload = {
            "off_rows": [r.__dict__ for r in off_rows],
            "on_rows": [r.__dict__ for r in on_rows],
            "comparison": comparison,
            "summary": eval_recall.graph_ab_summary(comparison),
        }

    if against_ref:
        against_result = _run_against(
            against_ref,
            rows=rows,
            labels_path=labels_path,
            k=k,
            profile=profile,
            config_names=tuple(config_names),
            labels_fingerprint=labels.fingerprint(),
        )
        if as_json:
            click.echo(json.dumps(against_result.__dict__, ensure_ascii=False, indent=2))
        else:
            color = "green" if against_result.passed else "red"
            mark = "✓" if against_result.passed else "✗"
            console.print(f"[{color}]{mark}[/{color}] vs {against_ref}: {against_result.message}")
        sys.exit(0 if against_result.passed else 1)

    if update_baseline:
        # Persist the FULL metrics (precision/noise ∪ avoid@k / avoid_leak@k) so
        # check_gate can enforce the ⛔ coverage/leakage floors — bare
        # gate_metrics wrote neither, leaving those checks vacuously true forever.
        metrics = eval_recall.full_gate_metrics(rows)
        payload = eval_recall.baseline_payload(
            rows,
            k=k,
            labels_fingerprint=labels.fingerprint(),
            corpus_fingerprint=corpus_fp,
        )
        # Stamp the owner so the other eval command's gate refuses this file
        # instead of silently comparing against it.
        payload["gate_command"] = GATE_RECALL
        bp = _baseline_path(cfg, GATE_RECALL)
        _atomic_write_json(bp, payload)
        console.print(
            f"[green]✓[/green] baseline saved: config {metrics['config']!r} · "
            f"prec@{k} {metrics['precision_at_k']} / "
            f"noise@{k} {metrics['noise_at_k']} → {bp}"
        )
        return

    if gate:
        result = _run_gate(
            rows,
            cfg,
            labels_fingerprint=labels.fingerprint(),
            k=k,
            corpus_fingerprint=corpus_fp,
        )
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
                    "graph_ab": graph_ab_payload,
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
    if graph_ab_payload is not None:
        comparison = cast(list[dict[str, object]], graph_ab_payload["comparison"])
        console.print(eval_recall.graph_ab_table(comparison))

    if detail:
        for r in rows:
            console.print(f"\n[bold]### {r.config}[/bold]")
            for d in r.detail:
                tag = "scored" if d["scored"] else "probe"
                console.print(f"  [{tag}] {d['prompt']}")
                for h in d["top"]:
                    flag = "NOISE" if h["noise"] else ("rel" if h["relevant"] else "—")
                    console.print(f"      {h['score']:>5}  {flag:<5}  {h['title']}")


@eval_group.command(name="chat")
@click.option(
    "--corpus",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=Path("eval/chat_regression_corpus.json"),
    show_default=True,
    help="JSON corpus (schema synapse.eval_chat.query.v1, rescued from synapse).",
)
@click.option("--only", default=None, help="Run only the query with this id.")
def eval_chat_cmd(corpus: Path, only: str | None) -> None:
    """Regression gate: run the corpus through the chat pipeline and check outputs."""
    from rich.table import Table

    from memo.chat.pipeline import chat_stream
    from memo.eval_chat import apply_checks

    data = json.loads(corpus.read_text(encoding="utf-8"))
    queries = [q for q in data.get("queries", []) if not only or q.get("id") == only]
    if not queries:
        raise click.ClickException(f"no query with id {only!r} in {corpus}")

    mem = _get_memory(Config.from_env())
    rows = []
    for query in queries:
        t0 = time.monotonic()
        # Regression gate, not a user-visible chat turn: don't let it inflate
        # access_count on the live corpus it queries (see chat_stream's
        # `_track_usage` docstring).
        events = list(chat_stream(mem, str(query.get("question") or ""), _track_usage=False))
        done = next((e for e in reversed(events) if e.get("type") in {"done", "error"}), {})
        rows.append(apply_checks(query, done, int((time.monotonic() - t0) * 1000)))

    table = Table(title=f"eval chat — {corpus.name}")
    table.add_column("id")
    table.add_column("passed")
    table.add_column("ms", justify="right")
    table.add_column("failed checks")
    for row in rows:
        failed = ", ".join(c["check"] for c in row["checks"] if not c["passed"])
        table.add_row(row["id"], "✓" if row["passed"] else "✗", str(row["total_ms"]), failed)
    console.print(table)

    latencies = sorted(r["total_ms"] for r in rows) or [0]
    p50 = latencies[len(latencies) // 2]
    p95_idx = max(0, min(len(latencies) - 1, int(len(latencies) * 0.95) - 1))
    console.print(f"p50={p50}ms p95={latencies[p95_idx]}ms")

    if any(not r["passed"] for r in rows):
        raise click.exceptions.Exit(1)


def _tokens_baseline_path(cfg: Config) -> Path:
    return cfg.state_dir / "eval" / "token_baseline.json"


@eval_group.command(name="tokens")
@click.option("--k", type=int, default=5, help="Top-K hits to render per P1 prompt.")
@click.option(
    "--labels",
    "labels_path",
    type=click.Path(exists=True, dir_okay=False),
    default="eval/regression_labels.json",
    help="P1 label set (schema memo.eval_recall.labels.v1).",
)
@click.option(
    "--corpus",
    "corpus_path",
    type=click.Path(exists=True, dir_okay=False),
    default="eval/token_corpus.json",
    help="P2 capture corpus (schema memo.token_corpus.v1).",
)
@click.option("--json", "as_json", is_flag=True, help="Emit raw JSON.")
@click.option("--update-baseline", is_flag=True, help="Save current per-lever metrics as baseline.")
@click.option(
    "--gate", is_flag=True, help="Exit non-zero if a passing lever regressed vs baseline."
)
@click.option("--force", is_flag=True, help="(accepted for parity; runs are never cached).")
def eval_tokens_cmd(
    k: int,
    labels_path: str,
    corpus_path: str,
    as_json: bool,
    update_baseline: bool,
    gate: bool,
    force: bool,
) -> None:
    """Measure each token-economy lever: Δtokens + Δquality, per plane.

    P1 (recall-output): render OFF vs ON under each lever, precision = expect_ids
    surviving into the injected block. P2 (capture): crush the corpus, quality =
    the labeled must-keep row surviving. A lever PASSes iff it cuts >=5% tokens
    AND does not drop quality.

    Gate (local, runs against the live index):
      memo eval tokens --update-baseline
      memo eval tokens --gate
    """
    from memo import eval_tokens

    cfg = Config.from_env()
    labels = eval_recall.load_labels(Path(labels_path))
    corpus = eval_tokens.load_capture_corpus(Path(corpus_path))
    mem = _get_memory(cfg)

    def _search(text: str) -> list:
        # `memo eval tokens` is not a user-visible retrieval: without this,
        # every search hit writes an access-log row (search_ops.py's
        # `_stage_record_usage`), inflating `access_count` on whichever
        # memories the eval surfaces — the same signal `memo usefulness` /
        # `dead_weight()` read to decide what's noise. Signature-introspected
        # like eval_recall.py's `_search_for_eval` so a test stub whose
        # `search()` doesn't take the kwarg (real Memory.search's keyword-only
        # shape, no **kwargs) is left untouched.
        search_kwargs: dict[str, Any] = {"limit": k}
        with contextlib.suppress(TypeError, ValueError):
            sig = inspect.signature(mem.search)
            has_var_kw = any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
            )
            if "_track_usage" in sig.parameters or has_var_kw:
                search_kwargs["_track_usage"] = False
        return list(mem.search(text, **search_kwargs))

    def _crush(content: str) -> tuple[str, str | None]:
        from memo.capture_core import maybe_crush_json_capture

        with eval_tokens.env_pins({"MEMO_CRUSHER_ENABLED": "1"}):
            return maybe_crush_json_capture(content, context="", config=cfg)

    rows = eval_tokens.run_all(
        prompts=labels.prompts, search=_search, corpus=corpus, crush_fn=_crush, k=k
    )
    metrics = eval_tokens.gate_metrics(rows)

    if update_baseline:
        bp = _tokens_baseline_path(cfg)
        _atomic_write_json(bp, metrics)
        console.print(f"[green]✓[/green] token baseline saved → {bp}")
        return

    if gate:
        bp = _tokens_baseline_path(cfg)
        if not bp.exists():
            raise click.ClickException(
                f"no token gate baseline at {bp} — seed it with "
                "`memo eval tokens --update-baseline`"
            )
        baseline = json.loads(bp.read_text(encoding="utf-8"))
        result = eval_tokens.check_gate(rows, baseline)
        if as_json:
            click.echo(json.dumps(result.__dict__, ensure_ascii=False, indent=2))
        else:
            color = "green" if result.passed else "red"
            mark = "✓" if result.passed else "✗"
            console.print(f"[{color}]{mark}[/{color}] token gate: {result.message}")
        sys.exit(0 if result.passed else 1)

    if as_json:
        click.echo(json.dumps(metrics, ensure_ascii=False, indent=2))
        return
    for r in rows:
        verdict = "PASS" if r.passed else "FAIL"
        color = "green" if r.passed else "yellow"
        console.print(
            f"[{color}]{verdict}[/{color}] {r.lever} [{r.plane}]  "
            f"saved {r.saved_frac * 100:+.1f}%  Δquality {r.quality_delta:+.2f}"
        )


@eval_group.command(name="ab")
@click.option("--k", type=int, default=5, show_default=True, help="Recall top-K used as context.")
@click.option(
    "--labels",
    "labels_path",
    type=click.Path(exists=True, dir_okay=False),
    default="eval/regression_labels.json",
    show_default=True,
    help="Label set (schema memo.eval_recall.labels.v1); only answerable prompts run.",
)
@click.option("--seed", type=int, default=42, show_default=True, help="Pair-order seed.")
@click.option(
    "--tie-band",
    type=float,
    default=0.05,
    show_default=True,
    help="Judge-score delta treated as a tie.",
)
@click.option(
    "--max-prompts",
    type=click.IntRange(min=1),
    default=None,
    help="Run only the first N answerable prompts (3 MLX chat calls each).",
)
@click.option("--model", "model_name", default=None, help="Chat model (default: cfg.llm_model).")
@click.option("--json", "as_json", is_flag=True, help="Emit raw JSON.")
def eval_ab_cmd(
    k: int,
    labels_path: str,
    seed: int,
    tie_band: float,
    max_prompts: int | None,
    model_name: str | None,
    as_json: bool,
) -> None:
    """Blind-judge A/B: answer each labeled prompt WITH recall context vs
    WITHOUT, judged blind by the local LLM (0-1 rubric, deterministic pair
    order, tie band). Reports win/tie/loss + context-token cost; the raw run
    persists under state_dir/eval/ for audit.

    Offline batch (3 MLX chat calls per prompt) — never the recall hook.

      memo eval ab --labels eval/regression_labels.json --k 5
    """
    from memo import eval_ab

    cfg = Config.from_env()
    try:
        labels = eval_recall.load_labels(Path(labels_path))
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    prompts = eval_ab.answerable_prompts(labels)
    if max_prompts is not None:
        prompts = prompts[:max_prompts]
    if not prompts:
        raise click.ClickException(f"no answerable prompts in {labels_path}")

    mem = _get_memory(cfg)
    model = model_name or cfg.llm_model

    # Recall-faithful ON retrieval: the shared rank_hits pipeline under the
    # live flag resolution — the same path the eval_recall gate measures — not
    # a raw search (see eval_ab.recall_search_fn).
    _search = eval_ab.recall_search_fn(mem, k=k)

    from memo.llm import MLXChat  # deferred — MLX invariant

    chat_backend = MLXChat()

    def _chat(system: str, user: str) -> str:
        out = chat_backend.chat(
            model,
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            options={"temperature": 0.0, "max_tokens": 512},
        )
        return str((out.get("message") or {}).get("content") or "")

    def _progress(index: int, total: int) -> None:
        if not as_json:
            console.print(f"[dim]ab: prompt {index}/{total} (3 chat calls)[/dim]")

    results = eval_ab.run_ab(
        prompts, search=_search, chat=_chat, k=k, seed=seed, tie_band=tie_band, progress=_progress
    )
    summary = eval_ab.summarize(results)
    payload = {
        "schema": eval_ab.AB_SCHEMA,
        "prompts_version": eval_ab.PROMPTS_VERSION,
        "model": model,
        "k": k,
        "seed": seed,
        "tie_band": tie_band,
        "labels_fingerprint": labels.fingerprint(),
        "summary": summary,
        "pairs": [r.__dict__ for r in results],
    }
    detail_path = eval_ab.write_detail(cfg.state_dir, payload)

    if as_json:
        click.echo(json.dumps({**payload, "detail_path": str(detail_path)}, ensure_ascii=False))
        return

    for r in results:
        mark = {"on": "green", "off": "red", "tie": "yellow"}[r.winner]
        flag = " (judge parse error)" if r.judge_parse_error else ""
        if r.leaked:
            flag = " (leaked — forced tie)"
        console.print(
            f"[{mark}]{r.winner.upper():<4}[/{mark}] on {r.mean_on:.2f} vs off {r.mean_off:.2f}  "
            f"ctx {r.context_tokens_on:>5} tok  {r.prompt[:56]}{flag}"
        )
    console.print(
        f"\n[bold]A/B (memo ON vs OFF):[/bold] {summary['wins_on']} win / "
        f"{summary['ties']} tie / {summary['losses_on']} loss over {summary['prompts']} prompts "
        f"(win rate {summary['win_rate_on']:.0%}, mean Δ {summary['mean_delta']:+.3f})"
    )
    sd = summary["sub_deltas"]
    console.print(
        f"  Δ correctness {sd['correctness']:+.3f} · groundedness {sd['groundedness']:+.3f} · "
        f"specificity {sd['specificity']:+.3f}"
    )
    console.print(
        f"  context tokens: ON {summary['context_tokens_on']} vs OFF "
        f"{summary['context_tokens_off']}"
    )
    if summary["judge_parse_errors"]:
        console.print(
            f"  [yellow]⚠ {summary['judge_parse_errors']} judge parse error(s) "
            f"(counted as ties)[/yellow]"
        )
    if summary["leaked_pairs"]:
        console.print(
            f"  [yellow]⚠ {summary['leaked_pairs']} leaked pair(s) — an answer named its "
            f"sources; judged blind impossible, forced tie[/yellow]"
        )
    console.print(f"  [dim]raw detail → {detail_path}[/dim]")


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
        click.echo(json.dumps(snap, ensure_ascii=False, indent=2))
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


@eval_group.command(name="expand-labels")
@click.option(
    "--labels",
    "labels_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Source label set (schema memo.eval_recall.labels.v1).",
)
@click.option(
    "--out",
    "out_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("eval/expanded_labels.json"),
    show_default=True,
    help="Where to write source prompts + paraphrases (never edits the source file).",
)
@click.option("--per-prompt", type=int, default=2, show_default=True)
@click.option(
    "--max-prompts",
    type=int,
    default=40,
    show_default=True,
    help="Cap on source prompts (each costs one MLX chat call).",
)
def expand_labels_cmd(labels_path: Path, out_path: Path, per_prompt: int, max_prompts: int) -> None:
    """Grow the eval label set with MLX paraphrases of expect_ids prompts.

    Offline batch (one local MLX chat call per source prompt, capped by
    --max-prompts) — never the recall hook. Paraphrases inherit the source
    prompt's expect_ids/project, so prec@K gets coverage on rephrasings."""
    if out_path.resolve() == labels_path.resolve():
        raise click.UsageError(
            "--out must differ from --labels (expand-labels never rewrites its source)."
        )
    try:
        raw = json.loads(labels_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise click.ClickException(f"labels: {exc}") from exc
    prompts = [p for p in (raw.get("prompts") or []) if isinstance(p, dict)]
    cfg = Config.from_env()

    def _generate(text: str, n: int) -> list[str]:
        from memo.llm import MLXChat  # deferred — MLX invariant

        ask = (
            f"Reescribí la siguiente pregunta de {n} formas distintas, en el "
            f"mismo idioma y con el mismo significado. Una por línea, sin "
            f"numerar ni comentar.\n\n{text}"
        )
        resp = MLXChat().chat(
            model=cfg.llm_model,
            messages=[{"role": "user", "content": ask}],
            options={"temperature": 0.7, "max_tokens": 200},
        )
        content = ((resp.get("message") or {}).get("content")) or ""
        lines = [re.sub(r"^\s*(?:\d+[.)]|[-•*])\s*", "", ln).strip() for ln in content.splitlines()]
        return [ln for ln in lines if ln]

    new = eval_recall.expand_labels(
        prompts, generate=_generate, per_prompt=per_prompt, max_prompts=max_prompts
    )
    label_set = {
        "schema": "memo.eval_recall.labels.v1",
        "_doc": (
            "Source prompts + MLX paraphrases (`memo eval expand-labels`). Each "
            "paraphrase inherits its source prompt's expect_ids (expanded_from "
            "records the source). Safe to regenerate."
        ),
        "prompts": prompts + new,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(label_set, ensure_ascii=False, indent=2), encoding="utf-8")
    console.print(f"[green]✓[/green] {len(new)} paraphrase label(s) added → {out_path}")


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
