"""`memo eval bench` — public long-memory benchmarks (LoCoMo / LongMemEval).

Offline batch. Each benchmark sample is ingested into an ISOLATED store
(`<state_dir>/bench/stores/<dataset>/<sample>/{data,state}` — never the live
corpus; the live Config supplies only model settings and the receipt
location). Retrieval metrics reuse eval_recall.run_config (the shared
rank_hits path); QA grading drives `memo ask` per question with a pluggable
judge (local MLX default; MEMO_BENCH_JUDGE=api for a stronger API judge).

Tip: export MEMO_SAVE_DEDUP_CHECK=0 during large ingests — the save-time
near-dup warning costs one vector search per turn and means nothing here.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import click

from memo import eval_bench
from memo.cli_common import console
from memo.cli_common import get_memory as _get_memory
from memo.config import Config
from memo.errors import MemoError


@click.group(name="bench")
def bench_group() -> None:
    """Public long-memory benchmarks in an isolated store."""


@bench_group.command(name="run")
@click.option(
    "--dataset",
    type=click.Choice(sorted(eval_bench.DATASET_URLS)),
    default="locomo",
    show_default=True,
    help="Which benchmark to run (also selects the parser for --file).",
)
@click.option(
    "--file",
    "file_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Already-downloaded dataset JSON (skips the download).",
)
@click.option("--url", default=None, help="Override the dataset download URL.")
@click.option("--k", type=int, default=5, show_default=True, help="Top-K to score.")
@click.option(
    "--max-samples",
    type=click.IntRange(min=1),
    default=None,
    help="Cap samples (LoCoMo: conversations; LongMemEval: questions).",
)
@click.option(
    "--max-qa", type=click.IntRange(min=1), default=None, help="Cap QA graded per sample."
)
@click.option("--retrieval-only", is_flag=True, help="Skip QA grading (no LLM, no judge).")
@click.option(
    "--contradict-scan",
    is_flag=True,
    help="Run memo's contradiction scanner on each isolated store after ingest "
    "and enable the contradiction penalty during scoring, so knowledge-update "
    "measures memo's real conflict handling (uses cfg.helper_model, not the "
    "answer LLM — off the 30B OOM path). Default off keeps raw-retrieval scoring.",
)
@click.option(
    "--workdir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Root for isolated stores + dataset cache (default <state_dir>/bench).",
)
@click.option("--json", "as_json", is_flag=True, help="Emit the receipt as JSON.")
def bench_run(
    dataset: str,
    file_path: Path | None,
    url: str | None,
    k: int,
    max_samples: int | None,
    max_qa: int | None,
    retrieval_only: bool,
    contradict_scan: bool,
    workdir: Path | None,
    as_json: bool,
) -> None:
    """Ingest a public benchmark into isolated stores and score memo on it."""
    live = Config.from_env()
    if contradict_scan:
        # The scoring pass reads this flag in mem.search; set it once for the run
        # so the penalty demotes the stale side of scanned update/conflict pairs.
        os.environ["MEMO_CONTRADICT_PENALTY_ENABLED"] = "1"
    bench_root = workdir or (live.state_dir / "bench")
    try:
        data_file = file_path or eval_bench.fetch_dataset(
            dataset, bench_root / "datasets", url=url
        )
        raw = json.loads(Path(data_file).read_text(encoding="utf-8"))
        samples = eval_bench.parse_dataset(dataset, raw)
    except (MemoError, ValueError, OSError, json.JSONDecodeError) as exc:
        raise click.ClickException(str(exc)) from exc
    if not samples:
        raise click.ClickException(f"dataset {dataset} yielded no samples")
    if max_samples:
        samples = samples[:max_samples]

    judge = None
    if not retrieval_only:
        try:
            judge = eval_bench.judge_from_flags(live)
        except MemoError as exc:
            raise click.ClickException(str(exc)) from exc

    per_sample_rows: list[dict] = []
    qa_results: list[eval_bench.QAResult] = []
    scan_examined = 0
    scan_inserted = 0
    for si, sample in enumerate(samples, start=1):
        root = bench_root / "stores" / dataset / eval_bench._safe_dir_name(sample.sample_id)
        bcfg = eval_bench.bench_store_config(root, live)
        mem = _get_memory(bcfg)
        try:
            if not as_json:
                console.print(
                    f"[dim]bench {dataset}: sample {si}/{len(samples)} "
                    f"({sample.sample_id}) — {len(sample.turns)} turns, "
                    f"{len(sample.qa)} QA[/dim]"
                )
            ingest = eval_bench.ingest_sample(mem, sample, root)
            if contradict_scan:
                ex, ins = eval_bench.scan_bench_contradictions(mem)
                scan_examined += ex
                scan_inserted += ins
            per_sample_rows.append(eval_bench.score_retrieval(mem, sample, ingest, k=k))
            if judge is not None:
                qa_results.extend(
                    eval_bench.grade_sample_qa(mem, sample, judge, k=k, max_qa=max_qa)
                )
        finally:
            mem.close()

    retrieval = eval_bench.aggregate_retrieval(per_sample_rows)
    receipt = {
        "schema": eval_bench.RECEIPT_SCHEMA,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "dataset": dataset,
        "k": k,
        "n_samples": len(samples),
        "judge": getattr(judge, "name", None),
        "llm_model": live.llm_model,
        "embedder_model": live.embedder_model,
        "retrieval": retrieval,
        "qa": eval_bench.qa_accuracy_by_category(qa_results) if qa_results else {},
        # Auxiliary capability-taxonomy rollup (Memoria-style 6-bucket view).
        "capability_retrieval": eval_bench.capability_retrieval(retrieval),
    }
    if qa_results:
        # First-class abstention metric + per-bucket QA accuracy.
        receipt["capability_qa"] = eval_bench.capability_qa(qa_results)
        receipt["abstention"] = eval_bench.abstention_summary(qa_results)
    if contradict_scan:
        receipt["contradict_scan"] = {"examined": scan_examined, "inserted": scan_inserted}
    path = eval_bench.write_receipt(live.state_dir, receipt)
    if as_json:
        click.echo(json.dumps(receipt, ensure_ascii=False, indent=2))
        return
    console.print(f"[green]✓[/green] bench receipt → {path}")
    click.echo(eval_bench.render_report([{**receipt, "_file": path.name}]))


@bench_group.command(name="report")
@click.option("--last", type=click.IntRange(min=1), default=3, show_default=True)
@click.option("--dataset", default=None, help="Only runs of this dataset.")
@click.option(
    "--out",
    "out_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Write the markdown report to a file instead of stdout.",
)
def bench_report(last: int, dataset: str | None, out_path: Path | None) -> None:
    """Markdown comparison of the most recent bench runs."""
    cfg = Config.from_env()
    md = eval_bench.render_report(
        eval_bench.load_receipts(cfg.state_dir, last=last, dataset=dataset)
    )
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(md, encoding="utf-8")
        console.print(f"[green]✓[/green] report → {out_path}")
    else:
        click.echo(md)
