"""`memo eval bench` CLI — isolated end-to-end run over a tiny inline dataset.

Network-free (dataset via --file), MLX-free (stub embedder + --retrieval-only)."""

from __future__ import annotations

import hashlib
import json

from click.testing import CliRunner

from memo.cli import cli

LOCOMO_TINY = [
    {
        "sample_id": "conv-1",
        "conversation": {
            "session_1_date_time": "1:56 pm on 8 May, 2023",
            "session_1": [
                {"speaker": "Caroline", "dia_id": "D1:1", "text": "I adopted a dog named Rex."},
                {"speaker": "Melanie", "dia_id": "D1:2", "text": "Congrats on the new dog!"},
            ],
        },
        "qa": [
            {
                "question": "What is the dog's name?",
                "answer": "Rex",
                "evidence": ["D1:1"],
                "category": 4,
            },
        ],
    }
]


def _stub_embed(self, inputs):
    out = []
    for s in inputs:
        h = hashlib.sha256((s or "").encode("utf-8")).digest()
        v = [((h[j] / 255.0) * 2.0) - 1.0 for j in range(4)]
        n = sum(x * x for x in v) ** 0.5
        out.append([x / n for x in v])
    return out


def test_bench_run_retrieval_only_isolated(tmp_path, monkeypatch):
    monkeypatch.setattr("memo.embedder.MLXEmbedder.embed", _stub_embed)
    data = tmp_path / "data"
    state = tmp_path / "state"
    data.mkdir()
    state.mkdir()
    dataset_file = tmp_path / "locomo-tiny.json"
    dataset_file.write_text(json.dumps(LOCOMO_TINY), encoding="utf-8")
    env = {
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_DATA_DIR": str(data),
        "MEMO_STATE_DIR": str(state),
        "MEMO_EMBEDDER_DIMS": "4",  # pin dims to the 4-dim stub (house rule)
        "MEMO_RERANKER_ENABLED": "0",
    }
    r = CliRunner().invoke(
        cli,
        [
            "eval",
            "bench",
            "run",
            "--dataset",
            "locomo",
            "--file",
            str(dataset_file),
            "--retrieval-only",
            "--k",
            "3",
            "--json",
        ],
        env=env,
    )
    assert r.exit_code == 0, r.output
    receipt = json.loads(r.output)
    assert receipt["schema"] == "memo.eval_bench.receipt.v1"
    assert receipt["dataset"] == "locomo"
    assert receipt["judge"] is None
    assert receipt["qa"] == {}
    assert "single_hop" in receipt["retrieval"]
    assert receipt["retrieval"]["single_hop"]["n_questions"] == 1
    # ISOLATION: no memory .md ever lands in the live data_dir
    assert list(data.rglob("*.md")) == []
    # the bench store + receipt live under state_dir/bench/
    assert (state / "bench" / "stores" / "locomo" / "conv-1" / "manifest.json").exists()
    assert list((state / "bench" / "runs").glob("locomo-*.json"))


def test_bench_report_renders_markdown(tmp_path):
    state = tmp_path / "state"
    runs = state / "bench" / "runs"
    runs.mkdir(parents=True)
    (tmp_path / "data").mkdir()
    receipt = {
        "schema": "memo.eval_bench.receipt.v1",
        "ts": "2026-07-03T12:00:00",
        "dataset": "locomo",
        "k": 5,
        "retrieval": {
            "single_hop": {
                "recall_at_k": 0.8,
                "ndcg_at_k": 0.7,
                "mrr": 0.6,
                "precision_at_k": 0.3,
                "n_questions": 10,
            }
        },
        "qa": {"single_hop": {"accuracy": 0.5, "n_questions": 10}},
    }
    (runs / "locomo-20260703-000000.json").write_text(json.dumps(receipt), encoding="utf-8")
    r = CliRunner().invoke(
        cli,
        ["eval", "bench", "report"],
        env={
            "MEMO_NONINTERACTIVE": "1",
            "MEMO_STATE_DIR": str(state),
            "MEMO_DATA_DIR": str(tmp_path / "data"),
        },
    )
    assert r.exit_code == 0, r.output
    assert "retrieval/single_hop/recall_at_k" in r.output
    assert "qa/single_hop/accuracy" in r.output


def test_bench_run_accepts_regime_option():
    res = CliRunner().invoke(cli, ["eval", "bench", "run", "--help"])
    assert res.exit_code == 0
    assert "--regime" in res.output
    assert "oracle" in res.output
