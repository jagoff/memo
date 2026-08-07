import json
from types import SimpleNamespace

from click.testing import CliRunner

from memo import eval_recall
from memo.cli_eval import eval_group


def test_eval_memory_json_and_text(tmp_path, monkeypatch):
    labels = SimpleNamespace(
        prompts=[SimpleNamespace(text="where is memo")],
        fingerprint=lambda: "labels",
    )
    row = SimpleNamespace(
        config="A",
        precision_at_k=1.0,
        recall_at_k=1.0,
        ndcg_at_k=1.0,
        mrr=1.0,
        noise_at_k=0.0,
        stale_at_k=0.0,
        canonical_hit_at_k=1.0,
        latency_ms_p50=1.0,
        graph_recall_gain=0.0,
        graph_noise_rate=0.0,
        graph_explanation_coverage=1.0,
    )

    class _Memory:
        def close(self):
            pass

    monkeypatch.setattr("memo.cli_eval._get_memory", lambda cfg: _Memory())
    monkeypatch.setattr("memo.eval_recall.load_labels", lambda path: labels)
    monkeypatch.setattr("memo.eval_recall.limit_label_set", lambda value, maximum: value)
    monkeypatch.setattr("memo.eval_recall.profile_configs", lambda profile: ["A"])
    monkeypatch.setattr("memo.eval_recall.evaluate", lambda *a, **k: [row])
    from click.testing import CliRunner

    runner = CliRunner()
    labels_path = tmp_path / "labels.json"
    labels_path.write_text("{}")
    result = runner.invoke(
        eval_group,
        ["memory", "--labels", str(labels_path), "--json"],
        env={"MEMO_DATA_DIR": str(tmp_path), "MEMO_STATE_DIR": str(tmp_path / "state")},
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["schema"] == "memo.eval.memory.v1"
    result = runner.invoke(eval_group, ["memory", "--labels", str(labels_path)])
    assert result.exit_code == 0, result.output
    assert "memory eval" in result.output


def test_eval_baseline_writes_snapshot(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMO_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("MEMO_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MEMO_NONINTERACTIVE", "1")
    # Keep the test MLX-free: stub the memory build and the labels + recall eval.
    monkeypatch.setattr("memo.cli_eval._get_memory", lambda cfg: object())
    monkeypatch.setattr(eval_recall, "load_labels", lambda path: object())
    monkeypatch.setattr(
        eval_recall,
        "evaluate",
        lambda mem, *, k, labels, configs, progress=None: [
            eval_recall.Row(config="A", precision_at_k=0.2, noise_at_k=0.0)
        ],
    )

    res = CliRunner().invoke(eval_group, ["baseline", "--labels", "x.json", "--json"])
    assert res.exit_code == 0, res.output

    snap = json.loads((tmp_path / "state" / "eval" / "baseline_snapshot.json").read_text())
    assert snap["schema"] == "memo.eval_baseline.v1"
    assert snap["offline"]["precision_at_k"] == 0.2
    assert snap["params_version"] == "base"


def test_expand_labels_cmd_writes_output(tmp_cfg, tmp_path, monkeypatch) -> None:
    import json

    from click.testing import CliRunner

    from memo.cli import cli

    src = tmp_path / "labels.json"
    src.write_text(
        json.dumps(
            {
                "schema": "memo.eval_recall.labels.v1",
                "prompts": [
                    {
                        "text": "cómo configuro el sync remoto?",
                        "relevant": True,
                        "expect_ids": ["aaaabbbb"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "expanded.json"

    class _StubChat:
        def chat(self, **kwargs):
            return {
                "message": {
                    "content": "1. cómo seteo el sync remoto?\n2. qué config lleva el sync remoto?"
                }
            }

    monkeypatch.setattr("memo.llm.MLXChat", lambda: _StubChat())
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["eval", "expand-labels", "--labels", str(src), "--out", str(out)],
        env={
            "MEMO_NONINTERACTIVE": "1",
            "MEMO_DATA_DIR": str(tmp_cfg.data_dir),
            "MEMO_STATE_DIR": str(tmp_cfg.state_dir),
        },
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    doc = json.loads(out.read_text(encoding="utf-8"))
    texts = [p["text"] for p in doc["prompts"]]
    assert "cómo seteo el sync remoto?" in texts
    assert all(p.get("expect_ids") == ["aaaabbbb"] for p in doc["prompts"])


def test_expand_labels_cmd_refuses_out_equals_labels(tmp_cfg, tmp_path) -> None:
    import json

    from click.testing import CliRunner

    from memo.cli import cli

    src = tmp_path / "labels.json"
    src.write_text(
        json.dumps(
            {
                "schema": "memo.eval_recall.labels.v1",
                "prompts": [
                    {
                        "text": "cómo configuro el sync remoto?",
                        "relevant": True,
                        "expect_ids": ["aaaabbbb"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["eval", "expand-labels", "--labels", str(src), "--out", str(src)],
        env={
            "MEMO_NONINTERACTIVE": "1",
            "MEMO_DATA_DIR": str(tmp_cfg.data_dir),
            "MEMO_STATE_DIR": str(tmp_cfg.state_dir),
        },
    )
    assert result.exit_code != 0
    assert "must differ" in result.output


def test_save_cache_drops_entries_past_their_ttl(tmp_path):
    """The TTL was only ever enforced on read, so every nightly A/B appended a
    key that never left. The file reached 211 MB on a live install — bigger than
    the index it measures. An entry past the TTL can never be served (the reader
    rejects it), so keeping it is pure waste."""
    import time

    from memo import cli_eval

    cfg = SimpleNamespace(state_dir=tmp_path)
    now = time.time()
    cli_eval._save_cache(
        cfg,
        {
            "fresh": {"ts": now, "rows": [1]},
            "expired": {"ts": now - cli_eval._CACHE_TTL_S - 1, "rows": [2]},
        },
    )

    on_disk = json.loads(cli_eval._cache_path(cfg).read_text(encoding="utf-8"))

    assert set(on_disk) == {"fresh"}


def test_save_cache_keeps_entries_it_cannot_date(tmp_path):
    """A malformed entry is not evidence of staleness — dropping it silently
    would turn an unrelated write bug into cache loss."""
    from memo import cli_eval

    cfg = SimpleNamespace(state_dir=tmp_path)
    cli_eval._save_cache(cfg, {"weird": "not-a-dict"})

    on_disk = json.loads(cli_eval._cache_path(cfg).read_text(encoding="utf-8"))

    assert set(on_disk) == {"weird"}
