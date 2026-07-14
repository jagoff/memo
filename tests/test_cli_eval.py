import json

from click.testing import CliRunner

from memo import eval_recall
from memo.cli_eval import eval_group


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
