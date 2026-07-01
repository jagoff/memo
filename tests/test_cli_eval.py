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
