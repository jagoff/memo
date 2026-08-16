"""`eval memory` and `eval recall` must not share a gate baseline.

They measure different pipelines. The file was shared and only `eval recall`
ever wrote it, so `eval memory --gate` compared its own numbers against a
baseline recorded from recall's — and `check_gate`'s k guard only catches that
when the two happen to run at different top-K. At equal k the comparison looked
healthy and meant nothing.
"""

from __future__ import annotations

import json
from pathlib import Path

import click
import pytest

from memo import cli_eval
from memo.cli_eval import (
    GATE_MEMORY,
    GATE_RECALL,
    _baseline_path,
    _reject_foreign_baseline,
)


class _Cfg:
    def __init__(self, state_dir: Path) -> None:
        self.state_dir = state_dir


def test_each_command_gets_its_own_baseline_file(tmp_path: Path) -> None:
    cfg = _Cfg(tmp_path)

    assert _baseline_path(cfg, GATE_RECALL) != _baseline_path(cfg, GATE_MEMORY)


def test_recall_keeps_the_historical_filename(tmp_path: Path) -> None:
    """Renaming it would silently invalidate every existing machine's baseline
    and force a re-seed on upgrade."""
    assert _baseline_path(_Cfg(tmp_path), GATE_RECALL).name == "recall_baseline.json"


def test_default_command_is_recall(tmp_path: Path) -> None:
    cfg = _Cfg(tmp_path)

    assert _baseline_path(cfg) == _baseline_path(cfg, GATE_RECALL)


def test_a_baseline_written_by_the_other_command_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "recall_baseline.json"

    with pytest.raises(click.ClickException, match="was written by `memo eval recall`"):
        _reject_foreign_baseline({"gate_command": GATE_RECALL}, GATE_MEMORY, path)


def test_a_baseline_from_the_same_command_is_accepted(tmp_path: Path) -> None:
    _reject_foreign_baseline({"gate_command": GATE_MEMORY}, GATE_MEMORY, tmp_path / "b.json")
    _reject_foreign_baseline({"gate_command": GATE_RECALL}, GATE_RECALL, tmp_path / "b.json")


def test_an_unstamped_baseline_counts_as_recall(tmp_path: Path) -> None:
    """Files written before the split only ever came from `eval recall`, so
    treating them as recall keeps existing baselines valid across the upgrade
    instead of forcing everyone to re-seed."""
    _reject_foreign_baseline({"precision_at_k": 0.6}, GATE_RECALL, tmp_path / "b.json")

    with pytest.raises(click.ClickException, match="was written by `memo eval recall`"):
        _reject_foreign_baseline({"precision_at_k": 0.6}, GATE_MEMORY, tmp_path / "b.json")


def _labels_file(tmp_path: Path) -> Path:
    path = tmp_path / "labels.json"
    path.write_text(json.dumps({"prompts": [{"text": "where is memo", "relevant": True}]}), "utf-8")
    return path


def _stub_eval(monkeypatch: pytest.MonkeyPatch) -> None:
    from memo import eval_recall

    labels = eval_recall.LabelSet(prompts=[eval_recall.Prompt("where is memo", relevant=True)])

    class _Mem:
        def close(self) -> None:  # eval memory closes the store before scoring
            pass

    monkeypatch.setattr(cli_eval, "_get_memory", lambda cfg: _Mem())
    monkeypatch.setattr(eval_recall, "load_labels", lambda path: labels)
    monkeypatch.setattr(eval_recall, "fingerprint_corpus", lambda mem: "corpus")
    monkeypatch.setattr(
        eval_recall,
        "evaluate",
        lambda mem, *, k, labels, configs, progress=None: [
            eval_recall.Row(config=configs[0].name, precision_at_k=1.0, noise_at_k=0.0)
        ],
    )


def _env(tmp_path: Path) -> dict[str, str]:
    return {
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_DATA_DIR": str(tmp_path / "data"),
        "MEMO_STATE_DIR": str(tmp_path / "state"),
    }


def test_each_command_writes_only_its_own_file_and_stamps_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real end-to-end check: seeding from one command must not produce or
    touch the other's baseline, and the file must say who wrote it."""
    from click.testing import CliRunner

    from memo.cli import cli

    _stub_eval(monkeypatch)
    labels = _labels_file(tmp_path)
    env = _env(tmp_path)
    eval_dir = tmp_path / "state" / "eval"

    for command, expected_file in (
        (GATE_RECALL, "recall_baseline.json"),
        (GATE_MEMORY, "memory_baseline.json"),
    ):
        for stale in eval_dir.glob("*_baseline.json"):
            stale.unlink()
        result = CliRunner().invoke(
            cli,
            ["eval", command, "--labels", str(labels), "--update-baseline"]
            + (["--force"] if command == GATE_RECALL else []),
            env=env,
        )

        assert result.exit_code == 0, result.output
        written = sorted(p.name for p in eval_dir.glob("*_baseline.json"))
        assert written == [expected_file], f"{command} wrote {written}"
        payload = json.loads((eval_dir / expected_file).read_text(encoding="utf-8"))
        assert payload["gate_command"] == command


def test_eval_memory_exposes_update_baseline() -> None:
    """Refusing the foreign baseline is only safe because `eval memory` can now
    seed its own; otherwise its gate would have no remedy at all."""
    names = {p.name for p in cli_eval.eval_memory_cmd.params}

    assert "update_baseline" in names
    assert "gate" in names


def test_gate_payload_round_trips_the_stamp(tmp_path: Path) -> None:
    path = tmp_path / "memory_baseline.json"
    path.write_text(json.dumps({"gate_command": GATE_MEMORY, "precision_at_k": 0.5}), "utf-8")

    loaded = json.loads(path.read_text(encoding="utf-8"))

    _reject_foreign_baseline(loaded, GATE_MEMORY, path)
    with pytest.raises(click.ClickException):
        _reject_foreign_baseline(loaded, GATE_RECALL, path)
