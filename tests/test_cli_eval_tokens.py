import json
from dataclasses import dataclass, field
from pathlib import Path

from click.testing import CliRunner

from memo import eval_tokens
from memo.cli import cli


@dataclass
class _RealishHit:
    """Mirrors the attributes render_recall_context reads off a real hit."""

    id: str
    title: str
    body: str
    score: float | None = 0.9
    tags: list = field(default_factory=list)


class _FakeMem:
    """Fake Memory whose search() mirrors the REAL Memory.search signature:

    keyword-only `limit`, no `k`, no **kwargs. If cli_eval's `_search`
    closure ever regresses to `mem.search(text, k=k)`, this raises
    TypeError and the test fails.
    """

    def search(self, text, *, limit):
        return [_RealishHit(id="aaaaaaaa11", title="t", body="b")][:limit]


def _env(tmp_path: Path) -> dict[str, str]:
    return {
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_DATA_DIR": str(tmp_path / "data"),
        "MEMO_STATE_DIR": str(tmp_path / "state"),
    }


def test_tokens_update_baseline_then_gate_pass(tmp_path, monkeypatch):
    canned = [eval_tokens.LeverRow("recall_format_compact", "recall_output", 100, 80, 1.0, 1.0)]
    monkeypatch.setattr(eval_tokens, "run_all", lambda **kw: canned)
    # Skip the live-index/labels wiring the stub doesn't need.
    monkeypatch.setattr("memo.cli_eval._get_memory", lambda cfg: object())

    runner = CliRunner()
    r1 = runner.invoke(cli, ["eval", "tokens", "--update-baseline"], env=_env(tmp_path))
    assert r1.exit_code == 0, r1.output
    baseline = tmp_path / "state" / "eval" / "token_baseline.json"
    assert baseline.exists()
    saved = json.loads(baseline.read_text())
    assert saved["recall_format_compact"]["passed"] is True

    r2 = runner.invoke(cli, ["eval", "tokens", "--gate"], env=_env(tmp_path))
    assert r2.exit_code == 0, r2.output


def test_tokens_gate_fails_on_regression(tmp_path, monkeypatch):
    monkeypatch.setattr("memo.cli_eval._get_memory", lambda cfg: object())
    good = [eval_tokens.LeverRow("recall_format_compact", "recall_output", 100, 80, 1.0, 1.0)]
    monkeypatch.setattr(eval_tokens, "run_all", lambda **kw: good)
    runner = CliRunner()
    runner.invoke(cli, ["eval", "tokens", "--update-baseline"], env=_env(tmp_path))

    bad = [eval_tokens.LeverRow("recall_format_compact", "recall_output", 100, 99, 1.0, 1.0)]
    monkeypatch.setattr(eval_tokens, "run_all", lambda **kw: bad)
    r = runner.invoke(cli, ["eval", "tokens", "--gate"], env=_env(tmp_path))
    assert r.exit_code == 1
    assert "FAIL" in r.output


def test_tokens_gate_without_baseline_errors(tmp_path, monkeypatch):
    monkeypatch.setattr("memo.cli_eval._get_memory", lambda cfg: object())
    monkeypatch.setattr(eval_tokens, "run_all", lambda **kw: [])
    r = CliRunner().invoke(cli, ["eval", "tokens", "--gate"], env=_env(tmp_path))
    assert r.exit_code != 0
    assert "baseline" in r.output.lower()


def test_tokens_cmd_drives_real_search_closure_with_limit_kwarg(tmp_path, monkeypatch):
    """Regression: cli_eval's `_search` closure must call mem.search(text,
    limit=k) — NOT mem.search(text, k=k). Memory.search's real signature is
    keyword-only `limit`, no `k`, no **kwargs, so the wrong keyword raises
    TypeError the moment a real search runs. Does NOT stub eval_tokens.run_all
    — this drives the actual `_search` closure against the committed
    eval/regression_labels.json + eval/token_corpus.json label/corpus files.
    """
    monkeypatch.setattr("memo.cli_eval._get_memory", lambda cfg: _FakeMem())
    # The real _get_memory()/Config.from_env() path creates state_dir before
    # eval_tokens_cmd runs; our fake bypasses that, so create it explicitly
    # (the P2 crusher path needs state_dir/crush_cache to exist).
    (tmp_path / "state").mkdir(parents=True, exist_ok=True)

    r = CliRunner().invoke(cli, ["eval", "tokens", "--update-baseline"], env=_env(tmp_path))
    assert r.exit_code == 0, r.output
    baseline = tmp_path / "state" / "eval" / "token_baseline.json"
    assert baseline.exists()


def _access_snapshot(mem) -> tuple[int, int, str | None]:
    row = mem.store._conn.execute(
        "SELECT COUNT(*), COALESCE(SUM(access_count), 0), MAX(last_accessed) FROM access"
    ).fetchone()
    return tuple(row)


def test_tokens_cmd_does_not_inflate_access_count(tmp_cfg, mem_with_stub, monkeypatch):
    """`memo eval tokens` runs the P1 recall-output levers against the live
    index (`_search` closure -> `mem.search`); without `_track_usage=False`
    threaded through, every search hit writes an access-log row
    (search_ops.py's `_stage_record_usage`), inflating access_count on
    whatever memory the eval surfaces — the same signal `memo usefulness` /
    `dead_weight()` read to decide what's noise.
    """
    mem = mem_with_stub
    rec = mem.save(
        content="deploy runbook for the lambda pipeline, step by step",
        title="Deploy runbook",
        auto_project=False,
    )
    monkeypatch.setattr("memo.cli_eval._get_memory", lambda cfg: mem)

    labels_path = tmp_cfg.state_dir / "labels.json"
    labels_path.write_text(
        json.dumps(
            {
                "schema": "memo.eval_recall.labels.v1",
                "prompts": [{"text": "deploy runbook lambda", "relevant": True}],
            }
        ),
        encoding="utf-8",
    )
    corpus_path = tmp_cfg.state_dir / "corpus.json"
    corpus_path.write_text(
        json.dumps({"schema": "memo.token_corpus.v1", "cases": []}), encoding="utf-8"
    )

    before = _access_snapshot(mem)
    r = CliRunner().invoke(
        cli,
        [
            "eval",
            "tokens",
            "--labels",
            str(labels_path),
            "--corpus",
            str(corpus_path),
            "--update-baseline",
        ],
        env=_env(tmp_cfg.data_dir.parent),
    )
    assert r.exit_code == 0, r.output
    assert _access_snapshot(mem) == before

    # Contrast: a real (non-eval) search against the same memory DOES bump
    # it — proving the assertion above isn't vacuously true because the
    # gate never actually matched a candidate.
    hits = mem.search("deploy runbook lambda")
    assert any(h.id == rec.id for h in hits)
    assert _access_snapshot(mem) != before
