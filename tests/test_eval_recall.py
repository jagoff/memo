"""Recall-quality eval — pure classification/loading + an isolated end-to-end run.

The end-to-end test uses the stubbed-embedder `mock_memory` fixture, so search
scores are deterministic-but-arbitrary; it asserts structure and value ranges,
not specific precision numbers (those only mean something on a real corpus).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from click.testing import CliRunner

from memo import eval_recall
from memo.cli import cli
from memo.eval_recall import LabelSet, Prompt


def _rec(**kw) -> SimpleNamespace:
    base = {"id": "a" * 32, "title": "", "tags": [], "path": "", "body": "", "score": 0.9}
    base.update(kw)
    return SimpleNamespace(**base)


# --- classification ---------------------------------------------------------


def test_is_noise_by_tag_and_path():
    labels = LabelSet(prompts=[], noise_tags={"04-archive"},
                      noise_path_fragments=("/old/",))
    assert eval_recall._is_noise(_rec(tags=["04-archive"]), labels)
    assert eval_recall._is_noise(_rec(path="memory/old/x.md"), labels)
    assert not eval_recall._is_noise(_rec(tags=["work"], path="memory/x.md"), labels)


def test_is_relevant_by_terms_excludes_noise():
    labels = LabelSet(prompts=[], relevant_terms={"synapse"}, noise_tags={"04-archive"})
    p = Prompt("q", relevant=True)
    assert eval_recall._is_relevant(_rec(title="synapse stack"), p, labels)
    # term present but record is noise → not relevant
    assert not eval_recall._is_relevant(
        _rec(title="synapse", tags=["04-archive"]), p, labels)
    assert not eval_recall._is_relevant(_rec(title="unrelated"), p, labels)


def test_is_relevant_by_expect_ids_prefix_match():
    labels = LabelSet(prompts=[])
    p = Prompt("q", expect_ids=["deadbeefdead"])
    assert eval_recall._is_relevant(_rec(id="deadbeefdead0000"), p, labels)
    assert not eval_recall._is_relevant(_rec(id="ffffffffffff"), p, labels)
    # expect_ids takes precedence over term heuristic
    assert not eval_recall._is_relevant(
        _rec(id="0000", title="synapse"), Prompt("q", expect_ids=["deadbeefdead"]), labels)


def test_id_matches_requires_8_char_floor():
    assert not eval_recall._id_matches("abc", ["abcd"])  # too short to prefix-match
    assert eval_recall._id_matches("abcdef12", ["abcdef12"])
    assert eval_recall._id_matches("abcdef1234", ["abcdef12"])


# --- label loading -----------------------------------------------------------


def test_load_labels_roundtrip(tmp_path: Path):
    p = tmp_path / "labels.json"
    p.write_text(json.dumps({
        "session_context": "ctx",
        "relevant_terms": ["Synapse"],
        "noise_tags": ["04-Archive"],
        "noise_path_fragments": ["/old/"],
        "prompts": [
            {"text": "where is the stack", "relevant": True, "expect_ids": ["abc12345"]},
            "a bare string prompt",
        ],
    }), encoding="utf-8")
    labels = eval_recall.load_labels(p)
    assert labels.session_context == "ctx"
    assert labels.relevant_terms == {"synapse"}  # lowercased
    assert labels.noise_tags == {"04-archive"}
    assert len(labels.prompts) == 2
    assert labels.prompts[0].relevant and labels.prompts[0].expect_ids == ["abc12345"]
    assert labels.prompts[1].relevant is False  # bare string defaults to a probe


def test_load_labels_rejects_malformed(tmp_path: Path):
    bad = tmp_path / "bad.json"
    bad.write_text("{ not json", encoding="utf-8")
    with pytest.raises(ValueError):
        eval_recall.load_labels(bad)

    no_prompts = tmp_path / "empty.json"
    no_prompts.write_text(json.dumps({"prompts": []}), encoding="utf-8")
    with pytest.raises(ValueError):
        eval_recall.load_labels(no_prompts)


def test_fingerprint_is_stable_and_content_sensitive():
    a = LabelSet(prompts=[Prompt("x", relevant=True)], relevant_terms={"t"})
    b = LabelSet(prompts=[Prompt("x", relevant=True)], relevant_terms={"t"})
    c = LabelSet(prompts=[Prompt("y", relevant=True)], relevant_terms={"t"})
    assert a.fingerprint() == b.fingerprint()
    assert a.fingerprint() != c.fingerprint()


# --- recommendation ----------------------------------------------------------


def test_recommend_prefers_baseline_when_it_wins():
    rows = [
        eval_recall.Row(config="A vec/0.60/keep", precision_at_k=0.9, noise_at_k=0.1),
        eval_recall.Row(config="B vec/0.72/excl", precision_at_k=0.5, noise_at_k=0.0),
    ]
    assert "Baseline" in eval_recall.recommend(rows)


def test_recommend_maps_winner_to_knobs():
    rows = [
        eval_recall.Row(config="A vec/0.60/keep", precision_at_k=0.4, noise_at_k=0.3),
        eval_recall.Row(config="C hyb/0.40/excl", precision_at_k=0.8, noise_at_k=0.1),
    ]
    out = eval_recall.recommend(rows)
    assert "C hyb/0.40/excl" in out
    assert "MEMO_RECALL_MODE=hybrid" in out
    assert "MEMO_RECALL_MIN_SIM=0.4" in out


def test_recommend_warns_when_winner_blows_hook_budget():
    rows = [
        eval_recall.Row(config="A vec/0.60/keep", precision_at_k=0.6, noise_at_k=0.4, latency_ms_p50=120),
        eval_recall.Row(config="D hyb/0.40/ctx", precision_at_k=1.0, noise_at_k=0.0, latency_ms_p50=14000),
    ]
    out = eval_recall.recommend(rows)
    assert "D hyb/0.40/ctx" in out
    assert "recall-hook budget" in out
    # a fast winner gets no latency warning
    fast = [
        eval_recall.Row(config="A vec/0.60/keep", precision_at_k=0.6, noise_at_k=0.4, latency_ms_p50=120),
        eval_recall.Row(config="B vec/0.72/excl", precision_at_k=0.9, noise_at_k=0.0, latency_ms_p50=130),
    ]
    assert "recall-hook budget" not in eval_recall.recommend(fast)


# --- end-to-end (isolated, stubbed embedder) --------------------------------


def test_evaluate_returns_one_row_per_config_in_range(mock_memory):
    mock_memory.save(content="synapse memflow memo stack architecture", title="Stack", tags=["stack"])
    mock_memory.save(content="old archived HR note", title="HR", tags=["04-archive"])

    labels = LabelSet(
        prompts=[Prompt("how is the stack architected", relevant=True),
                 Prompt("apple pie recipe", relevant=False)],
        relevant_terms={"synapse", "stack", "memo"},
        noise_tags={"04-archive"},
    )
    rows = eval_recall.evaluate(mock_memory, k=3, labels=labels)
    assert len(rows) == len(eval_recall.default_configs())
    for r in rows:
        assert 0.0 <= r.precision_at_k <= 1.0
        assert 0.0 <= r.noise_at_k <= 1.0
        assert r.latency_ms_p50 >= 0.0
        assert len(r.detail) == 2  # one entry per prompt
    # recommendation is a non-empty string referencing a known config or baseline
    assert eval_recall.recommend(rows)


# --- CLI layer (no MLX: bad --labels fails before Memory is built) ----------


def _env(tmp_path: Path) -> dict[str, str]:
    return {
        "MEMO_CONFIG_FILE": str(tmp_path / "memo-config.toml"),
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_DATA_DIR": str(tmp_path / "data"),
        "MEMO_STATE_DIR": str(tmp_path / "state"),
    }


def test_cli_eval_recall_help_lists_options():
    result = CliRunner().invoke(cli, ["eval", "recall", "--help"])
    assert result.exit_code == 0, result.output
    assert "--labels" in result.output
    assert "--force" in result.output


def test_cli_eval_recall_rejects_malformed_labels(tmp_path: Path):
    bad = tmp_path / "bad.json"
    bad.write_text("{ not json", encoding="utf-8")
    result = CliRunner().invoke(
        cli, ["eval", "recall", "--labels", str(bad)], env=_env(tmp_path))
    assert result.exit_code != 0
    assert "bad.json" in result.output
