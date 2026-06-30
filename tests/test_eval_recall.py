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


def test_select_configs_accepts_short_names_and_exact_names():
    rows = eval_recall.select_configs(["A", "D hyb/0.40/ctx"])

    assert [c.name for c in rows] == ["A vec/0.60/keep", "D hyb/0.40/ctx"]


def test_select_configs_quick_uses_one_fast_config():
    rows = eval_recall.select_configs(quick=True)

    assert [c.name for c in rows] == ["A vec/0.60/keep"]


def test_select_configs_rejects_unknown_name():
    with pytest.raises(ValueError, match="unknown recall eval config"):
        eval_recall.select_configs(["Z"])


def test_limit_label_set_keeps_metadata_and_slices_prompts():
    labels = LabelSet(
        prompts=[Prompt("a"), Prompt("b"), Prompt("c")],
        relevant_terms={"memo"},
        noise_tags={"old"},
        noise_path_fragments=("/old/",),
        session_context="ctx",
    )

    limited = eval_recall.limit_label_set(labels, 2)

    assert [p.text for p in limited.prompts] == ["a", "b"]
    assert limited.relevant_terms == labels.relevant_terms
    assert limited.noise_tags == labels.noise_tags
    assert limited.noise_path_fragments == labels.noise_path_fragments
    assert limited.session_context == labels.session_context


def test_run_config_reports_progress_per_prompt():
    calls: list[tuple[str, int, int]] = []
    labels = LabelSet(prompts=[Prompt("a"), Prompt("b")])
    mem = SimpleNamespace(search=lambda *_args, **_kwargs: [])

    eval_recall.run_config(
        mem,
        eval_recall.default_configs()[0],
        1,
        labels,
        progress=lambda cfg, index, total: calls.append((cfg.name, index, total)),
    )

    assert calls == [("A vec/0.60/keep", 1, 2), ("A vec/0.60/keep", 2, 2)]


# --- regression gate ---------------------------------------------------------


def _rows(*pairs):
    return [
        eval_recall.Row(config=f"cfg{i}", precision_at_k=p, noise_at_k=n)
        for i, (p, n) in enumerate(pairs)
    ]


def test_best_row_picks_highest_precision_then_lowest_noise():
    rows = _rows((0.4, 0.0), (0.8, 0.2), (0.8, 0.1))
    assert eval_recall.best_row(rows).config == "cfg2"  # 0.8 prec, lower noise


def test_gate_metrics_returns_best_pair():
    rows = _rows((0.4, 0.3), (0.9, 0.1))
    assert eval_recall.gate_metrics(rows) == {"precision_at_k": 0.9, "noise_at_k": 0.1}


def test_check_gate_passes_when_metrics_hold():
    rows = _rows((0.6, 0.1))
    res = eval_recall.check_gate(rows, {"precision_at_k": 0.6, "noise_at_k": 0.1})
    assert res.passed
    assert "PASS" in res.message


def test_check_gate_passes_when_metrics_improve():
    rows = _rows((0.8, 0.0))
    res = eval_recall.check_gate(rows, {"precision_at_k": 0.6, "noise_at_k": 0.1})
    assert res.passed


def test_check_gate_fails_on_precision_drop():
    rows = _rows((0.5, 0.1))
    res = eval_recall.check_gate(rows, {"precision_at_k": 0.6, "noise_at_k": 0.1})
    assert not res.passed
    assert "precision@k" in res.message


def test_check_gate_fails_on_noise_rise():
    rows = _rows((0.6, 0.3))
    res = eval_recall.check_gate(rows, {"precision_at_k": 0.6, "noise_at_k": 0.1})
    assert not res.passed
    assert "noise@k" in res.message


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
    assert "--quick" in result.output
    assert "--config" in result.output
    assert "--max-prompts" in result.output
    assert "--progress" in result.output
    assert "--gate" in result.output
    assert "--update-baseline" in result.output


def test_cli_eval_recall_rejects_malformed_labels(tmp_path: Path):
    bad = tmp_path / "bad.json"
    bad.write_text("{ not json", encoding="utf-8")
    result = CliRunner().invoke(
        cli, ["eval", "recall", "--labels", str(bad)], env=_env(tmp_path))
    assert result.exit_code != 0
    assert "bad.json" in result.output


def test_cli_eval_recall_rejects_unknown_config(tmp_path: Path):
    labels = tmp_path / "labels.json"
    labels.write_text(
        json.dumps({"prompts": [{"text": "where is memo", "relevant": True}]}),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        cli,
        ["eval", "recall", "--labels", str(labels), "--config", "Z"],
        env=_env(tmp_path),
    )

    assert result.exit_code != 0
    assert "unknown recall eval config" in result.output


# --- harvest labels from grounding.log --------------------------------------


def test_harvest_labels_joins_grounding_to_prompt(tmp_path: Path):
    from memo.dashboard import append_grounding_log, append_recall_log

    sd = tmp_path / "state"
    sd.mkdir(parents=True, exist_ok=True)
    # The prompt for (session, turn) lives in recall_hook.log (written when
    # append_recall_log gets a session_id).
    append_recall_log(
        sd,
        prompt="cómo configuro el daemon de recall warm",
        hits=[{"id": "deadbeef" + "0" * 24, "score": 0.9, "title": "T"}],
        session_id="s1",
        turn=3,
    )
    # A strongly-grounded row → becomes ground truth.
    append_grounding_log(
        sd, session_id="s1", turn=3, recall_id="deadbeef", used_score=0.88, method="both"
    )
    # A weakly-used row → filtered out (below --strong).
    append_grounding_log(
        sd, session_id="s1", turn=3, recall_id="cafe1234", used_score=0.10, method="lexical"
    )

    labels = eval_recall.harvest_labels(sd, strong=0.5)
    assert len(labels) == 1
    assert labels[0]["text"] == "cómo configuro el daemon de recall warm"
    assert labels[0]["relevant"] is True
    assert labels[0]["expect_ids"] == ["deadbeef"]


def test_harvest_labels_skips_rows_without_prompt(tmp_path: Path):
    from memo.dashboard import append_grounding_log

    sd = tmp_path / "state"
    sd.mkdir(parents=True, exist_ok=True)
    # Grounding row with no matching recall_hook.log prompt → skipped.
    append_grounding_log(
        sd, session_id="orphan", turn=1, recall_id="aaaa1111", used_score=0.9, method="both"
    )
    assert eval_recall.harvest_labels(sd, strong=0.5) == []


def test_merge_label_prompts_unions_expect_ids(tmp_path: Path):
    existing = [{"text": "cómo configuro el daemon de recall", "relevant": True, "expect_ids": ["aaaa1111"]}]
    harvested = [{"text": "cómo configuro el daemon de recall warm", "relevant": True, "expect_ids": ["bbbb2222"]}]
    merged = eval_recall.merge_label_prompts(existing, harvested)
    # Jaccard-similar → single entry with unioned ids, not a duplicate.
    assert len(merged) == 1
    assert set(merged[0]["expect_ids"]) == {"aaaa1111", "bbbb2222"}


# --- associative-recall eval labels ------------------------------------------


def test_label_parses_expect_associative_ids():
    from memo.eval_recall import Label, _label_from_dict  # parser helper

    lab = _label_from_dict({
        "prompt": "how does recall connect to the store?",
        "relevant": True,
        "expect_associative_ids": ["abcd1234", "ef567890"],
    })
    assert isinstance(lab, Label)
    assert lab.expect_associative_ids == ("abcd1234", "ef567890")
