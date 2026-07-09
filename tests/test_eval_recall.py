"""Recall-quality eval — pure classification/loading + an isolated end-to-end run.

The end-to-end test uses the stubbed-embedder `mock_memory` fixture, so search
scores are deterministic-but-arbitrary; it asserts structure and value ranges,
not specific precision numbers (those only mean something on a real corpus).
"""

from __future__ import annotations

import json
import math
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
    labels = LabelSet(prompts=[], noise_tags={"04-archive"}, noise_path_fragments=("/old/",))
    assert eval_recall._is_noise(_rec(tags=["04-archive"]), labels)
    assert eval_recall._is_noise(_rec(path="memory/old/x.md"), labels)
    assert not eval_recall._is_noise(_rec(tags=["work"], path="memory/x.md"), labels)


def test_is_relevant_by_terms_excludes_noise():
    labels = LabelSet(prompts=[], relevant_terms={"synapse"}, noise_tags={"04-archive"})
    p = Prompt("q", relevant=True)
    assert eval_recall._is_relevant(_rec(title="synapse stack"), p, labels)
    # term present but record is noise → not relevant
    assert not eval_recall._is_relevant(_rec(title="synapse", tags=["04-archive"]), p, labels)
    assert not eval_recall._is_relevant(_rec(title="unrelated"), p, labels)


def test_is_relevant_by_expect_ids_prefix_match():
    labels = LabelSet(prompts=[])
    p = Prompt("q", expect_ids=["deadbeefdead"])
    assert eval_recall._is_relevant(_rec(id="deadbeefdead0000"), p, labels)
    assert not eval_recall._is_relevant(_rec(id="ffffffffffff"), p, labels)
    # expect_ids takes precedence over term heuristic
    assert not eval_recall._is_relevant(
        _rec(id="0000", title="synapse"), Prompt("q", expect_ids=["deadbeefdead"]), labels
    )


def test_id_matches_requires_8_char_floor():
    assert not eval_recall._id_matches("abc", ["abcd"])  # too short to prefix-match
    assert eval_recall._id_matches("abcdef12", ["abcdef12"])
    assert eval_recall._id_matches("abcdef1234", ["abcdef12"])


# --- label loading -----------------------------------------------------------


def test_load_labels_roundtrip(tmp_path: Path):
    p = tmp_path / "labels.json"
    p.write_text(
        json.dumps(
            {
                "session_context": "ctx",
                "relevant_terms": ["Synapse"],
                "noise_tags": ["04-Archive"],
                "noise_path_fragments": ["/old/"],
                "prompts": [
                    {"text": "where is the stack", "relevant": True, "expect_ids": ["abc12345"]},
                    "a bare string prompt",
                ],
            }
        ),
        encoding="utf-8",
    )
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
        eval_recall.Row(
            config="A vec/0.60/keep", precision_at_k=0.6, noise_at_k=0.4, latency_ms_p50=120
        ),
        eval_recall.Row(
            config="D hyb/0.40/ctx", precision_at_k=1.0, noise_at_k=0.0, latency_ms_p50=14000
        ),
    ]
    out = eval_recall.recommend(rows)
    assert "D hyb/0.40/ctx" in out
    assert "recall-hook budget" in out
    # a fast winner gets no latency warning
    fast = [
        eval_recall.Row(
            config="A vec/0.60/keep", precision_at_k=0.6, noise_at_k=0.4, latency_ms_p50=120
        ),
        eval_recall.Row(
            config="B vec/0.72/excl", precision_at_k=0.9, noise_at_k=0.0, latency_ms_p50=130
        ),
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


def test_profile_configs_name_eval_roles() -> None:
    assert [c.name for c in eval_recall.profile_configs("quick")] == ["A vec/0.60/keep"]
    assert [c.name for c in eval_recall.profile_configs("default")] == [
        "A vec/0.60/keep",
        "B vec/0.72/excl",
        "C hyb/0.40/excl",
        "D hyb/0.40/ctx",
    ]
    assert [c.name for c in eval_recall.profile_configs("pre-push")] == [
        "A vec/0.60/keep",
        "B vec/0.72/excl",
        "E mmr/0.3",
        "F mmr/0.5",
        "G mmr/0.7",
        "H synth/0.05",
        "I synth/0.10",
    ]
    assert [c.name for c in eval_recall.profile_configs("matrix")] == [
        "A vec/0.60/keep",
        "B vec/0.72/excl",
        "C hyb/0.40/excl",
        "D hyb/0.40/ctx",
        "E mmr/0.3",
        "F mmr/0.5",
        "G mmr/0.7",
        "H synth/0.05",
        "I synth/0.10",
    ]
    assert [c.name for c in eval_recall.profile_configs("expensive")] == ["J hyb/0.40/hyde"]


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
    assert eval_recall.gate_metrics(rows) == {
        "precision_at_k": 0.9,
        "noise_at_k": 0.1,
        "stale_at_k": 0.0,
        "canonical_hit_at_k": 0.0,
        "pack_answerability": None,
        "compaction_safety": None,
    }


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
    mock_memory.save(
        content="synapse memflow memo stack architecture", title="Stack", tags=["stack"]
    )
    mock_memory.save(content="old archived HR note", title="HR", tags=["04-archive"])

    labels = LabelSet(
        prompts=[
            Prompt("how is the stack architected", relevant=True),
            Prompt("apple pie recipe", relevant=False),
        ],
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
    assert "--profile" in result.output
    assert "--config" in result.output
    assert "A-J" in result.output
    assert "--max-prompts" in result.output
    assert "--progress" in result.output
    assert "--gate" in result.output
    assert "--update-baseline" in result.output


def test_cli_eval_recall_rejects_malformed_labels(tmp_path: Path):
    bad = tmp_path / "bad.json"
    bad.write_text("{ not json", encoding="utf-8")
    result = CliRunner().invoke(cli, ["eval", "recall", "--labels", str(bad)], env=_env(tmp_path))
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


def test_cli_eval_recall_fresh_human_run_prints_progress(tmp_path: Path, monkeypatch):
    labels_path = tmp_path / "labels.json"
    labels_path.write_text(
        json.dumps({"prompts": [{"text": "where is memo", "relevant": True}]}),
        encoding="utf-8",
    )
    labels = LabelSet(prompts=[Prompt("where is memo", relevant=True)])

    monkeypatch.setattr("memo.cli_eval._get_memory", lambda cfg: object())
    monkeypatch.setattr(eval_recall, "load_labels", lambda path: labels)
    monkeypatch.setattr(eval_recall, "fingerprint_corpus", lambda mem: "corpus")

    def _evaluate(mem, *, k, labels, configs, progress=None):
        assert progress is not None
        progress(configs[0], 1, 1)
        return [eval_recall.Row(config=configs[0].name, precision_at_k=1.0, noise_at_k=0.0)]

    monkeypatch.setattr(eval_recall, "evaluate", _evaluate)

    result = CliRunner().invoke(
        cli,
        ["eval", "recall", "--labels", str(labels_path), "--force", "--no-cache"],
        env=_env(tmp_path),
    )

    assert result.exit_code == 0, result.output
    assert "Running recall eval: 4 config(s) x 1 prompt(s) = 4 search(es)." in result.output
    assert "eval A vec/0.60/keep: prompt 1/1" in result.output


def test_cli_eval_recall_profile_pre_push_selects_named_subset(
    tmp_path: Path, monkeypatch
) -> None:
    labels_path = tmp_path / "labels.json"
    labels_path.write_text(
        json.dumps({"prompts": [{"text": "where is memo", "relevant": True}]}),
        encoding="utf-8",
    )
    labels = LabelSet(prompts=[Prompt("where is memo", relevant=True)])
    seen: list[str] = []

    monkeypatch.setattr("memo.cli_eval._get_memory", lambda cfg: object())
    monkeypatch.setattr(eval_recall, "load_labels", lambda path: labels)
    monkeypatch.setattr(eval_recall, "fingerprint_corpus", lambda mem: "corpus")

    def _evaluate(mem, *, k, labels, configs, progress=None):
        seen.extend(c.name for c in configs)
        return [eval_recall.Row(config=configs[0].name, precision_at_k=1.0, noise_at_k=0.0)]

    monkeypatch.setattr(eval_recall, "evaluate", _evaluate)

    result = CliRunner().invoke(
        cli,
        ["eval", "recall", "--labels", str(labels_path), "--profile", "pre-push", "--force"],
        env=_env(tmp_path),
    )

    assert result.exit_code == 0, result.output
    assert seen == [
        "A vec/0.60/keep",
        "B vec/0.72/excl",
        "E mmr/0.3",
        "F mmr/0.5",
        "G mmr/0.7",
        "H synth/0.05",
        "I synth/0.10",
    ]


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
    existing = [
        {"text": "cómo configuro el daemon de recall", "relevant": True, "expect_ids": ["aaaa1111"]}
    ]
    harvested = [
        {
            "text": "cómo configuro el daemon de recall warm",
            "relevant": True,
            "expect_ids": ["bbbb2222"],
        }
    ]
    merged = eval_recall.merge_label_prompts(existing, harvested)
    # Jaccard-similar → single entry with unioned ids, not a duplicate.
    assert len(merged) == 1
    assert set(merged[0]["expect_ids"]) == {"aaaa1111", "bbbb2222"}


# --- associative-recall eval labels ------------------------------------------


def test_label_parses_expect_associative_ids():
    from memo.eval_recall import Label, _label_from_dict  # parser helper

    lab = _label_from_dict(
        {
            "prompt": "how does recall connect to the store?",
            "relevant": True,
            "expect_associative_ids": ["abcd1234", "ef567890"],
        }
    )
    assert isinstance(lab, Label)
    assert lab.expect_associative_ids == ("abcd1234", "ef567890")


def test_run_config_uses_faithful_ranking_dedup() -> None:
    """run_config must rank via rank_hits — so the same memory surfaced twice is
    deduped the way the daemon does, not counted as two separate top-K hits."""
    from dataclasses import dataclass, field
    from typing import Any

    from memo.eval_recall import Cfg, LabelSet, Prompt, run_config

    @dataclass
    class _Hit:
        id: str
        score: float | None
        title: str = ""
        body: str = ""
        type: str = "note"
        tags: list[str] = field(default_factory=list)
        path: str = "p.md"
        extra: dict[str, Any] = field(default_factory=dict)

    class _Mem:
        def search(self, *a: Any, **k: Any) -> list[Any]:
            # id 'a' surfaced twice (same memory) collapses; 'b' is distinct.
            return [
                _Hit("a", 0.9, title="arch a", body="distinct body a long enough"),
                _Hit("a", 0.85, title="arch a", body="distinct body a long enough"),
                _Hit("b", 0.8, title="arch b", body="distinct body b long enough"),
            ]

    labels = LabelSet(prompts=[Prompt("q", relevant=True, expect_ids=["a"])])
    row = run_config(_Mem(), Cfg("X vec/0.0/keep", "vec", 0.0, exclude_archived=False), 3, labels)
    # 'a' (deduped to one) + 'b' = two top hits, not three
    assert len(row.detail[0]["top"]) == 2
    assert row.precision_at_k <= 1.0


# --- recall-faithful knobs (knobs_from_flags) in run_config ------------------


def _capture_run_config_knobs(monkeypatch, cfg, k: int = 4):
    """Run run_config with a stubbed rank_hits; return the knobs it was given."""
    import memo.recall_logic as rl

    captured: dict = {}

    def fake_rank_hits(hits, knobs, **kw):
        captured["knobs"] = knobs
        return []

    monkeypatch.setattr(rl, "rank_hits", fake_rank_hits)
    mem = SimpleNamespace(search=lambda *a, **kw: [])
    labels = LabelSet(prompts=[Prompt("q", relevant=True)])
    eval_recall.run_config(mem, cfg, k, labels)
    return captured["knobs"]


def test_run_config_pins_eval_fields_and_inherits_live_flags(monkeypatch):
    monkeypatch.setenv("MEMO_RECALL_MMR_LAMBDA", "0.4")
    monkeypatch.setenv("MEMO_RECALL_SYNTHESIS_BOOST", "0.07")
    cfg = eval_recall.Cfg("X vec/0.33/keep", "vec", 0.33, exclude_archived=False)

    knobs = _capture_run_config_knobs(monkeypatch, cfg, k=4)

    # eval-specific pins
    assert knobs.top_k == 4
    assert knobs.min_sim == 0.33
    assert knobs.min_body_chars == 0
    assert knobs.mode == "vec"
    assert knobs.project_tag is None  # labels carry no project yet
    # everything else inherits the LIVE flag/overlay resolution
    assert knobs.mmr_lambda == 0.4
    assert knobs.synthesis_boost == 0.07


def test_run_config_knob_overrides_beat_env(monkeypatch):
    monkeypatch.setenv("MEMO_RECALL_MMR_LAMBDA", "0.4")
    cfg = eval_recall.Cfg(
        "X vec/0.33/keep",
        "vec",
        0.33,
        exclude_archived=False,
        knob_overrides={"mmr_lambda": 0.7, "synthesis_boost": 0.2},
    )

    knobs = _capture_run_config_knobs(monkeypatch, cfg)

    assert knobs.mmr_lambda == 0.7  # override wins over env flag
    assert knobs.synthesis_boost == 0.2


# --- grid: MMR / synthesis variants ------------------------------------------


def test_tuning_configs_are_named_only_mmr_and_synth_variants():
    cfgs = eval_recall.default_configs()
    names = [c.name for c in cfgs]
    assert names == [
        "A vec/0.60/keep",
        "B vec/0.72/excl",
        "C hyb/0.40/excl",
        "D hyb/0.40/ctx",
    ]

    tuning = eval_recall.tuning_configs()
    by_name = {c.name: c for c in tuning}
    assert by_name["E mmr/0.3"].knob_overrides == {"mmr_lambda": 0.3}
    assert by_name["F mmr/0.5"].knob_overrides == {"mmr_lambda": 0.5}
    assert by_name["G mmr/0.7"].knob_overrides == {"mmr_lambda": 0.7}
    assert by_name["H synth/0.05"].knob_overrides == {"synthesis_boost": 0.05}
    assert by_name["I synth/0.10"].knob_overrides == {"synthesis_boost": 0.10}
    # variants mirror the A baseline so the delta is attributable to the knob
    for n in ("E mmr/0.3", "F mmr/0.5", "G mmr/0.7", "H synth/0.05", "I synth/0.10"):
        assert by_name[n].mode == "vec"
        assert by_name[n].floor == 0.60
        assert by_name[n].exclude_archived is False
        assert by_name[n].injection_fidelity is False


def test_select_configs_accepts_new_letters():
    rows = eval_recall.select_configs(["G", "I"])
    assert [c.name for c in rows] == ["G mmr/0.7", "I synth/0.10"]


def _redundant_mem():
    """Synthetic corpus: two near-duplicate top hits + one diverse relevant hit."""
    from dataclasses import dataclass as _dc
    from dataclasses import field as _field

    @_dc
    class _Hit:
        id: str
        score: float | None
        title: str = ""
        body: str = ""
        type: str = "note"
        tags: list = _field(default_factory=list)
        path: str = "p.md"
        extra: dict = _field(default_factory=dict)

    shared = "sqlite vec store thread local connections wal busy timeout float32 blobs"

    class _Mem:
        def search(self, *a, **kw):
            return [
                _Hit("aaaaaaaa", 0.90, title="vec store notes", body=f"{shared} variant alpha"),
                _Hit("bbbbbbbb", 0.85, title="vec store notes bis", body=f"{shared} variant beta"),
                _Hit(
                    "cccccccc",
                    0.80,
                    title="release workflow",
                    body="bump tag push five manifests changelog keepachangelog",
                ),
            ]

    return _Mem()


def test_mmr_config_differs_from_baseline_on_redundant_corpus():
    labels = LabelSet(prompts=[Prompt("q", relevant=True, expect_ids=["cccccccc"])])
    base = eval_recall.Cfg("base vec/0.0/keep", "vec", 0.0, exclude_archived=False)
    mmr = eval_recall.Cfg(
        "mmr vec/0.0/keep",
        "vec",
        0.0,
        exclude_archived=False,
        knob_overrides={"mmr_lambda": 0.7},
    )

    base_row = eval_recall.run_config(_redundant_mem(), base, 2, labels)
    mmr_row = eval_recall.run_config(_redundant_mem(), mmr, 2, labels)

    base_top = [t["title"] for t in base_row.detail[0]["top"]]
    mmr_top = [t["title"] for t in mmr_row.detail[0]["top"]]
    # baseline keeps the redundant near-duplicate; MMR promotes the diverse hit
    assert base_top == ["vec store notes", "vec store notes bis"]
    assert mmr_top == ["vec store notes", "release workflow"]
    assert mmr_row.precision_at_k > base_row.precision_at_k


# --- injection fidelity (hook's post-rank skip/gap filters) -------------------


def test_injection_fidelity_defaults_off_and_applies_when_on(monkeypatch):
    monkeypatch.setenv("MEMO_RECALL_SKIP_BELOW", "0.95")  # every hit scores below
    labels = LabelSet(prompts=[Prompt("q", relevant=True, expect_ids=["cccccccc"])])
    plain = eval_recall.Cfg("plain vec/0.0/keep", "vec", 0.0, exclude_archived=False)
    faithful = eval_recall.Cfg(
        "faith vec/0.0/keep", "vec", 0.0, exclude_archived=False, injection_fidelity=True
    )
    assert plain.injection_fidelity is False  # Cfg default is OFF

    plain_row = eval_recall.run_config(_redundant_mem(), plain, 2, labels)
    faithful_row = eval_recall.run_config(_redundant_mem(), faithful, 2, labels)

    assert len(plain_row.detail[0]["top"]) == 2  # default: filters NOT applied
    assert faithful_row.detail[0]["top"] == []  # skip-below floor wiped the injection


def test_injection_fidelity_gap_trims_to_top_hit(monkeypatch):
    monkeypatch.setenv("MEMO_RECALL_SKIP_BELOW", "0")
    monkeypatch.setenv("MEMO_RECALL_GAP_THRESHOLD", "0.04")  # 0.90 - 0.85 > 0.04
    labels = LabelSet(prompts=[Prompt("q", relevant=True, expect_ids=["aaaaaaaa"])])
    faithful = eval_recall.Cfg(
        "faith vec/0.0/keep", "vec", 0.0, exclude_archived=False, injection_fidelity=True
    )

    row = eval_recall.run_config(_redundant_mem(), faithful, 2, labels)

    assert [t["title"] for t in row.detail[0]["top"]] == ["vec store notes"]


# --- recommend maps knob_overrides to env exports -----------------------------


def test_recommend_maps_mmr_winner_to_knob_export():
    rows = [
        eval_recall.Row(config="A vec/0.60/keep", precision_at_k=0.4, noise_at_k=0.3),
        eval_recall.Row(config="F mmr/0.5", precision_at_k=0.8, noise_at_k=0.1),
    ]
    out = eval_recall.recommend(rows)
    assert "F mmr/0.5" in out
    assert "MEMO_RECALL_MODE=vec" in out
    assert "MEMO_RECALL_MIN_SIM=0.6" in out
    assert "MEMO_RECALL_MMR_LAMBDA=0.5" in out


# --- per-label project (Fase 2 reader side) -----------------------------------


def test_project_tag_for_normalizes_formats():
    # stored format (what current_project_tag produces) passes through verbatim
    assert eval_recall._project_tag_for("project:memo") == "project:memo"
    # bare hand-written name gets slugified + prefixed
    assert eval_recall._project_tag_for("My Repo") == "project:my-repo"
    assert eval_recall._project_tag_for(None) is None
    assert eval_recall._project_tag_for("   ") is None


def test_load_labels_parses_project_and_old_files_keep_working(tmp_path: Path):
    # New-style file: schema marker + one prompt carrying `project`.
    new = tmp_path / "new.json"
    new.write_text(
        json.dumps(
            {
                "schema": eval_recall.LABELS_SCHEMA,
                "prompts": [
                    {"text": "how does sync work", "relevant": True, "project": "project:memo"},
                    {"text": "where is the socket", "relevant": True},
                ],
            }
        ),
        encoding="utf-8",
    )
    labels = eval_recall.load_labels(new)
    assert labels.prompts[0].project == "project:memo"
    assert labels.prompts[1].project is None

    # Old-style file (no schema key, no project anywhere) still loads.
    old = tmp_path / "old.json"
    old.write_text(
        json.dumps({"prompts": [{"text": "where is the stack", "relevant": True}]}),
        encoding="utf-8",
    )
    old_labels = eval_recall.load_labels(old)
    assert len(old_labels.prompts) == 1
    assert old_labels.prompts[0].project is None


def test_fingerprint_sensitive_to_project():
    a = LabelSet(prompts=[Prompt("x", relevant=True)])
    b = LabelSet(prompts=[Prompt("x", relevant=True, project="project:memo")])
    c = LabelSet(prompts=[Prompt("x", relevant=True, project="project:memo")])
    assert a.fingerprint() != b.fingerprint()
    assert b.fingerprint() == c.fingerprint()


def test_harvest_labels_propagates_project(tmp_path: Path):
    from memo.dashboard import append_grounding_log, append_recall_log

    sd = tmp_path / "state"
    sd.mkdir(parents=True, exist_ok=True)
    append_recall_log(
        sd,
        prompt="cómo configuro el daemon de recall warm",
        hits=[{"id": "deadbeef" + "0" * 24, "score": 0.9, "title": "T"}],
        session_id="s1",
        turn=3,
    )
    append_recall_log(
        sd,
        prompt="dónde vive el socket del recall daemon",
        hits=[{"id": "cafe1234" + "0" * 24, "score": 0.9, "title": "S"}],
        session_id="s1",
        turn=5,
    )
    # Row with a project context → label carries it.
    append_grounding_log(
        sd,
        session_id="s1",
        turn=3,
        recall_id="deadbeef",
        used_score=0.9,
        method="both",
        project="project:memo",
    )
    # Row without one → key stays absent (schema-additive).
    append_grounding_log(
        sd, session_id="s1", turn=5, recall_id="cafe1234", used_score=0.9, method="both"
    )

    labels = eval_recall.harvest_labels(sd, strong=0.5)
    by_text = {lab["text"]: lab for lab in labels}
    assert by_text["cómo configuro el daemon de recall warm"]["project"] == "project:memo"
    assert "project" not in by_text["dónde vive el socket del recall daemon"]


def test_harvest_labels_cluster_project_first_non_null_wins(tmp_path: Path):
    from memo.dashboard import append_grounding_log, append_recall_log

    sd = tmp_path / "state"
    sd.mkdir(parents=True, exist_ok=True)
    append_recall_log(
        sd,
        prompt="cómo configuro el daemon de recall warm",
        hits=[{"id": "deadbeef" + "0" * 24, "score": 0.9, "title": "T"}],
        session_id="s1",
        turn=3,
    )
    # Same cluster: first row has no project, second sets it, third differs.
    append_grounding_log(
        sd, session_id="s1", turn=3, recall_id="deadbeef", used_score=0.9, method="both"
    )
    append_grounding_log(
        sd,
        session_id="s1",
        turn=3,
        recall_id="aaaa1111",
        used_score=0.9,
        method="both",
        project="project:memo",
    )
    append_grounding_log(
        sd,
        session_id="s1",
        turn=3,
        recall_id="bbbb2222",
        used_score=0.9,
        method="both",
        project="project:otro",
    )

    labels = eval_recall.harvest_labels(sd, strong=0.5)
    assert len(labels) == 1
    assert labels[0]["project"] == "project:memo"  # first non-null, later one doesn't clobber


def test_merge_label_prompts_project_first_non_null_wins():
    existing = [
        {
            "text": "cómo configuro el daemon de recall",
            "relevant": True,
            "expect_ids": ["aaaa1111"],
        },
        {
            "text": "dónde vive el socket del daemon",
            "relevant": True,
            "expect_ids": ["cccc3333"],
            "project": "project:memo",
        },
    ]
    harvested = [
        {
            "text": "cómo configuro el daemon de recall warm",
            "relevant": True,
            "expect_ids": ["bbbb2222"],
            "project": "project:memo",
        },
        {
            "text": "dónde vive el socket del daemon warm",
            "relevant": True,
            "expect_ids": ["dddd4444"],
            "project": "project:otro",
        },
        {
            "text": "algo totalmente nuevo sobre synthesis nocturna",
            "relevant": True,
            "expect_ids": ["eeee5555"],
            "project": "project:nuevo",
        },
    ]
    merged = eval_recall.merge_label_prompts(existing, harvested)
    assert len(merged) == 3
    # existing without project adopts the harvested one
    assert merged[0]["project"] == "project:memo"
    # existing WITH project keeps its own (first non-null wins)
    assert merged[1]["project"] == "project:memo"
    # appended new label keeps its harvested project
    assert merged[2]["project"] == "project:nuevo"


def _capture_run_config_knobs_per_prompt(monkeypatch, cfg, labels, k: int = 3):
    """Run run_config with a stubbed rank_hits; return the knobs per prompt."""
    import memo.recall_logic as rl

    captured: list = []

    def fake_rank_hits(hits, knobs, **kw):
        captured.append(knobs)
        return []

    monkeypatch.setattr(rl, "rank_hits", fake_rank_hits)
    mem = SimpleNamespace(search=lambda *a, **kw: [])
    eval_recall.run_config(mem, cfg, k, labels)
    return captured


def test_run_config_per_label_project_tag_reaches_rank_hits(monkeypatch):
    cfg = eval_recall.Cfg("X vec/0.33/keep", "vec", 0.33, exclude_archived=False)
    labels = LabelSet(
        prompts=[
            Prompt("q with project", relevant=True, project="project:memo"),
            Prompt("q without project", relevant=True),
        ]
    )

    knobs = _capture_run_config_knobs_per_prompt(monkeypatch, cfg, labels)

    assert knobs[0].project_tag == "project:memo"
    assert knobs[1].project_tag is None
    # only project_tag differs — everything else stays the base resolution
    from dataclasses import replace as _replace

    assert _replace(knobs[0], project_tag=None) == knobs[1]


def test_run_config_projectless_labels_share_base_knobs(monkeypatch):
    """Aggregation unchanged for project-less sets: every prompt ranks with the
    SAME base knobs object (project_tag=None) — no per-label divergence."""
    cfg = eval_recall.Cfg("X vec/0.33/keep", "vec", 0.33, exclude_archived=False)
    labels = LabelSet(prompts=[Prompt("a", relevant=True), Prompt("b", relevant=True)])

    knobs = _capture_run_config_knobs_per_prompt(monkeypatch, cfg, labels)

    assert knobs[0] is knobs[1]  # identical object: the base knobs
    assert knobs[0].project_tag is None


def test_run_config_knob_override_project_tag_beats_label(monkeypatch):
    cfg = eval_recall.Cfg(
        "X vec/0.33/keep",
        "vec",
        0.33,
        exclude_archived=False,
        knob_overrides={"project_tag": "project:pinned"},
    )
    labels = LabelSet(prompts=[Prompt("q", relevant=True, project="project:memo")])

    knobs = _capture_run_config_knobs_per_prompt(monkeypatch, cfg, labels)

    assert knobs[0].project_tag == "project:pinned"  # overrides beat the label


def test_run_config_label_project_gated_on_project_boost(monkeypatch):
    """Hook-faithful gating: project_boost <= 0 means no project tiers at all,
    so the label's project must NOT set project_tag (same as cwd resolution)."""
    cfg = eval_recall.Cfg(
        "X vec/0.33/keep",
        "vec",
        0.33,
        exclude_archived=False,
        knob_overrides={"project_boost": 0.0},
    )
    labels = LabelSet(prompts=[Prompt("q", relevant=True, project="project:memo")])

    knobs = _capture_run_config_knobs_per_prompt(monkeypatch, cfg, labels)

    assert knobs[0].project_tag is None


def _project_corpus_mem():
    """Synthetic corpus: the project:alpha hit trails a project:beta hit on raw
    score, so only the project boost of a matching-project label flips the order."""
    from dataclasses import dataclass as _dc
    from dataclasses import field as _field

    @_dc
    class _Hit:
        id: str
        score: float | None
        title: str = ""
        body: str = ""
        type: str = "note"
        tags: list = _field(default_factory=list)
        path: str = "p.md"
        extra: dict = _field(default_factory=dict)

    class _Mem:
        def search(self, *a, **kw):
            return [
                _Hit(
                    "bbbbbbbb",
                    0.90,
                    title="beta note",
                    body="beta project body",
                    tags=["project:beta"],
                ),
                _Hit(
                    "aaaaaaaa",
                    0.80,
                    title="alpha note",
                    body="alpha project body",
                    tags=["project:alpha"],
                ),
            ]

    return _Mem()


def test_run_config_project_boost_applied_only_for_matching_label():
    # Pin the boosts via knob_overrides so the test is env-independent.
    def _cfg(name):
        return eval_recall.Cfg(
            name,
            "vec",
            0.0,
            exclude_archived=False,
            knob_overrides={"project_boost": 0.25, "global_boost": 0.10},
        )

    with_project = LabelSet(
        prompts=[Prompt("q", relevant=True, expect_ids=["aaaaaaaa"], project="project:alpha")]
    )
    without_project = LabelSet(prompts=[Prompt("q", relevant=True, expect_ids=["aaaaaaaa"])])

    row_p = eval_recall.run_config(_project_corpus_mem(), _cfg("P vec/0.0/keep"), 1, with_project)
    row_n = eval_recall.run_config(
        _project_corpus_mem(), _cfg("N vec/0.0/keep"), 1, without_project
    )

    # Matching-project label: alpha gets +0.25 (0.80 → 1.05) and outranks beta.
    assert [t["title"] for t in row_p.detail[0]["top"]] == ["alpha note"]
    assert row_p.precision_at_k == 1.0
    # Project-less label: raw score order — beta stays on top, alpha misses @1.
    assert [t["title"] for t in row_n.detail[0]["top"]] == ["beta note"]
    assert row_n.precision_at_k == 0.0


def test_avoid_ids_count_as_noise() -> None:
    from dataclasses import dataclass, field
    from typing import Any

    from memo.eval_recall import Cfg, LabelSet, Prompt, run_config

    @dataclass
    class _Hit:
        id: str
        score: float | None
        title: str = "t"
        body: str = "distinct body long enough for ranking"
        type: str = "note"
        tags: list[str] = field(default_factory=list)
        path: str = "p.md"
        extra: dict[str, Any] = field(default_factory=dict)

    class _Mem:
        def search(self, *a: Any, **k: Any) -> list[Any]:
            return [_Hit("bad1234567890abc", 0.9), _Hit("good567890abcdef", 0.8)]

    labels = LabelSet(prompts=[Prompt("q", relevant=False, avoid_ids=["bad12345"])])
    row = run_config(_Mem(), Cfg("X vec/0.0/keep", "vec", 0.0, exclude_archived=False), 2, labels)
    assert row.noise_at_k == 0.5  # 1 avoid-hit in top-2 / (1 prompt * k=2)


def test_label_parses_avoid_ids() -> None:
    from memo.eval_recall import _label_from_dict

    lab = _label_from_dict({"text": "q", "avoid_ids": ["bad12345"]})
    assert lab.avoid_ids == ["bad12345"]


def test_fingerprint_changes_with_avoid_ids() -> None:
    from memo.eval_recall import LabelSet, Prompt

    a = LabelSet(prompts=[Prompt("q")])
    b = LabelSet(prompts=[Prompt("q", avoid_ids=["bad12345"])])
    assert a.fingerprint() != b.fingerprint()


def test_harvest_negative_labels_from_verdict_log(tmp_path) -> None:
    from memo.dashboard import append_verdict_log
    from memo.eval_recall import harvest_negative_labels

    append_verdict_log(tmp_path, session_id="s1", turn=4, prior_turn=3,
                       verdict="negative", prompt="cómo configuro el sync remoto?",
                       reaction="no funciona", recall_ids=["aaaabbbb11112222"])
    append_verdict_log(tmp_path, session_id="s1", turn=6, prior_turn=5,
                       verdict="positive", prompt="otra", reaction="gracias",
                       recall_ids=["ccccdddd11112222"])
    out = harvest_negative_labels(tmp_path)
    assert out == [{"text": "cómo configuro el sync remoto?",
                    "relevant": False, "avoid_ids": ["aaaabbbb"]}]


def test_merge_expect_ids_beat_avoid_ids() -> None:
    from memo.eval_recall import merge_label_prompts

    existing = [{"text": "cómo configuro el sync remoto?", "relevant": True,
                 "expect_ids": ["aaaabbbb"]}]
    negatives = [{"text": "cómo configuro el sync remoto?", "relevant": False,
                  "avoid_ids": ["aaaabbbb", "eeeeffff"]}]
    merged = merge_label_prompts(existing, negatives)
    assert len(merged) == 1
    assert merged[0]["expect_ids"] == ["aaaabbbb"]
    assert merged[0]["avoid_ids"] == ["eeeeffff"]  # grounded evidence wins


def test_hyde_config_is_named_only_not_default() -> None:
    names = [c.name for c in eval_recall.default_configs()]
    assert not any("hyde" in n.lower() for n in names)  # default grid stays no-MLX
    assert not any("mmr" in n.lower() or "synth" in n.lower() for n in names)
    sel = eval_recall.select_configs(["J"])
    assert sel[0].mode == "hybrid"
    assert sel[0].flag_overrides == {"MEMO_HYDE_ENABLED": "1"}
    # default selection (no names) still returns exactly the default grid
    assert [c.name for c in eval_recall.select_configs(None)] == names


def test_run_config_pins_and_restores_flag_overrides(monkeypatch) -> None:
    import os

    seen: dict = {}

    class _Mem:
        def search(self, *a, **k):
            seen["hyde"] = os.environ.get("MEMO_HYDE_ENABLED")
            return []

    monkeypatch.delenv("MEMO_HYDE_ENABLED", raising=False)
    cfg = eval_recall.Cfg(
        "X vec/0.4/pin", "vec", 0.4, exclude_archived=False,
        flag_overrides={"MEMO_HYDE_ENABLED": "1"},
    )
    labels = LabelSet(prompts=[Prompt("some query", relevant=True)])
    eval_recall.run_config(_Mem(), cfg, 3, labels)
    assert seen["hyde"] == "1"          # pinned during the run
    assert "MEMO_HYDE_ENABLED" not in os.environ  # restored after


def test_expand_labels_copies_expect_ids_and_project() -> None:
    from memo.eval_recall import expand_labels

    prompts = [
        {"text": "cómo configuro el sync remoto?", "relevant": True,
         "expect_ids": ["aaaabbbb"], "project": "project:memo"},
        {"text": "prompt sin respuesta conocida", "relevant": False},
    ]

    def _gen(text: str, n: int) -> list[str]:
        return [f"paráfrasis {i} de: {text}" for i in range(n)]

    out = expand_labels(prompts, generate=_gen, per_prompt=2)
    assert len(out) == 2  # only the expect_ids prompt expands
    assert all(o["expect_ids"] == ["aaaabbbb"] for o in out)
    assert all(o["project"] == "project:memo" for o in out)
    assert all(o["relevant"] is True for o in out)
    assert out[0]["expanded_from"].startswith("cómo configuro")


def test_expand_labels_drops_duplicates_and_short() -> None:
    from memo.eval_recall import expand_labels

    prompts = [{"text": "cómo configuro el sync remoto?", "relevant": True,
                "expect_ids": ["aaaabbbb"]}]
    out = expand_labels(
        prompts,
        generate=lambda t, n: [t, "corto", "cómo seteo el sync remoto de memo?"],
        per_prompt=3,
    )
    assert [o["text"] for o in out] == ["cómo seteo el sync remoto de memo?"]


# --- ranked-retrieval metrics (R@K / NDCG@K / MRR) ---------------------------


def test_recall_at_k_fraction_of_expected_found():
    ranked = ["aaaaaaaa11111111", "bbbbbbbb22222222", "cccccccc33333333"]
    assert eval_recall.recall_at_k(ranked, ["aaaaaaaa", "dddddddd"], k=3) == 0.5
    assert eval_recall.recall_at_k(ranked, ["aaaaaaaa", "bbbbbbbb"], k=1) == 0.5
    assert eval_recall.recall_at_k(ranked, [], k=3) == 0.0


def test_mrr_at_k_first_relevant_rank():
    ranked = ["ffffffff00000000", "aaaaaaaa11111111"]
    assert eval_recall.mrr_at_k(ranked, ["aaaaaaaa"], k=5) == 0.5
    assert eval_recall.mrr_at_k(ranked, ["ffffffff"], k=5) == 1.0
    assert eval_recall.mrr_at_k(ranked, ["eeeeeeee"], k=5) == 0.0


def test_ndcg_at_k_binary_gains():
    # single expected id found at rank 2: DCG = 1/log2(3), IDCG = 1/log2(2)
    ranked = ["ffffffff00000000", "aaaaaaaa11111111"]
    got = eval_recall.ndcg_at_k(ranked, ["aaaaaaaa"], k=2)
    assert got == pytest.approx(math.log(2) / math.log(3))
    assert eval_recall.ndcg_at_k(ranked, ["ffffffff"], k=2) == pytest.approx(1.0)
    assert eval_recall.ndcg_at_k(ranked, ["eeeeeeee"], k=2) == 0.0
    assert eval_recall.ndcg_at_k(ranked, [], k=2) == 0.0


def test_run_config_reports_ranked_metrics(mock_memory):
    rec = mock_memory.save(content="the bench metric target note", title="bench target")
    labels = LabelSet(
        prompts=[Prompt("the bench metric target note", relevant=True, expect_ids=[rec.id])]
    )
    cfg = eval_recall.Cfg("t", "vec", -1.0, exclude_archived=False)
    row = eval_recall.run_config(mock_memory, cfg, 5, labels)
    # one record in the corpus + floor -1.0 → it must be in the top-5
    assert row.recall_at_k == 1.0
    assert 0.0 < row.mrr <= 1.0
    assert 0.0 < row.ndcg_at_k <= 1.0


def test_run_config_reports_quality_metrics() -> None:
    from dataclasses import dataclass, field
    from typing import Any

    @dataclass
    class _Hit:
        id: str
        score: float | None
        title: str = ""
        body: str = ""
        type: str = "note"
        tags: list[str] = field(default_factory=list)
        path: str = "p.md"
        extra: dict[str, Any] = field(default_factory=dict)

    class _Mem:
        def search(self, *a: Any, **k: Any) -> list[Any]:
            return [
                _Hit("stale", 0.9, title="Old", extra={"superseded_by": "canonical"}),
                _Hit("redundant", 0.8, title="Redundant", extra={"canonical_id": "canonical"}),
                _Hit("canonical", 0.7, title="Canonical", type="synthesis"),
            ]

    labels = LabelSet(prompts=[Prompt("q", relevant=True)])
    row = eval_recall.run_config(
        _Mem(),
        eval_recall.Cfg("X vec/0.0/keep", "vec", 0.0, exclude_archived=False),
        2,
        labels,
    )
    metrics = eval_recall.gate_metrics([row])

    assert metrics["stale_at_k"] == 0.5
    assert metrics["canonical_hit_at_k"] == 0.0
    assert metrics["pack_answerability"] is None
    assert metrics["compaction_safety"] is None


def test_quality_eval_metrics_counts_all_stale_quality_signals() -> None:
    hits = [
        _rec(id="invalidated-at", extra={"invalidated_at": "2026-01-01T00:00:00Z"}),
        _rec(id="rejected", verification_state="rejected"),
        _rec(id="contradiction-loser", extra={"contradiction_status": "lost"}),
        _rec(id="current"),
    ]

    metrics = eval_recall._quality_eval_metrics(hits, k=4)

    assert metrics["stale_at_k"] == 0.75


def test_run_config_counts_type_marked_canonical_hit() -> None:
    from dataclasses import dataclass, field
    from typing import Any

    @dataclass
    class _Hit:
        id: str
        score: float | None
        title: str = ""
        body: str = ""
        type: str = "note"
        tags: list[str] = field(default_factory=list)
        path: str = "p.md"
        extra: dict[str, Any] = field(default_factory=dict)

    class _Mem:
        def search(self, *a: Any, **k: Any) -> list[Any]:
            return [_Hit("canonical", 0.8, title="Canonical", type="synthesis")]

    labels = LabelSet(prompts=[Prompt("q", relevant=True)])
    row = eval_recall.run_config(
        _Mem(),
        eval_recall.Cfg("X vec/0.0/keep", "vec", 0.0, exclude_archived=False),
        1,
        labels,
    )

    assert eval_recall.gate_metrics([row])["canonical_hit_at_k"] == 1.0
