"""Tests for `memo debug-recall` (cli_debug_recall) + the rank_hits explain seam.

The CLI tests invoke the command object directly (it is wired into cli.py by
the orchestrator, not here) with an isolated Config via env pins and a
deterministic 4-dim keyword embedder stub. The rank_hits tests assert the
explain extension changes nothing about ranking when unused.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

import pytest
from click.testing import CliRunner

from memo.cli_debug_recall import debug_recall_cmd
from memo.config import Config
from memo.recall_logic import RankKnobs, rank_hits

# ---------------------------------------------------------------------------
# rank_hits explain seam (pure, no Memory needed)
# ---------------------------------------------------------------------------


@dataclass
class _Hit:
    id: str
    score: float | None
    title: str = ""
    body: str = ""
    type: str = "note"
    tags: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


def _mk(id: str, score: float | None, **kw: Any) -> _Hit:
    kw.setdefault("title", f"title {id}")
    kw.setdefault("body", f"distinct body for memory {id}, long enough to pass the gate")
    return _Hit(id=id, score=score, **kw)


def test_rank_hits_explain_none_path_is_identical() -> None:
    """explain=None (the hook path) must rank exactly like the explain path."""
    hits = [_mk("a", 0.9, tags=["project:x"]), _mk("b", 0.3), _mk("c", 0.7)]
    knobs = RankKnobs(min_sim=0.5, mode="vec", min_body_chars=0, project_tag="project:x")
    baseline = rank_hits(hits, knobs)
    explain: dict[str, dict[str, Any]] = {}
    explained = rank_hits(hits, knobs, explain=explain)
    assert [(h.id, h.score) for h in explained] == [(h.id, h.score) for h in baseline]


def test_rank_hits_explain_records_boosts_ranks_and_floor() -> None:
    hits = [_mk("a", 0.9, tags=["project:x"]), _mk("b", 0.3), _mk("c", 0.7)]
    knobs = RankKnobs(min_sim=0.5, mode="vec", min_body_chars=0, project_tag="project:x")
    explain: dict[str, dict[str, Any]] = {}
    out = rank_hits(hits, knobs, explain=explain)
    assert [h.id for h in out] == ["a", "c"]

    a = explain["a"]
    assert a["raw_score"] == pytest.approx(0.9)
    assert a["tier_boost"] == pytest.approx(0.25)  # tagged with the current project
    assert a["final_score"] == pytest.approx(1.15)
    assert a["passed_min_sim"] is True
    assert a["dropped"] is None
    assert a["rank"] == 1

    c = explain["c"]
    assert c["tier_boost"] == pytest.approx(0.10)  # untagged -> global tier
    assert c["rank"] == 2

    b = explain["b"]
    assert b["passed_min_sim"] is False
    assert b["dropped"] == "min_sim"
    assert "rank" not in b


def test_rank_hits_explain_marks_dedup_drop() -> None:
    a = _Hit(id="a", score=0.9, title="same title", body="same body, long enough to gate x")
    b = _Hit(id="b", score=0.8, title="same title", body="same body, long enough to gate x")
    knobs = RankKnobs(min_sim=0.0, min_body_chars=0)
    explain: dict[str, dict[str, Any]] = {}
    out = rank_hits([a, b], knobs, explain=explain)
    assert [h.id for h in out] == ["a"]
    assert explain["a"]["dropped"] is None
    assert explain["b"]["dropped"] == "dedup"


def test_rank_hits_explain_marks_min_body_drop() -> None:
    hits = [_mk("a", 0.9), _Hit(id="b", score=0.8, title="short", body="tiny")]
    knobs = RankKnobs(min_sim=0.5, mode="vec", min_body_chars=20)
    explain: dict[str, dict[str, Any]] = {}
    out = rank_hits(hits, knobs, explain=explain)
    assert [h.id for h in out] == ["a"]
    assert explain["b"]["passed_min_sim"] is True
    assert explain["b"]["passed_min_body"] is False
    assert explain["b"]["dropped"] == "min_body"


# ---------------------------------------------------------------------------
# CLI (CliRunner, isolated Config, 4-dim keyword stub embedder)
# ---------------------------------------------------------------------------


# Pinned in both the seeding Config and the CLI env so a developer shell's
# exported MEMO_EMBEDDER_MODEL can't leak through CliRunner's env overlay
# and trip the index/model-mismatch guard.
_STUB_MODEL = "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ"


def _install_keyword_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    """Deterministic 4-dim embedder: alpha/beta prompts map to orthogonal axes."""

    def _embed(self: Any, inputs: Any) -> list[list[float]]:
        out = []
        for s in inputs:
            low = s.lower()
            if "alpha" in low:
                out.append([1.0, 0.0, 0.0, 0.0])
            elif "beta" in low:
                out.append([0.0, 1.0, 0.0, 0.0])
            else:
                out.append([0.0, 0.0, 1.0, 0.0])
        return out

    monkeypatch.setattr("memo.embedder.MLXEmbedder.embed", _embed)


def _seed(tmp_cfg: Config) -> tuple[str, str]:
    """Save one alpha + one beta memory into the isolated store; return ids."""
    from memo.memory import Memory

    cfg = Config(
        data_dir=tmp_cfg.data_dir,
        vault_path=tmp_cfg.vault_path,
        state_dir=tmp_cfg.state_dir,
        embedder_model=_STUB_MODEL,
        embedder_dims=4,
        reranker_enabled=False,
    )
    mem = Memory(cfg)
    try:
        alpha = mem.save(
            content="the alpha rollout decision was made after deliberation about the gates",
            title="alpha rollout decision",
            auto_project=False,
        )
        beta = mem.save(
            content="the beta cache eviction policy uses LFU with a nightly prune floor",
            title="beta cache eviction",
            auto_project=False,
        )
    finally:
        mem.close()
    return alpha.id, beta.id


def _cli_env(tmp_cfg: Config) -> dict[str, str]:
    return {
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_DATA_DIR": str(tmp_cfg.data_dir),
        "MEMO_STATE_DIR": str(tmp_cfg.state_dir),
        "MEMO_VAULT_PATH": str(tmp_cfg.vault_path),
        "MEMO_EMBEDDER_MODEL": _STUB_MODEL,
        "MEMO_EMBEDDER_DIMS": "4",  # pinned to the stub's output dim
        "MEMO_RERANKER_ENABLED": "0",
        # cwd-independent test: 0 disables project-tag resolution + tier boosts
        "MEMO_RECALL_PROJECT_BOOST": "0",
        # Pin the post-rank output filters to their shipped defaults so a
        # developer shell's exported values can't change the injected verdict.
        "MEMO_RECALL_SKIP_BELOW": "0.45",
        "MEMO_RECALL_GAP_THRESHOLD": "0.10",
    }


_PROMPT = "what did we decide about the alpha rollout?"

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


@pytest.mark.float32_precision  # asserts vec_sim == 1.0 within abs=1e-6; int8 cosine is ~1/127-quantized
def test_debug_recall_json_shape(tmp_cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_keyword_stub(monkeypatch)
    alpha_id, beta_id = _seed(tmp_cfg)

    result = CliRunner().invoke(debug_recall_cmd, [_PROMPT, "--json"], env=_cli_env(tmp_cfg))
    assert result.exit_code == 0, result.output

    payload = json.loads(result.output)
    assert set(payload) == {"hits", "config"}

    cfg = payload["config"]
    assert cfg["min_sim"] == 0.5
    assert cfg["top_k"] == 3
    assert cfg["mode"] == "vec"
    assert cfg["reranker_enabled"] is False
    assert cfg["reranker_ran"] is False

    by_id = {h["id"]: h for h in payload["hits"]}
    a = by_id[alpha_id]
    assert a["id8"] == alpha_id[:8]
    assert a["rank"] == 1
    assert a["injected"] is True
    assert a["passed_min_sim"] is True
    assert a["dropped"] is None
    assert a["vec_sim"] == pytest.approx(1.0, abs=1e-6)

    b = by_id[beta_id]
    assert b["rank"] is None
    assert b["passed_min_sim"] is False
    assert b["dropped"] == "min_sim"
    assert b["vec_sim"] == pytest.approx(0.0, abs=1e-6)


def test_debug_recall_table_renders(tmp_cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_keyword_stub(monkeypatch)
    alpha_id, beta_id = _seed(tmp_cfg)

    result = CliRunner().invoke(debug_recall_cmd, [_PROMPT], env=_cli_env(tmp_cfg))
    assert result.exit_code == 0, result.output
    plain = _strip_ansi(result.output)
    # Thresholds line + both candidates by 8-char id + floor verdicts.
    assert "min_sim=0.5" in plain
    assert alpha_id[:8] in plain
    assert beta_id[:8] in plain
    assert "candidates" in plain


def test_debug_recall_config_echoes_skip_below_and_gap(
    tmp_cfg: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """MEMO_RECALL_SKIP_BELOW / MEMO_RECALL_GAP_THRESHOLD are echoed in both
    the --json config dict and the rendered thresholds header."""
    _install_keyword_stub(monkeypatch)
    _seed(tmp_cfg)
    env = {**_cli_env(tmp_cfg), "MEMO_RECALL_SKIP_BELOW": "0.3", "MEMO_RECALL_GAP_THRESHOLD": "0.2"}

    result = CliRunner().invoke(debug_recall_cmd, [_PROMPT, "--json"], env=env)
    assert result.exit_code == 0, result.output
    cfg = json.loads(result.output)["config"]
    assert cfg["skip_below"] == pytest.approx(0.3)
    assert cfg["gap_threshold"] == pytest.approx(0.2)
    assert cfg["skip_below_triggered"] is False  # alpha scores well above 0.3

    result = CliRunner().invoke(debug_recall_cmd, [_PROMPT], env=env)
    assert result.exit_code == 0, result.output
    plain = _strip_ansi(result.output)
    assert "skip_below=0.3" in plain
    assert "gap_threshold=0.2" in plain


def test_debug_recall_injected_honors_skip_below(
    tmp_cfg: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The hook skips recall entirely when the best score is under
    MEMO_RECALL_SKIP_BELOW — debug-recall must not show '● injected' then."""
    _install_keyword_stub(monkeypatch)
    _seed(tmp_cfg)
    # Gamma prompt: both memories score ~0.0; min_sim=0 lets them RANK, but the
    # real hook's skip_below floor (0.45) suppresses the whole injection.
    prompt = "anything about gamma at all?"
    env = {
        **_cli_env(tmp_cfg),
        "MEMO_RECALL_MIN_SIM": "0.0",
        "MEMO_RECALL_SKIP_BELOW": "0.45",
        "MEMO_RECALL_GAP_THRESHOLD": "0",
    }

    result = CliRunner().invoke(debug_recall_cmd, [prompt, "--json"], env=env)
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["config"]["skip_below_triggered"] is True
    ranked = [h for h in payload["hits"] if h["rank"] is not None]
    assert ranked  # hits DID rank — pre-fix they showed as injected
    assert all(h["injected"] is False for h in payload["hits"])

    # Same run with the floor disabled: rank 1 is injected again.
    result = CliRunner().invoke(
        debug_recall_cmd, [prompt, "--json"], env={**env, "MEMO_RECALL_SKIP_BELOW": "0"}
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["config"]["skip_below_triggered"] is False
    by_rank = {h["rank"]: h for h in payload["hits"] if h["rank"] is not None}
    assert by_rank[1]["injected"] is True


def test_debug_recall_injected_honors_gap_threshold(
    tmp_cfg: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rank-1/rank-2 score gap over MEMO_RECALL_GAP_THRESHOLD truncates the
    real hook's injection to top-1 — rank 2 must not show as injected."""
    _install_keyword_stub(monkeypatch)
    alpha_id, beta_id = _seed(tmp_cfg)
    # Alpha prompt: alpha ~1.0, beta ~0.0; min_sim=0 keeps beta ranked at 2.
    env = {
        **_cli_env(tmp_cfg),
        "MEMO_RECALL_MIN_SIM": "0.0",
        "MEMO_RECALL_SKIP_BELOW": "0",
        "MEMO_RECALL_GAP_THRESHOLD": "0.5",
    }

    result = CliRunner().invoke(debug_recall_cmd, [_PROMPT, "--json"], env=env)
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    by_id = {h["id"]: h for h in payload["hits"]}
    assert by_id[alpha_id]["rank"] == 1
    assert by_id[alpha_id]["injected"] is True
    assert by_id[beta_id]["rank"] == 2
    assert by_id[beta_id]["injected"] is False  # gap-truncated, hook injects top-1 only

    # Gap disabled: rank 2 is injected again (top_k=3 covers it).
    result = CliRunner().invoke(
        debug_recall_cmd, [_PROMPT, "--json"], env={**env, "MEMO_RECALL_GAP_THRESHOLD": "0"}
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    by_id = {h["id"]: h for h in payload["hits"]}
    assert by_id[beta_id]["injected"] is True


def test_debug_recall_renders_synthesis_boost_and_mmr(
    tmp_cfg: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The newer explain stages surface in the breakdown: synthesis_boost is a
    score delta in the boosts cell; mmr is a re-ORDER (not a delta) shown as a
    compact mmr=<greedy score> indicator."""
    _install_keyword_stub(monkeypatch)
    from memo.memory import Memory

    cfg = Config(
        data_dir=tmp_cfg.data_dir,
        vault_path=tmp_cfg.vault_path,
        state_dir=tmp_cfg.state_dir,
        embedder_model=_STUB_MODEL,
        embedder_dims=4,
        reranker_enabled=False,
    )
    mem = Memory(cfg)
    try:
        syn = mem.save(
            content="alpha rollout synthesis distilled from the gating deliberations",
            title="alpha rollout synthesis",
            type_="synthesis",
            auto_project=False,
        )
        mem.save(
            content="the alpha rollout decision was made after deliberation about the gates",
            title="alpha rollout decision",
            auto_project=False,
        )
    finally:
        mem.close()

    env = {
        **_cli_env(tmp_cfg),
        "MEMO_RECALL_MIN_SIM": "0.0",
        "MEMO_RECALL_SKIP_BELOW": "0",
        "MEMO_RECALL_GAP_THRESHOLD": "0",
        "MEMO_RECALL_SYNTHESIS_BOOST": "0.05",
        "MEMO_RECALL_MMR_LAMBDA": "0.7",
        "COLUMNS": "200",  # keep the boosts cell unwrapped for the substring asserts
    }

    result = CliRunner().invoke(debug_recall_cmd, [_PROMPT, "--json"], env=env)
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    by_id = {h["id"]: h for h in payload["hits"]}
    assert by_id[syn.id]["synthesis_boost"] == pytest.approx(0.05)
    mmr_entries = [h["mmr"] for h in payload["hits"] if h.get("mmr")]
    assert mmr_entries
    assert {"mmr_score", "max_sim_to_selected"} <= set(mmr_entries[0])

    result = CliRunner().invoke(debug_recall_cmd, [_PROMPT], env=env)
    assert result.exit_code == 0, result.output
    plain = _strip_ansi(result.output)
    # The header echoes "synth+0.05" once; the boosts cell adds a second one.
    assert plain.count("synth+0.05") >= 2
    assert "mmr=" in plain  # boosts-cell indicator (header says "mmr_lambda=")


def test_debug_recall_empty_index(tmp_cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_keyword_stub(monkeypatch)
    result = CliRunner().invoke(
        debug_recall_cmd, ["anything about gamma at all?"], env=_cli_env(tmp_cfg)
    )
    assert result.exit_code == 0, result.output
    assert "no candidates" in _strip_ansi(result.output)
